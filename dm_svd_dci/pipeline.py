#!/usr/bin/env python3
"""
Main pipeline: dmSVD + Krylov-dCI combined method.

Steps:
  1. PySCF molecule setup + CASCI reference
  2. dmSVD: occ-virt determinant partition → density matrix SVD → Schmidt basis
  3. Build H^emb = H_A + H_B + H_AB in Schmidt product basis (Path C + parallel sigma)
  4. Partition Schmidt basis: P = {n ∈ p_blocks}, Q = all other n
  5. Extract H_PP, H_PQ, H_QQ from full H^emb (方案 A)
  6. Krylov-dCI: m=0 MGS → m=1 MGS (NO SVD)
  7. Löwdin effective Hamiltonian + diagonalization
  8. Compare with CASCI reference

Supports two modes:
  - 'gs':  Ground-state only SVD (single-state ρ_A)
  - 'sa':  State-averaged SVD (multi-state ρ_A^SA / ρ_B^SA)
"""

import sys, os, time, json
import numpy as np
from numpy.linalg import eigh
from typing import Dict, List, Tuple, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


# ═══════════════════════════════════════════════════════════════
# Step 1: System setup
# ═══════════════════════════════════════════════════════════════

def setup_system(
    atom: str = 'N 0 0 0; N 0 0 1.098',
    basis: str = 'cc-pVDZ',
    n_active: int = 10,
    n_active_elec: Tuple[int, int] = (5, 5),
    n_core: int = 2,
    nroots: int = 1,
    verbose: bool = True,
) -> Dict:
    """Initialize PySCF molecule, RHF, CASCI, and backend."""
    from pyscf import gto, scf, mcscf
    from pyscf.fci import cistring
    from src_mf import QSpaceIndex, KDCIBackend
    from src.hamiltonian import Hamiltonian, _unpack_4fold

    t0 = time.perf_counter()

    mol = gto.M(atom=atom, basis=basis, verbose=0, spin=0)
    mf = scf.RHF(mol).run(verbose=0)

    n_elec_total = sum(n_active_elec)
    n_act = n_active

    cas = mcscf.CASCI(mf, n_act, n_elec_total)
    cas.frozen = n_core
    h1eff, ecore = cas.get_h1eff()
    h2eff = cas.get_h2eff()
    cas.kernel()
    fcivec = cas.ci
    ci_flat = fcivec.reshape(-1)
    E_fci = cas.e_tot  # total energy including core

    na, nb = n_active_elec
    alpha_strs = cistring.gen_strings4orblist(range(n_act), na)
    beta_strs = cistring.gen_strings4orblist(range(n_act), nb)

    q_idx = QSpaceIndex(alpha_strs, beta_strs, n_act, n_active_elec, h1eff, h2eff)
    backend = KDCIBackend(q_idx)
    M_all = q_idx.M

    h2_4d = _unpack_4fold(h2eff, n_act)
    ham = Hamiltonian(h1=h1eff, h2=h2_4d, E_nuc=ecore, E_HF=0.0)

    if verbose:
        print(f"  System:          {atom.strip()}, {basis}")
        print(f"  CAS({n_act},{n_elec_total}):      M={M_all:,} dets, n_core={n_core}")
        print(f"  CASCI total E:   {E_fci:.12f} Ha")
        print(f"  E_core (frozen): {ecore:.12f} Ha")
        print(f"  Active E:        {E_fci - ecore:.12f} Ha")
        print(f"  Setup done:      {time.perf_counter() - t0:.0f}s", flush=True)

    return {
        'mol': mol, 'mf': mf,
        'n_active': n_act, 'n_active_elec': n_active_elec, 'n_core': n_core,
        'h1eff': h1eff, 'h2eff': h2eff, 'h2_4d': h2_4d, 'ecore': ecore,
        'fcivec': fcivec, 'ci_flat': ci_flat, 'E_fci': E_fci,
        'alpha_strs': alpha_strs, 'beta_strs': beta_strs,
        'na': na, 'nb': nb, 'M_all': M_all,
        'q_idx': q_idx, 'backend': backend, 'ham': ham,
    }


# ═══════════════════════════════════════════════════════════════
# Step 3: Build H^emb = H_A + H_B + H_AB  (Path C + parallel sigma)
# ═══════════════════════════════════════════════════════════════

def build_hemb_parallel(
    schmidt_data: Dict[int, Dict],
    partition: Dict[int, Dict],
    q_idx,
    backend,
    h1_full: np.ndarray,
    h2_full: np.ndarray,
    n_occ: int,
    n_act: int,
    n_workers: int = 1,
    verbose: bool = True,
) -> Tuple[np.ndarray, List[Dict], Dict]:
    """Build H^emb = H_A + H_B + H_AB in Schmidt product basis.

    H_AB: sigma-vector projection (parallel ThreadPoolExecutor)
    H_A:  Path C — build H_A^det, project via U^† H_A^det U
    H_B:  Path C — build H_B^det, project via V^† H_B^det V

    Returns (H_emb, basis_info, norms_dict).
    """
    from dm_svd_embedding.embedded_hamiltonian import (
        _expand_schmidt_product_to_ci_matrix,
        _build_subspace_hamiltonian,
        _extract_subspace_integrals,
    )
    from dm_svd_dci.parallel_ops import compute_sigma_vectors_parallel

    # ── Build basis index map ──
    basis_info = []
    offset = 0
    block_offsets = {}

    for n_A in sorted(schmidt_data.keys()):
        sd = schmidt_data[n_A]
        r = sd['r']
        block_offsets[n_A] = (offset, r)
        for alpha in range(r):
            for beta in range(r):
                basis_info.append({
                    'n': n_A, 'alpha': alpha, 'beta': beta,
                    'flat_idx': offset + alpha * r + beta,
                })
        offset += r * r

    D = offset
    if D == 0:
        return np.zeros((0, 0)), [], {}

    if verbose:
        print(f"  Schmidt product basis dimension: D = Σ_n r_n² = {D}")
        print(f"  {'n':>4} {'r_n':>5} {'r_n²':>7}")
        print(f"  {'-'*18}")
        for n_A in sorted(schmidt_data.keys()):
            r = schmidt_data[n_A]['r']
            print(f"  {n_A:>4} {r:>5} {r*r:>7}")

    # ── Full CAS string info ──
    alpha_strs = q_idx.alpha_strs
    beta_strs = q_idx.beta_strs
    n_alpha_strs = len(alpha_strs)
    n_beta_strs = len(beta_strs)
    alpha_to_idx = {int(s): i for i, s in enumerate(alpha_strs)}
    beta_to_idx = {int(s): i for i, s in enumerate(beta_strs)}

    # ── Orbital indices for A and B spaces ──
    A_orb_indices = np.arange(n_occ, dtype=int)
    B_orb_indices = np.arange(n_occ, n_act, dtype=int)
    n_virt = n_act - n_occ

    if verbose:
        print(f"  Orbitals: A = [0..{n_occ-1}], B = [{n_occ}..{n_act-1}] ({n_virt} virt)")

    # ═══════════════════════════════════════════════════════
    # Part 1: H_AB via sigma-vector projection (parallel)
    # ═══════════════════════════════════════════════════════
    if verbose:
        t_expand = time.perf_counter()
        print(f"  [H_AB] Expanding {D} Schmidt states to CAS CI matrices...", flush=True)

    ci_mats = []
    for info in basis_info:
        n_A_val = info['n']
        alpha = info['alpha']
        beta = info['beta']
        blk_schmidt = schmidt_data[n_A_val]
        blk_partition = partition[n_A_val]
        ci_mat = _expand_schmidt_product_to_ci_matrix(
            alpha, beta, blk_schmidt, blk_partition,
            n_alpha_strs, n_beta_strs, n_occ,
            alpha_strs, beta_strs, alpha_to_idx, beta_to_idx)
        ci_mats.append(ci_mat)

    if verbose:
        print(f"    Expansion done: {time.perf_counter() - t_expand:.0f}s", flush=True)

    # Parallel sigma-vector computation
    if verbose:
        t1 = time.perf_counter()
        print(f"  [H_AB] Computing {D} sigma-vectors ({n_workers} workers)...", flush=True)

    sigmas = compute_sigma_vectors_parallel(
        backend.sigma_full, ci_mats, n_workers=n_workers, verbose=verbose)

    if verbose:
        elapsed = time.perf_counter() - t1
        print(f"    Sigma done: {elapsed:.0f}s ({elapsed/max(D,1):.2f}s/vector)", flush=True)

    # Project: H_emb_AB[k,l] = v_l · sigma_k
    if verbose:
        t_proj = time.perf_counter()
        print(f"  [H_AB] Projecting {D}×{D} matrix elements...", flush=True)

    ci_flat_mats = [cm.reshape(-1) for cm in ci_mats]
    sigma_flat = [sm.reshape(-1) for sm in sigmas]
    H_emb = np.zeros((D, D))

    for k in range(D):
        sk = sigma_flat[k]
        for l in range(D):
            H_emb[l, k] = np.dot(ci_flat_mats[l], sk)

    if verbose:
        elapsed = time.perf_counter() - t_proj
        print(f"    Projection done: {elapsed:.0f}s", flush=True)

    # Release memory for CI mats
    del ci_mats, sigmas, ci_flat_mats, sigma_flat

    # ═══════════════════════════════════════════════════════
    # Part 2: H_A + H_B via Path C
    # ═══════════════════════════════════════════════════════
    if verbose:
        t_pathc = time.perf_counter()
        print(f"  [Path C] Computing H_A and H_B in Schmidt basis...", flush=True)

    h1_A, h2_A = _extract_subspace_integrals(h1_full, h2_full, A_orb_indices)
    h1_B, h2_B = _extract_subspace_integrals(h1_full, h2_full, B_orb_indices)

    H_emb_HA = np.zeros((D, D))
    H_emb_HB = np.zeros((D, D))

    for n_A_val in sorted(schmidt_data.keys()):
        sd = schmidt_data[n_A_val]
        blk = partition[n_A_val]
        r = sd['r']
        if r == 0:
            continue

        a_dets = blk['a_dets']
        if len(a_dets) == 0:
            continue

        aA0, bA0 = a_dets[0]
        nA_alpha = aA0.bit_count()
        nA_beta = bA0.bit_count()

        # Build H_A^det in A-subspace determinant basis
        if n_occ > 0:
            HA_det = _build_subspace_hamiltonian(
                a_dets, h1_A, h2_A, n_occ, nA_alpha, nA_beta)
            HA_schmidt = sd['U'].T @ HA_det @ sd['U']
        else:
            HA_schmidt = np.zeros((r, r))

        # Build H_B^det in B-subspace determinant basis
        b_dets = blk['b_dets']
        if len(b_dets) > 0 and n_virt > 0:
            bB0, bB0b = b_dets[0]
            nB_alpha = bB0.bit_count()
            nB_beta = bB0b.bit_count()
            HB_det = _build_subspace_hamiltonian(
                b_dets, h1_B, h2_B, n_virt, nB_alpha, nB_beta)
            HB_schmidt = sd['V'].T @ HB_det @ sd['V']
        else:
            HB_schmidt = np.zeros((r, r))

        # Map to H_emb blocks: H_A = U^† H_A^det U ⊗ I_B
        #                        H_B = I_A ⊗ V^† H_B^det V
        offset_n, _ = block_offsets[n_A_val]
        for alpha in range(r):
            for beta in range(r):
                k_idx = offset_n + alpha * r + beta
                # H_A: δ_{βδ} HA_schmidt[α,γ]
                for gamma in range(r):
                    l_idx = offset_n + gamma * r + beta
                    H_emb_HA[l_idx, k_idx] = HA_schmidt[gamma, alpha]
                # H_B: δ_{αγ} HB_schmidt[β,δ]
                for delta in range(r):
                    l_idx = offset_n + alpha * r + delta
                    H_emb_HB[l_idx, k_idx] = HB_schmidt[delta, beta]

    if verbose:
        elapsed = time.perf_counter() - t_pathc
        print(f"    Path C done: {elapsed:.0f}s", flush=True)

    # ═══════════════════════════════════════════════════════
    # Combine: H_emb = H_AB + H_A + H_B
    # ═══════════════════════════════════════════════════════
    H_emb += H_emb_HA
    H_emb += H_emb_HB
    H_emb = 0.5 * (H_emb + H_emb.T)

    # Diagnostic norms
    ha_norm = np.linalg.norm(H_emb_HA)
    hb_norm = np.linalg.norm(H_emb_HB)
    hab_norm = np.linalg.norm(H_emb - H_emb_HA - H_emb_HB)
    total_norm = np.linalg.norm(H_emb)
    asym = np.abs(H_emb - H_emb.T).max()

    norms = {
        'norm_HA': float(ha_norm), 'norm_HB': float(hb_norm),
        'norm_HAB': float(hab_norm), 'norm_total': float(total_norm),
        'asymmetry': float(asym),
    }

    if verbose:
        print(f"  H^emb decomposition:")
        print(f"    ||H_A||  = {ha_norm:.2f}")
        print(f"    ||H_B||  = {hb_norm:.2f}")
        print(f"    ||H_AB|| = {hab_norm:.2f}")
        print(f"    ||H||    = {total_norm:.2f}")
        print(f"    max|H - H^T| = {asym:.2e}", flush=True)

    return H_emb, basis_info, norms


# ═══════════════════════════════════════════════════════════════
# Main pipeline entry point
# ═══════════════════════════════════════════════════════════════

def run_dm_svd_dci(
    atom: str = 'N 0 0 0; N 0 0 1.098',
    basis: str = 'cc-pVDZ',
    n_active: int = 10,
    n_active_elec: Tuple[int, int] = (5, 5),
    n_core: int = 2,
    n_occ: int = 5,
    ms: int = 0,
    svd_eps: float = 1e-3,
    sa_states: int = 1,
    p_blocks: List[int] = [8, 9, 10],
    m_max: int = 1,
    delta: float = 0.0,
    lindep_threshold: float = 1e-10,
    n_workers: int = 1,
    output_dir: Optional[str] = None,
    verbose: bool = True,
) -> Dict:
    """Run the complete dmSVD + Krylov-dCI pipeline."""
    t_total = time.perf_counter()
    timing = {}

    # ═══════════════════════════════════════════════════════
    # Step 1: System setup
    # ═══════════════════════════════════════════════════════
    if verbose:
        print("=" * 70)
        print("STEP 1: System Setup")
        print("=" * 70)

    sys_data = setup_system(
        atom=atom, basis=basis,
        n_active=n_active, n_active_elec=n_active_elec, n_core=n_core,
        nroots=sa_states, verbose=verbose)
    timing['1_setup'] = time.perf_counter() - t_total

    # ═══════════════════════════════════════════════════════
    # Step 2: dmSVD
    # ═══════════════════════════════════════════════════════
    if verbose:
        print(f"\n{'=' * 70}")
        print(f"STEP 2: dmSVD (occ-virt partition + Schmidt decomposition)")
        print(f"{'=' * 70}")

    t_step2 = time.perf_counter()
    from dm_svd_embedding.occ_virt_partition import (
        setup_partition, build_block_matrices)
    from dm_svd_embedding.density_matrix import (
        compute_schmidt_decomposition, compute_compression_metrics)

    partition, full_dets = setup_partition(n_active, sum(n_active_elec), n_occ, ms=ms)
    C_blocks = build_block_matrices(partition, sys_data['ci_flat'])

    state_average = None
    if sa_states > 1:
        if verbose:
            print(f"  State-averaged mode: {sa_states} states")
        from pyscf import mcscf
        cas2 = mcscf.CASCI(sys_data['mf'], n_active, sum(n_active_elec))
        cas2.frozen = n_core
        cas2.fcisolver.nroots = sa_states
        cas2.kernel()
        state_average = []
        for k in range(sa_states):
            Ck_blocks = build_block_matrices(partition, cas2.ci[k].reshape(-1))
            state_average.append(Ck_blocks)

    schmidt = compute_schmidt_decomposition(
        C_blocks, eps=svd_eps, state_average=state_average)
    metrics = compute_compression_metrics(schmidt, C_blocks, sys_data['ci_flat'])

    timing['2_dm_svd'] = time.perf_counter() - t_step2

    if verbose:
        print(f"  Schmidt decomposition results:")
        print(f"    r_total = {metrics['r_total']}/{metrics['dim_fci']} "
              f"(compression {metrics['compression_ratio']:.4%})")
        print(f"    discarded weight = {metrics['discarded_weight']:.2e}")
        print(f"    Per-block singular values (σ₁):")
        for n_A in sorted(schmidt.keys()):
            sd = schmidt[n_A]
            if sd['r'] > 0:
                sig_str = ", ".join(f"{s:.2e}" for s in sd['sigma'][:5])
                print(f"      n={n_A}: r={sd['r']}/{sd['dim_A']}×{sd['dim_B']}, "
                      f"σ₁={sd['sigma'][0]:.2e} [{sig_str}...]")
            else:
                print(f"      n={n_A}: r=0 (fully truncated)")
        print(f"  dmSVD done: {timing['2_dm_svd']:.0f}s", flush=True)

    # ═══════════════════════════════════════════════════════
    # Step 3: Build H^emb
    # ═══════════════════════════════════════════════════════
    if verbose:
        print(f"\n{'=' * 70}")
        print(f"STEP 3: Build H^emb = H_A + H_B + H_AB")
        print(f"{'=' * 70}")

    t_step3 = time.perf_counter()
    H_emb, basis_info, hemb_norms = build_hemb_parallel(
        schmidt, partition,
        sys_data['q_idx'], sys_data['backend'],
        h1_full=sys_data['h1eff'], h2_full=sys_data['h2_4d'],
        n_occ=n_occ, n_act=n_active,
        n_workers=n_workers, verbose=verbose)
    timing['3_build_hemb'] = time.perf_counter() - t_step3
    D = H_emb.shape[0]

    # ═══════════════════════════════════════════════════════
    # Step 4: Partition P/Q
    # ═══════════════════════════════════════════════════════
    if verbose:
        print(f"\n{'=' * 70}")
        print(f"STEP 4: Partition Schmidt Basis → P / Q")
        print(f"{'=' * 70}")

    from dm_svd_dci.schmidt_partition import partition_schmidt_basis, extract_subblocks

    part = partition_schmidt_basis(schmidt, p_blocks=p_blocks)
    H_PP, H_PQ, H_QQ = extract_subblocks(H_emb, part)

    if verbose:
        print(f"  Total Schmidt dim: D = {part['total_dim']}")
        print(f"  P-space (n ∈ {p_blocks}): |P| = {part['p_dim']}")
        print(f"  Q-space (n ∉ {p_blocks}): |Q| = {part['q_dim']}")
        print(f"  H_PP: {H_PP.shape}, H_PQ: {H_PQ.shape}, H_QQ: {H_QQ.shape}")
        print(f"  H_PP sparsity: {np.count_nonzero(H_PP)/(H_PP.shape[0]*H_PP.shape[1]):.2%}")
        print(f"  H_QQ sparsity: {np.count_nonzero(H_QQ)/(H_QQ.shape[0]*H_QQ.shape[1]):.2%}",
              flush=True)

    # ═══════════════════════════════════════════════════════
    # Step 5: Bare H_PP diagonalization
    # ═══════════════════════════════════════════════════════
    if part['p_dim'] == 0:
        print("  ERROR: P-space is empty! Check p_blocks setting.")
        return {'error': 'Empty P-space'}

    E_P, C_P = eigh(H_PP)
    E0 = E_P[0]

    if verbose:
        dE_bare = (E0 - sys_data['E_fci']) * 1000
        print(f"\n  Bare H_PP diagonalization:")
        print(f"    E0 (lowest)    = {E0:.12f} Ha")
        print(f"    ΔE vs FCI      = {dE_bare:+.3f} mH")
        if len(E_P) >= 5:
            print(f"    First 5 eigenvalues:")
            for k in range(min(5, len(E_P))):
                exc = (E_P[k] - E_P[0]) * 1000 if k > 0 else 0.0
                print(f"      S{k}: {E_P[k]:.12f} Ha  ({exc:+.1f} mH exc)")
        print(f"    ||H_PP||       = {np.linalg.norm(H_PP):.2f}", flush=True)

    # ═══════════════════════════════════════════════════════
    # Step 6: Krylov-dCI (MGS only, NO SVD)
    # ═══════════════════════════════════════════════════════
    if verbose:
        print(f"\n{'=' * 70}")
        print(f"STEP 6: Krylov-dCI (MGS only, no SVD)")
        print(f"{'=' * 70}")

    from dm_svd_dci.krylov_propagator import build_krylov_full
    from dm_svd_dci.effective_ham import run_effective_ham_at_m

    res = {}
    t_krylov = time.perf_counter()
    H_QQ_diag = np.diag(H_QQ)

    # ── m=0 ──
    if verbose:
        print(f"\n  --- m=0: Initial Krylov basis (A · H_QP → MGS) ---")

    B0, layers0, A_q = build_krylov_full(
        H_PQ, H_QQ, H_QQ_diag, E0, m_max=0,
        lindep_threshold=lindep_threshold, verbose=verbose)
    r0 = layers0[0]

    res_m0 = run_effective_ham_at_m(
        H_PP, H_PQ, H_QQ, E0, B0,
        delta=delta, n_states=min(sa_states, len(E_P)),
        C_ref=C_P, verbose=verbose)

    E_eff_m0 = res_m0['E_eff'][0]
    dE_m0_mH = (E_eff_m0 - sys_data['E_fci']) * 1000

    res['E_eff_m0'] = E_eff_m0
    res['dE_m0_mH'] = dE_m0_mH
    res['r0'] = r0

    if verbose:
        print(f"\n  m=0 SUMMARY: E = {E_eff_m0:.12f} Ha, "
              f"ΔE = {dE_m0_mH:+.3f} mH, r₀ = {r0}", flush=True)

    # ── m=1 ──
    if m_max >= 1 and r0 > 0:
        if verbose:
            print(f"\n  --- m=1: Propagation (MGS only) ---")

        B1, layer_sizes, _ = build_krylov_full(
            H_PQ, H_QQ, H_QQ_diag, E0, m_max=1,
            lindep_threshold=lindep_threshold, verbose=verbose)

        res_m1 = run_effective_ham_at_m(
            H_PP, H_PQ, H_QQ, E0, B1,
            delta=delta, n_states=min(sa_states, len(E_P)),
            C_ref=C_P, verbose=verbose)

        E_eff_m1 = res_m1['E_eff'][0]
        dE_m1_mH = (E_eff_m1 - sys_data['E_fci']) * 1000
        r1 = B1.shape[1]

        res['E_eff_m1'] = E_eff_m1
        res['dE_m1_mH'] = dE_m1_mH
        res['r1'] = r1
        res['layer_sizes'] = layer_sizes

        if verbose:
            print(f"\n  m=1 SUMMARY: E = {E_eff_m1:.12f} Ha, "
                  f"ΔE = {dE_m1_mH:+.3f} mH, r₁ = {r1} "
                  f"(layers: {layer_sizes})", flush=True)

    timing['6_krylov'] = time.perf_counter() - t_krylov

    # ═══════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════
    timing['total'] = time.perf_counter() - t_total

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"FINAL SUMMARY")
        print(f"{'=' * 70}")
        print(f"  E(FCI)      = {sys_data['E_fci']:.12f} Ha")
        print(f"  E(bare H_PP) = {E0:.12f} Ha  "
              f"(ΔE = {(E0 - sys_data['E_fci'])*1000:+.3f} mH)")
        print(f"  E(m=0)       = {E_eff_m0:.12f} Ha  "
              f"(ΔE = {dE_m0_mH:+.3f} mH)")
        if 'E_eff_m1' in res:
            print(f"  E(m=1)       = {res['E_eff_m1']:.12f} Ha  "
                  f"(ΔE = {res['dE_m1_mH']:+.3f} mH)")
        print(f"  Schmidt: r_total={metrics['r_total']}, D={D}, "
              f"|P|={part['p_dim']}, |Q|={part['q_dim']}")
        print(f"  H^emb norms: HA={hemb_norms['norm_HA']:.1f}, "
              f"HB={hemb_norms['norm_HB']:.1f}, "
              f"HAB={hemb_norms['norm_HAB']:.1f}")
        print(f"  Krylov: r₀={r0}" +
              (f", r₁={res.get('r1', 'N/A')}" if m_max >= 1 else ""))
        print(f"\n  Wall time breakdown:")
        for step, t in timing.items():
            pct = t / timing['total'] * 100
            print(f"    {step:20s} {t:8.1f}s  ({pct:5.1f}%)")
        print(f"    {'total':20s} {timing['total']:8.1f}s  (100.0%)", flush=True)

    # ── Build output dict ──
    output = {
        'E_fci': sys_data['E_fci'],
        'E_bare_P': E0,
        'E_eff_m0': E_eff_m0,
        'dE_bare_mH': (E0 - sys_data['E_fci']) * 1000,
        'dE_m0_mH': dE_m0_mH,
        'schmidt_metrics': {
            'r_total': metrics['r_total'],
            'dim_fci': metrics['dim_fci'],
            'compression_ratio': metrics['compression_ratio'],
            'discarded_weight': metrics['discarded_weight'],
        },
        'hemb_norms': hemb_norms,
        'partition_info': {
            'D_total': part['total_dim'],
            'P_dim': part['p_dim'], 'Q_dim': part['q_dim'],
            'p_blocks': p_blocks, 'n_blocks': part['n_blocks'],
        },
        'krylov_dims': {'r0': r0},
        'timing': {k: float(v) for k, v in timing.items()},
    }
    if 'E_eff_m1' in res:
        output['E_eff_m1'] = res['E_eff_m1']
        output['dE_m1_mH'] = res['dE_m1_mH']
        output['krylov_dims']['r1'] = res['r1']
        output['krylov_dims']['layer_sizes'] = [int(x) for x in res['layer_sizes']]

    # ── Save JSON ──
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        fname = os.path.join(output_dir, 'dm_svd_dci_results.json')
        output_serializable = _make_serializable(output)
        with open(fname, 'w') as f:
            json.dump(output_serializable, f, indent=2)
        if verbose:
            print(f"\n  Results saved to {fname}")

    return output


def _make_serializable(obj):
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    return obj