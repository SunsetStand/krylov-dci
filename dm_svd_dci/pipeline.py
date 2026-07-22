#!/usr/bin/env python3
"""
Main pipeline: dmSVD + Krylov-dCI combined method.

Steps:
  1. PySCF molecule setup + CASCI reference
  2. dmSVD: occ-virt determinant partition → density matrix SVD → Schmidt basis
  3. Build H^emb in Schmidt product basis (sigma-vector projection, parallel)
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


# ═══════════════════════════════════════════════════════════════════════════
# Step 1: System setup
# ═══════════════════════════════════════════════════════════════════════════

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
    E_fci = cas.e_tot

    na, nb = n_active_elec
    alpha_strs = cistring.gen_strings4orblist(range(n_act), na)
    beta_strs = cistring.gen_strings4orblist(range(n_act), nb)

    q_idx = QSpaceIndex(alpha_strs, beta_strs, n_act, n_active_elec, h1eff, h2eff)
    backend = KDCIBackend(q_idx)
    M_all = q_idx.M

    h2_4d = _unpack_4fold(h2eff, n_act)
    ham = Hamiltonian(h1=h1eff, h2=h2_4d, E_nuc=ecore, E_HF=0.0)

    if verbose:
        print(f"  System: {atom.strip()}, {basis}")
        print(f"  CAS({n_act},{n_elec_total}): M={M_all:,} dets, n_core={n_core}")
        print(f"  CASCI energy: {E_fci:.12f} Ha")
        print(f"  Setup done: {time.perf_counter() - t0:.0f}s", flush=True)

    return {
        'mol': mol, 'mf': mf,
        'n_active': n_act, 'n_active_elec': n_active_elec, 'n_core': n_core,
        'h1eff': h1eff, 'h2eff': h2eff, 'h2_4d': h2_4d, 'ecore': ecore,
        'fcivec': fcivec, 'ci_flat': ci_flat, 'E_fci': E_fci,
        'alpha_strs': alpha_strs, 'beta_strs': beta_strs,
        'na': na, 'nb': nb, 'M_all': M_all,
        'q_idx': q_idx, 'backend': backend, 'ham': ham,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Step 2-3: Build H^emb in Schmidt product basis (parallel sigma-vectors)
# ═══════════════════════════════════════════════════════════════════════════

def build_hemb_parallel(
    schmidt_data: Dict[int, Dict],
    partition: Dict[int, Dict],
    q_idx,
    backend,
    n_occ: int,
    n_act: int,
    n_workers: int = 1,
    verbose: bool = True,
) -> Tuple[np.ndarray, List[Dict]]:
    """Build H^emb in Schmidt product basis with parallel sigma-vectors."""
    from dm_svd_embedding.embedded_hamiltonian import _expand_schmidt_product_to_ci_matrix
    from dm_svd_dci.parallel_ops import compute_sigma_vectors_parallel

    # ── Build basis index map ──
    basis_info = []
    offset = 0
    block_offsets = {}

    for n_A in sorted(schmidt_data.keys()):
        sd = schmidt_data[n_A]
        r = sd['r']
        block_offsets[n_A] = offset
        for alpha in range(r):
            for beta in range(r):
                basis_info.append({
                    'n': n_A, 'alpha': alpha, 'beta': beta,
                    'flat_idx': offset + alpha * r + beta,
                })
        offset += r * r

    D = offset
    if D == 0:
        return np.zeros((0, 0)), []

    if verbose:
        print(f"  Schmidt product basis dimension: D = {D}")
        for n_A in sorted(schmidt_data.keys()):
            r = schmidt_data[n_A]['r']
            print(f"    n={n_A}: r={r}, r²={r*r}")

    alpha_strs = q_idx.alpha_strs
    beta_strs = q_idx.beta_strs
    n_alpha_strs = len(alpha_strs)
    n_beta_strs = len(beta_strs)
    alpha_to_idx = {int(s): i for i, s in enumerate(alpha_strs)}
    beta_to_idx = {int(s): i for i, s in enumerate(beta_strs)}

    # ── Pre-compute CI matrices for all D basis states ──
    if verbose:
        t0 = time.perf_counter()
        print(f"  Expanding {D} Schmidt basis states to full CAS CI matrices...", flush=True)

    ci_mats = []
    for info in basis_info:
        n_A = info['n']
        alpha = info['alpha']
        beta = info['beta']
        blk_schmidt = schmidt_data[n_A]
        blk_partition = partition[n_A]
        ci_mat = _expand_schmidt_product_to_ci_matrix(
            alpha, beta, blk_schmidt, blk_partition,
            n_alpha_strs, n_beta_strs, n_occ,
            alpha_strs, beta_strs, alpha_to_idx, beta_to_idx)
        ci_mats.append(ci_mat)

    if verbose:
        print(f"    Expansion done: {time.perf_counter() - t0:.0f}s", flush=True)

    # ── Compute sigma vectors in parallel ──
    if verbose:
        t1 = time.perf_counter()
        print(f"  Computing sigma-vectors for {D} basis states ({n_workers} workers)...", flush=True)

    sigmas = compute_sigma_vectors_parallel(
        backend.sigma_full, ci_mats, n_workers=n_workers, verbose=verbose)

    if verbose:
        elapsed = time.perf_counter() - t1
        print(f"    Sigma-vectors done: {elapsed:.0f}s ({elapsed/max(D,1):.2f}s/vector)", flush=True)

    # ── Project: H_emb[k,l] = v_l · sigma_k ──
    if verbose:
        t2 = time.perf_counter()
        print(f"  Projecting {D}×{D} matrix elements...", flush=True)

    ci_flat_mats = [cm.reshape(-1) for cm in ci_mats]
    sigma_flat = [sm.reshape(-1) for sm in sigmas]

    H_emb = np.zeros((D, D))
    for k in range(D):
        sk = sigma_flat[k]
        for l in range(D):
            H_emb[l, k] = np.dot(ci_flat_mats[l], sk)

    H_emb = 0.5 * (H_emb + H_emb.T)

    if verbose:
        elapsed = time.perf_counter() - t2
        asym = np.abs(H_emb - H_emb.T).max()
        print(f"    Projection done: {elapsed:.0f}s, max|H-H^T| = {asym:.2e}", flush=True)

    return H_emb, basis_info


# ═══════════════════════════════════════════════════════════════════════════
# Main pipeline entry point
# ═══════════════════════════════════════════════════════════════════════════

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

    # ═══════════════════════════════════════════════════════════════════════
    # Step 1: System setup
    # ═══════════════════════════════════════════════════════════════════════
    if verbose:
        print("=" * 70)
        print("Step 1: System Setup")
        print("=" * 70)

    sys_data = setup_system(
        atom=atom, basis=basis,
        n_active=n_active, n_active_elec=n_active_elec, n_core=n_core,
        nroots=sa_states, verbose=verbose)
    timing['setup'] = time.perf_counter() - t_total

    # ═══════════════════════════════════════════════════════════════════════
    # Step 2: dmSVD — occ-virt partition + density matrix SVD
    # ═══════════════════════════════════════════════════════════════════════
    if verbose:
        print(f"\n{'=' * 70}")
        print(f"Step 2: dmSVD (occ-virt partition + Schmidt decomposition)")
        print(f"{'=' * 70}")

    t_step2 = time.perf_counter()
    from dm_svd_embedding.occ_virt_partition import (
        setup_partition, build_block_matrices)
    from dm_svd_embedding.density_matrix import (
        compute_schmidt_decomposition, compute_compression_metrics)

    partition, full_dets = setup_partition(n_active, sum(n_active_elec), n_occ, ms=ms)
    C_blocks = build_block_matrices(partition, sys_data['ci_flat'])

    # State-averaging
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
            ci_k = cas2.ci[k]
            Ck_blocks = build_block_matrices(partition, ci_k.reshape(-1))
            state_average.append(Ck_blocks)

    schmidt = compute_schmidt_decomposition(
        C_blocks, eps=svd_eps, state_average=state_average)
    metrics = compute_compression_metrics(schmidt, C_blocks, sys_data['ci_flat'])

    timing['dm_svd'] = time.perf_counter() - t_step2

    if verbose:
        print(f"  r_total = {metrics['r_total']}, "
              f"compression ratio = {metrics['compression_ratio']:.4f}")
        print(f"  dmSVD done: {timing['dm_svd']:.0f}s", flush=True)

    # ═══════════════════════════════════════════════════════════════════════
    # Step 3: Build H^emb in Schmidt product basis
    # ═══════════════════════════════════════════════════════════════════════
    if verbose:
        print(f"\n{'=' * 70}")
        print(f"Step 3: Build H^emb (HA+HB+HAB, parallel sigma-vectors)")
        print(f"{'=' * 70}")

    t_step3 = time.perf_counter()
    H_emb, basis_info = build_hemb_parallel(
        schmidt, partition,
        sys_data['q_idx'], sys_data['backend'],
        n_occ=n_occ, n_act=n_active,
        n_workers=n_workers, verbose=verbose)
    timing['build_hemb'] = time.perf_counter() - t_step3
    D = H_emb.shape[0]

    # ═══════════════════════════════════════════════════════════════════════
    # Step 4: Partition P/Q
    # ═══════════════════════════════════════════════════════════════════════
    if verbose:
        print(f"\n{'=' * 70}")
        print(f"Step 4: Partition Schmidt Basis → P / Q")
        print(f"{'=' * 70}")

    from dm_svd_dci.schmidt_partition import partition_schmidt_basis, extract_subblocks

    part = partition_schmidt_basis(schmidt, p_blocks=p_blocks)
    H_PP, H_PQ, H_QQ = extract_subblocks(H_emb, part)

    if verbose:
        print(f"  Total Schmidt dim: D = {part['total_dim']}")
        print(f"  P-space: |P| = {part['p_dim']} (n ∈ {p_blocks})")
        print(f"  Q-space: |Q| = {part['q_dim']} (n ∉ {p_blocks})")
        print(f"  H_PP: {H_PP.shape}, H_PQ: {H_PQ.shape}, H_QQ: {H_QQ.shape}", flush=True)

    # ═══════════════════════════════════════════════════════════════════════
    # Step 5: Bare P-space diagonalization
    # ═══════════════════════════════════════════════════════════════════════
    if part['p_dim'] == 0:
        print("  WARNING: P-space is empty! Check p_blocks setting.")
        return {'error': 'Empty P-space'}

    E_P, C_P = eigh(H_PP)
    E0 = E_P[0]

    if verbose:
        dE_bare = (E0 - sys_data['E_fci']) * 1000
        print(f"\n  Bare H_PP: E0 = {E0:.12f} Ha, ΔE = {dE_bare:+.3f} mH vs CASCI", flush=True)

    # ═══════════════════════════════════════════════════════════════════════
    # Step 6: Krylov-dCI (MGS only, NO SVD)
    # ═══════════════════════════════════════════════════════════════════════
    if verbose:
        print(f"\n{'=' * 70}")
        print(f"Step 6: Krylov-dCI (MGS only, no SVD)")
        print(f"{'=' * 70}")

    from dm_svd_dci.krylov_propagator import build_krylov_full
    from dm_svd_dci.effective_ham import run_effective_ham_at_m

    res = {}
    t_krylov = time.perf_counter()
    H_QQ_diag = np.diag(H_QQ)

    # ── m=0 ──
    if verbose:
        print(f"\n  --- m=0 ---")

    B0, layers0, A_q = build_krylov_full(
        H_PQ, H_QQ, H_QQ_diag, E0, m_max=0,
        lindep_threshold=lindep_threshold, verbose=verbose)
    r0 = layers0[0]  # extract int from list

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
        print(f"  m=0 result: E = {E_eff_m0:.12f} Ha, ΔE = {dE_m0_mH:+.3f} mH, r₀ = {r0}", flush=True)

    # ── m=1 (if requested) ──
    if m_max >= 1 and r0 > 0:
        if verbose:
            print(f"\n  --- m=1 ---")

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
            print(f"  m=1 result: E = {E_eff_m1:.12f} Ha, ΔE = {dE_m1_mH:+.3f} mH, r₁ = {r1} (layers: {layer_sizes})", flush=True)

    timing['krylov'] = time.perf_counter() - t_krylov

    # ═══════════════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════════════
    timing['total'] = time.perf_counter() - t_total

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"Summary")
        print(f"{'=' * 70}")
        print(f"  CASCI:     {sys_data['E_fci']:.12f} Ha")
        print(f"  Bare H_PP: {E0:.12f} Ha  (ΔE = {(E0 - sys_data['E_fci'])*1000:+.3f} mH)")
        print(f"  m=0:       {E_eff_m0:.12f} Ha  (ΔE = {dE_m0_mH:+.3f} mH)")
        if 'E_eff_m1' in res:
            print(f"  m=1:       {res['E_eff_m1']:.12f} Ha  (ΔE = {res['dE_m1_mH']:+.3f} mH)")
        print(f"  Schmidt dim: D = {D}, |P| = {part['p_dim']}, |Q| = {part['q_dim']}")
        kdim = f"r₀ = {r0}"
        if 'r1' in res:
            kdim += f", r₁ = {res['r1']}"
        print(f"  Krylov: {kdim}")
        print(f"\n  Timing:")
        for step, t in timing.items():
            print(f"    {step}: {t:.0f}s")

    # ── Build return dict ──
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
        'partition_info': {
            'D_total': part['total_dim'],
            'P_dim': part['p_dim'],
            'Q_dim': part['q_dim'],
            'p_blocks': p_blocks,
            'n_blocks': part['n_blocks'],
        },
        'krylov_dims': {'r0': r0},
        'timing': timing,
    }
    if 'E_eff_m1' in res:
        output['E_eff_m1'] = res['E_eff_m1']
        output['dE_m1_mH'] = res['dE_m1_mH']
        output['krylov_dims']['r1'] = res['r1']
        output['krylov_dims']['layer_sizes'] = res['layer_sizes']

    # ── Save ──
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


def test_pipeline_small():
    """Run pipeline on N₂ CAS(6,6)/STO-3G (fast smoke test)."""
    print("=" * 70)
    print("Test: dmSVD + Krylov-dCI on N₂ CAS(6,6)/STO-3G")
    print("=" * 70)
    results = run_dm_svd_dci(
        atom='N 0 0 0; N 0 0 1.098', basis='sto-3g',
        n_active=6, n_active_elec=(3, 3), n_core=0, n_occ=3,
        svd_eps=1e-3, sa_states=1, p_blocks=[4, 5, 6],
        m_max=1, n_workers=1, verbose=True)
    assert 'E_fci' in results
    assert 'E_eff_m0' in results
    assert abs(results['E_eff_m0'] - results['E_fci']) < 1.0
    print("\n  ✓ Pipeline test passed")


if __name__ == "__main__":
    test_pipeline_small()