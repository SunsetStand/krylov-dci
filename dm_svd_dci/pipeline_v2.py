#!/usr/bin/env python3
"""
Main pipeline v2: dmSVD + Neumann series DCI (replaces Krylov-dCI).

Steps:
  1. PySCF molecule setup + CASCI reference
  2. dmSVD: occ-virt determinant partition → density matrix SVD → Schmidt basis
  3. Build H^emb = H_A + H_B + H_AB in Schmidt product basis
  4. Partition Schmidt basis: P = {n ∈ p_blocks}, Q_n for all other n
     Subdivide Q-space by electron count: Q₁, Q₂, ...
     Extract banded-diagonal Hamiltonian blocks using selection rules.
  5. Bare H_PP diagonalization → initial E₀
  6. Self-consistent Neumann solver (k=0 + k=1):
       loop: E₀ → A_n(E₀) → ΔH_PP^(k=0 + k=1) → diagonalize → E₀'
       until |ΔE| < tol
  7. Post-convergence: excited states via root tracking (single-shot)
  8. Compare with CASCI reference

Supports two density-matrix modes:
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


# ═══════════════════════════════════════════════════════════════════════════
# Main pipeline entry point
# ═══════════════════════════════════════════════════════════════════════════

def run_neumann_dci(
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
    k_max: int = 1,
    delta: float = 0.0,
    scf_tol: float = 1e-8,
    scf_max_iter: int = 100,
    n_workers: int = 1,
    scheme: str = 'A',
    batch_size: int = 32,
    output_dir: Optional[str] = None,
    verbose: bool = True,
) -> Dict:
    """Run the complete dmSVD + Neumann series DCI pipeline.

    Args:
        atom:          PySCF molecular geometry string.
        basis:         Basis set name.
        n_active:      Number of active orbitals.
        n_active_elec: (n_alpha, n_beta) active electrons.
        n_core:        Number of frozen core orbitals.
        n_occ:         Number of occupied orbitals (A-space).
        ms:            Spin projection (2*Sz).
        svd_eps:       SVD truncation threshold (relative to σ_max).
        sa_states:     Number of states for state-averaging (1 = ground-state only).
        p_blocks:      List of n values in P-space (e.g. [8,9,10]).
        k_max:         Neumann expansion order (0 or 1).
        delta:         Energy shift Δ (default 0, preserved for future exploration).
        scf_tol:       Self-consistent convergence threshold (Hartree).
        scf_max_iter:  Maximum SCF iterations.
        n_workers:     Number of parallel threads for sigma-vector computation.
        scheme:        'A' = full H^emb, 'B'/'B_streaming' = direct blocks.
        batch_size:    Batch size for streaming mode.
        output_dir:    Directory for JSON output (None = no file output).
        verbose:       Print progress.

    Returns:
        dict with all results.
    """
    t_total = time.perf_counter()
    timing = {}

    # ═══════════════════════════════════════════════════════════════
    # Step 1: System setup
    # ═══════════════════════════════════════════════════════════════
    if verbose:
        print("=" * 70)
        print("STEP 1: System Setup")
        print("=" * 70)

    sys_data = setup_system(
        atom=atom, basis=basis,
        n_active=n_active, n_active_elec=n_active_elec, n_core=n_core,
        nroots=sa_states, verbose=verbose)
    timing['1_setup'] = time.perf_counter() - t_total

    # ═══════════════════════════════════════════════════════════════
    # Step 2: dmSVD
    # ═══════════════════════════════════════════════════════════════
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
    E_casci_list = None
    if sa_states > 1:
        if verbose:
            print(f"  State-averaged mode: {sa_states} states")
        from pyscf import mcscf
        cas2 = mcscf.CASCI(sys_data['mf'], n_active, sum(n_active_elec))
        cas2.frozen = n_core
        cas2.fcisolver.nroots = sa_states
        cas2.kernel()
        if hasattr(cas2, 'e_tot'):
            E_casci_list = np.atleast_1d(np.asarray(cas2.e_tot)).flatten()
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
        for n_A in sorted(schmidt.keys()):
            sd = schmidt[n_A]
            if sd['r'] > 0:
                sig_str = ", ".join(f"{s:.2e}" for s in sd['sigma'][:3])
                print(f"      n={n_A}: r={sd['r']} [{sig_str}...]")
        print(f"  dmSVD done: {timing['2_dm_svd']:.0f}s", flush=True)

    # ═══════════════════════════════════════════════════════════════
    # Step 3+4: Build H^emb + Q-space subdivision
    # ═══════════════════════════════════════════════════════════════
    from dm_svd_dci.schmidt_partition import partition_schmidt_basis
    from dm_svd_dci.qspace_partition import (
        partition_qspace_by_n,
        extract_q_blocks_scheme_a,
    )

    part = partition_schmidt_basis(schmidt, p_blocks=p_blocks)
    D = part['total_dim']

    # Subdivide Q-space by electron count n
    q_partition = partition_qspace_by_n(part, schmidt, p_blocks=p_blocks)

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"STEP 3+4: Build Hamiltonian Blocks + Q-Space Subdivision  (scheme={scheme})")
        print(f"{'=' * 70}")
        print(f"  Total Schmidt dim: D = {D}")
        print(f"  P-space (n ∈ {p_blocks}): |P| = {part['p_dim']}")
        print(f"  Q-space: |Q| = {part['q_dim']} → subdivided into {q_partition['n_total']} blocks")
        print(f"  Active Q_n (|Δn|≤2): {q_partition['active_q']}")

    t_step3 = time.perf_counter()
    hemb_norms = {}
    q_blocks_data = None

    if scheme == 'A':
        # Build full H^emb (reuse existing build_hemb_parallel from legacy pipeline)
        from dm_svd_dci._legacy_pipeline import build_hemb_parallel

        H_emb, basis_info, hemb_norms = build_hemb_parallel(
            schmidt, partition,
            sys_data['q_idx'], sys_data['backend'],
            h1_full=sys_data['h1eff'], h2_full=sys_data['h2_4d'],
            n_occ=n_occ, n_act=n_active,
            n_workers=n_workers, verbose=verbose)
        if D > 0:
            H_emb += sys_data['ecore'] * np.eye(D)

        # Extract Q-blocks from full H^emb
        q_blocks_data = extract_q_blocks_scheme_a(
            H_emb, part, q_partition, p_blocks=p_blocks, verbose=verbose)

    elif scheme in ('B', 'B_streaming'):
        from dm_svd_dci.qspace_partition import extract_q_blocks_scheme_b

        q_blocks_data = extract_q_blocks_scheme_b(
            schmidt, partition, part, q_partition,
            p_blocks, sys_data['backend'],
            n_occ, n_active,
            n_workers=n_workers, ecore=sys_data['ecore'],
            verbose=verbose)

        # Build H^emb norms from the extracted blocks (approximate)
        hemb_norms = {
            'norm_HA': 0.0, 'norm_HB': 0.0,
            'norm_HAB': np.linalg.norm(q_blocks_data['H_PP']),
            'norm_total': np.linalg.norm(q_blocks_data['H_PP']),
            'asymmetry': 0.0,
        }
    else:
        raise ValueError(f"Unknown scheme: {scheme}")

    timing['3_build_hemb'] = time.perf_counter() - t_step3

    H_PP = q_blocks_data['H_PP']
    H_PQ = q_blocks_data['H_PQ']
    H_QQ_blocks = q_blocks_data['H_QQ_blocks']
    H_QQ_diag = q_blocks_data['H_QQ_diag']

    if verbose:
        print(f"\n  Extracted blocks summary:")
        print(f"    H_PP: {H_PP.shape}")
        for n_Q in sorted(H_PQ.keys()):
            print(f"    H_PQ_{n_Q}: {H_PQ[n_Q].shape}")
        print(f"    H_QQ diagonal blocks: {len(H_QQ_diag)} blocks")
        print(flush=True)

    # ═══════════════════════════════════════════════════════════════
    # Step 5: Bare H_PP diagonalization
    # ═══════════════════════════════════════════════════════════════
    if part['p_dim'] == 0:
        print("  ERROR: P-space is empty! Check p_blocks setting.")
        return {'error': 'Empty P-space'}

    E_P, C_P = eigh(H_PP)
    E0_bare = E_P[0]

    if verbose:
        dE_bare = (E0_bare - sys_data['E_fci']) * 1000
        print(f"\n  Bare H_PP diagonalization:")
        print(f"    E0 (lowest)    = {E0_bare:.12f} Ha")
        print(f"    ΔE vs FCI      = {dE_bare:+.3f} mH")
        if len(E_P) >= 5:
            print(f"    First 5 eigenvalues:")
            for k in range(min(5, len(E_P))):
                exc = (E_P[k] - E_P[0]) * 1000 if k > 0 else 0.0
                print(f"      S{k}: {E_P[k]:.12f} Ha  ({exc:+.1f} mH exc)")
        print(flush=True)

    # ═══════════════════════════════════════════════════════════════
    # Step 6: Self-consistent Neumann solver
    # ═══════════════════════════════════════════════════════════════
    if verbose:
        print(f"\n{'=' * 70}")
        print(f"STEP 6: Self-Consistent Neumann Solver (k={k_max})")
        print(f"{'=' * 70}")

    from dm_svd_dci.self_consistent_solver import (
        solve_self_consistent,
        evaluate_excited_states,
    )

    t_scf = time.perf_counter()

    sc_result = solve_self_consistent(
        H_PP, H_PQ, H_QQ_blocks, H_QQ_diag,
        delta=delta, k_max=k_max,
        tol=scf_tol, max_iter=scf_max_iter,
        E0_init=E0_bare, verbose=verbose)

    timing['6_neumann_scf'] = time.perf_counter() - t_scf

    E_conv = sc_result['E_conv']
    dE_conv_mH = (E_conv - sys_data['E_fci']) * 1000

    if verbose:
        print(f"\n  Neumann SCF SUMMARY:")
        print(f"    E(bare H_PP)   = {E0_bare:.12f} Ha  "
              f"(ΔE = {(E0_bare - sys_data['E_fci'])*1000:+.3f} mH)")
        print(f"    E(conv)        = {E_conv:.12f} Ha  "
              f"(ΔE = {dE_conv_mH:+.3f} mH)")
        print(f"    Iterations      = {sc_result['n_iter']}, "
              f"converged={sc_result['converged']}")
        print(flush=True)

    # ═══════════════════════════════════════════════════════════════
    # Step 7: Excited states (single-shot, non-self-consistent)
    # ═══════════════════════════════════════════════════════════════
    exc_result = None
    if sa_states > 1:
        if verbose:
            print(f"\n  --- Excited states (single-shot at E₀={E_conv:.12f}) ---")

        exc_result = evaluate_excited_states(
            H_PP, H_PQ, H_QQ_blocks, H_QQ_diag,
            E0_gs=E_conv, delta=delta, k_max=k_max,
            n_states=sa_states, C_ref=C_P, verbose=verbose)

    # ═══════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════
    timing['total'] = time.perf_counter() - t_total

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"FINAL SUMMARY")
        print(f"{'=' * 70}")
        print(f"  E(FCI)          = {sys_data['E_fci']:.12f} Ha")
        print(f"  E(bare H_PP)     = {E0_bare:.12f} Ha  "
              f"(ΔE = {(E0_bare - sys_data['E_fci'])*1000:+.3f} mH)")
        print(f"  E(Neumann k={k_max}) = {E_conv:.12f} Ha  "
              f"(ΔE = {dE_conv_mH:+.3f} mH)")
        print(f"  Schmidt: r_total={metrics['r_total']}, D={D}, "
              f"|P|={part['p_dim']}, |Q|={part['q_dim']}")
        if H_QQ_diag:
            q_dims = {n: len(d) for n, d in H_QQ_diag.items()}
            print(f"  Q_n dims: {q_dims}")
        if hemb_norms:
            print(f"  H^emb norms: HA={hemb_norms.get('norm_HA', 0):.1f}, "
                  f"HB={hemb_norms.get('norm_HB', 0):.1f}, "
                  f"HAB={hemb_norms.get('norm_HAB', 0):.1f}")

        if exc_result is not None and sa_states > 1:
            print(f"\n  Excited states:")
            print(f"  {'State':>5} {'E_neumann':>16} {'ΔE vs CASCI':>14}")
            print(f"  {'-'*43}")
            for k in range(min(sa_states, len(exc_result['E_excited']))):
                E_ps = exc_result['E_excited'][k]
                E_ref_k = (E_casci_list[k] if E_casci_list is not None
                           and k < len(E_casci_list) else sys_data['E_fci'])
                dE = (E_ps - E_ref_k) * 1000
                print(f"  {'S'+str(k):>5} {E_ps:>16.12f} {dE:>+13.3f} mH")

        print(f"\n  Wall time breakdown:")
        for step, t in timing.items():
            pct = t / timing['total'] * 100
            print(f"    {step:20s} {t:8.1f}s  ({pct:5.1f}%)")
        print(f"    {'total':20s} {timing['total']:8.1f}s  (100.0%)", flush=True)

    # ── Build output dict ──
    output = {
        'E_fci': sys_data['E_fci'],
        'E_bare_P': E0_bare,
        'dE_bare_mH': (E0_bare - sys_data['E_fci']) * 1000,
        'E_neumann_k0': E_conv if k_max == 0 else None,
        'E_neumann_k1': E_conv if k_max >= 1 else None,
        'E_conv': E_conv,
        'dE_conv_mH': dE_conv_mH,
        'scf_converged': sc_result['converged'],
        'scf_n_iter': sc_result['n_iter'],
        'scf_E_history': sc_result['E_history'],
        'schmidt_metrics': {
            'r_total': metrics['r_total'],
            'dim_fci': metrics['dim_fci'],
            'compression_ratio': metrics['compression_ratio'],
            'discarded_weight': metrics['discarded_weight'],
        },
        'hemb_norms': {k: float(v) for k, v in hemb_norms.items()}
        if hemb_norms else {},
        'partition_info': {
            'D_total': part['total_dim'],
            'P_dim': part['p_dim'],
            'Q_dim': part['q_dim'],
            'Q_by_n': {int(n): q_partition['q_blocks'][n]['dim']
                       for n in q_partition['q_n_list']},
            'p_blocks': p_blocks,
            'n_blocks': part['n_blocks'],
            'needed_q': q_blocks_data.get('needed_q', []),
        },
        'correction_norms': {
            'norm_Delta_k0': float(np.linalg.norm(sc_result['Delta_k0'])),
            'norm_Delta_k1': float(np.linalg.norm(sc_result['Delta_k1'])),
            'norm_H_PP': float(np.linalg.norm(H_PP)),
        },
        'timing': {k: float(v) for k, v in timing.items()},
    }

    if exc_result is not None:
        output['E_excited'] = [float(e) for e in exc_result['E_excited']]
        output['overlaps_excited'] = [float(o) for o in exc_result['overlaps']]
        if E_casci_list is not None:
            output['E_casci_excited'] = [float(e) for e in E_casci_list]

    # ── Save JSON ──
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        fname = os.path.join(output_dir, 'neumann_dci_results.json')
        output_serializable = _make_serializable(output)
        with open(fname, 'w') as f:
            json.dump(output_serializable, f, indent=2)
        if verbose:
            print(f"\n  Results saved to {fname}")

    return output


def _make_serializable(obj):
    """Recursively convert numpy types to native Python for JSON serialization."""
    if isinstance(obj, dict):
        return {str(k): _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif callable(obj):
        return '<callable>'
    return obj


# ═══════════════════════════════════════════════════════════════════════════
# Quick test entry point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("pipeline_v2.py — Neumann DCI pipeline")
    print("Usage: from dm_svd_dci.pipeline_v2 import run_neumann_dci")
    print("This module is designed to be imported, not run directly.")