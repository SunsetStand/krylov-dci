#!/usr/bin/env python3
"""
Phase 1 extension: Excited-state energy accuracy with ground-state Schmidt basis.

Pipeline:
  1. Set up N₂/cc-pVDZ → HF → CASCI(10e,10o) with nroots → multiple reference states
  2. Partition determinants by electron count in Space A (5 occ) vs B (5 virt)
  3. SVD on ρ_A^(n) using ONLY ground-state CI vector (single-state mode)
  4. Build H^emb via C-level sigma-vector projection
  5. Diagonalize H^emb → compare first K eigenvalues with CASCI reference states

Key question: How well does the ground-state Schmidt basis describe excited states?

Usage:
  python run_n2_excited.py [--nroots 5] [--eps 1e-3]
"""

import sys, os, time
import argparse
import numpy as np

# Allow imports from parent krylov-dci directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from pyscf import gto, scf, mcscf
from pyscf.fci import cistring, direct_spin1
from pyscf import ao2mo

from src.hamiltonian import _unpack_4fold
from src_mf.pyscf_backend import QSpaceIndex, KDCIBackend

from dm_svd_embedding.occ_virt_partition import (
    setup_partition, build_block_matrices,
)
from dm_svd_embedding.density_matrix import (
    compute_schmidt_decomposition, compute_compression_metrics,
)
from dm_svd_embedding.embedded_hamiltonian import build_h_emb


def main():
    parser = argparse.ArgumentParser(
        description="Excited-state energy accuracy with ground-state Schmidt basis")
    parser.add_argument('--nroots', type=int, default=5,
                        help='Number of CASCI roots / embedded states (default: 5)')
    parser.add_argument('--eps', type=float, default=1e-3,
                        help='SVD truncation threshold (default: 1e-3)')
    args = parser.parse_args()

    N_ROOTS = args.nroots
    EPS = args.eps

    t_total_start = time.perf_counter()

    # ═══════════════════════════════════════════════════════════════
    # 1. Molecule setup: N₂ / cc-pVDZ
    # ═══════════════════════════════════════════════════════════════
    r_n2 = 1.098  # equilibrium bond length in Å
    mol = gto.M(
        atom=f'N 0 0 0; N 0 0 {r_n2}',
        basis='cc-pVDZ',
        verbose=3,
    )
    mf = scf.RHF(mol)
    mf.kernel()
    E_HF = mf.e_tot

    # CAS(10,10): freeze 2 core (N 1s ×2), active = next 10 MOs
    n_core = 2
    n_act = 10
    n_elec = 10
    n_occ = 5   # Space A: HF-occupied active orbitals (indices 0..4 in active)
    n_virt = 5  # Space B: HF-virtual active orbitals (indices 5..9 in active)
    ms = 0

    na = n_elec // 2   # 5
    nb = n_elec - na   # 5

    print(f"{'='*60}")
    print(f"N₂/cc-pVDZ  CAS({n_act},{n_elec})  r(N-N) = {r_n2} Å")
    print(f"Excited-state test: ground-state Schmidt basis, {N_ROOTS} roots")
    print(f"{'='*60}")
    print(f"  Total MOs: {mol.nao}, electrons: {mol.nelec}")
    print(f"  Frozen core: {n_core} orbitals (N 1s ×2)")
    print(f"  Active: {n_act} orbitals, {n_elec} electrons")
    print(f"  Partition: A = [0..{n_occ-1}] (occ), B = [{n_occ}..{n_act-1}] (virt)")
    print(f"  E_HF = {E_HF:.10f}")
    print(f"  SVD threshold: ε = {EPS}")
    print(f"  CASCI nroots: {N_ROOTS}")
    print()

    # ═══════════════════════════════════════════════════════════════
    # 2. CASCI reference (multiple roots)
    # ═══════════════════════════════════════════════════════════════
    print(f"{'─'*60}")
    print(f"Step 1: CASCI reference ({n_act}o, {n_elec}e, nroots={N_ROOTS})")
    print(f"{'─'*60}")

    t_casci = time.perf_counter()
    cas = mcscf.CASCI(mf, n_act, n_elec)
    cas.frozen = n_core
    # NOTE: CASCI nroots must be set on fcisolver, not CASCI object itself
    cas.fcisolver.nroots = N_ROOTS
    h1eff, ecore = cas.get_h1eff()
    h2eff = cas.get_h2eff()
    cas.kernel()

    # CASCI multiple roots: cas.e_tot is list, cas.ci is list of CI matrices
    E_casci_list = cas.e_tot if isinstance(cas.e_tot, (list, tuple, np.ndarray)) else [cas.e_tot]
    ci_list = cas.ci if isinstance(cas.ci, list) else [cas.ci]
    actual_nroots = len(E_casci_list)

    # Ground-state CI vector for SVD
    fcivec_gs = ci_list[0]    # shape (n_alpha_strs, n_beta_strs)
    ci_flat_gs = fcivec_gs.reshape(-1)
    dim_fci = ci_flat_gs.size

    t_casci_elapsed = time.perf_counter() - t_casci

    print(f"  FCI dimension: {dim_fci} determinants")
    print(f"  CASCI reference energies ({actual_nroots} roots):")
    for k in range(actual_nroots):
        dE_ref = (E_casci_list[k] - E_casci_list[0]) * 1000  # mH
        print(f"    State {k}: E = {E_casci_list[k]:.10f}  "
              f"(ΔE from GS = {dE_ref:+.3f} mH)")
    print(f"  CASCI time: {t_casci_elapsed:.1f}s")
    print()

    # ═══════════════════════════════════════════════════════════════
    # 3. PySCF backend: QSpaceIndex + KDCIBackend
    # ═══════════════════════════════════════════════════════════════
    print(f"{'─'*60}")
    print(f"Step 2: Build PySCF backend (QSpaceIndex)")
    print(f"{'─'*60}")

    t_backend = time.perf_counter()
    alpha_strs = cistring.gen_strings4orblist(range(n_act), na)
    beta_strs = cistring.gen_strings4orblist(range(n_act), nb)
    q_idx = QSpaceIndex(alpha_strs, beta_strs, n_act, (na, nb), h1eff, h2eff)
    backend = KDCIBackend(q_idx)
    h2_4d = _unpack_4fold(h2eff, n_act)
    t_backend_elapsed = time.perf_counter() - t_backend

    print(f"  Alpha strings: {q_idx.n_alpha}, Beta strings: {q_idx.n_beta}")
    print(f"  Q-space dim: {q_idx.M}")
    print(f"  Backend setup: {t_backend_elapsed:.1f}s")
    print()

    # ═══════════════════════════════════════════════════════════════
    # 4. Partition determinants by A/B electron count
    #    (same partition for all states)
    # ═══════════════════════════════════════════════════════════════
    print(f"{'─'*60}")
    print(f"Step 3: Occ/Virt partition")
    print(f"{'─'*60}")

    t_part = time.perf_counter()
    partition, full_dets = setup_partition(n_act, n_elec, n_occ, ms=ms)
    C_blocks = build_block_matrices(partition, ci_flat_gs)
    t_part_elapsed = time.perf_counter() - t_part

    # Verify partition coverage
    total_partitioned = sum(blk['n_entries'] for blk in partition.values())
    assert total_partitioned == dim_fci, \
        f"Partition mismatch: {total_partitioned} vs {dim_fci}"

    print(f"  {len(partition)} electron-number blocks:")
    for n_A in sorted(partition.keys(), reverse=True):
        blk = partition[n_A]
        print(f"    n={n_A}: dim_A={blk['dim_A']}, dim_B={blk['dim_B']}, "
              f"n_entries={blk['n_entries']}")
    print(f"  Partition time: {t_part_elapsed:.1f}s")
    print()

    # ═══════════════════════════════════════════════════════════════
    # 5. Density matrix SVD → Schmidt basis (GROUND STATE ONLY)
    # ═══════════════════════════════════════════════════════════════
    print(f"{'─'*60}")
    print(f"Step 4: ρ_A SVD → Schmidt decomposition (ε = {EPS}, GS only)")
    print(f"{'─'*60}")

    t_svd = time.perf_counter()
    schmidt = compute_schmidt_decomposition(C_blocks, eps=EPS)
    metrics = compute_compression_metrics(schmidt, C_blocks, ci_flat_gs)
    t_svd_elapsed = time.perf_counter() - t_svd

    # Print compression results
    print(f"  Compression summary:")
    print(f"    r_total         = {metrics['r_total']}")
    print(f"    dim_FCI         = {metrics['dim_fci']}")
    print(f"    compression_ratio = {metrics['compression_ratio']:.4f} "
          f"({metrics['r_total']}/{metrics['dim_fci']} = "
          f"{metrics['compression_ratio']*100:.1f}%)")
    print(f"    discarded weight  = {metrics['discarded_weight']:.6e}")
    print()
    print(f"  Per-block singular value spectra (n = electrons in Space A):")
    for n_A in sorted(schmidt.keys(), reverse=True):
        sd = schmidt[n_A]
        blk = metrics['per_block'][n_A]
        s_full = sd['sigma_full']
        if len(s_full) > 0:
            s1 = s_full[0]
            s_min = s_full[-1]
            ratio = s_min / max(s1, 1e-14)
            print(f"    n={n_A:2d}: r={blk['r']:3d}/{blk['dim_product']:6d}, "
                  f"σ₁={s1:.4e}, σ_min/σ₁={ratio:.4f}, "
                  f"range=[{s_full[0]:.4e}, {s_full[-1]:.4e}] "
                  f"({blk['r']/max(blk['dim_product'],1)*100:.1f}% retained)")
        else:
            print(f"    n={n_A:2d}: r=0/{blk['dim_product']:6d} (empty block)")
    print(f"  SVD time: {t_svd_elapsed:.1f}s")
    print()

    # ═══════════════════════════════════════════════════════════════
    # 6. Build H^emb in Schmidt product basis
    # ═══════════════════════════════════════════════════════════════
    print(f"{'─'*60}")
    print(f"Step 5: Build H^emb in Schmidt product basis")
    print(f"{'─'*60}")

    t_hemb = time.perf_counter()
    H_emb, basis_info, decomps = build_h_emb(
        schmidt, partition, q_idx, backend, h1eff, h2_4d,
        n_occ, n_act, verbose=True)
    D = H_emb.shape[0]
    t_hemb_elapsed = time.perf_counter() - t_hemb
    print(f"  H^emb build time: {t_hemb_elapsed:.1f}s")
    print()

    # ═══════════════════════════════════════════════════════════════
    # 7. Diagonalize H^emb → compare with CASCI reference
    # ═══════════════════════════════════════════════════════════════
    print(f"{'─'*60}")
    print(f"Step 6: Diagonalize H^emb (D = {D})")
    print(f"{'─'*60}")

    if D == 0:
        print("  ERROR: H^emb has dimension 0 — no Schmidt pairs retained.")
        print("  Try decreasing truncation threshold ε.")
        return 1

    t_diag = time.perf_counter()
    evals_emb, evecs_emb = np.linalg.eigh(H_emb)
    t_diag_elapsed = time.perf_counter() - t_diag

    # Compare first K embedded eigenvalues with CASCI reference
    K_compare = min(actual_nroots, D)
    E_emb_list = evals_emb[:K_compare] + ecore

    print(f"  Excited-state comparison (ground-state Schmidt basis):")
    print(f"  {'State':>6s}  {'E_emb (H)':>14s}  {'E_CASCI (H)':>14s}  "
          f"{'ΔE (mH)':>10s}  {'ΔΔE (mH)':>10s}")
    print(f"  {'─'*6}  {'─'*14}  {'─'*14}  {'─'*10}  {'─'*10}")

    dE_list = []
    for k in range(K_compare):
        dE_abs = (E_emb_list[k] - E_casci_list[k]) * 1000  # mH, absolute error
        dE_rel = ((E_emb_list[k] - E_emb_list[0])
                  - (E_casci_list[k] - E_casci_list[0])) * 1000  # mH, excitation energy error
        dE_list.append(dE_abs)
        print(f"  {k:6d}  {E_emb_list[k]:14.10f}  {E_casci_list[k]:14.10f}  "
              f"{dE_abs:+10.3f}  {dE_rel:+10.3f}")

    if D > K_compare:
        print(f"  ... ({D - K_compare} more embedded states not compared)")

    # Hermiticity check
    asym = np.abs(H_emb - H_emb.T).max()

    # Also print the full excitation spectrum from CASCI for reference
    print()
    print(f"  Full CASCI excitation spectrum ({actual_nroots} roots):")
    print(f"  {'State':>6s}  {'E (H)':>14s}  {'ΔE (mH)':>12s}  "
          f"{'ΔE (eV)':>10s}")
    print(f"  {'─'*6}  {'─'*14}  {'─'*12}  {'─'*10}")
    for k in range(actual_nroots):
        dE_mH = (E_casci_list[k] - E_casci_list[0]) * 1000
        dE_eV = (E_casci_list[k] - E_casci_list[0]) * 27.2114
        marker = " ← GS" if k == 0 else ""
        print(f"  {k:6d}  {E_casci_list[k]:14.10f}  {dE_mH:+12.3f}  "
              f"{dE_eV:+10.4f}{marker}")

    # ═══════════════════════════════════════════════════════════════
    # 8. Summary
    # ═══════════════════════════════════════════════════════════════
    t_total = time.perf_counter() - t_total_start
    print()
    print(f"{'='*60}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  System:        N₂/cc-pVDZ r={r_n2}Å  CAS({n_act},{n_elec})")
    print(f"  FCI dimension: {dim_fci}")
    print(f"  Schmidt rank:  r_total = {metrics['r_total']} "
          f"({metrics['compression_ratio']*100:.1f}%)")
    print(f"  H^emb dim:     D = Σ r_n² = {D}")
    print(f"  SVD basis:     ground-state only (single-state)")
    print()
    print(f"  Energies:")
    for k in range(K_compare):
        print(f"    State {k}: E_emb = {E_emb_list[k]:12.8f}  "
              f"E_CASCI = {E_casci_list[k]:12.8f}  "
              f"ΔE = {dE_list[k]:+.3f} mH")
    print()
    print(f"  H = H_A + H_B + H_AB:")
    print(f"    ||H_A||  = {decomps['norm_HA']:.4f}")
    print(f"    ||H_B||  = {decomps['norm_HB']:.4f}")
    print(f"    ||H_AB|| = {decomps['norm_HAB']:.4f}")
    print(f"  Hermiticity:  max|H - H^T| = {asym:.2e}")
    print(f"  Discarded σ²: {metrics['discarded_weight']:.6e}")
    print()
    print(f"  Timings:")
    print(f"    CASCI ({actual_nroots} roots): {t_casci_elapsed:5.1f}s")
    print(f"    Backend:      {t_backend_elapsed:5.1f}s")
    print(f"    Partition:    {t_part_elapsed:5.1f}s")
    print(f"    SVD:          {t_svd_elapsed:5.1f}s")
    print(f"    H^emb build:  {t_hemb_elapsed:5.1f}s")
    print(f"    Diagonalize:  {t_diag_elapsed:5.1f}s")
    print(f"    ─────────────────────")
    print(f"    Total:        {t_total:5.1f}s")
    print()

    # Save results
    save_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
    os.makedirs(save_dir, exist_ok=True)
    np.savez(
        os.path.join(save_dir, 'n2_excited_results.npz'),
        evals_emb=evals_emb,
        E_casci_list=np.array(E_casci_list),
        E_emb_list=E_emb_list,
        dE_list=np.array(dE_list),
        ecore=ecore,
        nroots=actual_nroots,
        sigma_spectra={str(k): v for k, v in metrics['sigma_spectra'].items()},
        compression_ratio=metrics['compression_ratio'],
        r_total=metrics['r_total'],
        dim_fci=dim_fci,
        D_emb=D,
        eps=EPS,
        t_casci=t_casci_elapsed,
        t_hemb=t_hemb_elapsed,
        t_total=t_total,
    )
    print(f"  Results saved to: {save_dir}/n2_excited_results.npz")

    # Verdict
    max_abs_dE = max(abs(d) for d in dE_list)
    if max_abs_dE < 1.0:
        print(f"  ✓ PASS: max|ΔE| = {max_abs_dE:.3f} mH < 1 mH threshold")
    elif max_abs_dE < 10.0:
        print(f"  ~ ACCEPTABLE: max|ΔE| = {max_abs_dE:.3f} mH < 10 mH threshold")
    else:
        print(f"  ✗ LARGE ERROR: max|ΔE| = {max_abs_dE:.3f} mH > 10 mH — "
              f"ground-state Schmidt basis insufficient for excited states")

    return 0


if __name__ == "__main__":
    exit(main())