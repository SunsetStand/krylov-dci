#!/usr/bin/env python3
"""
Phase 1 prototype: Density Matrix SVD Embedding on N₂ CAS(10,10)/cc-pVDZ.

Pipeline:
  1. Set up N₂/cc-pVDZ → HF → CASCI(10e,10o) with frozen core (N 1s ×2)
  2. Partition determinants by electron count in Space A (5 occ) vs B (5 virt)
  3. SVD on ρ_A^(n) → Schmidt basis (single-state, ε = 1e-3)
  4. Build H^emb via C-level sigma-vector projection
  5. Diagonalize H^emb → compare with CASCI ground state energy

Outputs:
  - Singular value spectra per electron block
  - Compression ratio (r_total / dim_FCI)
  - ||H_A||, ||H_B||, ||H_AB|| norms
  - ΔE = E^emb - E^CASCI (mH)
  - Runtime breakdown
"""

import sys, os, time
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
    print(f"{'='*60}")
    print(f"  Total MOs: {mol.nao}, electrons: {mol.nelec}")
    print(f"  Frozen core: {n_core} orbitals (N 1s ×2)")
    print(f"  Active: {n_act} orbitals, {n_elec} electrons")
    print(f"  Partition: A = [{0}..{n_occ-1}] (occ), B = [{n_occ}..{n_act-1}] (virt)")
    print(f"  E_HF = {E_HF:.10f}")
    print()

    # ═══════════════════════════════════════════════════════════════
    # 2. CASCI reference
    # ═══════════════════════════════════════════════════════════════
    print(f"{'─'*60}")
    print(f"Step 1: CASCI reference ({n_act}o, {n_elec}e)")
    print(f"{'─'*60}")

    t_casci = time.perf_counter()
    cas = mcscf.CASCI(mf, n_act, n_elec)
    cas.frozen = n_core
    h1eff, ecore = cas.get_h1eff()
    h2eff = cas.get_h2eff()
    cas.kernel()
    E_casci = cas.e_tot
    fcivec = cas.ci    # shape (n_alpha_strs, n_beta_strs)
    ci_flat = fcivec.reshape(-1)
    dim_fci = ci_flat.size
    t_casci_elapsed = time.perf_counter() - t_casci

    print(f"  FCI dimension: {dim_fci} determinants")
    print(f"  E_CASCI = {E_casci:.10f}")
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
    # ═══════════════════════════════════════════════════════════════
    print(f"{'─'*60}")
    print(f"Step 3: Occ/Virt partition")
    print(f"{'─'*60}")

    t_part = time.perf_counter()
    partition, full_dets = setup_partition(n_act, n_elec, n_occ, ms=ms)
    C_blocks = build_block_matrices(partition, ci_flat)
    t_part_elapsed = time.perf_counter() - t_part

    # Verify partition coverage
    total_partitioned = sum(blk['n_entries'] for blk in partition.values())
    assert total_partitioned == dim_fci, \
        f"Partition mismatch: {total_partitioned} vs {dim_fci}"

    print(f"  {len(partition)} electron-number blocks:")
    for n_A in sorted(partition.keys()):
        blk = partition[n_A]
        print(f"    n={n_A}: dim_A={blk['dim_A']}, dim_B={blk['dim_B']}, "
              f"n_entries={blk['n_entries']}")
    print(f"  Partition time: {t_part_elapsed:.1f}s")
    print()

    # ═══════════════════════════════════════════════════════════════
    # 5. Density matrix SVD → Schmidt basis
    # ═══════════════════════════════════════════════════════════════
    print(f"{'─'*60}")
    print(f"Step 4: ρ_A SVD → Schmidt decomposition (ε = 1e-3)")
    print(f"{'─'*60}")

    t_svd = time.perf_counter()
    schmidt = compute_schmidt_decomposition(C_blocks, eps=1e-3)
    metrics = compute_compression_metrics(schmidt, C_blocks, ci_flat)
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
    print(f"  Per-block singular value spectra:")
    for n_A in sorted(schmidt.keys()):
        sd = schmidt[n_A]
        blk = metrics['per_block'][n_A]
        s_full = sd['sigma_full']
        if len(s_full) > 0:
            s1 = s_full[0]
            s_min = s_full[-1]
            ratio = s_min / max(s1, 1e-14)
            print(f"    n={n_A}: r={blk['r']}/{blk['dim_product']}, "
                  f"σ₁={s1:.4e}, σ_min/σ₁={ratio:.4f}, "
                  f"range=[{s_full[0]:.4e}, {s_full[-1]:.4e}]")
        else:
            print(f"    n={n_A}: r=0/{blk['dim_product']} (empty block)")
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
    # 7. Diagonalize H^emb
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
    E_emb_gs = evals_emb[0] + ecore
    t_diag_elapsed = time.perf_counter() - t_diag

    dE = (E_emb_gs - E_casci) * 1000  # mH

    # Hermiticity check
    asym = np.abs(H_emb - H_emb.T).max()

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
    print()
    print(f"  Energies:")
    print(f"    E_HF          = {E_HF:12.8f}")
    print(f"    E_CASCI       = {E_casci:12.8f}")
    print(f"    E_emb (GS)    = {E_emb_gs:12.8f}")
    print(f"    ΔE (mH)       = {dE:+12.6f}")
    print()
    print(f"  H = H_A + H_B + H_AB:")
    print(f"    ||H_A||  = {decomps['norm_HA']:.4f}")
    print(f"    ||H_B||  = {decomps['norm_HB']:.4f}")
    print(f"    ||H_AB|| = {decomps['norm_HAB']:.4f}")
    print(f"  Hermiticity:  max|H - H^T| = {asym:.2e}")
    print(f"  Discarded σ²: {metrics['discarded_weight']:.6e}")
    print()
    print(f"  Timings:")
    print(f"    CASCI:        {t_casci_elapsed:5.1f}s")
    print(f"    Backend:      {t_backend_elapsed:5.1f}s")
    print(f"    Partition:    {t_part_elapsed:5.1f}s")
    print(f"    SVD:          {t_svd_elapsed:5.1f}s")
    print(f"    H^emb build:  {t_hemb_elapsed:5.1f}s")
    print(f"    Diagonalize:  {t_diag_elapsed:5.1f}s")
    print(f"    ─────────────────────")
    print(f"    Total:        {t_total:5.1f}s")
    print()

    # Save H_emb eigenvalues for analysis
    np.savez(
        os.path.join(os.path.dirname(__file__), '..', 'logs', 'n2_prototype_results.npz'),
        evals_emb=evals_emb,
        E_casci=E_casci,
        ecore=ecore,
        sigma_spectra={str(k): v for k, v in metrics['sigma_spectra'].items()},
        compression_ratio=metrics['compression_ratio'],
        r_total=metrics['r_total'],
        dim_fci=dim_fci,
        D_emb=D,
        dE_mH=dE,
        t_casci=t_casci_elapsed,
        t_hemb=t_hemb_elapsed,
        t_total=t_total,
    )

    # Verdict
    if abs(dE) < 1.0:
        print(f"  ✓ PASS: ΔE = {dE:+.3f} mH < 1 mH threshold")
    elif abs(dE) < 10.0:
        print(f"  ~ ACCEPTABLE: ΔE = {dE:+.3f} mH < 10 mH threshold")
    else:
        print(f"  ✗ LARGE ERROR: ΔE = {dE:+.3f} mH > 10 mH — check truncation")

    return 0


if __name__ == "__main__":
    exit(main())