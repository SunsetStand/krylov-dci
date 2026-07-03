#!/usr/bin/env python3
"""
N₂/cc-pVDZ P-Space Strategy Test — Phase 4

Tests multiple P-space selection strategies at m=0 (no Krylov layers)
within a CAS(8,8) active space, frozen-core.

Key metrics (per strategy, per bond length):
  - N_det_P (P-space size)
  - ΔE vs CASCI reference (mH)
  - t_wall (seconds)
  - Effective compression: N_det_P / N_det_CAS

Strategies tested:
  A. CAS(4,4): P = full CAS(4,4) = 36 dets
  B. CAS(6,6): P = full CAS(6,6) = 400 dets  
  C. Energy-window: P = dets with |H_ii - E_HF| < ΔE
  D. PT2 selection: P = dets with |<HF|H|det>|²/ΔE > θ
  E. Single-det: P = HF = 1 det (extreme limit)

Geometries: Re=1.098Å, 1.5Re, 2.0Re, 2.5Re

Output: tables for report + timing data
"""

import sys, os, time
import numpy as np

sys.path.insert(0, '/data/home/wangcx/krylov-dci/src')

from pyscf import gto, scf, mcscf, ao2mo
from pyscf.fci.direct_nosym import FCI
from hamiltonian import Hamiltonian, from_pyscf, _unpack_4fold
from determinants import generate_determinants_ms
from partitioning import partition_cas, compute_reference_energy
from krylov import compute_A, compute_H_off_diag, build_H_QP
from effective_h import (
    build_H_Qtilde_Qtilde, build_H_PQtilde,
    build_effective_H, self_consistent_iteration
)
from svd_compression import build_weighted_coupling, svd_truncate, compress_layer

np.set_printoptions(linewidth=120, precision=6, suppress=True)

# ============================================================================
# Configuration
# ============================================================================
N_CORE = 3        # Frozen core orbitals (N 1s ×2 + N 2s? depends on ordering)
N_ACTIVE_ORB = 8  # CAS orbitals
N_ACTIVE_ELEC = 8 # Active electrons
BOND_LENGTHS = [1.098, 1.647, 2.196, 2.745]  # Re, 1.5Re, 2.0Re, 2.5Re (Å)

# ============================================================================
# Helper: Extract CAS Integrals from PySCF CASCI
# ============================================================================

def get_cas_integrals(mol, mf, n_core, n_active_orb, n_active_elec):
    """Extract active-space integrals and CASCI reference energy.

    Uses PySCF CASCI with frozen core to:
      1. Compute the correct CASCI total energy (our reference)
      2. Transform AO integrals to active-space MO basis
      3. Compute effective 1e integrals (with frozen core folded in)

    Returns:
        dict with h1_eff, h2_eff, E_casci, E_nuc, active_mo_coeff, n_orb_active
    """
    # Run CASCI to get the proper orbital selection and energy
    mycas = mcscf.CASCI(mf, n_active_orb, n_active_elec)
    mycas.frozen = n_core
    mycas.verbose = 0
    mycas.kernel()
    E_casci = mycas.e_tot

    # Extract the active-space MO coefficients
    mo_coeff = mycas.mo_coeff  # CASCI-optimized orbitals
    n_core_orb = n_core
    active_mo = mo_coeff[:, n_core_orb:n_core_orb + n_active_orb]

    # AO integrals
    h1_ao = mol.intor_symmetric('int1e_kin') + mol.intor_symmetric('int1e_nuc')
    eri_ao = mol.intor('int2e')

    # Active-space 1e integrals (bare, no frozen core)
    h1_active_bare = active_mo.T @ h1_ao @ active_mo

    # Active-space 2e integrals
    eri_active_packed = ao2mo.incore.full(eri_ao, active_mo)
    h2_active = _unpack_4fold(eri_active_packed, n_active_orb)

    # Compute ecore (everything except active-space energy)
    # ecore = E_casci - E_active(FCI in bare h1/h2)
    fci_solver = FCI()
    fci_solver.verbose = 0
    n_alpha = n_active_elec // 2
    n_beta = n_active_elec - n_alpha
    E_active_bare, _ = fci_solver.kernel(
        h1_active_bare, h2_active, n_active_orb,
        (n_alpha, n_beta), ecore=0.0
    )
    ecore = E_casci - E_active_bare

    # Effective 1e integrals: force FCI(h1_eff, h2) + ecore = E_casci
    # h1_eff = h1_active_bare (already includes core effects via CASCI MOs)
    h1_eff = h1_active_bare

    return {
        'h1_eff': h1_eff,
        'h2_eff': h2_active,
        'E_casci': E_casci,
        'ecore': ecore,
        'n_active_orb': n_active_orb,
        'n_active_elec': n_active_elec,
    }


# ============================================================================
# P-Space Selection Strategies
# ============================================================================

def strategy_sub_cas(dets, ham, n_active_orb, n_active_elec,
                     sub_n_orb, sub_n_elec):
    """P = full CAS(sub_n_orb, sub_n_elec) within the active space."""
    p_idx, q_idx = partition_cas(
        n_active_orb, n_active_elec,
        n_active_orb=sub_n_orb, n_active_elec=sub_n_elec
    )
    return p_idx, q_idx


def strategy_energy_window(dets, ham, E_hf, delta_e):
    """P = determinants with |H_ii - E_HF| < delta_e."""
    p_idx = []
    q_idx = []
    for i, (a, b) in enumerate(dets):
        e_diag = ham.diagonal_element(a, b)
        if abs(e_diag - E_hf) < delta_e:
            p_idx.append(i)
        else:
            q_idx.append(i)
    return np.array(p_idx), np.array(q_idx)


def strategy_pt2(dets, ham, hf_idx, E_hf, threshold):
    """P = determinants with PT2 contribution > threshold.

    PT2 contribution: |<HF|H|det>|² / (E_HF - H_ii)
    """
    hf_det = dets[hf_idx]
    pt2_contribs = []
    for i, det_i in enumerate(dets):
        if i == hf_idx:
            continue
        hij = ham.matrix_element(hf_det, det_i)
        if abs(hij) < 1e-14:
            pt2_contribs.append((i, 0.0))
        else:
            e_i = ham.diagonal_element(det_i[0], det_i[1])
            denom = E_hf - e_i
            if abs(denom) < 1e-14:
                contrib = 1e10  # Near-degenerate → always in P
            else:
                contrib = abs(hij**2 / denom)
            pt2_contribs.append((i, contrib))

    p_idx = [hf_idx]  # HF always in P
    q_idx = []
    for i, contrib in pt2_contribs:
        if contrib > threshold:
            p_idx.append(i)
        else:
            q_idx.append(i)
    return np.array(p_idx), np.array(q_idx)


def strategy_single_det(dets, hf_idx):
    """P = single HF determinant (extreme limit)."""
    M = len(dets)
    p_idx = np.array([hf_idx])
    q_idx = np.array([i for i in range(M) if i != hf_idx])
    return p_idx, q_idx


# ============================================================================
# Run single configuration
# ============================================================================

def run_strategy(name, p_idx, q_idx, dets, ham, E_casci, ecore, verbose=True):
    """Run Krylov-dCI m=0 for a given P/Q partition and return results."""
    t0 = time.perf_counter()

    p_dets = [dets[i] for i in p_idx]
    q_dets = [dets[i] for i in q_idx]
    N = len(p_dets)
    M = len(q_dets)

    # Reference energy from H_PP
    E0 = compute_reference_energy(ham, dets, p_idx)

    # SVD compression at m=0 (layer 0 only)
    diag_H_QQ = np.array([ham.diagonal_element(a, b) for a, b in q_dets])
    A_diag = compute_A(E0, diag_H_QQ)
    H_QP_mat = build_H_QP(ham, p_dets, q_dets)

    # Build H_PP
    H_PP = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            H_PP[i, j] = ham.matrix_element(p_dets[i], p_dets[j])

    # Layer 0 with SVD compression
    from krylov import generate_layer_0
    layer0_raw = generate_layer_0(H_QP_mat, A_diag)

    if layer0_raw.shape[1] > 0:
        U_comp, sigma_comp, r = compress_layer(
            layer0_raw, A_diag, threshold=1e-3, verbose=False
        )
    else:
        U_comp = np.zeros((M, 0))
        r = 0

    if r > 0:
        # Orthonormalize
        from krylov import modified_gram_schmidt
        U_comp, _ = modified_gram_schmidt(U_comp, np.zeros((M, 0)))
        d = U_comp.shape[1]

        # Build compressed H blocks
        H_QQ_full = compute_H_off_diag(ham, q_dets) + np.diag(diag_H_QQ)
        H_QQ_tilde = build_H_Qtilde_Qtilde(ham, U_comp, q_dets,
                                           H_QQ_full=H_QQ_full)
        H_PQ_tilde = build_H_PQtilde(ham, U_comp, p_dets, q_dets)

        # Self-consistent effective Hamiltonian
        result = self_consistent_iteration(
            H_PP, H_PQ_tilde, H_QQ_tilde, E0, verbose=False
        )
        E_method = result['E_final']
        n_iter = result['n_iter']
    else:
        # No Q-space: effective H = H_PP
        from numpy.linalg import eigh
        eigvals, _ = eigh(H_PP)
        E_method = eigvals[0]
        d = 0
        n_iter = 1

    t_wall = time.perf_counter() - t0

    # Total energy = effective energy + ecore
    E_total = E_method + ecore
    dE_mH = (E_total - E_casci) * 1000.0

    return {
        'name': name,
        'N_det_P': N,
        'N_det_Q': M,
        'd_compressed': d,
        'E_total': E_total,
        'dE_mH': dE_mH,
        'n_iter': n_iter,
        't_wall': t_wall,
    }


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 80)
    print("Phase 4: N₂/cc-pVDZ P-Space Strategy Test (m=0)")
    print("=" * 80)

    all_results = []

    for R in BOND_LENGTHS:
        print(f"\n{'='*60}")
        print(f"Bond length: R = {R:.3f} Å")
        print(f"{'='*60}")

        t_setup = time.perf_counter()

        # Build N₂ molecule
        mol = gto.M(
            atom=f'N 0 0 0; N 0 0 {R}',
            basis='cc-pVDZ',
            verbose=0
        )
        mf = scf.RHF(mol)
        mf.kernel()

        # Get CAS integrals via PySCF CASCI
        cas = get_cas_integrals(mol, mf, N_CORE, N_ACTIVE_ORB, N_ACTIVE_ELEC)
        E_casci = cas['E_casci']
        ecore = cas['ecore']
        n_orb = cas['n_active_orb']
        n_elec = cas['n_active_elec']
        print(f"  E_CASCI(8,8) = {E_casci:.10f}")
        print(f"  E_HF          = {mf.e_tot:.10f}")

        # Build Hamiltonian in active space
        ham = Hamiltonian(
            h1=cas['h1_eff'],
            h2=cas['h2_eff'],
            E_nuc=0.0,  # ecore handled separately
            E_HF=mf.e_tot  # Not used here
        )

        # Generate all CAS-space determinants
        dets = generate_determinants_ms(n_orb, n_elec, ms=0)
        n_dets_total = len(dets)
        print(f"  CAS dim = {n_dets_total} determinants")
        t_setup_elapsed = time.perf_counter() - t_setup
        print(f"  Setup time: {t_setup_elapsed:.2f}s")

        # Find HF index
        # HF in active space: lowest n_alpha α orbitals + lowest n_beta β
        n_alpha = n_elec // 2
        n_beta = n_elec - n_alpha
        alpha_hf = (1 << n_alpha) - 1  # bits 0..n_alpha-1 set
        beta_hf = (1 << n_beta) - 1
        hf_idx = None
        for i, (a, b) in enumerate(dets):
            if a == alpha_hf and b == beta_hf:
                hf_idx = i
                break
        if hf_idx is None:
            print("  ⚠ HF determinant not found in active space!")
            continue
        E_hf_det = ham.diagonal_element(alpha_hf, beta_hf)
        print(f"  HF det energy: {E_hf_det:.10f}")

        # ================================================================
        # Strategy A: Sub-CAS(4,4)
        # ================================================================
        if n_elec >= 4 and n_orb >= 4:
            p_a, q_a = strategy_sub_cas(
                dets, ham, n_orb, n_elec, sub_n_orb=4, sub_n_elec=4
            )
            result = run_strategy(
                "A: CAS(4,4)", p_a, q_a, dets, ham, E_casci, ecore
            )
            print(f"  {result['name']:20s}: P={result['N_det_P']:4d}, "
                  f"d={result['d_compressed']:3d}, "
                  f"ΔE={result['dE_mH']:+.3f} mH, "
                  f"{result['t_wall']:.3f}s")
            all_results.append((R, result))

        # ================================================================
        # Strategy B: Sub-CAS(6,6)
        # ================================================================
        if n_elec >= 6 and n_orb >= 6:
            p_b, q_b = strategy_sub_cas(
                dets, ham, n_orb, n_elec, sub_n_orb=6, sub_n_elec=6
            )
            result = run_strategy(
                "B: CAS(6,6)", p_b, q_b, dets, ham, E_casci, ecore
            )
            print(f"  {result['name']:20s}: P={result['N_det_P']:4d}, "
                  f"d={result['d_compressed']:3d}, "
                  f"ΔE={result['dE_mH']:+.3f} mH, "
                  f"{result['t_wall']:.3f}s")
            all_results.append((R, result))

        # ================================================================
        # Strategy C: Energy Window (multiple widths)
        # ================================================================
        for de in [0.5, 1.0, 2.0]:
            p_c, q_c = strategy_energy_window(dets, ham, E_hf_det, de)
            if len(p_c) > 0:
                label = f"C: EW ΔE={de:.1f}"
                result = run_strategy(
                    label, p_c, q_c, dets, ham, E_casci, ecore
                )
                print(f"  {result['name']:20s}: P={result['N_det_P']:4d}, "
                      f"d={result['d_compressed']:3d}, "
                      f"ΔE={result['dE_mH']:+.3f} mH, "
                      f"{result['t_wall']:.3f}s")
                all_results.append((R, result))

        # ================================================================
        # Strategy D: PT2 Selection
        # ================================================================
        for thresh in [1e-5, 1e-4, 1e-3]:
            p_d, q_d = strategy_pt2(dets, ham, hf_idx, E_hf_det, thresh)
            if len(p_d) > 1:
                label = f"D: PT2 θ={thresh:.0e}"
                result = run_strategy(
                    label, p_d, q_d, dets, ham, E_casci, ecore
                )
                print(f"  {result['name']:20s}: P={result['N_det_P']:4d}, "
                      f"d={result['d_compressed']:3d}, "
                      f"ΔE={result['dE_mH']:+.3f} mH, "
                      f"{result['t_wall']:.3f}s")
                all_results.append((R, result))

        # ================================================================
        # Strategy E: Single Determinant (Extreme Limit)
        # ================================================================
        p_e, q_e = strategy_single_det(dets, hf_idx)
        result = run_strategy(
            "E: Single-det", p_e, q_e, dets, ham, E_casci, ecore
        )
        print(f"  {result['name']:20s}: P={result['N_det_P']:4d}, "
              f"d={result['d_compressed']:3d}, "
              f"ΔE={result['dE_mH']:+.3f} mH, "
              f"{result['t_wall']:.3f}s")
        all_results.append((R, result))

    # ================================================================
    # Summary Table
    # ================================================================
    print("\n" + "=" * 80)
    print("SUMMARY: ΔE vs N_det_P at m=0")
    print("=" * 80)
    for R in BOND_LENGTHS:
        print(f"\n  R = {R:.3f} Å:")
        print(f"  {'Strategy':25s}  {'P_size':>6s}  {'d_comp':>6s}  "
              f"{'ΔE(mH)':>10s}  {'t(s)':>8s}")
        print(f"  {'-'*25}  {'-'*6}  {'-'*6}  {'-'*10}  {'-'*8}")
        for r_r, r_data in all_results:
            if abs(r_r - R) < 0.01:
                print(f"  {r_data['name']:25s}  {r_data['N_det_P']:6d}  "
                      f"{r_data['d_compressed']:6d}  "
                      f"{r_data['dE_mH']:+10.3f}  {r_data['t_wall']:8.3f}")

    print("\nPhase 4 N₂ analysis complete.")


if __name__ == '__main__':
    main()
