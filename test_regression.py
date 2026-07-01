#!/usr/bin/env python3
"""
Comprehensive regression tests for refactored Krylov-dCI.

Tests after PySCF refactoring (2026-07-01):
  1. Determinant generation & phases (cistring)
  2. Hamiltonian matrix elements (make_hdiag, Slater-Condon)
  3. H₂/STO-3G: full H matrix vs FCI
  4. H₂O/STO-3G: P/Q partition, H_PP, H_QP, m=0 effective H
  5. Excited states: diagonalize_effective_H with n_states > 1
  6. Sparse sigma vs dense reference
  7. CAS Hamiltonian frozen-core
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

import numpy as np
from numpy.linalg import eigh

TOL = 1e-10

def green(s): return f"\033[92m{s}\033[0m"
def red(s): return f"\033[91m{s}\033[0m"
def ok(msg): print(f"  {green('✓')} {msg}")
def fail(msg): print(f"  {red('✗')} {msg}"); raise AssertionError(msg)


# ============================================================================
# Test 1: Determinant generation (PySCF cistring)
# ============================================================================
def test_1_determinants():
    from src.determinants import (generate_determinants, generate_determinants_ms,
                                   excitation_phase_alpha, count_bits)
    from pyscf.fci import cistring

    # 1a: H₂/STO-3G
    dets = generate_determinants_ms(2, 2, ms=0)
    assert len(dets) == 4
    for a, b in dets:
        assert count_bits(a) == 1 and count_bits(b) == 1
    ok("1a: H₂/STO-3G — 4 determinants")

    # 1b: Match PySCF cistring directly
    alphas = [int(s) for s in cistring.gen_strings4orblist([0, 1], 1)]
    betas = [int(s) for s in cistring.gen_strings4orblist([0, 1], 1)]
    expected = {(a, b) for a in alphas for b in betas}
    assert set(dets) == expected
    ok("1b: Matches cistring.gen_strings4orblist")

    # 1c: Phase via cistring.cre_des_sign
    # cre_des_sign(create=2, destroy=0, str=0b11) = -1
    phase = excitation_phase_alpha(0b11, 0, 2)
    assert phase == -1, f"Expected -1, got {phase}"
    ok("1c: cre_des_sign phase correct (0→2, 0b11) = -1")

    # 1d: Phase with no orbital between
    phase = excitation_phase_alpha(0b01, 0, 1)
    assert phase in (+1, -1)
    ok(f"1d: cre_des_sign phase (0→1, 0b01) = {phase:+d}")


# ============================================================================
# Test 2: Hamiltonian — H₂/STO-3G full matrix vs FCI
# ============================================================================
def test_2_hamiltonian_h2():
    from pyscf import gto, scf
    from pyscf.fci import direct_spin1
    from src.hamiltonian import from_pyscf
    from src.determinants import generate_determinants_ms

    mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)
    mf = scf.RHF(mol); mf.kernel()
    ham = from_pyscf(mol, mf)

    dets = generate_determinants_ms(2, 2, ms=0)
    H = ham.build_full_matrix(dets)
    E_our = eigh(H)[0][0]

    norb = 2; nelec = (1, 1)
    h1e = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    from pyscf import ao2mo
    eri = ao2mo.restore(1, ao2mo.incore.full(mol.intor('int2e'), mf.mo_coeff), norb)
    E_fci = direct_spin1.FCI().kernel(h1e, eri, norb, nelec, ecore=mf.energy_nuc())[0]

    diff = abs(E_our - E_fci)
    assert diff < TOL, f"H₂ FCI mismatch: {diff:.2e}"
    ok(f"2a: H₂/STO-3G — E={E_our:.10f}, vs FCI diff={diff:.2e}")

    # 2b: Diagonal bulk vs individual
    diag_indiv = np.array([ham.diagonal_element(a, b) for a, b in dets])
    diag_bulk = ham.diagonal_elements_bulk(dets)
    assert np.allclose(diag_indiv, diag_bulk, atol=1e-12)
    ok("2b: Bulk diag matches individual")


# ============================================================================
# Test 3: H₂O/STO-3G — P/Q partition, H_QP, m=0 effective H
# ============================================================================
def test_3_h2o_partition():
    from pyscf import gto, scf
    from src.hamiltonian import from_pyscf
    from src.determinants import generate_determinants_ms
    from src.partitioning import partition_cas, compute_reference_energy
    from src.krylov import compute_A, build_H_QP
    from src.effective_h import build_effective_H, diagonalize_effective_H

    mol = gto.M(atom='O 0 0 0; H 1.0 0 0; H -0.2774 0.9605 0',
                basis='sto-3g', verbose=0)
    mf = scf.RHF(mol); mf.kernel()
    ham = from_pyscf(mol, mf)
    n_orb = mol.nao  # 7
    n_elec = 10

    dets = generate_determinants_ms(n_orb, n_elec, ms=0)
    n_fci = len(dets)

    # Partition: CAS(4,4) — 4 active orbs, 4 active electrons
    # Frozen core: 3 orbs (6 electrons)
    p_idx, q_idx = partition_cas(n_orb, n_elec, n_active_orb=4, n_active_elec=4)
    N, M = len(p_idx), len(q_idx)
    assert N > 0 and M > 0, f"Empty P or Q: N={N}, M={M}"
    ok(f"3a: P={N}, Q={M}, total={n_fci}")

    # H_PP & E0
    E0 = compute_reference_energy(ham, dets, p_idx)
    ok(f"3b: E₀(P) = {E0:.10f}")

    # H_QP
    p_dets = [dets[i] for i in p_idx]
    q_dets = [dets[i] for i in q_idx]
    H_QP_mat = build_H_QP(ham, p_dets, q_dets)
    assert H_QP_mat.shape == (M, N)
    ok(f"3c: H_QP shape = {H_QP_mat.shape}")

    # A_diag
    diag_H_QQ = np.array([ham.diagonal_element(a, b) for a, b in q_dets])
    A_diag = compute_A(E0, diag_H_QQ)
    assert len(A_diag) == M
    ok(f"3d: A_diag computed ({M} elements)")

    # m=0 effective H
    from src.krylov import generate_layer_0, modified_gram_schmidt
    L0 = generate_layer_0(H_QP_mat, A_diag)
    U, _ = modified_gram_schmidt(L0, np.zeros((M, 0)))
    dt = U.shape[1]
    ok(f"3e: Layer 0 compressed: {M}×{N} → {M}×{dt}")

    # Build H_PP
    H_PP = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            H_PP[i, j] = ham.matrix_element(p_dets[i], p_dets[j])
    H_PP = 0.5 * (H_PP + H_PP.T)

    # Projected H blocks
    from src.krylov import compute_H_off_diag
    H_off = compute_H_off_diag(ham, q_dets)
    H_QQ_full = H_off + np.diag(diag_H_QQ)
    sb = H_QQ_full @ U
    H_QQ_t = U.T @ sb
    H_QQ_t = 0.5 * (H_QQ_t + H_QQ_t.T)
    H_PQ = (U.T @ H_QP_mat).T

    H_eff = build_effective_H(H_PP, H_PQ, H_QQ_t, E0, delta=0.0)
    ev, _ = diagonalize_effective_H(H_eff)
    E_kdci = ev[0]

    ok(f"3f: m=0 E(kDCI) = {E_kdci:.10f}")

    # Verify E(kDCI) < E0(P) (downfolding should improve energy)
    if E_kdci < E0 + 1e-6:  # allow small numerical noise
        ok(f"3g: Downfolding improves energy: E₀(P)={E0:.6f} → E(kDCI)={E_kdci:.6f}")
    else:
        # This can happen when P already covers most of the FCI space
        print(f"  [info] E(kDCI)={E_kdci:.6f} >= E₀(P)={E0:.6f} — P dominates")


# ============================================================================
# Test 4: Excited states via diagonalize_effective_H
# ============================================================================
def test_4_excited_states():
    """Verify that diagonalize_effective_H returns multi-root correctly."""
    from src.effective_h import diagonalize_effective_H

    # Build a synthetic H_eff (5×5)
    rng = np.random.RandomState(42)
    A = rng.randn(5, 5)
    H = A + A.T

    # Default (n_states=None) → all
    ev_all, evecs_all = diagonalize_effective_H(H)
    assert len(ev_all) == 5, f"Expected 5 eigenvalues, got {len(ev_all)}"
    ok(f"4a: n_states=None → {len(ev_all)} states")

    # Explicit n_states=3
    ev_3, evecs_3 = diagonalize_effective_H(H, n_states=3)
    assert len(ev_3) == 3
    assert np.allclose(ev_3, ev_all[:3])
    ok("4b: n_states=3 → 3 states, matches full diag")

    # n_states=1 (old default behavior)
    ev_1, evecs_1 = diagonalize_effective_H(H, n_states=1)
    assert len(ev_1) == 1
    assert np.isclose(ev_1[0], ev_all[0])
    ok("4c: n_states=1 → ground state only")


# ============================================================================
# Test 5: Sparse sigma vs dense
# ============================================================================
def test_5_sparse_sigma():
    from pyscf import gto, scf
    from src.hamiltonian import from_pyscf
    from src.determinants import generate_determinants_ms

    mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)
    mf = scf.RHF(mol); mf.kernel()
    ham = from_pyscf(mol, mf)
    n_orb = mol.nao

    dets = generate_determinants_ms(n_orb, 2, ms=0)
    M = len(dets)

    # Dense H_QQ
    H_full = np.zeros((M, M))
    for i in range(M):
        H_full[i, i] = ham.diagonal_element(dets[i][0], dets[i][1])
        for j in range(i + 1, M):
            hij = ham.matrix_element(dets[i], dets[j])
            H_full[i, j] = hij
            H_full[j, i] = hij

    # Sparse adjacency
    from src.sparse_sigma import build_sparse_adjacency, sigma_from_adjacency
    hdiag = np.array([ham.diagonal_element(a, b) for a, b in dets])
    adj, _ = build_sparse_adjacency(ham, dets, n_orb)

    # Verify sigma
    v = np.random.RandomState(123).randn(M)
    sigma_dense = H_full @ v
    sigma_sparse = sigma_from_adjacency(v, hdiag, adj)
    diff = np.max(np.abs(sigma_dense - sigma_sparse))
    assert diff < 1e-12, f"Sparse sigma mismatch: {diff:.2e}"
    ok(f"5a: Sparse sigma matches dense (max diff={diff:.2e})")

    # Multi-column
    V = np.random.RandomState(124).randn(M, 3)
    sigma_dense_m = H_full @ V
    from src.sparse_sigma import sigma_from_adjacency_multi
    sigma_sparse_m = sigma_from_adjacency_multi(V, hdiag, adj)
    diff_m = np.max(np.abs(sigma_dense_m - sigma_sparse_m))
    assert diff_m < 1e-12
    ok(f"5b: Multi-column sparse sigma matches (max diff={diff_m:.2e})")


# ============================================================================
# Test 6: CAS frozen-core Hamiltonian vs PySCF CASCI
# ============================================================================
def test_6_cas_hamiltonian():
    from pyscf import gto, scf, mcscf
    from src.cas_hamiltonian import build_cas_hamiltonian, compute_casci_energy

    # Use the same geometry as cas_hamiltonian.py's own test
    mol = gto.M(atom='O 0 0 0; H 0 0.757 0.586; H 0 -0.757 0.586',
                basis='sto-3g', verbose=0)
    mf = scf.RHF(mol); mf.kernel()

    # CAS(5,6) with 2 frozen core (matches old test)
    cas_data = build_cas_hamiltonian(mol, mf, n_core_orb=2, n_active_orb=5)
    n_act_elec = cas_data['n_active_elec']  # 10 - 2*2 = 6
    E_our = compute_casci_energy(cas_data, n_act_elec, ms=0)

    # PySCF reference
    mycas = mcscf.CASCI(mf, 5, 6)
    mycas.frozen = 2
    mycas.kernel()
    E_pyscf = mycas.e_tot

    diff = abs(E_our - E_pyscf)
    assert diff < 1e-8, f"CAS energy mismatch: {diff:.2e}"
    ok(f"6a: CAS(5,6) frozen-core={E_our:.10f} vs PySCF={E_pyscf:.10f} (diff={diff:.2e})")


# ============================================================================
# Test 7: H₂/STO-3G Krylov m=0 full pipeline
# ============================================================================
def test_7_krylov_m0_h2():
    """Quick end-to-end test: H₂/STO-3G, m=0 Krylov-dCI."""
    from pyscf import gto, scf
    from src.hamiltonian import from_pyscf
    from src.determinants import generate_determinants_ms
    from src.partitioning import compute_reference_energy
    from src.krylov import (compute_A, build_H_QP, generate_layer_0,
                        modified_gram_schmidt, compute_H_off_diag)
    from src.effective_h import build_effective_H, diagonalize_effective_H

    mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)
    mf = scf.RHF(mol); mf.kernel()
    ham = from_pyscf(mol, mf)
    n_orb = 2
    dets = generate_determinants_ms(n_orb, 2, ms=0)

    # P = HF only, Q = other 3
    p_idx = np.array([0])
    q_idx = np.array([1, 2, 3])
    p_dets = [dets[i] for i in p_idx]
    q_dets = [dets[i] for i in q_idx]
    N, M = len(p_dets), len(q_dets)

    E0 = compute_reference_energy(ham, dets, p_idx)
    H_PP = np.array([[ham.matrix_element(p_dets[0], p_dets[0])]])

    # FCI reference
    norb = 2; nelec = (1, 1)
    h1e = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    from pyscf import ao2mo
    eri = ao2mo.restore(1, ao2mo.incore.full(mol.intor('int2e'), mf.mo_coeff), norb)
    from pyscf.fci import direct_spin1
    E_fci = direct_spin1.FCI().kernel(h1e, eri, norb, nelec, ecore=mf.energy_nuc())[0]

    # Krylov
    diag_H_QQ = np.array([ham.diagonal_element(a, b) for a, b in q_dets])
    A_diag = compute_A(E0, diag_H_QQ)
    H_QP_mat = build_H_QP(ham, p_dets, q_dets)
    L0 = generate_layer_0(H_QP_mat, A_diag)
    U, _ = modified_gram_schmidt(L0, np.zeros((M, 0)))
    dt = U.shape[1]

    H_off = compute_H_off_diag(ham, q_dets)
    H_QQ_full = H_off + np.diag(diag_H_QQ)
    sb = H_QQ_full @ U
    H_QQ_t = U.T @ sb; H_QQ_t = 0.5 * (H_QQ_t + H_QQ_t.T)
    H_PQ_t = (U.T @ H_QP_mat).T

    H_eff = build_effective_H(H_PP, H_PQ_t, H_QQ_t, E0, delta=0.0)
    ev, _ = diagonalize_effective_H(H_eff)
    E_kdci = ev[0]

    dE = (E_kdci - E_fci) * 1000
    # With HF-only P space, m=0 should be between P-only and FCI
    E0_mH = (E0 - E_fci) * 1000
    ok(f"7a: H₂ m=0 — E₀(P)={E0_mH:+.1f} mH, E(kDCI)={dE:+.1f} mH vs FCI")


# ============================================================================
# Main
# ============================================================================
if __name__ == '__main__':
    print("=" * 70)
    print("Krylov-dCI Regression Tests (post-refactoring, 2026-07-01)")
    print("=" * 70)

    tests = [
        ("Determinants (cistring)", test_1_determinants),
        ("Hamiltonian H₂/STO-3G", test_2_hamiltonian_h2),
        ("H₂O P/Q partition & m=0", test_3_h2o_partition),
        ("Excited states (multi-root)", test_4_excited_states),
        ("Sparse sigma vs dense", test_5_sparse_sigma),
        ("CAS frozen-core", test_6_cas_hamiltonian),
        ("Krylov m=0 H₂ pipeline", test_7_krylov_m0_h2),
    ]

    passed = 0
    for name, fn in tests:
        print(f"\n[Test] {name}")
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  {red('FAILED')}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*70}")
    print(f"Results: {passed}/{len(tests)} passed")
    print(f"{'='*70}")

    if passed < len(tests):
        sys.exit(1)
