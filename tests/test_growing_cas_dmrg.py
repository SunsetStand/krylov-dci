#!/usr/bin/env python3
"""
Smoke tests for GrowingCASDMRG pipeline.

Tests:
  1. ChainedTransform unit tests
  2. build_T0_from_schmidt unit test
  3. block_svd_multi_orbital unit tests
  4. Full pipeline smoke test on H₂O/STO-3G CAS(5,6) — small enough to run quickly
  5. Full pipeline on N₂/cc-pVDZ CAS(10,10) — integration test (optional, requires PySCF)
"""

import sys, os
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def test_chained_transform_basic():
    """Test ChainedTransform: init, extend, get_full_transform."""
    from dm_svd_dci.growing_cas_dmrg import ChainedTransform

    # T_0 with 3 blocks
    T0 = {
        0: np.eye(4, 2),
        1: np.eye(6, 3),
        2: np.eye(3, 2),
    }
    ct = ChainedTransform(T0)
    assert ct.total_dimension == 7  # 2+3+2
    assert ct.n_blocks == {0, 1, 2}
    print("  ✓ ChainedTransform init: correct")

    # Extend with SVD result for block 1 only
    U_new = {1: np.eye(3, 2)}
    ct.extend(U_new)
    assert ct.total_dimension == 6  # 2+2+2
    print("  ✓ ChainedTransform extend: dimension reduced correctly")

    # Full transform
    T_full = ct.get_full_transform(1)
    assert T_full.shape == (6, 2)  # T0[1](6,3) @ U_new[1](3,2) = (6,2)
    print("  ✓ ChainedTransform get_full_transform: shape correct")


def test_chained_transform_compress():
    """Test ChainedTransform.compress_ci_matrix."""
    from dm_svd_dci.growing_cas_dmrg import ChainedTransform

    T0 = {1: np.eye(6, 3)}  # (6, 3)
    ct = ChainedTransform(T0)
    U_new = {1: np.eye(3, 2)}  # (3, 2)
    ct.extend(U_new)

    C = np.random.randn(6, 4)  # block 1: d_old=6, d_B=4
    C_tilde = ct.compress_ci_matrix(C, 1)
    assert C_tilde.shape == (2, 4)  # (r=2, d_B=4)
    print("  ✓ ChainedTransform compress_ci_matrix: shape correct")

    # Verify: T_full.T @ C should equal C_tilde
    T_full = ct.get_full_transform(1)
    expected = T_full.T @ C
    assert np.allclose(C_tilde, expected, atol=1e-12)
    print("  ✓ ChainedTransform compress_ci_matrix: values correct")


def test_chained_transform_missing_block():
    """Test compress_ci_matrix for a block not in T_0."""
    from dm_svd_dci.growing_cas_dmrg import ChainedTransform

    T0 = {0: np.eye(2, 1)}
    ct = ChainedTransform(T0)

    # Block n_A=5 not in T_0; should return unchanged
    C = np.random.randn(10, 3)
    C_tilde = ct.compress_ci_matrix(C, 5)
    assert C_tilde.shape == (10, 3)
    assert np.allclose(C_tilde, C)
    print("  ✓ ChainedTransform: missing block handled correctly")


def test_chained_transform_multi_extend():
    """Test multiple extensions."""
    from dm_svd_dci.growing_cas_dmrg import ChainedTransform

    T0 = {0: np.eye(8, 5)}
    ct = ChainedTransform(T0)
    assert ct.total_dimension == 5

    # Round 1: reduce
    ct.extend({0: np.eye(5, 3)})
    assert ct.total_dimension == 3

    # Round 2: reduce further
    ct.extend({0: np.eye(3, 2)})
    assert ct.total_dimension == 2

    T_final = ct.get_full_transform(0)
    assert T_final.shape == (8, 2)
    assert np.allclose(T_final, np.eye(8, 5) @ np.eye(5, 3) @ np.eye(3, 2))
    print("  ✓ ChainedTransform: multi-extension chaining correct")


def test_build_T0():
    """Test build_T0_from_schmidt."""
    from dm_svd_dci.growing_cas_dmrg import build_T0_from_schmidt

    U = np.array([[1.0, 0.0], [0.0, 0.5]])
    V = np.array([[0.8, 0.0], [0.0, 0.6]])
    schmidt = {
        2: {
            'U': U, 'V': V, 'r': 2,
            'dim_A': 2, 'dim_B': 2,
            'sigma': np.array([0.8, 0.3]),
            'sigma_full': np.array([0.8, 0.3]),
        },
        3: {
            'U': np.zeros((3, 0)),
            'V': np.zeros((3, 0)),
            'r': 0,
            'dim_A': 3, 'dim_B': 3,
            'sigma': np.array([]),
            'sigma_full': np.array([]),
        },
    }
    T0 = build_T0_from_schmidt(schmidt)
    assert 2 in T0
    assert T0[2].shape == (4, 2)  # d_A*d_B=4, r=2
    assert T0[3].shape == (0, 0)  # empty block

    # Column 0: outer(U[:,0], V[:,0])
    expected_col0 = np.outer(U[:, 0], V[:, 0]).ravel()
    assert np.allclose(T0[2][:, 0], expected_col0)

    # Column 1: outer(U[:,1], V[:,1])
    expected_col1 = np.outer(U[:, 1], V[:, 1]).ravel()
    assert np.allclose(T0[2][:, 1], expected_col1)

    print("  ✓ build_T0_from_schmidt: correct")


def test_block_svd_multi_orbital_rank1():
    """Test block_svd_multi_orbital with rank-1 matrix."""
    from dm_svd_dci.block_svd_general import block_svd_multi_orbital

    D, d_B = 10, 16
    u = np.random.randn(D)
    u /= np.linalg.norm(u)
    v = np.random.randn(d_B)
    v /= np.linalg.norm(v)
    C = np.outer(u, v)
    psi = C.ravel()

    result = block_svd_multi_orbital(psi, D, d_B, eps=1e-3, verbose=False)
    assert result['D_new'] == 1
    assert result['discarded_weight'] < 1e-14

    W = result['W']
    assert np.allclose(W.T @ W, np.eye(1), atol=1e-12)

    # U_trunc should map to u
    U_trunc = result['U_trunc']
    assert U_trunc.shape == (D, 1)
    assert np.allclose(np.abs(U_trunc.ravel()), np.abs(u), atol=1e-12)

    print("  ✓ block_svd_multi_orbital: rank-1 test passed")


def test_block_svd_multi_orbital_truncation():
    """Test block_svd_multi_orbital truncation behavior."""
    from dm_svd_dci.block_svd_general import block_svd_multi_orbital

    D, d_B = 20, 15
    s_true = np.array([5.0, 2.0, 0.5, 0.01, 0.001])
    k = len(s_true)
    U_true = np.eye(D)[:, :k]
    V_true = np.eye(d_B)[:, :k]
    C = U_true @ np.diag(s_true) @ V_true.T
    psi = C.ravel()

    # eps=0.01: keep s > 0.01*5.0 = 0.05 → keep 3 (5.0, 2.0, 0.5)
    r1 = block_svd_multi_orbital(psi, D, d_B, eps=0.01, verbose=False)
    assert r1['D_new'] == 3, f"Expected 3, got {r1['D_new']}"

    # eps=1e-4: keep s > 1e-4*5.0 = 5e-4 → keep all 5
    r2 = block_svd_multi_orbital(psi, D, d_B, eps=1e-4, verbose=False)
    assert r2['D_new'] == 5, f"Expected 5, got {r2['D_new']}"

    print("  ✓ block_svd_multi_orbital: truncation test passed")


def test_block_svd_zero():
    """Test block_svd_multi_orbital edge cases."""
    from dm_svd_dci.block_svd_general import block_svd_multi_orbital

    # Zero D
    r = block_svd_multi_orbital(np.zeros(0), 0, 4, eps=1e-3, verbose=False)
    assert r['D_new'] == 0
    assert r['W'].shape == (0, 0)

    # Zero CI vector
    r = block_svd_multi_orbital(np.zeros(30), 10, 3, eps=1e-3, verbose=False)
    assert r['D_new'] == 0

    print("  ✓ block_svd_multi_orbital: edge cases handled")


def test_smoke_h2o_sto3g():
    """Smoke test: Growing CAS DMRG on H₂O/STO-3G CAS(5,6).

    This is a minimal integration test to verify the pipeline runs end-to-end.
    H₂O/STO-3G is small enough to run in seconds.
    """
    from pyscf import gto, scf
    from dm_svd_dci.growing_cas_dmrg import GrowingCASDMRG

    mol = gto.M(
        atom='O 0 0 0; H 0 0.757 0.586; H 0 -0.757 0.586',
        basis='sto-3g',
        verbose=0,
    )
    mf = scf.RHF(mol)
    mf.kernel()

    # CAS(5,6): 2 core frozen, 5 active orbitals, 6 active electrons
    # Round 0: A₀=3, B₀=1 (4 orbitals total)
    # Round 1: B₁=1 (5 orbitals total = full CAS)
    grower = GrowingCASDMRG(
        mol, mf,
        n_active=5,
        n_elec=(3, 3),    # 6 electrons total
        n_core=2,
        n_occ_A=3,        # A: first 3 active MOs
        n_orb_B0=1,       # B0: 1 orbital
        n_orb_Bt=1,       # Bt: 1 orbital per round
        eps_svd=1e-3,
    )

    results = grower.run(verbose=False)

    # Check basic structure
    assert 'E_history' in results
    assert 'D_history' in results
    assert 'E_fci_ref' in results
    assert 'dE_final_mH' in results

    n_rounds = results['n_rounds']
    assert n_rounds >= 2, f"Expected at least 2 rounds, got {n_rounds}"

    # E_history should have n_rounds entries
    assert len(results['E_history']) == n_rounds
    assert len(results['D_history']) == n_rounds

    # D_history should be monotonically approximately non-increasing
    # (allow small increases from new blocks entering)
    # At minimum, the last round should be ≤ first round
    # Actually D_k is total compressed dimension which grows when new n_A blocks
    # appear. So non-monotonicity isn't guaranteed. Just check it's reasonable.
    for D in results['D_history']:
        assert D >= 0, f"D should be non-negative, got {D}"

    print(f"\n  H₂O/STO-3G smoke test results:")
    print(f"    Rounds: {n_rounds}")
    print(f"    D_history: {results['D_history']}")
    print(f"    E_history: {[f'{e:.10f}' for e in results['E_history']]}")
    print(f"    E_FCI_ref: {results['E_fci_ref']:.10f}")
    print(f"    dE_final:  {results['dE_final_mH']:.3f} mH")
    print("  ✓ Smoke test on H₂O/STO-3G passed")


def test_smoke_n2_ccpvdz_small():
    """Smoke test: Growing CAS DMRG on N₂/cc-pVDZ with minimal rounds.

    Uses only 2 rounds (CAS(7,10) → CAS(8,10)) to keep runtime reasonable.
    Verifies pipeline structure and basic correctness.
    """
    from pyscf import gto, scf
    from dm_svd_dci.growing_cas_dmrg import GrowingCASDMRG

    print("\n  Setting up N₂/cc-pVDZ (this may take ~30s)...")

    mol = gto.M(
        atom='N 0 0 0; N 0 0 1.098',
        basis='cc-pVDZ',
        verbose=0,
    )
    mf = scf.RHF(mol)
    mf.kernel()

    # Use a subset: round 0 on 7 orbitals, then 1 extension to 8
    # n_active=8 means we pre-allocate up to 8, but only use a subset
    grower = GrowingCASDMRG(
        mol, mf,
        n_active=8,        # Only test up to 8 orbitals
        n_elec=(5, 5),     # 10 electrons
        n_core=2,
        n_occ_A=5,         # A: first 5 active MOs
        n_orb_B0=2,        # B0: 2 orbitals
        n_orb_Bt=1,        # Bt: 1 orbital per round
        eps_svd=1e-3,
        max_rounds=1,      # Only 1 extension round
    )

    results = grower.run(verbose=False)

    assert 'E_history' in results
    assert len(results['E_history']) == 2  # round 0 + 1 extension
    assert len(results['D_history']) == 2

    print(f"\n  N₂/cc-pVDZ CAS(8,10) smoke test results:")
    print(f"    Rounds: {results['n_rounds']}")
    print(f"    D_history: {results['D_history']}")
    print(f"    E_history: {[f'{e:.10f}' for e in results['E_history']]}")
    print(f"    E_FCI_ref: {results['E_fci_ref']:.10f}")
    print(f"    dE_final:  {results['dE_final_mH']:.3f} mH")
    print("  ✓ Smoke test on N₂/cc-pVDZ CAS(8,10) passed")


if __name__ == "__main__":
    print("=" * 60)
    print("Running unit tests...")
    print("=" * 60)
    test_chained_transform_basic()
    test_chained_transform_compress()
    test_chained_transform_missing_block()
    test_chained_transform_multi_extend()
    test_build_T0()
    test_block_svd_multi_orbital_rank1()
    test_block_svd_multi_orbital_truncation()
    test_block_svd_zero()
    print("\nAll unit tests passed!")

    print("\n" + "=" * 60)
    print("Running smoke tests...")
    print("=" * 60)
    test_smoke_h2o_sto3g()
    test_smoke_n2_ccpvdz_small()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)