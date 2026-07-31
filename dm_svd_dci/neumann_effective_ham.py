#!/usr/bin/env python3
"""
Neumann series effective Hamiltonian builder (k=0 and k=1).

Replaces the Krylov-subspace + Löwdin-resolvent approach with a direct
Neumann series expansion of the Q-space Green's function:

  (E₀ I - H_QQ)⁻¹ ≈ A + A·H'_QQ·A

where A = (E₀ I - D_QQ)⁻¹ is the diagonal resolvent and
H'_QQ = H_QQ - D_QQ is the off-diagonal part.

Selection rules (|Δn| ≤ 2) restrict the needed blocks to:
  - H_{P Q₁}, H_{P Q₂}   (P–Q couplings)
  - H_{Q₁ Q₁}, H_{Q₂ Q₂} (Q-block diagonals, of which we take only the diagonal elements)
  - H_{Q₁ Q₂}             (Q₁–Q₂ off-diagonal coupling)

The k=0 correction uses only the diagonal resolvent:
  Δ⁽⁰⁾ = H_{P Q₁} A₁ H_{Q₁ P} + H_{P Q₂} A₂ H_{Q₂ P}

The k=1 correction adds four terms:
  Δ⁽¹⁾ = Δ_{Q₁→Q₁} + Δ_{Q₁→Q₂} + Δ_{Q₂→Q₁} + Δ_{Q₂→Q₂}

Total effective Hamiltonian:
  H_PP^{eff} = H_PP + Δ⁽⁰⁾ + Δ⁽¹⁾
"""

import numpy as np
from numpy.linalg import eigh
from typing import Dict, Tuple, Optional, List
import time


# ═══════════════════════════════════════════════════════════════════════════
# Diagonal resolvent A_n = 1 / (E - D_n)
# ═══════════════════════════════════════════════════════════════════════════

def build_resolvent(
    D: np.ndarray,
    E: float,
    regularize: float = 1e-10,
) -> np.ndarray:
    """Build diagonal resolvent A = (E·I - D)^{-1}.

    Args:
        D:    (d,) diagonal elements of a Q_n block.
        E:    Reference energy E₀ (possibly shifted by Δ).
        regularize: Threshold for near-zero denominators.

    Returns:
        A: (d,) vector of 1/(E - D_i).
    """
    denom = E - D
    A = np.where(np.abs(denom) > regularize, 1.0 / denom, 0.0)
    return A


# ═══════════════════════════════════════════════════════════════════════════
# k=0 correction
# ═══════════════════════════════════════════════════════════════════════════

def build_neumann_correction_k0(
    H_PQ: Dict[int, np.ndarray],
    D_by_n: Dict[int, np.ndarray],
    E0: float,
    active_n: Optional[List[int]] = None,
) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
    """Compute k=0 Neumann correction: Δ = Σ_n H_{P Q_n} A_n H_{Q_n P}.

    Args:
        H_PQ:     Dict[n] → H_{P Q_n}  (|P|, d_n).
        D_by_n:   Dict[n] → diag(H_{Q_n Q_n})  (d_n,).
        E0:       Reference energy.
        active_n: Which n values to include (default: all keys in H_PQ).

    Returns:
        (Delta_k0, resolvents):
          Delta_k0: (|P|, |P|) k=0 correction matrix.
          resolvents: Dict[n] → A_n (d_n,) resolvent vectors.
    """
    if active_n is None:
        active_n = sorted(H_PQ.keys())

    p_dim = 0
    for n_val in active_n:
        if n_val in H_PQ:
            p_dim = H_PQ[n_val].shape[0]
            break

    if p_dim == 0:
        return np.zeros((0, 0)), {}

    Delta = np.zeros((p_dim, p_dim))
    resolvents = {}

    for n_val in active_n:
        if n_val not in H_PQ or n_val not in D_by_n:
            continue
        H_PQ_n = H_PQ[n_val]   # (|P|, d_n)
        D_n = D_by_n[n_val]    # (d_n,)

        if H_PQ_n.shape[1] == 0:
            continue

        A_n = build_resolvent(D_n, E0)  # (d_n,)
        resolvents[n_val] = A_n

        # H_PQ_n * A_n  (column-wise scaling)
        H_PQ_weighted = H_PQ_n * A_n[np.newaxis, :]  # (|P|, d_n)

        # Δ += H_PQ_weighted @ H_PQ_n^T
        Delta += H_PQ_weighted @ H_PQ_n.T

    return Delta, resolvents


# ═══════════════════════════════════════════════════════════════════════════
# k=1 correction
# ═══════════════════════════════════════════════════════════════════════════

def build_neumann_correction_k1(
    H_PQ: Dict[int, np.ndarray],
    H_QQ_blocks: Dict[Tuple[int, int], np.ndarray],
    D_by_n: Dict[int, np.ndarray],
    E0: float,
    active_n: Optional[List[int]] = None,
    q_pairs: Optional[List[Tuple[int, int]]] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Compute k=1 Neumann correction: four fluctuation/transition terms.

    The four contributions are:
      (a) Q₁→Q₁:  H_{P Q₁} A₁ (H_{Q₁ Q₁} - diag(D₁)) A₁ H_{Q₁ P}
      (b) Q₁→Q₂:  H_{P Q₁} A₁ H_{Q₁ Q₂} A₂ H_{Q₂ P}
      (c) Q₂→Q₁:  Δ_{(b)}^†
      (d) Q₂→Q₂:  H_{P Q₂} A₂ (H_{Q₂ Q₂} - diag(D₂)) A₂ H_{Q₂ P}

    More generally, for all active Q_m, Q_n with |m-n| ≤ 2:
      Δ⁽¹⁾ = Σ_{m,n} H_{P Q_m} A_m H'_{Q_m Q_n} A_n H_{Q_n P}
    where H'_{Q_m Q_n} = H_{Q_m Q_n} - δ_{mn}·diag(D_m).

    Args:
        H_PQ:        Dict[n] → H_{P Q_n}  (|P|, d_n).
        H_QQ_blocks: Dict[(m,n)] → H_{Q_m Q_n}  (d_m, d_n).
        D_by_n:      Dict[n] → diag(H_{Q_n Q_n})  (d_n,).
        E0:          Reference energy.
        active_n:    All active Q_n values.
        q_pairs:     List of (m,n) pairs with m < n, |m-n| ≤ 2.

    Returns:
        (Delta_k1, components):
          Delta_k1:   (|P|, |P|) k=1 correction matrix.
          components: Dict[str → ndarray] individual term contributions
                      for diagnostics.
    """
    if active_n is None:
        active_n = sorted(H_PQ.keys())

    p_dim = 0
    for n_val in active_n:
        if n_val in H_PQ:
            p_dim = H_PQ[n_val].shape[0]
            break

    if p_dim == 0:
        return np.zeros((0, 0)), {}

    # Precompute resolvents A_n
    A_by_n = {}
    for n_val in active_n:
        if n_val in D_by_n and len(D_by_n[n_val]) > 0:
            A_by_n[n_val] = build_resolvent(D_by_n[n_val], E0)

    Delta = np.zeros((p_dim, p_dim))
    components = {}

    # ── (a) Diagonal fluctuations: H_{P Q_n} A_n (H_{Q_n Q_n} - diag(D_n)) A_n H_{Q_n P} ──
    for n_val in active_n:
        if n_val not in H_PQ or n_val not in A_by_n:
            continue
        H_PQ_n = H_PQ[n_val]       # (|P|, d_n)
        A_n = A_by_n[n_val]        # (d_n,)
        D_n = D_by_n.get(n_val, np.zeros(0))
        d_n = len(D_n)
        if d_n == 0:
            continue

        # H'_{nn} = H_{Q_n Q_n} - diag(D_n)
        H_nn = H_QQ_blocks.get((n_val, n_val))
        if H_nn is None or H_nn.shape[0] == 0:
            continue
        H_prime_nn = H_nn - np.diag(D_n)   # (d_n, d_n)

        # M_nn = A_n * H'_{nn} * A_n  (element-wise outer product)
        # M_nn[i,j] = A_n[i] * H'_{nn}[i,j] * A_n[j]
        M_nn = A_n[:, np.newaxis] * H_prime_nn * A_n[np.newaxis, :]  # (d_n, d_n)

        # Δ_{n→n} = H_PQ_n @ M_nn @ H_PQ_n^T
        Delta_nn = H_PQ_n @ M_nn @ H_PQ_n.T  # (|P|, |P|)
        Delta += Delta_nn
        components[f'Q{n_val}→Q{n_val}'] = Delta_nn

    # ── (b,c) Off-diagonal transitions: H_{P Q_m} A_m H_{Q_m Q_n} A_n H_{Q_n P} ──
    if q_pairs is None:
        # Build all pairs from active_n with |m-n| ≤ 2
        q_pairs = []
        sorted_n = sorted(active_n)
        for i, m in enumerate(sorted_n):
            for n in sorted_n[i + 1:]:
                if abs(m - n) <= 2:
                    q_pairs.append((m, n))

    for m, n in q_pairs:
        if m not in H_PQ or n not in H_PQ:
            continue
        if m not in A_by_n or n not in A_by_n:
            continue

        H_PQ_m = H_PQ[m]  # (|P|, d_m)
        H_PQ_n = H_PQ[n]  # (|P|, d_n)
        A_m = A_by_n[m]   # (d_m,)
        A_n = A_by_n[n]   # (d_n,)

        H_mn = H_QQ_blocks.get((m, n))
        if H_mn is None:
            H_mn = H_QQ_blocks.get((n, m))
            if H_mn is not None:
                H_mn = H_mn.T
        if H_mn is None or H_mn.size == 0:
            continue

        # M_mn = A_m * H_{mn} * A_n  (element-wise)
        M_mn = A_m[:, np.newaxis] * H_mn * A_n[np.newaxis, :]  # (d_m, d_n)

        # Δ_{m→n} = H_PQ_m @ M_mn @ H_PQ_n^T
        Delta_mn = H_PQ_m @ M_mn @ H_PQ_n.T  # (|P|, |P|)
        Delta += Delta_mn

        # Δ_{n→m} = Δ_{m→n}^T  (hermitian conjugate)
        Delta_nm = Delta_mn.T
        Delta += Delta_nm

        components[f'Q{m}→Q{n}'] = Delta_mn
        components[f'Q{n}→Q{m}'] = Delta_nm

    return Delta, components


# ═══════════════════════════════════════════════════════════════════════════
# Full effective Hamiltonian builder
# ═══════════════════════════════════════════════════════════════════════════

def build_effective_hamiltonian_neumann(
    H_PP: np.ndarray,
    H_PQ: Dict[int, np.ndarray],
    H_QQ_blocks: Dict[Tuple[int, int], np.ndarray],
    D_by_n: Dict[int, np.ndarray],
    E0: float,
    delta: float = 0.0,
    k_max: int = 1,
    verbose: bool = True,
) -> Dict:
    """Build Neumann-series effective Hamiltonian at given E₀.

    H_PP^{eff}(E₀) = H_PP + Δ⁽⁰⁾(E₀) + Δ⁽¹⁾(E₀)

    Args:
        H_PP:        (|P|, |P|) P-space Hamiltonian.
        H_PQ:        Dict[n] → H_{P Q_n}  (|P|, d_n).
        H_QQ_blocks: Dict[(m,n)] → H_{Q_m Q_n}  (d_m, d_n).
        D_by_n:      Dict[n] → diag(H_{Q_n Q_n})  (d_n,).
        E0:          Reference energy.
        delta:       Optional energy shift Δ (default 0).
        k_max:       Max Neumann order (0 or 1).
        verbose:     Print diagnostics.

    Returns:
        dict:
          'H_eff':    (|P|,|P|) effective Hamiltonian.
          'Delta_k0': (|P|,|P|) k=0 correction.
          'Delta_k1': (|P|,|P|) k=1 correction (if k_max ≥ 1).
          'A_by_n':   Dict[n] → resolvent vectors.
          'E_used':   E0 + delta.
    """
    E = E0 + delta
    p_dim = H_PP.shape[0]

    if verbose:
        t0 = time.perf_counter()

    # ── k=0 ──
    active_n = sorted(H_PQ.keys())
    q_pairs = []
    sorted_n = sorted(active_n)
    for i, m in enumerate(sorted_n):
        for n in sorted_n[i + 1:]:
            if abs(m - n) <= 2:
                q_pairs.append((m, n))

    Delta_k0, A_by_n = build_neumann_correction_k0(H_PQ, D_by_n, E, active_n=active_n)

    # ── k=1 ──
    Delta_k1 = np.zeros((p_dim, p_dim))
    components_k1 = {}
    if k_max >= 1:
        Delta_k1, components_k1 = build_neumann_correction_k1(
            H_PQ, H_QQ_blocks, D_by_n, E,
            active_n=active_n, q_pairs=q_pairs)

    # ── Assemble ──
    H_eff = H_PP + Delta_k0 + Delta_k1
    H_eff = 0.5 * (H_eff + H_eff.T)  # enforce hermiticity

    if verbose:
        elapsed = time.perf_counter() - t0
        norm_k0 = np.linalg.norm(Delta_k0)
        norm_k1 = np.linalg.norm(Delta_k1)
        norm_pp = np.linalg.norm(H_PP)
        print(f"  H^eff (Neumann k={k_max}): |P|={p_dim}, "
              f"||Δ_k0||/||H_PP||={norm_k0/max(norm_pp, 1e-15):.4f}, "
              f"||Δ_k1||/||H_PP||={norm_k1/max(norm_pp, 1e-15):.4f} "
              f"({elapsed:.1f}s)", flush=True)

    result = {
        'H_eff': H_eff,
        'Delta_k0': Delta_k0,
        'A_by_n': A_by_n,
        'E_used': E,
    }
    if k_max >= 1:
        result['Delta_k1'] = Delta_k1
        result['k1_components'] = components_k1

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Convenience: diagonalize and track roots
# ═══════════════════════════════════════════════════════════════════════════

def diagonalize_and_track(
    H_eff: np.ndarray,
    C_ref: Optional[np.ndarray] = None,
    H_PP: Optional[np.ndarray] = None,
    n_states: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Diagonalize H_eff and match roots to reference eigenvectors.

    If C_ref is not provided, uses H_PP eigenvectors as reference.

    Args:
        H_eff:  (N, N) effective Hamiltonian.
        C_ref:  (N, n_ref) reference eigenvectors (columns).
        H_PP:   (N, N) P-space Hamiltonian (used if C_ref is None).
        n_states: Number of lowest states to track.

    Returns:
        (eigvals, eigvecs, overlaps):
          eigvals:  (n_states,) matched eigenvalues.
          eigvecs:  (N, n_states) matched eigenvectors.
          overlaps: (n_states,) max |overlap| for each match.
    """
    if H_eff.shape[0] == 0:
        return np.array([]), np.zeros((0, 0)), np.array([])

    if C_ref is None and H_PP is not None:
        _, C_ref = eigh(H_PP)

    evals, evecs = eigh(H_eff)
    n_track = min(n_states, len(evals))

    if C_ref is not None and C_ref.shape[1] > 0:
        matched_evals = np.zeros(n_track)
        matched_evecs = np.zeros((H_eff.shape[0], n_track))
        overlaps = np.zeros(n_track)

        for k in range(n_track):
            ref_vec = C_ref[:, min(k, C_ref.shape[1] - 1)]
            ovlp = np.abs(evecs.T @ ref_vec)
            m_star = np.argmax(ovlp)
            matched_evals[k] = evals[m_star]
            matched_evecs[:, k] = evecs[:, m_star]
            overlaps[k] = ovlp[m_star]

        return matched_evals, matched_evecs, overlaps
    else:
        return evals[:n_track], evecs[:, :n_track], np.ones(n_track)


# ═══════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════

def test_build_resolvent():
    """Test diagonal resolvent construction."""
    D = np.array([1.0, 2.0, 3.0])
    E0 = 0.5
    A = build_resolvent(D, E0)
    # A_i = 1 / (0.5 - D_i)
    expected = 1.0 / (0.5 - D)
    assert np.allclose(A, expected)
    print("  ✓ build_resolvent: correct")


def test_k0_toy():
    """Test k=0 correction on a toy system."""
    # |P|=2, Q₁ has 2 states, Q₂ has 1 state
    p_dim = 2
    H_PQ = {
        1: np.array([[1.0, 0.5],
                      [0.0, 0.3]]),    # (2, 2)
        2: np.array([[0.2],
                      [0.4]]),          # (2, 1)
    }
    D_by_n = {
        1: np.array([2.0, 3.0]),       # (2,)
        2: np.array([4.0]),             # (1,)
    }
    E0 = 1.0

    Delta, A = build_neumann_correction_k0(H_PQ, D_by_n, E0)

    assert Delta.shape == (p_dim, p_dim)
    # A1 = 1/(1-2)=-1, 1/(1-3)=-0.5
    assert np.allclose(A[1], [-1.0, -0.5])
    # A2 = 1/(1-4) = -1/3
    assert np.allclose(A[2], [-1.0 / 3.0])

    # Verify hermiticity: Δ should be symmetric (since H_PQ·A·H_QP is hermitian)
    assert np.allclose(Delta, Delta.T, atol=1e-12)
    print("  ✓ build_neumann_correction_k0: toy test passed")


def test_k1_toy():
    """Test k=1 correction on a toy system."""
    p_dim = 2
    H_PQ = {
        1: np.array([[1.0, 0.0],
                      [0.0, 0.5]]),    # (2, 2)
    }
    H_QQ_blocks = {
        (1, 1): np.array([[2.0, 0.3],
                           [0.3, 3.0]]),  # (2, 2)
    }
    D_by_n = {
        1: np.array([2.0, 3.0]),       # (2,)
    }
    E0 = 1.0

    Delta, comps = build_neumann_correction_k1(
        H_PQ, H_QQ_blocks, D_by_n, E0)

    assert Delta.shape == (p_dim, p_dim)
    assert np.allclose(Delta, Delta.T, atol=1e-12)
    # There should be a Q₁→Q₁ diagonal fluctuation term
    assert 'Q1→Q1' in comps
    print("  ✓ build_neumann_correction_k1: toy test passed")


def test_k1_cross_terms():
    """Test k=1 correction with Q₁↔Q₂ cross terms."""
    p_dim = 2
    H_PQ = {
        1: np.array([[1.0, 0.0],
                      [0.0, 0.5]]),    # (2, 2)
        2: np.array([[0.3],
                      [0.1]]),          # (2, 1)
    }
    H_QQ_blocks = {
        (1, 1): np.diag([2.0, 3.0]),   # (2, 2)
        (2, 2): np.diag([4.0]),         # (1, 1)
        (1, 2): np.array([[0.2],
                           [0.1]]),     # (2, 1)
    }
    H_QQ_blocks[(2, 1)] = H_QQ_blocks[(1, 2)].T  # (1, 2)

    D_by_n = {
        1: np.array([2.0, 3.0]),
        2: np.array([4.0]),
    }
    E0 = 1.0

    q_pairs = [(1, 2)]
    Delta, comps = build_neumann_correction_k1(
        H_PQ, H_QQ_blocks, D_by_n, E0,
        active_n=[1, 2], q_pairs=q_pairs)

    assert Delta.shape == (p_dim, p_dim)
    assert np.allclose(Delta, Delta.T, atol=1e-12)
    # Should have Q₁→Q₁, Q₁→Q₂, Q₂→Q₁, Q₂→Q₂
    assert 'Q1→Q1' in comps
    assert 'Q1→Q2' in comps
    assert 'Q2→Q1' in comps
    assert 'Q2→Q2' in comps

    # Q₂→Q₁ should be the transpose of Q₁→Q₂
    assert np.allclose(comps['Q2→Q1'], comps['Q1→Q2'].T, atol=1e-12)

    print("  ✓ k=1 cross terms: all four contributions present and correct")


def test_full_neumann():
    """Test full build_effective_hamiltonian_neumann."""
    p_dim = 2
    H_PP = np.array([[0.0, 0.1],
                      [0.1, 0.5]])
    H_PP = 0.5 * (H_PP + H_PP.T)

    H_PQ = {
        1: np.array([[0.5, 0.0],
                      [0.0, 0.2]]),
        2: np.array([[0.1],
                      [0.3]]),
    }

    H_QQ_blocks = {
        (1, 1): np.array([[2.0, 0.3],
                           [0.3, 3.0]]),
        (2, 2): np.array([[4.0]]),
        (1, 2): np.array([[0.2],
                           [0.1]]),
    }
    H_QQ_blocks[(2, 1)] = H_QQ_blocks[(1, 2)].T

    D_by_n = {
        1: np.array([2.0, 3.0]),
        2: np.array([4.0]),
    }

    E0 = eigh(H_PP)[0][0]

    # k=0 only
    res_k0 = build_effective_hamiltonian_neumann(
        H_PP, H_PQ, H_QQ_blocks, D_by_n, E0, k_max=0, verbose=False)
    assert res_k0['H_eff'].shape == (p_dim, p_dim)
    assert np.allclose(res_k0['H_eff'], res_k0['H_eff'].T, atol=1e-12)

    # k=1
    res_k1 = build_effective_hamiltonian_neumann(
        H_PP, H_PQ, H_QQ_blocks, D_by_n, E0, k_max=1, verbose=False)
    assert res_k1['H_eff'].shape == (p_dim, p_dim)
    assert np.allclose(res_k1['H_eff'], res_k1['H_eff'].T, atol=1e-12)
    assert 'Delta_k1' in res_k1

    # k=1 correction should be non-zero (off-diagonal coupling exists)
    assert np.linalg.norm(res_k1['Delta_k1']) > 0

    # Check that k=1 result differs from k=0
    assert not np.allclose(res_k0['H_eff'], res_k1['H_eff'])

    # Check that H_eff eigenvalues are real
    evals = eigh(res_k1['H_eff'])[0]
    assert np.all(np.isreal(evals))

    print("  ✓ build_effective_hamiltonian_neumann: full pipeline passed")


def test_delta_shift():
    """Test that delta parameter correctly shifts the energy."""
    H_PP = np.eye(2) * 0.5
    H_PQ = {1: np.ones((2, 1)) * 0.5}
    H_QQ_blocks = {(1, 1): np.eye(1) * 2.0}
    D_by_n = {1: np.array([2.0])}
    E0 = 0.0

    res_d0 = build_effective_hamiltonian_neumann(
        H_PP, H_PQ, H_QQ_blocks, D_by_n, E0, delta=0.0, k_max=0, verbose=False)
    res_d1 = build_effective_hamiltonian_neumann(
        H_PP, H_PQ, H_QQ_blocks, D_by_n, E0, delta=1.0, k_max=0, verbose=False)

    # With delta=1.0, E_used = 1.0 → A = 1/(1-2) = -1 (vs -0.5 for E=0)
    # Corrections should differ
    assert not np.allclose(res_d0['Delta_k0'], res_d1['Delta_k0'])

    print("  ✓ delta shift: correct energy dependence")


if __name__ == "__main__":
    test_build_resolvent()
    test_k0_toy()
    test_k1_toy()
    test_k1_cross_terms()
    test_full_neumann()
    test_delta_shift()
    print("All neumann_effective_ham tests passed.")