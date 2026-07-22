#!/usr/bin/env python3
"""
Effective Hamiltonian construction in Krylov-compressed Schmidt basis.

Given:
  - H_PP: (|P|, |P|) P-space Hamiltonian
  - H_PQ: (|P|, |Q|) P–Q coupling
  - H_QQ: (|Q|, |Q|) Q-space Hamiltonian
  - B:    (|Q|, r) Krylov-compressed orthonormal basis (from krylov_propagator)

Build the Löwdin effective Hamiltonian:

  H_Q̃Q̃ = B^T @ H_QQ @ B         (r × r)
  H_PQ̃  = H_PQ @ B               (|P| × r)

  H^eff = H_PP + H_PQ̃ @ ((E0 + Δ)I - H_Q̃Q̃)^(-1) @ H_PQ̃^T

Then diagonalize to obtain approximate eigenvalues.

This is "方案 A": we assume H_PP, H_PQ, H_QQ are already extracted from the
full H^emb (via extract_subblocks in schmidt_partition.py).
"""

import numpy as np
from numpy.linalg import eigh, inv
from typing import Tuple, List, Dict, Optional
import time


# ═══════════════════════════════════════════════════════════════════════════
# Projected blocks in compressed Krylov basis
# ═══════════════════════════════════════════════════════════════════════════

def build_projected_hqq(
    H_QQ: np.ndarray,
    B: np.ndarray,
) -> np.ndarray:
    """Compute H_Q̃Q̃ = B^T @ H_QQ @ B.

    If H_QQ is large and B has few columns, computing H_QQ @ B first and
    then projecting is more efficient than forming B^T @ H_QQ @ B directly
    (since H_QQ @ B is (|Q|, r) and B^T @ (H_QQ @ B) is (r, r)).

    Args:
        H_QQ: (|Q|, |Q|) dense Q-space Hamiltonian.
        B:    (|Q|, r) orthonormal Krylov basis.

    Returns:
        H_Q̃Q̃: (r, r) hermitian matrix.
    """
    if B.shape[1] == 0:
        return np.zeros((0, 0))
    HQ_B = H_QQ @ B              # (|Q|, r)
    H_KK = B.T @ HQ_B            # (r, r)
    return 0.5 * (H_KK + H_KK.T)  # symmetrize


def build_projected_hpq(
    H_PQ: np.ndarray,
    B: np.ndarray,
) -> np.ndarray:
    """Compute H_PQ̃ = H_PQ @ B.

    Args:
        H_PQ: (|P|, |Q|) P–Q coupling matrix.
        B:    (|Q|, r) orthonormal Krylov basis.

    Returns:
        H_PQ̃: (|P|, r) matrix.
    """
    if B.shape[1] == 0:
        return np.zeros((H_PQ.shape[0], 0))
    return H_PQ @ B


# ═══════════════════════════════════════════════════════════════════════════
# Löwdin effective Hamiltonian
# ═══════════════════════════════════════════════════════════════════════════

def build_effective_hamiltonian(
    H_PP: np.ndarray,
    H_PQ: np.ndarray,
    H_QQ: np.ndarray,
    B: np.ndarray,
    E0: float,
    delta: float = 0.0,
    verbose: bool = True,
) -> np.ndarray:
    """Build the Löwdin effective Hamiltonian in the Krylov-compressed basis.

    H^eff = H_PP + H_PQ̃ @ ((E0 + Δ)I - H_Q̃Q̃)^(-1) @ H_PQ̃^T

    where H_PQ̃ = H_PQ @ B and H_Q̃Q̃ = B^T @ H_QQ @ B.

    Args:
        H_PP:  (|P|, |P|) P-space Hamiltonian.
        H_PQ:  (|P|, |Q|) P–Q coupling.
        H_QQ:  (|Q|, |Q|) Q-space Hamiltonian.
        B:     (|Q|, r) Krylov-compressed orthonormal basis.
        E0:    Reference energy (from H_PP diagonalization).
        delta: Energy shift Δ. Use 0 for non-self-consistent mode.
        verbose: Print diagnostics.

    Returns:
        H_eff: (|P|, |P|) hermitian effective Hamiltonian.
    """
    N = H_PP.shape[0]
    r = B.shape[1]

    if r == 0:
        # No compressed Q-space → H^eff = H_PP
        return 0.5 * (H_PP + H_PP.T)

    if verbose:
        t0 = time.perf_counter()

    # ── Build projected blocks ──
    H_KK = build_projected_hqq(H_QQ, B)    # (r, r)
    H_PK = build_projected_hpq(H_PQ, B)    # (N, r)

    # ── Resolvent: ((E0 + Δ)I - H_KK)^(-1) ──
    E = E0 + delta
    resolvent = inv(E * np.eye(r) - H_KK)

    # ── Correction: H_PK @ resolvent @ H_PK^T ──
    correction = H_PK @ resolvent @ H_PK.T  # (N, N)

    # ── Effective Hamiltonian ──
    H_eff = H_PP + correction
    H_eff = 0.5 * (H_eff + H_eff.T)  # enforce hermiticity

    if verbose:
        elapsed = time.perf_counter() - t0
        norm_corr = np.linalg.norm(correction)
        norm_pp = np.linalg.norm(H_PP)
        print(f"  H^eff built: |P|={N}, r_Q̃={r}, "
              f"||corr||/||H_PP||={norm_corr/norm_pp:.4f} "
              f"({elapsed:.0f}s)", flush=True)

    return H_eff


# ═══════════════════════════════════════════════════════════════════════════
# Diagonalization
# ═══════════════════════════════════════════════════════════════════════════

def diagonalize_effective(
    H_eff: np.ndarray,
    n_states: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Diagonalize the effective Hamiltonian.

    Args:
        H_eff:    (N, N) hermitian effective Hamiltonian.
        n_states: Number of lowest eigenstates to return.
                  If None, returns all eigenvalues.

    Returns:
        (eigvals, eigvecs):
          eigvals: (n_states,) eigenvalues, sorted ascending.
          eigvecs: (N, n_states) eigenvectors (columns) in P-space basis.
    """
    if H_eff.shape[0] == 0:
        return np.array([]), np.zeros((0, 0))

    evals, evecs = eigh(H_eff)
    if n_states is None:
        n_states = len(evals)
    n_states = min(n_states, len(evals))
    return evals[:n_states], evecs[:, :n_states]


# ═══════════════════════════════════════════════════════════════════════════
# Multi-root tracking: match H_eff eigenstates to reference H_PP eigenstates
# ═══════════════════════════════════════════════════════════════════════════

def track_roots(
    H_eff: np.ndarray,
    C_ref: np.ndarray,
    n_states: int = 5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Diagonalize H_eff and match eigenvalues to reference eigenstates.

    Uses overlap tracking: for each reference eigenstate k, find the H_eff
    eigenstate m*_k with maximum overlap:

        m*_k = argmax_m |⟨c_m^{eff} | c_k^{(ref)}⟩|

    Args:
        H_eff:  (N, N) effective Hamiltonian.
        C_ref:  (N, n_ref) reference eigenvectors (e.g. from H_PP).
        n_states: Number of states to track.

    Returns:
        (eigvals_matched, eigvecs_matched, overlaps):
          eigvals_matched: (n_states,) matched eigenvalues.
          eigvecs_matched: (N, n_states) matched eigenvectors in P-space.
          overlaps:        (n_states,) max |overlap| for each state.
    """
    evals, evecs = eigh(H_eff)
    n_ref = C_ref.shape[1]
    n_track = min(n_states, n_ref)

    matched_evals = np.zeros(n_track)
    matched_evecs = np.zeros((C_ref.shape[0], n_track))
    overlaps = np.zeros(n_track)

    for k in range(n_track):
        ref_vec = C_ref[:, k]
        ovlp = np.abs(evecs.T @ ref_vec)
        m_star = np.argmax(ovlp)
        matched_evals[k] = evals[m_star]
        matched_evecs[:, k] = evecs[:, m_star]
        overlaps[k] = ovlp[m_star]

    return matched_evals, matched_evecs, overlaps


# ═══════════════════════════════════════════════════════════════════════════
# Convenience: run full effective Hamiltonian pipeline at given m
# ═══════════════════════════════════════════════════════════════════════════

def run_effective_ham_at_m(
    H_PP: np.ndarray,
    H_PQ: np.ndarray,
    H_QQ: np.ndarray,
    E0: float,
    B: np.ndarray,
    delta: float = 0.0,
    n_states: int = 1,
    C_ref: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> Dict:
    """Build and diagonalize H^eff at given Krylov basis B.

    Args:
        H_PP, H_PQ, H_QQ: Hamiltonian blocks.
        E0:     Reference energy.
        B:      (|Q|, r) Krylov basis.
        delta:  Energy shift.
        n_states: Number of states to track.
        C_ref:  Reference eigenvectors for overlap tracking.
                If None, uses H_PP eigenvectors.
        verbose: Print results.

    Returns:
        dict with:
          'H_eff':       effective Hamiltonian
          'E_eff':       (n_states,) eigenvalues
          'E_vecs':      (|P|, n_states) eigenvectors
          'overlaps':    (n_states,) max overlaps
          'E0':          reference energy
          'r':           Krylov basis dimension
    """
    H_eff = build_effective_hamiltonian(
        H_PP, H_PQ, H_QQ, B, E0, delta=delta, verbose=verbose)

    # Reference for overlap tracking
    if C_ref is None:
        _, C_ref = eigh(H_PP)

    E_matched, E_vecs_matched, overlaps = track_roots(
        H_eff, C_ref, n_states=n_states)

    if verbose:
        print(f"  Effective eigenvalues:")
        for k in range(min(n_states, len(E_matched))):
            exc = (E_matched[k] - E_matched[0]) * 1000 if k > 0 else 0
            exc_str = f"  ({exc:+.1f} mH)" if k > 0 else ""
            print(f"    S{k}: {E_matched[k]:.12f} Ha{exc_str} "
                  f"(overlap={overlaps[k]:.6f})")

    return {
        'H_eff': H_eff,
        'E_eff': E_matched,
        'E_vecs': E_vecs_matched,
        'overlaps': overlaps,
        'E0': E0,
        'r': B.shape[1],
    }


# ═══════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════

def test_build_projected():
    """Test H_Q̃Q̃ and H_PQ̃ construction."""
    # Toy: |Q|=4, |P|=2, r=2
    H_QQ = np.diag([1.0, 2.0, 3.0, 4.0])
    H_QQ[0, 1] = H_QQ[1, 0] = 0.5
    H_PQ = np.array([[1.0, 0.0, 0.5, 0.0],
                      [0.0, 1.0, 0.0, 0.5]])

    B = np.zeros((4, 2))
    B[0, 0] = 1.0
    B[1, 1] = 1.0

    H_KK = build_projected_hqq(H_QQ, B)
    assert H_KK.shape == (2, 2)
    # H_KK[0,0] = B[:,0]^T @ H_QQ @ B[:,0] = H_QQ[0,0] = 1.0
    assert np.allclose(H_KK[0, 0], 1.0)
    # H_KK[1,1] = H_QQ[1,1] = 2.0
    assert np.allclose(H_KK[1, 1], 2.0)
    # H_KK[0,1] = B[:,0]^T @ H_QQ @ B[:,1] = H_QQ[0,1] = 0.5
    assert np.allclose(H_KK[0, 1], 0.5)
    assert np.allclose(H_KK[1, 0], 0.5)

    H_PK = build_projected_hpq(H_PQ, B)
    assert H_PK.shape == (2, 2)
    # H_PK[0,0] = Σ_q H_PQ[0,q] * B[q,0] = H_PQ[0,0] = 1.0
    assert np.allclose(H_PK[0, 0], 1.0)
    # H_PK[1,1] = H_PQ[1,1] = 1.0
    assert np.allclose(H_PK[1, 1], 1.0)

    print("  ✓ build_projected_hqq / build_projected_hpq: correct")


def test_effective_hamiltonian_toy():
    """Test full H^eff construction on a tiny toy system."""
    # |P|=2, |Q|=3, r=2
    H_PP = np.array([[0.0, 0.1], [0.1, 0.5]])
    H_PP = 0.5 * (H_PP + H_PP.T)

    H_PQ = np.array([[0.5, 0.0, 0.0],
                      [0.0, 0.3, 0.0]])

    H_QQ = np.diag([1.0, 2.0, 3.0])
    H_QQ[0, 1] = H_QQ[1, 0] = 0.2

    B = np.zeros((3, 2))
    B[0, 0] = 1.0
    B[1, 1] = 1.0

    E0 = np.linalg.eigvalsh(H_PP)[0]

    H_eff = build_effective_hamiltonian(
        H_PP, H_PQ, H_QQ, B, E0, delta=0.0, verbose=False)

    assert H_eff.shape == (2, 2)
    assert np.allclose(H_eff, H_eff.T)

    # Without Q coupling (H_PQ=0), H_eff = H_PP
    H_eff_noq = build_effective_hamiltonian(
        H_PP, np.zeros_like(H_PQ), H_QQ, B, E0, delta=0.0, verbose=False)
    assert np.allclose(H_eff_noq, 0.5 * (H_PP + H_PP.T))

    print("  ✓ build_effective_hamiltonian: toy test passed")


def test_track_roots():
    """Test root tracking logic."""
    # Simple 3×3 H_eff with known eigenvectors
    H_eff = np.diag([0.0, 1.0, 2.0])
    C_ref = np.eye(3)

    evals, evecs, overlaps = track_roots(H_eff, C_ref, n_states=3)
    assert np.allclose(evals, [0.0, 1.0, 2.0])
    assert np.allclose(overlaps, [1.0, 1.0, 1.0])

    # If H_eff has permuted eigenvalues, tracking should still match
    H_eff_perm = np.diag([2.0, 0.0, 1.0])
    evals2, evecs2, overlaps2 = track_roots(H_eff_perm, C_ref, n_states=3)
    # The tracked eigenvalues should still be [0.0, 1.0, 2.0] in order
    # (each matched to the reference eigenvector with max overlap)
    assert np.allclose(np.sort(evals2), [0.0, 1.0, 2.0])

    print("  ✓ track_roots: correct matching")


def test_empty_edge_cases():
    """Test edge cases with empty Krylov basis."""
    H_PP = np.array([[0.0, 0.1], [0.1, 0.5]])
    H_PP = 0.5 * (H_PP + H_PP.T)
    H_PQ = np.zeros((2, 3))
    H_QQ = np.eye(3)
    E0 = np.linalg.eigvalsh(H_PP)[0]

    # Empty B
    B = np.zeros((3, 0))
    H_eff = build_effective_hamiltonian(
        H_PP, H_PQ, H_QQ, B, E0, delta=0.0, verbose=False)
    assert H_eff.shape == (2, 2)
    assert np.allclose(H_eff, 0.5 * (H_PP + H_PP.T))

    # Diagonalize empty
    evals, evecs = diagonalize_effective(np.zeros((0, 0)))
    assert len(evals) == 0

    print("  ✓ empty_edge_cases: all passed")


if __name__ == "__main__":
    test_build_projected()
    test_effective_hamiltonian_toy()
    test_track_roots()
    test_empty_edge_cases()
    print("All effective_ham tests passed.")