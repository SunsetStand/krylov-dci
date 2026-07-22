#!/usr/bin/env python3
"""
Krylov subspace propagation using Modified Gram-Schmidt (MGS) only.

NO SVD truncation is performed at any stage. The only dimensional reduction
comes from the dmSVD Schmidt decomposition itself (σ_α > ε · σ_max).

Algorithm (m=0):
  1. Build initial Krylov vectors: T₀ = A · H_QP
     where A_q = 1/(E0 - H_QQ[q,q]) for q ∈ Q
  2. MGS orthonormalize → B₀ (size: q_dim × r₀, r₀ ≤ q_dim)

Algorithm (m=1):
  1. Compute residual: Y₁[:,k] = H_QQ @ B₀[:,k] - H_QQ_diag * B₀[:,k]
  2. Weight: X₁[:,k] = A * Y₁[:,k]
  3. MGS orthogonalize X₁ against B₀ and itself → B_incr
  4. B₁ = [B₀, B_incr]
"""

import numpy as np
from numpy.linalg import norm
from typing import Tuple, List, Optional
import time


# ═══════════════════════════════════════════════════════════════════════════
# Modified Gram-Schmidt (MGS)
# ═══════════════════════════════════════════════════════════════════════════

def modified_gram_schmidt(
    new_vectors: np.ndarray,
    existing_basis: Optional[np.ndarray] = None,
    lindep_threshold: float = 1e-10,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Orthonormalize new_vectors via MGS against existing_basis and itself.

    For each column v of new_vectors:
      1. Orthogonalize against all columns of existing_basis
      2. Orthogonalize against previously processed new vectors
      3. If ||v|| < lindep_threshold → discard (linear dependence)
      4. Normalize v → ||v|| = 1

    Modified Gram-Schmidt (not classical) for numerical stability.

    Args:
        new_vectors:      (M, k) array with k new vectors.
        existing_basis:   (M, d) array with d existing orthonormal vectors.
                          If None or empty, treated as empty.
        lindep_threshold: Norm below which vectors are considered linearly
                          dependent and discarded.
        verbose:          Print retained/discarded counts.

    Returns:
        (orthonormal_vectors, retained_indices):
          orthonormal_vectors: (M, r) with r ≤ k, orthonormal columns.
          retained_indices:    (r,) int array of column indices kept.
    """
    M, k = new_vectors.shape
    d = existing_basis.shape[1] if existing_basis is not None and existing_basis.size > 0 else 0

    V = new_vectors.copy()
    retained = []

    for j in range(k):
        v = V[:, j]

        # Orthogonalize against existing basis
        if d > 0:
            for i in range(d):
                v -= np.dot(existing_basis[:, i], v) * existing_basis[:, i]

        # Orthogonalize against previously accepted new vectors
        for r_idx in retained:
            w = V[:, r_idx]  # already orthonormalized
            v -= np.dot(w, v) * w

        nrm = norm(v)
        if nrm > lindep_threshold:
            V[:, j] = v / nrm
            retained.append(j)

    r = len(retained)
    if r == 0:
        if verbose:
            print(f"    MGS: {k} vectors → 0 retained (all linearly dependent)")
        return np.zeros((M, 0)), np.array([], dtype=int)

    result = V[:, retained]
    if verbose:
        print(f"    MGS: {k} vectors → {r} retained "
              f"({100 * (k - r) / max(k, 1):.0f}% discarded)")

    return result, np.array(retained, dtype=int)


# ═══════════════════════════════════════════════════════════════════════════
# Krylov m=0: Initial basis from H_QP
# ═══════════════════════════════════════════════════════════════════════════

def build_krylov_basis_mgs(
    H_PQ: np.ndarray,
    H_QQ_diag: np.ndarray,
    E0: float,
    lindep_threshold: float = 1e-10,
    verbose: bool = True,
) -> Tuple[np.ndarray, int, np.ndarray]:
    """Build initial Krylov basis B₀ = MGS(A · H_QP^T).

    Note: H_PQ is (|P|, |Q|) so H_QP = H_PQ^T is (|Q|, |P|).
    Each column of H_QP corresponds to one P basis state's coupling to all Q
    basis states.

    T₀[:, p] = A_q · H_QP[:, p]  where A_q = 1/(E0 - H_QQ[q,q])

    Then MGS(T₀) → B₀ (|Q|, r₀).

    No SVD truncation is applied.

    Args:
        H_PQ:        (|P|, |Q|) P–Q coupling matrix from H^emb.
        H_QQ_diag:   (|Q|,) diagonal of H_QQ.
        E0:          Reference energy (lowest eigenvalue of H_PP).
        lindep_threshold: Threshold for linear dependence detection.
        verbose:     Print progress.

    Returns:
        (B_0, r_0, A_q):
          B_0:  (|Q|, r₀) orthonormal Krylov basis.
          r_0:  Number of retained basis vectors.
          A_q:  (|Q|,) diagonal resolvent A_q = 1/(E0 - H_QQ[q,q]).
    """
    q_dim = H_QQ_diag.shape[0]
    p_dim = H_PQ.shape[0]

    if q_dim == 0 or p_dim == 0:
        return np.zeros((q_dim, 0)), 0, np.ones(q_dim)

    # ── Diagonal resolvent A = (E0·I - H_QQ_diag)^{-1} ──
    denom = E0 - H_QQ_diag
    # Regularize near-zero denominators
    A_q = np.where(np.abs(denom) > 1e-10, 1.0 / denom, 0.0)

    # ── T₀ = A · H_QP = A · H_PQ^T ──
    # H_PQ shape: (|P|, |Q|) → H_QP = H_PQ.T shape: (|Q|, |P|)
    H_QP = H_PQ.T  # (|Q|, |P|)
    T0 = H_QP * A_q[:, np.newaxis]  # (|Q|, |P|) — Lorentzian weighting

    # ── MGS (no SVD!) ──
    if verbose:
        t0 = time.perf_counter()
        print(f"  [m=0] Building initial Krylov basis from {p_dim} P-basis "
              f"columns...", flush=True)

    B0, retained = modified_gram_schmidt(
        T0, existing_basis=None, lindep_threshold=lindep_threshold,
        verbose=verbose)
    r0 = B0.shape[1]

    if verbose:
        elapsed = time.perf_counter() - t0
        print(f"  [m=0] Done: r₀ = {r0} ({elapsed:.0f}s)", flush=True)

    return B0, r0, A_q


# ═══════════════════════════════════════════════════════════════════════════
# Krylov m=1: Propagate with H_QQ, then MGS (no SVD)
# ═══════════════════════════════════════════════════════════════════════════

def propagate_krylov_mgs(
    B_current: np.ndarray,
    H_QQ: np.ndarray,
    H_QQ_diag: np.ndarray,
    A_q: np.ndarray,
    lindep_threshold: float = 1e-10,
    verbose: bool = True,
) -> Tuple[np.ndarray, int]:
    """Propagate Krylov basis: B_new = MGS(A · (H_QQ - D_QQ) · B_current).

    For each column b_k of B_current:
      1. residual = H_QQ @ b_k - H_QQ_diag * b_k  (off-diagonal action)
      2. x_k = A_q * residual                       (energy weighting)
      3. MGS(x_k, existing=B_current) → new directions

    No SVD truncation. Only MGS is used to remove linear dependencies.

    Args:
        B_current:        (|Q|, r_current) current orthonormal Krylov basis.
        H_QQ:             (|Q|, |Q|) Q-space Hamiltonian (dense sub-block).
        H_QQ_diag:        (|Q|,) diagonal of H_QQ.
        A_q:              (|Q|,) diagonal resolvent from build_krylov_basis_mgs.
        lindep_threshold: Linear dependence threshold.
        verbose:          Print progress.

    Returns:
        (B_new, r_new):
          B_new: (|Q|, r_new) extended orthonormal basis.
          r_new: Total number of vectors after propagation.
    """
    q_dim = B_current.shape[0]
    r_current = B_current.shape[1]

    if r_current == 0 or q_dim == 0:
        return B_current.copy(), r_current

    # ── Step 1: residual = H_QQ @ B - diag(H_QQ) * B ──
    # This is the off-diagonal action: (H_QQ - D_QQ) @ B
    # For efficiency, compute H_QQ @ B as dense matmul.
    if verbose:
        t0 = time.perf_counter()
        print(f"  [m=1] Propagating {r_current} basis vectors...", flush=True)

    HQQ_B = H_QQ @ B_current  # (|Q|, r_current)
    residual = HQQ_B - H_QQ_diag[:, np.newaxis] * B_current

    # ── Step 2: X = A · residual ──
    X = residual * A_q[:, np.newaxis]  # (|Q|, r_current)

    if verbose:
        print(f"    Residuals computed ({time.perf_counter() - t0:.0f}s)", flush=True)
        t1 = time.perf_counter()

    # ── Step 3: MGS against existing basis + new directions ──
    B_incr, retained = modified_gram_schmidt(
        X, existing_basis=B_current, lindep_threshold=lindep_threshold,
        verbose=verbose)

    r_incr = B_incr.shape[1]

    if r_incr > 0:
        B_new = np.hstack([B_current, B_incr])
    else:
        B_new = B_current.copy()

    r_new = B_new.shape[1]

    if verbose:
        elapsed = time.perf_counter() - t1
        print(f"  [m=1] Done: r_new = {r_new} (+{r_incr} new, "
              f"{r_current + r_incr - r_new} lost to lindep) "
              f"({elapsed:.0f}s MGS)", flush=True)

    return B_new, r_new


# ═══════════════════════════════════════════════════════════════════════════
# Convenience: build Krylov basis at m=0 or m=0+1
# ═══════════════════════════════════════════════════════════════════════════

def build_krylov_full(
    H_PQ: np.ndarray,
    H_QQ: np.ndarray,
    H_QQ_diag: np.ndarray,
    E0: float,
    m_max: int = 1,
    lindep_threshold: float = 1e-10,
    verbose: bool = True,
) -> Tuple[np.ndarray, List[int], np.ndarray]:
    """Build full Krylov basis up to m=m_max using MGS only.

    Args:
        H_PQ:        (|P|, |Q|) P–Q coupling.
        H_QQ:        (|Q|, |Q|) Q–Q Hamiltonian.
        H_QQ_diag:   (|Q|,) diagonal of H_QQ.
        E0:          Reference energy (from H_PP).
        m_max:       Max Krylov order (0 or 1 typical).
        lindep_threshold: Linear dependence threshold.
        verbose:     Print progress.

    Returns:
        (B_final, layer_sizes, A_q):
          B_final:     (|Q|, r_total) orthonormal basis.
          layer_sizes: [r₀, r₁_incr, ...] retained per layer.
          A_q:         (|Q|,) diagonal resolvent.
    """
    # m=0
    B0, r0, A_q = build_krylov_basis_mgs(
        H_PQ, H_QQ_diag, E0, lindep_threshold, verbose)
    layer_sizes = [r0]

    B_current = B0
    r_current = r0

    # m ≥ 1
    for m in range(1, m_max + 1):
        if r_current == 0:
            if verbose:
                print(f"  m={m}: no basis vectors, stopping")
            break

        r_before = r_current
        B_current, r_current = propagate_krylov_mgs(
            B_current, H_QQ, H_QQ_diag, A_q,
            lindep_threshold, verbose)
        r_incr = r_current - r_before
        layer_sizes.append(r_incr)

        if r_current == r_before:
            if verbose:
                print(f"  m={m}: no new directions, stopping propagation")
            break

    return B_current, layer_sizes, A_q


# ═══════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════

def test_mgs_basic():
    """Test MGS with simple independent vectors."""
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([1.0, 1.0, 0.0])
    new_vecs = np.column_stack([v1, v2])

    orth, retained = modified_gram_schmidt(new_vecs, None, verbose=False)
    assert orth.shape == (3, 2)
    assert np.allclose(np.dot(orth[:, 0], orth[:, 1]), 0.0, atol=1e-12)
    assert np.allclose(norm(orth[:, 0]), 1.0)
    assert np.allclose(norm(orth[:, 1]), 1.0)
    assert np.array_equal(retained, [0, 1])
    print("  ✓ MGS: 2 independent vectors → 2 orthonormal")


def test_mgs_lindep():
    """Test MGS discards linearly dependent vectors."""
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([2.0, 0.0, 0.0])  # parallel to v1
    v3 = np.array([0.0, 1.0, 0.0])  # independent
    new_vecs = np.column_stack([v1, v2, v3])

    orth, retained = modified_gram_schmidt(new_vecs, None, verbose=False)
    assert orth.shape == (3, 2)
    assert np.array_equal(retained, [0, 2])
    print("  ✓ MGS: linear dependence detected (v2 discarded, v3 kept)")


def test_mgs_with_existing():
    """Test MGS with existing basis."""
    existing = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])  # spans xy plane
    # Two identical columns [1,1,1]^T; after MGS against xy-plane → only z survives
    v = np.array([1.0, 1.0, 1.0])
    new = np.column_stack([v, v])

    orth, retained = modified_gram_schmidt(new, existing, verbose=False)
    assert orth.shape == (3, 1)  # only z-direction survives
    # The retained vector should be normalized [0,0,1]
    assert abs(orth[2, 0]) > 0.99
    print("  ✓ MGS with existing basis: projections correctly removed")


def test_build_krylov_mgs_toy():
    """Test build_krylov_basis_mgs on a toy 3×3 system."""
    # Q-space dimension = 3, P-space dimension = 2
    # H_PQ is (2, 3): P→Q coupling
    H_PQ = np.array([
        [1.0, 0.5, 0.0],
        [0.0, 0.5, 1.0],
    ])
    H_QQ_diag = np.array([2.0, 3.0, 4.0])
    E0 = 1.0  # reference energy

    B0, r0, A_q = build_krylov_basis_mgs(
        H_PQ, H_QQ_diag, E0, lindep_threshold=1e-12, verbose=False)

    # A_q computation: 1/(1-2)=-1, 1/(1-3)=-0.5, 1/(1-4)=-1/3
    assert np.allclose(A_q, [-1.0, -0.5, -1.0 / 3.0])

    # With 2 independent columns of H_QP (transpose of H_PQ), expect r0 ≤ 2
    assert r0 > 0
    assert r0 <= 2

    # B0 should be orthonormal
    if r0 >= 2:
        assert np.allclose(B0.T @ B0, np.eye(r0), atol=1e-12)
    print(f"  ✓ build_krylov_basis_mgs: r₀ = {r0}, B₀ orthonormal")


def test_propagate_krylov_mgs_toy():
    """Test propagate_krylov_mgs on toy system."""
    # Build a 4×4 Q-space Hamiltonian
    q_dim = 4
    H_QQ = np.array([
        [2.0, 0.5, 0.0, 0.0],
        [0.5, 3.0, 0.3, 0.0],
        [0.0, 0.3, 4.0, 0.2],
        [0.0, 0.0, 0.2, 5.0],
    ])
    H_QQ = 0.5 * (H_QQ + H_QQ.T)  # symmetrize
    H_QQ_diag = np.diag(H_QQ)

    A_q = 1.0 / (1.0 - H_QQ_diag)  # E0 = 1.0

    # Start with 2 basis vectors covering first 2 columns of H_QQ
    B0 = np.zeros((q_dim, 2))
    B0[0, 0] = 1.0
    B0[1, 1] = 1.0

    B_new, r_new = propagate_krylov_mgs(
        B0, H_QQ, H_QQ_diag, A_q, lindep_threshold=1e-12, verbose=False)

    # Propagation should produce at least 2 new directions from off-diagonal
    # coupling (H_QQ[0,1]=0.5, H_QQ[1,2]=0.3, H_QQ[2,3]=0.2)
    assert r_new >= 2
    # New basis should be orthonormal
    assert np.allclose(B_new.T @ B_new, np.eye(r_new), atol=1e-12)
    print(f"  ✓ propagate_krylov_mgs: r = {r_new}, basis orthonormal")


def test_build_krylov_full_toy():
    """Test full build_krylov_full pipeline on toy system."""
    H_PQ = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    H_QQ = np.array([
        [2.0, 0.5, 0.0],
        [0.5, 3.0, 0.3],
        [0.0, 0.3, 4.0],
    ])
    H_QQ = 0.5 * (H_QQ + H_QQ.T)
    H_QQ_diag = np.diag(H_QQ)
    E0 = 1.0

    B, layer_sizes, A_q = build_krylov_full(
        H_PQ, H_QQ, H_QQ_diag, E0, m_max=1,
        lindep_threshold=1e-12, verbose=False)

    assert len(layer_sizes) >= 1  # at least m=0
    assert layer_sizes[0] > 0       # m=0 should produce at least 1 vector
    assert B.shape[1] == sum(layer_sizes)
    # Orthonormality
    if B.shape[1] > 0:
        assert np.allclose(B.T @ B, np.eye(B.shape[1]), atol=1e-12)
    print(f"  ✓ build_krylov_full: layers={layer_sizes}, total r={B.shape[1]}")


if __name__ == "__main__":
    test_mgs_basic()
    test_mgs_lindep()
    test_mgs_with_existing()
    test_build_krylov_mgs_toy()
    test_propagate_krylov_mgs_toy()
    test_build_krylov_full_toy()
    print("All krylov_propagator tests passed.")