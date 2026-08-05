#!/usr/bin/env python3
"""
Generalized multi-orbital block-SVD for DMRG-style growing active space.

Generalizes block_svd.py (which handles single-orbital B space with d_B=4)
to arbitrary B-space dimensions. Given a CI vector ψ in the composite space
S ⊗ F(B), reshapes to (D, d_B) and performs SVD with truncation.

The singular vectors U are the optimal compressed basis (minimizing
Frobenius-norm truncation error). Truncation: keep σ_α > ε · σ_max.

Returns:
  - W: (D*d_B, D_new) compression isometry (W†W = I)
  - U_trunc: (D, D_new) — maps S_{k+1} → S_k (for chain transform)
  - s_all, s_kept, D_new: singular values and new dimension
"""

import numpy as np
from typing import Dict, Optional


def block_svd_multi_orbital(
    psi: np.ndarray,
    D: int,
    d_B: int,
    eps: float = 1e-3,
    verbose: bool = True,
) -> Dict:
    """SVD on CI vector reshaped as (D, d_B).

    Args:
        psi: (D * d_B,) CI vector in the composite S ⊗ F(B) space.
             Index layout: psi[d + D*b] = coefficient for
             S-basis state d × B-occupation b.
        D:   Dimension of the S-basis (current compressed dimension).
        d_B: Number of B-space determinants (for this n_A block).
        eps: Relative truncation threshold (keep σ > ε·σ_max).

    Returns:
        dict with keys:
          'W':       (D*d_B, D_new) isometry matrix (W†W = I).
          'U_trunc': (D, D_new) truncated left singular vectors.
          'V_trunc': (d_B, D_new) truncated right singular vectors.
          's_all':   all singular values, sorted descending.
          's_kept':  kept singular values.
          'D_new':   new compressed dimension.
          'D_old':   old compressed dimension (= D).
          'discarded_weight': Σ_{dropped} σ² / Σ_all σ².
    """
    if D == 0 or d_B == 0:
        return {
            'W': np.zeros((0, 0)),
            'U_trunc': np.zeros((0, 0)),
            'V_trunc': np.zeros((0, 0)),
            's_all': np.array([]),
            's_kept': np.array([]),
            'D_new': 0,
            'D_old': D,
            'discarded_weight': 0.0,
        }

    C = psi.reshape(D, d_B)  # (D, d_B)

    k = min(D, d_B)
    U, s, Vh = np.linalg.svd(C, full_matrices=False)
    # U: (D, k), s: (k,), Vh: (k, d_B)
    V = Vh.T  # (d_B, k)

    s_max = s[0] if len(s) > 0 else 0.0
    if s_max < 1e-15:
        keep = np.zeros(len(s), dtype=bool)
    else:
        keep = s > eps * s_max

    D_new = int(np.sum(keep))
    kept_indices = np.where(keep)[0]

    # Build compression isometry W: (D*d_B, D_new)
    # W[d + D*b, α] = U[d, α] * V[b, α]
    W = np.zeros((D * d_B, D_new))
    for idx_new, alpha in enumerate(kept_indices):
        u_alpha = U[:, alpha]      # (D,)
        v_alpha = V[:, alpha]      # (d_B,)
        W[:, idx_new] = np.outer(u_alpha, v_alpha).ravel()

    s_kept = s[keep]
    s_dropped = s[~keep] if len(s) > D_new else np.array([])
    total_sq = float(np.sum(s**2))
    discarded_weight = float(np.sum(s_dropped**2)) / max(total_sq, 1e-30)

    U_trunc = U[:, kept_indices]   # (D, D_new)
    V_trunc = V[:, kept_indices]   # (d_B, D_new)

    if verbose:
        print(f"  [block-SVD] D: {D} → {D_new} "
              f"(ε={eps}, kept={D_new}/{len(s)}, "
              f"σ₁={s[0]:.4e}, σ_min_kept={s_kept[-1]:.4e}"
              f"{f', σ_max_dropped={s_dropped[0]:.4e}' if len(s_dropped) > 0 else ''}, "
              f"discarded={discarded_weight:.2e})",
              flush=True)

    return {
        'W': W,
        'U_trunc': U_trunc,
        'V_trunc': V_trunc,
        's_all': s,
        's_kept': s_kept,
        'D_new': D_new,
        'D_old': D,
        'discarded_weight': discarded_weight,
    }


def compress_vector(
    v: np.ndarray,
    W: np.ndarray,
) -> np.ndarray:
    """Compress a vector in (D*d_B) space to D_new via W†."""
    return W.T @ v


def uncompress_vector(
    v_compressed: np.ndarray,
    W: np.ndarray,
) -> np.ndarray:
    """Expand a vector from D_new to (D*d_B) via W."""
    return W @ v_compressed


# ═══════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════

def test_block_svd_multi_rank1():
    """Test multi-orbital block-SVD on a rank-1 matrix."""
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

    # W should be an isometry
    W = result['W']
    assert np.allclose(W.T @ W, np.eye(1), atol=1e-12)
    # W @ W† @ psi ≈ psi
    psi_roundtrip = W @ (W.T @ psi)
    assert np.allclose(psi_roundtrip, psi, atol=1e-12)

    # U_trunc should map correctly
    U_trunc = result['U_trunc']
    assert U_trunc.shape == (D, 1)
    assert np.allclose(np.abs(U_trunc.ravel()), np.abs(u), atol=1e-12)

    print("  ✓ block_svd_multi: rank-1 test passed")


def test_block_svd_multi_truncation():
    """Test truncation with multiple singular values."""
    D, d_B = 20, 15
    s_true = np.array([5.0, 2.0, 0.5, 0.01, 0.001])
    k = len(s_true)
    U_true = np.eye(D)[:, :k]
    V_true = np.eye(d_B)[:, :k]
    C = U_true @ np.diag(s_true) @ V_true.T
    psi = C.ravel()

    # eps=0.01: keep s > 0.01*5 = 0.05 → keep [5.0, 2.0, 0.5, 0.01]
    result = block_svd_multi_orbital(psi, D, d_B, eps=0.01, verbose=False)
    assert result['D_new'] == 4

    # eps=1e-4: keep s > 1e-4*5 = 0.0005 → keep all 5
    result2 = block_svd_multi_orbital(psi, D, d_B, eps=1e-4, verbose=False)
    assert result2['D_new'] == 5

    # Verify isometry
    for r in [result, result2]:
        W = r['W']
        assert np.allclose(W.T @ W, np.eye(r['D_new']), atol=1e-12)

    print("  ✓ block_svd_multi: truncation test passed")


def test_block_svd_multi_zero():
    """Test edge case: zero D or d_B."""
    # Zero D
    result = block_svd_multi_orbital(np.zeros(0), 0, 4, eps=1e-3, verbose=False)
    assert result['D_new'] == 0
    assert result['W'].shape == (0, 0)

    # Zero CI vector
    result = block_svd_multi_orbital(np.zeros(30), 10, 3, eps=1e-3, verbose=False)
    assert result['D_new'] == 0
    assert result['discarded_weight'] == 0.0

    print("  ✓ block_svd_multi: zero edge cases handled")


if __name__ == "__main__":
    test_block_svd_multi_rank1()
    test_block_svd_multi_truncation()
    test_block_svd_multi_zero()
    print("All block_svd_general tests passed.")