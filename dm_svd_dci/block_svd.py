#!/usr/bin/env python3
"""
Quantum-number-aware block-SVD for DMRG-style growing active space.

Given a CI vector ψ in the composite space S ⊗ F(B) where B has
1 spatial orbital (4 occupation states), reshape to (D, 4) and SVD.

The 4 B occupations correspond to sectors:
  col 0: |0⟩     — 0 electrons in B, Sz=0
  col 1: |↑⟩     — 1 electron in B,  Sz=+½
  col 2: |↓⟩     — 1 electron in B,  Sz=-½
  col 3: |↑↓⟩    — 2 electrons in B, Sz=0

Global SVD on C = ψ.reshape(D,4) is equivalent to diagonalizing the
reduced density matrix ρ_S = ψψ†. The singular vectors U are the
optimal compressed basis (minimizing Frobenius-norm truncation error).

Truncation: keep σ_α > ε · σ_max.
"""

import numpy as np
from typing import Tuple, Dict, List, Optional


def block_svd(
    psi: np.ndarray,
    D: int,
    eps: float = 1e-3,
    verbose: bool = True,
) -> Dict:
    """Perform SVD on CI vector ψ reshaped to (D, 4).

    Args:
        psi: (D*4,) CI vector in the composite S ⊗ F(B) space.
             Index layout: psi[d + D*b] = coefficient for
             S-basis state d × B-occupation b.
        D:   Dimension of the S-basis (D_{t-1}).
        eps: Relative truncation threshold (keep σ > ε·σ_max).

    Returns:
        dict:
          'W':       (D*4, D_new) compression matrix (isometry: W†W = I).
          's_all':   (min(D,4),) all singular values.
          's_kept':  (D_new,) kept singular values.
          'D_new':   int — new Schmidt dimension.
          'D_old':   int — old Schmidt dimension (= D).
          'discarded_weight': float — Σ_{dropped} σ² / Σ_all σ².
    """
    C = psi.reshape(D, 4)  # (D, 4)

    U, s, Vh = np.linalg.svd(C, full_matrices=False)
    # U: (D, k), s: (k,), Vh: (k, 4) where k = min(D, 4)

    s_max = s[0] if len(s) > 0 else 0.0
    if s_max < 1e-15:
        keep = np.zeros(len(s), dtype=bool)
    else:
        keep = s > eps * s_max

    D_new = int(np.sum(keep))

    # Build compression isometry W: (D*4, D_new)
    # W[d + D*b, α] = U[d, α] * Vh[α, b]
    # This maps the new Schmidt basis state α to the composite space
    W = np.zeros((D * 4, D_new))

    kept_indices = np.where(keep)[0]
    for idx_new, alpha in enumerate(kept_indices):
        u_alpha = U[:, alpha]     # (D,)
        v_alpha = Vh[alpha, :]    # (4,)
        # Outer product: W[:, idx_new] = u_alpha ⊗ v_alpha (flattened)
        W[:, idx_new] = np.outer(u_alpha, v_alpha).ravel()

    # Verify isometry: W†W ≈ I
    # WWt_test = W.T @ W
    # assert np.allclose(WWt_test, np.eye(D_new), atol=1e-12)

    s_kept = s[keep]
    s_dropped = s[~keep]
    discarded_weight = float(np.sum(s_dropped**2)) / max(float(np.sum(s**2)), 1e-30)

    if verbose:
        print(f"  [block-SVD] D: {D} → {D_new} "
              f"(ε={eps}, kept={D_new}/{len(s)}, "
              f"σ₁={s[0]:.4f}, σ_min_kept={s_kept[-1]:.4e}"
              f"{f', σ_max_dropped={s_dropped[0]:.4e}' if len(s_dropped) > 0 else ''}, "
              f"discarded={discarded_weight:.2e})",
              flush=True)

    return {
        'W': W,
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
    """Compress a vector in (D*4) space to D_new via W†.

    v_new = W† @ v_old
    """
    return W.T @ v


def uncompress_vector(
    v_compressed: np.ndarray,
    W: np.ndarray,
) -> np.ndarray:
    """Expand a vector from D_new to (D*4) via W.

    v_expanded = W @ v_compressed
    """
    return W @ v_compressed


# ═══════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════

def test_block_svd_basic():
    """Test block-SVD on a simple toy CI vector."""
    # Build a known rank-1 (D,4) matrix
    D = 5
    u = np.array([1.0, 0.5, 0.3, 0.2, 0.1])
    u /= np.linalg.norm(u)
    v = np.array([0.5, 0.3, -0.2, 0.1])
    v /= np.linalg.norm(v)
    C = np.outer(u, v)  # rank 1
    psi = C.ravel()

    result = block_svd(psi, D, eps=1e-3, verbose=False)

    # Should keep exactly 1 singular value (rank 1)
    assert result['D_new'] == 1
    assert result['discarded_weight'] < 1e-14

    # W should map back: W @ (W† @ psi) ≈ psi
    psi_compressed = compress_vector(psi, result['W'])
    psi_recovered = uncompress_vector(psi_compressed, result['W'])
    assert np.allclose(psi_recovered, psi, atol=1e-12)

    print("  ✓ block_svd: rank-1 test passed")


def test_block_svd_truncation():
    """Test truncation with multiple singular values."""
    D = 10
    # Build C with known singular values
    s_true = np.array([3.0, 0.5, 0.01, 0.001])
    U_true = np.eye(D)[:, :4]
    V_true = np.eye(4)
    C = U_true @ np.diag(s_true) @ V_true
    psi = C.ravel()

    # eps=0.01: should keep s > 0.01*3 = 0.03 → keep [3.0, 0.5]
    result = block_svd(psi, D, eps=0.01, verbose=False)
    assert result['D_new'] == 2

    # eps=1e-3: should keep s > 1e-3*3 = 0.003 → keep [3.0, 0.5, 0.01, 0.001]
    result2 = block_svd(psi, D, eps=1e-4, verbose=False)
    assert result2['D_new'] == 4

    # Verify isometry property
    for r in [result, result2]:
        W = r['W']
        WtW = W.T @ W
        assert np.allclose(WtW, np.eye(r['D_new']), atol=1e-12)

    print("  ✓ block_svd: truncation test passed")


def test_compress_uncompress():
    """Test compress/uncompress round-trip."""
    D = 5
    C = np.random.randn(D, 4)
    psi = C.ravel()

    result = block_svd(psi, D, eps=1e-3, verbose=False)

    # Compress
    psi_c = compress_vector(psi, result['W'])
    assert psi_c.shape == (result['D_new'],)

    # Uncompress
    psi_u = uncompress_vector(psi_c, result['W'])
    assert psi_u.shape == (D * 4,)

    # Check isometry: W† @ W = I
    assert np.allclose(result['W'].T @ result['W'],
                       np.eye(result['D_new']), atol=1e-12)

    # Compress then uncompress: W @ W† is a projector
    psi_roundtrip = uncompress_vector(compress_vector(psi, result['W']),
                                      result['W'])
    assert np.allclose(result['W'].T @ psi_roundtrip,
                       result['W'].T @ psi, atol=1e-12)

    print("  ✓ block_svd: compress/uncompress test passed")


def test_edge_case_zero():
    """Test edge case: zero CI vector."""
    psi = np.zeros(20)
    result = block_svd(psi, 5, eps=1e-3, verbose=False)
    assert result['D_new'] == 0
    assert result['W'].shape == (20, 0)
    print("  ✓ block_svd: zero edge case handled")


if __name__ == "__main__":
    test_block_svd_basic()
    test_block_svd_truncation()
    test_compress_uncompress()
    test_edge_case_zero()
    print("All block_svd tests passed.")
