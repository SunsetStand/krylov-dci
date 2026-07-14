"""Vectorized P-space operations for Krylov-dCI (parallel-opt).

Drop-in replacements for the per-iteration scalar Python hot loops in the
phaseA state-average pipeline. Numerically equivalent to the original scalar
reference (see tests/test_pspace_ops.py), but use numpy vectorization (SIMD/BLAS)
instead of Python-level per-determinant loops over M (M ~ 4e6 for CAS(14,10)).
"""
import numpy as np


def embed_pspace_vec(coeffs, p_full_idx, M):
    """Scatter P-space coefficients into a full length-M CI vector.

    Replaces: `vec = np.zeros(M); for li,gi in enumerate(p_full_idx): vec[gi]=coeffs[li]`
    """
    vec = np.zeros(M)
    vec[np.asarray(p_full_idx, dtype=np.int64)] = coeffs
    return vec


def build_pmask(p_set_or_idx, M):
    """Boolean mask (M,) True where determinant index is already in P-space."""
    m = np.zeros(M, dtype=bool)
    idx = np.fromiter(p_set_or_idx, dtype=np.int64, count=len(p_set_or_idx))
    m[idx] = True
    return m


def score_and_select(sigmas, hdiag, p_mask, batch,
                     c2_floor=1e-24, denom_floor=1e-8):
    """State-average CIPSI scoring + top-`batch` selection (vectorized).

    weights[q] = sum_roots |sigma_root[q]|^2 / max(|E_root - H_qq|, denom_floor),
    with contributions where |sigma|^2 < c2_floor dropped, and P-space zeroed.

    Args:
        sigmas:  list of (E_ref, sigma_vec) ; sigma_vec real, shape (M,).
        hdiag:   (M,) Hamiltonian diagonal.
        p_mask:  (M,) bool, True for determinants already in P-space.
        batch:   max number of new determinants to select.
    Returns:
        sel     : (<=batch,) int64 selected Q-space indices, weight-descending
                  (ties broken by ascending index, matching the scalar reference).
        max_w   : float, largest weight (0.0 if no candidates).
        weights : (M,) full weight array (P-space entries zeroed).
    """
    M = hdiag.shape[0]
    weights = np.zeros(M)
    for E_ref, sk in sigmas:
        c2 = sk * sk
        denom = np.maximum(np.abs(E_ref - hdiag), denom_floor)
        weights += np.where(c2 >= c2_floor, c2 / denom, 0.0)
    weights[p_mask] = 0.0
    cand = np.nonzero(weights > 0.0)[0]
    if cand.size == 0:
        return np.empty(0, dtype=np.int64), 0.0, weights
    order = cand[np.argsort(-weights[cand], kind='stable')]
    sel = order[:batch].astype(np.int64)
    return sel, float(weights[order[0]]), weights
