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


def _flat_indices(dets, aidx, bidx, nb):
    return np.array([aidx[int(a)] * nb + bidx[int(b)] for (a, b) in dets],
                    dtype=np.int64)


def build_hpp_sigma(dets, backend, aidx, bidx, na, nb):
    """H_PP over `dets` via C-level sigma_full columns (OMP-parallel libfci),
    replacing the O(P^2) pure-Python Slater-Condon `matrix_element` loop.

    For each determinant d: sigma = H|d>  (one contract_2e), then the column
    H[:, d] restricted to `dets` is sigma[flat_indices(dets)].
    """
    n = len(dets)
    flat = _flat_indices(dets, aidx, bidx, nb)
    H = np.zeros((n, n))
    for j, (a, b) in enumerate(dets):
        ci = np.zeros((na, nb)); ci[aidx[int(a)], bidx[int(b)]] = 1.0
        sig = backend.sigma_full(ci).reshape(-1)
        H[:, j] = sig[flat]
    return 0.5 * (H + H.T)


def extend_hpp_sigma(H_old, old_dets, new_dets, backend, aidx, bidx, na, nb):
    """Grow H_PP by appending `new_dets` (rows+cols) via C-level sigma_full,
    reusing the existing H_old block. Numerically matches extend via matrix_element.
    """
    No = len(old_dets); m = len(new_dets)
    all_dets = list(old_dets) + list(new_dets)
    flat_all = _flat_indices(all_dets, aidx, bidx, nb)
    Hn = np.zeros((No + m, No + m))
    Hn[:No, :No] = H_old
    for jl, (a, b) in enumerate(new_dets):
        ci = np.zeros((na, nb)); ci[aidx[int(a)], bidx[int(b)]] = 1.0
        sig = backend.sigma_full(ci).reshape(-1)
        col = sig[flat_all]
        r = No + jl
        Hn[:, r] = col
        Hn[r, :] = col
    return Hn
