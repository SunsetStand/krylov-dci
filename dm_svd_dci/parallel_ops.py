#!/usr/bin/env python3
"""
Parallel sigma-vector computation using ThreadPoolExecutor.

PySCF's contract_2e is C-level (libfci) and releases the GIL, so Python
threads provide real parallelism when computing many independent sigma-vectors.

Usage:
    from dm_svd_dci.parallel_ops import compute_sigma_vectors_parallel
    sigmas = compute_sigma_vectors_parallel(backend, ci_mats, n_workers=8)
"""

import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Callable, Optional


def compute_sigma_vectors_parallel(
    sigma_fn: Callable[[np.ndarray], np.ndarray],
    vectors: List[np.ndarray],
    n_workers: int = 1,
    verbose: bool = True,
) -> List[np.ndarray]:
    """Compute H·v for each vector v in `vectors`, optionally in parallel.

    Args:
        sigma_fn: Function that takes (n_alpha, n_beta) CI matrix and returns
                  H·v as a CI matrix of the same shape.
                  Signature: sigma_mat = sigma_fn(ci_mat)
        vectors:  List of CI matrices, each shape (n_alpha, n_beta).
        n_workers: Number of worker threads. If 1, runs serially.
        verbose:  Print progress.

    Returns:
        List of sigma matrices, same length as vectors.
    """
    n_vecs = len(vectors)
    if n_vecs == 0:
        return []

    if n_workers <= 1:
        return _serial_sigma(sigma_fn, vectors, verbose)

    return _parallel_sigma(sigma_fn, vectors, n_workers, verbose)


def _serial_sigma(sigma_fn, vectors, verbose):
    """Serial computation with progress reporting."""
    n_vecs = len(vectors)
    results = []
    t0 = time.perf_counter()
    for k, ci_mat in enumerate(vectors):
        results.append(sigma_fn(ci_mat))
        if verbose and (k + 1) % max(1, n_vecs // 10) == 0:
            elapsed = time.perf_counter() - t0
            eta = elapsed / (k + 1) * (n_vecs - k - 1)
            print(f"    sigma {k + 1}/{n_vecs} "
                  f"({elapsed:.0f}s, ETA {eta:.0f}s)", flush=True)
    if verbose:
        elapsed = time.perf_counter() - t0
        print(f"    sigma done: {elapsed:.0f}s "
              f"({elapsed / max(n_vecs, 1):.2f}s/vector)", flush=True)
    return results


def _parallel_sigma(sigma_fn, vectors, n_workers, verbose):
    """Thread-parallel computation."""
    n_vecs = len(vectors)
    t0 = time.perf_counter()

    # Use ThreadPoolExecutor: contract_2e is C-level, releases GIL
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        # Submit all tasks with index
        future_map = {}
        for k, ci_mat in enumerate(vectors):
            future = pool.submit(_sigma_one, sigma_fn, ci_mat, k)
            future_map[future] = k

        results = [None] * n_vecs
        n_done = 0
        for future in as_completed(future_map):
            idx, sigma_mat = future.result()
            results[idx] = sigma_mat
            n_done += 1
            if verbose and n_done % max(1, n_vecs // 10) == 0:
                elapsed = time.perf_counter() - t0
                eta = elapsed / n_done * (n_vecs - n_done)
                print(f"    sigma {n_done}/{n_vecs} "
                      f"({elapsed:.0f}s, ETA {eta:.0f}s)", flush=True)

    if verbose:
        elapsed = time.perf_counter() - t0
        print(f"    sigma done ({n_workers} workers): {elapsed:.0f}s "
              f"({elapsed / max(n_vecs, 1):.2f}s/vector)", flush=True)

    return results


def _sigma_one(sigma_fn, ci_mat, idx):
    """Wrapper for pool.submit — returns (idx, result)."""
    return idx, sigma_fn(ci_mat)


def build_sigma_flat(
    sigma_fn: Callable[[np.ndarray], np.ndarray],
    vectors: List[np.ndarray],
    n_workers: int = 1,
    verbose: bool = True,
) -> np.ndarray:
    """Compute sigma vectors and return them stacked as a flat (M, n_vecs) array.

    Each sigma vector is reshaped from (n_alpha, n_beta) to (M,) flat.

    Args:
        sigma_fn: Function ci_mat → sigma_mat (both (na, nb)).
        vectors:  List of CI matrices.
        n_workers: Number of parallel workers.
        verbose:  Print progress.

    Returns:
        (M, n_vecs) float64 array.
    """
    sigmas = compute_sigma_vectors_parallel(sigma_fn, vectors, n_workers, verbose)
    if not sigmas:
        return np.zeros((0, 0))
    M = sigmas[0].size
    result = np.empty((M, len(sigmas)))
    for k, sm in enumerate(sigmas):
        result[:, k] = sm.reshape(-1)
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Convenience: parallel build of embedded Hamiltonian columns
# ═══════════════════════════════════════════════════════════════════════════

def build_hemb_columns_parallel(
    sigma_fn: Callable[[np.ndarray], np.ndarray],
    ci_mats: List[np.ndarray],
    n_workers: int = 1,
    verbose: bool = True,
) -> np.ndarray:
    """Build H^emb columns (or rows) in parallel via sigma-vector calls.

    For each CI matrix (representing a Schmidt basis state expanded into
    full CAS space), compute sigma = H·v, then project (dot with all other
    basis states) to get one column of H^emb.

    This function returns the raw sigma vectors; the projection (dot products
    with other basis states) is done by the caller to form H^emb.

    Args:
        sigma_fn:  Function ci_mat → sigma_mat (both (na, nb)).
        ci_mats:   List of CI matrices for all D Schmidt product states.
        n_workers: Number of parallel threads.
        verbose:   Print progress.

    Returns:
        Sigma matrix of shape (M_flat, D) where M_flat = na * nb.
    """
    return build_sigma_flat(sigma_fn, ci_mats, n_workers, verbose)


# ═══════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════

def test_compute_sigma_parallel_toy():
    """Test parallel sigma on a toy function (not real contract_2e)."""
    # Toy sigma function: identity (just returns copy)
    def toy_sigma(ci_mat):
        return ci_mat.copy()

    vecs = [np.ones((5, 5)) * i for i in range(8)]

    # Serial
    res_serial = compute_sigma_vectors_parallel(toy_sigma, vecs, n_workers=1, verbose=False)
    assert len(res_serial) == 8
    for i in range(8):
        assert np.allclose(res_serial[i], vecs[i])

    # Parallel
    res_parallel = compute_sigma_vectors_parallel(toy_sigma, vecs, n_workers=2, verbose=False)
    assert len(res_parallel) == 8
    for i in range(8):
        assert np.allclose(res_parallel[i], vecs[i])

    print("  ✓ compute_sigma_vectors_parallel: toy test passed")


def test_build_sigma_flat():
    """Test stacking sigma vectors."""
    def toy_sigma(ci_mat):
        return ci_mat * 2.0

    vecs = [np.array([[1.0, 0.0], [0.0, 0.0]]),
            np.array([[0.0, 1.0], [0.0, 0.0]])]

    result = build_sigma_flat(toy_sigma, vecs, n_workers=1, verbose=False)
    assert result.shape == (4, 2)  # 2×2=4 flat, 2 vectors
    assert result[0, 0] == 2.0  # first vector, first element doubled
    assert result[2, 1] == 0.0  # second vector, third element

    print("  ✓ build_sigma_flat: test passed")


if __name__ == "__main__":
    test_compute_sigma_parallel_toy()
    test_build_sigma_flat()
    print("All parallel_ops tests passed.")