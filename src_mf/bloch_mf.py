"""
Matrix-free Bloch effective Hamiltonian correction.

Replaces dense H_QP construction (build_hqp) with σ-vector accumulation:
    correction[i,j] = Σ_{q∈Q} A_q[q] · ⟨p_i|H|q⟩ · ⟨q|H|p_j⟩

where A_q = (E0 + Δ - H_QQ[q])^{-1} (diagonal resolvent, m=0).

Key advantage: avoids storing the Q×P H_QP matrix (e.g. 63k×2000 ≈ 1GB),
replacing it with ~N σ-vector computations + dot products.

Algorithm (batched, memory-efficient):
  1. For each p_i ∈ P: compute σ_i = H|p_i⟩, zero P rows, store to memmap
  2. Compute correction via batched dot products: Σ_q A_q[q] · σ_i[q] · σ_j[q]

Complexity: O(N·C_sigma + N²·M/2) vs O(N·C_sigma) for H_QP construction.
  - N=2000, M=63504: σ building ≈ 127M ops, dot products ≈ 127B ops
  - With BLAS: ~10-60s on 32-core CPU for N2/CAS(10,10)

For production-large systems: use batching with on-disk σ storage.

Usage:
    from src_mf.bloch_mf import compute_bloch_correction_mf
    correction = compute_bloch_correction_mf(backend, p_dets, H_PP, E0, delta=0.0)
    H_eff = H_PP + correction

Author: 雷塞, 2026-07-06
"""
import numpy as np
from numpy.linalg import eigh
import time, os, tempfile
from typing import List, Tuple, Optional


def compute_all_sigma_vectors(backend, p_dets, p_mask=None, verbose=True):
    """Compute σ_i = H|p_i⟩ for all P determinants.

    Zeroes out P-space components (these belong to H_PP, not Q coupling).

    Args:
        backend:  KDCIBackend instance.
        p_dets:   List of (alpha_str, beta_str) tuples.
        p_mask:   Boolean mask over Q-space for P indices.
                  Computed automatically if None.
        verbose:  Print timing.

    Returns:
        (sigma_all, p_mask) where:
          sigma_all: (M, N) numpy memmap array, each column = H|p_i⟩
          p_mask:    (M,) bool mask marking P-space indices
    """
    M = backend.q_idx.M
    N = len(p_dets)

    if p_mask is None:
        p_indices = backend.q_idx.p_indices(p_dets)
        p_mask = np.zeros(M, dtype=bool)
        p_mask[p_indices[p_indices >= 0]] = True

    # Use a temp memmap file for sigma storage (avoid 1GB RAM)
    tmpdir = tempfile.mkdtemp(prefix='bloch_sigma_')
    sigma_path = os.path.join(tmpdir, 'sigma_all.dat')
    sigma_all = np.memmap(sigma_path, dtype='float64', mode='w+',
                          shape=(M, N))

    t0 = time.perf_counter()
    for p in range(N):
        pa, pb = int(p_dets[p][0]), int(p_dets[p][1])
        ia = backend.q_idx._alpha_idx.get(pa)
        ib = backend.q_idx._beta_idx.get(pb)
        if ia is None or ib is None:
            continue

        # Build unit vector at position p
        idx = backend.q_idx.flat_index(pa, pb)
        unit = np.zeros(M)
        unit[idx] = 1.0

        # σ = H|p_i⟩ via selected_ci.contract_2e
        sigma_p = backend.sigma(unit)

        # Zero out P-space rows (these are intra-P couplings, counted in H_PP)
        sigma_p[p_mask] = 0.0

        sigma_all[:, p] = sigma_p

        if verbose and (p + 1) % 100 == 0:
            elapsed = time.perf_counter() - t0
            rate = (p + 1) / elapsed
            eta = (N - p - 1) / rate
            print("  sigma {:4d}/{}  |  {:.1f}/s  |  ETA {:.0f}s".format(
                p + 1, N, rate, eta), flush=True)

    sigma_all.flush()
    t_total = time.perf_counter() - t0
    if verbose:
        print("  Built {} sigma vectors in {:.1f}s ({:.1f}/s)".format(
            N, t_total, N / t_total), flush=True)
        print("  sigma stored at: {} ({:.1f} MB)".format(
            sigma_path, sigma_all.nbytes / 1e6), flush=True)

    return sigma_all, p_mask, sigma_path


def bloch_correction_batched(sigma_all, A_q_diag, p_mask,
                              batch_size=100, verbose=True):
    """Compute Bloch correction matrix from pre-computed σ vectors.

    correction[i,j] = Σ_q A_q[q] · σ_i[q] · σ_j[q]

    Batched to avoid O(N²·M) memory for full correction intermediate.

    Args:
        sigma_all:  (M, N) array of σ vectors.
        A_q_diag:   (M,) diagonal resolvent weights.
        p_mask:     (M,) bool mask (unused here; σ vectors already P-zeroed).
        batch_size: Number of rows to process at once.
        verbose:    Print timing.

    Returns:
        correction: (N, N) symmetric correction matrix.
    """
    M, N = sigma_all.shape
    correction = np.zeros((N, N))

    # Element-wise multiply each σ column by sqrt(A_q) for numerical stability
    # correction[i,j] = Σ_q (sqrt(A_q)·σ_i[q]) · (sqrt(A_q)·σ_j[q])
    # This avoids separate weighting step, using BLAS dot products directly.
    sqrt_A = np.sqrt(np.abs(A_q_diag))
    sqrt_A[A_q_diag < 0] = 0.0  # negative A_q → zero weight

    t0 = time.perf_counter()

    # Process in batches of rows to reduce memory
    for i in range(0, N, batch_size):
        i_end = min(i + batch_size, N)
        batch_len = i_end - i

        # Weighted σ for batch rows: shape (M, batch_len)
        weighted_batch = sigma_all[:, i:i_end] * sqrt_A[:, np.newaxis]

        # Accumulate: correction[i:i_end, :] = weighted_batch^T @ sigma_all
        # Only need upper triangle + diagonal for symmetric result
        # But for simplicity, compute full block then symmetrize
        for j_start in range(0, N, batch_size):
            j_end = min(j_start + batch_size, N)
            block = weighted_batch.T @ sigma_all[:, j_start:j_end]
            correction[i:i_end, j_start:j_end] += block

        if verbose:
            elapsed = time.perf_counter() - t0
            progress = (i + batch_len) / N * 100
            rate = (i + batch_len) / elapsed
            eta = (N - i - batch_len) / rate
            print("  Bloch {:5.1f}%  |  {:.1f}s  |  ETA {:.0f}s".format(
                progress, elapsed, eta), flush=True)

    # Symmetrize
    correction = 0.5 * (correction + correction.T)

    t_total = time.perf_counter() - t0
    if verbose:
        print("  Bloch correction built in {:.1f}s ({:.0f} GFLOPs est.)".format(
            t_total, N * N * M / 2 / 1e9), flush=True)

    return correction


def compute_bloch_correction_mf(backend, p_dets, H_PP, E0,
                                 delta=0.0, batch_size=100,
                                 cache_sigma_path=None,
                                 verbose=True):
    """Complete matrix-free Bloch correction pipeline.

    Args:
        backend:            KDCIBackend instance.
        p_dets:             List of (alpha, beta) P-determinant tuples.
        H_PP:               (N, N) P-space Hamiltonian.
        E0:                 Reference energy (or array for per-state).
        delta:              Level shift (default 0).
        batch_size:         Batch size for dot products (higher = faster but
                            more memory).
        cache_sigma_path:   If provided, save/load σ vectors from this path
                            (reuse across per-state calls).
        verbose:            Print progress.

    Returns:
        If E0 is scalar:
          correction: (N, N) matrix, H_eff = H_PP + correction
        If E0 is array of length nroots:
          corrections: list of (N, N) matrices, one per root
    """
    M = backend.q_idx.M
    N = len(p_dets)
    H_QQ_diag = backend.q_idx.hdiag

    # ── Step 1: Compute σ vectors (or load from cache) ──
    if cache_sigma_path and os.path.exists(cache_sigma_path):
        if verbose:
            print("  Loading cached σ vectors from {}".format(cache_sigma_path))
        sigma_all = np.memmap(cache_sigma_path, dtype='float64', mode='r',
                              shape=(M, N))
        p_indices = backend.q_idx.p_indices(p_dets)
        p_mask = np.zeros(M, dtype=bool)
        p_mask[p_indices[p_indices >= 0]] = True
    else:
        sigma_all, p_mask, tmp_sigma_path = compute_all_sigma_vectors(
            backend, p_dets, verbose=verbose)
        if cache_sigma_path:
            if verbose:
                print("  Saving σ cache to {}".format(cache_sigma_path))
            import shutil
            shutil.copy(tmp_sigma_path, cache_sigma_path)
            sigma_all = np.memmap(cache_sigma_path, dtype='float64', mode='r',
                                  shape=(M, N))

    # ── Step 2: Compute Bloch correction ──
    E0_list = np.atleast_1d(E0)

    corrections = []
    for k, E0_k in enumerate(E0_list):
        if verbose and len(E0_list) > 1:
            print("\n  --- root {}/{} (E0 = {:.8f} Ha) ---".format(
                k, len(E0_list) - 1, E0_k), flush=True)

        A_q_diag = 1.0 / (E0_k + delta - H_QQ_diag)
        A_q_diag = np.clip(A_q_diag, -1e10, 1e10)

        correction_k = bloch_correction_batched(
            sigma_all, A_q_diag, p_mask if 'p_mask' in dir() else None,
            batch_size=batch_size, verbose=verbose)
        corrections.append(correction_k)

    # Cleanup temp file if we created one
    if 'tmp_sigma_path' in dir() and not cache_sigma_path:
        try:
            os.unlink(tmp_sigma_path)
            os.rmdir(os.path.dirname(tmp_sigma_path))
        except OSError:
            pass

    if len(corrections) == 1:
        return corrections[0]
    return corrections


def compute_bloch_heff_mf(backend, p_dets, H_PP, nroots=6,
                           delta=0.0, batch_size=100,
                           sigma_cache_dir=None, verbose=True):
    """Full matrix-free Bloch H^eff pipeline: σ → correction → H_eff → diagonalize.

    Args:
        backend:          KDCIBackend instance.
        p_dets:           P determinant list.
        H_PP:             (N, N) dense P-space Hamiltonian.
        nroots:           Number of roots to extract.
        delta:            Level shift.
        batch_size:       Dot product batch size.
        sigma_cache_dir:  Directory for persistent σ cache (reuse across runs).
        verbose:          Print progress.

    Returns:
        dict with keys:
          E_bare:     bare H_PP eigenvalues (N,)
          E_bloch:    Bloch corrected eigenvalues for each root (nroots,)
          corrections: list of (N,N) correction matrices
          sigma_path: path to cached σ vectors
          wall_time:  total wall time
    """
    t0 = time.perf_counter()

    N = len(p_dets)
    M = backend.q_idx.M

    # Diagonalize H_PP to get E0_vals
    H_PP_sym = 0.5 * (H_PP + H_PP.T)
    E0_vals, _ = eigh(H_PP_sym)
    E0_vals = E0_vals[:nroots]

    if verbose:
        print("Matrix-free Bloch H^eff")
        print("  P = {}, M = {}, nroots = {}".format(N, M, nroots))
        print("  E0_vals = [{} ...]".format(
            ", ".join("{:.6f}".format(e) for e in E0_vals[:3])), flush=True)

    # Cache σ vectors
    sigma_path = None
    if sigma_cache_dir:
        os.makedirs(sigma_cache_dir, exist_ok=True)
        sigma_path = os.path.join(sigma_cache_dir, 'sigma_P{:d}.dat'.format(N))

    # Compute per-state corrections (reuses σ via cache)
    corrections = compute_bloch_correction_mf(
        backend, p_dets, H_PP_sym, E0_vals,
        delta=delta, batch_size=batch_size,
        cache_sigma_path=sigma_path, verbose=verbose)

    # Build H_eff for each root and diagonalize
    E_bloch = []
    for k in range(nroots):
        H_eff_k = H_PP_sym + corrections[k]
        H_eff_k = 0.5 * (H_eff_k + H_eff_k.T)
        ev, _ = eigh(H_eff_k)
        E_bloch.append(float(ev[k]))

    wall_total = time.perf_counter() - t0
    if verbose:
        print("\n  Total wall time: {:.1f}s".format(wall_total))

    return {
        'E_bare': E0_vals,
        'E_bloch': np.array(E_bloch),
        'corrections': corrections,
        'sigma_path': sigma_path,
        'wall_time': wall_total,
    }
