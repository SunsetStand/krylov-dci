#!/usr/bin/env python3
"""
Streaming (memory-efficient) Scheme B operations for building H_PP, H_PQ
without constructing the full D×D H^emb matrix.

Key idea: Expand Schmidt basis states to full CAS CI matrices ONE AT A TIME
(or in small batches), compute sigma-vectors, and project immediately.
Memory peak = O(batch_size × M) instead of O(|P| × M) or O(D × M).

For CAS(14,10) with M=4,008,004 and batch_size=32:
  Peak memory = 2 × 32 × 4M × 8 bytes ≈ 2 GB  (vs 192 GB for full expansion)

Usage:
    from dm_svd_dci.streaming_ops import StreamBuilder

    builder = StreamBuilder(schmidt_data, partition, part_info, backend,
                            n_occ, n_act, batch_size=32, n_workers=16)
    H_PP = builder.build_hpp()
    H_PQ = builder.build_hpq()
    # For Krylov propagation:
    H_QQ_v = builder.apply_hqq(v)        # single vector
    H_QQ_B = builder.apply_hqq_batch(B)  # batch of vectors
"""

import sys, os, time
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


# ═══════════════════════════════════════════════════════════════════════════
# Single-state expansion (shared helper)
# ═══════════════════════════════════════════════════════════════════════════

def _expand_one(
    elem_info: Dict,
    schmidt_data: Dict[int, Dict],
    partition: Dict[int, Dict],
    n_alpha_strs: int,
    n_beta_strs: int,
    n_occ: int,
    alpha_strs: np.ndarray,
    beta_strs: np.ndarray,
    alpha_to_idx: Dict[int, int],
    beta_to_idx: Dict[int, int],
) -> np.ndarray:
    """Expand a single Schmidt product state to a full CAS CI matrix.

    This is equivalent to _expand_schmidt_product_to_ci_matrix in
    dm_svd_embedding.embedded_hamiltonian, but accepts the same elem_info
    dict format used by partition_schmidt_basis().

    Args:
        elem_info: {'n': n_A, 'alpha': α, 'beta': β}
        (other args same as _expand_schmidt_product_to_ci_matrix)

    Returns:
        CI matrix of shape (n_alpha_strs, n_beta_strs).
    """
    n_A_val = elem_info['n']
    alpha = elem_info['alpha']
    beta = elem_info['beta']

    blk_schmidt = schmidt_data[n_A_val]
    blk_partition = partition[n_A_val]

    U = blk_schmidt['U']   # (dim_A, r)
    V = blk_schmidt['V']   # (dim_B, r)
    a_dets = blk_partition['a_dets']
    b_dets = blk_partition['b_dets']

    ci_mat = np.zeros((n_alpha_strs, n_beta_strs))

    for i, (aA, bA) in enumerate(a_dets):
        u_coef = U[i, alpha]
        if abs(u_coef) < 1e-14:
            continue
        for j, (aB, bB) in enumerate(b_dets):
            v_coef = V[j, beta]
            if abs(v_coef) < 1e-14:
                continue
            # Reconstruct full-CAS determinant
            a_full = aA | (aB << n_occ)
            b_full = bA | (bB << n_occ)
            ia = alpha_to_idx.get(int(a_full))
            ib = beta_to_idx.get(int(b_full))
            if ia is not None and ib is not None:
                ci_mat[ia, ib] += u_coef * v_coef

    return ci_mat


# ═══════════════════════════════════════════════════════════════════════════
# Precomputed index table for fast CI expansion (optional optimization)
# ═══════════════════════════════════════════════════════════════════════════

def _precompute_expansion_indices(
    block_n: int,
    schmidt_data: Dict[int, Dict],
    partition: Dict[int, Dict],
    n_occ: int,
    alpha_to_idx: Dict[int, int],
    beta_to_idx: Dict[int, int],
) -> Tuple[List[Tuple[int, int, int, int]], int, int]:
    """Precompute (i, j, ia, ib) index mapping for a given occupation block n.

    This avoids repeated bit operations and dict lookups when expanding
    multiple Schmidt states from the same block.

    Args:
        block_n: Occupation number n for this block.
        schmidt_data, partition: Standard data structures.
        n_occ: Number of A-space orbitals.
        alpha_to_idx, beta_to_idx: String → index maps for full CAS.

    Returns:
        (index_pairs, dim_A, dim_B):
          index_pairs: List of (i, j, ia, ib) for valid (ia, ib) combinations.
          dim_A: Number of A-subspace determinants.
          dim_B: Number of B-subspace determinants.
    """
    blk_partition = partition[block_n]
    a_dets = blk_partition['a_dets']
    b_dets = blk_partition['b_dets']
    dim_A = len(a_dets)
    dim_B = len(b_dets)

    index_pairs = []
    for i, (aA, bA) in enumerate(a_dets):
        for j, (aB, bB) in enumerate(b_dets):
            a_full = aA | (aB << n_occ)
            b_full = bA | (bB << n_occ)
            ia = alpha_to_idx.get(int(a_full))
            ib = beta_to_idx.get(int(b_full))
            if ia is not None and ib is not None:
                index_pairs.append((i, j, ia, ib))

    return index_pairs, dim_A, dim_B


def _expand_one_fast(
    elem_info: Dict,
    schmidt_data: Dict[int, Dict],
    precomputed: Dict[int, Tuple[List, int, int]],
    n_alpha_strs: int,
    n_beta_strs: int,
) -> np.ndarray:
    """Fast expansion using precomputed index tables.

    Args:
        elem_info: {'n': n_A, 'alpha': α, 'beta': β}
        schmidt_data: Schmidt decomposition data (for U, V matrices).
        precomputed: Dict[n] → (index_pairs, dim_A, dim_B) from _precompute_expansion_indices.
        n_alpha_strs, n_beta_strs: Full CAS CI matrix dimensions.

    Returns:
        CI matrix of shape (n_alpha_strs, n_beta_strs).
    """
    n_A_val = elem_info['n']
    alpha = elem_info['alpha']
    beta = elem_info['beta']

    blk_schmidt = schmidt_data[n_A_val]
    U = blk_schmidt['U']   # (dim_A, r)
    V = blk_schmidt['V']   # (dim_B, r)

    index_pairs, dim_A, dim_B = precomputed[n_A_val]

    ci_mat = np.zeros((n_alpha_strs, n_beta_strs))

    for i, j, ia, ib in index_pairs:
        u_coef = U[i, alpha]
        if abs(u_coef) < 1e-14:
            continue
        v_coef = V[j, beta]
        if abs(v_coef) < 1e-14:
            continue
        ci_mat[ia, ib] += u_coef * v_coef

    return ci_mat


# ═══════════════════════════════════════════════════════════════════════════
# Parallel sigma helper (single vector)
# ═══════════════════════════════════════════════════════════════════════════

def _compute_sigmas_batch(
    sigma_fn: Callable,
    ci_mats: List[np.ndarray],
    n_workers: int = 1,
) -> List[np.ndarray]:
    """Compute sigma-vectors for a batch of CI matrices (parallel).

    Uses ThreadPoolExecutor since contract_2e is C-level and releases GIL.
    """
    n = len(ci_mats)
    if n == 0:
        return []
    if n_workers <= 1:
        return [sigma_fn(ci) for ci in ci_mats]

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(sigma_fn, ci): idx for idx, ci in enumerate(ci_mats)}
        results = [None] * n
        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()
    return results


# ═══════════════════════════════════════════════════════════════════════════
# StreamBuilder: main class for streaming Scheme B construction
# ═══════════════════════════════════════════════════════════════════════════

class StreamBuilder:
    """Streaming (memory-efficient) builder for H_PP, H_PQ, and H_QQ@v.

    Attributes:
        batch_size: Max number of CI matrices to keep in memory simultaneously.
        n_workers: Thread-parallel workers for sigma-vector computation.
        precomputed: Optional precomputed index tables for fast expansion.
    """

    def __init__(
        self,
        schmidt_data: Dict[int, Dict],
        partition: Dict[int, Dict],
        part_info: Dict,
        backend,
        n_occ: int,
        n_act: int,
        batch_size: int = 32,
        n_workers: int = 1,
        use_precompute: bool = True,
        verbose: bool = True,
    ):
        """Initialize StreamBuilder.

        Args:
            schmidt_data: Schmidt decomposition data (from density_matrix.py).
            partition: Occ/virt partition data (from occ_virt_partition.py).
            part_info: P/Q partition info (from partition_schmidt_basis).
            backend: KDCIBackend for sigma_full.
            n_occ: Number of A-space orbitals.
            n_act: Total active orbitals.
            batch_size: Batch size for memory control (default 32).
            n_workers: Number of parallel threads for sigma computation.
            use_precompute: Whether to precompute expansion index tables.
            verbose: Print progress.
        """
        self.schmidt_data = schmidt_data
        self.partition = partition
        self.part_info = part_info
        self.backend = backend
        self.n_occ = n_occ
        self.n_act = n_act
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.verbose = verbose

        # Extract basis lists
        self.p_basis = part_info['p_basis']
        self.q_basis = part_info['q_basis']
        self.p_dim = part_info['p_dim']
        self.q_dim = part_info['q_dim']

        # CAS string info
        self.alpha_strs = backend.q_idx.alpha_strs
        self.beta_strs = backend.q_idx.beta_strs
        self.n_alpha_strs = len(self.alpha_strs)
        self.n_beta_strs = len(self.beta_strs)
        self.M = self.n_alpha_strs * self.n_beta_strs
        self.alpha_to_idx = {int(s): i for i, s in enumerate(self.alpha_strs)}
        self.beta_to_idx = {int(s): i for i, s in enumerate(self.beta_strs)}

        # Precompute expansion index tables (optional, ~MB memory)
        self.precomputed = None
        if use_precompute:
            self._build_precomputed()

        # Lazy cache for Q-basis CI flat vectors (for H_QQ@v)
        self._q_ci_flat_cache: Dict[int, np.ndarray] = {}

        if verbose:
            print(f"  [StreamBuilder] CAS({n_act},{sum(backend.q_idx.nelec)}), "
                  f"M={self.M:,}, |P|={self.p_dim}, |Q|={self.q_dim}")
            print(f"    batch_size={batch_size}, n_workers={n_workers}, "
                  f"precompute={'on' if self.precomputed else 'off'}")
            ci_mb = self.n_alpha_strs * self.n_beta_strs * 8 / (1024 * 1024)
            print(f"    CI matrix size: {ci_mb:.1f} MB, "
                  f"peak mem: ~{2 * batch_size * ci_mb:.0f} MB")

    def _build_precomputed(self):
        """Precompute index tables for all occupation blocks."""
        t0 = time.perf_counter()
        self.precomputed = {}
        for n_A in sorted(self.schmidt_data.keys()):
            if self.schmidt_data[n_A]['r'] == 0:
                continue
            self.precomputed[n_A] = _precompute_expansion_indices(
                n_A, self.schmidt_data, self.partition,
                self.n_occ, self.alpha_to_idx, self.beta_to_idx)
        if self.verbose:
            total_pairs = sum(len(p[0]) for p in self.precomputed.values())
            print(f"    Precomputed {total_pairs:,} index pairs "
                  f"({time.perf_counter() - t0:.1f}s)", flush=True)

    # ── Expansion ──

    def expand_one(self, elem_info: Dict) -> np.ndarray:
        """Expand a single Schmidt basis state to CI matrix."""
        if self.precomputed is not None:
            return _expand_one_fast(
                elem_info, self.schmidt_data, self.precomputed,
                self.n_alpha_strs, self.n_beta_strs)
        else:
            return _expand_one(
                elem_info, self.schmidt_data, self.partition,
                self.n_alpha_strs, self.n_beta_strs, self.n_occ,
                self.alpha_strs, self.beta_strs,
                self.alpha_to_idx, self.beta_to_idx)

    def expand_batch(self, basis_slice: List[Dict]) -> List[np.ndarray]:
        """Expand a batch of Schmidt basis states to CI matrices."""
        return [self.expand_one(info) for info in basis_slice]

    def expand_all_to_flat(self, basis_list: List[Dict]) -> np.ndarray:
        """Expand all basis states and stack as (M, N) flat array.

        WARNING: This allocates M × N memory. Only use when |basis| is small
        and M × |basis| fits in RAM. For large |basis|, use expand_batch()
        with streaming.
        """
        N = len(basis_list)
        result = np.empty((self.M, N))
        for k, info in enumerate(basis_list):
            ci = self.expand_one(info)
            result[:, k] = ci.reshape(-1)
        return result

    # ── Sigma-vector computation ──

    def sigma_one(self, ci_mat: np.ndarray) -> np.ndarray:
        """Compute H·v for a single CI matrix."""
        return self.backend.sigma_full(ci_mat)

    def sigma_batch(self, ci_mats: List[np.ndarray]) -> List[np.ndarray]:
        """Compute H·v for a batch of CI matrices (parallel)."""
        return _compute_sigmas_batch(
            self.backend.sigma_full, ci_mats, n_workers=self.n_workers)

    def sigma_batch_from_basis(self, basis_slice: List[Dict]) -> List[np.ndarray]:
        """Expand a batch of basis states then compute sigmas.

        Peak memory: batch_size × (1 CI expand + 1 sigma) = 2 × batch_size × CI_size
        """
        ci_mats = self.expand_batch(basis_slice)
        sigmas = self.sigma_batch(ci_mats)
        return ci_mats, sigmas

    # ═══════════════════════════════════════════════════════════
    # H_PP: streaming construction
    # ═══════════════════════════════════════════════════════════

    def build_hpp(self) -> np.ndarray:
        """Build H_PP (|P|×|P|) using streaming batches.

        Algorithm:
          For each outer batch of P states (size B):
            1. Expand k-batch → CI mats
            2. Compute sigmas for k-batch
            3. For each inner batch of P states (size B):
                 a. Expand l-batch → CI mats
                 b. BLAS3: H_PP[l_batch, k_batch] = C_l^T @ S_k
            4. Free k-batch CI mats and sigmas

        Memory: O(2 × batch_size × M) peak.
        Sigma calls: |P| (one per P-basis state).
        """
        p_dim = self.p_dim
        if p_dim == 0:
            return np.zeros((0, 0))

        H_PP = np.zeros((p_dim, p_dim))
        B = min(self.batch_size, p_dim)

        if self.verbose:
            t0 = time.perf_counter()
            print(f"  [H_PP streaming] |P|={p_dim}, batch={B}, "
                  f"sigma calls={p_dim}", flush=True)

        # Pre-expand ALL P-basis states once to flat (M, |P|) matrix
        # and cache it. This trades memory for speed: avoids O(|P|²) expansions.
        # If |P|×M doesn't fit, fall back to per-batch expansion.
        p_dim_mem_gb = p_dim * self.M * 8 / (1024**3)
        use_full_cache = p_dim_mem_gb < 4.0  # cache if < 4 GB

        if use_full_cache:
            if self.verbose:
                print(f"    Full P-cache mode ({p_dim_mem_gb:.1f} GB for "
                      f"{p_dim}×{self.M:,})", flush=True)
            # Expand all P states once
            C_P_full = np.empty((self.M, p_dim))
            S_P_full = np.empty((self.M, p_dim))
            for k_start in range(0, p_dim, B):
                k_end = min(k_start + B, p_dim)
                if self.verbose:
                    print(f"    Expanding & sigma P[{k_start}:{k_end}]...", flush=True)
                ci_mats_k, sigmas_k = self.sigma_batch_from_basis(
                    self.p_basis[k_start:k_end])
                for idx_in_batch, (ci, sigma) in enumerate(zip(ci_mats_k, sigmas_k)):
                    k = k_start + idx_in_batch
                    C_P_full[:, k] = ci.reshape(-1)
                    S_P_full[:, k] = sigma.reshape(-1)
                del ci_mats_k, sigmas_k

            # Single BLAS3: H_PP = C^T @ S
            H_PP = C_P_full.T @ S_P_full
            del C_P_full, S_P_full
        else:
            if self.verbose:
                print(f"    Batch mode ({p_dim_mem_gb:.1f} GB would exceed "
                      f"cache limit)", flush=True)
            # Streaming: outer loop over k-batches, inner over l-batches
            for k_start in range(0, p_dim, B):
                k_end = min(k_start + B, p_dim)
                if self.verbose:
                    print(f"    Outer batch P[{k_start}:{k_end}] / {p_dim}...",
                          flush=True)

                ci_mats_k, sigmas_k = self.sigma_batch_from_basis(
                    self.p_basis[k_start:k_end])
                S_k_flat = np.empty((self.M, k_end - k_start))
                for idx_in_batch, sigma in enumerate(sigmas_k):
                    S_k_flat[:, idx_in_batch] = sigma.reshape(-1)
                del sigmas_k

                # Inner loop over l-batches
                for l_start in range(0, p_dim, B):
                    l_end = min(l_start + B, p_dim)
                    ci_mats_l = self.expand_batch(self.p_basis[l_start:l_end])
                    C_l_flat = np.empty((self.M, l_end - l_start))
                    for idx_in_batch, ci in enumerate(ci_mats_l):
                        C_l_flat[:, idx_in_batch] = ci.reshape(-1)
                    del ci_mats_l

                    # BLAS3 block
                    H_PP[l_start:l_end, k_start:k_end] = C_l_flat.T @ S_k_flat
                    del C_l_flat

                del ci_mats_k, S_k_flat

        H_PP = 0.5 * (H_PP + H_PP.T)

        if self.verbose:
            elapsed = time.perf_counter() - t0
            print(f"    H_PP done: {elapsed:.0f}s", flush=True)

        return H_PP

    # ═══════════════════════════════════════════════════════════
    # H_PQ: streaming construction
    # ═══════════════════════════════════════════════════════════

    def build_hpq(self) -> np.ndarray:
        """Build H_PQ (|P|×|Q|) using streaming batches.

        Strategy: sigma on the SMALLER side, project to the larger side.
        Memory: O(2 × batch_size × M) peak.
        Sigma calls: min(|P|, |Q|).
        """
        p_dim = self.p_dim
        q_dim = self.q_dim
        if p_dim == 0 or q_dim == 0:
            return np.zeros((p_dim, q_dim))

        H_PQ = np.zeros((p_dim, q_dim))
        B = min(self.batch_size, p_dim, q_dim)

        if self.verbose:
            t0 = time.perf_counter()
            print(f"  [H_PQ streaming] |P|={p_dim}, |Q|={q_dim}, batch={B}",
                  flush=True)

        # Check if we can cache the larger side's CI expansions
        p_mem_gb = p_dim * self.M * 8 / (1024**3)
        q_mem_gb = q_dim * self.M * 8 / (1024**3)
        cache_limit_gb = 4.0

        if q_dim <= p_dim:
            # Sigma on Q (smaller side), project to P (larger side)
            if self.verbose:
                print(f"    Sigma on Q ({q_dim}), project to P ({p_dim})",
                      flush=True)

            # Can we cache all P states?
            cache_P = p_mem_gb < cache_limit_gb
            if cache_P:
                if self.verbose:
                    print(f"    Pre-caching all P states ({p_mem_gb:.1f} GB)...",
                          flush=True)
                C_P_full = self.expand_all_to_flat(self.p_basis)

            for k_start in range(0, q_dim, B):
                k_end = min(k_start + B, q_dim)
                if self.verbose:
                    print(f"    Q batch [{k_start}:{k_end}] / {q_dim}...",
                          flush=True)

                ci_mats_q, sigmas_q = self.sigma_batch_from_basis(
                    self.q_basis[k_start:k_end])
                S_k_flat = np.empty((self.M, k_end - k_start))
                for idx_s, sigma in enumerate(sigmas_q):
                    S_k_flat[:, idx_s] = sigma.reshape(-1)
                del sigmas_q, ci_mats_q

                if cache_P:
                    # H_PQ[:, k_start:k_end] = C_P_full.T @ S_k_flat
                    H_PQ[:, k_start:k_end] = C_P_full.T @ S_k_flat
                else:
                    # Batch over P
                    for l_start in range(0, p_dim, B):
                        l_end = min(l_start + B, p_dim)
                        ci_mats_p = self.expand_batch(self.p_basis[l_start:l_end])
                        C_l_flat = np.empty((self.M, l_end - l_start))
                        for idx_c, ci in enumerate(ci_mats_p):
                            C_l_flat[:, idx_c] = ci.reshape(-1)
                        del ci_mats_p
                        H_PQ[l_start:l_end, k_start:k_end] = (
                            C_l_flat.T @ S_k_flat)
                        del C_l_flat

                del S_k_flat

            if cache_P:
                del C_P_full
        else:
            # Sigma on P (smaller side), project to Q → transpose
            if self.verbose:
                print(f"    Sigma on P ({p_dim}), project to Q ({q_dim}) "
                      f"→ transpose", flush=True)

            H_QP = np.zeros((q_dim, p_dim))

            cache_Q = q_mem_gb < cache_limit_gb
            if cache_Q:
                if self.verbose:
                    print(f"    Pre-caching all Q states ({q_mem_gb:.1f} GB)...",
                          flush=True)
                C_Q_full = self.expand_all_to_flat(self.q_basis)

            for k_start in range(0, p_dim, B):
                k_end = min(k_start + B, p_dim)
                if self.verbose:
                    print(f"    P batch [{k_start}:{k_end}] / {p_dim}...",
                          flush=True)

                ci_mats_p, sigmas_p = self.sigma_batch_from_basis(
                    self.p_basis[k_start:k_end])
                S_k_flat = np.empty((self.M, k_end - k_start))
                for idx_s, sigma in enumerate(sigmas_p):
                    S_k_flat[:, idx_s] = sigma.reshape(-1)
                del sigmas_p, ci_mats_p

                if cache_Q:
                    H_QP[:, k_start:k_end] = C_Q_full.T @ S_k_flat
                else:
                    for l_start in range(0, q_dim, B):
                        l_end = min(l_start + B, q_dim)
                        ci_mats_q = self.expand_batch(self.q_basis[l_start:l_end])
                        C_l_flat = np.empty((self.M, l_end - l_start))
                        for idx_c, ci in enumerate(ci_mats_q):
                            C_l_flat[:, idx_c] = ci.reshape(-1)
                        del ci_mats_q
                        H_QP[l_start:l_end, k_start:k_end] = (
                            C_l_flat.T @ S_k_flat)
                        del C_l_flat

                del S_k_flat

            if cache_Q:
                del C_Q_full

            H_PQ = H_QP.T

        if self.verbose:
            elapsed = time.perf_counter() - t0
            print(f"    H_PQ done: {elapsed:.0f}s", flush=True)

        return H_PQ

    # ═══════════════════════════════════════════════════════════
    # H_QQ @ v: matrix-free matvec (with optional Q-cache)
    # ═══════════════════════════════════════════════════════════

    def _get_q_ci_flat(self, q_idx_flat: int) -> np.ndarray:
        """Get CI flat vector for Q-basis state q_idx_flat (lazy cache)."""
        if q_idx_flat not in self._q_ci_flat_cache:
            ci = self.expand_one(self.q_basis[q_idx_flat])
            self._q_ci_flat_cache[q_idx_flat] = ci.reshape(-1).copy()
        return self._q_ci_flat_cache[q_idx_flat]

    def prewarm_q_cache(self):
        """Pre-expand and cache ALL Q-basis CI flat vectors.

        This is a one-time cost that dramatically speeds up subsequent
        H_QQ@v calls during Krylov propagation. Only use when
        |Q| × M fits in RAM.
        """
        q_dim = self.q_dim
        mem_gb = q_dim * self.M * 8 / (1024**3)
        if self.verbose:
            print(f"  [Q cache] Expanding {q_dim} Q-basis states "
                  f"({mem_gb:.1f} GB)...", flush=True)
        t0 = time.perf_counter()
        for q_idx in range(q_dim):
            self._get_q_ci_flat(q_idx)
            if self.verbose and (q_idx + 1) % max(1, q_dim // 10) == 0:
                print(f"    {q_idx + 1}/{q_dim} "
                      f"({time.perf_counter() - t0:.0f}s)", flush=True)
        if self.verbose:
            print(f"    Q cache done: {time.perf_counter() - t0:.0f}s, "
                  f"{mem_gb:.1f} GB", flush=True)

    def apply_hqq(self, v: np.ndarray) -> np.ndarray:
        """Compute H_QQ @ v on-the-fly (matrix-free).

        Steps:
          1. Build combined CI matrix: C = Σ_q v[q] · CI_mat_q
          2. Compute sigma = H · C
          3. Project back: result[q] = ⟨CI_mat_q | sigma⟩

        If Q-cache is warmed, steps 1 and 3 use cached CI flat vectors.
        Otherwise, each Q-state is expanded on-the-fly (slower but lower memory).

        Args:
            v: (|Q|,) vector in Q-space.

        Returns:
            result: (|Q|,) = H_QQ @ v
        """
        q_dim = self.q_dim
        if q_dim == 0:
            return np.zeros(0)

        use_cache = len(self._q_ci_flat_cache) > 0

        if use_cache:
            # Fast path: cached CI flat vectors
            # Step 1: linear combination
            ci_combined_flat = np.zeros(self.M)
            for q_idx in range(q_dim):
                v_val = v[q_idx]
                if abs(v_val) < 1e-14:
                    continue
                ci_combined_flat += v_val * self._get_q_ci_flat(q_idx)

            # Step 2: sigma
            sigma_mat = self.backend.sigma_full(
                ci_combined_flat.reshape(self.n_alpha_strs, self.n_beta_strs))
            sigma_flat = sigma_mat.reshape(-1)

            # Step 3: projection
            result = np.zeros(q_dim)
            for q_idx in range(q_dim):
                result[q_idx] = np.dot(self._get_q_ci_flat(q_idx), sigma_flat)
        else:
            # Slow path: on-the-fly expansion (no Q cache)
            ci_combined = np.zeros((self.n_alpha_strs, self.n_beta_strs))
            for q_idx in range(q_dim):
                v_val = v[q_idx]
                if abs(v_val) < 1e-14:
                    continue
                ci_q = self.expand_one(self.q_basis[q_idx])
                ci_combined += v_val * ci_q

            sigma_mat = self.backend.sigma_full(ci_combined)
            sigma_flat = sigma_mat.reshape(-1)

            result = np.zeros(q_dim)
            for q_idx in range(q_dim):
                ci_q = self.expand_one(self.q_basis[q_idx])
                result[q_idx] = np.dot(ci_q.reshape(-1), sigma_flat)

        return result

    def apply_hqq_batch(self, B: np.ndarray) -> np.ndarray:
        """Compute H_QQ @ B for a batch of vectors (columns of B).

        Args:
            B: (|Q|, r) matrix.

        Returns:
            HQQ_B: (|Q|, r) = H_QQ @ B
        """
        q_dim, r = B.shape
        if q_dim == 0 or r == 0:
            return np.zeros((q_dim, r))

        result = np.zeros((q_dim, r))
        for j in range(r):
            result[:, j] = self.apply_hqq(B[:, j])
            if self.verbose and r > 10 and (j + 1) % max(1, r // 10) == 0:
                print(f"    H_QQ@B col {j+1}/{r}", flush=True)

        return result

    def get_hqq_diag(self) -> np.ndarray:
        """Extract the diagonal of H_QQ.

        H_QQ[q,q] = ⟨CI_q | H | CI_q⟩ — the expectation value of H in
        each Q-basis Schmidt state. This can be computed efficiently
        using the backend's diagonal.
        """
        q_dim = self.q_dim
        if q_dim == 0:
            return np.zeros(0)

        # H_QQ_diag[q] = ⟨CI_q | H | CI_q⟩
        # Since CI_q is a linear combination of full CAS determinants,
        # we can't just use the CAS diagonal. We compute it via sigma.
        hqq_diag = np.zeros(q_dim)
        for q_idx in range(q_dim):
            ci_q = self.expand_one(self.q_basis[q_idx])
            sigma_q = self.backend.sigma_full(ci_q)
            hqq_diag[q_idx] = np.dot(ci_q.reshape(-1), sigma_q.reshape(-1))
            if self.verbose and q_dim > 100 and (q_idx + 1) % max(1, q_dim // 10) == 0:
                print(f"    H_QQ_diag {q_idx+1}/{q_dim}", flush=True)
        return hqq_diag

    # ═══════════════════════════════════════════════════════════
    # Convenience: build all blocks
    # ═══════════════════════════════════════════════════════════

    def build_all(
        self,
        prewarm_q: bool = False,
    ) -> Dict:
        """Build H_PP, H_PQ, and return H_QQ matvec callables.

        Args:
            prewarm_q: If True, pre-expand and cache all Q-basis CI flat
                       vectors. This trades memory (~|Q|×M×8 bytes) for
                       speed in subsequent H_QQ@v calls.

        Returns:
            dict with:
              'H_PP': (|P|,|P|) hermitian matrix
              'H_PQ': (|P|,|Q|) matrix
              'H_QQ_matvec': callable (|Q|,) → (|Q|,) for H_QQ@v
              'H_QQ_batch': callable (|Q|,r) → (|Q|,r) for batch H_QQ@B
              'H_QQ_diag': (|Q|,) array of H_QQ diagonal elements
              'builder': self (for further operations)
        """
        H_PP = self.build_hpp()
        H_PQ = self.build_hpq()

        if prewarm_q:
            self.prewarm_q_cache()

        H_QQ_diag = self.get_hqq_diag()

        return {
            'H_PP': H_PP,
            'H_PQ': H_PQ,
            'H_QQ_matvec': self.apply_hqq,
            'H_QQ_batch': self.apply_hqq_batch,
            'H_QQ_diag': H_QQ_diag,
            'builder': self,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════

def test_precompute_indices():
    """Test precomputed index tables on toy data."""
    # Toy: n_occ=2, n_act=4, A=2 orbs, B=2 orbs
    # A subspace: 2 alpha electrons in 2 orbs → 1 det
    # B subspace: 1 electron in 2 orbs → 2 dets
    # Full CAS: 3 alpha electrons in 4 orbs → ... (just verify no crash)
    n_occ = 2
    n_act = 4

    # Build minimal schmidt_data and partition manually
    # For testing, just verify the function doesn't crash
    # and returns correct dimensions.

    # Simple: A has 1 det (bit 0b11, bit 0b0), B has 1 det (0b1, 0b0)
    schmidt = {
        3: {'r': 1, 'U': np.eye(1), 'V': np.eye(1), 'sigma': np.array([1.0])},
    }
    part = {
        3: {
            'a_dets': [(0b11, 0b0)],   # 2 alpha in A (bits 0,1), 0 beta
            'b_dets': [(0b1, 0b0)],    # 1 alpha in B (bit 0), 0 beta
            'a_index': {(0b11, 0b0): 0},
            'b_index': {(0b1, 0b0): 0},
        },
    }

    # Full CAS alpha strings (3 alpha in 4 orbs)
    # We need at least the strings that will be accessed
    alpha_strs = np.array([0b111, 0b1011, 0b1101, 0b1110], dtype=np.int64)
    beta_strs = np.array([0b0], dtype=np.int64)
    alpha_to_idx = {int(s): i for i, s in enumerate(alpha_strs)}
    beta_to_idx = {int(s): i for i, s in enumerate(beta_strs)}

    index_pairs, dim_A, dim_B = _precompute_expansion_indices(
        3, schmidt, part, n_occ, alpha_to_idx, beta_to_idx)

    assert dim_A == 1
    assert dim_B == 1
    # A-det: aA=0b11, bA=0b0. B-det: aB=0b1, bB=0b0
    # a_full = 0b11 | (0b1 << 2) = 0b111 → alpha_to_idx[0b111] = 0
    # b_full = 0b0 | (0b0 << 2) = 0b0 → beta_to_idx[0b0] = 0
    assert len(index_pairs) >= 1
    assert index_pairs[0] == (0, 0, 0, 0)

    print("  ✓ _precompute_expansion_indices: toy test passed")


def test_expand_one_fast():
    """Test fast expansion vs slow expansion."""
    n_occ = 2
    n_act = 4

    schmidt = {
        3: {'r': 1, 'U': np.array([[1.0]]), 'V': np.array([[1.0]]),
            'sigma': np.array([1.0])},
    }
    part = {
        3: {
            'a_dets': [(0b11, 0b0)],
            'b_dets': [(0b1, 0b0)],
            'a_index': {(0b11, 0b0): 0},
            'b_index': {(0b1, 0b0): 0},
        },
    }

    alpha_strs = np.array([0b111, 0b1011, 0b1101, 0b1110], dtype=np.int64)
    beta_strs = np.array([0b0], dtype=np.int64)
    alpha_to_idx = {int(s): i for i, s in enumerate(alpha_strs)}
    beta_to_idx = {int(s): i for i, s in enumerate(beta_strs)}

    elem_info = {'n': 3, 'alpha': 0, 'beta': 0}

    # Slow expansion
    ci_slow = _expand_one(
        elem_info, schmidt, part,
        len(alpha_strs), len(beta_strs), n_occ,
        alpha_strs, beta_strs, alpha_to_idx, beta_to_idx)

    # Fast expansion
    precomputed = {}
    for n_A in schmidt:
        if schmidt[n_A]['r'] > 0:
            precomputed[n_A] = _precompute_expansion_indices(
                n_A, schmidt, part, n_occ, alpha_to_idx, beta_to_idx)

    ci_fast = _expand_one_fast(
        elem_info, schmidt, precomputed,
        len(alpha_strs), len(beta_strs))

    assert np.allclose(ci_slow, ci_fast)
    print("  ✓ _expand_one_fast: matches slow expansion")


if __name__ == "__main__":
    test_precompute_indices()
    test_expand_one_fast()
    print("All streaming_ops tests passed.")