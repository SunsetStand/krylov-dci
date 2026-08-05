#!/usr/bin/env python3
"""
Partition Q-space into Q₁, Q₂, ... sub-blocks by electron count n.

Given the P/Q partition from schmidt_partition.py, further subdivide the
Q-space into sectors Q_n = {states with B-space electron count n}.
Exploits selection rules |Δn| ≤ 2 to build only the banded-diagonal blocks
of H_QQ and the non-zero H_{P Q_n} couplings.

Supports two modes:
  - Scheme A: Extract sub-blocks from a pre-built full H^emb (D×D).
  - Scheme B: Build blocks directly via sigma-vector projection (matrix-free).

Selection Rules (from two-body Hamiltonian):
  H_{P Q_n}  ≠ 0  only for |n - n_P| ≤ 2  (n_P ∈ p_blocks)
  H_{Q_m Q_n} ≠ 0  only for |m - n| ≤ 2
"""

import numpy as np
import time as time_mod
from typing import Dict, List, Tuple, Optional, Set
from concurrent.futures import ThreadPoolExecutor, as_completed


# ═══════════════════════════════════════════════════════════════════════════
# Step 1: Subdivide Q-space by electron number n
# ═══════════════════════════════════════════════════════════════════════════

def partition_qspace_by_n(
    part_info: Dict,
    schmidt_data: Dict,
    p_blocks: List[int],
) -> Dict:
    """Subdivide Q-space states by their B-space electron count n.

    Input `part_info` comes from `schmidt_partition.partition_schmidt_basis()`,
    which already separates P and Q states.  Here we further group the Q states
    into Q₁, Q₂, …, Q_N according to their `n` label.

    Args:
        part_info:    Output of partition_schmidt_basis().
        schmidt_data: Schmidt decomposition data (needed for per-block rank).
        p_blocks:     n-values included in P-space.

    Returns:
        dict with:
          'q_blocks':     Dict[int] → {
                              'basis':   List[Dict] — Q-states in this block,
                              'dim':     int — d_n = number of states,
                              'indices': (d_n,) int64 — flat indices in H^emb,
                              'r':       int — Schmidt rank for this n,
                              'n':       int — B-space electron count,
                          }
          'q_n_list':     List[int] — sorted Q_n values.
          'active_q':     List[int] — Q_n blocks that can couple to P
                                     (|n - n_P| ≤ 2 for some n_P ∈ p_blocks).
          'n_total':      int — total number of Q-blocks.
    """
    p_set = set(p_blocks)
    q_basis = part_info.get('q_basis', [])

    if len(q_basis) == 0:
        return {
            'q_blocks': {},
            'q_n_list': [],
            'active_q': [],
            'n_total': 0,
        }

    # Group Q-basis states by n
    q_by_n: Dict[int, List[Dict]] = {}
    for st in q_basis:
        n_val = st['n']
        q_by_n.setdefault(n_val, []).append(st)

    # Build per-block data
    q_blocks = {}
    for n_val in sorted(q_by_n.keys()):
        states = q_by_n[n_val]
        indices = np.array([s['flat_idx_q'] for s in states], dtype=np.int64)
        # Recover Schmidt rank from schmidt_data
        r_n = schmidt_data.get(n_val, {}).get('r', 0)
        q_blocks[n_val] = {
            'basis': states,
            'dim': len(states),
            'indices': indices,
            'r': r_n,
            'n': n_val,
        }

    q_n_list = sorted(q_blocks.keys())

    # Determine which Q_n blocks are "active" (can couple to P)
    # For each n_Q, check if there exists n_P ∈ p_blocks with |n_Q - n_P| ≤ 2
    active_q = []
    for n_Q in q_n_list:
        for n_P in p_set:
            if abs(n_Q - n_P) <= 2:
                active_q.append(n_Q)
                break

    return {
        'q_blocks': q_blocks,
        'q_n_list': q_n_list,
        'active_q': active_q,
        'n_total': len(q_n_list),
    }


def _get_active_q_pairs(
    active_q: List[int],
    p_blocks: List[int],
) -> Tuple[List[int], List[Tuple[int, int]]]:
    """Determine which Q_n blocks and Q_m↔Q_n pairs are needed for k=1 theory.

    Based on selection rules:
      - H_{P Q_n}  ≠ 0  requires ∃ n_P ∈ p_blocks with |n - n_P| ≤ 2.
      - H_{Q_m Q_n} ≠ 0  requires |m - n| ≤ 2.

    For k=1 Neumann expansion, we only need Q_n blocks that are within
    distance 2 of P.  Higher Q_n only couple indirectly and their effect
    would require higher-order expansions.

    Returns:
        (needed_q_list, q_pairs):
          needed_q_list: sorted list of Q_n values that couple directly to P.
          q_pairs:       list of (m, n) tuples with m < n and |m-n| ≤ 2,
                         restricted to needed_q_list.
    """
    p_set = set(p_blocks)
    needed_q = set()

    for n_Q in active_q:
        if n_Q in p_set:
            continue  # skip P-space blocks (they should not appear in active_q
                      # in practice, but defensive coding)
        for n_P in p_set:
            if abs(n_Q - n_P) <= 2:
                needed_q.add(n_Q)
                break

    needed_q_list = sorted(needed_q)

    # Build Q_m ↔ Q_n pairs within needed_q_list
    q_pairs = []
    for i, m in enumerate(needed_q_list):
        for n in needed_q_list[i + 1:]:
            if abs(m - n) <= 2:
                q_pairs.append((m, n))

    return needed_q_list, q_pairs


# ═══════════════════════════════════════════════════════════════════════════
# Step 2: Extract Hamiltonian sub-blocks from full H^emb  (Scheme A)
# ═══════════════════════════════════════════════════════════════════════════

def extract_q_blocks_scheme_a(
    H_emb: np.ndarray,
    part_info: Dict,
    q_partition: Dict,
    p_blocks: List[int],
    verbose: bool = True,
) -> Dict:
    """Extract needed H_{P Q_n}, H_{Q_m Q_n} blocks from a full H^emb.

    Scheme A: Full H^emb was already built (D×D), we slice out the required
    sub-blocks using pre-computed flat indices.

    Args:
        H_emb:       Full embedded Hamiltonian, shape (D, D).
        part_info:   Output of partition_schmidt_basis().
        q_partition: Output of partition_qspace_by_n().
        p_blocks:    P-space n-values.
        verbose:     Print extraction summary.

    Returns:
        dict with:
          'H_PP':       (|P|, |P|) P-space block.
          'H_PQ':       Dict[n] → H_{P Q_n} (|P|, d_n) for each active Q_n.
          'H_QQ_blocks': Dict[(m,n)] → H_{Q_m Q_n} (d_m, d_n) for |m-n|≤2.
          'H_QQ_diag':  Dict[n] → (d_n,) diagonal of H_{Q_n Q_n}.
          'dims':       {'P': int, 'Q_by_n': Dict[n→int]}.
    """
    p_indices = part_info['p_indices']   # (|P|,) flat indices in H^emb
    q_blocks = q_partition['q_blocks']
    active_q = q_partition.get('active_q', sorted(q_blocks.keys()))

    needed_q, q_pairs = _get_active_q_pairs(active_q, p_blocks)

    if verbose:
        print(f"  [Q-partition] Active Q_n blocks: {active_q}")
        print(f"  [Q-partition] Needed for k=1: {needed_q}")
        print(f"  [Q-partition] Q_m↔Q_n pairs: {q_pairs}")

    p_dim = len(p_indices)

    # ── Extract H_PP ──
    if p_dim > 0:
        H_PP = H_emb[np.ix_(p_indices, p_indices)]
        H_PP = 0.5 * (H_PP + H_PP.T)
    else:
        H_PP = np.zeros((0, 0))

    # ── Extract H_{P Q_n} for needed Q_n ──
    H_PQ = {}
    for n_Q in needed_q:
        if n_Q not in q_blocks:
            continue
        q_idx = q_blocks[n_Q]['indices']  # flat indices in Q-space
        # Map to full H^emb: q_indices_full = part_info['q_indices'][q_idx]
        q_idx_full = part_info['q_indices'][q_idx]
        if p_dim > 0 and len(q_idx_full) > 0:
            H_PQ[n_Q] = H_emb[np.ix_(p_indices, q_idx_full)]
        else:
            H_PQ[n_Q] = np.zeros((p_dim, len(q_idx_full)))

    # ── Extract H_{Q_m Q_n} for |m-n| ≤ 2 ──
    H_QQ_blocks = {}
    H_QQ_diag = {}

    # Diagonals: H_{Q_n Q_n}
    for n_Q in needed_q:
        if n_Q not in q_blocks:
            continue
        q_idx = q_blocks[n_Q]['indices']
        q_idx_full = part_info['q_indices'][q_idx]
        d_n = len(q_idx_full)
        if d_n > 0:
            H_nn = H_emb[np.ix_(q_idx_full, q_idx_full)]
            H_nn = 0.5 * (H_nn + H_nn.T)
            H_QQ_blocks[(n_Q, n_Q)] = H_nn
            H_QQ_diag[n_Q] = np.diag(H_nn).copy()
        else:
            H_QQ_blocks[(n_Q, n_Q)] = np.zeros((0, 0))
            H_QQ_diag[n_Q] = np.zeros(0)

    # Off-diagonals: H_{Q_m Q_n} for m < n, |m-n| ≤ 2
    for m, n in q_pairs:
        if m not in q_blocks or n not in q_blocks:
            continue
        q_idx_m = q_blocks[m]['indices']
        q_idx_n = q_blocks[n]['indices']
        q_idx_full_m = part_info['q_indices'][q_idx_m]
        q_idx_full_n = part_info['q_indices'][q_idx_n]
        d_m, d_n = len(q_idx_full_m), len(q_idx_full_n)
        if d_m > 0 and d_n > 0:
            H_mn = H_emb[np.ix_(q_idx_full_m, q_idx_full_n)]
            H_QQ_blocks[(m, n)] = H_mn
            H_QQ_blocks[(n, m)] = H_mn.T  # hermitian conjugate
        else:
            H_QQ_blocks[(m, n)] = np.zeros((d_m, d_n))
            H_QQ_blocks[(n, m)] = np.zeros((d_n, d_m))

    # ── Build dims summary ──
    dims = {
        'P': p_dim,
        'Q_by_n': {n_Q: q_blocks[n_Q]['dim'] for n_Q in needed_q if n_Q in q_blocks},
    }

    if verbose:
        print(f"  [Q-partition] Extracted blocks:")
        print(f"    H_PP: {H_PP.shape}")
        for n_Q in needed_q:
            if n_Q in H_PQ:
                print(f"    H_PQ_{n_Q}: {H_PQ[n_Q].shape}")
        for (m, n), mat in sorted(H_QQ_blocks.items()):
            if m <= n:
                print(f"    H_Q{m}Q{n}: {mat.shape}")
        print(f"    Diagonals: { {n: len(d) for n, d in H_QQ_diag.items()} }")

    return {
        'H_PP': H_PP,
        'H_PQ': H_PQ,
        'H_QQ_blocks': H_QQ_blocks,
        'H_QQ_diag': H_QQ_diag,
        'dims': dims,
        'needed_q': needed_q,
        'q_pairs': q_pairs,
        'active_q': active_q,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Step 2b: Build sub-blocks directly via sigma-vector projection  (Scheme B)
# ═══════════════════════════════════════════════════════════════════════════

def _expand_q_basis_to_ci_mats(
    basis_list: List[Dict],
    schmidt_data: Dict,
    partition: Dict,
    alpha_strs: np.ndarray,
    beta_strs: np.ndarray,
    alpha_to_idx: Dict[int, int],
    beta_to_idx: Dict[int, int],
    n_occ: int,
) -> List[np.ndarray]:
    """Helper: expand a list of Q-basis Schmidt states to full CAS CI matrices."""
    from dm_svd_embedding.embedded_hamiltonian import _expand_schmidt_product_to_ci_matrix

    n_alpha_strs = len(alpha_strs)
    n_beta_strs = len(beta_strs)
    ci_mats = []
    for info in basis_list:
        n_val = info['n']
        alpha = info['alpha']
        beta = info['beta']
        blk_schmidt = schmidt_data[n_val]
        blk_partition = partition[n_val]
        ci_mat = _expand_schmidt_product_to_ci_matrix(
            alpha, beta, blk_schmidt, blk_partition,
            n_alpha_strs, n_beta_strs, n_occ,
            alpha_strs, beta_strs, alpha_to_idx, beta_to_idx)
        ci_mats.append(ci_mat)
    return ci_mats


def _project_onto_basis(
    sigmas: List[np.ndarray],
    basis_list: List[Dict],
    dim_target: int,
) -> np.ndarray:
    """Project sigma-vectors onto target basis.

    Given sigma_k = H · |v_k⟩ (sigma on source basis k) and a target basis
    list, compute H_target_source[l, k] = ⟨target_l | sigma_k⟩.

    Args:
        sigmas:     List of (n_alpha, n_beta) sigma matrices, length = n_source.
        basis_list: List of target basis state dicts, length = n_target.
        dim_target: Number of target basis states.

    Returns:
        H_block: (n_target, n_source) matrix.
    """
    n_source = len(sigmas)
    if n_source == 0 or dim_target == 0:
        return np.zeros((dim_target, n_source))

    M_flat = sigmas[0].size
    S_flat = np.empty((M_flat, n_source))
    for k in range(n_source):
        S_flat[:, k] = sigmas[k].reshape(-1)

    # Target basis CI matrices
    from dm_svd_embedding.embedded_hamiltonian import _expand_schmidt_product_to_ci_matrix
    # We need schmidt_data, partition, etc. — but this path is complex.
    # For now, we provide the interface; full implementation uses
    # the same expansion logic as build_hpp_direct / build_hpq_direct
    # in schmidt_partition.py.

    # Placeholder: callers should use the helper directly.
    raise NotImplementedError("Use _expand_q_basis_to_ci_mats + BLAS3 projection directly.")


def extract_q_blocks_scheme_b(
    schmidt_data: Dict,
    partition: Dict,
    part_info: Dict,
    q_partition: Dict,
    p_blocks: List[int],
    backend,
    n_occ: int,
    n_act: int,
    n_workers: int = 1,
    ecore: float = 0.0,
    verbose: bool = True,
) -> Dict:
    """Build Q-space sub-blocks via direct sigma-vector projection (Scheme B).

    For each needed Q_n block and Q_m↔Q_n pair, expand Schmidt basis states
    to CI determinant matrices, compute sigma-vectors, and project.

    Selection rules are used to skip zero blocks (|m-n| > 2).

    Args:
        schmidt_data: Schmidt decomposition data.
        partition:    Occ/virt partition data.
        part_info:    Output of partition_schmidt_basis().
        q_partition:  Output of partition_qspace_by_n().
        p_blocks:     P-space n-values.
        backend:      KDCIBackend for sigma_full.
        n_occ, n_act: Orbital counts.
        n_workers:    Parallel threads for sigma-vector computation.
        ecore:        Core energy (added to diagonal blocks).
        verbose:      Print progress.

    Returns:
        dict: Same structure as extract_q_blocks_scheme_a().
    """
    from dm_svd_dci.parallel_ops import compute_sigma_vectors_parallel
    from dm_svd_dci.schmidt_partition import build_hpp_direct, build_hpq_direct

    q_blocks = q_partition['q_blocks']
    active_q = q_partition.get('active_q', sorted(q_blocks.keys()))
    needed_q, q_pairs = _get_active_q_pairs(active_q, p_blocks)

    if verbose:
        print(f"  [Scheme B] Q-partition: active={active_q}, needed={needed_q}")
        print(f"  [Scheme B] Q_m↔Q_n pairs: {q_pairs}")

    # ── Build H_PP (reuse existing Scheme B function) ──
    H_PP = build_hpp_direct(
        schmidt_data, partition, part_info,
        backend, n_occ, n_act, n_workers=n_workers, verbose=verbose)
    H_PP = H_PP + ecore * np.eye(H_PP.shape[0])

    # ── Alpha/beta string info ──
    alpha_strs = backend.qspace_index.alpha_strs
    beta_strs = backend.qspace_index.beta_strs
    n_alpha_strs = len(alpha_strs)
    n_beta_strs = len(beta_strs)
    alpha_to_idx = {int(s): i for i, s in enumerate(alpha_strs)}
    beta_to_idx = {int(s): i for i, s in enumerate(beta_strs)}

    # ── Build H_{P Q_n} for needed Q_n (parallel across n) ──
    H_PQ = {}
    p_dim = H_PP.shape[0]
    p_basis = part_info['p_basis']

    def _build_hpq_for_n(n_Q):
        if n_Q not in q_blocks:
            return n_Q, np.zeros((p_dim, 0))
        q_basis_n = q_blocks[n_Q]['basis']
        if len(q_basis_n) == 0:
            return n_Q, np.zeros((p_dim, 0))

        # Expand P basis
        ci_mats_P = _expand_q_basis_to_ci_mats(
            p_basis, schmidt_data, partition,
            alpha_strs, beta_strs, alpha_to_idx, beta_to_idx, n_occ)

        # Expand Q_n basis
        ci_mats_Q = _expand_q_basis_to_ci_mats(
            q_basis_n, schmidt_data, partition,
            alpha_strs, beta_strs, alpha_to_idx, beta_to_idx, n_occ)

        # Sigma on Q_n states
        sigmas_Q = compute_sigma_vectors_parallel(
            backend.sigma_full, ci_mats_Q, n_workers=n_workers, verbose=False)

        # Project: H_PQ[l,k] = ⟨P_l | sigma_Q_k⟩
        M_flat = ci_mats_P[0].size
        C_flat_P = np.empty((M_flat, p_dim))
        S_flat_Q = np.empty((M_flat, len(q_basis_n)))
        for k in range(p_dim):
            C_flat_P[:, k] = ci_mats_P[k].reshape(-1)
        for k in range(len(q_basis_n)):
            S_flat_Q[:, k] = sigmas_Q[k].reshape(-1)
        H_PQ_n = C_flat_P.T @ S_flat_Q

        del ci_mats_P, ci_mats_Q, sigmas_Q, C_flat_P, S_flat_Q
        return n_Q, H_PQ_n

    if verbose:
        print(f"  [Scheme B] Building H_PQ_n for {len(needed_q)} Q blocks...", flush=True)

    # Parallel over Q_n blocks
    with ThreadPoolExecutor(max_workers=min(n_workers, len(needed_q))) as ex:
        futures = {ex.submit(_build_hpq_for_n, n_Q): n_Q for n_Q in needed_q}
        for fut in as_completed(futures):
            n_Q, mat = fut.result()
            H_PQ[n_Q] = mat
            if verbose:
                print(f"    H_PQ_{n_Q}: {mat.shape}")

    # ── Build H_{Q_n Q_n} diagonals and H_{Q_m Q_n} off-diagonals ──
    # For each needed Q_n, build its diagonal block H_{Q_n Q_n}.
    # For each (m,n) pair with |m-n|≤2, build H_{Q_m Q_n}.

    H_QQ_blocks = {}
    H_QQ_diag = {}

    def _build_hqq_diag(n_Q):
        """Build H_{Q_n Q_n} for a single Q_n block."""
        if n_Q not in q_blocks:
            return n_Q, np.zeros((0, 0)), np.zeros(0)
        q_basis_n = q_blocks[n_Q]['basis']
        d_n = len(q_basis_n)
        if d_n == 0:
            return n_Q, np.zeros((0, 0)), np.zeros(0)

        ci_mats_Q = _expand_q_basis_to_ci_mats(
            q_basis_n, schmidt_data, partition,
            alpha_strs, beta_strs, alpha_to_idx, beta_to_idx, n_occ)

        sigmas_Q = compute_sigma_vectors_parallel(
            backend.sigma_full, ci_mats_Q, n_workers=n_workers, verbose=False)

        # H_nn = C^T @ S
        M_flat = ci_mats_Q[0].size
        C_flat = np.empty((M_flat, d_n))
        S_flat = np.empty((M_flat, d_n))
        for k in range(d_n):
            C_flat[:, k] = ci_mats_Q[k].reshape(-1)
            S_flat[:, k] = sigmas_Q[k].reshape(-1)
        H_nn = C_flat.T @ S_flat
        H_nn = 0.5 * (H_nn + H_nn.T)
        H_nn += ecore * np.eye(d_n)

        D_n = np.diag(H_nn).copy()

        del ci_mats_Q, sigmas_Q, C_flat, S_flat
        return n_Q, H_nn, D_n

    def _build_hqq_offdiag(m, n):
        """Build H_{Q_m Q_n} for m < n (hermitian conjugate for n < m)."""
        if m not in q_blocks or n not in q_blocks:
            return (m, n), np.zeros((0, 0))
        q_basis_m = q_blocks[m]['basis']
        q_basis_n = q_blocks[n]['basis']
        d_m, d_n = len(q_basis_m), len(q_basis_n)
        if d_m == 0 or d_n == 0:
            return (m, n), np.zeros((d_m, d_n))

        ci_mats_M = _expand_q_basis_to_ci_mats(
            q_basis_m, schmidt_data, partition,
            alpha_strs, beta_strs, alpha_to_idx, beta_to_idx, n_occ)
        ci_mats_N = _expand_q_basis_to_ci_mats(
            q_basis_n, schmidt_data, partition,
            alpha_strs, beta_strs, alpha_to_idx, beta_to_idx, n_occ)

        # Sigma on N states, project onto M
        sigmas_N = compute_sigma_vectors_parallel(
            backend.sigma_full, ci_mats_N, n_workers=n_workers, verbose=False)

        M_flat = ci_mats_M[0].size
        C_flat_M = np.empty((M_flat, d_m))
        S_flat_N = np.empty((M_flat, d_n))
        for k in range(d_m):
            C_flat_M[:, k] = ci_mats_M[k].reshape(-1)
        for k in range(d_n):
            S_flat_N[:, k] = sigmas_N[k].reshape(-1)
        H_mn = C_flat_M.T @ S_flat_N

        del ci_mats_M, ci_mats_N, sigmas_N, C_flat_M, S_flat_N
        return (m, n), H_mn

    # Parallel build of diagonal blocks
    if verbose:
        print(f"  [Scheme B] Building H_QnQn for {len(needed_q)} blocks...", flush=True)

    with ThreadPoolExecutor(max_workers=min(n_workers, len(needed_q))) as ex:
        futures = {ex.submit(_build_hqq_diag, n_Q): n_Q for n_Q in needed_q}
        for fut in as_completed(futures):
            n_Q, H_nn, D_n = fut.result()
            H_QQ_blocks[(n_Q, n_Q)] = H_nn
            H_QQ_diag[n_Q] = D_n
            if verbose:
                print(f"    H_Q{n_Q}Q{n_Q}: {H_nn.shape}")

    # Parallel build of off-diagonal blocks
    if q_pairs:
        if verbose:
            print(f"  [Scheme B] Building H_QmQn for {len(q_pairs)} off-diagonal pairs...",
                  flush=True)

        with ThreadPoolExecutor(max_workers=min(n_workers, len(q_pairs))) as ex:
            futures = {ex.submit(_build_hqq_offdiag, m, n): (m, n) for m, n in q_pairs}
            for fut in as_completed(futures):
                (m, n), H_mn = fut.result()
                H_QQ_blocks[(m, n)] = H_mn
                H_QQ_blocks[(n, m)] = H_mn.T
                if verbose:
                    print(f"    H_Q{m}Q{n}: {H_mn.shape}")

    # ── Build dims summary ──
    dims = {
        'P': p_dim,
        'Q_by_n': {n_Q: q_blocks[n_Q]['dim'] for n_Q in needed_q if n_Q in q_blocks},
    }

    return {
        'H_PP': H_PP,
        'H_PQ': H_PQ,
        'H_QQ_blocks': H_QQ_blocks,
        'H_QQ_diag': H_QQ_diag,
        'dims': dims,
        'needed_q': needed_q,
        'q_pairs': q_pairs,
        'active_q': active_q,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════

def test_partition_qspace_by_n():
    """Test Q-space subdivision on toy data."""
    # Create toy schmidt_data: n=0..4
    schmidt = {}
    for n_val in range(5):
        schmidt[n_val] = {'r': n_val + 1}

    # Build mock part_info with q_basis
    q_basis = []
    for n_val in range(5):
        for alpha in range(n_val + 1):
            for beta in range(n_val + 1):
                q_basis.append({
                    'n': n_val,
                    'alpha': alpha,
                    'beta': beta,
                    'flat_idx': 0,
                    'flat_idx_q': len(q_basis),
                })

    part_info = {
        'q_basis': q_basis,
        'q_indices': np.arange(len(q_basis)),
        'p_basis': [],
        'p_indices': np.array([], dtype=np.int64),
        'p_dim': 0,
        'q_dim': len(q_basis),
        'total_dim': len(q_basis),
        'block_offsets': {},
        'p_blocks': [2, 3],
        'n_blocks': list(range(5)),
    }

    # P = {2, 3}
    qp = partition_qspace_by_n(part_info, schmidt, p_blocks=[2, 3])

    assert qp['n_total'] == 5
    assert set(qp['q_n_list']) == {0, 1, 2, 3, 4}

    # Q_n block sizes: dim_n = r_n² = (n+1)²
    for n_val in range(5):
        assert qp['q_blocks'][n_val]['dim'] == (n_val + 1) ** 2

    # Active Q_n: those with |n - n_P| ≤ 2 for n_P ∈ {2,3}
    # n=0: |0-2|=2 ✓, |0-3|=3 ✗ → ✓
    # n=1: |1-2|=1 ✓ → ✓
    # n=2: in P, still in Q if scheme wants (but partition_qspace doesn't filter)
    # Wait — partition_qspace_by_n does NOT filter; it just labels all Q_n.
    # The filtering is in _get_active_q_pairs.
    assert set(qp['q_n_list']) == {0, 1, 2, 3, 4}  # all present

    # _get_active_q_pairs filters:
    needed, pairs = _get_active_q_pairs(qp['q_n_list'], [2, 3])
    # n=0: |0-2|=2 ✓, |0-3|=3 ✗ → included because ∃ n_P=2 with |0-2|≤2
    # n=1: ✓
    # n=4: |4-2|=2 ✓, |4-3|=1 ✓ → included
    # n=2,3: excluded (in P)
    assert set(needed) == {0, 1, 4}
    print("  ✓ partition_qspace_by_n: toy test passed")


def test_extract_q_blocks_scheme_a():
    """Test block extraction from a toy H^emb."""
    # Build a small H^emb:
    # n=0: r=1 → 1 state  (Q0)
    # n=1: r=2 → 4 states (Q1)
    # n=2: r=1 → 1 state  (P)
    # Total D = 6
    # P = {2}
    schmidt = {
        0: {'r': 1},
        1: {'r': 2},
        2: {'r': 1},
    }

    # Mock part_info
    # Layout: n=0 (1 state), n=1 (4 states), n=2 (1 state)
    p_basis = [{'n': 2, 'alpha': 0, 'beta': 0, 'flat_idx': 5, 'flat_idx_p': 0}]
    q_basis = [
        {'n': 0, 'alpha': 0, 'beta': 0, 'flat_idx': 0, 'flat_idx_q': 0},
        {'n': 1, 'alpha': 0, 'beta': 0, 'flat_idx': 1, 'flat_idx_q': 1},
        {'n': 1, 'alpha': 0, 'beta': 1, 'flat_idx': 2, 'flat_idx_q': 2},
        {'n': 1, 'alpha': 1, 'beta': 0, 'flat_idx': 3, 'flat_idx_q': 3},
        {'n': 1, 'alpha': 1, 'beta': 1, 'flat_idx': 4, 'flat_idx_q': 4},
    ]

    part_info = {
        'p_basis': p_basis,
        'q_basis': q_basis,
        'p_indices': np.array([5], dtype=np.int64),
        'q_indices': np.array([0, 1, 2, 3, 4], dtype=np.int64),
        'p_dim': 1,
        'q_dim': 5,
        'total_dim': 6,
        'block_offsets': {0: (0, 1), 1: (1, 2), 2: (5, 1)},
        'p_blocks': [2],
        'n_blocks': [0, 1, 2],
    }

    q_partition = partition_qspace_by_n(part_info, schmidt, p_blocks=[2])

    # Build a toy H^emb with known values
    D = 6
    H_emb = np.arange(D * D, dtype=float).reshape(D, D)
    H_emb = 0.5 * (H_emb + H_emb.T)

    blocks = extract_q_blocks_scheme_a(H_emb, part_info, q_partition, p_blocks=[2], verbose=False)

    # P = {2}, needed Q = {0, 1} (both within distance 2)
    assert 'H_PP' in blocks
    assert blocks['H_PP'].shape == (1, 1)

    # Q0 has 1 state, Q1 has 4 states
    assert blocks['H_PQ'][0].shape == (1, 1)
    assert blocks['H_PQ'][1].shape == (1, 4)

    # H_Q0Q0: (1,1), H_Q1Q1: (4,4), H_Q0Q1: (1,4)
    assert blocks['H_QQ_blocks'][(0, 0)].shape == (1, 1)
    assert blocks['H_QQ_blocks'][(1, 1)].shape == (4, 4)
    if (0, 1) in blocks['H_QQ_blocks']:
        assert blocks['H_QQ_blocks'][(0, 1)].shape == (1, 4)

    # Diagonals
    assert len(blocks['H_QQ_diag'][0]) == 1
    assert len(blocks['H_QQ_diag'][1]) == 4

    print("  ✓ extract_q_blocks_scheme_a: toy test passed")


if __name__ == "__main__":
    test_partition_qspace_by_n()
    test_extract_q_blocks_scheme_a()
    print("All qspace_partition tests passed.")