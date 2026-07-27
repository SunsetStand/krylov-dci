#!/usr/bin/env python3
"""
Partition Schmidt product basis into P and Q spaces by electron count n.

Given the Schmidt decomposition data (from dm_svd_embedding.density_matrix),
partition the product basis states {|Ã_α^(n)⟩ ⊗ |B̃_β^(n)⟩} into:
  - P-space: blocks with n ∈ p_blocks (e.g. n=8,9,10)
  - Q-space: blocks with n ∉ p_blocks

Returns index mappings, dimension info, and sub-block extraction utilities.
"""

import numpy as np
import time as time_mod
from typing import Dict, List, Tuple, Optional


def partition_schmidt_basis(
    schmidt_data: Dict[int, Dict],
    p_blocks: List[int],
) -> Dict:
    """Partition Schmidt product basis into P and Q spaces.

    The Schmidt product basis has states |Ã_α^(n)⟩ ⊗ |B̃_β^(n)⟩ for each
    electron-number block n. Total dimension D = Σ_n r_n².

    Args:
        schmidt_data: Output of compute_schmidt_decomposition().
            Dict[n] → {'r': int, 'U': ndarray, 'V': ndarray, ...}
        p_blocks: List of n values to include in P-space.
            Example: [8, 9, 10] for N₂-like systems.

    Returns:
        dict with keys:
          'p_basis':     List[dict] — each P basis state {n, alpha, beta, flat_idx_p}
          'q_basis':     List[dict] — each Q basis state {n, alpha, beta, flat_idx_q}
          'p_dim':       int — |P| = number of Schmidt product states in P
          'q_dim':       int — |Q| = number of Schmidt product states in Q
          'total_dim':   int — D = Σ_n r_n²
          'p_indices':   (|P|,) int64 — flat indices in full H^emb
          'q_indices':   (|Q|,) int64 — flat indices in full H^emb
          'block_offsets': Dict[n] → (offset, r_n) — starting flat index and rank
          'p_blocks':    List[int] — the p_blocks input (for reference)
          'n_blocks':    List[int] — sorted list of all n blocks present
    """
    # ── Build global index map ──
    n_sorted = sorted(schmidt_data.keys())
    block_offsets = {}
    offset = 0
    for n_A in n_sorted:
        r = schmidt_data[n_A]['r']
        block_offsets[n_A] = (offset, r)
        offset += r * r

    total_dim = offset  # D = Σ_n r_n²

    if total_dim == 0:
        return {
            'p_basis': [], 'q_basis': [],
            'p_dim': 0, 'q_dim': 0, 'total_dim': 0,
            'p_indices': np.array([], dtype=np.int64),
            'q_indices': np.array([], dtype=np.int64),
            'block_offsets': block_offsets,
            'p_blocks': list(p_blocks),
            'n_blocks': n_sorted,
        }

    p_set = set(p_blocks)

    # ── Build basis lists ──
    p_basis = []
    q_basis = []
    p_indices_list = []
    q_indices_list = []

    for n_A in n_sorted:
        r = schmidt_data[n_A]['r']
        if r == 0:
            continue
        offset_n, _ = block_offsets[n_A]
        in_p = (n_A in p_set)

        for alpha in range(r):
            for beta in range(r):
                flat_idx = offset_n + alpha * r + beta
                info = {'n': n_A, 'alpha': alpha, 'beta': beta, 'flat_idx': flat_idx}
                if in_p:
                    info['flat_idx_p'] = len(p_basis)
                    p_basis.append(info)
                    p_indices_list.append(flat_idx)
                else:
                    info['flat_idx_q'] = len(q_basis)
                    q_basis.append(info)
                    q_indices_list.append(flat_idx)

    return {
        'p_basis': p_basis,
        'q_basis': q_basis,
        'p_dim': len(p_basis),
        'q_dim': len(q_basis),
        'total_dim': total_dim,
        'p_indices': np.array(p_indices_list, dtype=np.int64),
        'q_indices': np.array(q_indices_list, dtype=np.int64),
        'block_offsets': block_offsets,
        'p_blocks': list(p_blocks),
        'n_blocks': n_sorted,
    }


def extract_subblocks(
    H_emb: np.ndarray,
    part: Dict,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract H_PP, H_PQ, H_QQ from the full H^emb using partition indices.

    This is the "方案 A" approach: build full H^emb first, then slice.

    Args:
        H_emb: Full embedded Hamiltonian, shape (D, D).
        part:   Output of partition_schmidt_basis().

    Returns:
        (H_PP, H_PQ, H_QQ):
          H_PP: (|P|, |P|) hermitian
          H_PQ: (|P|, |Q|)
          H_QQ: (|Q|, |Q|) hermitian
    """
    p_idx = part['p_indices']
    q_idx = part['q_indices']

    if len(p_idx) == 0 or len(q_idx) == 0:
        p_dim = len(p_idx)
        q_dim = len(q_idx)
        H_PP = H_emb[np.ix_(p_idx, p_idx)] if p_dim > 0 else np.zeros((0, 0))
        H_PQ = np.zeros((p_dim, q_dim))
        H_QQ = H_emb[np.ix_(q_idx, q_idx)] if q_dim > 0 else np.zeros((0, 0))
        return H_PP, H_PQ, H_QQ

    H_PP = H_emb[np.ix_(p_idx, p_idx)]
    H_PQ = H_emb[np.ix_(p_idx, q_idx)]
    H_QQ = H_emb[np.ix_(q_idx, q_idx)]

    # Enforce hermiticity (numerical noise from sigma-vector projection)
    H_PP = 0.5 * (H_PP + H_PP.T)
    H_QQ = 0.5 * (H_QQ + H_QQ.T)

    return H_PP, H_PQ, H_QQ


# TODO(方案B): Build H_PP, H_PQ, H_QQ directly without constructing full H^emb.
# For large D (e.g. > 100,000), storing the full D×D matrix is prohibitive.
#
# Approach:
#   H_PP:  Only project sigma-vectors of P-basis states onto P-basis states.
#          Requires |P| sigma-vector calls.
#   H_PQ:  Project sigma-vectors of Q-basis states onto P-basis states, or
#          equivalently, project sigma-vectors of P-basis states onto Q-basis
#          states and transpose. Requires max(|P|, |Q|) sigma-vector calls.
#   H_QQ:  Only needed for Krylov propagation via H_QQ @ B product, which can
#          be done on-the-fly: H_QQ @ v = (H^emb @ v_full)[q_indices] where
#          v_full[q_indices] = v and v_full[p_indices] = 0.
#          This avoids storing the |Q|×|Q| matrix.
#
# The current 方案A is adequate when D ≲ 20,000 (N₂ CAS(10,10) with r_total
# ~ 100-230 gives D = Σ r_n² ~ 4,700-15,200). At this scale, storing a
# 15k×15k float64 matrix takes ~1.8 GB, which is manageable on a single node.
#
# IMPLEMENTATION: Scheme B functions below.


# ═══════════════════════════════════════════════════════════════════════════
# 方案 B: Direct construction of H_PP, H_PQ without full H^emb
# ═══════════════════════════════════════════════════════════════════════════

def _expand_basis_to_ci_mats(
    basis_list: List[Dict],
    schmidt_data: Dict[int, Dict],
    partition: Dict[int, Dict],
    n_alpha_strs: int,
    n_beta_strs: int,
    n_occ: int,
    alpha_strs: np.ndarray,
    beta_strs: np.ndarray,
    alpha_to_idx: Dict[int, int],
    beta_to_idx: Dict[int, int],
) -> List[np.ndarray]:
    """Expand a list of Schmidt basis states to full CAS CI matrices.

    Helper used by both build_hpp_direct and build_hpq_direct.
    """
    from dm_svd_embedding.embedded_hamiltonian import _expand_schmidt_product_to_ci_matrix

    ci_mats = []
    for info in basis_list:
        blk_schmidt = schmidt_data[info['n']]
        blk_partition = partition[info['n']]
        ci_mat = _expand_schmidt_product_to_ci_matrix(
            info['alpha'], info['beta'],
            blk_schmidt, blk_partition,
            n_alpha_strs, n_beta_strs, n_occ,
            alpha_strs, beta_strs,
            alpha_to_idx, beta_to_idx)
        ci_mats.append(ci_mat)
    return ci_mats


def build_hpp_direct(
    schmidt_data: Dict[int, Dict],
    partition: Dict[int, Dict],
    part_info: Dict,
    backend,
    n_occ: int,
    n_act: int,
    n_workers: int = 1,
    verbose: bool = True,
) -> np.ndarray:
    """Build H_PP directly: sigma-vector project P-basis onto P-basis.

    Computes sigma = H·v for each P-basis Schmidt state v, then projects
    onto all other P-basis states: H_PP[l,k] = v_l · sigma_k.

    Only |P| sigma-vector calls needed.

    Args:
        schmidt_data: Schmidt decomposition data.
        partition: Occ/virt partition data.
        part_info: Output of partition_schmidt_basis().
        backend: KDCIBackend for sigma_full.
        n_occ: Number of A-space orbitals.
        n_act: Total active orbitals.
        n_workers: Parallel threads.
        verbose: Print progress.

    Returns:
        H_PP: (|P|, |P|) hermitian matrix.
    """
    from dm_svd_dci.parallel_ops import compute_sigma_vectors_parallel

    p_dim = part_info['p_dim']
    if p_dim == 0:
        return np.zeros((0, 0))

    alpha_strs = backend.qspace_index.alpha_strs
    beta_strs = backend.qspace_index.beta_strs
    n_alpha_strs = len(alpha_strs)
    n_beta_strs = len(beta_strs)
    alpha_to_idx = {int(s): i for i, s in enumerate(alpha_strs)}
    beta_to_idx = {int(s): i for i, s in enumerate(beta_strs)}

    p_basis = part_info['p_basis']

    if verbose:
        print(f"  [Scheme B] Expanding {p_dim} P-basis states to CI mats...",
              flush=True)

    ci_mats_P = _expand_basis_to_ci_mats(
        p_basis, schmidt_data, partition,
        n_alpha_strs, n_beta_strs, n_occ,
        alpha_strs, beta_strs, alpha_to_idx, beta_to_idx)

    if verbose:
        print(f"  [Scheme B] Computing {p_dim} sigma-vectors for H_PP...",
              flush=True)

    sigmas_P = compute_sigma_vectors_parallel(
        backend.sigma_full, ci_mats_P, n_workers=n_workers, verbose=verbose)

    # BLAS3 projection: H_PP = C^T @ S
    M_flat = ci_mats_P[0].size
    C_flat = np.empty((M_flat, p_dim))
    S_flat = np.empty((M_flat, p_dim))
    for k in range(p_dim):
        C_flat[:, k] = ci_mats_P[k].reshape(-1)
        S_flat[:, k] = sigmas_P[k].reshape(-1)
    H_PP = C_flat.T @ S_flat
    H_PP = 0.5 * (H_PP + H_PP.T)

    del ci_mats_P, sigmas_P, C_flat, S_flat

    if verbose:
        print(f"  [Scheme B] H_PP ({p_dim}×{p_dim}) built.", flush=True)

    return H_PP


def build_hpq_direct(
    schmidt_data: Dict[int, Dict],
    partition: Dict[int, Dict],
    part_info: Dict,
    backend,
    n_occ: int,
    n_act: int,
    n_workers: int = 1,
    verbose: bool = True,
) -> np.ndarray:
    """Build H_PQ directly: sigma-vector project Q-basis onto P-basis.

    Computes sigma = H·v for each Q-basis Schmidt state, then projects
    onto all P-basis states: H_PQ[l,k] = v_l^P · sigma_k^Q.

    Requires min(|P|, |Q|) sigma calls. Uses the smaller side for sigma,
    then projects to the other.

    Args:
        (same as build_hpp_direct_v2)
        part_info: Must contain both p_basis and q_basis.

    Returns:
        H_PQ: (|P|, |Q|) matrix.
    """
    from dm_svd_dci.parallel_ops import compute_sigma_vectors_parallel

    p_dim = part_info['p_dim']
    q_dim = part_info['q_dim']
    if p_dim == 0 or q_dim == 0:
        return np.zeros((p_dim, q_dim))

    alpha_strs = backend.qspace_index.alpha_strs
    beta_strs = backend.qspace_index.beta_strs
    n_alpha_strs = len(alpha_strs)
    n_beta_strs = len(beta_strs)
    alpha_to_idx = {int(s): i for i, s in enumerate(alpha_strs)}
    beta_to_idx = {int(s): i for i, s in enumerate(beta_strs)}

    p_basis = part_info['p_basis']
    q_basis = part_info['q_basis']

    # Strategy: sigma on smaller side, project to larger side
    if q_dim <= p_dim:
        # Compute sigma for Q states, project onto P
        if verbose:
            print(f"  [Scheme B] Expanding {q_dim} Q-basis states for H_PQ...",
                  flush=True)

        ci_mats_Q = _expand_basis_to_ci_mats(
            q_basis, schmidt_data, partition,
            n_alpha_strs, n_beta_strs, n_occ,
            alpha_strs, beta_strs, alpha_to_idx, beta_to_idx)

        if verbose:
            print(f"  [Scheme B] Computing {q_dim} sigma-vectors for H_PQ...",
                  flush=True)
        sigmas_Q = compute_sigma_vectors_parallel(
            backend.sigma_full, ci_mats_Q, n_workers=n_workers, verbose=verbose)

        # Also expand P states for projection
        ci_mats_P = _expand_basis_to_ci_mats(
            p_basis, schmidt_data, partition,
            n_alpha_strs, n_beta_strs, n_occ,
            alpha_strs, beta_strs, alpha_to_idx, beta_to_idx)

        # H_PQ[l,k] = v_l^P · sigma_k^Q → C_P^T @ S_Q
        M_flat = ci_mats_P[0].size
        C_flat_P = np.empty((M_flat, p_dim))
        S_flat_Q = np.empty((M_flat, q_dim))
        for k in range(p_dim):
            C_flat_P[:, k] = ci_mats_P[k].reshape(-1)
        for k in range(q_dim):
            S_flat_Q[:, k] = sigmas_Q[k].reshape(-1)
        H_PQ = C_flat_P.T @ S_flat_Q

        del ci_mats_P, ci_mats_Q, sigmas_Q, C_flat_P, S_flat_Q
    else:
        # Compute sigma for P states, project onto Q → transpose
        if verbose:
            print(f"  [Scheme B] Expanding {p_dim} P-basis states for H_PQ...",
                  flush=True)

        ci_mats_P = _expand_basis_to_ci_mats(
            p_basis, schmidt_data, partition,
            n_alpha_strs, n_beta_strs, n_occ,
            alpha_strs, beta_strs, alpha_to_idx, beta_to_idx)

        if verbose:
            print(f"  [Scheme B] Computing {p_dim} sigma-vectors for H_PQ...",
                  flush=True)
        sigmas_P = compute_sigma_vectors_parallel(
            backend.sigma_full, ci_mats_P, n_workers=n_workers, verbose=verbose)

        # Expand Q states for projection
        ci_mats_Q = _expand_basis_to_ci_mats(
            q_basis, schmidt_data, partition,
            n_alpha_strs, n_beta_strs, n_occ,
            alpha_strs, beta_strs, alpha_to_idx, beta_to_idx)

        # H_QP[l,k] = v_l^Q · sigma_k^P → after transpose: H_PQ = H_QP^T
        M_flat = ci_mats_Q[0].size
        C_flat_Q = np.empty((M_flat, q_dim))
        S_flat_P = np.empty((M_flat, p_dim))
        for k in range(q_dim):
            C_flat_Q[:, k] = ci_mats_Q[k].reshape(-1)
        for k in range(p_dim):
            S_flat_P[:, k] = sigmas_P[k].reshape(-1)
        H_QP = C_flat_Q.T @ S_flat_P  # (|Q|, |P|)
        H_PQ = H_QP.T  # (|P|, |Q|)

        del ci_mats_P, ci_mats_Q, sigmas_P, C_flat_Q, S_flat_P

    if verbose:
        print(f"  [Scheme B] H_PQ ({p_dim}×{q_dim}) built.", flush=True)

    return H_PQ


def apply_hqq_on_the_fly(
    v: np.ndarray,
    schmidt_data: Dict[int, Dict],
    partition: Dict[int, Dict],
    part_info: Dict,
    backend,
    n_occ: int,
    n_act: int,
    n_workers: int = 1,
    verbose: bool = False,
) -> np.ndarray:
    """Compute H_QQ @ v on-the-fly without storing H_QQ (Matrix-Free).

    For a vector v ∈ R^{|Q|}, this:
      1. Expands v to full Schmidt basis: v_full[p_indices]=0, v_full[q_indices]=v
      2. Expands v_full to CI determinant basis → one CI matrix
      3. Computes sigma = H·CI_matrix via contract_2e
      4. Projects sigma back to Q-space: result[q] = ⟨q| sigma ⟩
      5. Returns result ∈ R^{|Q|}

    This uses exactly 1 sigma-vector call per evaluation of H_QQ@v.

    Args:
        v: (|Q|,) vector in Q-space.
        schmidt_data, partition, part_info: Standard data structures.
        backend: KDCIBackend.
        n_occ, n_act: Orbital counts.
        n_workers: Unused (single sigma call).
        verbose: Print diagnostics.

    Returns:
        result: (|Q|,) = H_QQ @ v
    """
    import time as time_mod
    q_dim = part_info['q_dim']
    p_dim = part_info['p_dim']
    D = p_dim + q_dim

    if q_dim == 0:
        return np.zeros(0)

    # ── Step 1: Build full CI matrix as linear combination ──
    alpha_strs = backend.qspace_index.alpha_strs
    beta_strs = backend.qspace_index.beta_strs
    n_alpha_strs = len(alpha_strs)
    n_beta_strs = len(beta_strs)
    alpha_to_idx = {int(s): i for i, s in enumerate(alpha_strs)}
    beta_to_idx = {int(s): i for i, s in enumerate(beta_strs)}

    q_basis = part_info['q_basis']

    # Build linear combination: CI_mat = Σ_q v[q] * CI_mat_q
    ci_combined = np.zeros((n_alpha_strs, n_beta_strs))

    for q_idx_flat, elem_info in enumerate(q_basis):
        v_val = v[q_idx_flat]
        if abs(v_val) < 1e-14:
            continue

        n_A_val = elem_info['n']
        alpha = elem_info['alpha']
        beta = elem_info['beta']
        blk_schmidt = schmidt_data[n_A_val]
        blk_partition = partition[n_A_val]

        from dm_svd_embedding.embedded_hamiltonian import _expand_schmidt_product_to_ci_matrix
        ci_mat_q = _expand_schmidt_product_to_ci_matrix(
            alpha, beta, blk_schmidt, blk_partition,
            n_alpha_strs, n_beta_strs, n_occ,
            alpha_strs, beta_strs, alpha_to_idx, beta_to_idx)

        ci_combined += v_val * ci_mat_q

    # ── Step 2: Compute sigma = H @ ci_combined ──
    sigma_mat = backend.sigma_full(ci_combined)
    sigma_flat = sigma_mat.reshape(-1)

    # ── Step 3: Project sigma back to Q-space ──
    # result[q] = ⟨q| sigma⟩ = Σ_{I,J} CI_mat_q[I,J] · sigma[I,J]
    result = np.zeros(q_dim)

    # For efficiency, pre-compute all Q-basis CI mat flat vectors
    # But this is heavy. Instead, we batch-expand Q-basis and dot.
    # Since this is on-the-fly, we accept the cost.
    # Re-expand each Q-basis state to compute dot product.
    for q_idx_flat, elem_info in enumerate(q_basis):
        n_A_val = elem_info['n']
        alpha = elem_info['alpha']
        beta = elem_info['beta']
        blk_schmidt = schmidt_data[n_A_val]
        blk_partition = partition[n_A_val]

        from dm_svd_embedding.embedded_hamiltonian import _expand_schmidt_product_to_ci_matrix
        ci_mat_q = _expand_schmidt_product_to_ci_matrix(
            alpha, beta, blk_schmidt, blk_partition,
            n_alpha_strs, n_beta_strs, n_occ,
            alpha_strs, beta_strs, alpha_to_idx, beta_to_idx)

        result[q_idx_flat] = np.dot(ci_mat_q.reshape(-1), sigma_flat)

    return result


def apply_hqq_batch(
    B: np.ndarray,
    schmidt_data: Dict[int, Dict],
    partition: Dict[int, Dict],
    part_info: Dict,
    backend,
    n_occ: int,
    n_act: int,
    n_workers: int = 1,
    verbose: bool = False,
) -> np.ndarray:
    """Compute H_QQ @ B for a batch of vectors (columns of B).

    This is a batched version of apply_hqq_on_the_fly that computes
    sigma for each column and projects back to Q-space.

    Args:
        B: (|Q|, r) matrix, each column is a vector in Q-space.
        (other args same as apply_hqq_on_the_fly)

    Returns:
        HQQ_B: (|Q|, r) = H_QQ @ B
    """
    q_dim, r = B.shape
    if q_dim == 0 or r == 0:
        return np.zeros((q_dim, r))

    result = np.zeros((q_dim, r))
    for j in range(r):
        result[:, j] = apply_hqq_on_the_fly(
            B[:, j], schmidt_data, partition, part_info,
            backend, n_occ, n_act, n_workers=n_workers,
            verbose=(verbose and j == 0))
        if verbose and (j + 1) % max(1, r // 5) == 0:
            print(f"    H_QQ@{j+1}/{r} done", flush=True)

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════

def test_partition_schmidt_basis():
    """Test partition logic on a toy Schmidt data dict."""
    # Toy: 3 blocks with r=[1, 2, 3]
    # n=0: r=1 → 1 state
    # n=1: r=2 → 4 states
    # n=2: r=3 → 9 states
    # Total D = 14
    schmidt = {
        0: {'r': 1, 'U': np.eye(1), 'V': np.eye(1), 'sigma': np.array([1.0])},
        1: {'r': 2, 'U': np.eye(2), 'V': np.eye(2), 'sigma': np.array([0.5, 0.3])},
        2: {'r': 3, 'U': np.eye(3), 'V': np.eye(3), 'sigma': np.array([0.2, 0.1, 0.05])},
    }

    # P = {1, 2} → |P| = 4 + 9 = 13, |Q| = 1
    part = partition_schmidt_basis(schmidt, p_blocks=[1, 2])
    assert part['total_dim'] == 14
    assert part['p_dim'] == 13
    assert part['q_dim'] == 1
    assert len(part['p_basis']) == 13
    assert len(part['q_basis']) == 1
    assert part['q_basis'][0]['n'] == 0
    assert part['q_basis'][0]['flat_idx_q'] == 0

    # Block offsets
    assert part['block_offsets'][0] == (0, 1)
    assert part['block_offsets'][1] == (1, 2)
    assert part['block_offsets'][2] == (5, 3)

    # Indices arrays
    assert len(part['p_indices']) == 13
    assert len(part['q_indices']) == 1
    # n=0 is Q, its only state is at flat_idx = 0
    assert part['q_indices'][0] == 0
    # n=1 (P) starts at offset=1, r=2 → indices 1..4
    # n=2 (P) starts at offset=5, r=3 → indices 5..13
    assert set(part['p_indices']) == set(range(1, 14))

    print("  ✓ partition_schmidt_basis: toy test passed")


def test_extract_subblocks():
    """Test slicing H_emb with partition indices."""
    # Build a toy 5×5 H_emb with known pattern:
    # D=5: n=0 (r=2→4 states), n=1 (r=1→1 state)
    # P = {0} → indices 0,1,2,3; Q = {1} → index 4
    D = 5
    H_emb = np.arange(D * D, dtype=float).reshape(D, D)
    H_emb = 0.5 * (H_emb + H_emb.T)  # make symmetric-ish
    # Mark diagonals for easier checking
    for i in range(D):
        H_emb[i, i] = float(i * 100)

    schmidt = {
        0: {'r': 2, 'U': np.eye(2), 'V': np.eye(2), 'sigma': np.array([1.0, 0.5])},
        1: {'r': 1, 'U': np.eye(1), 'V': np.eye(1), 'sigma': np.array([0.1])},
    }
    part = partition_schmidt_basis(schmidt, p_blocks=[0])  # n=0 in P

    H_PP, H_PQ, H_QQ = extract_subblocks(H_emb, part)

    # P indices: 0,1,2,3 (n=0, r=2 → 4 states)
    # Q indices: 4 (n=1, r=1 → 1 state)
    assert H_PP.shape == (4, 4)
    assert H_PQ.shape == (4, 1)
    assert H_QQ.shape == (1, 1)

    # Check H_PP is correct slice
    assert np.allclose(H_PP[0, 0], 0.0)   # H_emb[0,0] = 0
    assert np.allclose(H_PP[3, 3], 300.0)  # H_emb[3,3] = 300
    assert H_QQ[0, 0] == 400.0             # H_emb[4,4] = 400

    # H_PQ[0,0] should be H_emb[0,4]
    assert np.allclose(H_PQ[0, 0], H_emb[0, 4])

    print("  ✓ extract_subblocks: slicing test passed")


def test_empty_edge_cases():
    """Test edge cases: empty P, empty Q, zero-rank blocks."""
    # All-zero rank
    schmidt_empty = {0: {'r': 0}}
    part = partition_schmidt_basis(schmidt_empty, p_blocks=[0])
    assert part['total_dim'] == 0
    assert part['p_dim'] == 0
    assert part['q_dim'] == 0

    # All blocks in P → Q empty
    schmidt_all_p = {0: {'r': 2}, 1: {'r': 1}}
    part = partition_schmidt_basis(schmidt_all_p, p_blocks=[0, 1])
    assert part['p_dim'] == 5   # 4 + 1
    assert part['q_dim'] == 0

    # All blocks in Q → P empty
    part = partition_schmidt_basis(schmidt_all_p, p_blocks=[])
    assert part['p_dim'] == 0
    assert part['q_dim'] == 5

    print("  ✓ empty_edge_cases: all passed")


if __name__ == "__main__":
    test_partition_schmidt_basis()
    test_extract_subblocks()
    test_empty_edge_cases()
    print("All schmidt_partition tests passed.")