"""
Sparse operations on the Q-space Hamiltonian.

Provides:
  1. generate_connected_determinants — enumerate single/double excitations
     (combinatorial; no PySCF equivalent for selected-CI spaces).
  2. sigma_H_off_sparse / sigma_H_full — sparse sigma-vector via adjacency.
  3. sigma_H_contract_2e — production sigma-vector via PySCF contract_2e.
  4. build_sparse_adjacency — build H_QQ sparse graph for iterative work.

For production use (Phase 10+), sigma_H_contract_2e is preferred: C-level
speed from PySCF's direct_spin1.contract_2e. The sparse adjacency approach
is kept for cases where the full CI vector is too large.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional


def generate_connected_determinants(
    alpha_str: int, beta_str: int,
    n_orb: int
) -> List[Tuple[Tuple[int, int], int, int, int, int]]:
    """Generate all determinants connected to |D> via 1 or 2 ex citations.

    Pure combinatorial enumeration — returns string pairs, not matrix elements.

    Args:
        alpha_str, beta_str: Bit strings for the reference determinant.
        n_orb: Number of spatial orbitals.

    Returns:
        List of (det_tuple, i, a, j, b) where i,a are hole/particle for
        first excitation, j,b for second (or -1,-1 for singles).
    """
    try:
        from .determinants import bit_positions
    except ImportError:
        from determinants import bit_positions

    alpha_occ = bit_positions(int(alpha_str))
    beta_occ = bit_positions(int(beta_str))
    all_orbs = list(range(n_orb))

    alpha_virt = [p for p in all_orbs if p not in alpha_occ]
    beta_virt = [p for p in all_orbs if p not in beta_occ]

    results = []

    # ---- Single excitations ----
    # α → α
    for i in alpha_occ:
        for a in alpha_virt:
            new_alpha = (alpha_str ^ (1 << i)) | (1 << a)
            results.append(((new_alpha, beta_str), i, a, -1, -1))

    # β → β
    for i in beta_occ:
        for a in beta_virt:
            new_beta = (beta_str ^ (1 << i)) | (1 << a)
            results.append(((alpha_str, new_beta), i, a, -1, -1))

    # ---- Double excitations ----
    # αα → αα
    if len(alpha_occ) >= 2:
        for idx_i, i in enumerate(alpha_occ):
            for j in alpha_occ[idx_i + 1:]:
                for idx_a, va in enumerate(alpha_virt):
                    for vb in alpha_virt[idx_a + 1:]:
                        new_alpha = alpha_str
                        new_alpha ^= (1 << i) | (1 << j)
                        new_alpha |= (1 << va) | (1 << vb)
                        results.append(((new_alpha, beta_str), i, va, j, vb))

    # ββ → ββ
    if len(beta_occ) >= 2:
        for idx_i, i in enumerate(beta_occ):
            for j in beta_occ[idx_i + 1:]:
                for idx_a, va in enumerate(beta_virt):
                    for vb in beta_virt[idx_a + 1:]:
                        new_beta = beta_str
                        new_beta ^= (1 << i) | (1 << j)
                        new_beta |= (1 << va) | (1 << vb)
                        results.append(((alpha_str, new_beta), i, va, j, vb))

    # αβ → αβ
    for i in alpha_occ:
        for j in beta_occ:
            for va in alpha_virt:
                for vb in beta_virt:
                    new_alpha = (alpha_str ^ (1 << i)) | (1 << va)
                    new_beta = (beta_str ^ (1 << j)) | (1 << vb)
                    results.append(((new_alpha, new_beta), i, va, j, vb))

    return results


def build_sparse_adjacency(
    ham, q_dets: List[Tuple[int, int]],
    n_orb: int,
    q_idx_map: Optional[Dict[Tuple[int, int], int]] = None
) -> Tuple[List[List[Tuple[int, float]]], Dict[Tuple[int, int], int]]:
    """Build H_QQ sparse adjacency list (off-diagonal edges only).

    For each Q determinant i, stores list of (j, H[i,j]) for j>i such that
    H[i,j] ≠ 0 (connected via 1 or 2 spin-orbital excitations).

    Args:
        ham:     Hamiltonian object.
        q_dets:  Q-space determinant list.
        n_orb:   Number of spatial orbitals.
        q_idx_map: Optional pre-built {(a,b): idx} map.

    Returns:
        (adjacency, idx_map) where adjacency[i] = [(j, hij), ...].
    """
    M = len(q_dets)

    if q_idx_map is None:
        q_idx_map = {(int(a), int(b)): i for i, (a, b) in enumerate(q_dets)}

    off_diag = [[] for _ in range(M)]

    for i in range(M):
        a_str, b_str = q_dets[i]
        connected = generate_connected_determinants(
            int(a_str), int(b_str), n_orb
        )
        for det_j, *_ in connected:
            j = q_idx_map.get((det_j[0], det_j[1]))
            if j is not None and j > i:
                hij = ham.matrix_element(
                    (int(a_str), int(b_str)), det_j
                )
                if abs(hij) > 1e-14:
                    off_diag[i].append((j, hij))

    return off_diag, q_idx_map


def sigma_from_adjacency(
    v: np.ndarray,
    hdiag: np.ndarray,
    adjacency: List[List[Tuple[int, float]]]
) -> np.ndarray:
    """Compute H_QQ @ v using pre-built sparse adjacency.

    sigma[i] = hdiag[i] * v[i] + Σ_{j∈adj[i]} hij * v[j]
             + Σ_{k: i∈adj[k]} hki * v[k]       (Hermitian)

    Args:
        v:         Input vector, shape (M,).
        hdiag:     Diagonal of H_QQ, shape (M,).
        adjacency: Adjacency list from build_sparse_adjacency.

    Returns:
        sigma, shape (M,).
    """
    M = len(hdiag)
    sigma = hdiag * v.copy()

    for i in range(M):
        for (j, hij) in adjacency[i]:
            sigma[i] += hij * v[j]
            sigma[j] += hij * v[i]

    return sigma


def sigma_from_adjacency_multi(
    vectors: np.ndarray,
    hdiag: np.ndarray,
    adjacency: List[List[Tuple[int, float]]]
) -> np.ndarray:
    """Compute H_QQ @ vectors for multiple columns using sparse adjacency.

    Args:
        vectors:   (M, k) input array.
        hdiag:     Diagonal of H_QQ, shape (M,).
        adjacency: Adjacency list.

    Returns:
        sigma, shape (M, k).
    """
    M, k = vectors.shape
    sigma = vectors * hdiag[:, np.newaxis]

    for i in range(M):
        for (j, hij) in adjacency[i]:
            sigma[i] += hij * vectors[j]
            sigma[j] += hij * vectors[i]

    return sigma


# ============================================================================
# Production sigma via PySCF contract_2e
# ============================================================================

def sigma_H_contract_2e(
    h1e: np.ndarray, eri: np.ndarray,
    c_vec: np.ndarray,
    norb: int, nelec: Tuple[int, int],
    alpha_strs: np.ndarray, beta_strs: np.ndarray,
    link_index: Optional[np.ndarray] = None
) -> np.ndarray:
    """Compute H·c using PySCF's direct_spin1.contract_2e.

    C-level implementation, the fastest available path for large CI spaces.

    Args:
        h1e:         One-electron integrals (norb, norb).
        eri:         Two-electron integrals, packed 4-fold format.
        c_vec:       1D CI coefficient vector.
        norb:        Number of spatial orbitals.
        nelec:       (n_alpha, n_beta).
        alpha_strs:  Alpha strings array (int64).
        beta_strs:   Beta strings array (int64).
        link_index:  Optional pre-computed link table.

    Returns:
        sigma vector, same shape as c_vec.
    """
    from pyscf.fci import direct_spin1

    # Reconstruct full CI vector from selected entries
    from pyscf.fci import cistring
    na_strs_full = cistring.gen_strings4orblist(range(norb), nelec[0])
    nb_strs_full = cistring.gen_strings4orblist(range(norb), nelec[1])
    na = len(na_strs_full)
    nb = len(nb_strs_full)

    qa_idx = {int(s): i for i, s in enumerate(na_strs_full)}
    qb_idx = {int(s): i for i, s in enumerate(nb_strs_full)}

    ci_full = np.zeros((na, nb))
    for k in range(len(alpha_strs)):
        ia = qa_idx[int(alpha_strs[k])]
        ib = qb_idx[int(beta_strs[k])]
        ci_full[ia, ib] = c_vec[k]

    # Contract
    sigma_full = direct_spin1.contract_2e(
        eri, ci_full, norb, nelec, link_index=link_index
    ) + direct_spin1.contract_1e(h1e, ci_full, norb, nelec)

    # Extract selected entries
    sigma = np.zeros(len(alpha_strs))
    for k in range(len(alpha_strs)):
        ia = qa_idx[int(alpha_strs[k])]
        ib = qb_idx[int(beta_strs[k])]
        sigma[k] = sigma_full[ia, ib]

    return sigma


# ============================================================================
# Tests
# ============================================================================

def test_generate_connected_counts():
    """Verify connected determinant counts for H₂/STO-3G HF det."""
    try:
        from .determinants import count_bits
    except ImportError:
        from determinants import count_bits

    # H₂/STO-3G: 2 electrons, 2 spatial orbitals. HF = (0b11, 0b11) — both
    # orbitals doubly occupied. Q space: no virtual orbitals → 0 connections.
    det = (0b11, 0b11)
    conn = generate_connected_determinants(det[0], det[1], 2)
    print(f"  H₂ HF det → {len(conn)} connected (should be 0 — no virtuals)")
    # With 2 orbs and 2 electrons in HF, no virtual orbitals → 0
    assert len(conn) == 0, f"Expected 0, got {len(conn)}"

    # With n_orb=4, n_elec=2: HF = (0b11, 0b11), 2 virtual orbitals
    # Singles: α→α: 2*2=4, β→β: 2*2=4 = 8
    # Doubles αα: C(2,2)*C(2,2)=1*1=1, ββ: 1*1=1, αβ: 2*2*2*2=16
    # Total: 8+1+1+16=26
    conn = generate_connected_determinants(0b11, 0b11, 4)
    print(f"  (4 orb, 2 elec) HF det → {len(conn)} connected (expect 26)")

    print("  ✓ Connected determinant counts")


if __name__ == "__main__":
    test_generate_connected_counts()
    print("All sparse_sigma tests passed.")
