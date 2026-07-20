#!/usr/bin/env python3
"""
Occ/Virt determinant space partition.

Partitions the CAS active-space MOs into:
  - Space A: N_occ occupied orbitals (indices 0 .. N_occ-1)
  - Space B: N_virt virtual orbitals (indices N_occ .. N_act-1)

Each full-CAS determinant is uniquely factorised as:

    |Φ_I⟩ = |a_i^(n)⟩ ⊗ |b_j^(N-n)⟩

where n = number of electrons residing in Space A.

For each electron-number block n, we build:
  - F_A(n): all n-electron determinants within the N_occ A-space orbitals
  - F_B(N-n): all (N-n)-electron determinants within the N_virt B-space orbitals
  - C^(n): CI coefficient matrix of shape (dim F_A(n), dim F_B(N-n))

References:
  - DensityMatrix_SVD_Embedding_Proposal.md, Sec. 2.1-2.2
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
import sys, os


# ---------- helpers ----------

def _bit_popcount(x: int) -> int:
    """Population count (number of set bits)."""
    return x.bit_count()


def _partition_alpha_beta(
    alpha_str: int, beta_str: int, n_occ: int,
) -> Tuple[int, int, int, int, int]:
    """Split a full-CAS (alpha, beta) into A and B parts.

    Orbitals 0 .. n_occ-1 belong to Space A.
    Orbitals n_occ .. belong to Space B.

    Returns:
        (alpha_A, beta_A, alpha_B, beta_B, n_A) where:
        - alpha_A, beta_A: bits in Space A, 0-indexed within A
        - alpha_B, beta_B: bits in Space B, 0-indexed within B
        - n_A: total electrons in A = popcount(alpha_A) + popcount(beta_A)
    """
    a_mask = (1 << n_occ) - 1
    alpha_A = alpha_str & a_mask
    beta_A = beta_str & a_mask
    alpha_B = alpha_str >> n_occ
    beta_B = beta_str >> n_occ
    n_A = _bit_popcount(alpha_A) + _bit_popcount(beta_A)
    return alpha_A, beta_A, alpha_B, beta_B, n_A


def _generate_subspace_strings(
    n_orb: int, n_alpha: int, n_beta: int,
) -> Tuple[List[int], List[int]]:
    """Generate all alpha and beta strings for a subspace.

    Uses PySCF cistring.gen_strings4orblist.

    Returns:
        (alpha_strs, beta_strs) as lists of Python ints.
    """
    from pyscf.fci import cistring
    orb_list = list(range(n_orb))
    alphas = [int(s) for s in cistring.gen_strings4orblist(orb_list, n_alpha)]
    betas = [int(s) for s in cistring.gen_strings4orblist(orb_list, n_beta)]
    return alphas, betas


# ---------- main API ----------

def generate_subspace_determinants(
    n_orb: int, n_alpha: int, n_beta: int,
) -> List[Tuple[int, int]]:
    """Generate all determinants within a subspace.

    Args:
        n_orb: Number of spatial orbitals in the subspace.
        n_alpha, n_beta: Electron counts.

    Returns:
        List of (alpha_str, beta_str) tuples.
    """
    alphas, betas = _generate_subspace_strings(n_orb, n_alpha, n_beta)
    dets = []
    for a in alphas:
        for b in betas:
            dets.append((a, b))
    return dets


def partition_determinants(
    full_dets: List[Tuple[int, int]],
    n_occ: int,
    n_alpha_total: int,
    n_beta_total: int,
) -> Dict[int, Dict]:
    """Partition full-CAS determinants into occ/virt blocks.

    For each determinant, compute n = electron count in Space A.
    Returns a dict mapping n → block metadata.

    Args:
        full_dets: List of all (alpha_str, beta_str) tuples for full CAS.
        n_occ: Number of spatial orbitals in Space A (occupied).
        n_alpha_total: Total alpha electrons in CAS.
        n_beta_total: Total beta electrons in CAS.

    Returns:
        Dict[n] = {
            'n': int,
            'a_dets': list of (alpha_A, beta_A) tuples for F_A(n),
            'b_dets': list of (alpha_B, beta_B) tuples for F_B(N-n),
            'a_index': dict (alpha_A, beta_A) → index in a_dets,
            'b_index': dict (alpha_B, beta_B) → index in b_dets,
            'dim_A': int,
            'dim_B': int,
            'coeff_map': list of (i, j, coeff) for building C^(n),
                where i = a_index[(alpha_A,beta_A)], j = b_index[(alpha_B,beta_B)].
            'n_entries': int (number of full dets in this block).
        }
    """
    # Determine n_act = n_occ + n_virt. n_virt is implied by max orbital index.
    max_orb = 0
    for a, b in full_dets:
        for s in [a, b]:
            if s > 0:
                max_orb = max(max_orb, s.bit_length() - 1)
    n_act = max_orb + 1
    n_virt = n_act - n_occ

    # ── Step 1: Count determinants per n and collect A/B dets ──
    # We do two passes:
    #   Pass 1: collect unique A and B subspace dets per n.
    #   Pass 2: build the index maps and coeff entries.

    # Pass 1: collect
    a_dets_by_n: Dict[int, set] = {}
    b_dets_by_n: Dict[int, set] = {}
    coeff_entries_by_n: Dict[int, list] = {}

    for det_idx, (a_full, b_full) in enumerate(full_dets):
        alpha_A, beta_A, alpha_B, beta_B, n_A = _partition_alpha_beta(
            a_full, b_full, n_occ)
        if n_A not in a_dets_by_n:
            a_dets_by_n[n_A] = set()
            b_dets_by_n[n_A] = set()
            coeff_entries_by_n[n_A] = []
        a_dets_by_n[n_A].add((alpha_A, beta_A))
        b_dets_by_n[n_A].add((alpha_B, beta_B))

    # ── Step 2: Build sorted index maps ──
    result = {}
    for n_A in sorted(a_dets_by_n.keys()):
        n_A_alpha = (n_A + 0)  # to be determined from the dets
        n_A_beta = 0

        a_dets_list = sorted(a_dets_by_n[n_A])
        b_dets_list = sorted(b_dets_by_n[n_A])

        a_index = {d: i for i, d in enumerate(a_dets_list)}
        b_index = {d: j for j, d in enumerate(b_dets_list)}

        dim_A = len(a_dets_list)
        dim_B = len(b_dets_list)

        result[n_A] = {
            'n': n_A,
            'a_dets': a_dets_list,
            'b_dets': b_dets_list,
            'a_index': a_index,
            'b_index': b_index,
            'dim_A': dim_A,
            'dim_B': dim_B,
            'coeff_map': [],   # populated in Pass 2
            'n_entries': 0,
        }

    # Pass 2: build coeff maps
    for det_idx, (a_full, b_full) in enumerate(full_dets):
        alpha_A, beta_A, alpha_B, beta_B, n_A = _partition_alpha_beta(
            a_full, b_full, n_occ)
        blk = result[n_A]
        blk['coeff_map'].append((
            blk['a_index'][(alpha_A, beta_A)],
            blk['b_index'][(alpha_B, beta_B)],
            det_idx,  # store global det index; CI coeff filled later
        ))
        blk['n_entries'] += 1

    return result


def build_block_matrices(
    partition: Dict[int, Dict],
    ci_vector: np.ndarray,
) -> Dict[int, np.ndarray]:
    """Build C^(n) coefficient matrices from the full CI vector.

    C^(n)[i, j] = CI coefficient of determinant |a_i^(n)⟩ ⊗ |b_j^(N-n)⟩.

    Args:
        partition: Output of partition_determinants().
        ci_vector: Full CI vector (length = total number of CAS dets).

    Returns:
        Dict[n] → C^(n) matrix of shape (dim_A, dim_B).
    """
    matrices = {}
    for n_A, blk in partition.items():
        C = np.zeros((blk['dim_A'], blk['dim_B']))
        for (i, j, det_idx) in blk['coeff_map']:
            C[i, j] = ci_vector[det_idx]
        matrices[n_A] = C
    return matrices


# ---------- convenience: generate full CAS + partition in one go ----------

def setup_partition(
    n_act: int,
    n_elec: int,
    n_occ: int,
    ms: int = 0,
) -> Tuple[Dict[int, Dict], List[Tuple[int, int]]]:
    """Generate all CAS determinants and partition by occ/virt.

    Convenience wrapper combining generate_determinants_ms + partition_determinants.

    Args:
        n_act: Total active spatial orbitals.
        n_elec: Total active electrons.
        n_occ: Number of occupied orbitals (Space A).
        ms: 2*Sz (default 0).

    Returns:
        (partition, full_dets):
          partition: Dict[n] → block metadata (from partition_determinants).
          full_dets: List of all (alpha, beta) tuples.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from src.determinants import generate_determinants_ms

    n_alpha = (n_elec + ms) // 2
    n_beta = n_elec - n_alpha
    full_dets = generate_determinants_ms(n_act, n_elec, ms=ms)

    partition = partition_determinants(
        full_dets, n_occ, n_alpha, n_beta)

    return partition, full_dets


# ---------- tests ----------

def test_partition_h2o_sto3g():
    """Verify partition on H₂O/STO-3G CAS(6,5): 2 core frozen, 5 active MOs."""
    n_act, n_elec = 5, 6
    n_occ = 3  # occupied in HF: 2 O-H σ, 1 O lone-pair-ish
    n_virt = n_act - n_occ  # 2 virtual

    partition, full_dets = setup_partition(n_act, n_elec, n_occ, ms=0)

    # Check that total determinant count sums correctly
    total_dets = sum(blk['n_entries'] for blk in partition.values())
    assert total_dets == len(full_dets), \
        f"Partition lost dets: {total_dets} vs {len(full_dets)}"

    # Check that C matrices have consistent dimensions
    for n_A, blk in partition.items():
        n_B = n_elec - n_A  # electrons in B
        # Verify that dim_A and dim_B match the combos
        assert blk['dim_A'] == len(blk['a_index'])
        assert blk['dim_B'] == len(blk['b_index'])
        print(f"  n={n_A}: dim_A={blk['dim_A']}, dim_B={blk['dim_B']}, "
              f"n_entries={blk['n_entries']}")

    # Verify the factorization: for each full det, the A and B parts
    # should reconstruct the original
    a_mask = (1 << n_occ) - 1
    for a_full, b_full in full_dets:
        alpha_A = a_full & a_mask
        beta_A = b_full & a_mask
        alpha_B = a_full >> n_occ
        beta_B = b_full >> n_occ
        n_A = _bit_popcount(alpha_A) + _bit_popcount(beta_A)
        blk = partition[n_A]
        assert (alpha_A, beta_A) in blk['a_index']
        assert (alpha_B, beta_B) in blk['b_index']

    print(f"  ✓ H₂O/STO-3G CAS(5,6): {len(full_dets)} dets → "
          f"{len(partition)} blocks, all consistent")


if __name__ == "__main__":
    test_partition_h2o_sto3g()
    print("All occ_virt_partition tests passed.")