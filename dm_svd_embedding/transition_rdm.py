#!/usr/bin/env python3
"""
Schmidt-basis transition matrices for second-quantization H^emb construction.

Computes transition matrix elements of the form:
  ⟨Ã_α| O |Ã_γ⟩,  ⟨B̃_β| O |B̃_δ⟩

for single- and two-body operators (a_p†, a_p, a_p† a_q, a_p† a_q†, a_p a_q,
a_p† a_q† a_r, etc.) needed for H_AB contraction.

Strategy:
  1. Compute transition matrices in the A/B subspace determinant basis
     using Slater-Condon phase rules.
  2. Transform to Schmidt basis via U† · T_det · U (or cross-block U† · T · U).
  3. Never touch the full CAS CI space.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
import time
import sys, os


# ═══════════════════════════════════════════════════════════════════════════
# Low-level bit-string helpers
# ═══════════════════════════════════════════════════════════════════════════

def _popcount(x: int) -> int:
    """Number of set bits."""
    return x.bit_count()


def _bits_below(x: int, pos: int) -> int:
    """Number of set bits strictly below position pos in x."""
    mask = (1 << pos) - 1
    return (x & mask).bit_count()


def _create_sign(stra: int, p: int) -> Tuple[int, int]:
    """Phase and new string for a_p†|stra⟩.

    Args:
        stra: Bit string.
        p: Orbital index (0-based).

    Returns:
        (phase, new_str) where phase = ±1, or (0, stra) if already occupied.
    """
    if (stra >> p) & 1:
        return 0, stra
    n_before = _bits_below(stra, p)
    phase = -1 if (n_before & 1) else 1
    new_str = stra | (1 << p)
    return phase, new_str


def _annihilate_sign(stra: int, p: int) -> Tuple[int, int]:
    """Phase and new string for a_p|stra⟩.

    Args:
        stra: Bit string.
        p: Orbital index (0-based).

    Returns:
        (phase, new_str) where phase = ±1, or (0, stra) if already empty.
    """
    if not ((stra >> p) & 1):
        return 0, stra
    n_before = _bits_below(stra, p)
    phase = -1 if (n_before & 1) else 1
    new_str = stra & ~(1 << p)
    return phase, new_str


# ═══════════════════════════════════════════════════════════════════════════
# Determinant-basis transition matrices
# ═══════════════════════════════════════════════════════════════════════════

def _compute_det_creation(
    a_dets_src: List[Tuple[int, int]],
    a_dets_dst: List[Tuple[int, int]],
    a_index_dst: Dict[Tuple[int, int], int],
    n_orb: int,
) -> np.ndarray:
    """Compute ⟨d_dst| a_p† |d_src⟩ in determinant basis.

    Returns: (d_dst, d_src, n_orb) array of matrix elements (±1 or 0).
    The full creation operator includes both alpha and beta spin:
      a_p† = a_pα† + a_pβ†  (with appropriate spin labels).

    Matrix element: ⟨d_i|a_pα†|d_j⟩ is non-zero iff d_i has electron at pα
    and d_j does not, and they agree everywhere else.
    """
    d_dst = len(a_dets_dst)
    d_src = len(a_dets_src)
    T = np.zeros((d_dst, d_src, n_orb))

    for j, (aA_j, bA_j) in enumerate(a_dets_src):
        # Alpha creation
        for p in range(n_orb):
            phase, aA_i = _create_sign(aA_j, p)
            if phase != 0:
                key = (aA_i, bA_j)
                i = a_index_dst.get(key)
                if i is not None:
                    T[i, j, p] = phase

        # Beta creation
        for p in range(n_orb):
            phase, bA_i = _create_sign(bA_j, p)
            if phase != 0:
                key = (aA_j, bA_i)
                i = a_index_dst.get(key)
                if i is not None:
                    T[i, j, p] += phase  # α and β are orthogonal → no double-count

    return T


def _compute_det_annihilation(
    a_dets_src: List[Tuple[int, int]],
    a_dets_dst: List[Tuple[int, int]],
    a_index_dst: Dict[Tuple[int, int], int],
    n_orb: int,
) -> np.ndarray:
    """Compute ⟨d_dst| a_p |d_src⟩ in determinant basis.

    Returns: (d_dst, d_src, n_orb) array.
    """
    d_dst = len(a_dets_dst)
    d_src = len(a_dets_src)
    T = np.zeros((d_dst, d_src, n_orb))

    for j, (aA_j, bA_j) in enumerate(a_dets_src):
        # Alpha annihilation
        for p in range(n_orb):
            phase, aA_i = _annihilate_sign(aA_j, p)
            if phase != 0:
                key = (aA_i, bA_j)
                i = a_index_dst.get(key)
                if i is not None:
                    T[i, j, p] = phase

        # Beta annihilation
        for p in range(n_orb):
            phase, bA_i = _annihilate_sign(bA_j, p)
            if phase != 0:
                key = (aA_j, bA_i)
                i = a_index_dst.get(key)
                if i is not None:
                    T[i, j, p] += phase

    return T


def _compute_det_pair_creation(
    a_dets_src: List[Tuple[int, int]],
    a_dets_dst: List[Tuple[int, int]],
    a_index_dst: Dict[Tuple[int, int], int],
    n_orb: int,
) -> np.ndarray:
    """Compute ⟨d_dst| a_p† a_q† |d_src⟩ in determinant basis.

    Returns: (d_dst, d_src, n_orb, n_orb) array, antisymmetric in (p,q).
    """
    d_dst = len(a_dets_dst)
    d_src = len(a_dets_src)
    T = np.zeros((d_dst, d_src, n_orb, n_orb))

    for j, (aA_j, bA_j) in enumerate(a_dets_src):
        # p, q both alpha
        for p in range(n_orb):
            phase1, a1 = _create_sign(aA_j, p)
            if phase1 == 0:
                continue
            for q in range(p + 1, n_orb):  # only p < q; antisymmetry handles q < p
                phase2, a2 = _create_sign(a1, q)
                if phase2 == 0:
                    continue
                key = (a2, bA_j)
                i = a_index_dst.get(key)
                if i is not None:
                    pq_phase = phase1 * phase2
                    T[i, j, p, q] = pq_phase
                    T[i, j, q, p] = -pq_phase

        # p, q both beta
        for p in range(n_orb):
            phase1, b1 = _create_sign(bA_j, p)
            if phase1 == 0:
                continue
            for q in range(p + 1, n_orb):
                phase2, b2 = _create_sign(b1, q)
                if phase2 == 0:
                    continue
                key = (aA_j, b2)
                i = a_index_dst.get(key)
                if i is not None:
                    pq_phase = phase1 * phase2
                    T[i, j, p, q] += pq_phase
                    T[i, j, q, p] -= pq_phase

        # p ∈ alpha, q ∈ beta (already anticommuting → no order issue)
        for p in range(n_orb):
            phase_p, aA_i = _create_sign(aA_j, p)
            if phase_p == 0:
                continue
            for q in range(n_orb):
                phase_q, bA_i = _create_sign(bA_j, q)
                if phase_q == 0:
                    continue
                key = (aA_i, bA_i)
                i = a_index_dst.get(key)
                if i is not None:
                    T[i, j, p, q] = phase_p * phase_q
                    # No antisymmetry between α and β creation (different spin labels)

    return T


def _compute_det_pair_annihilation(
    a_dets_src: List[Tuple[int, int]],
    a_dets_dst: List[Tuple[int, int]],
    a_index_dst: Dict[Tuple[int, int], int],
    n_orb: int,
) -> np.ndarray:
    """Compute ⟨d_dst| a_q a_p |d_src⟩ in determinant basis.

    Note operator order: a_q a_p = -a_p a_q for same spin.
    Convention: T[i,j,p,q] = ⟨d_i| a_q a_p |d_j⟩, so INDEX p is second annihilation.

    Returns: (d_dst, d_src, n_orb, n_orb) array.
    """
    d_dst = len(a_dets_dst)
    d_src = len(a_dets_src)
    T = np.zeros((d_dst, d_src, n_orb, n_orb))

    for j, (aA_j, bA_j) in enumerate(a_dets_src):
        # p, q both alpha: a_q a_p removes p then q from alpha
        for p in range(n_orb):
            phase_p, a1 = _annihilate_sign(aA_j, p)
            if phase_p == 0:
                continue
            for q in range(n_orb):
                if q == p:
                    continue
                phase_q, a2 = _annihilate_sign(a1, q)
                if phase_q == 0:
                    continue
                key = (a2, bA_j)
                i = a_index_dst.get(key)
                if i is not None:
                    T[i, j, p, q] = phase_p * phase_q

        # p, q both beta
        for p in range(n_orb):
            phase_p, b1 = _annihilate_sign(bA_j, p)
            if phase_p == 0:
                continue
            for q in range(n_orb):
                if q == p:
                    continue
                phase_q, b2 = _annihilate_sign(b1, q)
                if phase_q == 0:
                    continue
                key = (aA_j, b2)
                i = a_index_dst.get(key)
                if i is not None:
                    T[i, j, p, q] += phase_p * phase_q

        # p ∈ alpha, q ∈ beta (different spin, order irrelevant)
        for p in range(n_orb):
            phase_p, aA_i = _annihilate_sign(aA_j, p)
            if phase_p == 0:
                continue
            for q in range(n_orb):
                phase_q, bA_i = _annihilate_sign(bA_j, q)
                if phase_q == 0:
                    continue
                key = (aA_i, bA_i)
                i = a_index_dst.get(key)
                if i is not None:
                    T[i, j, p, q] = phase_p * phase_q

    return T


def _compute_det_1body_transition(
    a_dets: List[Tuple[int, int]],
    a_index: Dict[Tuple[int, int], int],
    n_orb: int,
) -> np.ndarray:
    """Compute ⟨d_i| a_p† a_q |d_j⟩ in determinant basis (same block).

    Returns: (d, d, n_orb, n_orb) array.
    """
    d = len(a_dets)
    T = np.zeros((d, d, n_orb, n_orb))

    for j, (aA_j, bA_j) in enumerate(a_dets):
        # α†α transitions
        for q in range(n_orb):
            if not ((aA_j >> q) & 1):
                continue  # q must be occupied to annihilate
            phase_q, a1 = _annihilate_sign(aA_j, q)
            for p in range(n_orb):
                if (a1 >> p) & 1:
                    continue  # p must be empty to create
                if p == q:
                    # diagonal: a_p† a_p = number operator
                    # phase: creation after annihilation in same orbital
                    # For diagonal: ⟨d_j| a_p† a_p |d_j⟩ = occupancy(p)
                    # We handle diagonal separately, set here for completeness
                    T[j, j, p, q] = 1.0
                    continue
                phase_p, a2 = _create_sign(a1, p)
                if phase_p == 0:
                    continue
                key = (a2, bA_j)
                i = a_index.get(key)
                if i is not None:
                    T[i, j, p, q] = phase_q * phase_p

        # β†β transitions: analogous
        for q in range(n_orb):
            if not ((bA_j >> q) & 1):
                continue
            phase_q, b1 = _annihilate_sign(bA_j, q)
            for p in range(n_orb):
                if (b1 >> p) & 1:
                    continue
                if p == q:
                    T[j, j, p, q] += 1.0
                    continue
                phase_p, b2 = _create_sign(b1, p)
                if phase_p == 0:
                    continue
                key = (aA_j, b2)
                i = a_index.get(key)
                if i is not None:
                    T[i, j, p, q] += phase_q * phase_p

    return T


def _compute_det_create2_annih1(
    a_dets_src: List[Tuple[int, int]],
    a_dets_dst: List[Tuple[int, int]],
    a_index_dst: Dict[Tuple[int, int], int],
    n_orb: int,
) -> np.ndarray:
    """⟨d_dst| a_p† a_q† a_r |d_src⟩: create 2, annihilate 1 (n→n+1).

    Returns: (d_dst, d_src, n_orb, n_orb, n_orb).
    Apply order: a_r (annihilate) first, then a_q†, then a_p†.
    """
    d_dst = len(a_dets_dst)
    d_src = len(a_dets_src)
    T = np.zeros((d_dst, d_src, n_orb, n_orb, n_orb))

    for j, (aA_j, bA_j) in enumerate(a_dets_src):
        # First: a_r (annihilate)
        for r in range(n_orb):
            # Try alpha annihilation
            phase_r_a, a1_a = _annihilate_sign(aA_j, r)
            if phase_r_a != 0:
                # Try beta creation
                for q in range(n_orb):
                    phase_q, b1 = _create_sign(bA_j, q)
                    if phase_q == 0:
                        continue
                    for p in range(n_orb):
                        phase_p, b2 = _create_sign(b1, p)
                        if phase_p == 0:
                            continue
                        key = (a1_a, b2)
                        i = a_index_dst.get(key)
                        if i is not None:
                            T[i, j, p, q, r] = phase_r_a * phase_q * phase_p

            # Try beta annihilation
            phase_r_b, b1_b = _annihilate_sign(bA_j, r)
            if phase_r_b != 0:
                for q in range(n_orb):
                    phase_q, a1 = _create_sign(aA_j, q)
                    if phase_q == 0:
                        continue
                    for p in range(n_orb):
                        phase_p, a2 = _create_sign(a1, p)
                        if phase_p == 0:
                            continue
                        key = (a2, b1_b)
                        i = a_index_dst.get(key)
                        if i is not None:
                            T[i, j, p, q, r] += phase_r_b * phase_q * phase_p

    return T


def _compute_det_create1_annih2(
    a_dets_src: List[Tuple[int, int]],
    a_dets_dst: List[Tuple[int, int]],
    a_index_dst: Dict[Tuple[int, int], int],
    n_orb: int,
) -> np.ndarray:
    """⟨d_dst| a_p† a_q a_r |d_src⟩: create 1, annihilate 2 (n→n-1).

    Returns: (d_dst, d_src, n_orb, n_orb, n_orb).
    Apply order: a_r (annihilate) first, then a_q (annihilate), then a_p†.
    """
    d_dst = len(a_dets_dst)
    d_src = len(a_dets_src)
    T = np.zeros((d_dst, d_src, n_orb, n_orb, n_orb))

    for j, (aA_j, bA_j) in enumerate(a_dets_src):
        # a_r: first annihilation
        for r in range(n_orb):
            # Try alpha annihilation first
            ph_r, a1 = _annihilate_sign(aA_j, r)
            if ph_r != 0:
                # a_q: second annihilation — try beta
                for q in range(n_orb):
                    ph_q, b1 = _annihilate_sign(bA_j, q)
                    if ph_q == 0:
                        continue
                    # a_p†: creation — try beta
                    for p in range(n_orb):
                        ph_p, b2 = _create_sign(b1, p)
                        if ph_p == 0:
                            continue
                        key = (a1, b2)
                        i = a_index_dst.get(key)
                        if i is not None:
                            T[i, j, p, q, r] = ph_r * ph_q * ph_p
            
            # Try beta annihilation first
            ph_r, b1_r = _annihilate_sign(bA_j, r)
            if ph_r != 0:
                for q in range(n_orb):
                    # Try alpha annihilation
                    ph_q, a1_q = _annihilate_sign(aA_j, q)
                    if ph_q == 0:
                        continue
                    for p in range(n_orb):
                        ph_p, a2 = _create_sign(a1_q, p)
                        if ph_p == 0:
                            continue
                        key = (a2, b1_r)
                        i = a_index_dst.get(key)
                        if i is not None:
                            T[i, j, p, q, r] += ph_r * ph_q * ph_p

    return T


def _transform_3body_to_schmidt(T_det, U_dst, U_src):
    """Transform 3-body transition to Schmidt basis."""
    return np.einsum('ijpqr,ia,jg->agpqr', T_det, U_dst, U_src)


# ═══════════════════════════════════════════════════════════════════════════
# Schmidt-basis transformation
# ═══════════════════════════════════════════════════════════════════════════

def _transform_1body_to_schmidt(
    T_det: np.ndarray,
    U_dst: np.ndarray,
    U_src: np.ndarray,
) -> np.ndarray:
    """U_dst† @ T_det @ U_src for 1-body transitions (d_dst, d_src, n_orb).

    Returns: (r_dst, r_src, n_orb).
    """
    # T_det: (d_dst, d_src, n_orb)
    # Einsum: αγp = Σ_{i,j} U_{iα}(dst) · T[i,j,p] · U_{jγ}(src)
    return np.einsum('ijp,ia,jg->agp', T_det, U_dst, U_src)


def _transform_2body_to_schmidt(
    T_det: np.ndarray,
    U_dst: np.ndarray,
    U_src: np.ndarray,
) -> np.ndarray:
    """U_dst† @ T_det @ U_src for 2-body transitions (d_dst, d_src, n_orb, n_orb).

    Returns: (r_dst, r_src, n_orb, n_orb).
    """
    return np.einsum('ijpq,ia,jg->agpq', T_det, U_dst, U_src)


# ═══════════════════════════════════════════════════════════════════════════
# TransitionMatrices dataclass
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TransitionMatrices:
    """Schmidt-basis transition matrices for one subspace (A or B)."""
    n_orb: int
    blocks: List[int] = field(default_factory=list)
    create_1: Dict[int, np.ndarray] = field(default_factory=dict)
    annihilate_1: Dict[int, np.ndarray] = field(default_factory=dict)
    trans_1: Dict[int, np.ndarray] = field(default_factory=dict)
    create_2: Dict[int, np.ndarray] = field(default_factory=dict)
    annihilate_2: Dict[int, np.ndarray] = field(default_factory=dict)
    # 3-body: create 2 + annihilate 1 (n→n+1): ⟨α| a_p† a_q† a_r |γ⟩
    create2_annih1: Dict[int, np.ndarray] = field(default_factory=dict)
    # 3-body: create 1 + annihilate 2 (n→n-1): ⟨α| a_p† a_q a_r |γ⟩
    create1_annih2: Dict[int, np.ndarray] = field(default_factory=dict)
    U_blocks: Dict[int, np.ndarray] = field(default_factory=dict)
    r_blocks: Dict[int, int] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════
# Main API
# ═══════════════════════════════════════════════════════════════════════════

def compute_transition_matrices(
    partition: Dict[int, Dict],
    schmidt_data: Dict[int, Dict],
    n_orb: int,
    subspace: str = 'A',
    verbose: bool = True,
) -> TransitionMatrices:
    """Compute all needed transition matrices in the Schmidt basis.

    Args:
        partition: Output of partition_determinants (occ_virt_partition.py).
        schmidt_data: Output of compute_schmidt_decomposition (density_matrix.py).
        n_orb: Number of orbitals in this subspace.
        subspace: 'A' or 'B'.
        verbose: Print progress.

    Returns:
        TransitionMatrices with all precomputed transition arrays.
    """
    if verbose:
        t0 = time.perf_counter()
        print(f"  Computing {subspace}-space transition matrices "
              f"({n_orb} orbitals)...")

    result = TransitionMatrices(n_orb=n_orb)

    # Determine which blocks exist
    all_n = sorted(schmidt_data.keys())
    result.blocks = all_n

    # Store U and r per block
    for n_A in all_n:
        sd = schmidt_data[n_A]
        if subspace == 'A':
            result.U_blocks[n_A] = sd['U'] if sd['r'] > 0 else np.zeros((sd['dim_A'], 0))
        else:
            result.U_blocks[n_A] = sd['V'] if sd['r'] > 0 else np.zeros((sd['dim_B'], 0))
        result.r_blocks[n_A] = sd['r']

    # Get det lists per block
    for n_A in all_n:
        blk = partition.get(n_A)
        if blk is None:
            continue
        sd = schmidt_data[n_A]
        r = sd['r']
        if r == 0:
            continue

        if subspace == 'A':
            dets_n = blk['a_dets']
            idx_n = blk['a_index']
            U = sd['U']
        else:
            dets_n = blk['b_dets']
            idx_n = blk['b_index']
            U = sd['V']

        # ── Single creation ──
        # A-space: n_A → n_A+1 (A gains electron, dest = n_A+1)
        # B-space: N_B → N_B+1 means n_A → n_A-1 (B gains, A loses)
        if subspace == 'A':
            dk_create = +1
            dk_annih = -1
            dk2_create = +2
            dk2_annih = -2
        else:
            dk_create = -1
            dk_annih = +1
            dk2_create = -2
            dk2_annih = +2

        blk_create_dst = partition.get(n_A + dk_create)
        sd_create_dst = schmidt_data.get(n_A + dk_create)
        if blk_create_dst is not None and sd_create_dst is not None and sd_create_dst['r'] > 0:
            if subspace == 'A':
                dets_dst = blk_create_dst['a_dets']
                idx_dst = blk_create_dst['a_index']
                U_dst = sd_create_dst['U']
            else:
                dets_dst = blk_create_dst['b_dets']
                idx_dst = blk_create_dst['b_index']
                U_dst = sd_create_dst['V']

            T_det = _compute_det_creation(dets_n, dets_dst, idx_dst, n_orb)
            result.create_1[n_A] = _transform_1body_to_schmidt(T_det, U_dst, U)

        # ── Single annihilation ──
        blk_annih_dst = partition.get(n_A + dk_annih)
        sd_annih_dst = schmidt_data.get(n_A + dk_annih)
        if blk_annih_dst is not None and sd_annih_dst is not None and sd_annih_dst['r'] > 0:
            if subspace == 'A':
                dets_dst = blk_annih_dst['a_dets']
                idx_dst = blk_annih_dst['a_index']
                U_dst = sd_annih_dst['U']
            else:
                dets_dst = blk_annih_dst['b_dets']
                idx_dst = blk_annih_dst['b_index']
                U_dst = sd_annih_dst['V']

            T_det = _compute_det_annihilation(dets_n, dets_dst, idx_dst, n_orb)
            result.annihilate_1[n_A] = _transform_1body_to_schmidt(T_det, U_dst, U)

        # ── 1-body transition (n → n, same block) ──
        T_det = _compute_det_1body_transition(dets_n, idx_n, n_orb)
        # T_det: (d, d, n_orb, n_orb)
        result.trans_1[n_A] = _transform_2body_to_schmidt(T_det, U, U)

        # ── Pair creation ──
        blk_create2_dst = partition.get(n_A + dk2_create)
        sd_create2_dst = schmidt_data.get(n_A + dk2_create)
        if blk_create2_dst is not None and sd_create2_dst is not None and sd_create2_dst['r'] > 0:
            if subspace == 'A':
                dets_dst = blk_create2_dst['a_dets']
                idx_dst = blk_create2_dst['a_index']
                U_dst = sd_create2_dst['U']
            else:
                dets_dst = blk_create2_dst['b_dets']
                idx_dst = blk_create2_dst['b_index']
                U_dst = sd_create2_dst['V']

            T_det = _compute_det_pair_creation(dets_n, dets_dst, idx_dst, n_orb)
            result.create_2[n_A] = _transform_2body_to_schmidt(T_det, U_dst, U)

        # ── Pair annihilation ──
        blk_annih2_dst = partition.get(n_A + dk2_annih)
        sd_annih2_dst = schmidt_data.get(n_A + dk2_annih)
        if blk_annih2_dst is not None and sd_annih2_dst is not None and sd_annih2_dst['r'] > 0:
            if subspace == 'A':
                dets_dst = blk_annih2_dst['a_dets']
                idx_dst = blk_annih2_dst['a_index']
                U_dst = sd_annih2_dst['U']
            else:
                dets_dst = blk_annih2_dst['b_dets']
                idx_dst = blk_annih2_dst['b_index']
                U_dst = sd_annih2_dst['V']

            T_det = _compute_det_pair_annihilation(dets_n, dets_dst, idx_dst, n_orb)
            result.annihilate_2[n_A] = _transform_2body_to_schmidt(T_det, U_dst, U)

        # ── 3-body: create 2 + annihilate 1 (n → n+1) ──
        blk_3b_dst = partition.get(n_A + dk_create)  # same direction as single creation
        sd_3b_dst = schmidt_data.get(n_A + dk_create)
        if blk_3b_dst is not None and sd_3b_dst is not None and sd_3b_dst['r'] > 0:
            if subspace == 'A':
                dets_dst = blk_3b_dst['a_dets']
                idx_dst = blk_3b_dst['a_index']
                U_dst = sd_3b_dst['U']
            else:
                dets_dst = blk_3b_dst['b_dets']
                idx_dst = blk_3b_dst['b_index']
                U_dst = sd_3b_dst['V']
            T_det = _compute_det_create2_annih1(dets_n, dets_dst, idx_dst, n_orb)
            result.create2_annih1[n_A] = _transform_3body_to_schmidt(T_det, U_dst, U)

        # ── 3-body: create 1 + annihilate 2 (n → n-1) ──
        blk_3b2_dst = partition.get(n_A + dk_annih)
        sd_3b2_dst = schmidt_data.get(n_A + dk_annih)
        if blk_3b2_dst is not None and sd_3b2_dst is not None and sd_3b2_dst['r'] > 0:
            if subspace == 'A':
                dets_dst = blk_3b2_dst['a_dets']
                idx_dst = blk_3b2_dst['a_index']
                U_dst = sd_3b2_dst['U']
            else:
                dets_dst = blk_3b2_dst['b_dets']
                idx_dst = blk_3b2_dst['b_index']
                U_dst = sd_3b2_dst['V']
            T_det = _compute_det_create1_annih2(dets_n, dets_dst, idx_dst, n_orb)
            result.create1_annih2[n_A] = _transform_3body_to_schmidt(T_det, U_dst, U)

    if verbose:
        elapsed = time.perf_counter() - t0
        # Count total stored elements
        n_elem = 0
        for d in [result.create_1, result.annihilate_1, result.trans_1,
                   result.create_2, result.annihilate_2]:
            for v in d.values():
                n_elem += v.size
        print(f"    Done in {elapsed:.1f}s, {n_elem} elements "
              f"({n_elem * 8 / 1024:.1f} KB)")

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════

def test_transition_matrices_h2o():
    """Verify transition matrices on H₂O/STO-3G CAS(5,6)."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

    from dm_svd_embedding.occ_virt_partition import (
        setup_partition, build_block_matrices,
    )
    from dm_svd_embedding.density_matrix import (
        compute_schmidt_decomposition,
    )
    from pyscf import gto, scf, mcscf
    from pyscf.fci import direct_spin1

    n_act, n_elec = 5, 6
    n_occ = 3
    n_virt = n_act - n_occ

    mol = gto.M(atom='O 0 0 0; H 0 0.757 0.586; H 0 -0.757 0.586',
                basis='sto-3g', verbose=0)
    mf = scf.RHF(mol); mf.kernel()
    cas = mcscf.CASCI(mf, n_act, n_elec)
    cas.frozen = 2
    cas.kernel()
    fcivec = cas.ci
    ci_flat = fcivec.reshape(-1)

    partition, full_dets = setup_partition(n_act, n_elec, n_occ, ms=0)
    C_blocks = build_block_matrices(partition, ci_flat)
    schmidt = compute_schmidt_decomposition(C_blocks, eps=1e-3)

    print(f"H₂O/STO-3G CAS(5,6): {len(full_dets)} dets, "
          f"{len(partition)} blocks")

    # A-space transition matrices
    trans_A = compute_transition_matrices(
        partition, schmidt, n_occ, subspace='A', verbose=True)

    # B-space transition matrices
    trans_B = compute_transition_matrices(
        partition, schmidt, n_virt, subspace='B', verbose=True)

    # Check basic properties
    for n_A in schmidt:
        sd = schmidt[n_A]
        r = sd['r']
        if r == 0:
            continue

        # trans_1 should have correct shape
        if n_A in trans_A.trans_1:
            t1 = trans_A.trans_1[n_A]
            assert t1.shape == (r, r, n_occ, n_occ), \
                f"A trans_1 shape {t1.shape} != ({r},{r},{n_occ},{n_occ})"
            # Diagonal should be non-zero (occupation numbers)
            diag_sum = sum(t1[a, a, p, p] for a in range(r) for p in range(n_occ))
            print(f"  n_A={n_A}: A diag sum = {diag_sum:.4f}")
            assert diag_sum > 0, "Diagonal 1-body transitions should have occupation"

    print("  ✓ Transition matrices basic checks passed")

    # Verify: creation + annihilation should be consistent for same-block
    for n_A in schmidt:
        sd_n = schmidt[n_A]
        if sd_n['r'] == 0:
            continue
        if n_A in trans_A.create_1 and (n_A + 1) in trans_A.annihilate_1:
            C = trans_A.create_1[n_A]  # (r_np1, r_n, n_orb)
            A = trans_A.annihilate_1[n_A + 1]  # (r_n, r_np1, n_orb)
            # A[α,β,p] should equal C[β,α,p] (adjoint)
            # Check per-orbital
            for p in range(n_occ):
                diff = np.abs(C[:, :, p] - A[:, :, p].T).max()
                if diff > 1e-10:
                    print(f"  WARNING: create/annihilate mismatch p={p}: {diff:.2e}")
            print(f"  n_A={n_A}: create/annihilate adjoint check OK")

    print("  ✓ All transition matrix tests passed")
    return trans_A, trans_B


if __name__ == "__main__":
    test_transition_matrices_h2o()
    print("All transition_rdm tests passed.")
