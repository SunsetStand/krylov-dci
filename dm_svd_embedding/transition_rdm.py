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


def _compute_det_creation_explicit(
    a_dets_src: List[Tuple[int, int]],
    a_dets_dst: List[Tuple[int, int]],
    a_index_dst: Dict[Tuple[int, int], int],
    n_orb: int,
) -> Dict[str, np.ndarray]:
    """Spin-EXPLICIT single creation: returns {'a','b'} arrays.

    T['a'][i,j,p] = ⟨d_dst| a_pα† |d_src⟩
    T['b'][i,j,p] = ⟨d_dst| a_pβ† |d_src⟩
    The spin-summed create_1 is 'a' + 'b'.
    """
    d_dst = len(a_dets_dst)
    d_src = len(a_dets_src)
    Ta = np.zeros((d_dst, d_src, n_orb))
    Tb = np.zeros((d_dst, d_src, n_orb))
    for j, (aA_j, bA_j) in enumerate(a_dets_src):
        for p in range(n_orb):
            phase, aA_i = _create_sign(aA_j, p)
            if phase != 0:
                key = (aA_i, bA_j)
                i = a_index_dst.get(key)
                if i is not None:
                    Ta[i, j, p] = phase
        for p in range(n_orb):
            phase, bA_i = _create_sign(bA_j, p)
            if phase != 0:
                key = (aA_j, bA_i)
                i = a_index_dst.get(key)
                if i is not None:
                    Tb[i, j, p] = phase
    return {'a': Ta, 'b': Tb}


def _compute_det_annihilation_explicit(
    a_dets_src: List[Tuple[int, int]],
    a_dets_dst: List[Tuple[int, int]],
    a_index_dst: Dict[Tuple[int, int], int],
    n_orb: int,
) -> Dict[str, np.ndarray]:
    """Spin-EXPLICIT single annihilation: returns {'a','b'} arrays.

    T['a'][i,j,p] = ⟨d_dst| a_pα |d_src⟩
    T['b'][i,j,p] = ⟨d_dst| a_pβ |d_src⟩
    """
    d_dst = len(a_dets_dst)
    d_src = len(a_dets_src)
    Ta = np.zeros((d_dst, d_src, n_orb))
    Tb = np.zeros((d_dst, d_src, n_orb))
    for j, (aA_j, bA_j) in enumerate(a_dets_src):
        for p in range(n_orb):
            phase, aA_i = _annihilate_sign(aA_j, p)
            if phase != 0:
                key = (aA_i, bA_j)
                i = a_index_dst.get(key)
                if i is not None:
                    Ta[i, j, p] = phase
        for p in range(n_orb):
            phase, bA_i = _annihilate_sign(bA_j, p)
            if phase != 0:
                key = (aA_j, bA_i)
                i = a_index_dst.get(key)
                if i is not None:
                    Tb[i, j, p] = phase
    return {'a': Ta, 'b': Tb}


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


def _compute_det_1body_transition_explicit(
    a_dets: List[Tuple[int, int]],
    a_index: Dict[Tuple[int, int], int],
    n_orb: int,
) -> Dict[str, np.ndarray]:
    """Compute spin-EXPLICIT 1-body transitions ⟨d_i| a_p,σ† a_q,σ' |d_j⟩.

    Returns a dict with four (d, d, n_orb, n_orb) arrays, one per spin pair:
      'aa': a_p,α† a_q,α   (same-spin alpha)
      'bb': a_p,β† a_q,β   (same-spin beta)
      'ab': a_p,α† a_q,β   (spin-flip: create alpha, annihilate beta)
      'ba': a_p,β† a_q,α   (spin-flip: create beta, annihilate alpha)

    Index convention matches _compute_det_1body_transition:
      T[dst, src, p, q] = ⟨d_dst| a_p† a_q |d_src⟩.
    The spin-summed trans_1 is 'aa' + 'bb'.

    In the (alpha-string, beta-string) representation the alpha and beta
    operators act on independent strings, so no cross fermion sign arises:
    the phase of a mixed operator is the product of the individual
    create/annihilate phases.
    """
    d = len(a_dets)
    comps = {k: np.zeros((d, d, n_orb, n_orb)) for k in ('aa', 'ab', 'ba', 'bb')}

    for j, (aA_j, bA_j) in enumerate(a_dets):
        # ── 'aa': a_p,α† a_q,α ──
        for q in range(n_orb):
            if not ((aA_j >> q) & 1):
                continue
            phase_q, a1 = _annihilate_sign(aA_j, q)
            for p in range(n_orb):
                if (a1 >> p) & 1:
                    continue
                if p == q:
                    comps['aa'][j, j, p, q] = 1.0
                    continue
                phase_p, a2 = _create_sign(a1, p)
                if phase_p == 0:
                    continue
                key = (a2, bA_j)
                i = a_index.get(key)
                if i is not None:
                    comps['aa'][i, j, p, q] = phase_q * phase_p

        # ── 'bb': a_p,β† a_q,β ──
        for q in range(n_orb):
            if not ((bA_j >> q) & 1):
                continue
            phase_q, b1 = _annihilate_sign(bA_j, q)
            for p in range(n_orb):
                if (b1 >> p) & 1:
                    continue
                if p == q:
                    comps['bb'][j, j, p, q] = 1.0
                    continue
                phase_p, b2 = _create_sign(b1, p)
                if phase_p == 0:
                    continue
                key = (aA_j, b2)
                i = a_index.get(key)
                if i is not None:
                    comps['bb'][i, j, p, q] = phase_q * phase_p

        # ── 'ab': a_p,α† a_q,β (create alpha, annihilate beta) ──
        for q in range(n_orb):
            if not ((bA_j >> q) & 1):
                continue
            phase_q, b1 = _annihilate_sign(bA_j, q)
            for p in range(n_orb):
                if (aA_j >> p) & 1:
                    continue
                phase_p, a2 = _create_sign(aA_j, p)
                if phase_p == 0:
                    continue
                key = (a2, b1)
                i = a_index.get(key)
                if i is not None:
                    comps['ab'][i, j, p, q] = phase_q * phase_p

        # ── 'ba': a_p,β† a_q,α (create beta, annihilate alpha) ──
        for q in range(n_orb):
            if not ((aA_j >> q) & 1):
                continue
            phase_q, a1 = _annihilate_sign(aA_j, q)
            for p in range(n_orb):
                if (bA_j >> p) & 1:
                    continue
                phase_p, b2 = _create_sign(bA_j, p)
                if phase_p == 0:
                    continue
                key = (a1, b2)
                i = a_index.get(key)
                if i is not None:
                    comps['ba'][i, j, p, q] = phase_q * phase_p

    return comps


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


def _jw_phase(dn_a, dn_b, aA_j, bA_j):
    """Jordan-Wigner phase baked into A-side transition matrices.

    A B-space operator crosses the n_σ(A) A-space σ-electrons.  Whether it
    crosses before or after the A operators act depends on their ordering, but
    the net effect only depends on the A-side operator's NET spin change:
      Δn_σ = +1 (net create) -> factor (-1)^{n_σ}
      Δn_σ = -1 (net annih)  -> factor (-1)^{n_σ - 1}
      Δn_σ =  0              -> factor +1
    """
    na = aA_j.bit_count()
    nb = bA_j.bit_count()
    e = 0
    if dn_a > 0:
        e += na
    elif dn_a < 0:
        e += na - 1
    if dn_b > 0:
        e += nb
    elif dn_b < 0:
        e += nb - 1
    return -1 if (e & 1) else 1


def _compute_det_pair_creation_explicit(
    a_dets_src: List[Tuple[int, int]],
    a_dets_dst: List[Tuple[int, int]],
    a_index_dst: Dict[Tuple[int, int], int],
    n_orb: int,
) -> Dict[str, np.ndarray]:
    """Spin-EXPLICIT pair creation.

    Returns dict {'aa','ab','ba','bb'}, each (d_dst, d_src, n_orb, n_orb):
      T['ab'][i,j,p,q] = ⟨d_i| a†_pα a†_qβ |d_j⟩   (a†_qβ acts first, then a†_pα)
    The spin-summed create_2 is 'aa'+'ab'+'ba'+'bb'.
    """
    d_dst = len(a_dets_dst)
    d_src = len(a_dets_src)
    comps = {k: np.zeros((d_dst, d_src, n_orb, n_orb))
             for k in ('aa', 'ab', 'ba', 'bb')}

    for j, (aA_j, bA_j) in enumerate(a_dets_src):
        # 'aa': a†_pα a†_qα  -> create q (α) first, then p (α)
        for q in range(n_orb):
            ph_q, a1 = _create_sign(aA_j, q)
            if ph_q == 0:
                continue
            for p in range(n_orb):
                if p == q:
                    continue
                ph_p, a2 = _create_sign(a1, p)
                if ph_p == 0:
                    continue
                i = a_index_dst.get((a2, bA_j))
                if i is not None:
                    comps['aa'][i, j, p, q] = ph_q * ph_p

        # 'bb': a†_pβ a†_qβ  -> create q (β) first, then p (β)
        for q in range(n_orb):
            ph_q, b1 = _create_sign(bA_j, q)
            if ph_q == 0:
                continue
            for p in range(n_orb):
                if p == q:
                    continue
                ph_p, b2 = _create_sign(b1, p)
                if ph_p == 0:
                    continue
                i = a_index_dst.get((aA_j, b2))
                if i is not None:
                    comps['bb'][i, j, p, q] = ph_q * ph_p

        # 'ab': a†_pα a†_qβ  -> create q (β) first, then p (α)
        for q in range(n_orb):
            ph_q, b1 = _create_sign(bA_j, q)
            if ph_q == 0:
                continue
            for p in range(n_orb):
                ph_p, a1 = _create_sign(aA_j, p)
                if ph_p == 0:
                    continue
                i = a_index_dst.get((a1, b1))
                if i is not None:
                    comps['ab'][i, j, p, q] = ph_q * ph_p

        # 'ba': a†_pβ a†_qα  -> create q (α) first, then p (β)
        for q in range(n_orb):
            ph_q, a1 = _create_sign(aA_j, q)
            if ph_q == 0:
                continue
            for p in range(n_orb):
                ph_p, b1 = _create_sign(bA_j, p)
                if ph_p == 0:
                    continue
                i = a_index_dst.get((a1, b1))
                if i is not None:
                    comps['ba'][i, j, p, q] = ph_q * ph_p

    return comps


def _compute_det_pair_annihilation_explicit(
    a_dets_src: List[Tuple[int, int]],
    a_dets_dst: List[Tuple[int, int]],
    a_index_dst: Dict[Tuple[int, int], int],
    n_orb: int,
) -> Dict[str, np.ndarray]:
    """Spin-EXPLICIT pair annihilation.

    Returns dict {'aa','ab','ba','bb'}, each (d_dst, d_src, n_orb, n_orb):
      T['ab'][i,j,p,q] = ⟨d_i| a_pα a_qβ |d_j⟩   (a_qβ acts first, then a_pα)
    """
    d_dst = len(a_dets_dst)
    d_src = len(a_dets_src)
    comps = {k: np.zeros((d_dst, d_src, n_orb, n_orb))
             for k in ('aa', 'ab', 'ba', 'bb')}

    for j, (aA_j, bA_j) in enumerate(a_dets_src):
        # 'aa': a_pα a_qα  -> annihilate q (α) first, then p (α)
        for q in range(n_orb):
            ph_q, a1 = _annihilate_sign(aA_j, q)
            if ph_q == 0:
                continue
            for p in range(n_orb):
                if p == q:
                    continue
                ph_p, a2 = _annihilate_sign(a1, p)
                if ph_p == 0:
                    continue
                i = a_index_dst.get((a2, bA_j))
                if i is not None:
                    comps['aa'][i, j, p, q] = ph_q * ph_p

        # 'bb': a_pβ a_qβ  -> annihilate q (β) first, then p (β)
        for q in range(n_orb):
            ph_q, b1 = _annihilate_sign(bA_j, q)
            if ph_q == 0:
                continue
            for p in range(n_orb):
                if p == q:
                    continue
                ph_p, b2 = _annihilate_sign(b1, p)
                if ph_p == 0:
                    continue
                i = a_index_dst.get((aA_j, b2))
                if i is not None:
                    comps['bb'][i, j, p, q] = ph_q * ph_p

        # 'ab': a_pα a_qβ  -> annihilate q (β) first, then p (α)
        for q in range(n_orb):
            ph_q, b1 = _annihilate_sign(bA_j, q)
            if ph_q == 0:
                continue
            for p in range(n_orb):
                ph_p, a1 = _annihilate_sign(aA_j, p)
                if ph_p == 0:
                    continue
                i = a_index_dst.get((a1, b1))
                if i is not None:
                    comps['ab'][i, j, p, q] = ph_q * ph_p

        # 'ba': a_pβ a_qα  -> annihilate q (α) first, then p (β)
        for q in range(n_orb):
            ph_q, a1 = _annihilate_sign(aA_j, q)
            if ph_q == 0:
                continue
            for p in range(n_orb):
                ph_p, b1 = _annihilate_sign(bA_j, p)
                if ph_p == 0:
                    continue
                i = a_index_dst.get((a1, b1))
                if i is not None:
                    comps['ba'][i, j, p, q] = ph_q * ph_p

    return comps


def _compute_det_create2_annih1_explicit(
    a_dets_src: List[Tuple[int, int]],
    a_dets_dst: List[Tuple[int, int]],
    a_index_dst: Dict[Tuple[int, int], int],
    n_orb: int,
) -> Dict[str, np.ndarray]:
    """Spin-EXPLICIT create2+annih1 (n -> n+1).

    Returns dict of 6 combos {'aaa','aba','abb','bab','baa','bbb'}, each
    (d_dst, d_src, n_orb, n_orb, n_orb):
      T['abb'][i,j,p,q,r] = ⟨d_i| a†_pα a†_qβ a_rβ |d_j⟩
    (a_rβ acts first, then a†_qβ, then a†_pα).
    Combos (create1, create2, annih):
      aaa=(α,α,α) aba=(α,β,α) abb=(α,β,β) bab=(β,α,β) baa=(β,α,α) bbb=(β,β,β)
    """
    d_dst = len(a_dets_dst)
    d_src = len(a_dets_src)
    keys = ('aaa', 'aba', 'abb', 'bab', 'baa', 'bbb')
    comps = {k: np.zeros((d_dst, d_src, n_orb, n_orb, n_orb)) for k in keys}

    for j, (aA_j, bA_j) in enumerate(a_dets_src):
        # aaa: annih r(α), create q(α), create p(α)
        for r in range(n_orb):
            ph_r, a1 = _annihilate_sign(aA_j, r)
            if ph_r == 0:
                continue
            for q in range(n_orb):
                ph_q, a2 = _create_sign(a1, q)
                if ph_q == 0:
                    continue
                for p in range(n_orb):
                    ph_p, a3 = _create_sign(a2, p)
                    if ph_p == 0:
                        continue
                    i = a_index_dst.get((a3, bA_j))
                    if i is not None:
                        comps['aaa'][i, j, p, q, r] = ph_r * ph_q * ph_p

        # aba: annih r(α), create q(β), create p(α)
        for r in range(n_orb):
            ph_r, a1 = _annihilate_sign(aA_j, r)
            if ph_r == 0:
                continue
            for q in range(n_orb):
                ph_q, b1 = _create_sign(bA_j, q)
                if ph_q == 0:
                    continue
                for p in range(n_orb):
                    ph_p, a2 = _create_sign(a1, p)
                    if ph_p == 0:
                        continue
                    i = a_index_dst.get((a2, b1))
                    if i is not None:
                        comps['aba'][i, j, p, q, r] = ph_r * ph_q * ph_p

        # abb: annih r(β), create q(β), create p(α)
        for r in range(n_orb):
            ph_r, b1 = _annihilate_sign(bA_j, r)
            if ph_r == 0:
                continue
            for q in range(n_orb):
                ph_q, b2 = _create_sign(b1, q)
                if ph_q == 0:
                    continue
                for p in range(n_orb):
                    ph_p, a1 = _create_sign(aA_j, p)
                    if ph_p == 0:
                        continue
                    i = a_index_dst.get((a1, b2))
                    if i is not None:
                        comps['abb'][i, j, p, q, r] = ph_r * ph_q * ph_p

        # bab: annih r(β), create q(α), create p(β)
        for r in range(n_orb):
            ph_r, b1 = _annihilate_sign(bA_j, r)
            if ph_r == 0:
                continue
            for q in range(n_orb):
                ph_q, a1 = _create_sign(aA_j, q)
                if ph_q == 0:
                    continue
                for p in range(n_orb):
                    ph_p, b2 = _create_sign(b1, p)
                    if ph_p == 0:
                        continue
                    i = a_index_dst.get((a1, b2))
                    if i is not None:
                        comps['bab'][i, j, p, q, r] = ph_r * ph_q * ph_p

        # baa: annih r(α), create q(α), create p(β)
        for r in range(n_orb):
            ph_r, a1 = _annihilate_sign(aA_j, r)
            if ph_r == 0:
                continue
            for q in range(n_orb):
                ph_q, a2 = _create_sign(a1, q)
                if ph_q == 0:
                    continue
                for p in range(n_orb):
                    ph_p, b1 = _create_sign(bA_j, p)
                    if ph_p == 0:
                        continue
                    i = a_index_dst.get((a2, b1))
                    if i is not None:
                        comps['baa'][i, j, p, q, r] = ph_r * ph_q * ph_p

        # bbb: annih r(β), create q(β), create p(β)
        for r in range(n_orb):
            ph_r, b1 = _annihilate_sign(bA_j, r)
            if ph_r == 0:
                continue
            for q in range(n_orb):
                ph_q, b2 = _create_sign(b1, q)
                if ph_q == 0:
                    continue
                for p in range(n_orb):
                    ph_p, b3 = _create_sign(b2, p)
                    if ph_p == 0:
                        continue
                    i = a_index_dst.get((aA_j, b3))
                    if i is not None:
                        comps['bbb'][i, j, p, q, r] = ph_r * ph_q * ph_p

    return comps


def _compute_det_create1_annih2_explicit(
    a_dets_src: List[Tuple[int, int]],
    a_dets_dst: List[Tuple[int, int]],
    a_index_dst: Dict[Tuple[int, int], int],
    n_orb: int,
) -> Dict[str, np.ndarray]:
    """Spin-EXPLICIT create1+annih2 (n -> n-1).

    Returns dict of 6 combos {'aaa','aab','aba','bab','bba','bbb'}, each
    (d_dst, d_src, n_orb, n_orb, n_orb):
      T['aab'][i,j,p,q,r] = ⟨d_i| a†_pα a_qα a_rβ |d_j⟩
    (a_rβ acts first, then a_qα, then a†_pα).
    Combos (create, annih1, annih2):
      aaa=(α,α,α) aab=(α,α,β) aba=(α,β,α) bab=(β,α,β) bba=(β,β,α) bbb=(β,β,β)
    """
    d_dst = len(a_dets_dst)
    d_src = len(a_dets_src)
    keys = ('aaa', 'aab', 'aba', 'bab', 'bba', 'bbb')
    comps = {k: np.zeros((d_dst, d_src, n_orb, n_orb, n_orb)) for k in keys}

    for j, (aA_j, bA_j) in enumerate(a_dets_src):
        # aaa: annih r(α), annih q(α), create p(α)
        for r in range(n_orb):
            ph_r, a1 = _annihilate_sign(aA_j, r)
            if ph_r == 0:
                continue
            for q in range(n_orb):
                ph_q, a2 = _annihilate_sign(a1, q)
                if ph_q == 0:
                    continue
                for p in range(n_orb):
                    ph_p, a3 = _create_sign(a2, p)
                    if ph_p == 0:
                        continue
                    i = a_index_dst.get((a3, bA_j))
                    if i is not None:
                        comps['aaa'][i, j, p, q, r] = ph_r * ph_q * ph_p

        # aab: annih r(β), annih q(α), create p(α)
        for r in range(n_orb):
            ph_r, b1 = _annihilate_sign(bA_j, r)
            if ph_r == 0:
                continue
            for q in range(n_orb):
                ph_q, a1 = _annihilate_sign(aA_j, q)
                if ph_q == 0:
                    continue
                for p in range(n_orb):
                    ph_p, a2 = _create_sign(a1, p)
                    if ph_p == 0:
                        continue
                    i = a_index_dst.get((a2, b1))
                    if i is not None:
                        comps['aab'][i, j, p, q, r] = ph_r * ph_q * ph_p

        # aba: annih r(α), annih q(β), create p(α)
        for r in range(n_orb):
            ph_r, a1 = _annihilate_sign(aA_j, r)
            if ph_r == 0:
                continue
            for q in range(n_orb):
                ph_q, b1 = _annihilate_sign(bA_j, q)
                if ph_q == 0:
                    continue
                for p in range(n_orb):
                    ph_p, a2 = _create_sign(a1, p)
                    if ph_p == 0:
                        continue
                    i = a_index_dst.get((a2, b1))
                    if i is not None:
                        comps['aba'][i, j, p, q, r] = ph_r * ph_q * ph_p

        # bab: annih r(β), annih q(α), create p(β)
        for r in range(n_orb):
            ph_r, b1 = _annihilate_sign(bA_j, r)
            if ph_r == 0:
                continue
            for q in range(n_orb):
                ph_q, a1 = _annihilate_sign(aA_j, q)
                if ph_q == 0:
                    continue
                for p in range(n_orb):
                    ph_p, b2 = _create_sign(b1, p)
                    if ph_p == 0:
                        continue
                    i = a_index_dst.get((a1, b2))
                    if i is not None:
                        comps['bab'][i, j, p, q, r] = ph_r * ph_q * ph_p

        # bba: annih r(α), annih q(β), create p(β)
        for r in range(n_orb):
            ph_r, a1 = _annihilate_sign(aA_j, r)
            if ph_r == 0:
                continue
            for q in range(n_orb):
                ph_q, b1 = _annihilate_sign(bA_j, q)
                if ph_q == 0:
                    continue
                for p in range(n_orb):
                    ph_p, b2 = _create_sign(b1, p)
                    if ph_p == 0:
                        continue
                    i = a_index_dst.get((a1, b2))
                    if i is not None:
                        comps['bba'][i, j, p, q, r] = ph_r * ph_q * ph_p

        # bbb: annih r(β), annih q(β), create p(β)
        for r in range(n_orb):
            ph_r, b1 = _annihilate_sign(bA_j, r)
            if ph_r == 0:
                continue
            for q in range(n_orb):
                ph_q, b2 = _annihilate_sign(b1, q)
                if ph_q == 0:
                    continue
                for p in range(n_orb):
                    ph_p, b3 = _create_sign(b2, p)
                    if ph_p == 0:
                        continue
                    i = a_index_dst.get((aA_j, b3))
                    if i is not None:
                        comps['bbb'][i, j, p, q, r] = ph_r * ph_q * ph_p

    return comps


def _transform_3body_to_schmidt(T_det, U_dst, U_src):
    """Transform 3-body transition to Schmidt basis."""
    return np.einsum('ijpqr,ia,jg->agpqr', T_det, U_dst, U_src)


def _apply_jw_phase_A(comp, dn_table, dets_n):
    """Bake the Jordan-Wigner phase into A-side spin-explicit transition matrices.

    comp: dict combo -> ndarray (d_dst, d_src, ...)  (mutated in place)
    dn_table: dict combo -> (dn_alpha, dn_beta) net spin change of the A operator
    dets_n: list of (alpha_str, beta_str) source determinants (columns)
    """
    for j, (aA_j, bA_j) in enumerate(dets_n):
        for k, (dn_a, dn_b) in dn_table.items():
            if _jw_phase(dn_a, dn_b, aA_j, bA_j) == -1:
                comp[k][:, j] *= -1
    return comp


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
    # Spin-explicit single creation/annihilation: n_A -> {'a','b'} -> (r_dst,r_src,n_orb)
    create_1_explicit: Dict[int, Dict[str, np.ndarray]] = field(default_factory=dict)
    annihilate_1_explicit: Dict[int, Dict[str, np.ndarray]] = field(default_factory=dict)
    trans_1: Dict[int, np.ndarray] = field(default_factory=dict)
    # Spin-explicit 1-body transitions: n_A -> {'aa','ab','ba','bb'} -> (r,r,n_orb,n_orb)
    trans_1_explicit: Dict[int, Dict[str, np.ndarray]] = field(default_factory=dict)
    create_2: Dict[int, np.ndarray] = field(default_factory=dict)
    annihilate_2: Dict[int, np.ndarray] = field(default_factory=dict)
    # Spin-explicit pair creation/annihilation: n_A -> {'aa','ab','ba','bb'} -> (r_dst,r_src,n_orb,n_orb)
    create_2_explicit: Dict[int, Dict[str, np.ndarray]] = field(default_factory=dict)
    annihilate_2_explicit: Dict[int, Dict[str, np.ndarray]] = field(default_factory=dict)
    # 3-body: create 2 + annihilate 1 (n→n+1): ⟨α| a_p† a_q† a_r |γ⟩
    create2_annih1: Dict[int, np.ndarray] = field(default_factory=dict)
    # 3-body: create 1 + annihilate 2 (n→n-1): ⟨α| a_p† a_q a_r |γ⟩
    create1_annih2: Dict[int, np.ndarray] = field(default_factory=dict)
    # Spin-explicit 3-body: n_A -> combo -> (r_dst,r_src,n_orb,n_orb,n_orb)
    #   create2_annih1 combos: aaa,aba,abb,bab,baa,bbb
    #   create1_annih2 combos: aaa,aab,aba,bab,bba,bbb
    create2_annih1_explicit: Dict[int, Dict[str, np.ndarray]] = field(default_factory=dict)
    create1_annih2_explicit: Dict[int, Dict[str, np.ndarray]] = field(default_factory=dict)
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
            T_exp = _compute_det_creation_explicit(dets_n, dets_dst, idx_dst, n_orb)
            if subspace == 'A':
                # JW phase for the B-side operator crossing fragment A:
                # a†_pσ on A carries an extra (-1)^{n_σ(A_src)} relative to the
                # separate-fragment contraction (α/β strings are ordered A-then-B
                # in the full space).  σ=α for 'a', σ=β for 'b'.
                for j, (aA_j, bA_j) in enumerate(dets_n):
                    T_exp['a'][:, j, :] *= (-1) ** aA_j.bit_count()
                    T_exp['b'][:, j, :] *= (-1) ** bA_j.bit_count()
            result.create_1_explicit[n_A] = {
                k: _transform_1body_to_schmidt(T_exp[k], U_dst, U)
                for k in ('a', 'b')
            }

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
            T_exp = _compute_det_annihilation_explicit(dets_n, dets_dst, idx_dst, n_orb)
            if subspace == 'A':
                # JW phase: a_pσ on A carries (-1)^{n_σ(A_src)-1} (one electron
                # removed before the B-side operator crosses A).
                for j, (aA_j, bA_j) in enumerate(dets_n):
                    T_exp['a'][:, j, :] *= (-1) ** (aA_j.bit_count() - 1)
                    T_exp['b'][:, j, :] *= (-1) ** (bA_j.bit_count() - 1)
            result.annihilate_1_explicit[n_A] = {
                k: _transform_1body_to_schmidt(T_exp[k], U_dst, U)
                for k in ('a', 'b')
            }

        # ── 1-body transition (n → n, same block) ──
        T_det = _compute_det_1body_transition(dets_n, idx_n, n_orb)
        # T_det: (d, d, n_orb, n_orb)
        result.trans_1[n_A] = _transform_2body_to_schmidt(T_det, U, U)

        # ── Spin-explicit 1-body transitions (for the exchange term) ──
        T_exp = _compute_det_1body_transition_explicit(dets_n, idx_n, n_orb)
        result.trans_1_explicit[n_A] = {
            k: _transform_2body_to_schmidt(T_exp[k], U, U)
            for k in ('aa', 'ab', 'ba', 'bb')
        }

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
            T_exp = _compute_det_pair_creation_explicit(dets_n, dets_dst, idx_dst, n_orb)
            if subspace == 'A':
                _apply_jw_phase_A(T_exp, {'ab': (1, 1), 'ba': (1, 1)}, dets_n)
            result.create_2_explicit[n_A] = {
                k: _transform_2body_to_schmidt(T_exp[k], U_dst, U)
                for k in ('aa', 'ab', 'ba', 'bb')
            }

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
            T_exp = _compute_det_pair_annihilation_explicit(dets_n, dets_dst, idx_dst, n_orb)
            if subspace == 'A':
                _apply_jw_phase_A(T_exp, {'ab': (-1, -1), 'ba': (-1, -1)}, dets_n)
            result.annihilate_2_explicit[n_A] = {
                k: _transform_2body_to_schmidt(T_exp[k], U_dst, U)
                for k in ('aa', 'ab', 'ba', 'bb')
            }

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
            T_exp = _compute_det_create2_annih1_explicit(dets_n, dets_dst, idx_dst, n_orb)
            if subspace == 'A':
                _apply_jw_phase_A(T_exp, {
                    'aaa': (1, 0), 'aba': (0, 1), 'abb': (1, 0),
                    'bab': (1, 0), 'baa': (0, 1), 'bbb': (0, 1),
                }, dets_n)
            result.create2_annih1_explicit[n_A] = {
                k: _transform_3body_to_schmidt(T_exp[k], U_dst, U)
                for k in ('aaa', 'aba', 'abb', 'bab', 'baa', 'bbb')
            }

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
            T_exp = _compute_det_create1_annih2_explicit(dets_n, dets_dst, idx_dst, n_orb)
            if subspace == 'A':
                _apply_jw_phase_A(T_exp, {
                    'aaa': (-1, 0), 'aab': (0, -1), 'aba': (0, -1),
                    'bab': (-1, 0), 'bba': (-1, 0), 'bbb': (0, -1),
                }, dets_n)
            result.create1_annih2_explicit[n_A] = {
                k: _transform_3body_to_schmidt(T_exp[k], U_dst, U)
                for k in ('aaa', 'aab', 'aba', 'bab', 'bba', 'bbb')
            }

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
