#!/usr/bin/env python3
"""
H_AB construction via DMRG-style complementary operator decomposition.

Based on block2's Normal/Complementary partitioning (Chan et al., JCP 2016):
  H = H^A ⊗ 1^B + 1^A ⊗ H^B + Σ_α O_A^α ⊗ O_B^α

where O_B^α are complementary operators with integrals pre-contracted.
This avoids the non-factorizable 4-operator product problem that
plagues naive TA×TB factorization.

References:
  - block2 docs: Normal/Complementary Partitioning
  - Chan, Keselman, Nakatani, Li, White, JCP 145, 014102 (2016)
"""

import numpy as np
from typing import Dict
import time


# ═══════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════

def build_hab_rdm(
    H_AB: np.ndarray,
    schmidt_data: Dict[int, Dict],
    block_offsets: Dict[int, int],
    trans_A,
    trans_B,
    h1_full: np.ndarray,
    h2_full: np.ndarray,
    n_occ: int,
    n_act: int,
    verbose: bool = True,
):
    """Build H_AB matrix via DMRG complementary operator contraction."""
    n_virt = n_act - n_occ
    all_blocks = sorted(schmidt_data.keys())

    if verbose:
        t0 = time.perf_counter()

    # ── 1e cross terms ──
    for n_A in all_blocks:
        _add_1e_cross_block_rdm(
            H_AB, block_offsets, n_A, trans_A, trans_B,
            h1_full, n_occ, n_act, n_virt
        )

    # ── 2e n_A-conserved: complementary operator approach ──
    # Term (b): Σ_{ij∈A} B_{ij} ⊗ Q_{ij}^B  (same-spin, dominant)
    # Term (c): -Σ_{il∈A} B_{il} ⊗ Q'^B_{il} (cross-spin, same-spin part)
    for n_A in all_blocks:
        sd = schmidt_data[n_A]
        if sd['r'] == 0:
            continue
        os = block_offsets.get(n_A)
        if os is None:
            continue
        _add_nconserved_complementary(
            H_AB, os, n_A, trans_A, trans_B,
            h2_full, n_occ, n_act, n_virt
        )

    # ── 2e pair transfer: A_{ik} ⊗ P_{ik}^B + h.c. ──
    for n_A in all_blocks:
        sd = schmidt_data[n_A]
        if sd['r'] == 0:
            continue
        _add_pair_transfer_complementary(
            H_AB, block_offsets, n_A, trans_A, trans_B,
            h2_full, n_occ, n_act, n_virt
        )

    # ── 2e: 3-body terms (3A+1B and 1A+3B) ──
    for n_A in all_blocks:
        _add_3body_patterns(
            H_AB, block_offsets, n_A, trans_A, trans_B,
            h2_full, n_occ, n_act, n_virt
        )
        _add_1a3b_patterns(
            H_AB, block_offsets, n_A, trans_A, trans_B,
            h2_full, n_occ, n_act, n_virt
        )

    if verbose:
        elapsed = time.perf_counter() - t0
        print(f"    H_AB comp-op: {elapsed:.1f}s, "
              f"||H_AB||={np.linalg.norm(H_AB):.4f}")


# ═══════════════════════════════════════════════════════════════════════════
# 2e n_A-conserved: complementary operator approach
# ═══════════════════════════════════════════════════════════════════════════

def _add_nconserved_complementary(
    H_AB, offset, n_A, trans_A, trans_B,
    h2_full, n_occ, n_act, n_virt
):
    """n_A-conserved 2e terms (block2 normal/complementary, Chan et al. 2016).

    Same-spin (direct), coefficient +1:
        + Σ_{ij∈A} B_ij ⊗ Q^B_ij
          B_ij = Σ_σ a†_iσ a_jσ,   Q^B_ij = Σ_{kl∈B,σ'} v_ijkl a†_kσ' a_lσ'

    Exchange (complementary) term, spin-explicit:
        − Σ_{il∈A} Σ_{σσ'} B'_il,σσ' ⊗ Q'^B_il,σσ'
          B'_il,σσ' = a†_iσ a_lσ',   Q'^B_il,σσ' = Σ_{jk∈B} v_ijkl a†_kσ' a_jσ

    The direct term uses spin-SUMMED transitions (trans_1).  The exchange term
    uses spin-EXPLICIT transitions (trans_1_explicit) because its A-side
    operator a†_iσ a_lσ' mixes spins (σ ≠ σ').

    SIGN STRUCTURE: the −1 coefficient applies only to the SAME-SPIN (σ = σ')
    exchange, which is the genuine Fermi-exchange contribution.  The CROSS-SPIN
    (σ ≠ σ') terms carry +1 — they are the spin-flip Coulomb terms, not Fermi
    exchange, so the −1 does NOT apply.  (Verified numerically: flipping the
    cross-spin sign fixes the odd-n_A diagonal blocks of H^emb to machine
    precision against the Slater-Condon reference.)
    """
    TA = trans_A.trans_1.get(n_A)
    TB = trans_B.trans_1.get(n_A)
    if TA is None or TB is None:
        return

    r, _, nA_orb, _ = TA.shape

    # ── Term (b): same-spin (direct), spin-summed, coefficient +1 ──
    # Q[i,j,b_dst,b_src] = Σ_{k,l∈B} h2_full[i,j,k_B,l_B] × TB[b_dst,b_src,k,l]
    Q = np.zeros((n_occ, n_occ, r, r))
    for i in range(n_occ):
        for j in range(n_occ):
            for k in range(n_virt):
                for l in range(n_virt):
                    v = h2_full[i, j, k + n_occ, l + n_occ]
                    if abs(v) < 1e-14:
                        continue
                    Q[i, j] += v * TB[:, :, k, l]

    _contract_A_tensor_B(H_AB, offset, r, TA, Q, n_occ, sign=+1.0)

    # ── Term (c): exchange (complementary), spin-explicit ──
    TAe = trans_A.trans_1_explicit.get(n_A)
    TBe = trans_B.trans_1_explicit.get(n_A)
    if TAe is None or TBe is None:
        return

    # Spin pairing (A-side create σ / annihilate σ', B-side create σ' / annihilate σ):
    #   A 'aa' ↔ B 'aa', A 'bb' ↔ B 'bb', A 'ab' ↔ B 'ba', A 'ba' ↔ B 'ab'
    # Sign: −1 for same-spin (Fermi exchange).  The cross-spin (σ ≠ σ') terms
    # carry an extra JW phase (−1)^{n_A} from the B-side spin-flip operator
    # crossing the n_A electrons of fragment A, giving overall (−1)^{n_A+1}.
    spin_pairs = [('aa', 'aa'), ('bb', 'bb'), ('ab', 'ba'), ('ba', 'ab')]
    cross_sign = -1.0 if (n_A % 2 == 0) else +1.0
    spin_sign = {('aa', 'aa'): -1.0, ('bb', 'bb'): -1.0,
                 ('ab', 'ba'): cross_sign, ('ba', 'ab'): cross_sign}

    # Pre-contract the integral-weighted B-side operator per spin pair:
    #   Qp[sA][i, l, b_dst, b_src] = Σ_{j,k∈B} h2_full[i,j_B,k_B,l] · TBe[sB][b_dst,b_src,k,j]
    Qp = {sA: np.zeros((n_occ, n_occ, r, r)) for sA, _ in spin_pairs}
    for i in range(n_occ):
        for l in range(n_occ):
            for j in range(n_virt):
                for k in range(n_virt):
                    v = h2_full[i, j + n_occ, k + n_occ, l]
                    if abs(v) < 1e-14:
                        continue
                    for sA, sB in spin_pairs:
                        Qp[sA][i, l] += v * TBe[sB][:, :, k, j]

    # Contract A-side: H_AB[αβ,γδ] += sign × Σ_{i,l} TAe[sA][α,γ,i,l] · Qp[sA][i,l,β,δ]
    for sA, sB in spin_pairs:
        _contract_A_tensor_B(H_AB, offset, r, TAe[sA], Qp[sA], n_occ,
                             sign=spin_sign[(sA, sB)])


def _contract_A_tensor_B(H_AB, offset, r, TA, QB, n_indices, sign):
    """Contract H_AB[αβ,γδ] += sign × Σ_{a,b} TA[α,γ,a,b] × QB[a,b,β,δ].

    TA: (r, r, n, n) — A-space 1-body transition matrix.
    QB: (n, n, r, r) — B-space complementary operator (integral-weighted).
    """
    for a_dst in range(r):
        for a_src in range(r):
            for b_dst in range(r):
                for b_src in range(r):
                    val = 0.0
                    for p in range(n_indices):
                        for q in range(n_indices):
                            ta = TA[a_dst, a_src, p, q]
                            if abs(ta) < 1e-14:
                                continue
                            val += ta * QB[p, q, b_dst, b_src]
                    if abs(val) > 1e-14:
                        s = offset + a_src * r + b_src
                        d = offset + a_dst * r + b_dst
                        H_AB[d, s] += sign * val


# ═══════════════════════════════════════════════════════════════════════════
# 2e pair transfer: complementary operator approach
# ═══════════════════════════════════════════════════════════════════════════

def _add_pair_transfer_complementary(
    H_AB, block_offsets, n_A, trans_A, trans_B,
    h2_full, n_occ, n_act, n_virt
):
    """2e pair transfer (2A+2B, n_A → n_A ± 2), spin-explicit.

    n_A → n_A+2:  ½ Σ_{ik∈A} Σ_{jl∈B} Σ_{στ} (ik|jl) c2_A[σ,τ][i,k] · a2_B[τ,σ][l,j]
    n_A → n_A-2:  ½ Σ_{jl∈B} Σ_{ik∈A} Σ_{στ} (jl|ik) a2_A[τ,σ][k,i] · c2_B[σ,τ][j,l]

    Spin pairing A[s1,s2] ↔ B[s2,s1]: aa↔aa, ab↔ba, ba↔ab, bb↔bb.
    """
    spin_pairs = [('aa', 'aa'), ('ab', 'ba'), ('ba', 'ab'), ('bb', 'bb')]

    # ---- n_A → n_A + 2 : A creates 2, B annihilates 2 ----
    c2_A = trans_A.create_2_explicit.get(n_A)
    a2_B = trans_B.annihilate_2_explicit.get(n_A)
    os = block_offsets.get(n_A)
    od = block_offsets.get(n_A + 2)
    if c2_A is not None and a2_B is not None and os is not None and od is not None:
        r_src = c2_A['aa'].shape[1]
        r_dst = c2_A['aa'].shape[0]
        for sA, sB in spin_pairs:
            TA = c2_A[sA]
            TB = a2_B[sB]
            rB_dst, rB_src = TB.shape[0], TB.shape[1]
            # P[i,k,b_dst,b_src] = Σ_{j,l∈B} (ik|jl) · TB[b_dst,b_src,l,j]
            P = np.zeros((n_occ, n_occ, rB_dst, rB_src))
            for i in range(n_occ):
                for k in range(n_occ):
                    for j in range(n_virt):
                        for l in range(n_virt):
                            v = h2_full[i, j + n_occ, k, l + n_occ]
                            if abs(v) < 1e-14:
                                continue
                            P[i, k] += v * TB[:, :, l, j]
            _contract_pair_block(H_AB, os, od, TA, P, r_src, r_dst, 0.5)

    # ---- n_A → n_A - 2 : A annihilates 2, B creates 2 ----
    a2_A = trans_A.annihilate_2_explicit.get(n_A)
    c2_B = trans_B.create_2_explicit.get(n_A)
    os2 = block_offsets.get(n_A)
    od2 = block_offsets.get(n_A - 2)
    if a2_A is not None and c2_B is not None and os2 is not None and od2 is not None:
        r_src = a2_A['aa'].shape[1]
        r_dst = a2_A['aa'].shape[0]
        for sA, sB in spin_pairs:
            TA = a2_A[sA]   # [a_dst, a_src, k, i]
            TB = c2_B[sB]   # [b_dst, b_src, j, l]
            rB_dst, rB_src = TB.shape[0], TB.shape[1]
            # P[k,i,b_dst,b_src] = Σ_{j,l∈B} (jl|ik) · TB[b_dst,b_src,j,l]
            P = np.zeros((n_occ, n_occ, rB_dst, rB_src))
            for i in range(n_occ):
                for k in range(n_occ):
                    for j in range(n_virt):
                        for l in range(n_virt):
                            v = h2_full[j + n_occ, i, l + n_occ, k]
                            if abs(v) < 1e-14:
                                continue
                            P[k, i] += v * TB[:, :, j, l]
            _contract_pair_block(H_AB, os2, od2, TA, P, r_src, r_dst, 0.5)


def _contract_pair_block(H_AB, os, od, TA, P, r_src, r_dst, sign):
    """H_AB[od+a_dst*r_dst+b_dst, os+a_src*r_src+b_src] += sign·Σ_{i,k} TA[a_dst,a_src,i,k]·P[i,k,b_dst,b_src]."""
    rA_dst, rA_src = TA.shape[0], TA.shape[1]
    n_i, n_k = P.shape[0], P.shape[1]
    rB_dst, rB_src = P.shape[2], P.shape[3]
    for a_dst in range(rA_dst):
        for a_src in range(rA_src):
            for b_dst in range(rB_dst):
                for b_src in range(rB_src):
                    val = 0.0
                    for i in range(n_i):
                        for k in range(n_k):
                            ta = TA[a_dst, a_src, i, k]
                            if abs(ta) < 1e-14:
                                continue
                            val += ta * P[i, k, b_dst, b_src]
                    if abs(val) > 1e-14:
                        H_AB[od + a_dst * r_dst + b_dst,
                              os + a_src * r_src + b_src] += sign * val


# ═══════════════════════════════════════════════════════════════════════════
# 3A+1B and 1A+3B (kept from original — minor contributions)
# ═══════════════════════════════════════════════════════════════════════════

def _add_3body_patterns(
    H_AB, block_offsets, n_A,
    trans_A, trans_B, h2_full, n_occ, n_act, n_virt
):
    """3A+1B (n_A→n_A±1): A-side 3-body op, B-side single op (spin-explicit)."""
    # n_A → n_A+1 : A create2_annih1, B annihilate_1
    TA3 = trans_A.create2_annih1_explicit.get(n_A)
    TB1 = trans_B.annihilate_1_explicit.get(n_A)
    os = block_offsets.get(n_A)
    od = block_offsets.get(n_A + 1)
    if TA3 is not None and TB1 is not None and os is not None and od is not None:
        r_src = TA3['aaa'].shape[1]
        r_dst = TA3['aaa'].shape[0]
        _contract_3A1B(H_AB, os, od, TA3, TB1, r_src, r_dst, h2_full, n_occ, n_virt,
                       [('aaa', 'a'), ('abb', 'a'), ('baa', 'b'), ('bbb', 'b')], 'rB')
        _contract_3A1B(H_AB, os, od, TA3, TB1, r_src, r_dst, h2_full, n_occ, n_virt,
                       [('aaa', 'a'), ('aba', 'b'), ('bab', 'a'), ('bbb', 'b')], 'sB')

    # n_A → n_A-1 : A create1_annih2, B create_1
    TA3b = trans_A.create1_annih2_explicit.get(n_A)
    TB1b = trans_B.create_1_explicit.get(n_A)
    os2 = block_offsets.get(n_A)
    od2 = block_offsets.get(n_A - 1)
    if TA3b is not None and TB1b is not None and os2 is not None and od2 is not None:
        r_src = TA3b['aaa'].shape[1]
        r_dst = TA3b['aaa'].shape[0]
        _contract_3A1B(H_AB, os2, od2, TA3b, TB1b, r_src, r_dst, h2_full, n_occ, n_virt,
                       [('aaa', 'a'), ('bba', 'a'), ('aab', 'b'), ('bbb', 'b')], 'pB')
        _contract_3A1B(H_AB, os2, od2, TA3b, TB1b, r_src, r_dst, h2_full, n_occ, n_virt,
                       [('aaa', 'a'), ('aba', 'b'), ('bab', 'a'), ('bbb', 'b')], 'qB')


def _add_1a3b_patterns(
    H_AB, block_offsets, n_A,
    trans_A, trans_B, h2_full, n_occ, n_act, n_virt
):
    """1A+3B (n_A→n_A±1): A-side single op, B-side 3-body op (spin-explicit)."""
    # n_A → n_A+1 : A create_1, B create1_annih2
    cre_A = trans_A.create_1_explicit.get(n_A)
    c1a2_B = trans_B.create1_annih2_explicit.get(n_A)
    os = block_offsets.get(n_A)
    od = block_offsets.get(n_A + 1)
    if cre_A is not None and c1a2_B is not None and os is not None and od is not None:
        r_src = cre_A['a'].shape[1]
        r_dst = cre_A['a'].shape[0]
        _contract_1A3B(H_AB, os, od, cre_A, c1a2_B, r_src, r_dst, h2_full, n_occ, n_virt,
                       [('a', 'aaa'), ('a', 'bba'), ('b', 'aab'), ('b', 'bbb')], 'pA')
        _contract_1A3B(H_AB, os, od, cre_A, c1a2_B, r_src, r_dst, h2_full, n_occ, n_virt,
                       [('a', 'aaa'), ('b', 'aba'), ('a', 'bab'), ('b', 'bbb')], 'qA')

    # n_A → n_A-1 : A annihilate_1, B create2_annih1
    ann_A = trans_A.annihilate_1_explicit.get(n_A)
    c2a1_B = trans_B.create2_annih1_explicit.get(n_A)
    os2 = block_offsets.get(n_A)
    od2 = block_offsets.get(n_A - 1)
    if ann_A is not None and c2a1_B is not None and os2 is not None and od2 is not None:
        r_src = ann_A['a'].shape[1]
        r_dst = ann_A['a'].shape[0]
        _contract_1A3B(H_AB, os2, od2, ann_A, c2a1_B, r_src, r_dst, h2_full, n_occ, n_virt,
                       [('a', 'aaa'), ('a', 'abb'), ('b', 'baa'), ('b', 'bbb')], 'rA')
        _contract_1A3B(H_AB, os2, od2, ann_A, c2a1_B, r_src, r_dst, h2_full, n_occ, n_virt,
                       [('a', 'aaa'), ('b', 'aba'), ('a', 'bab'), ('b', 'bbb')], 'sA')


# ═══════════════════════════════════════════════════════════════════════════
# Low-level contraction helpers
# ═══════════════════════════════════════════════════════════════════════════

def _contract_3A1B(H_AB, os, od, TA3, TB1, r_src, r_dst, h2_full, n_occ, n_virt,
                   pairs, integ_kind):
    """Contract 3A+1B: A-side 3-body (TA3: combo→5D), B-side single (TB1: spin→3D).

    integ_kind selects the B-orbital slot in (p,q,r,s):
      'rB': B=3rd slot, A indices (p,q,s)  — c2a1, n→n+1
      'sB': B=4th slot, A indices (p,q,r)  — c2a1, n→n+1
      'pB': B=1st slot, A indices (q,s,r)  — c1a2, n→n-1
      'qB': B=2nd slot, A indices (p,s,r)  — c1a2, n→n-1
    """
    rA_dst, rA_src = TA3['aaa'].shape[0], TA3['aaa'].shape[1]
    rB_dst, rB_src = TB1['a'].shape[0], TB1['a'].shape[1]
    for combo, spin in pairs:
        TA = TA3[combo]
        TB = TB1[spin]
        # Same-spin combos ('aaa','bbb') in the 'sB' sub-case (c2a1, B annihilate SECOND):
        # the B operator crosses one fewer A electron because the A-side annihilated the
        # SAME spin first.  Extra (-1) only for this sub-case.
        extra = -1.0 if (combo in ('aaa', 'bbb') and integ_kind == 'sB') else 1.0
        P = np.zeros((n_occ, n_occ, n_occ, rB_dst, rB_src))
        for x in range(n_occ):
            for y in range(n_occ):
                for z in range(n_occ):
                    for b in range(n_virt):
                        if integ_kind == 'rB':
                            v = h2_full[x, b + n_occ, y, z]
                        elif integ_kind == 'sB':
                            v = h2_full[x, z, y, b + n_occ]
                        elif integ_kind == 'pB':
                            v = h2_full[b + n_occ, z, x, y]
                        else:  # 'qB'
                            v = h2_full[x, z, b + n_occ, y]
                        if abs(v) < 1e-14:
                            continue
                        P[x, y, z] += v * TB[:, :, b]
        for a_dst in range(rA_dst):
            for a_src in range(rA_src):
                for b_dst in range(rB_dst):
                    for b_src in range(rB_src):
                        val = 0.0
                        for x in range(n_occ):
                            for y in range(n_occ):
                                for z in range(n_occ):
                                    ta = TA[a_dst, a_src, x, y, z]
                                    if abs(ta) < 1e-14:
                                        continue
                                    val += ta * P[x, y, z, b_dst, b_src]
                        if abs(val) > 1e-14:
                            H_AB[od + a_dst * r_dst + b_dst,
                                  os + a_src * r_src + b_src] += 0.5 * extra * val


def _contract_1A3B(H_AB, os, od, TA1, TB3, r_src, r_dst, h2_full, n_occ, n_virt,
                   pairs, integ_kind):
    """Contract 1A+3B: A-side single (TA1: spin→3D), B-side 3-body (TB3: combo→5D).

    integ_kind selects the A-orbital slot in (p,q,r,s):
      'pA': A=1st slot, B indices (q,s,r)  — c1a2, n→n+1
      'qA': A=2nd slot, B indices (p,s,r)  — c1a2, n→n+1
      'rA': A=3rd slot, B indices (p,q,s)  — c2a1, n→n-1
      'sA': A=4th slot, B indices (p,q,r)  — c2a1, n→n-1
    """
    rA_dst, rA_src = TA1['a'].shape[0], TA1['a'].shape[1]
    rB_dst, rB_src = TB3['aaa'].shape[0], TB3['aaa'].shape[1]
    for spin, combo in pairs:
        TA = TA1[spin]
        TB = TB3[combo]
        P = np.zeros((n_virt, n_virt, n_virt, rB_dst, rB_src))
        for x in range(n_virt):
            for y in range(n_virt):
                for z in range(n_virt):
                    for a in range(n_occ):
                        if integ_kind == 'pA':
                            v = h2_full[a, z + n_occ, x + n_occ, y + n_occ]
                        elif integ_kind == 'qA':
                            v = h2_full[x + n_occ, z + n_occ, a, y + n_occ]
                        elif integ_kind == 'rA':
                            v = h2_full[x + n_occ, a, y + n_occ, z + n_occ]
                        else:  # 'sA'
                            v = h2_full[x + n_occ, z + n_occ, y + n_occ, a]
                        if abs(v) < 1e-14:
                            continue
                        P[x, y, z] += v * TA[:, :, a]
        for a_dst in range(rA_dst):
            for a_src in range(rA_src):
                for b_dst in range(rB_dst):
                    for b_src in range(rB_src):
                        val = 0.0
                        for x in range(n_virt):
                            for y in range(n_virt):
                                for z in range(n_virt):
                                    tb = TB[b_dst, b_src, x, y, z]
                                    if abs(tb) < 1e-14:
                                        continue
                                    val += tb * P[x, y, z, b_dst, b_src]
                        if abs(val) > 1e-14:
                            H_AB[od + a_dst * r_dst + b_dst,
                                  os + a_src * r_src + b_src] += 0.5 * val


# ═══════════════════════════════════════════════════════════════════════════
# 1e cross terms (unchanged from original — verified correct)
# ═══════════════════════════════════════════════════════════════════════════

def _add_1e_cross_block_rdm(
    H_AB, block_offsets, n_A,
    trans_A, trans_B, h1_full, n_occ, n_act, n_virt
):
    """1e cross: h_pr a_p†(A) a_r(B) + h.c.  (SPIN-DIAGONAL).

    H_1e = Σ_{pq} h_pq Σ_σ a†_pσ a_qσ.  The cross part (p∈A, r∈B) is
      Σ_{p∈A,r∈B,σ} h_pr a†_pσ a_rσ  +  h.c.
    which is spin-diagonal (σ on both create and annihilate).  A spin-SUMMED
    contraction (Σ_σ a†_pσ)(Σ_σ' a_rσ') would spuriously include σ≠σ' terms, so
    we use the spin-explicit transitions and contract same-spin only.
    """
    cre_A = trans_A.create_1_explicit.get(n_A)
    ann_B = trans_B.annihilate_1_explicit.get(n_A)
    if cre_A is not None and ann_B is not None:
        _add_1e_cross_pair(H_AB, block_offsets, n_A, n_A + 1,
                           cre_A, ann_B, h1_full, n_occ, rev_B=False)

    ann_A = trans_A.annihilate_1_explicit.get(n_A)
    cre_B = trans_B.create_1_explicit.get(n_A)
    if ann_A is not None and cre_B is not None:
        _add_1e_cross_pair(H_AB, block_offsets, n_A, n_A - 1,
                           ann_A, cre_B, h1_full, n_occ, rev_B=True)


def _add_1e_cross_pair(H_AB, block_offsets, n_A_src, n_A_dst,
                       TA, TB, h1_full, n_occ, rev_B=False):
    """Add the 1e cross contribution with spin-explicit transitions.

    TA, TB: dict {'a','b'} of spin-explicit single (create/annihilate)
    transitions in Schmidt basis, shape (r_dst, r_src, n_orb).

    rev_B=False: h_{pr} a_p†(A) a_r(B), JW = (-1)^{n_A}
    rev_B=True:  h_{rp} a_r†(B) a_p(A), JW = (-1)^{n_A-1}
    """
    os = block_offsets.get(n_A_src)
    od = block_offsets.get(n_A_dst)
    if os is None or od is None:
        return

    r_dA, r_sA = TA['a'].shape[0], TA['a'].shape[1]
    r_dB, r_sB = TB['a'].shape[0], TB['a'].shape[1]

    # JW sign is now baked into the A-side transition matrices
    # (spin-dependent (-1)^{n_σ(A_src)} / (-1)^{n_σ(A_src)-1} in
    # transition_rdm.py).  No extra factor here.
    jw = 1

    for a_dst in range(r_dA):
        for a_src in range(r_sA):
            for b_dst in range(r_dB):
                for b_src in range(r_sB):
                    val = 0.0
                    for p in range(n_occ):
                        for r_sub in range(TB['a'].shape[2]):
                            r_full = r_sub + n_occ
                            if rev_B:
                                h = h1_full[r_full, p]
                            else:
                                h = h1_full[p, r_full]
                            if abs(h) < 1e-15:
                                continue
                            # same-spin contraction (αα + ββ), no σ≠σ' terms
                            s = (TA['a'][a_dst, a_src, p] * TB['a'][b_dst, b_src, r_sub]
                                 + TA['b'][a_dst, a_src, p] * TB['b'][b_dst, b_src, r_sub])
                            val += h * s
                    if abs(val) > 1e-15:
                        s_idx = os + a_src * r_sA + b_src
                        d_idx = od + a_dst * r_dA + b_dst
                        H_AB[d_idx, s_idx] += jw * val
