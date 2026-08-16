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
    """Term (a): ½ Σ_{ik∈A} A_{ik} ⊗ P_{ik}^B + h.c.

    A_{ik} = a_i† a_k† (pair creation on A)
    P_{ik}^B = Σ_{jl∈B} v_{ijkl} a_l a_j (pair annihilation on B, weighted)

    n_A → n_A+2: TA.create_2 × P_B (direct)
    n_A → n_A-2: TA.annihilate_2 × P_B† (Hermitian conjugate)
    """
    # n_A → n_A+2
    TA_cre = trans_A.create_2.get(n_A)
    TB_ann = trans_B.annihilate_2.get(n_A)
    sd_src = block_offsets.get(n_A)
    sd_dst = block_offsets.get(n_A + 2)
    if TA_cre is not None and TB_ann is not None and sd_src is not None and sd_dst is not None:
        r_dst_A, r_src_A, nA_orb, _ = TA_cre.shape
        r_dst_B, r_src_B = TB_ann.shape[0], TB_ann.shape[1]

        # P[i,k,b_dst,b_src] = Σ_{j,l∈B} h2_full[i,j_B,k,l_B] × TB_ann[b_dst,b_src,j,l]
        P = np.zeros((n_occ, n_occ, r_dst_B, r_src_B))
        for i in range(n_occ):
            for k in range(n_occ):
                for j in range(n_virt):
                    for l_idx in range(n_virt):
                        v = h2_full[i, j + n_occ, k, l_idx + n_occ]
                        if abs(v) < 1e-14:
                            continue
                        # TB_ann[b_dst,b_src,j,l] = ⟨b_dst| a_l a_j |b_src⟩
                        P[i, k] += 0.5 * v * TB_ann[:, :, j, l_idx]

        _contract_pair(H_AB, sd_src, sd_dst,
                       r_dst_A, r_src_A, r_dst_B, r_src_B,
                       TA_cre, P, n_occ)

    # n_A → n_A-2 (Hermitian conjugate: TA.annihilate_2 × P_B†)
    TA_ann = trans_A.annihilate_2.get(n_A)
    TB_cre = trans_B.create_2.get(n_A)
    sd_src2 = block_offsets.get(n_A)
    sd_dst2 = block_offsets.get(n_A - 2)
    if TA_ann is not None and TB_cre is not None and sd_src2 is not None and sd_dst2 is not None:
        r_dst_A, r_src_A = TA_ann.shape[0], TA_ann.shape[1]
        r_dst_B, r_src_B = TB_cre.shape[0], TB_cre.shape[1]

        # Pdag[i,k,b_dst,b_src] = Σ_{j,l∈B} h2_full[i,j_B,k,l_B] × TB_cre[b_dst,b_src,j,l]
        Pdag = np.zeros((n_occ, n_occ, r_dst_B, r_src_B))
        for i in range(n_occ):
            for k in range(n_occ):
                for j in range(n_virt):
                    for l_idx in range(n_virt):
                        v = h2_full[i, j + n_occ, k, l_idx + n_occ]
                        if abs(v) < 1e-14:
                            continue
                        Pdag[i, k] += 0.5 * v * TB_cre[:, :, j, l_idx]

        _contract_pair(H_AB, sd_src2, sd_dst2,
                       r_dst_A, r_src_A, r_dst_B, r_src_B,
                       TA_ann, Pdag, n_occ)


def _contract_pair(H_AB, os, od, rA_dst, rA_src, rB_dst, rB_src, TA, PB, n_indices):
    """H_AB[α_dst,β_dst, α_src,β_src] += Σ_{i,k} TA[α_dst,α_src,i,k] × PB[i,k,β_dst,β_src]."""
    for a_dst in range(rA_dst):
        for a_src in range(rA_src):
            for b_dst in range(rB_dst):
                for b_src in range(rB_src):
                    val = 0.0
                    for i in range(n_indices):
                        for k in range(n_indices):
                            ta = TA[a_dst, a_src, i, k]
                            if abs(ta) < 1e-14:
                                continue
                            val += ta * PB[i, k, b_dst, b_src]
                    if abs(val) > 1e-14:
                        s = os + a_src * rB_src + b_src
                        d = od + a_dst * rB_dst + b_dst
                        H_AB[d, s] += val


# ═══════════════════════════════════════════════════════════════════════════
# 3A+1B and 1A+3B (kept from original — minor contributions)
# ═══════════════════════════════════════════════════════════════════════════

def _add_3body_patterns(
    H_AB, block_offsets, n_A,
    trans_A, trans_B, h2_full, n_occ, n_act, n_virt
):
    """3A+1B (n_A→n_A+1 with 3 A ops) and 3A+1B-rev (n_A→n_A-1)."""
    TA3 = trans_A.create2_annih1.get(n_A)
    TB1 = trans_B.annihilate_1.get(n_A)
    if TA3 is not None and TB1 is not None:
        _contract_3body(H_AB, block_offsets, n_A, n_A + 1,
                        TA3, TB1, h2_full, n_occ, n_act, n_virt)

    TA3b = trans_A.create1_annih2.get(n_A)
    TB1b = trans_B.create_1.get(n_A)
    if TA3b is not None and TB1b is not None:
        _contract_3body(H_AB, block_offsets, n_A, n_A - 1,
                        TA3b, TB1b, h2_full, n_occ, n_act, n_virt, swap=True)


def _add_1a3b_patterns(
    H_AB, block_offsets, n_A,
    trans_A, trans_B, h2_full, n_occ, n_act, n_virt
):
    """1A+3B (1 A op + 3 B ops, n_A→n_A±1)."""
    cre_A = trans_A.create_1.get(n_A)
    cre1_ann2_B = trans_B.create1_annih2.get(n_A)
    if cre_A is not None and cre1_ann2_B is not None:
        _contract_1a3b(
            H_AB, block_offsets, n_A, n_A + 1,
            cre_A, cre1_ann2_B, h2_full, n_occ, n_act, n_virt,
            sign=1, net_A_create=True
        )

    ann_A = trans_A.annihilate_1.get(n_A)
    cre2_ann1_B = trans_B.create2_annih1.get(n_A)
    if ann_A is not None and cre2_ann1_B is not None:
        _contract_1a3b(
            H_AB, block_offsets, n_A, n_A - 1,
            ann_A, cre2_ann1_B, h2_full, n_occ, n_act, n_virt,
            sign=1, net_A_create=False
        )


# ═══════════════════════════════════════════════════════════════════════════
# Low-level contraction helpers
# ═══════════════════════════════════════════════════════════════════════════

def _contract_3body(H_AB, block_offsets, n_A_src, n_A_dst,
                    TA, TB, h2_full, n_occ, n_act, n_virt, swap=False):
    """Contract 3A+1B (swap=False) or rev (swap=True)."""
    os = block_offsets.get(n_A_src)
    od = block_offsets.get(n_A_dst)
    if os is None or od is None:
        return

    r_dA, r_sA = TA.shape[0], TA.shape[1]
    r_dB, r_sB = TB.shape[0], TB.shape[1]
    nA = TA.shape[2]
    nB = TB.shape[2]

    if swap:
        for a_dst in range(r_dA):
            for a_src in range(r_sA):
                for b_dst in range(r_dB):
                    for b_src in range(r_sB):
                        v = 0.0
                        for p in range(nA):
                            for q in range(nA):
                                for r in range(nA):
                                    ta = TA[a_dst, a_src, p, q, r]
                                    if abs(ta) < 1e-15:
                                        continue
                                    for s in range(nB):
                                        tb = TB[b_dst, b_src, s]
                                        if abs(tb) < 1e-15:
                                            continue
                                        integ = 0.5 * h2_full[p, s + n_occ, q, r]
                                        v += integ * ta * tb
                        if abs(v) > 1e-15:
                            s = os + a_src * r_sA + b_src
                            d = od + a_dst * r_dA + b_dst
                            H_AB[d, s] += v
    else:
        for a_dst in range(r_dA):
            for a_src in range(r_sA):
                for b_dst in range(r_dB):
                    for b_src in range(r_sB):
                        v = 0.0
                        for p in range(nA):
                            for q in range(nA):
                                for r in range(nA):
                                    ta = TA[a_dst, a_src, p, q, r]
                                    if abs(ta) < 1e-15:
                                        continue
                                    for s in range(nB):
                                        tb = TB[b_dst, b_src, s]
                                        if abs(tb) < 1e-15:
                                            continue
                                        integ = 0.5 * h2_full[p, q, s + n_occ, r]
                                        v += integ * ta * tb
                        if abs(v) > 1e-15:
                            s = os + a_src * r_sA + b_src
                            d = od + a_dst * r_dA + b_dst
                            H_AB[d, s] += v


def _contract_1a3b(H_AB, block_offsets, n_A_src, n_A_dst,
                   TA, TB, h2_full, n_occ, n_act, n_virt,
                   sign, net_A_create):
    """Contract 1A+3B."""
    os = block_offsets.get(n_A_src)
    od = block_offsets.get(n_A_dst)
    if os is None or od is None:
        return

    r_dA, r_sA = TA.shape[0], TA.shape[1]
    nA_orb = TA.shape[2]
    r_dB, r_sB = TB.shape[0], TB.shape[1]
    nB_orb = TB.shape[2]

    if net_A_create:
        # Sub-case 1: p ∈ A, (q,s,r) ∈ B
        for a_dst in range(r_dA):
            for a_src in range(r_sA):
                for b_dst in range(r_dB):
                    for b_src in range(r_sB):
                        v = 0.0
                        for p_sub in range(nA_orb):
                            ta = TA[a_dst, a_src, p_sub]
                            if abs(ta) < 1e-15: continue
                            for qb in range(nB_orb):
                                for sb in range(nB_orb):
                                    for rb in range(nB_orb):
                                        tb = TB[b_dst, b_src, qb, sb, rb]
                                        if abs(tb) < 1e-15: continue
                                        integ = 0.5 * h2_full[
                                            p_sub, qb + n_occ,
                                            rb + n_occ, sb + n_occ]
                                        v += sign * integ * ta * tb
                        if abs(v) > 1e-15:
                            s = os + a_src * r_sA + b_src
                            d = od + a_dst * r_dA + b_dst
                            H_AB[d, s] += v

        # Sub-case 2: q ∈ A, (p,s,r) ∈ B
        for a_dst in range(r_dA):
            for a_src in range(r_sA):
                for b_dst in range(r_dB):
                    for b_src in range(r_sB):
                        v = 0.0
                        for q_sub in range(nA_orb):
                            ta = TA[a_dst, a_src, q_sub]
                            if abs(ta) < 1e-15: continue
                            for pb in range(nB_orb):
                                for sb in range(nB_orb):
                                    for rb in range(nB_orb):
                                        tb = TB[b_dst, b_src, pb, sb, rb]
                                        if abs(tb) < 1e-15: continue
                                        integ = 0.5 * h2_full[
                                            pb + n_occ, q_sub,
                                            rb + n_occ, sb + n_occ]
                                        v += sign * integ * ta * tb
                        if abs(v) > 1e-15:
                            s = os + a_src * r_sA + b_src
                            d = od + a_dst * r_dA + b_dst
                            H_AB[d, s] += v
    else:
        # Sub-case 1: r ∈ A, (p,q,s) ∈ B
        for a_dst in range(r_dA):
            for a_src in range(r_sA):
                for b_dst in range(r_dB):
                    for b_src in range(r_sB):
                        v = 0.0
                        for r_sub in range(nA_orb):
                            ta = TA[a_dst, a_src, r_sub]
                            if abs(ta) < 1e-15: continue
                            for pb in range(nB_orb):
                                for qb in range(nB_orb):
                                    for sb in range(nB_orb):
                                        tb = TB[b_dst, b_src, pb, qb, sb]
                                        if abs(tb) < 1e-15: continue
                                        integ = 0.5 * h2_full[
                                            pb + n_occ, qb + n_occ,
                                            sb + n_occ, r_sub]
                                        v += sign * integ * ta * tb
                        if abs(v) > 1e-15:
                            s = os + a_src * r_sA + b_src
                            d = od + a_dst * r_dA + b_dst
                            H_AB[d, s] += v

        # Sub-case 2: s ∈ A, (p,q,r) ∈ B
        for a_dst in range(r_dA):
            for a_src in range(r_sA):
                for b_dst in range(r_dB):
                    for b_src in range(r_sB):
                        v = 0.0
                        for s_sub in range(nA_orb):
                            ta = TA[a_dst, a_src, s_sub]
                            if abs(ta) < 1e-15: continue
                            for pb in range(nB_orb):
                                for qb in range(nB_orb):
                                    for rb in range(nB_orb):
                                        tb = TB[b_dst, b_src, pb, qb, rb]
                                        if abs(tb) < 1e-15: continue
                                        integ = 0.5 * h2_full[
                                            pb + n_occ, qb + n_occ,
                                            rb + n_occ, s_sub]
                                        v += sign * integ * ta * tb
                        if abs(v) > 1e-15:
                            s = os + a_src * r_sA + b_src
                            d = od + a_dst * r_dA + b_dst
                            H_AB[d, s] += v


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
