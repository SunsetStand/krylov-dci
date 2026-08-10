#!/usr/bin/env python3
"""
H_AB construction via second-quantization RDM contraction.

Computes H_AB matrix elements directly from Schmidt-basis transition
matrices and integrals, without CI expansion.

Key physics: B-space operators pick up Jordan-Wigner factor (-1)^{n_A}
when passing through A-space electrons in the sorted Fock basis.
"""

import numpy as np
from typing import Dict


# ═══════════════════════════════════════════════════════════════════════════
# Jordan-Wigner sign
# ═══════════════════════════════════════════════════════════════════════════

def _jw(n_A_cur: int) -> int:
    """Jordan-Wigner sign for one B-operator: (-1)^{n_A_cur}."""
    return 1 if (n_A_cur % 2 == 0) else -1


def _add_to_hab(H_AB, offset_src, offset_dst, r_src, r_dst,
                val_agbd, a_dst, a_src, b_dst, b_src, factor):
    """Safely add a contribution to H_AB."""
    if abs(factor) < 1e-16:
        return
    s = offset_src + a_src * r_src + b_src
    d = offset_dst + a_dst * r_dst + b_dst
    H_AB[d, s] += factor


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
    """Build H_AB matrix via RDM contraction. Fills H_AB in-place."""
    import time

    n_virt = n_act - n_occ

    # Set of all n_A blocks that exist
    all_blocks = sorted(schmidt_data.keys())

    if verbose:
        t0 = time.perf_counter()

    # ── 2e: n_A conserved (2A + 2B, 1-body transitions) ──
    # This is the dominant contribution. We iterate over all integral
    # index assignments with 2 indices in A, 2 in B.
    for n_A in all_blocks:
        sd = schmidt_data[n_A]
        rA_sch = sd['r']
        if rA_sch == 0:
            continue
        os = block_offsets.get(n_A)
        if os is None:
            continue

        TA = trans_A.trans_1.get(n_A)
        TB = trans_B.trans_1.get(n_A)
        if TA is None or TB is None:
            continue

        # Loop over all integral index assignments with exactly 2 in A
        _contract_2e_nA_conserved(
            H_AB, os, n_A, TA, TB, h2_full,
            n_occ, n_act, n_virt, rA_sch
        )

    # ── 2e: n_A → n_A+2 (pair creation in A, pair annihilation in B) ──
    for n_A in all_blocks:
        sd = schmidt_data[n_A]
        sd2 = schmidt_data.get(n_A + 2)
        if sd is None or sd2 is None:
            continue
        rA_src = sd['r']
        rA_dst = sd2['r']
        if rA_src == 0 or rA_dst == 0:
            continue

        TA = trans_A.create_2.get(n_A)
        TB = trans_B.annihilate_2.get(n_A)
        if TA is not None and TB is not None:
            _contract_2e_pair_transfer(
                H_AB, block_offsets, n_A, n_A + 2,
                TA, TB, h2_full, n_occ, n_act, n_virt,
                rA_src, rA_dst, jw_sign=1
            )

    # ── 2e: n_A → n_A-2 (pair annihilation in A, pair creation in B) ──
    for n_A in all_blocks:
        sd = schmidt_data[n_A]
        sd2 = schmidt_data.get(n_A - 2)
        if sd is None or sd2 is None:
            continue
        rA_src = sd['r']
        rA_dst = sd2['r']
        if rA_src == 0 or rA_dst == 0:
            continue

        TA = trans_A.annihilate_2.get(n_A)
        TB = trans_B.create_2.get(n_A)
        if TA is not None and TB is not None:
            _contract_2e_pair_transfer(
                H_AB, block_offsets, n_A, n_A - 2,
                TA, TB, h2_full, n_occ, n_act, n_virt,
                rA_src, rA_dst, jw_sign=1, swap_AB=True
            )

    # ── 2e: 3A+1B (3 ops in A + 1 op in B, n_A → n_A±1) ──
    for n_A in all_blocks:
        _add_3body_patterns(
            H_AB, block_offsets, n_A, trans_A, trans_B,
            h2_full, n_occ, n_act, n_virt
        )

    # ── 2e: 1A+3B (1 op in A + 3 ops in B, n_A → n_A±1) ──
    for n_A in all_blocks:
        _add_1a3b_patterns(
            H_AB, block_offsets, n_A, trans_A, trans_B,
            h2_full, n_occ, n_act, n_virt
        )

    # ── 1e cross terms ──
    for n_A in all_blocks:
        _add_1e_cross_block_rdm(
            H_AB, block_offsets, n_A, trans_A, trans_B,
            h1_full, n_occ, n_act, n_virt
        )

    if verbose:
        elapsed = time.perf_counter() - t0
        print(f"    H_AB RDM: {elapsed:.1f}s, ||H_AB||={np.linalg.norm(H_AB):.4f}")


# ═══════════════════════════════════════════════════════════════════════════
# 2e: n_A conserved (dominant contribution)
# ═══════════════════════════════════════════════════════════════════════════

def _contract_2e_nA_conserved(
    H_AB, offset_src, n_A_val,
    TA, TB, h2_full,
    n_occ, n_act, n_virt, rA_sch
):
    """Contract 2A+2B with n_A conserved.

    Enumerates all integral index patterns with 2 in A, 2 in B,
    computes JW sign from operator application order, contracts
    TA (A-space 1-body transition) and TB (B-space 1-body transition).
    """
    rA = TA.shape[0]  # Schmidt rank of A
    rB = TB.shape[0]  # Schmidt rank of B

    # Loop over all orbital index combinations
    for p in range(n_act):
        p_in_A = p < n_occ
        for q in range(n_act):
            q_in_A = q < n_occ
            for r in range(n_act):
                r_in_A = r < n_occ  # NOTE: this is a BOOL, different from rA!
                for s in range(n_act):
                    s_in_A = s < n_occ

                    n_in_A = p_in_A + q_in_A + r_in_A + s_in_A
                    if n_in_A != 2:
                        continue

                    # n_A conserved requires exactly 1 creation + 1 annihilation in A.
                    # p,q are creation ops; r,s are annihilation ops in a_p†a_q†a_s a_r.
                    n_create_A = p_in_A + q_in_A
                    n_annih_A  = r_in_A + s_in_A
                    if n_create_A != 1 or n_annih_A != 1:
                        # both A-ops are same type → n_A changes by ±2
                        # (handled by _contract_2e_pair_transfer below)
                        continue

                    integ = h2_full[p, q, r, s]
                    if abs(integ) < 1e-14:
                        continue
                    integ *= 0.5  # 1/2 prefactor in H = 1/2 Σ (pq|rs) a_p† a_q† a_s a_r

                    # Compute JW sign by tracking A-count through
                    # right-to-left operator application
                    cur = n_A_val
                    jw = 1

                    # a_r (rightmost, applied first)
                    if r_in_A:
                        cur -= 1
                    else:
                        jw *= _jw(cur)

                    # a_s (second)
                    if s_in_A:
                        cur -= 1
                    else:
                        jw *= _jw(cur)

                    # a_q† (third)
                    if q_in_A:
                        cur += 1
                    else:
                        jw *= _jw(cur)

                    # a_p† (fourth, leftmost)
                    if p_in_A:
                        cur += 1
                    else:
                        jw *= _jw(cur)

                    # Map to subspace orbital indices
                    p_sub = p if p_in_A else p - n_occ
                    q_sub = q if q_in_A else q - n_occ
                    r_sub = r if r_in_A else r - n_occ
                    s_sub = s if s_in_A else s - n_occ

                    # Collect A/B operators with explicit creation/annihilation type.
                    # p,q are creation (a_p†, a_q†); r,s are annihilation (a_r, a_s).
                    a_create = []
                    a_annih  = []
                    b_create = []
                    b_annih  = []

                    for op_type, in_A, sub_idx in [
                        ('c', p_in_A, p_sub),  # a_p†
                        ('c', q_in_A, q_sub),  # a_q†
                        ('a', s_in_A, s_sub),  # a_s  (annihilation)
                        ('a', r_in_A, r_sub),  # a_r  (annihilation)
                    ]:
                        if in_A:
                            if op_type == 'c':
                                a_create.append(sub_idx)
                            else:
                                a_annih.append(sub_idx)
                        else:
                            if op_type == 'c':
                                b_create.append(sub_idx)
                            else:
                                b_annih.append(sub_idx)

                    # n_A conserved filter: Bug 1 already guarantees
                    # 1 creation + 1 annihilation in each subspace.
                    assert len(a_create) == len(a_annih) == 1
                    assert len(b_create) == len(b_annih) == 1

                    ac, aa = a_create[0], a_annih[0]
                    bc, ba = b_create[0], b_annih[0]

                    # TA[:,:,creation,annihilation] = ⟨Ã_α|a_{ac}† a_{aa}|Ã_γ⟩
                    ta_slice = TA[:, :, ac, aa]
                    # TB[:,:,creation,annihilation] = ⟨B̃_β|a_{bc}† a_{ba}|B̃_δ⟩
                    tb_slice = TB[:, :, bc, ba]

                    factor = integ * jw

                    # Add to H_AB
                    for a_dst in range(rA):
                        for a_src in range(rA):
                            ta = ta_slice[a_dst, a_src]
                            if abs(ta) < 1e-15:
                                continue
                            for b_dst in range(rB):
                                for b_src in range(rB):
                                    tb = tb_slice[b_dst, b_src]
                                    if abs(tb) < 1e-15:
                                        continue
                                    v = factor * ta * tb
                                    if abs(v) < 1e-15:
                                        continue
                                    s = offset_src + a_src * rB + b_src
                                    d = offset_src + a_dst * rB + b_dst
                                    H_AB[d, s] += v


# ═══════════════════════════════════════════════════════════════════════════
# 2e: n_A ± 2 (pair transfer)
# ═══════════════════════════════════════════════════════════════════════════

def _contract_2e_pair_transfer(
    H_AB, block_offsets, n_A_src, n_A_dst,
    TA, TB, h2_full, n_occ, n_act, n_virt,
    r_src, r_dst, jw_sign, swap_AB=False
):
    """Contract 2e pair-transfer terms.

    TA: A-space pair creation/annihilation (r_dst, r_src, n_occ, n_occ)
    TB: B-space pair annihilation/creation (rB_dst, rB_src, n_virt, n_virt)

    For n_A→n_A+2 (swap_AB=False):
        Σ_{p,q∈A, r,s∈B} (pq|rs) · TA[α,γ,p,q] · TB[β,δ,r,s]
    For n_A→n_A-2 (swap_AB=True):
        Σ_{p,q∈B, r,s∈A} (pq|rs) · TA[γ,α,p,q] · TB[δ,β,r,s]
    (where TA now stores the A-space annihilation, TB stores B-space creation,
     but the integral pattern is swapped)
    """
    os = block_offsets.get(n_A_src)
    od = block_offsets.get(n_A_dst)
    if os is None or od is None:
        return

    r_dA, r_sA = TA.shape[0], TA.shape[1]
    r_dB, r_sB = TB.shape[0], TB.shape[1]

    if swap_AB:
        # (B,B,A,A): p,q∈B, r,s∈A → n_A → n_A-2
        # TA = A-space pair annihilation, shape (r_dA, r_sA, n_occ, n_occ)
        # TB = B-space pair creation, shape (r_dB, r_sB, n_virt, n_virt)
        # Contract: Σ_{pa,qa∈A, pb,qb∈B} (pb,qb|pa,qa) · TA[pa,qa] · TB[pb,qb]
        for a_dst in range(r_dA):
            for a_src in range(r_sA):
                for b_dst in range(r_dB):
                    for b_src in range(r_sB):
                        val = 0.0
                        for r_sub in range(n_occ):      # A-space (TA) indices
                            for s_sub in range(n_occ):
                                ta = TA[a_dst, a_src, r_sub, s_sub]
                                if abs(ta) < 1e-15:
                                    continue
                                for p_sub in range(n_virt):  # B-space (TB) indices
                                    for q_sub in range(n_virt):
                                        tb = TB[b_dst, b_src, p_sub, q_sub]
                                        if abs(tb) < 1e-15:
                                            continue
                                        integ = 0.5 * h2_full[
                                            p_sub + n_occ, q_sub + n_occ,
                                            r_sub, s_sub]
                                        val += integ * ta * tb * jw_sign
                        if abs(val) > 1e-15:
                            s = os + a_src * r_sA + b_src
                            d = od + a_dst * r_dA + b_dst
                            H_AB[d, s] += val
    else:
        # (A,A,B,B): p,q∈A, r,s∈B
        for a_dst in range(r_dA):
            for a_src in range(r_sA):
                for b_dst in range(r_dB):
                    for b_src in range(r_sB):
                        val = 0.0
                        for p_sub in range(n_occ):
                            for q_sub in range(n_occ):
                                ta = TA[a_dst, a_src, p_sub, q_sub]
                                if abs(ta) < 1e-15:
                                    continue
                                for r_sub in range(n_virt):
                                    for s_sub in range(n_virt):
                                        tb = TB[b_dst, b_src, r_sub, s_sub]
                                        if abs(tb) < 1e-15:
                                            continue
                                        integ = 0.5 * h2_full[
                                            p_sub, q_sub,
                                            r_sub + n_occ, s_sub + n_occ]
                                        val += integ * ta * tb * jw_sign
                        if abs(val) > 1e-15:
                            s = os + a_src * r_sA + b_src
                            d = od + a_dst * r_dA + b_dst
                            H_AB[d, s] += val


# ═══════════════════════════════════════════════════════════════════════════
# 3A+1B and 1A+3B
# ═══════════════════════════════════════════════════════════════════════════

def _add_3body_patterns(
    H_AB, block_offsets, n_A,
    trans_A, trans_B, h2_full, n_occ, n_act, n_virt
):
    """Add 3A+1B (n_A→n_A+1) and 1A+3B (n_A→n_A-1)."""
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


def _contract_3body(H_AB, block_offsets, n_A_src, n_A_dst,
                    TA, TB, h2_full, n_occ, n_act, n_virt, swap=False):
    """Contract 3A+1B (swap=False) or 1A+3B (swap=True).

    TA: A 3-body (r_dA, r_sA, n_occ, n_occ, n_occ).
    TB: B 1-body (r_dB, r_sB, n_orb).
    """
    os = block_offsets.get(n_A_src)
    od = block_offsets.get(n_A_dst)
    if os is None or od is None:
        return

    r_dA, r_sA = TA.shape[0], TA.shape[1]
    r_dB, r_sB = TB.shape[0], TB.shape[1]
    nA = TA.shape[2]
    nB = TB.shape[2]

    if swap:
        # 1A+3B: B creates 1, A annihilates 2 + creates 1 → n_A-1
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
                                        v += integ * ta * tb * _jw(n_A_src - 2)
                        if abs(v) > 1e-15:
                            s = os + a_src * r_sA + b_src
                            d = od + a_dst * r_dA + b_dst
                            H_AB[d, s] += v
    else:
        # 3A+1B: A creates 2 + annihilates 1, B annihilates 1 → n_A+1
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
                                        v += integ * ta * tb * _jw(n_A_src - 1)
                        if abs(v) > 1e-15:
                            s = os + a_src * r_sA + b_src
                            d = od + a_dst * r_dA + b_dst
                            H_AB[d, s] += v


# ═══════════════════════════════════════════════════════════════════════════
# 1A+3B patterns (1 op in A + 3 ops in B, n_A → n_A±1)
# ═══════════════════════════════════════════════════════════════════════════

def _add_1a3b_patterns(
    H_AB, block_offsets, n_A,
    trans_A, trans_B, h2_full, n_occ, n_act, n_virt
):
    """Add true 1A+3B patterns: 1 operator in A, 3 operators in B.

    These are the complementary patterns to _add_3body_patterns (3A+1B).
    Handles both n_A → n_A+1 and n_A → n_A-1.
    """
    # Pattern: 1 create in A + (1 create + 2 annih) in B → n_A → n_A+1
    # TA = create_1 (A creates 1), TB = create1_annih2 (B: 1 create + 2 annih)
    cre_A = trans_A.create_1.get(n_A)
    cre1_ann2_B = trans_B.create1_annih2.get(n_A)
    if cre_A is not None and cre1_ann2_B is not None:
        _contract_1a3b(
            H_AB, block_offsets, n_A, n_A + 1,
            cre_A, cre1_ann2_B, h2_full, n_occ, n_act, n_virt,
            jw_sign=_jw(n_A + 1), net_A_create=True
        )

    # Pattern: 1 annih in A + (2 create + 1 annih) in B → n_A → n_A-1
    # TA = annihilate_1, TB = create2_annih1
    ann_A = trans_A.annihilate_1.get(n_A)
    cre2_ann1_B = trans_B.create2_annih1.get(n_A)
    if ann_A is not None and cre2_ann1_B is not None:
        _contract_1a3b(
            H_AB, block_offsets, n_A, n_A - 1,
            ann_A, cre2_ann1_B, h2_full, n_occ, n_act, n_virt,
            jw_sign=_jw(n_A - 1), net_A_create=False
        )


def _contract_1a3b(H_AB, block_offsets, n_A_src, n_A_dst,
                   TA, TB, h2_full, n_occ, n_act, n_virt,
                   jw_sign, net_A_create):
    """Contract 1A+3B: 1 A-op × 3 B-ops.

    TA: A 1-body (r_dA, r_sA, n_occ) — create_1 or annihilate_1.
    TB: B 3-body (r_dB, r_sB, n_virt, n_virt, n_virt) — create1_annih2 or
        create2_annih1.

    net_A_create=True:  A creates, B has 1 create + 2 annih.
        Hamiltonian: a_p†(A) a_q†(B) a_s(B) a_r(B) or a_q†(A) a_p†(B) a_s(B) a_r(B)
        JW sign = (-1)^{n_A + 1} (verified by anticommutation + JW expansion).

    net_A_create=False: A annihilates, B has 2 create + 1 annih.
        Hamiltonian: a_s(A) a_p†(B) a_q†(B) a_r(B) or a_r(A) a_p†(B) a_q†(B) a_s(B)
        JW sign = (-1)^{n_A - 1}.
    """
    os = block_offsets.get(n_A_src)
    od = block_offsets.get(n_A_dst)
    if os is None or od is None:
        return

    r_dA, r_sA = TA.shape[0], TA.shape[1]
    nA_orb = TA.shape[2]
    r_dB, r_sB = TB.shape[0], TB.shape[1]
    nB_orb = TB.shape[2]

    if net_A_create:
        # p or q in A, the rest in B.
        # TB indices: (create, annih1, annih2) per create1_annih2 convention.
        # Integral: h2_full[p, q, r, s] with appropriate subspace mapping.

        # Sub-case 1: p ∈ A, (q, s, r) ∈ B
        for a_dst in range(r_dA):
            for a_src in range(r_sA):
                for b_dst in range(r_dB):
                    for b_src in range(r_sB):
                        v = 0.0
                        for p_sub in range(nA_orb):       # A create (p or q)
                            ta = TA[a_dst, a_src, p_sub]
                            if abs(ta) < 1e-15:
                                continue
                            # B: a_q†(B) a_s(B) a_r(B)
                            for qb in range(nB_orb):      # create in B
                                for sb in range(nB_orb):  # annih in B
                                    for rb in range(nB_orb):  # annih in B
                                        tb = TB[b_dst, b_src, qb, sb, rb]
                                        if abs(tb) < 1e-15:
                                            continue
                                        # p ∈ A (create), q ∈ B (create),
                                        # r ∈ B (annih), s ∈ B (annih)
                                        integ = 0.5 * h2_full[
                                            p_sub, qb + n_occ,
                                            rb + n_occ, sb + n_occ]
                                        v += integ * ta * tb * jw_sign
                        if abs(v) > 1e-15:
                            s = os + a_src * r_sA + b_src
                            d = od + a_dst * r_dA + b_dst
                            H_AB[d, s] += v

        # Sub-case 2: q ∈ A, (p, s, r) ∈ B
        for a_dst in range(r_dA):
            for a_src in range(r_sA):
                for b_dst in range(r_dB):
                    for b_src in range(r_sB):
                        v = 0.0
                        for q_sub in range(nA_orb):       # A create (q)
                            ta = TA[a_dst, a_src, q_sub]
                            if abs(ta) < 1e-15:
                                continue
                            for pb in range(nB_orb):      # create in B
                                for sb in range(nB_orb):
                                    for rb in range(nB_orb):
                                        tb = TB[b_dst, b_src, pb, sb, rb]
                                        if abs(tb) < 1e-15:
                                            continue
                                        # q ∈ A (create), p ∈ B (create),
                                        # r ∈ B (annih), s ∈ B (annih)
                                        integ = 0.5 * h2_full[
                                            pb + n_occ, q_sub,
                                            rb + n_occ, sb + n_occ]
                                        v += integ * ta * tb * jw_sign
                        if abs(v) > 1e-15:
                            s = os + a_src * r_sA + b_src
                            d = od + a_dst * r_dA + b_dst
                            H_AB[d, s] += v
    else:
        # net_A_create=False: A annihilates, B has 2 create + 1 annih.
        # TB indices: (create1, create2, annih) per create2_annih1 convention.

        # Sub-case 1: r ∈ A, (p, q, s) ∈ B
        for a_dst in range(r_dA):
            for a_src in range(r_sA):
                for b_dst in range(r_dB):
                    for b_src in range(r_sB):
                        v = 0.0
                        for r_sub in range(nA_orb):       # A annih (r or s)
                            ta = TA[a_dst, a_src, r_sub]
                            if abs(ta) < 1e-15:
                                continue
                            for pb in range(nB_orb):      # create in B
                                for qb in range(nB_orb):  # create in B
                                    for sb in range(nB_orb):  # annih in B
                                        tb = TB[b_dst, b_src, pb, qb, sb]
                                        if abs(tb) < 1e-15:
                                            continue
                                        # r ∈ A (annih), p ∈ B (create),
                                        # q ∈ B (create), s ∈ B (annih)
                                        integ = 0.5 * h2_full[
                                            pb + n_occ, qb + n_occ,
                                            sb + n_occ, r_sub]
                                        v += integ * ta * tb * jw_sign
                        if abs(v) > 1e-15:
                            s = os + a_src * r_sA + b_src
                            d = od + a_dst * r_dA + b_dst
                            H_AB[d, s] += v

        # Sub-case 2: s ∈ A, (p, q, r) ∈ B
        for a_dst in range(r_dA):
            for a_src in range(r_sA):
                for b_dst in range(r_dB):
                    for b_src in range(r_sB):
                        v = 0.0
                        for s_sub in range(nA_orb):       # A annih (s)
                            ta = TA[a_dst, a_src, s_sub]
                            if abs(ta) < 1e-15:
                                continue
                            for pb in range(nB_orb):
                                for qb in range(nB_orb):
                                    for rb in range(nB_orb):
                                        tb = TB[b_dst, b_src, pb, qb, rb]
                                        if abs(tb) < 1e-15:
                                            continue
                                        # s ∈ A (annih), p ∈ B (create),
                                        # q ∈ B (create), r ∈ B (annih)
                                        integ = 0.5 * h2_full[
                                            pb + n_occ, qb + n_occ,
                                            rb + n_occ, s_sub]
                                        v += integ * ta * tb * jw_sign
                        if abs(v) > 1e-15:
                            s = os + a_src * r_sA + b_src
                            d = od + a_dst * r_dA + b_dst
                            H_AB[d, s] += v


# ═══════════════════════════════════════════════════════════════════════════
# 1e cross terms
# ═══════════════════════════════════════════════════════════════════════════

def _add_1e_cross_block_rdm(
    H_AB, block_offsets, n_A,
    trans_A, trans_B, h1_full, n_occ, n_act, n_virt
):
    """1e cross: h_pr a_p†(A) a_r(B) + h.c."""
    # h_pr a_p†(A) a_r(B), sign = (-1)^{n_A}
    cre_A = trans_A.create_1.get(n_A)
    ann_B = trans_B.annihilate_1.get(n_A)
    if cre_A is not None and ann_B is not None:
        _add_1e_cross_pair(H_AB, block_offsets, n_A, n_A + 1,
                           cre_A, ann_B, h1_full, n_occ,
                           jw_sign=_jw(n_A), rev_B=False)

    # h_rp a_r†(B) a_p(A), sign = (-1)^{n_A-1}
    ann_A = trans_A.annihilate_1.get(n_A)
    cre_B = trans_B.create_1.get(n_A)
    if ann_A is not None and cre_B is not None:
        _add_1e_cross_pair(H_AB, block_offsets, n_A, n_A - 1,
                           ann_A, cre_B, h1_full, n_occ,
                           jw_sign=_jw(n_A - 1), rev_B=True)


def _add_1e_cross_pair(H_AB, block_offsets, n_A_src, n_A_dst,
                       TA, TB, h1_full, n_occ,
                       jw_sign, rev_B=False):
    """Add 1e cross contribution.

    TA: A-op transition (r_dA, r_sA, n_occ)
    TB: B-op transition (r_dB, r_sB, n_virt)
    rev_B=False: h[p,r] * TA[α,γ,p] * TB[β,δ,r]
    rev_B=True:  h[r,p] * TA[α,γ,p] * TB[β,δ,r]
    """
    os = block_offsets.get(n_A_src)
    od = block_offsets.get(n_A_dst)
    if os is None or od is None:
        return

    r_dA, r_sA = TA.shape[0], TA.shape[1]
    r_dB, r_sB = TB.shape[0], TB.shape[1]
    n_vB = TB.shape[2]

    for a_dst in range(r_dA):
        for a_src in range(r_sA):
            for b_dst in range(r_dB):
                for b_src in range(r_sB):
                    val = 0.0
                    for p in range(n_occ):
                        ta = TA[a_dst, a_src, p]
                        if abs(ta) < 1e-15:
                            continue
                        for r_sub in range(n_vB):
                            tb = TB[b_dst, b_src, r_sub]
                            if abs(tb) < 1e-15:
                                continue
                            r_full = r_sub + n_occ
                            if rev_B:
                                val += h1_full[r_full, p] * ta * tb * jw_sign
                            else:
                                val += h1_full[p, r_full] * ta * tb * jw_sign
                    if abs(val) > 1e-15:
                        s = os + a_src * r_sA + b_src
                        d = od + a_dst * r_dA + b_dst
                        H_AB[d, s] += val
