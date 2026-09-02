#!/usr/bin/env python3
"""Definitive brute-force check of the pair-transfer contraction.

Computes <d_dst| H_pair |d_src> two ways in the RAW determinant basis:
  (A) direct operator-application sum over p,q in A, r,s in B, sigma,tau
  (B) my contraction logic (spin-explicit transition matrices + h2)
and compares element-wise to find the exact factor/sign.
"""
import numpy as np
import sys
sys.path.insert(0, '/data/home/wangcx/krylov-dci')

from dm_svd_embedding.embedded_hamiltonian import _setup_h2o_system
from dm_svd_embedding.transition_rdm import (
    _create_sign, _annihilate_sign,
)


def apply_op(d, ops):
    """Apply a list of (kind, spin, orb) ops to det (alpha,beta).  kind in {'c','a'}."""
    aA, bA = d
    phase = 1
    for kind, spin, orb in ops:
        if spin == 'a':
            if kind == 'c':
                ph, aA = _create_sign(aA, orb)
            else:
                ph, aA = _annihilate_sign(aA, orb)
        else:
            if kind == 'c':
                ph, bA = _create_sign(bA, orb)
            else:
                ph, bA = _annihilate_sign(bA, orb)
        if ph == 0:
            return None
        phase *= ph
    return (aA, bA), phase


def brute_pair(d_src, d_dst, h2, n_occ, n_act):
    """<d_dst| H_pair |d_src> = 1/2 sum_{pq in A, rs in B} sum_st (pq|rs) <d_dst| c_p c_q a_s a_r |d_src>."""
    val = 0.0
    n_virt = n_act - n_occ
    for p in range(n_occ):
        for q in range(n_occ):
            for r in range(n_virt):
                for s in range(n_virt):
                    g = h2[p, q, r + n_occ, s + n_occ]
                    if abs(g) < 1e-14:
                        continue
                    for sig in ('a', 'b'):
                        for tau in ('a', 'b'):
                            # <d_dst| c_p,sig c_q,tau a_s,tau a_r,sig |d_src>
                            # apply right-to-left: a_r,sig ; a_s,tau ; c_q,tau ; c_p,sig
                            res = apply_op(d_src, [('a', sig, r + n_occ), ('a', tau, s + n_occ),
                                                   ('c', tau, q), ('c', sig, p)])
                            if res is None:
                                continue
                            final, phase = res
                            if final == d_dst:
                                val += 0.5 * g * phase
    return val


def contraction_pair(d_src, d_dst, h2, n_occ, n_act, c2_A, a2_B):
    """My contraction: 1/2 sum_st sum_{ik in A, jl in B} (ik|jl) c2_A[sA][i,k] a2_B[sB][l,j]."""
    n_virt = n_act - n_occ
    spin_pairs = [('aa', 'aa'), ('ab', 'ba'), ('ba', 'ab'), ('bb', 'bb')]
    val = 0.0
    for sA, sB in spin_pairs:
        for i in range(n_occ):
            for k in range(n_occ):
                for j in range(n_virt):
                    for l in range(n_virt):
                        g = h2[i, k, j + n_occ, l + n_occ]
                        if abs(g) < 1e-14:
                            continue
                        # c2_A[sA][a_dst, a_src, i, k] * a2_B[sB][b_dst, b_src, l, j]
                        val += 0.5 * g * c2_A[sA][i, k] * a2_B[sB][l, j]
    return val


def main():
    (mol, mf, cas, q_idx, backend, h1eff, h2_4d, fcivec, ci_flat,
     n_act, n_elec, n_occ, na, nb, ecore, E_fci) = _setup_h2o_system()
    n_virt = n_act - n_occ

    from dm_svd_embedding.transition_rdm import (
        _compute_det_pair_creation_explicit, _compute_det_pair_annihilation_explicit,
    )
    from dm_svd_embedding.occ_virt_partition import setup_partition
    from dm_svd_embedding.density_matrix import compute_schmidt_decomposition
    from dm_svd_embedding.transition_rdm import compute_transition_matrices
    from dm_svd_embedding.occ_virt_partition import build_block_matrices

    partition, full_dets = setup_partition(n_act, n_elec, n_occ, ms=0)
    C_blocks = build_block_matrices(partition, ci_flat)
    schmidt = compute_schmidt_decomposition(C_blocks, eps=1e-3)

    # pick source block n_A=2, dest block n_A=4 (A subspace)
    blk_src = partition[2]
    blk_dst = partition[4]
    dets_src = blk_src['a_dets']
    dets_dst = blk_dst['a_dets']
    idx_dst = blk_dst['a_index']

    # raw determinant-basis spin-explicit pair creation (A side, n_A=2 -> 4)
    c2_A = _compute_det_pair_creation_explicit(dets_src, dets_dst, idx_dst, n_occ)

    # B side: pair annihilation, B-block(2) -> B-block(4)  (B loses 2 electrons)
    blk_bsrc = partition[2]['b_dets']
    blk_bdst = partition[4]['b_dets']
    idx_bdst = partition[4]['b_index']
    a2_B = _compute_det_pair_annihilation_explicit(blk_bsrc, blk_bdst, idx_bdst, n_virt)

    # Compare element-wise for a few pairs
    print("comparing brute vs contraction (A determinant index, B determinant index):")
    import itertools
    ntest = 0
    maxabs = 0.0
    for ai in range(min(len(dets_dst), 3)):
        for aj in range(min(len(dets_src), 3)):
            for bi in range(min(len(blk_bdst), 3)):
                for bj in range(min(len(blk_bsrc), 3)):
                    d_src = (dets_src[aj][0], dets_src[aj][1],
                             blk_bsrc[bj][0], blk_bsrc[bj][1])
                    d_dst = (dets_dst[ai][0], dets_dst[ai][1],
                             blk_bdst[bi][0], blk_bdst[bi][1])
                    # brute: full 4-index (A,B) determinant
                    full_src = (dets_src[aj][0] | (blk_bsrc[bj][0] << n_occ),
                                dets_src[aj][1] | (blk_bsrc[bj][1] << n_occ))
                    full_dst = (dets_dst[ai][0] | (blk_bdst[bi][0] << n_occ),
                                dets_dst[ai][1] | (blk_bdst[bi][1] << n_occ))
                    bval = brute_pair(full_src, full_dst, h2_4d, n_occ, n_act)
                    # contraction (spin-explicit, NO JW phase yet -- test raw operator contraction)
                    cval = 0.0
                    spin_pairs = [('aa', 'aa'), ('ab', 'ba'), ('ba', 'ab'), ('bb', 'bb')]
                    for sA, sB in spin_pairs:
                        for i in range(n_occ):
                            for k in range(n_occ):
                                for j in range(n_virt):
                                    for l in range(n_virt):
                                        g = h2_4d[i, k, j + n_occ, l + n_occ]
                                        if abs(g) < 1e-14:
                                            continue
                                        cval += 0.5 * g * c2_A[sA][ai, aj, i, k] * a2_B[sB][bi, bj, l, j]
                    diff = abs(bval - cval)
                    maxabs = max(maxabs, diff)
                    if abs(bval) > 1e-8 or abs(cval) > 1e-8:
                        ntest += 1
                        if ntest <= 6:
                            print(f"  brute={bval:+.8f}  contraction={cval:+.8f}  diff={diff:.2e}")
    print(f"max abs diff over sampled elements = {maxabs:.3e}")


if __name__ == "__main__":
    main()
