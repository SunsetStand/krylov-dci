#!/usr/bin/env python3
"""Scan ALL determinant pairs (n+1 2->3) for brute-vs-contraction residual."""
import numpy as np
import sys
sys.path.insert(0, '/data/home/wangcx/krylov-dci')

from dm_svd_embedding.embedded_hamiltonian import _setup_h2o_system
from dm_svd_embedding.transition_rdm import _create_sign, _annihilate_sign
from dm_svd_embedding.occ_virt_partition import setup_partition
from dm_svd_embedding.transition_rdm import (
    _compute_det_create2_annih1_explicit, _compute_det_annihilation_explicit,
    _compute_det_creation_explicit, _compute_det_create1_annih2_explicit,
)

(mol, mf, cas, q_idx, backend, h1eff, h2_4d, fcivec, ci_flat,
 n_act, n_elec, n_occ, na, nb, ecore, E_fci) = _setup_h2o_system()
n_virt = n_act - n_occ
partition, full_dets = setup_partition(n_act, n_elec, n_occ, ms=0)

dets_src = partition[2]['a_dets']; dets_dst = partition[3]['a_dets']
idx_dst = partition[3]['a_index']
c2a1_A = _compute_det_create2_annih1_explicit(dets_src, dets_dst, idx_dst, n_occ)
c1_A = _compute_det_creation_explicit(dets_src, dets_dst, idx_dst, n_occ)
bsrc = partition[2]['b_dets']; bdst = partition[3]['b_dets']
bidx = partition[3]['b_index']
a1_B = _compute_det_annihilation_explicit(bsrc, bdst, bidx, n_virt)
c1a2_B = _compute_det_create1_annih2_explicit(bsrc, bdst, bidx, n_virt)


def apply_op(d, ops):
    aA, bA = d; phase = 1
    for kind, spin, orb in ops:
        if spin == 'a':
            ph, aA = (_create_sign(aA, orb) if kind == 'c' else _annihilate_sign(aA, orb))
        else:
            ph, bA = (_create_sign(bA, orb) if kind == 'c' else _annihilate_sign(bA, orb))
        if ph == 0:
            return None
        phase *= ph
    return (aA, bA), phase


def brute(full_src, full_dst):
    val = 0.0
    for p in range(n_act):
        for q in range(n_act):
            for r in range(n_act):
                for s in range(n_act):
                    g = h2_4d[p,q,r,s]
                    if abs(g) < 1e-14: continue
                    for sig in ('a','b'):
                        for tau in ('a','b'):
                            res = apply_op(full_src, [('a',sig,q),('a',tau,s),('c',tau,r),('c',sig,p)])
                            if res is None: continue
                            f, ph = res
                            if f == full_dst: val += 0.5*g*ph
    return val


def contract(aj, bj, ai, bi):
    na_src = dets_src[aj][0].bit_count(); nb_src = dets_src[aj][1].bit_count()
    jw_c2a1 = {'aaa': (-1)**na_src, 'aba': (-1)**nb_src, 'abb': (-1)**na_src,
               'bab': (-1)**na_src, 'baa': (-1)**nb_src, 'bbb': (-1)**nb_src}
    jw_c1 = {'a': (-1)**na_src, 'b': (-1)**nb_src}
    val = 0.0
    # 3a1b c2a1 on A
    for integ_kind, pairs in (('rB', [('aaa','a'),('abb','a'),('baa','b'),('bbb','b')]),
                              ('sB', [('aaa','a'),('aba','b'),('bab','a'),('bbb','b')])):
        for combo, spin in pairs:
            extra = -1.0 if (combo in ('aaa','bbb') and integ_kind == 'sB') else 1.0
            for x in range(n_occ):
                for y in range(n_occ):
                    for z in range(n_occ):
                        ta = c2a1_A[combo][ai,aj,x,y,z]
                        if abs(ta) < 1e-14: continue
                        for b in range(n_virt):
                            tb = a1_B[spin][bi,bj,b]
                            if abs(tb) < 1e-14: continue
                            v = h2_4d[x, b+n_occ, y, z] if integ_kind == 'rB' else h2_4d[x, z, y, b+n_occ]
                            val += 0.5*v*ta*tb*jw_c2a1[combo]*extra
    # 1a3b c1a2 on B
    for integ_kind, pairs in (('pA', [('a','aaa'),('a','bba'),('b','aab'),('b','bbb')]),
                              ('qA', [('a','aaa'),('b','aba'),('a','bab'),('b','bbb')])):
        for spin, combo in pairs:
            extra = -1.0 if (combo in ('aaa','bbb') and integ_kind == 'qA') else 1.0
            for x in range(n_virt):
                for y in range(n_virt):
                    for z in range(n_virt):
                        tb = c1a2_B[combo][bi,bj,x,y,z]
                        if abs(tb) < 1e-14: continue
                        for a in range(n_occ):
                            ta = c1_A[spin][ai,aj,a]
                            if abs(ta) < 1e-14: continue
                            v = h2_4d[a, z+n_occ, x+n_occ, y+n_occ] if integ_kind == 'pA' else h2_4d[x+n_occ, z+n_occ, a, y+n_occ]
                            val += 0.5*v*ta*tb*jw_c1[spin]*extra
    return val


maxdiff = 0.0; nmis = 0; worst = None
for aj in range(len(dets_src)):
    for bj in range(len(bsrc)):
        full_src = (dets_src[aj][0] | (bsrc[bj][0] << n_occ), dets_src[aj][1] | (bsrc[bj][1] << n_occ))
        for ai in range(len(dets_dst)):
            for bi in range(len(bdst)):
                full_dst = (dets_dst[ai][0] | (bdst[bi][0] << n_occ), dets_dst[ai][1] | (bdst[bi][1] << n_occ))
                bv = brute(full_src, full_dst)
                cv = contract(aj, bj, ai, bi)
                d = abs(bv - cv)
                if d > maxdiff:
                    maxdiff = d; worst = (aj, bj, ai, bi, bv, cv)
                if d > 1e-8:
                    nmis += 1

print(f"scanned 2->3: {len(dets_src)*len(bsrc)*len(dets_dst)*len(bdst)} pairs, {nmis} mismatches")
print(f"max diff = {maxdiff:.3e}")
if worst:
    aj, bj, ai, bi, bv, cv = worst
    print(f"worst pair: src A=({dets_src[aj][0]:b},{dets_src[aj][1]:b}) B=({bsrc[bj][0]:b},{bsrc[bj][1]:b})  dst A=({dets_dst[ai][0]:b},{dets_dst[ai][1]:b}) B=({bdst[bi][0]:b},{bdst[bi][1]:b})")
    print(f"  brute={bv:+.8f} contract={cv:+.8f} diff={bv-cv:+.8f}")
