#!/usr/bin/env python3
"""Break down n-1 (3->2) 3-body by spin combo: brute vs contraction (3a1b c1a2-on-A + 1a3b c2a1-on-B)."""
import numpy as np
import sys
sys.path.insert(0, '/data/home/wangcx/krylov-dci')

from dm_svd_embedding.embedded_hamiltonian import _setup_h2o_system
from dm_svd_embedding.transition_rdm import _create_sign, _annihilate_sign
from dm_svd_embedding.occ_virt_partition import setup_partition
from dm_svd_embedding.transition_rdm import (
    _compute_det_create1_annih2_explicit, _compute_det_creation_explicit,
    _compute_det_create2_annih1_explicit, _compute_det_annihilation_explicit,
)

(mol, mf, cas, q_idx, backend, h1eff, h2_4d, fcivec, ci_flat,
 n_act, n_elec, n_occ, na, nb, ecore, E_fci) = _setup_h2o_system()
n_virt = n_act - n_occ

partition, full_dets = setup_partition(n_act, n_elec, n_occ, ms=0)

# source n_A=3 -> dest n_A=2 (A loses 1 = c1a2 on A; B gains 1 = c1 on B)
dets_src = partition[3]['a_dets']; dets_dst = partition[2]['a_dets']
idx_dst = partition[2]['a_index']
c1a2_A = _compute_det_create1_annih2_explicit(dets_src, dets_dst, idx_dst, n_occ)

bsrc = partition[3]['b_dets']; bdst = partition[2]['b_dets']
bidx = partition[2]['b_index']
c1_B = _compute_det_creation_explicit(bsrc, bdst, bidx, n_virt)

# 1a3b n-1: A annihilate (a1 on A), B c2a1 (create2+annih1 on B)
a1_A = _compute_det_annihilation_explicit(dets_src, dets_dst, idx_dst, n_occ)
c2a1_B = _compute_det_create2_annih1_explicit(bsrc, bdst, bidx, n_virt)


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


# find a large element
target = None
for aj in range(len(dets_src)):
    for bj in range(len(bsrc)):
        full_src = (dets_src[aj][0] | (bsrc[bj][0] << n_occ), dets_src[aj][1] | (bsrc[bj][1] << n_occ))
        for ai in range(len(dets_dst)):
            for bi in range(len(bdst)):
                full_dst = (dets_dst[ai][0] | (bdst[bi][0] << n_occ), dets_dst[ai][1] | (bdst[bi][1] << n_occ))
                tot = 0.0
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
                                        if f == full_dst: tot += 0.5*g*ph
                if abs(tot) > 0.2:
                    target = (aj, bj, ai, bi, full_src, full_dst)
                    break
            if target: break
        if target: break
    if target: break

aj, bj, ai, bi, full_src, full_dst = target
na_src = dets_src[aj][0].bit_count(); nb_src = dets_src[aj][1].bit_count()
print(f"src A=({dets_src[aj][0]:b},{dets_src[aj][1]:b}) B=({bsrc[bj][0]:b},{bsrc[bj][1]:b}) nA=3")
print(f"dst A=({dets_dst[ai][0]:b},{dets_dst[ai][1]:b}) B=({bdst[bi][0]:b},{bdst[bi][1]:b}) nA=2")

# brute by (sigma,tau)
brute_st = {}
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
                        if f == full_dst:
                            brute_st[(sig,tau)] = brute_st.get((sig,tau),0.0) + 0.5*g*ph
print("\nBrute by (sigma,tau):")
for k in sorted(brute_st):
    print(f"  ({k[0]},{k[1]}): {brute_st[k]:+.6f}")

# 3a1b (c1a2 on A) contraction
jw_c1a2 = {'aaa': (-1)**(na_src-1), 'aab': (-1)**(nb_src-1), 'aba': (-1)**(nb_src-1),
           'bab': (-1)**(na_src-1), 'bba': (-1)**(na_src-1), 'bbb': (-1)**(nb_src-1)}
pairs_pB = [('aaa','a'),('bba','a'),('aab','b'),('bbb','b')]
pairs_qB = [('aaa','a'),('aba','b'),('bab','a'),('bbb','b')]
cont_3a1b = {}
for integ_kind, pairs in (('pB', pairs_pB), ('qB', pairs_qB)):
    for combo, spin in pairs:
        extra = -1.0 if (combo in ('aaa','bbb') and integ_kind == 'qB') else 1.0
        for x in range(n_occ):
            for y in range(n_occ):
                for z in range(n_occ):
                    ta = c1a2_A[combo][ai,aj,x,y,z]
                    if abs(ta) < 1e-14: continue
                    for b in range(n_virt):
                        tb = c1_B[spin][bi,bj,b]
                        if abs(tb) < 1e-14: continue
                        if integ_kind == 'pB':
                            v = h2_4d[b+n_occ, z, x, y]
                        else:  # qB
                            v = h2_4d[x, z, b+n_occ, y]
                        cont_3a1b[combo+'-'+integ_kind] = cont_3a1b.get(combo+'-'+integ_kind,0.0) + 0.5*v*ta*tb*jw_c1a2[combo]*extra

print("\n3a1b (c1a2 on A) contraction:")
for k in sorted(cont_3a1b):
    print(f"  {k}: {cont_3a1b[k]:+.6f}")

# 1a3b (c2a1 on B) contraction: A-side a1 (annihilate), JW phase (-1)^{n-1}
jw_a1 = {'a': (-1)**(na_src-1), 'b': (-1)**(nb_src-1)}
pairs_rA = [('a','aaa'),('a','abb'),('b','baa'),('b','bbb')]
pairs_sA = [('a','aaa'),('b','aba'),('a','bab'),('b','bbb')]
cont_1a3b = {}
for integ_kind, pairs in (('rA', pairs_rA), ('sA', pairs_sA)):
    for spin, combo in pairs:
        extra = -1.0 if (combo in ('aaa','bbb') and integ_kind == 'sA') else 1.0
        for x in range(n_virt):
            for y in range(n_virt):
                for z in range(n_virt):
                    tb = c2a1_B[combo][bi,bj,x,y,z]
                    if abs(tb) < 1e-14: continue
                    for a in range(n_occ):
                        ta = a1_A[spin][ai,aj,a]
                        if abs(ta) < 1e-14: continue
                        if integ_kind == 'rA':
                            v = h2_4d[x+n_occ, a, y+n_occ, z+n_occ]
                        else:  # sA
                            v = h2_4d[x+n_occ, z+n_occ, y+n_occ, a]
                        cont_1a3b[combo+'-'+integ_kind] = cont_1a3b.get(combo+'-'+integ_kind,0.0) + 0.5*v*ta*tb*jw_a1[spin]*extra

print("\n1a3b (c2a1 on B) contraction:")
for k in sorted(cont_1a3b):
    print(f"  {k}: {cont_1a3b[k]:+.6f}")

print(f"\nbrute total={sum(brute_st.values()):+.6f}  3a1b total={sum(cont_3a1b.values()):+.6f}  1a3b total={sum(cont_1a3b.values()):+.6f}")
