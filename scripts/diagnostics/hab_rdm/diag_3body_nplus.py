#!/usr/bin/env python3
"""n+1 (2->3) full 3-body breakdown: brute vs contraction (3a1b c2a1-on-A + 1a3b c1a2-on-B)."""
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

# source n_A=2 -> dest n_A=3 (A gains 1)
dets_src = partition[2]['a_dets']; dets_dst = partition[3]['a_dets']
idx_dst = partition[3]['a_index']
c2a1_A = _compute_det_create2_annih1_explicit(dets_src, dets_dst, idx_dst, n_occ)
a1_A = _compute_det_annihilation_explicit(dets_src, dets_dst, idx_dst, n_occ)
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
print(f"src A=({dets_src[aj][0]:b},{dets_src[aj][1]:b}) B=({bsrc[bj][0]:b},{bsrc[bj][1]:b}) nA=2")
print(f"dst A=({dets_dst[ai][0]:b},{dets_dst[ai][1]:b}) B=({bdst[bi][0]:b},{bdst[bi][1]:b}) nA=3")

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

# 3a1b (c2a1 on A): A create2+annih1, B annihilate. JW = (-1)^{n_spin} (net-created)
jw_c2a1 = {'aaa': (-1)**na_src, 'aba': (-1)**nb_src, 'abb': (-1)**na_src,
           'bab': (-1)**na_src, 'baa': (-1)**nb_src, 'bbb': (-1)**nb_src}
pairs_rB = [('aaa','a'),('abb','a'),('baa','b'),('bbb','b')]
pairs_sB = [('aaa','a'),('aba','b'),('bab','a'),('bbb','b')]
cont_3a1b = {}
for integ_kind, pairs in (('rB', pairs_rB), ('sB', pairs_sB)):
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
                        if integ_kind == 'rB':
                            v = h2_4d[x, b+n_occ, y, z]
                        else:  # sB
                            v = h2_4d[x, z, y, b+n_occ]
                        cont_3a1b[combo+'-'+integ_kind] = cont_3a1b.get(combo+'-'+integ_kind,0.0) + 0.5*v*ta*tb*jw_c2a1[combo]*extra

print("\n3a1b (c2a1 on A) contraction:")
for k in sorted(cont_3a1b):
    print(f"  {k}: {cont_3a1b[k]:+.6f}")

# 1a3b (c1a2 on B): A create, B create1+annih2. A-side JW = (-1)^{n_spin}
jw_c1 = {'a': (-1)**na_src, 'b': (-1)**nb_src}
pairs_pA = [('a','aaa'),('a','bba'),('b','aab'),('b','bbb')]
pairs_qA = [('a','aaa'),('b','aba'),('a','bab'),('b','bbb')]
cont_1a3b = {}
for integ_kind, pairs in (('pA', pairs_pA), ('qA', pairs_qA)):
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
                        if integ_kind == 'pA':
                            v = h2_4d[a, z+n_occ, x+n_occ, y+n_occ]
                        else:  # qA
                            v = h2_4d[x+n_occ, z+n_occ, a, y+n_occ]
                        cont_1a3b[combo+'-'+integ_kind] = cont_1a3b.get(combo+'-'+integ_kind,0.0) + 0.5*v*ta*tb*jw_c1[spin]*extra

print("\n1a3b (c1a2 on B) contraction:")
for k in sorted(cont_1a3b):
    print(f"  {k}: {cont_1a3b[k]:+.6f}")

print(f"\nbrute total={sum(brute_st.values()):+.6f}  3a1b total={sum(cont_3a1b.values()):+.6f}  1a3b total={sum(cont_1a3b.values()):+.6f}")
