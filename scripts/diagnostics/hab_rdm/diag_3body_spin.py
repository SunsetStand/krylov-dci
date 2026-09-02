#!/usr/bin/env python3
"""Break down a 3-body element by spin combo (brute vs contraction)."""
import numpy as np
import sys
sys.path.insert(0, '/data/home/wangcx/krylov-dci')

from dm_svd_embedding.embedded_hamiltonian import _setup_h2o_system
from dm_svd_embedding.transition_rdm import _create_sign, _annihilate_sign
from dm_svd_embedding.occ_virt_partition import setup_partition
from dm_svd_embedding.transition_rdm import (
    _compute_det_create2_annih1_explicit, _compute_det_annihilation_explicit,
)

(mol, mf, cas, q_idx, backend, h1eff, h2_4d, fcivec, ci_flat,
 n_act, n_elec, n_occ, na, nb, ecore, E_fci) = _setup_h2o_system()
n_virt = n_act - n_occ

partition, full_dets = setup_partition(n_act, n_elec, n_occ, ms=0)
dets_src = partition[2]['a_dets']; dets_dst = partition[3]['a_dets']
idx_dst = partition[3]['a_index']
c2a1_A = _compute_det_create2_annih1_explicit(dets_src, dets_dst, idx_dst, n_occ)
bsrc = partition[2]['b_dets']; bdst = partition[3]['b_dets']
bidx = partition[3]['b_index']
a1_B = _compute_det_annihilation_explicit(bsrc, bdst, bidx, n_virt)


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
                # brute total
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

# brute broken down by (sigma, tau)
print("\nBrute by (sigma,tau):")
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
for k in sorted(brute_st):
    print(f"  ({k[0]},{k[1]}): {brute_st[k]:+.6f}")

# contraction broken down by combo (Case1 + Case2)
print("\nContraction by combo (JW on):")
jw = {'aaa': (-1)**na_src, 'aba': (-1)**nb_src, 'abb': (-1)**na_src,
      'bab': (-1)**na_src, 'baa': (-1)**nb_src, 'bbb': (-1)**nb_src}
pairs1 = [('aaa','a'),('abb','a'),('baa','b'),('bbb','b')]  # rB (B slot 2)
pairs2 = [('aaa','a'),('aba','b'),('bab','a'),('bbb','b')]  # sB (B slot 4)
cont = {}
for combo, spin in pairs1:
    for x in range(n_occ):
        for y in range(n_occ):
            for z in range(n_occ):
                ta = c2a1_A[combo][ai,aj,x,y,z]
                if abs(ta) < 1e-14: continue
                for b in range(n_virt):
                    tb = a1_B[spin][bi,bj,b]
                    if abs(tb) < 1e-14: continue
                    v = h2_4d[x, b+n_occ, y, z]
                    cont[combo+'-rB'] = cont.get(combo+'-rB',0.0) + 0.5*v*ta*tb*jw[combo]
for combo, spin in pairs2:
    for x in range(n_occ):
        for y in range(n_occ):
            for z in range(n_occ):
                ta = c2a1_A[combo][ai,aj,x,y,z]
                if abs(ta) < 1e-14: continue
                for b in range(n_virt):
                    tb = a1_B[spin][bi,bj,b]
                    if abs(tb) < 1e-14: continue
                    v = h2_4d[x, z, y, b+n_occ]
                    cont[combo+'-sB'] = cont.get(combo+'-sB',0.0) + 0.5*v*ta*tb*jw[combo]
for k in sorted(cont):
    print(f"  {k}: {cont[k]:+.6f}")
print(f"\nbrute total={sum(brute_st.values()):+.6f}  contraction total={sum(cont.values()):+.6f}")
