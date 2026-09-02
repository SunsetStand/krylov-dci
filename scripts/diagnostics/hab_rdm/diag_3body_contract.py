#!/usr/bin/env python3
"""Isolate 3-body contraction (determinant basis) vs brute-force chemist H_2e."""
import numpy as np
import sys
sys.path.insert(0, '/data/home/wangcx/krylov-dci')

from dm_svd_embedding.embedded_hamiltonian import _setup_h2o_system
from dm_svd_embedding.transition_rdm import _create_sign, _annihilate_sign
from dm_svd_embedding.occ_virt_partition import setup_partition
from src.hamiltonian import Hamiltonian

(mol, mf, cas, q_idx, backend, h1eff, h2_4d, fcivec, ci_flat,
 n_act, n_elec, n_occ, na, nb, ecore, E_fci) = _setup_h2o_system()
n_virt = n_act - n_occ
ham = Hamiltonian(h1=np.zeros_like(h1eff), h2=h2_4d)

from dm_svd_embedding.transition_rdm import (
    _compute_det_create2_annih1_explicit, _compute_det_create1_annih2_explicit,
    _compute_det_creation_explicit, _compute_det_annihilation_explicit,
)

partition, full_dets = setup_partition(n_act, n_elec, n_occ, ms=0)

# 3A+1B n+1: source block n_A=2, dest n_A=3 (A subspace)
dets_src = partition[2]['a_dets']
dets_dst = partition[3]['a_dets']
idx_dst = partition[3]['a_index']
c2a1_A = _compute_det_create2_annih1_explicit(dets_src, dets_dst, idx_dst, n_occ)

bsrc = partition[2]['b_dets']
bdst = partition[3]['b_dets']
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


def brute_2e(src, dst, h2, norb):
    """chemist: 1/2 sum (pq|rs) a+_p,s a+_r,t a_s,t a_q,s"""
    val = 0.0
    for p in range(norb):
        for q in range(norb):
            for r in range(norb):
                for s in range(norb):
                    g = h2[p, q, r, s]
                    if abs(g) < 1e-14:
                        continue
                    for sig in ('a', 'b'):
                        for tau in ('a', 'b'):
                            res = apply_op(src, [('a', sig, q), ('a', tau, s),
                                                 ('c', tau, r), ('c', sig, p)])
                            if res is None:
                                continue
                            final, phase = res
                            if final == dst:
                                val += 0.5 * g * phase
    return val


def contraction_3body(ai, aj, bi, bj, jw_on=True):
    """My 3A+1B n+1 contraction (2 sub-cases) in determinant basis, with JW phase."""
    val = 0.0
    # JW phase: c2a1 net +1 spin -> (-1)^{n_spin(src)}
    na_src = dets_src[aj][0].bit_count()
    nb_src = dets_src[aj][1].bit_count()
    jw = {'aaa': (-1)**na_src, 'aba': (-1)**nb_src, 'abb': (-1)**na_src,
          'bab': (-1)**na_src, 'baa': (-1)**nb_src, 'bbb': (-1)**nb_src}
    if not jw_on:
        jw = {k: 1 for k in jw}
    # Case 1 (B = q, annihilate sigma): c2a1[sigma,tau,tau], pairs aaa/abb/baa/bbb
    pairs1 = [('aaa', 'a'), ('abb', 'a'), ('baa', 'b'), ('bbb', 'b')]
    for combo, spin in pairs1:
        for x in range(n_occ):
            for y in range(n_occ):
                for z in range(n_occ):
                    ta = c2a1_A[combo][ai, aj, x, y, z]
                    if abs(ta) < 1e-14:
                        continue
                    for b in range(n_virt):
                        tb = a1_B[spin][bi, bj, b]
                        if abs(tb) < 1e-14:
                            continue
                        v = h2_4d[x, b + n_occ, y, z]  # rB: B at slot 2
                        val += 0.5 * v * ta * tb * jw[combo]
    # Case 2 (B = s, annihilate tau): c2a1[sigma,tau,sigma], pairs aaa/aba/bab/bbb
    pairs2 = [('aaa', 'a'), ('aba', 'b'), ('bab', 'a'), ('bbb', 'b')]
    for combo, spin in pairs2:
        for x in range(n_occ):
            for y in range(n_occ):
                for z in range(n_occ):
                    ta = c2a1_A[combo][ai, aj, x, y, z]
                    if abs(ta) < 1e-14:
                        continue
                    for b in range(n_virt):
                        tb = a1_B[spin][bi, bj, b]
                        if abs(tb) < 1e-14:
                            continue
                        v = h2_4d[x, z, y, b + n_occ]  # sB: B at slot 4
                        val += 0.5 * v * ta * tb * jw[combo]
    return val


# find a few nonzero matrix elements and compare
print("comparing 3A+1B n+1 (n_A=2 -> 3) brute vs contraction:")
shown = 0
for aj in range(len(dets_src)):
    for bj in range(len(bsrc)):
        full_src = (dets_src[aj][0] | (bsrc[bj][0] << n_occ), dets_src[aj][1] | (bsrc[bj][1] << n_occ))
        for ai in range(len(dets_dst)):
            for bi in range(len(bdst)):
                full_dst = (dets_dst[ai][0] | (bdst[bi][0] << n_occ), dets_dst[ai][1] | (bdst[bi][1] << n_occ))
                bval = brute_2e(full_src, full_dst, h2_4d, n_act)
                cval = contraction_3body(ai, aj, bi, bj, jw_on=True)
                if abs(bval) > 1e-6 and abs(cval) > 1e-6:
                    print(f"  brute={bval:+.6f} contraction={cval:+.6f} ratio={cval/bval:+.4f}")
                    shown += 1
                    if shown >= 8:
                        break
            if shown >= 8:
                break
        if shown >= 8:
            break
    if shown >= 8:
        break
