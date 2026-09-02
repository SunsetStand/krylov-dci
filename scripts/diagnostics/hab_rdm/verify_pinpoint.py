#!/usr/bin/env python3
"""
Pin down the direct/exchange discrepancy.

(1) Verify aa+bb == module trans_1 (known-correct spin-summed).
(2) Verify ab, ba against explicit determinant expansion (ground truth).
(3) For the largest-residual diagonal element, print ref/direct/exchange.
"""
import numpy as np
import sys
sys.path.insert(0, '/data/home/wangcx/krylov-dci')

from dm_svd_embedding.embedded_hamiltonian import (
    build_h_emb, _setup_h2o_system,
)
from dm_svd_embedding.occ_virt_partition import setup_partition, build_block_matrices
from dm_svd_embedding.density_matrix import compute_schmidt_decomposition
from dm_svd_embedding.transition_rdm import (
    compute_transition_matrices, _create_sign, _annihilate_sign, _compute_det_1body_transition,
)

(mol, mf, cas, q_idx, backend, h1eff, h2_4d, fcivec, ci_flat,
 n_act, n_elec, n_occ, na, nb, ecore, E_fci) = _setup_h2o_system()
n_virt = n_act - n_occ

partition, full_dets = setup_partition(n_act, n_elec, n_occ, ms=0)
C_blocks = build_block_matrices(partition, ci_flat)
schmidt = compute_schmidt_decomposition(C_blocks, eps=1e-3)

trans_A = compute_transition_matrices(partition, schmidt, n_occ, subspace='A', verbose=False)
trans_B = compute_transition_matrices(partition, schmidt, n_virt, subspace='B', verbose=False)

n_A = 3
sd = schmidt[n_A]; r = sd['r']; blk = partition[n_A]
U, V = sd['U'], sd['V']
a_dets = blk['a_dets']; b_dets = blk['b_dets']
a_index = blk['a_index']; b_index = blk['b_index']

# ---- my spin-explicit (copy of verify_clean) ----
def spin_explicit_1body(dets, idx, n_orb):
    d = len(dets)
    comps = {k: np.zeros((d, d, n_orb, n_orb)) for k in ('aa','ab','ba','bb')}
    for jj, (aA_j, bA_j) in enumerate(dets):
        for q in range(n_orb):
            if (aA_j >> q) & 1:
                phq, a1 = _annihilate_sign(aA_j, q)
                for p in range(n_orb):
                    if (a1 >> p) & 1: continue
                    if p == q: comps['aa'][jj,jj,p,q]=1.0; continue
                    php, a2 = _create_sign(a1, p)
                    if php==0: continue
                    ii = idx.get((a2, bA_j))
                    if ii is not None: comps['aa'][ii,jj,p,q]=phq*php
        for q in range(n_orb):
            if (bA_j >> q) & 1:
                phq, b1 = _annihilate_sign(bA_j, q)
                for p in range(n_orb):
                    if (b1 >> p) & 1: continue
                    if p == q: comps['bb'][jj,jj,p,q]=1.0; continue
                    php, b2 = _create_sign(b1, p)
                    if php==0: continue
                    ii = idx.get((aA_j, b2))
                    if ii is not None: comps['bb'][ii,jj,p,q]=phq*php
        for q in range(n_orb):
            if (bA_j >> q) & 1:
                phq, b1 = _annihilate_sign(bA_j, q)
                for p in range(n_orb):
                    if (aA_j >> p) & 1: continue
                    php, a2 = _create_sign(aA_j, p)
                    if php==0: continue
                    ii = idx.get((a2, b1))
                    if ii is not None: comps['ab'][ii,jj,p,q]=phq*php
        for q in range(n_orb):
            if (aA_j >> q) & 1:
                phq, a1 = _annihilate_sign(aA_j, q)
                for p in range(n_orb):
                    if (bA_j >> p) & 1: continue
                    php, b2 = _create_sign(bA_j, p)
                    if php==0: continue
                    ii = idx.get((a1, b2))
                    if ii is not None: comps['ba'][ii,jj,p,q]=phq*php
    return comps

compA = spin_explicit_1body(a_dets, a_index, n_occ)
compB = spin_explicit_1body(b_dets, b_index, n_virt)

# --- (1) aa+bb vs module trans_1 ---
TA_module = trans_A.trans_1[n_A]  # (r,r,n_occ,n_occ) — module, verified
TA_mine = np.einsum('ijpq,ia,jg->agpq', compA['aa']+compA['bb'], U, U)
print(f"[1] A-space aa+bb vs module trans_1: max diff = {np.abs(TA_module - TA_mine).max():.2e}")

TB_module = trans_B.trans_1[n_A]
TB_mine = np.einsum('ijpq,ia,jg->agpq', compB['aa']+compB['bb'], V, V)
print(f"    B-space aa+bb vs module trans_1: max diff = {np.abs(TB_module - TB_mine).max():.2e}")

# --- (2) ab/ba against direct det expansion ---
# Ground truth: T_det = _compute_det_1body_transition (spin-summed only).
# For spin-flip, verify against manual: <a_i|a^+_p,alpha a_q,beta|a_j> etc.
# Build module's determinant-basis 1-body (alpha-only and beta-only separately)
def alpha_only_1body(dets, idx, n_orb):
    d = len(dets); T = np.zeros((d,d,n_orb,n_orb))
    for jj,(aA_j,bA_j) in enumerate(dets):
        for q in range(n_orb):
            if not ((aA_j>>q)&1): continue
            phq,a1 = _annihilate_sign(aA_j,q)
            for p in range(n_orb):
                if (a1>>p)&1: continue
                if p==q: T[jj,jj,p,q]=1.0; continue
                php,a2=_create_sign(a1,p)
                if php==0: continue
                ii=idx.get((a2,bA_j))
                if ii is not None: T[ii,jj,p,q]=phq*php
    return T

T_aa_det = alpha_only_1body(a_dets, a_index, n_occ)
print(f"[2] A-space 'aa' vs alpha-only det expansion: max diff = {np.abs(compA['aa'] - T_aa_det).max():.2e}")

# verify 'ab' = a+_p,alpha a_q,beta via brute: for each det j, annihilate beta q, create alpha p
d = len(a_dets)
T_ab_ref = np.zeros((d,d,n_occ,n_occ))
for jj,(aA_j,bA_j) in enumerate(a_dets):
    for q in range(n_occ):
        if not ((bA_j>>q)&1): continue
        phq,b1 = _annihilate_sign(bA_j,q)
        for p in range(n_occ):
            if (aA_j>>p)&1: continue
            php,a2 = _create_sign(aA_j,p)
            if php==0: continue
            ii = a_index.get((a2,b1))
            if ii is not None: T_ab_ref[ii,jj,p,q] = phq*php
print(f"    A-space 'ab' vs manual det expansion: max diff = {np.abs(compA['ab'] - T_ab_ref).max():.2e}")

# --- (3) reference H_AB block, direct+exchange, residual ---
H_ref, _, decomps_ref = build_h_emb(schmidt, partition, q_idx, backend, h1eff, h2_4d, n_occ, n_act, verbose=False)
HAB_ref = decomps_ref['HAB']
block_offsets = {}
off=0
for nA_ in sorted(schmidt.keys()):
    block_offsets[nA_]=off; off += schmidt[nA_]['r']**2
bo = block_offsets[n_A]

TA = {k: np.einsum('ijpq,ia,jg->agpq', compA[k], U, U) for k in compA}
TB = {k: np.einsum('ijpq,ia,jg->agpq', compB[k], V, V) for k in compB}
TA_sum = TA['aa']+TA['bb']; TB_sum = TB['aa']+TB['bb']

direct = np.zeros((r*r, r*r)); exch = np.zeros((r*r, r*r))
spin_pairs = [('aa','aa'),('bb','bb'),('ab','ba'),('ba','ab')]
for a_src in range(r):
    for b_src in range(r):
        ik = a_src*r+b_src
        for a_dst in range(r):
            for b_dst in range(r):
                ib = a_dst*r+b_dst
                vd = 0.0
                for i in range(n_occ):
                    for j in range(n_occ):
                        ta = TA_sum[a_dst,a_src,i,j]
                        if abs(ta)<1e-14: continue
                        for k in range(n_virt):
                            for l in range(n_virt):
                                tb = TB_sum[b_dst,b_src,k,l]
                                if abs(tb)<1e-14: continue
                                vd += h2_4d[i,j,k+n_occ,l+n_occ]*ta*tb
                direct[ib,ik]=vd
                ve = 0.0
                for i in range(n_occ):
                    for l in range(n_occ):
                        for j in range(n_virt):
                            for k in range(n_virt):
                                v = h2_4d[i,j+n_occ,k+n_occ,l]
                                if abs(v)<1e-14: continue
                                s = 0.0
                                for sA,sB in spin_pairs:
                                    s += TA[sA][a_dst,a_src,i,l]*TB[sB][b_dst,b_src,k,j]
                                ve += v*s
                exch[ib,ik]=-ve

ref_block = HAB_ref[bo:bo+r*r, bo:bo+r*r]
resid = ref_block - direct - exch

# find largest residual DIAGONAL element
diag = np.diag(resid)
ii = np.argmax(np.abs(diag))
a_, b_ = ii//r, ii%r
print(f"\n[3] largest residual diag: (a={a_},b={b_})")
print(f"    ref     = {ref_block[ii,ii]:+.8f}")
print(f"    direct  = {direct[ii,ii]:+.8f}")
print(f"    exchange= {exch[ii,ii]:+.8f}")
print(f"    residual= {resid[ii,ii]:+.8f}")

# Also check: what does the FULL sigma-based H_AB diagonal look like vs my total?
print(f"\n    ||ref diag||    = {np.linalg.norm(np.diag(ref_block)):.4f}")
print(f"    ||direct diag|| = {np.linalg.norm(np.diag(direct)):.4f}")
print(f"    ||exch diag||   = {np.linalg.norm(np.diag(exch)):.4f}")
print(f"    ||resid diag||  = {np.linalg.norm(diag):.4f}")
