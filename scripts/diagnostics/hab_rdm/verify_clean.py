#!/usr/bin/env python3
"""
Clean within-block H_AB decomposition for n_A=3, with CORRECT spin-sector handling.

Compare three things:
  (a) sigma-based reference HAB_ref block  (trusted: reproduces FCI to 0.15 mH)
  (b) direct (spin-summed) + exchange (spin-EXPLICIT) via transition matrices
  (c) residual (a)-(b), analyzed by (A-exc, B-exc) pattern
"""
import numpy as np
import sys
sys.path.insert(0, '/data/home/wangcx/krylov-dci')

from dm_svd_embedding.embedded_hamiltonian import (
    build_h_emb, _setup_h2o_system, _expand_schmidt_product_to_ci_matrix,
)
from dm_svd_embedding.occ_virt_partition import setup_partition, build_block_matrices
from dm_svd_embedding.density_matrix import compute_schmidt_decomposition
from dm_svd_embedding.transition_rdm import _create_sign, _annihilate_sign

(mol, mf, cas, q_idx, backend, h1eff, h2_4d, fcivec, ci_flat,
 n_act, n_elec, n_occ, na, nb, ecore, E_fci) = _setup_h2o_system()
n_virt = n_act - n_occ

partition, full_dets = setup_partition(n_act, n_elec, n_occ, ms=0)
C_blocks = build_block_matrices(partition, ci_flat)
schmidt = compute_schmidt_decomposition(C_blocks, eps=1e-3)

H_ref, basis_ref, decomps_ref = build_h_emb(
    schmidt, partition, q_idx, backend, h1eff, h2_4d, n_occ, n_act, verbose=False)
HAB_ref = decomps_ref['HAB']

block_offsets = {}
off = 0
for nA in sorted(schmidt.keys()):
    block_offsets[nA] = off
    off += schmidt[nA]['r'] ** 2

n_A = 3
sd = schmidt[n_A]; r = sd['r']; blk = partition[n_A]
U, V = sd['U'], sd['V']
a_dets = blk['a_dets']; b_dets = blk['b_dets']
a_index = blk['a_index']; b_index = blk['b_index']
bo = block_offsets[n_A]

# ---- spin-explicit 1-body transition matrices ----
def spin_explicit_1body(dets, idx, n_orb):
    d = len(dets)
    comps = {k: np.zeros((d, d, n_orb, n_orb)) for k in ('aa','ab','ba','bb')}
    for jj, (aA_j, bA_j) in enumerate(dets):
        for q in range(n_orb):
            if (aA_j >> q) & 1:  # aa: annihilate alpha q, create alpha p
                phq, a1 = _annihilate_sign(aA_j, q)
                for p in range(n_orb):
                    if (a1 >> p) & 1: continue
                    if p == q:
                        comps['aa'][jj, jj, p, q] = 1.0; continue
                    php, a2 = _create_sign(a1, p)
                    if php == 0: continue
                    ii = idx.get((a2, bA_j))
                    if ii is not None: comps['aa'][ii, jj, p, q] = phq * php
        for q in range(n_orb):
            if (bA_j >> q) & 1:  # bb
                phq, b1 = _annihilate_sign(bA_j, q)
                for p in range(n_orb):
                    if (b1 >> p) & 1: continue
                    if p == q:
                        comps['bb'][jj, jj, p, q] = 1.0; continue
                    php, b2 = _create_sign(b1, p)
                    if php == 0: continue
                    ii = idx.get((aA_j, b2))
                    if ii is not None: comps['bb'][ii, jj, p, q] = phq * php
        for q in range(n_orb):
            if (bA_j >> q) & 1:  # ab: annihilate beta q, create alpha p
                phq, b1 = _annihilate_sign(bA_j, q)
                for p in range(n_orb):
                    if (aA_j >> p) & 1: continue
                    php, a2 = _create_sign(aA_j, p)
                    if php == 0: continue
                    ii = idx.get((a2, b1))
                    if ii is not None: comps['ab'][ii, jj, p, q] = phq * php
        for q in range(n_orb):
            if (aA_j >> q) & 1:  # ba: annihilate alpha q, create beta p
                phq, a1 = _annihilate_sign(aA_j, q)
                for p in range(n_orb):
                    if (bA_j >> p) & 1: continue
                    php, b2 = _create_sign(bA_j, p)
                    if php == 0: continue
                    ii = idx.get((a1, b2))
                    if ii is not None: comps['ba'][ii, jj, p, q] = phq * php
    return comps

compA = spin_explicit_1body(a_dets, a_index, n_occ)
compB = spin_explicit_1body(b_dets, b_index, n_virt)
TA = {k: np.einsum('ijpq,ia,jg->agpq', compA[k], U, U) for k in compA}
TB = {k: np.einsum('ijpq,ia,jg->agpq', compB[k], V, V) for k in compB}
TA_sum = TA['aa'] + TA['bb']
TB_sum = TB['aa'] + TB['bb']

# ---- direct (spin-summed) ----
direct = np.zeros((r*r, r*r))
for a_src in range(r):
    for b_src in range(r):
        ik = a_src * r + b_src
        for a_dst in range(r):
            for b_dst in range(r):
                ib = a_dst * r + b_dst
                val = 0.0
                for i in range(n_occ):
                    for j in range(n_occ):
                        ta = TA_sum[a_dst, a_src, i, j]
                        if abs(ta) < 1e-14: continue
                        for k in range(n_virt):
                            for l in range(n_virt):
                                tb = TB_sum[b_dst, b_src, k, l]
                                if abs(tb) < 1e-14: continue
                                val += h2_4d[i, j, k+n_occ, l+n_occ] * ta * tb
                direct[ib, ik] = val

# ---- exchange (spin-explicit) ----
exch = np.zeros((r*r, r*r))
spin_pairs = [('aa','aa'), ('bb','bb'), ('ab','ba'), ('ba','ab')]
for a_src in range(r):
    for b_src in range(r):
        ik = a_src * r + b_src
        for a_dst in range(r):
            for b_dst in range(r):
                ib = a_dst * r + b_dst
                val = 0.0
                for i in range(n_occ):
                    for l in range(n_occ):
                        for j in range(n_virt):
                            for k in range(n_virt):
                                v = h2_4d[i, j+n_occ, k+n_occ, l]
                                if abs(v) < 1e-14: continue
                                s = 0.0
                                for sA, sB in spin_pairs:
                                    s += TA[sA][a_dst, a_src, i, l] * TB[sB][b_dst, b_src, k, j]
                                val += v * s
                exch[ib, ik] = -val

total = direct + exch
ref_block = HAB_ref[bo:bo+r*r, bo:bo+r*r]
resid = ref_block - total

print(f"n_A={n_A}, r={r}")
print(f"  ||ref||      = {np.linalg.norm(ref_block):.6f}")
print(f"  ||direct||   = {np.linalg.norm(direct):.6f}")
print(f"  ||exchange|| = {np.linalg.norm(exch):.6f}")
print(f"  ||total||    = {np.linalg.norm(total):.6f}")
print(f"  ||residual|| = {np.linalg.norm(resid):.6f}")

# analyze residual: on-diagonal vs off-diagonal in Schmidt product basis
diag_mask = np.eye(r*r, dtype=bool)
print(f"  ||resid diag||    = {np.linalg.norm(resid[diag_mask]):.6f}")
print(f"  ||resid offdiag|| = {np.linalg.norm(resid[~diag_mask]):.6f}")

# largest residual elements
idx = np.argsort(np.abs(resid).ravel())[-5:][::-1]
print("  top-5 residual elements (flat_idx, value):")
for ii in idx:
    print(f"    {ii}: {resid.ravel()[ii]:+.6f}")
