#!/usr/bin/env python3
"""Focused: within-block n_A=3, direct vs exchange vs reference (projected)."""
import numpy as np
import sys
sys.path.insert(0, '/data/home/wangcx/krylov-dci')

from dm_svd_embedding.embedded_hamiltonian import (
    build_h_emb, _setup_h2o_system,
    compatible_product_mask, _project_physical_subspace,
)
from dm_svd_embedding.occ_virt_partition import setup_partition, build_block_matrices
from dm_svd_embedding.density_matrix import compute_schmidt_decomposition
from dm_svd_embedding.transition_rdm import compute_transition_matrices

(mol, mf, cas, q_idx, backend, h1eff, h2_4d, fcivec, ci_flat,
 n_act, n_elec, n_occ, na, nb, ecore, E_fci) = _setup_h2o_system()
n_virt = n_act - n_occ

partition, full_dets = setup_partition(n_act, n_elec, n_occ, ms=0)
C_blocks = build_block_matrices(partition, ci_flat)
schmidt = compute_schmidt_decomposition(C_blocks, eps=1e-3)

H_ref, _, decomps_ref = build_h_emb(
    schmidt, partition, q_idx, backend, h1eff, h2_4d, n_occ, n_act, verbose=False)
D = H_ref.shape[0]
HAB_ref = decomps_ref['HAB']

trans_A = compute_transition_matrices(partition, schmidt, n_occ, subspace='A', verbose=False)
trans_B = compute_transition_matrices(partition, schmidt, n_virt, subspace='B', verbose=False)

block_offsets = {}
off = 0
for nA in sorted(schmidt.keys()):
    block_offsets[nA] = off
    off += schmidt[nA]['r'] ** 2

n_A = 3
sd = schmidt[n_A]; r = sd['r']
bo = block_offsets[n_A]
sl = slice(bo, bo + r*r)

# reference within-block (already projected by build_h_emb)
ref_blk = HAB_ref[sl, sl]

# direct (same-spin) — coefficient +1, spin-summed
TA = trans_A.trans_1[n_A]  # (r,r,n_occ,n_occ)
TB = trans_B.trans_1[n_A]  # (r,r,n_virt,n_virt)
direct = np.zeros((r*r, r*r))
for a_src in range(r):
    for b_src in range(r):
        ik = a_src*r + b_src
        for a_dst in range(r):
            for b_dst in range(r):
                ib = a_dst*r + b_dst
                v = 0.0
                for i in range(n_occ):
                    for j in range(n_occ):
                        ta = TA[a_dst, a_src, i, j]
                        if abs(ta) < 1e-14: continue
                        for k in range(n_virt):
                            for l in range(n_virt):
                                tb = TB[b_dst, b_src, k, l]
                                if abs(tb) < 1e-14: continue
                                v += h2_4d[i, j, k+n_occ, l+n_occ] * ta * tb
                direct[ib, ik] = v

# exchange (different-spin) — coefficient -1, spin-explicit
TAe = trans_A.trans_1_explicit[n_A]
TBe = trans_B.trans_1_explicit[n_A]
spin_pairs = [('aa','aa'), ('bb','bb'), ('ab','ba'), ('ba','ab')]
exch = np.zeros((r*r, r*r))
for a_src in range(r):
    for b_src in range(r):
        ik = a_src*r + b_src
        for a_dst in range(r):
            for b_dst in range(r):
                ib = a_dst*r + b_dst
                v = 0.0
                for i in range(n_occ):
                    for l in range(n_occ):
                        for j in range(n_virt):
                            for k in range(n_virt):
                                integ = h2_4d[i, j+n_occ, k+n_occ, l]
                                if abs(integ) < 1e-14: continue
                                s = 0.0
                                for sA, sB in spin_pairs:
                                    s += TAe[sA][a_dst, a_src, i, l] * TBe[sB][b_dst, b_src, k, j]
                                v += integ * s
                exch[ib, ik] = -v

total = direct + exch

print(f"n_A={n_A}, r={r}")
print(f"  ||ref (within-block)|| = {np.linalg.norm(ref_blk):.6f}")
print(f"  ||direct||            = {np.linalg.norm(direct):.6f}")
print(f"  ||exchange||          = {np.linalg.norm(exch):.6f}")
print(f"  ||direct+exchange||   = {np.linalg.norm(total):.6f}")
print(f"  max|total - ref|      = {np.abs(total - ref_blk).max():.6f}")

# how much of ref is direct-like vs exchange-like? compare component-wise
print(f"\n  max|direct - ref|     = {np.abs(direct - ref_blk).max():.6f}")

# print the reference block and the total for the first few entries
print("\n  ref_blk (upper 4x4):")
print(np.round(ref_blk[:4,:4], 4))
print("  total (upper 4x4):")
print(np.round(total[:4,:4], 4))
print("  direct (upper 4x4):")
print(np.round(direct[:4,:4], 4))
