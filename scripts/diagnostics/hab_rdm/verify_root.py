#!/usr/bin/env python3
"""
Definitive: locate the bug in build_h_emb's HA/HB/HAB decomposition.

Compare, for within-block n_A=3:
  (a) H_emb_block (sigma-vector)           [build_h_emb return value]
  (b) H_full_schmidt (independent Slater-Condon)
  (c) HA_schmidt, HB_schmidt (Path C)
  (d) W^T H_A_full W, W^T H_B_full W (independent intra-A/B)
  (e) H_AB = (a) - HA - HB  vs  independent H_cross
"""
import numpy as np
import sys
sys.path.insert(0, '/data/home/wangcx/krylov-dci')

from dm_svd_embedding.embedded_hamiltonian import (
    build_h_emb, _setup_h2o_system, _build_subspace_hamiltonian,
    _extract_subspace_integrals,
)
from dm_svd_embedding.occ_virt_partition import setup_partition, build_block_matrices
from dm_svd_embedding.density_matrix import compute_schmidt_decomposition
from src.hamiltonian import Hamiltonian

(mol, mf, cas, q_idx, backend, h1eff, h2_4d, fcivec, ci_flat,
 n_act, n_elec, n_occ, na, nb, ecore, E_fci) = _setup_h2o_system()
n_virt = n_act - n_occ

partition, full_dets = setup_partition(n_act, n_elec, n_occ, ms=0)
C_blocks = build_block_matrices(partition, ci_flat)
schmidt = compute_schmidt_decomposition(C_blocks, eps=1e-3)

ham = Hamiltonian(h1=h1eff, h2=h2_4d, E_nuc=0.0, E_HF=0.0)

n_A = 3
sd = schmidt[n_A]; r = sd['r']; blk = partition[n_A]
U, V = sd['U'], sd['V']
a_dets = blk['a_dets']; b_dets = blk['b_dets']
da = len(a_dets); db = len(b_dets)

def prod_to_full(i_a, k_b):
    aA, aB = a_dets[i_a]; bA, bB = b_dets[k_b]
    return (aA | (bA << n_occ), aB | (bB << n_occ))

valid_ik = []
for i in range(da):
    for k in range(db):
        f = prod_to_full(i, k)
        if f[0].bit_count() == na and f[1].bit_count() == nb:
            valid_ik.append((i, k))
Nv = len(valid_ik)

H_full = np.zeros((Nv, Nv))
for x in range(Nv):
    i, k = valid_ik[x]; fx = prod_to_full(i, k)
    for y in range(x, Nv):
        j, l = valid_ik[y]; fy = prod_to_full(j, l)
        h = ham.matrix_element(fx, fy)
        H_full[x, y] = h; H_full[y, x] = h

h1_A, h2_A = _extract_subspace_integrals(h1eff, h2_4d, np.arange(n_occ))
h1_B, h2_B = _extract_subspace_integrals(h1eff, h2_4d, np.arange(n_occ, n_act))
aA0, aB0 = a_dets[0]
HA_det = _build_subspace_hamiltonian(a_dets, h1_A, h2_A, n_occ, aA0.bit_count(), aB0.bit_count())
bB0, bB0b = b_dets[0]
HB_det = _build_subspace_hamiltonian(b_dets, h1_B, h2_B, n_virt, bB0.bit_count(), bB0b.bit_count())

H_A_full = np.zeros((Nv, Nv)); H_B_full = np.zeros((Nv, Nv))
for x in range(Nv):
    i, k = valid_ik[x]
    for y in range(Nv):
        j, l = valid_ik[y]
        if k == l: H_A_full[x, y] += HA_det[i, j]
        if i == j: H_B_full[x, y] += HB_det[k, l]

H_cross = H_full - H_A_full - H_B_full

W = np.zeros((Nv, r*r))
for x in range(Nv):
    i, k = valid_ik[x]
    for a_ in range(r):
        for b_ in range(r):
            W[x, a_*r + b_] = U[i, a_] * V[k, b_]

H_full_sch = W.T @ H_full @ W
H_A_sch = W.T @ H_A_full @ W
H_B_sch = W.T @ H_B_full @ W
H_cross_sch = W.T @ H_cross @ W

# build_h_emb reference
H_emb_full, _, decomps = build_h_emb(schmidt, partition, q_idx, backend, h1eff, h2_4d, n_occ, n_act, verbose=False)
HA_ref = decomps['HA']; HB_ref = decomps['HB']; HAB_ref = decomps['HAB']

block_offsets = {}
off = 0
for nA_ in sorted(schmidt.keys()):
    block_offsets[nA_] = off; off += schmidt[nA_]['r']**2
bo = block_offsets[n_A]
sl = slice(bo, bo+r*r)

H_emb_blk = H_emb_full[sl, sl]
HA_blk = HA_ref[sl, sl]; HB_blk = HB_ref[sl, sl]; HAB_blk = HAB_ref[sl, sl]

print(f"n_A={n_A}, r={r}, Nv={Nv}")
print(f"  [a] H_emb (sigma) block  = {np.linalg.norm(H_emb_blk):.6f}")
print(f"  [b] H_full_sch (indep)    = {np.linalg.norm(H_full_sch):.6f}")
print(f"      max|a - b|            = {np.abs(H_emb_blk - H_full_sch).max():.2e}")
print()
print(f"  [c] HA (Path C) block     = {np.linalg.norm(HA_blk):.6f}")
print(f"  [d] H_A_sch (indep)       = {np.linalg.norm(H_A_sch):.6f}")
print(f"      max|HA - H_A_sch|     = {np.abs(HA_blk - H_A_sch).max():.2e}")
print(f"  [c] HB (Path C) block     = {np.linalg.norm(HB_blk):.6f}")
print(f"  [d] H_B_sch (indep)       = {np.linalg.norm(H_B_sch):.6f}")
print(f"      max|HB - H_B_sch|     = {np.abs(HB_blk - H_B_sch).max():.2e}")
print()
print(f"  [e] HAB (sigma-HA-HB)     = {np.linalg.norm(HAB_blk):.6f}")
print(f"  [e] H_cross_sch (indep)   = {np.linalg.norm(H_cross_sch):.6f}")
print(f"      max|HAB - H_cross_sch|= {np.abs(HAB_blk - H_cross_sch).max():.2e}")

# does H_emb = HA + HB + HAB (should reconstruct)?
recon = HA_blk + HB_blk + HAB_blk
print(f"\n  max|H_emb - (HA+HB+HAB)|  = {np.abs(H_emb_blk - recon).max():.2e}")
# does independent reconstruct?
recon2 = H_A_sch + H_B_sch + H_cross_sch
print(f"  max|H_full_sch - (A+B+cross)| = {np.abs(H_full_sch - recon2).max():.2e}")
