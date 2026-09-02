#!/usr/bin/env python3
"""
Fully independent ground truth for the within-block cross Hamiltonian H_AB.

Build the FULL Hamiltonian in the product basis |a_i b_k> (a_dets x b_dets)
via Slater-Condon (trusted).  Then compute H_cross = H_full - H_A - H_B
where H_A/H_B are the pure intra-block parts (also via Slater-Condon on the
A/B subspaces).  Project to Schmidt basis via W = U (x) V.

Compare three things:
  (a) H_cross_schmidt (this independent ground truth)
  (b) sigma-based HAB_ref from build_h_emb
  (c) my direct + exchange
"""
import numpy as np
import sys
sys.path.insert(0, '/data/home/wangcx/krylov-dci')

from dm_svd_embedding.embedded_hamiltonian import (
    build_h_emb, _setup_h2o_system,
)
from dm_svd_embedding.occ_virt_partition import setup_partition, build_block_matrices
from dm_svd_embedding.density_matrix import compute_schmidt_decomposition
from dm_svd_embedding.transition_rdm import _create_sign, _annihilate_sign
from src.hamiltonian import Hamiltonian, _unpack_4fold

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

# ---- full H in product basis (da*db)^2, only valid spin-sector pairs ----
# IMPORTANT: only keep product dets whose total alpha=na, beta=nb (valid Sz sector)
valid_ik = []
for i in range(da):
    for k in range(db):
        f = prod_to_full(i, k)
        na_full = f[0].bit_count(); nb_full = f[1].bit_count()
        if na_full == na and nb_full == nb:
            valid_ik.append((i, k))
print(f"valid product dets: {len(valid_ik)} / {da*db}")

# build H_full over valid product dets
Nv = len(valid_ik)
H_full = np.zeros((Nv, Nv))
for x in range(Nv):
    i, k = valid_ik[x]
    fx = prod_to_full(i, k)
    for y in range(x, Nv):
        j, l = valid_ik[y]
        fy = prod_to_full(j, l)
        h = ham.matrix_element(fx, fy)
        H_full[x, y] = h
        H_full[y, x] = h

# ---- H_A and H_B in product basis (over the SAME valid dets) ----
# H_A: intra-A only. <a_i b_k | H_A | a_j b_l> = delta_{k,l} <a_i|H_A|a_j>
# Build A-space Hamiltonian in a_dets basis
from dm_svd_embedding.embedded_hamiltonian import _build_subspace_hamiltonian, _extract_subspace_integrals
h1_A, h2_A = _extract_subspace_integrals(h1eff, h2_4d, np.arange(n_occ))
aA0, aB0 = a_dets[0]
nA_a = aA0.bit_count(); nA_b = aB0.bit_count()
HA_det = _build_subspace_hamiltonian(a_dets, h1_A, h2_A, n_occ, nA_a, nA_b)
h1_B, h2_B = _extract_subspace_integrals(h1eff, h2_4d, np.arange(n_occ, n_act))
bB0, bB0b = b_dets[0]
nB_a = bB0.bit_count(); nB_b = bB0b.bit_count()
HB_det = _build_subspace_hamiltonian(b_dets, h1_B, h2_B, n_virt, nB_a, nB_b)

# map valid_ik -> (i,k); build index lookup
ik_to_x = {ik: x for x, ik in enumerate(valid_ik)}

H_A_full = np.zeros((Nv, Nv))
H_B_full = np.zeros((Nv, Nv))
for x in range(Nv):
    i, k = valid_ik[x]
    for y in range(Nv):
        j, l = valid_ik[y]
        if k == l:
            H_A_full[x, y] += HA_det[i, j]
        if i == j:
            H_B_full[x, y] += HB_det[k, l]

H_cross = H_full - H_A_full - H_B_full

# ---- project to Schmidt basis ----
# Schmidt product state |A_alpha B_beta> = sum_{ik} U[i,alpha] V[k,beta] |a_i b_k>
# But only valid ik contribute. Build W matrix (Nv, r*r).
W = np.zeros((Nv, r*r))
for x in range(Nv):
    i, k = valid_ik[x]
    for a_ in range(r):
        for b_ in range(r):
            W[x, a_*r + b_] = U[i, a_] * V[k, b_]

H_cross_schmidt = W.T @ H_cross @ W
H_full_schmidt = W.T @ H_full @ W

# ---- compare with build_h_emb reference ----
H_ref, _, decomps_ref = build_h_emb(schmidt, partition, q_idx, backend, h1eff, h2_4d, n_occ, n_act, verbose=False)
HAB_ref = decomps_ref['HAB']
block_offsets = {}
off = 0
for nA_ in sorted(schmidt.keys()):
    block_offsets[nA_] = off; off += schmidt[nA_]['r']**2
bo = block_offsets[n_A]
ref_block = HAB_ref[bo:bo+r*r, bo:bo+r*r]

print(f"\n=== within-block n_A={n_A} ===")
print(f"  ||H_cross_schmidt (independent)|| = {np.linalg.norm(H_cross_schmidt):.6f}")
print(f"  ||HAB_ref (build_h_emb)||         = {np.linalg.norm(ref_block):.6f}")
print(f"  ||H_full_schmidt||                = {np.linalg.norm(H_full_schmidt):.6f}")

# first check: independent cross == build_h_emb ref?
print(f"  max|H_cross_schmidt - ref_block|  = {np.abs(H_cross_schmidt - ref_block).max():.2e}")

# check H_full_schmidt == sigma-based H_emb block?
# build_h_emb doesn't return H_emb, but HAB_ref + HA + HB should equal H_emb block
# Instead verify H_full_schmidt is symmetric and matches sigma via a different route:
# sigma-vector H_emb block = W^T H_full W (since sigma = H v exactly)
# We already trust Slater-Condon == sigma (verified). So H_full_schmidt is the truth.

# print the reference vs independent diag
diag_ind = np.diag(H_cross_schmidt)
diag_ref = np.diag(ref_block)
print(f"\n  ||diag(cross independent)|| = {np.linalg.norm(diag_ind):.6f}")
print(f"  ||diag(ref)||               = {np.linalg.norm(diag_ref):.6f}")
print(f"  ||diag diff||               = {np.linalg.norm(diag_ind - diag_ref):.6f}")
