#!/usr/bin/env python3
"""Isolate the Path-C HA bug: compare delta-placement vs direct W^T H_A W."""
import numpy as np
import sys
sys.path.insert(0, '/data/home/wangcx/krylov-dci')

from dm_svd_embedding.embedded_hamiltonian import (
    build_h_emb, _setup_h2o_system, _build_subspace_hamiltonian,
    _extract_subspace_integrals,
)
from dm_svd_embedding.occ_virt_partition import setup_partition, build_block_matrices
from dm_svd_embedding.density_matrix import compute_schmidt_decomposition

(mol, mf, cas, q_idx, backend, h1eff, h2_4d, fcivec, ci_flat,
 n_act, n_elec, n_occ, na, nb, ecore, E_fci) = _setup_h2o_system()
n_virt = n_act - n_occ

partition, full_dets = setup_partition(n_act, n_elec, n_occ, ms=0)
C_blocks = build_block_matrices(partition, ci_flat)
schmidt = compute_schmidt_decomposition(C_blocks, eps=1e-3)

n_A = 3
sd = schmidt[n_A]; r = sd['r']; blk = partition[n_A]
U, V = sd['U'], sd['V']
a_dets = blk['a_dets']

h1_A, h2_A = _extract_subspace_integrals(h1eff, h2_4d, np.arange(n_occ))
aA0, aB0 = a_dets[0]
HA_det = _build_subspace_hamiltonian(a_dets, h1_A, h2_A, n_occ, aA0.bit_count(), aB0.bit_count())
HA_schmidt = U.T @ HA_det @ U   # (r,r)

# Path C placement: H[gamma*beta, alpha*beta] = HA_schmidt[gamma, alpha]
HA_placed = np.zeros((r*r, r*r))
for alpha in range(r):
    for beta in range(r):
        k = alpha*r + beta
        for gamma in range(r):
            l = gamma*r + beta
            HA_placed[l, k] = HA_schmidt[gamma, alpha]

# build_h_emb reference HA block
_, _, decomps = build_h_emb(schmidt, partition, q_idx, backend, h1eff, h2_4d, n_occ, n_act, verbose=False)
HA_ref = decomps['HA']
block_offsets = {}
off = 0
for nA_ in sorted(schmidt.keys()):
    block_offsets[nA_] = off; off += schmidt[nA_]['r']**2
bo = block_offsets[n_A]
HA_blk = HA_ref[bo:bo+r*r, bo:bo+r*r]

print(f"n_A={n_A}, r={r}")
print(f"  HA_schmidt (U^T HA_det U) shape = {HA_schmidt.shape}")
print(f"  max|HA_placed - HA_blk| = {np.abs(HA_placed - HA_blk).max():.2e}")
print(f"  ||HA_placed|| = {np.linalg.norm(HA_placed):.6f}")
print(f"  ||HA_blk||    = {np.linalg.norm(HA_blk):.6f}")

# If they match, then the bug is elsewhere. Check whether HA_det itself is
# block-diagonal in spin-sector (na_A, nb_A):
print(f"\n  A-space spin sectors (na_A, nb_A):")
sectors = {}
for i, (aA, aB) in enumerate(a_dets):
    s = (aA.bit_count(), aB.bit_count())
    sectors.setdefault(s, []).append(i)
for s, idx in sorted(sectors.items()):
    print(f"    {s}: {len(idx)} dets")
# check block-diagonality of HA_det
max_cross = 0.0
for i in range(len(a_dets)):
    si = (a_dets[i][0].bit_count(), a_dets[i][1].bit_count())
    for j in range(len(a_dets)):
        sj = (a_dets[j][0].bit_count(), a_dets[j][1].bit_count())
        if si != sj:
            max_cross = max(max_cross, abs(HA_det[i,j]))
print(f"  max |HA_det| cross-sector = {max_cross:.2e}  (0 => block-diagonal)")

# Does U mix spin sectors?  Check support of each singular vector.
print(f"\n  U spin-sector support per column:")
for a in range(r):
    sup = set()
    for i in range(len(a_dets)):
        if abs(U[i,a]) > 1e-10:
            sup.add((a_dets[i][0].bit_count(), a_dets[i][1].bit_count()))
    print(f"    column {a}: sectors {sup}")
