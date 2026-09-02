#!/usr/bin/env python3
"""Check: is the 1e cross block of h1eff nonzero?  Is ref HAB 5->6 really 0?"""
import numpy as np
import sys
sys.path.insert(0, '/data/home/wangcx/krylov-dci')

from dm_svd_embedding.embedded_hamiltonian import (
    build_h_emb, _setup_h2o_system, compatible_product_mask, _project_physical_subspace,
)
from dm_svd_embedding.occ_virt_partition import setup_partition, build_block_matrices
from dm_svd_embedding.density_matrix import compute_schmidt_decomposition

(mol, mf, cas, q_idx, backend, h1eff, h2_4d, fcivec, ci_flat,
 n_act, n_elec, n_occ, na, nb, ecore, E_fci) = _setup_h2o_system()

print(f"n_occ={n_occ}, n_act={n_act}")
print("h1eff (full 5x5):")
print(np.round(h1eff, 5))
print(f"\nA-B block (p<{n_occ}, r>={n_occ}): norm = {np.linalg.norm(h1eff[:n_occ, n_occ:]):.6f}")

# Is h1eff = core hamiltonian or Fock? check the object
print(f"\nh1eff type: {type(h1eff)}, is hermitian: {np.abs(h1eff-h1eff.T).max():.2e}")
print(f"ecore = {ecore:.8f}")

# Build reference and check the 5->6 block exactly
partition, full_dets = setup_partition(n_act, n_elec, n_occ, ms=0)
C_blocks = build_block_matrices(partition, ci_flat)
schmidt = compute_schmidt_decomposition(C_blocks, eps=1e-3)

H_ref, _, decomps_ref = build_h_emb(
    schmidt, partition, q_idx, backend, h1eff, h2_4d, n_occ, n_act, verbose=False)
HAB_ref = decomps_ref['HAB']

mask = compatible_product_mask(schmidt, partition, (na, nb))
HAB_ref = _project_physical_subspace(HAB_ref, mask)

# block offsets
off = 0
bo = {}
for nA in sorted(schmidt.keys()):
    bo[nA] = off
    off += schmidt[nA]['r']**2

for nA in sorted(schmidt.keys()):
    print(f"n_A={nA}: r={schmidt[nA]['r']}")

# 5->6 block
r5, r6 = schmidt[5]['r'], schmidt[6]['r']
os5, os6 = bo[5], bo[6]
blk56 = HAB_ref[os6:os6+r6*r6, os5:os5+r5*r5]
print(f"\nReference HAB[6,5] block ({r6*r6}x{r5*r5}):")
print(np.round(blk56, 5))
print(f"norm = {np.linalg.norm(blk56):.6e}")

# Also check 2->3 block norm
r2, r3 = schmidt[2]['r'], schmidt[3]['r']
blk23 = HAB_ref[bo[3]:bo[3]+r3*r3, bo[2]:bo[2]+r2*r2]
print(f"\nReference HAB[3,2] block norm = {np.linalg.norm(blk23):.6f}")
