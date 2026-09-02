#!/usr/bin/env python3
"""Projected block-by-block: which physical-block transitions are still wrong?"""
import numpy as np
import sys
sys.path.insert(0, '/data/home/wangcx/krylov-dci')

from dm_svd_embedding.embedded_hamiltonian import (
    build_h_emb, _setup_h2o_system, compatible_product_mask, _project_physical_subspace,
)
from dm_svd_embedding.occ_virt_partition import setup_partition, build_block_matrices
from dm_svd_embedding.density_matrix import compute_schmidt_decomposition
from dm_svd_embedding.transition_rdm import compute_transition_matrices
from dm_svd_embedding.hab_rdm_contract import (
    _add_1e_cross_block_rdm, _add_nconserved_complementary,
    _add_pair_transfer_complementary, _add_3body_patterns, _add_1a3b_patterns,
)

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

HAB_1e = np.zeros((D, D)); HAB_nc = np.zeros((D, D)); HAB_pair = np.zeros((D, D))
HAB_3a1b = np.zeros((D, D)); HAB_1a3b = np.zeros((D, D))

for nA in sorted(schmidt.keys()):
    if schmidt[nA]['r'] == 0: continue
    _add_1e_cross_block_rdm(HAB_1e, block_offsets, nA, trans_A, trans_B, h1eff, n_occ, n_act, n_virt)
    _add_nconserved_complementary(HAB_nc, block_offsets[nA], nA, trans_A, trans_B, h2_4d, n_occ, n_act, n_virt)
    _add_pair_transfer_complementary(HAB_pair, block_offsets, nA, trans_A, trans_B, h2_4d, n_occ, n_act, n_virt)
    _add_3body_patterns(HAB_3a1b, block_offsets, nA, trans_A, trans_B, h2_4d, n_occ, n_act, n_virt)
    _add_1a3b_patterns(HAB_1a3b, block_offsets, nA, trans_A, trans_B, h2_4d, n_occ, n_act, n_virt)

# project EVERY term onto physical subspace FIRST
mask = compatible_product_mask(schmidt, partition, (na, nb))
def P(M): return _project_physical_subspace(M, mask)
ref = P(HAB_ref)
r_1e = P(HAB_1e); r_nc = P(HAB_nc); r_pair = P(HAB_pair)
r_3a1b = P(HAB_3a1b); r_1a3b = P(HAB_1a3b)
rdm = r_1e + r_nc + r_pair + r_3a1b + r_1a3b

print(f"  [PROJECTED, physical block only]")
print(f"  ||ref|| = {np.linalg.norm(ref):.4f}")
print(f"  ||1e||  = {np.linalg.norm(r_1e):.4f}")
print(f"  ||nc||  = {np.linalg.norm(r_nc):.4f}")
print(f"  ||pair||= {np.linalg.norm(r_pair):.4f}")
print(f"  ||3a1b||= {np.linalg.norm(r_3a1b):.4f}")
print(f"  ||1a3b||= {np.linalg.norm(r_1a3b):.4f}")
print(f"  ||rdm|| = {np.linalg.norm(rdm):.4f}")
print(f"  max|rdm - ref| = {np.abs(rdm - ref).max():.4f}")

# block-by-block projected diff
print("\n  {:>8} | {:>8} {:>8} {:>8}".format("nA->nA'", "ref", "rdm", "diff"))
for ns in sorted(schmidt.keys()):
    for nd_ in sorted(schmidt.keys()):
        if schmidt[ns]['r'] == 0 or schmidt[nd_]['r'] == 0: continue
        os = block_offsets[ns]; od = block_offsets[nd_]
        rs = schmidt[ns]['r']; rd = schmidt[nd_]['r']
        slr = slice(od, od+rd*rd); slc = slice(os, os+rs*rs)
        r = ref[slr, slc]; m = rdm[slr, slc]
        nr = np.linalg.norm(r)
        if nr < 1e-6 and np.linalg.norm(m) < 1e-6: continue
        print(f"  {ns}->{nd_:<4} | {nr:8.4f} {np.linalg.norm(m):8.4f} {np.abs(m-r).max():8.4f}")
