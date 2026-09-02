#!/usr/bin/env python3
"""Block-by-block: which off-block H_AB transitions are wrong in the RDM path?"""
import numpy as np
import sys
sys.path.insert(0, '/data/home/wangcx/krylov-dci')

from dm_svd_embedding.embedded_hamiltonian import (
    build_h_emb, _setup_h2o_system,
)
from dm_svd_embedding.occ_virt_partition import setup_partition, build_block_matrices
from dm_svd_embedding.density_matrix import compute_schmidt_decomposition
from dm_svd_embedding.transition_rdm import compute_transition_matrices
from dm_svd_embedding.hab_rdm_contract import (
    _add_1e_cross_block_rdm, _add_nconserved_complementary,
    _add_pair_transfer_complementary,
    _add_3body_patterns, _add_1a3b_patterns,
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

HAB_1e = np.zeros((D, D))
HAB_nc = np.zeros((D, D))
HAB_pair = np.zeros((D, D))
HAB_3a1b = np.zeros((D, D))
HAB_1a3b = np.zeros((D, D))

for nA in sorted(schmidt.keys()):
    if schmidt[nA]['r'] == 0: continue
    _add_1e_cross_block_rdm(HAB_1e, block_offsets, nA, trans_A, trans_B, h1eff, n_occ, n_act, n_virt)
    _add_nconserved_complementary(HAB_nc, block_offsets[nA], nA, trans_A, trans_B, h2_4d, n_occ, n_act, n_virt)
    _add_pair_transfer_complementary(HAB_pair, block_offsets, nA, trans_A, trans_B, h2_4d, n_occ, n_act, n_virt)
    _add_3body_patterns(HAB_3a1b, block_offsets, nA, trans_A, trans_B, h2_4d, n_occ, n_act, n_virt)
    _add_1a3b_patterns(HAB_1a3b, block_offsets, nA, trans_A, trans_B, h2_4d, n_occ, n_act, n_virt)

HAB_rdm = HAB_1e + HAB_nc + HAB_pair + HAB_3a1b + HAB_1a3b

# block-by-block comparison
hdr = "{:>8} | {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9}".format(
    "nA->nA'", "ref", "1e", "nc", "pair", "3a1b", "1a3b", "rdm", "diff")
print(hdr)
blocks = sorted(schmidt.keys())
for n_src in blocks:
    for n_dst in blocks:
        if schmidt[n_src]['r'] == 0 or schmidt[n_dst]['r'] == 0:
            continue
        os = block_offsets[n_src]; od = block_offsets[n_dst]
        rs = schmidt[n_src]['r']; rd = schmidt[n_dst]['r']
        if rs == 0 or rd == 0: continue
        def norm(M):
            return np.linalg.norm(M[od:od+rd*rd, os:os+rs*rs])
        r_ref = norm(HAB_ref); r_1e = norm(HAB_1e); r_nc = norm(HAB_nc)
        r_pair = norm(HAB_pair); r_3a1b = norm(HAB_3a1b); r_1a3b = norm(HAB_1a3b)
        r_rdm = norm(HAB_rdm)
        diff = np.abs(HAB_ref[od:od+rd*rd, os:os+rs*rs] - HAB_rdm[od:od+rd*rd, os:os+rs*rs]).max()
        tag = f"{n_src}->{n_dst}"
        print(f"{tag:>8} | {r_ref:9.3f} {r_1e:9.3f} {r_nc:9.3f} {r_pair:9.3f} {r_3a1b:9.3f} {r_1a3b:9.3f} {r_rdm:9.3f} {diff:9.3f}")
