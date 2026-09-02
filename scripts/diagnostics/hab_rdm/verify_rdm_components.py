#!/usr/bin/env python3
"""RDM H_AB vs (now-correct) reference H_AB, component by component."""
import numpy as np
import sys
sys.path.insert(0, '/data/home/wangcx/krylov-dci')

from dm_svd_embedding.embedded_hamiltonian import (
    build_h_emb, build_hemb_via_rdm, _setup_h2o_system,
)
from dm_svd_embedding.occ_virt_partition import setup_partition, build_block_matrices
from dm_svd_embedding.density_matrix import compute_schmidt_decomposition
from dm_svd_embedding.transition_rdm import compute_transition_matrices
from dm_svd_embedding.hab_rdm_contract import (
    _add_1e_cross_block_rdm, _add_nconserved_complementary,
    _add_nconserved_complementary_b_to_a, _add_pair_transfer_complementary,
    _add_3body_patterns, _add_1a3b_patterns,
)

(mol, mf, cas, q_idx, backend, h1eff, h2_4d, fcivec, ci_flat,
 n_act, n_elec, n_occ, na, nb, ecore, E_fci) = _setup_h2o_system()
n_virt = n_act - n_occ

partition, full_dets = setup_partition(n_act, n_elec, n_occ, ms=0)
C_blocks = build_block_matrices(partition, ci_flat)
schmidt = compute_schmidt_decomposition(C_blocks, eps=1e-3)

H_ref, basis_ref, decomps_ref = build_h_emb(
    schmidt, partition, q_idx, backend, h1eff, h2_4d, n_occ, n_act, verbose=False)
D = H_ref.shape[0]
HAB_ref = decomps_ref['HAB']

trans_A = compute_transition_matrices(partition, schmidt, n_occ, subspace='A', verbose=False)
trans_B = compute_transition_matrices(partition, schmidt, n_virt, subspace='B', verbose=False)

# block offsets (same as build_h_emb)
block_offsets = {}
off = 0
for nA in sorted(schmidt.keys()):
    block_offsets[nA] = off
    off += schmidt[nA]['r'] ** 2

# build each RDM component separately
HAB_1e = np.zeros((D, D))
HAB_nc = np.zeros((D, D))
HAB_nc2 = np.zeros((D, D))   # b_to_a
HAB_pair = np.zeros((D, D))
HAB_3a1b = np.zeros((D, D))
HAB_1a3b = np.zeros((D, D))

for nA in sorted(schmidt.keys()):
    if schmidt[nA]['r'] == 0: continue
    _add_1e_cross_block_rdm(HAB_1e, block_offsets, nA, trans_A, trans_B, h1eff, n_occ, n_act, n_virt)
    _add_nconserved_complementary(HAB_nc, block_offsets[nA], nA, trans_A, trans_B, h2_4d, n_occ, n_act, n_virt)
    _add_nconserved_complementary_b_to_a(HAB_nc2, block_offsets[nA], nA, trans_A, trans_B, h2_4d, n_occ, n_act, n_virt)
    _add_pair_transfer_complementary(HAB_pair, block_offsets, nA, trans_A, trans_B, h2_4d, n_occ, n_act, n_virt)
    _add_3body_patterns(HAB_3a1b, block_offsets, nA, trans_A, trans_B, h2_4d, n_occ, n_act, n_virt)
    _add_1a3b_patterns(HAB_1a3b, block_offsets, nA, trans_A, trans_B, h2_4d, n_occ, n_act, n_virt)

HAB_rdm = HAB_1e + HAB_nc + HAB_nc2 + HAB_pair + HAB_3a1b + HAB_1a3b

print(f"D = {D}")
for name, comp in [("1e cross", HAB_1e), ("n-conserved (b,c)", HAB_nc),
                   ("n-conserved b_to_a (d,e)", HAB_nc2), ("pair transfer", HAB_pair),
                   ("3A+1B", HAB_3a1b), ("1A+3B", HAB_1a3b),
                   ("RDM total", HAB_rdm)]:
    print(f"  ||{name:24s}|| = {np.linalg.norm(comp):.6f}")
print(f"  ||reference HAB||      = {np.linalg.norm(HAB_ref):.6f}")
print(f"  max|RDM total - ref|   = {np.abs(HAB_rdm - HAB_ref).max():.6f}")

# per-component diff against reference (projected onto physical subspace via mask)
from dm_svd_embedding.embedded_hamiltonian import compatible_product_mask, _project_physical_subspace
mask = compatible_product_mask(schmidt, partition, (na, nb))
HAB_ref_p = _project_physical_subspace(HAB_ref, mask)
HAB_rdm_p = _project_physical_subspace(HAB_rdm, mask)
print(f"\n  [physical block only]")
print(f"  max|RDM total - ref|   = {np.abs(HAB_rdm_p - HAB_ref_p).max():.6f}")
