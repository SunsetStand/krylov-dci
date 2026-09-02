#!/usr/bin/env python3
"""Print RDM 1e-cross vs 1e-ref elements for the 5->6 block to see the sign relation."""
import numpy as np
import sys
sys.path.insert(0, '/data/home/wangcx/krylov-dci')

from dm_svd_embedding.embedded_hamiltonian import (
    _setup_h2o_system, _expand_schmidt_product_to_ci_matrix,
    compatible_product_mask, _project_physical_subspace,
)
from dm_svd_embedding.occ_virt_partition import setup_partition, build_block_matrices
from dm_svd_embedding.density_matrix import compute_schmidt_decomposition
from dm_svd_embedding.transition_rdm import compute_transition_matrices
from dm_svd_embedding.hab_rdm_contract import _add_1e_cross_block_rdm
from pyscf.fci import direct_spin1

(mol, mf, cas, q_idx, backend, h1eff, h2_4d, fcivec, ci_flat,
 n_act, n_elec, n_occ, na, nb, ecore, E_fci) = _setup_h2o_system()
partition, full_dets = setup_partition(n_act, n_elec, n_occ, ms=0)
C_blocks = build_block_matrices(partition, ci_flat)
schmidt = compute_schmidt_decomposition(C_blocks, eps=1e-3)

n_alpha_strs = len(q_idx.alpha_strs); n_beta_strs = len(q_idx.beta_strs)
alpha_to_idx = {int(s): i for i, s in enumerate(q_idx.alpha_strs)}
beta_to_idx = {int(s): i for i, s in enumerate(q_idx.beta_strs)}

basis_info = []; off = 0; bo = {}
for nA in sorted(schmidt.keys()):
    bo[nA] = off; r = schmidt[nA]['r']
    for a in range(r):
        for b in range(r):
            basis_info.append({'n': nA, 'alpha': a, 'beta': b})
    off += r*r
D = off

C = np.empty((n_alpha_strs*n_beta_strs, D)); S1 = np.empty_like(C)
for k, info in enumerate(basis_info):
    ci = _expand_schmidt_product_to_ci_matrix(
        info['alpha'], info['beta'], schmidt[info['n']], partition[info['n']],
        n_alpha_strs, n_beta_strs, n_occ,
        q_idx.alpha_strs, q_idx.beta_strs, alpha_to_idx, beta_to_idx)
    C[:, k] = ci.reshape(-1)
    S1[:, k] = direct_spin1.contract_1e(h1eff, ci, n_act, (na, nb)).reshape(-1)
H1 = C.T @ S1

trans_A = compute_transition_matrices(partition, schmidt, n_occ, subspace='A', verbose=False)
trans_B = compute_transition_matrices(partition, schmidt, n_act-n_occ, subspace='B', verbose=False)
H1r = np.zeros((D, D))
for nA in sorted(schmidt.keys()):
    if schmidt[nA]['r']: _add_1e_cross_block_rdm(H1r, bo, nA, trans_A, trans_B, h1eff, n_occ, n_act, n_act-n_occ)

mask = compatible_product_mask(schmidt, partition, (na, nb))
H1 = _project_physical_subspace(H1, mask); H1r = _project_physical_subspace(H1r, mask)

r5, r6 = schmidt[5]['r'], schmidt[6]['r']
slr = slice(bo[6], bo[6]+r6*r6); slc = slice(bo[5], bo[5]+r5*r5)
ref = H1[slr, slc].ravel(); rdm = H1r[slr, slc].ravel()
print("5->6 block:  idx   ref        rdm        rdm/ref")
for i in range(len(ref)):
    if abs(ref[i]) > 1e-8:
        print(f"  {i:3d}  {ref[i]:+.6f}  {rdm[i]:+.6f}  {rdm[i]/ref[i]:+.4f}")

print(f"\nnorm(ref)={np.linalg.norm(ref):.6f}  norm(rdm)={np.linalg.norm(rdm):.6f}")
print(f"norm(rdm+ref)={np.linalg.norm(rdm+ref):.6e}  (0 => uniform flip)")
print(f"norm(rdm-ref)={np.linalg.norm(rdm-ref):.6e}  (0 => identical)")
