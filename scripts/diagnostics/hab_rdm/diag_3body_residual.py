#!/usr/bin/env python3
"""Element-wise: fixed 3-body (3a1b+1a3b) vs 2e reference, find residual."""
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
from dm_svd_embedding.hab_rdm_contract import _add_3body_patterns, _add_1a3b_patterns
from src_mf.pyscf_backend import QSpaceIndex, KDCIBackend

(mol, mf, cas, q_idx, backend, h1eff, h2_4d, fcivec, ci_flat,
 n_act, n_elec, n_occ, na, nb, ecore, E_fci) = _setup_h2o_system()
n_virt = n_act - n_occ

h2eff = cas.get_h2eff()
q2 = QSpaceIndex(q_idx.alpha_strs, q_idx.beta_strs, n_act, (na, nb),
                 np.zeros_like(h1eff), h2eff)
backend2 = KDCIBackend(q2)

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

C = np.empty((n_alpha_strs*n_beta_strs, D)); S2 = np.empty_like(C)
for k, info in enumerate(basis_info):
    ci = _expand_schmidt_product_to_ci_matrix(
        info['alpha'], info['beta'], schmidt[info['n']], partition[info['n']],
        n_alpha_strs, n_beta_strs, n_occ,
        q_idx.alpha_strs, q_idx.beta_strs, alpha_to_idx, beta_to_idx)
    C[:, k] = ci.reshape(-1)
    S2[:, k] = backend2.sigma_full(ci).reshape(-1)
H2 = C.T @ S2

trans_A = compute_transition_matrices(partition, schmidt, n_occ, subspace='A', verbose=False)
trans_B = compute_transition_matrices(partition, schmidt, n_virt, subspace='B', verbose=False)

R3 = np.zeros((D, D)); R13 = np.zeros((D, D))
for nA in sorted(schmidt.keys()):
    if schmidt[nA]['r'] == 0:
        continue
    _add_3body_patterns(R3, bo, nA, trans_A, trans_B, h2_4d, n_occ, n_act, n_virt)
    _add_1a3b_patterns(R13, bo, nA, trans_A, trans_B, h2_4d, n_occ, n_act, n_virt)

mask = compatible_product_mask(schmidt, partition, (na, nb))
H2 = _project_physical_subspace(H2, mask)
R3 = _project_physical_subspace(R3, mask)
R13 = _project_physical_subspace(R13, mask)

# compare 2->3 block element-wise
ns, nd = 2, 3
rs, rd = schmidt[ns]['r'], schmidt[nd]['r']
slr = slice(bo[nd], bo[nd]+rd*rd); slc = slice(bo[ns], bo[ns]+rs*rs)
ref = H2[slr, slc]; rdm = R3[slr, slc] + R13[slr, slc]
print(f"2->3 block: ref norm={np.linalg.norm(ref):.6f}, rdm(3+13) norm={np.linalg.norm(rdm):.6f}")
print("max |rdm - ref| elements:")
d = np.abs(rdm - ref)
flat = np.argsort(d.ravel())[::-1]
for idx in flat[:10]:
    i, j = np.unravel_index(idx, d.shape)
    if d[i,j] > 1e-6:
        print(f"  [{i},{j}] ref={ref[i,j]:+.6f} rdm={rdm[i,j]:+.6f} diff={rdm[i,j]-ref[i,j]:+.6f}")

# also show 3->4 block
ns, nd = 3, 4
rs, rd = schmidt[ns]['r'], schmidt[nd]['r']
slr = slice(bo[nd], bo[nd]+rd*rd); slc = slice(bo[ns], bo[ns]+rs*rs)
ref = H2[slr, slc]; rdm = R3[slr, slc] + R13[slr, slc]
print(f"\n3->4 block: ref norm={np.linalg.norm(ref):.6f}, rdm(3+13) norm={np.linalg.norm(rdm):.6f}")
d = np.abs(rdm - ref)
flat = np.argsort(d.ravel())[::-1]
for idx in flat[:8]:
    i, j = np.unravel_index(idx, d.shape)
    if d[i,j] > 1e-6:
        print(f"  [{i},{j}] ref={ref[i,j]:+.6f} rdm={rdm[i,j]:+.6f} diff={rdm[i,j]-ref[i,j]:+.6f}")
