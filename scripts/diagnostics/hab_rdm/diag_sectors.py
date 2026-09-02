#!/usr/bin/env python3
"""Print residual elements' spin sectors and R3/R13/H2 decomposition."""
import numpy as np
import sys
sys.path.insert(0, '/data/home/wangcx/krylov-dci')

from dm_svd_embedding.embedded_hamiltonian import (
    _setup_h2o_system, _expand_schmidt_product_to_ci_matrix,
    compatible_product_mask, schmidt_spin_sectors, _project_physical_subspace,
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

sectors = schmidt_spin_sectors(schmidt, partition)

# print spin sectors of all Schmidt states in blocks 2 and 3
for nA in (2, 3):
    a_sec, b_sec = sectors[nA]
    r = schmidt[nA]['r']
    print(f"block n={nA} (r={r}):")
    for a in range(r):
        for b in range(r):
            sa = a_sec[a]; sb = b_sec[b]
            flat = bo[nA] + a*r + b
            print(f"  (alpha={a},beta={b}) flat={flat}: A_sec={sa} B_sec={sb} sum={(sa[0]+sb[0] if sa and sb else None, sa[1]+sb[1] if sa and sb else None)}")

# total spin target
print(f"\ntarget (na,nb)=({na},{nb})")
