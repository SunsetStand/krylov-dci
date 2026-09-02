#!/usr/bin/env python3
"""Determine if the RDM 1e-cross term is correct: compare against a
1e-only reference (h2=0 sigma-vector projection)."""
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

from pyscf.fci import cistring, direct_spin1
from src_mf.pyscf_backend import QSpaceIndex, KDCIBackend

(mol, mf, cas, q_idx, backend, h1eff, h2_4d, fcivec, ci_flat,
 n_act, n_elec, n_occ, na, nb, ecore, E_fci) = _setup_h2o_system()

partition, full_dets = setup_partition(n_act, n_elec, n_occ, ms=0)
C_blocks = build_block_matrices(partition, ci_flat)
schmidt = compute_schmidt_decomposition(C_blocks, eps=1e-3)

# --- 1e-only reference via direct_spin1.contract_1e ---
n_alpha_strs = len(q_idx.alpha_strs)
n_beta_strs = len(q_idx.beta_strs)
alpha_to_idx = {int(s): i for i, s in enumerate(q_idx.alpha_strs)}
beta_to_idx = {int(s): i for i, s in enumerate(q_idx.beta_strs)}

# build basis_info
basis_info = []
offset = 0
bo = {}
for nA in sorted(schmidt.keys()):
    bo[nA] = offset
    r = schmidt[nA]['r']
    for alpha in range(r):
        for beta in range(r):
            basis_info.append({'n': nA, 'alpha': alpha, 'beta': beta})
    offset += r*r
D = offset

C_flat = np.empty((n_alpha_strs * n_beta_strs, D))
S1_flat = np.empty((n_alpha_strs * n_beta_strs, D))
Sfull_flat = np.empty((n_alpha_strs * n_beta_strs, D))

for k, info in enumerate(basis_info):
    ci = _expand_schmidt_product_to_ci_matrix(
        info['alpha'], info['beta'], schmidt[info['n']], partition[info['n']],
        n_alpha_strs, n_beta_strs, n_occ,
        q_idx.alpha_strs, q_idx.beta_strs, alpha_to_idx, beta_to_idx)
    C_flat[:, k] = ci.reshape(-1)
    # 1e-only sigma
    s1 = direct_spin1.contract_1e(h1eff, ci, n_act, (na, nb))
    S1_flat[:, k] = s1.reshape(-1)
    # full sigma
    Sfull_flat[:, k] = backend.sigma_full(ci).reshape(-1)

H1_emb = C_flat.T @ S1_flat
Hfull_emb = C_flat.T @ Sfull_flat

mask = compatible_product_mask(schmidt, partition, (na, nb))
H1_emb = _project_physical_subspace(H1_emb, mask)
Hfull_emb = _project_physical_subspace(Hfull_emb, mask)

# --- RDM 1e cross ---
trans_A = compute_transition_matrices(partition, schmidt, n_occ, subspace='A', verbose=False)
trans_B = compute_transition_matrices(partition, schmidt, n_virt := n_act-n_occ, subspace='B', verbose=False)
H1_rdm = np.zeros((D, D))
for nA in sorted(schmidt.keys()):
    if schmidt[nA]['r'] == 0:
        continue
    _add_1e_cross_block_rdm(H1_rdm, bo, nA, trans_A, trans_B, h1eff, n_occ, n_act, n_virt)
H1_rdm = _project_physical_subspace(H1_rdm, mask)

# compare off-diagonal 1e blocks
print(f"{'block':>7} | {'H1_ref':>10} {'H1_rdm':>10} {'maxdiff':>10} | {'Hfull_ref':>10}")
for ns in sorted(schmidt.keys()):
    for nd in sorted(schmidt.keys()):
        if ns == nd or schmidt[ns]['r']==0 or schmidt[nd]['r']==0:
            continue
        rs, rd = schmidt[ns]['r'], schmidt[nd]['r']
        slr = slice(bo[nd], bo[nd]+rd*rd); slc = slice(bo[ns], bo[ns]+rs*rs)
        r1 = H1_emb[slr, slc]; m1 = H1_rdm[slr, slc]; rf = Hfull_emb[slr, slc]
        n1 = np.linalg.norm(r1); nm = np.linalg.norm(m1); nf = np.linalg.norm(rf)
        if n1 < 1e-8 and nm < 1e-8:
            continue
        md = np.abs(m1 - r1).max()
        print(f"{ns}->{nd:<3} | {n1:10.5f} {nm:10.5f} {md:10.5f} | {nf:10.5f}")
