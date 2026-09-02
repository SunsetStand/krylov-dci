#!/usr/bin/env python3
"""2e-only reference vs RDM 2e cross terms (3body + pair).

Determine whether the 3-body / pair-transfer terms are off by a sign,
a constant factor, or something else, by comparing against a 2e-only
sigma-vector reference (h1=0).
"""
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
from dm_svd_embedding.hab_rdm_contract import (
    _add_3body_patterns, _add_1a3b_patterns, _add_pair_transfer_complementary,
)
from pyscf.fci import cistring
from src_mf.pyscf_backend import QSpaceIndex, KDCIBackend

(mol, mf, cas, q_idx, backend, h1eff, h2_4d, fcivec, ci_flat,
 n_act, n_elec, n_occ, na, nb, ecore, E_fci) = _setup_h2o_system()

# 2e-only backend: h1 = 0
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
trans_B = compute_transition_matrices(partition, schmidt, n_act-n_occ, subspace='B', verbose=False)

R3 = np.zeros((D, D)); R13 = np.zeros((D, D)); Rp = np.zeros((D, D))
for nA in sorted(schmidt.keys()):
    if schmidt[nA]['r'] == 0: continue
    _add_3body_patterns(R3, bo, nA, trans_A, trans_B, h2_4d, n_occ, n_act, n_act-n_occ)
    _add_1a3b_patterns(R13, bo, nA, trans_A, trans_B, h2_4d, n_occ, n_act, n_act-n_occ)
    _add_pair_transfer_complementary(Rp, bo, nA, trans_A, trans_B, h2_4d, n_occ, n_act, n_act-n_occ)

mask = compatible_product_mask(schmidt, partition, (na, nb))
def P(M): return _project_physical_subspace(M, mask)
H2 = P(H2); R3 = P(R3); R13 = P(R13); Rp = P(Rp)

print(f"{'block':>7} | {'H2_ref':>10} {'3a1b':>10} {'1a3b':>10} {'pair':>10} | {'3+13':>10} {'ratio':>8}")
for ns in sorted(schmidt.keys()):
    for nd in sorted(schmidt.keys()):
        if ns == nd or schmidt[ns]['r']==0 or schmidt[nd]['r']==0: continue
        rs, rd = schmidt[ns]['r'], schmidt[nd]['r']
        slr = slice(bo[nd], bo[nd]+rd*rd); slc = slice(bo[ns], bo[ns]+rs*rs)
        r = H2[slr, slc]
        nr = np.linalg.norm(r)
        dn = nd - ns
        if dn in (1, -1):
            m = R3[slr, slc] + R13[slr, slc]
            n3 = np.linalg.norm(R3[slr, slc]); n13 = np.linalg.norm(R13[slr, slc])
            nm = np.linalg.norm(m)
            ratio = nm/nr if nr > 1e-8 else float('nan')
            print(f"{ns}->{nd:<3} | {nr:10.5f} {n3:10.5f} {n13:10.5f} {'--':>10} | {nm:10.5f} {ratio:8.3f}")
        elif dn in (2, -2):
            n3 = np.linalg.norm(R3[slr, slc]); n13 = np.linalg.norm(R13[slr, slc])
            npair = np.linalg.norm(Rp[slr, slc])
            ratio = npair/nr if nr > 1e-8 else float('nan')
            print(f"{ns}->{nd:<3} | {nr:10.5f} {'--':>10} {'--':>10} {npair:10.5f} | {npair:10.5f} {ratio:8.3f}")
