#!/usr/bin/env python3
"""Pinpoint bug5: decompose off-diagonal H_AB blocks by term.

For each off-diagonal block (n_A -> n_A +-1, n_A +-2), show the reference
norm and each RDM term's contribution, to identify which term is wrong.
"""
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

terms = {}
for name in ('1e', 'nc', 'pair', '3a1b', '1a3b'):
    terms[name] = np.zeros((D, D))

for nA in sorted(schmidt.keys()):
    if schmidt[nA]['r'] == 0:
        continue
    _add_1e_cross_block_rdm(terms['1e'], block_offsets, nA, trans_A, trans_B, h1eff, n_occ, n_act, n_virt)
    _add_nconserved_complementary(terms['nc'], block_offsets[nA], nA, trans_A, trans_B, h2_4d, n_occ, n_act, n_virt)
    _add_pair_transfer_complementary(terms['pair'], block_offsets, nA, trans_A, trans_B, h2_4d, n_occ, n_act, n_virt)
    _add_3body_patterns(terms['3a1b'], block_offsets, nA, trans_A, trans_B, h2_4d, n_occ, n_act, n_virt)
    _add_1a3b_patterns(terms['1a3b'], block_offsets, nA, trans_A, trans_B, h2_4d, n_occ, n_act, n_virt)

mask = compatible_product_mask(schmidt, partition, (na, nb))
def P(M): return _project_physical_subspace(M, mask)

ref = P(HAB_ref)
terms = {k: P(v) for k, v in terms.items()}

print(f"n_occ={n_occ}, n_act={n_act}, n_virt={n_virt}, blocks={sorted(schmidt.keys())}")
print()
print("OFF-DIAGONAL blocks: ref norm vs per-term contribution")
print(f"{'block':>8} | {'ref':>9} {'1e':>9} {'3a1b':>9} {'1a3b':>9} {'pair':>9} | {'sum':>9} {'maxdiff':>9}")
print("-" * 90)

for ns in sorted(schmidt.keys()):
    for nd in sorted(schmidt.keys()):
        if ns == nd:
            continue
        if schmidt[ns]['r'] == 0 or schmidt[nd]['r'] == 0:
            continue
        os = block_offsets[ns]; od = block_offsets[nd]
        rs = schmidt[ns]['r']; rd = schmidt[nd]['r']
        slr = slice(od, od + rd*rd); slc = slice(os, os + rs*rs)
        r = ref[slr, slc]
        nr = np.linalg.norm(r)
        if nr < 1e-8:
            # still show if RDM gives nonzero (spurious coupling)
            tot = sum(terms[k][slr, slc] for k in terms)
            ntot = np.linalg.norm(tot)
            if ntot > 1e-8:
                n1 = np.linalg.norm(terms['1e'][slr, slc])
                n3 = np.linalg.norm(terms['3a1b'][slr, slc])
                n13 = np.linalg.norm(terms['1a3b'][slr, slc])
                npair = np.linalg.norm(terms['pair'][slr, slc])
                print(f"{ns}->{nd:<3} | {nr:9.4f} {n1:9.4f} {n3:9.4f} {n13:9.4f} {npair:9.4f} | {ntot:9.4f} SPURIOUS")
            continue
        n1 = np.linalg.norm(terms['1e'][slr, slc])
        n3 = np.linalg.norm(terms['3a1b'][slr, slc])
        n13 = np.linalg.norm(terms['1a3b'][slr, slc])
        npair = np.linalg.norm(terms['pair'][slr, slc])
        tot = sum(terms[k][slr, slc] for k in terms)
        ntot = np.linalg.norm(tot)
        md = np.abs(tot - r).max()
        print(f"{ns}->{nd:<3} | {nr:9.4f} {n1:9.4f} {n3:9.4f} {n13:9.4f} {npair:9.4f} | {ntot:9.4f} {md:9.4f}")
