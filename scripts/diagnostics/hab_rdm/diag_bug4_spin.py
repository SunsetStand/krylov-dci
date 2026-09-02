#!/usr/bin/env python3
"""Decompose the exchange term into its 4 spin-pairs for the odd blocks.

For the problematic element (n_A=3 block, flat (0,5)) show which spin-pair
contribution has the wrong sign vs the Slater-Condon reference.
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
from dm_svd_embedding.hab_rdm_contract import _contract_A_tensor_B

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

mask = compatible_product_mask(schmidt, partition, (na, nb))
def project(M): return _project_physical_subspace(M, mask)
ref = project(HAB_ref)

spin_pairs = [('aa', 'aa'), ('bb', 'bb'), ('ab', 'ba'), ('ba', 'ab')]

for nA in sorted(schmidt.keys()):
    r = schmidt[nA]['r']
    if r == 0 or nA not in (3, 5):
        continue
    os = block_offsets[nA]

    # direct term
    H_dir = np.zeros((D, D))
    TA = trans_A.trans_1.get(nA)
    TB = trans_B.trans_1.get(nA)
    if TA is not None:
        Q = np.zeros((n_occ, n_occ, r, r))
        for i in range(n_occ):
            for j in range(n_occ):
                for k in range(n_virt):
                    for l in range(n_virt):
                        Q[i, j] += h2_4d[i, j, k + n_occ, l + n_occ] * TB[:, :, k, l]
        _contract_A_tensor_B(H_dir, os, r, TA, Q, n_occ, sign=+1.0)

    # exchange term, per spin pair
    TAe = trans_A.trans_1_explicit.get(nA)
    TBe = trans_B.trans_1_explicit.get(nA)
    H_spin = {}
    for sA, sB in spin_pairs:
        Hs = np.zeros((D, D))
        Qp = np.zeros((n_occ, n_occ, r, r))
        for i in range(n_occ):
            for l in range(n_occ):
                for j in range(n_virt):
                    for k in range(n_virt):
                        v = h2_4d[i, j + n_occ, k + n_occ, l]
                        if abs(v) < 1e-14:
                            continue
                        Qp[i, l] += v * TBe[sB][:, :, k, j]
        _contract_A_tensor_B(Hs, os, r, TAe[sA], Qp, n_occ, sign=-1.0)
        H_spin[(sA, sB)] = project(Hs)

    sl = slice(os, os + r * r)
    r_ref = ref[sl, sl]

    # find the largest residual element
    r_dir = project(H_dir)[sl, sl]
    r_sum = r_dir
    for Hs in H_spin.values():
        r_sum = r_sum + Hs[sl, sl]

    resid = r_sum - r_ref
    idx = np.unravel_index(np.argmax(np.abs(resid)), resid.shape)
    print(f"\n=== n_A = {nA} (r={r}), residual element {idx} ===")
    print(f"  ref = {r_ref[idx]:+.6f}")
    print(f"  dir = {r_dir[idx]:+.6f}")
    for (sA, sB), Hs in H_spin.items():
        print(f"  ex[{sA}<->{sB}] = {Hs[sl, sl][idx]:+.6f}")
    print(f"  sum = {r_sum[idx]:+.6f}  (ref-sum = {r_ref[idx]-r_sum[idx]:+.6f})")
