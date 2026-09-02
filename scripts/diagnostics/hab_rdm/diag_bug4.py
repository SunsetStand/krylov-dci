#!/usr/bin/env python3
"""Pinpoint bug4: split n-conserved H_AB into direct vs exchange per block.

Compare against the Slater-Condon reference (build_h_emb HAB) block-by-block,
for the DIAGONAL (n_A -> n_A) blocks, which are contributed ONLY by the
n-conserved 2e term.
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

def project(M):
    return _project_physical_subspace(M, mask)

ref = project(HAB_ref)

print("=" * 72)
print("DIAGONAL (n_A -> n_A) blocks: direct vs exchange vs reference")
print("=" * 72)

for nA in sorted(schmidt.keys()):
    r = schmidt[nA]['r']
    if r == 0:
        continue
    os = block_offsets[nA]

    # Build direct term only (+1, spin-summed)
    H_dir = np.zeros((D, D))
    TA = trans_A.trans_1.get(nA)
    TB = trans_B.trans_1.get(nA)
    if TA is not None and TB is not None:
        Q = np.zeros((n_occ, n_occ, r, r))
        for i in range(n_occ):
            for j in range(n_occ):
                for k in range(n_virt):
                    for l in range(n_virt):
                        v = h2_4d[i, j, k + n_occ, l + n_occ]
                        Q[i, j] += v * TB[:, :, k, l]
        _contract_A_tensor_B(H_dir, os, r, TA, Q, n_occ, sign=+1.0)

    # Build exchange term only (-1, spin-explicit)
    H_ex = np.zeros((D, D))
    TAe = trans_A.trans_1_explicit.get(nA)
    TBe = trans_B.trans_1_explicit.get(nA)
    if TAe is not None and TBe is not None:
        spin_pairs = [('aa', 'aa'), ('bb', 'bb'), ('ab', 'ba'), ('ba', 'ab')]
        Qp = {sA: np.zeros((n_occ, n_occ, r, r)) for sA, _ in spin_pairs}
        for i in range(n_occ):
            for l in range(n_occ):
                for j in range(n_virt):
                    for k in range(n_virt):
                        v = h2_4d[i, j + n_occ, k + n_occ, l]
                        if abs(v) < 1e-14:
                            continue
                        for sA, sB in spin_pairs:
                            Qp[sA][i, l] += v * TBe[sB][:, :, k, j]
        for sA, _ in spin_pairs:
            _contract_A_tensor_B(H_ex, os, r, TAe[sA], Qp[sA], n_occ, sign=-1.0)

    H_dir = project(H_dir)
    H_ex = project(H_ex)

    sl = slice(os, os + r * r)
    r_ref = ref[sl, sl]
    r_dir = H_dir[sl, sl]
    r_ex = H_ex[sl, sl]
    r_tot = r_dir + r_ex

    d_dir = np.abs(r_dir - r_ref).max()
    d_ex = np.abs(r_ex - r_ref).max()
    d_tot = np.abs(r_tot - r_ref).max()
    n_ref = np.linalg.norm(r_ref)

    print(f"\nn_A = {nA}  (r = {r}, ||ref|| = {n_ref:.4f})")
    print(f"  ||direct|| = {np.linalg.norm(r_dir):.4f}   ||exchange|| = {np.linalg.norm(r_ex):.4f}   ||sum|| = {np.linalg.norm(r_tot):.4f}")
    print(f"  max|direct-ref| = {d_dir:.4e}")
    print(f"  max|exchange-ref| = {d_ex:.4e}")
    print(f"  max|(dir+ex)-ref| = {d_tot:.4e}")
    print(f"  max|sum-ref| = {d_tot:.4e}")

    # Also: does the reference equal direct + exchange?
    resid = r_tot - r_ref
    idx = np.unravel_index(np.argmax(np.abs(resid)), resid.shape)
    print(f"  largest residual element at {idx}: ref={r_ref[idx]:.6f} sum={r_tot[idx]:.6f}")
