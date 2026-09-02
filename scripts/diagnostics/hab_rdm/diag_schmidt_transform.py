#!/usr/bin/env python3
"""Compare compute_transition_matrices output vs manual (RAW + JW + einsum) Schmidt transform."""
import numpy as np
import sys
sys.path.insert(0, '/data/home/wangcx/krylov-dci')

from dm_svd_embedding.embedded_hamiltonian import _setup_h2o_system
from dm_svd_embedding.occ_virt_partition import setup_partition, build_block_matrices
from dm_svd_embedding.density_matrix import compute_schmidt_decomposition
from dm_svd_embedding.transition_rdm import (
    compute_transition_matrices, _compute_det_create2_annih1_explicit,
    _compute_det_create1_annih2_explicit, _create_sign, _annihilate_sign,
)

(mol, mf, cas, q_idx, backend, h1eff, h2_4d, fcivec, ci_flat,
 n_act, n_elec, n_occ, na, nb, ecore, E_fci) = _setup_h2o_system()
n_virt = n_act - n_occ

partition, full_dets = setup_partition(n_act, n_elec, n_occ, ms=0)
C_blocks = build_block_matrices(partition, ci_flat)
schmidt = compute_schmidt_decomposition(C_blocks, eps=1e-3)

trans_A = compute_transition_matrices(partition, schmidt, n_occ, subspace='A', verbose=False)

# manually recompute create2_annih1_explicit[2] (A side, source n=2 -> dest n=3)
n_A = 2
blk = partition[n_A]
sd = schmidt[n_A]
dets_src = blk['a_dets']
blk_dst = partition[n_A + 1]
sd_dst = schmidt[n_A + 1]
dets_dst = blk_dst['a_dets']
idx_dst = blk_dst['a_index']
U = sd['U']; U_dst = sd_dst['U']

T_raw = _compute_det_create2_annih1_explicit(dets_src, dets_dst, idx_dst, n_occ)

# manual JW (per source det)
dn = {'aaa': (1, 0), 'aba': (0, 1), 'abb': (1, 0), 'bab': (1, 0), 'baa': (0, 1), 'bbb': (0, 1)}
for j, (aA_j, bA_j) in enumerate(dets_src):
    na_ = aA_j.bit_count(); nb_ = bA_j.bit_count()
    for k, (dna, dnb) in dn.items():
        e = 0
        e += na_ if dna > 0 else (na_ - 1 if dna < 0 else 0)
        e += nb_ if dnb > 0 else (nb_ - 1 if dnb < 0 else 0)
        if e & 1:
            T_raw[k][:, j] *= -1

# manual Schmidt transform
def schmidt_transform(T, Ud, Us):
    return np.einsum('ijpqr,ia,jg->agpqr', T, Ud, Us)

manual = {k: schmidt_transform(T_raw[k], U_dst, U) for k in T_raw}

# compare
for k in manual:
    got = trans_A.create2_annih1_explicit[n_A][k]
    ref = manual[k]
    d = np.abs(got - ref).max()
    print(f"  c2a1[{k}] max|diff| = {d:.3e} {'OK' if d < 1e-12 else 'FAIL'}")

# also check create1_annih2 (n-1) for a sanity check of the other operator
n_A = 2
blk_dst2 = partition[n_A - 1]
sd_dst2 = schmidt[n_A - 1]
dets_dst2 = blk_dst2['a_dets']
idx_dst2 = blk_dst2['a_index']
T_raw2 = _compute_det_create1_annih2_explicit(dets_src, dets_dst2, idx_dst2, n_occ)
dn2 = {'aaa': (-1, 0), 'aab': (0, -1), 'aba': (0, -1), 'bab': (-1, 0), 'bba': (-1, 0), 'bbb': (0, -1)}
for j, (aA_j, bA_j) in enumerate(dets_src):
    na_ = aA_j.bit_count(); nb_ = bA_j.bit_count()
    for k, (dna, dnb) in dn2.items():
        e = 0
        e += na_ if dna > 0 else (na_ - 1 if dna < 0 else 0)
        e += nb_ if dnb > 0 else (nb_ - 1 if dnb < 0 else 0)
        if e & 1:
            T_raw2[k][:, j] *= -1
manual2 = {k: schmidt_transform(T_raw2[k], sd_dst2['U'], U) for k in T_raw2}
print("\nc1a2 (n-1):")
for k in manual2:
    got = trans_A.create1_annih2_explicit[n_A][k]
    ref = manual2[k]
    d = np.abs(got - ref).max()
    print(f"  c1a2[{k}] max|diff| = {d:.3e} {'OK' if d < 1e-12 else 'FAIL'}")
