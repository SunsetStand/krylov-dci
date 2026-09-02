#!/usr/bin/env python3
"""Verify B-side (V-based) 3-body transition matrix Schmidt transform + A-side c1a2."""
import numpy as np
import sys
sys.path.insert(0, '/data/home/wangcx/krylov-dci')

from dm_svd_embedding.embedded_hamiltonian import _setup_h2o_system
from dm_svd_embedding.occ_virt_partition import setup_partition, build_block_matrices
from dm_svd_embedding.density_matrix import compute_schmidt_decomposition
from dm_svd_embedding.transition_rdm import (
    compute_transition_matrices, _compute_det_create1_annih2_explicit,
    _compute_det_create2_annih1_explicit,
)

(mol, mf, cas, q_idx, backend, h1eff, h2_4d, fcivec, ci_flat,
 n_act, n_elec, n_occ, na, nb, ecore, E_fci) = _setup_h2o_system()
n_virt = n_act - n_occ

partition, full_dets = setup_partition(n_act, n_elec, n_occ, ms=0)
C_blocks = build_block_matrices(partition, ci_flat)
schmidt = compute_schmidt_decomposition(C_blocks, eps=1e-3)

trans_A = compute_transition_matrices(partition, schmidt, n_occ, subspace='A', verbose=False)
trans_B = compute_transition_matrices(partition, schmidt, n_virt, subspace='B', verbose=False)


def schmidt_transform(T, Ud, Us):
    return np.einsum('ijpqr,ia,jg->agpqr', T, Ud, Us)


# B-side c1a2 (create1_annih2): B loses electron, n_A=2 -> 3.  source B block n=2, dest n=3.
n_A = 2
bsrc = partition[n_A]['b_dets']
bdst = partition[n_A + 1]['b_dets']
bidx = partition[n_A + 1]['b_index']
Vs = schmidt[n_A]['V']; Vd = schmidt[n_A + 1]['V']
T_raw = _compute_det_create1_annih2_explicit(bsrc, bdst, bidx, n_virt)
manual = {k: schmidt_transform(T_raw[k], Vd, Vs) for k in T_raw}
print("B-side c1a2 (1a3b n+1):")
for k in manual:
    got = trans_B.create1_annih2_explicit[n_A][k]
    d = np.abs(got - manual[k]).max()
    print(f"  c1a2[{k}] max|diff| = {d:.3e} {'OK' if d < 1e-12 else 'FAIL'}")

# A-side c1a2 (create1_annih2): A loses electron, n_A=3 -> 2.
n_A = 3
asrc = partition[n_A]['a_dets']
adst = partition[n_A - 1]['a_dets']
aidx = partition[n_A - 1]['a_index']
Us = schmidt[n_A]['U']; Ud = schmidt[n_A - 1]['U']
T_raw2 = _compute_det_create1_annih2_explicit(asrc, adst, aidx, n_occ)
# apply JW
dn2 = {'aaa': (-1, 0), 'aab': (0, -1), 'aba': (0, -1), 'bab': (-1, 0), 'bba': (-1, 0), 'bbb': (0, -1)}
for j, (aA_j, bA_j) in enumerate(asrc):
    na_ = aA_j.bit_count(); nb_ = bA_j.bit_count()
    for k, (dna, dnb) in dn2.items():
        e = 0
        e += na_ if dna > 0 else (na_ - 1 if dna < 0 else 0)
        e += nb_ if dnb > 0 else (nb_ - 1 if dnb < 0 else 0)
        if e & 1:
            T_raw2[k][:, j] *= -1
manual2 = {k: schmidt_transform(T_raw2[k], Ud, Us) for k in T_raw2}
print("\nA-side c1a2 (3a1b n-1):")
for k in manual2:
    got = trans_A.create1_annih2_explicit[n_A][k]
    d = np.abs(got - manual2[k]).max()
    print(f"  c1a2[{k}] max|diff| = {d:.3e} {'OK' if d < 1e-12 else 'FAIL'}")

# B-side c2a1 (1a3b n-1): B gains electron, n_A=3 -> 2.
n_A = 3
bsrc2 = partition[n_A]['b_dets']
bdst2 = partition[n_A - 1]['b_dets']
bidx2 = partition[n_A - 1]['b_index']
Vs2 = schmidt[n_A]['V']; Vd2 = schmidt[n_A - 1]['V']
T_raw3 = _compute_det_create2_annih1_explicit(bsrc2, bdst2, bidx2, n_virt)
manual3 = {k: schmidt_transform(T_raw3[k], Vd2, Vs2) for k in T_raw3}
print("\nB-side c2a1 (1a3b n-1):")
for k in manual3:
    got = trans_B.create2_annih1_explicit[n_A][k]
    d = np.abs(got - manual3[k]).max()
    print(f"  c2a1[{k}] max|diff| = {d:.3e} {'OK' if d < 1e-12 else 'FAIL'}")
