#!/usr/bin/env python3
"""End-to-end: build H^emb via RDM path, diagonalize, compare to FCI."""
import numpy as np
import sys
sys.path.insert(0, '/data/home/wangcx/krylov-dci')

from dm_svd_embedding.embedded_hamiltonian import (
    build_h_emb, build_hemb_via_rdm, _setup_h2o_system,
    compatible_product_mask, _project_physical_subspace,
)
from dm_svd_embedding.occ_virt_partition import setup_partition, build_block_matrices
from dm_svd_embedding.density_matrix import compute_schmidt_decomposition
from dm_svd_embedding.transition_rdm import compute_transition_matrices

(mol, mf, cas, q_idx, backend, h1eff, h2_4d, fcivec, ci_flat,
 n_act, n_elec, n_occ, na, nb, ecore, E_fci) = _setup_h2o_system()
n_virt = n_act - n_occ

partition, full_dets = setup_partition(n_act, n_elec, n_occ, ms=0)
C_blocks = build_block_matrices(partition, ci_flat)
schmidt = compute_schmidt_decomposition(C_blocks, eps=1e-3)

# Reference path (Slater-Condon direct)
H_ref, _, _ = build_h_emb(
    schmidt, partition, q_idx, backend, h1eff, h2_4d, n_occ, n_act, verbose=False)

# RDM path
trans_A = compute_transition_matrices(partition, schmidt, n_occ, subspace='A', verbose=False)
trans_B = compute_transition_matrices(partition, schmidt, n_virt, subspace='B', verbose=False)
H_rdm, decomps, basis_info = build_hemb_via_rdm(
    schmidt, partition, h1eff, h2_4d, n_occ, n_act, trans_A, trans_B, verbose=False)

# Project onto physical subspace for a fair eigenvalue comparison
mask = compatible_product_mask(schmidt, partition, (na, nb))
H_ref_p = _project_physical_subspace(H_ref, mask)
H_rdm_p = _project_physical_subspace(H_rdm, mask)

e_ref = np.linalg.eigvalsh(H_ref_p)
e_rdm = np.linalg.eigvalsh(H_rdm_p)

print("=" * 60)
print(f"FCI ground state   : {E_fci:.10f} Ha")
print(f"Ref H^emb (SC) gs  : {e_ref[0]:.10f} Ha   dE = {(e_ref[0]-E_fci)*1000:+.6f} mH")
print(f"RDM H^emb gs       : {e_rdm[0]:.10f} Ha   dE = {(e_rdm[0]-E_fci)*1000:+.6f} mH")
print(f"Ref vs RDM gs      : {(e_rdm[0]-e_ref[0])*1000:+.6f} mH")
print(f"max|H_rdm - H_ref| (projected) = {np.abs(H_rdm_p - H_ref_p).max():.4e}")
print("=" * 60)
print("First few RDM eigenvalues (mH vs FCI):")
for i in range(min(5, len(e_rdm))):
    print(f"  state {i}: {(e_rdm[i]-E_fci)*1000:+.4f} mH")
