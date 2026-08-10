"""Test eigenvalues of H_emb via RDM vs sigma-vector on H2O and N2."""
import numpy as np
import sys
sys.path.insert(0, '/data/home/wangcx/krylov-dci')

from dm_svd_embedding.occ_virt_partition import setup_partition, build_block_matrices
from dm_svd_embedding.density_matrix import compute_schmidt_decomposition
from dm_svd_embedding.embedded_hamiltonian import _setup_h2o_system, build_h_emb, build_hemb_via_rdm
from dm_svd_embedding.transition_rdm import compute_transition_matrices

(mol, mf, cas, q_idx, backend, h1eff, h2_4d, fcivec, ci_flat,
 n_act, n_elec, n_occ, na, nb, ecore, E_fci) = _setup_h2o_system()
n_virt = n_act - n_occ

partition, full_dets = setup_partition(n_act, n_elec, n_occ, ms=0)
C_blocks = build_block_matrices(partition, ci_flat)
schmidt = compute_schmidt_decomposition(C_blocks, eps=1e-3)

# Reference
H_ref, basis_ref, _ = build_h_emb(schmidt, partition, q_idx, backend, h1eff, h2_4d, n_occ, n_act, verbose=False)

# RDM
trans_A = compute_transition_matrices(partition, schmidt, n_occ, subspace='A', verbose=False)
trans_B = compute_transition_matrices(partition, schmidt, n_virt, subspace='B', verbose=False)
H_rdm, basis_rdm, decomps = build_hemb_via_rdm(schmidt, partition, h1eff, h2_4d, n_occ, n_act, trans_A, trans_B, verbose=False)

D = H_rdm.shape[0]
print(f"H2O: D={D}")

# Eigenvalues
ev_ref = np.sort(np.linalg.eigvalsh(H_ref))
ev_rdm = np.sort(np.linalg.eigvalsh(H_rdm))
print(f"\nEigenvalues (Ha):")
for i in range(min(8, D)):
    print(f"  E[{i}]: ref={ev_ref[i]:.6f}, rdm={ev_rdm[i]:.6f}, diff={(ev_rdm[i]-ev_ref[i])*1000:+.2f} mH")

# Component norms
HA_ref = decomps['HA']
HB_ref = decomps['HB']
HAB_ref = H_ref - HA_ref - HB_ref
HAB_rdm = decomps['HAB']
print(f"\n||HA||={np.linalg.norm(HA_ref):.2f}, ||HB||={np.linalg.norm(HB_ref):.2f}")
print(f"||HAB_ref||={np.linalg.norm(HAB_ref):.2f}, ||HAB_rdm||={np.linalg.norm(HAB_rdm):.2f}")

# Now just compare same-block diagonal using term(b) only (no pair transfer, no 3body)
H_AB_tb = np.zeros_like(H_rdm)
import dm_svd_embedding.hab_rdm_contract as hrc

# Block offsets
block_offsets = {}
offset = 0
for na in sorted(schmidt.keys()):
    rr = schmidt[na]['r']
    block_offsets[na] = offset
    offset += rr * rr

# Just term(b)
for n_A in sorted(schmidt.keys()):
    sd = schmidt[n_A]
    if sd['r'] == 0: continue
    os = block_offsets.get(n_A)
    if os is None: continue
    hrc._add_nconserved_complementary(H_AB_tb, os, n_A, trans_A, trans_B, h2_4d, n_occ, n_act, n_virt)

H_tb = HA_ref + HB_ref + H_AB_tb
ev_tb = np.sort(np.linalg.eigvalsh(H_tb))
print(f"\nTerm(b) only: ||H_AB_tb||={np.linalg.norm(H_AB_tb):.2f}")
for i in range(min(8, D)):
    print(f"  E[{i}]: ref={ev_ref[i]:.6f}, tb={ev_tb[i]:.6f}, diff={(ev_tb[i]-ev_ref[i])*1000:+.2f} mH")
