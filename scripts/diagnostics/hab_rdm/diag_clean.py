#!/usr/bin/env python3
"""
Clean comparison: RDM-based H^emb vs sigma-vector H^emb.
Identify exactly which components are wrong.
"""
import numpy as np
import sys
sys.path.insert(0, '/data/home/wangcx/krylov-dci')

from dm_svd_embedding.embedded_hamiltonian import (
    build_h_emb, build_hemb_via_rdm, _setup_h2o_system,
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

# === Reference: sigma-vector ===
print("=== Reference H^emb (sigma-vector) ===")
H_ref, basis_ref, decomps_ref = build_h_emb(
    schmidt, partition, q_idx, backend, h1eff, h2_4d,
    n_occ, n_act, verbose=False)
D = H_ref.shape[0]
HA_ref = decomps_ref['HA']
HB_ref = decomps_ref['HB']
HAB_ref = decomps_ref['HAB']

# === RDM-based H^emb ===
print("\n=== RDM H^emb ===")
trans_A = compute_transition_matrices(partition, schmidt, n_occ, subspace='A', verbose=False)
trans_B = compute_transition_matrices(partition, schmidt, n_virt, subspace='B', verbose=False)

H_rdm, basis_rdm, decomps_rdm = build_hemb_via_rdm(
    schmidt, partition, h1eff, h2_4d,
    n_occ, n_act, trans_A, trans_B, verbose=False)

HA_rdm = decomps_rdm['HA']
HB_rdm = decomps_rdm['HB']
HAB_rdm = decomps_rdm['HAB']

# === Comparison ===
print(f"\n=== Global comparison (D={D}) ===")
print(f"  max|H_ref - H_rdm|          = {np.abs(H_ref - H_rdm).max():.6e}")
print(f"  max|HA_ref - HA_rdm|        = {np.abs(HA_ref - HA_rdm).max():.6e}")
print(f"  max|HB_ref - HB_rdm|        = {np.abs(HB_ref - HB_rdm).max():.6e}")
print(f"  max|HAB_ref - HAB_rdm|      = {np.abs(HAB_ref - HAB_rdm).max():.6e}")
print(f"  ||H_ref|| = {np.linalg.norm(H_ref):.6f}")
print(f"  ||H_rdm|| = {np.linalg.norm(H_rdm):.6f}")
print(f"  ||HAB_ref|| = {np.linalg.norm(HAB_ref):.6f}")
print(f"  ||HAB_rdm|| = {np.linalg.norm(HAB_rdm):.6f}")

# === Eigenvalues ===
ev_ref, _ = np.linalg.eigh(H_ref)
ev_rdm, _ = np.linalg.eigh(H_rdm)
print(f"\n=== Eigenvalue comparison ===")
for i in range(min(8, D)):
    diff = abs(ev_ref[i] - ev_rdm[i])
    print(f"  E[{i}]: ref={ev_ref[i]:.8f}, rdm={ev_rdm[i]:.8f}, diff={diff:.6e}")

dE0 = (ev_rdm[0] - ev_ref[0]) * 1000
print(f"\n  dE0 (RDM - ref) = {dE0:+.3f} mH")

# === Per-block contributions ===
print(f"\n=== Per-block H_AB comparison ===")
block_offsets = {}
off = 0
for na in sorted(schmidt.keys()):
    block_offsets[na] = off
    rn = schmidt[na]['r']
    off += rn * rn

for na in sorted(schmidt.keys()):
    rn = schmidt[na]['r']
    if rn == 0: continue
    bo = block_offsets[na]
    block_ref = HAB_ref[bo:bo+rn*rn, bo:bo+rn*rn]
    block_rdm = HAB_rdm[bo:bo+rn*rn, bo:bo+rn*rn]
    diff = np.abs(block_ref - block_rdm).max()
    nr = np.linalg.norm(block_ref)
    nd = np.linalg.norm(block_rdm)
    print(f"  n={na}: r={rn}, ||ref||={nr:.4f}, ||rdm||={nd:.4f}, max|diff|={diff:.4f}")

# === H_AB component breakdown ===
print(f"\n=== H_AB RDM component norms ===")
from dm_svd_embedding.hab_rdm_contract import (
    build_hab_rdm, _add_1e_cross_block_rdm, _add_nconserved_complementary,
    _add_pair_transfer_complementary, _add_3body_patterns, _add_1a3b_patterns,
)

# Rebuild H_AB piece by piece
HAB_1e = np.zeros((D, D))
HAB_nc = np.zeros((D, D))
HAB_pair = np.zeros((D, D))
HAB_3a1b = np.zeros((D, D))
HAB_1a3b = np.zeros((D, D))

for na in sorted(schmidt.keys()):
    if schmidt[na]['r'] == 0: continue
    _add_1e_cross_block_rdm(HAB_1e, block_offsets, na, trans_A, trans_B, h1eff, n_occ, n_act, n_virt)

for na in sorted(schmidt.keys()):
    if schmidt[na]['r'] == 0: continue
    os = block_offsets.get(na)
    if os is not None:
        _add_nconserved_complementary(HAB_nc, os, na, trans_A, trans_B, h2_4d, n_occ, n_act, n_virt)

for na in sorted(schmidt.keys()):
    if schmidt[na]['r'] == 0: continue
    _add_pair_transfer_complementary(HAB_pair, block_offsets, na, trans_A, trans_B, h2_4d, n_occ, n_act, n_virt)

for na in sorted(schmidt.keys()):
    _add_3body_patterns(HAB_3a1b, block_offsets, na, trans_A, trans_B, h2_4d, n_occ, n_act, n_virt)
    _add_1a3b_patterns(HAB_1a3b, block_offsets, na, trans_A, trans_B, h2_4d, n_occ, n_act, n_virt)

for name, hmat in [
    ("1e cross", HAB_1e), ("n-conserved 2e", HAB_nc), ("pair transfer", HAB_pair),
    ("3A+1B", HAB_3a1b), ("1A+3B", HAB_1a3b)
]:
    nrm = np.linalg.norm(hmat)
    print(f"  {name:20s}: ||H||={nrm:.6f}")

# Check: is HAB_rdm = sum of components?
HAB_recon = HAB_1e + HAB_nc + HAB_pair + HAB_3a1b + HAB_1a3b
print(f"\n  HAB_rdm vs sum of components: max|diff|={np.abs(HAB_rdm - HAB_recon).max():.2e}")

# === Most wrong block ===
print(f"\n=== Worst block: n=3 (H₂O example) ===")
na = 3
rn = schmidt[na]['r']
bo = block_offsets[na]
for a_src in range(rn):
    for b_src in range(rn):
        for a_dst in range(rn):
            for b_dst in range(rn):
                s = bo + a_src * rn + b_src
                d = bo + a_dst * rn + b_dst
                diff = abs(HAB_ref[d, s] - HAB_rdm[d, s])
                ref_v = abs(HAB_ref[d, s])
                if diff > 0.1 and ref_v > 1.0:
                    print(f"  HAB[({a_dst},{b_dst}),({a_src},{b_src})]: ref={HAB_ref[d,s]:.6f}, rdm={HAB_rdm[d,s]:.6f}, diff={diff:.4f}")

print("\n=== Done ===")
