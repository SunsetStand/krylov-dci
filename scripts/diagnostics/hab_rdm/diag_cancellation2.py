#!/usr/bin/env python3
"""
Test: Is H_full[αβ, γβ] = 0 for α≠γ a GENERAL property?

If yes: H_AB_offdiag = -(HA_offdiag + HB_offdiag) is exact.
"""
import numpy as np
import sys
sys.path.insert(0, '/data/home/wangcx/krylov-dci')

from dm_svd_embedding.embedded_hamiltonian import (
    build_h_emb, _setup_h2o_system, _extract_subspace_integrals,
    _build_subspace_hamiltonian,
)
from dm_svd_embedding.occ_virt_partition import setup_partition, build_block_matrices
from dm_svd_embedding.density_matrix import compute_schmidt_decomposition

(mol, mf, cas, q_idx, backend, h1eff, h2_4d, fcivec, ci_flat,
 n_act, n_elec, n_occ, na, nb, ecore, E_fci) = _setup_h2o_system()
n_virt = n_act - n_occ

partition, full_dets = setup_partition(n_act, n_elec, n_occ, ms=0)
C_blocks = build_block_matrices(partition, ci_flat)
schmidt = compute_schmidt_decomposition(C_blocks, eps=1e-3)

H_ref, basis_ref, decomps_ref = build_h_emb(
    schmidt, partition, q_idx, backend, h1eff, h2_4d,
    n_occ, n_act, verbose=False)

HA_ref = decomps_ref['HA']
HB_ref = decomps_ref['HB']
HAB_ref = decomps_ref['HAB']

# === Test cancellation: H_full vs HA+HB ===
print("=== Cancellation test: H_full vs HA⊗I + I⊗HB ===")
print("(For OFF-diagonal, does H_AB exactly cancel HA+HB?)")

D = H_ref.shape[0]
mask_offdiag = (np.eye(D) == 0)
mask_diag = (np.eye(D) == 1)

# H_full = HA + HB + HAB
# If cancellation is exact, then for off-diagonal: H_full ≈ 0
# which means HAB ≈ -(HA + HB) for off-diagonal elements

h_full_offdiag_norm = np.linalg.norm(H_ref * mask_offdiag)
ha_hb_offdiag_norm = np.linalg.norm((HA_ref + HB_ref) * mask_offdiag)
hab_offdiag_norm = np.linalg.norm(HAB_ref * mask_offdiag)

print(f"  ||H_full[offdiag]||         = {h_full_offdiag_norm:.6f}")
print(f"  ||HA+HB[offdiag]||          = {ha_hb_offdiag_norm:.6f}")
print(f"  ||HAB[offdiag]||             = {hab_offdiag_norm:.6f}")
print(f"  ||HAB + (HA+HB)[offdiag]||   = {np.linalg.norm((HAB_ref + HA_ref + HB_ref) * mask_offdiag):.6e}")

# Off-diagonal per-block
block_offsets = {}
off = 0
all_blocks = sorted(schmidt.keys())
for na in all_blocks:
    block_offsets[na] = off
    off += schmidt[na]['r'] ** 2

print(f"\n=== Per-block: within-block vs cross-block off-diagonal ===")
within_cancel_err = 0.0
cross_cancel_err = 0.0
for na_src in all_blocks:
    rs = schmidt[na_src]['r']
    if rs == 0: continue
    os = block_offsets[na_src]
    for na_dst in all_blocks:
        rd = schmidt[na_dst]['r']
        if rd == 0: continue
        od = block_offsets[na_dst]
        for a in range(rs):
            for b in range(rs):
                s = os + a * rs + b
                for g in range(rd):
                    for d in range(rd):
                        t = od + g * rd + d
                        if s == t: continue
                        hf = H_ref[t, s]
                        hs = HA_ref[t, s] + HB_ref[t, s]
                        cancel = abs(hf)  # should be 0 if HAB = -(HA+HB)
                        if na_src == na_dst:
                            if cancel > within_cancel_err:
                                within_cancel_err = cancel
                        else:
                            if cancel > cross_cancel_err:
                                cross_cancel_err = cancel

print(f"  Max within-block |H_full[offdiag]| = {within_cancel_err:.6e}")
print(f"  Max cross-block |H_full[offdiag]|  = {cross_cancel_err:.6e}")

# === KEY TEST: per-block within-block H_full off-diagonal ===
print(f"\n=== Detailed: within-block H_full off-diagonals ===")
for na in all_blocks:
    rn = schmidt[na]['r']
    if rn < 2: continue
    bo = block_offsets[na]

    max_err = 0.0
    max_ha = 0.0
    max_hb = 0.0
    worst = None

    for a in range(rn):
        for b in range(rn):
            s = bo + a * rn + b
            for g in range(rn):
                for d_idx in range(rn):
                    t = bo + g * rn + d_idx
                    if s == t: continue

                    hf = H_ref[t, s]
                    ha = HA_ref[t, s]
                    hb = HB_ref[t, s]
                    hab = HAB_ref[t, s]

                    if abs(ha) > max_ha: max_ha = abs(ha)
                    if abs(hb) > max_hb: max_hb = abs(hb)

                    err = abs(hf)
                    if err > max_err:
                        max_err = err
                        worst = (a, b, g, d_idx, hf, ha, hb, ha+hb+hab)

    print(f"  n={na} r={rn}: max|HA_offdiag|={max_ha:.4f}, max|HB_offdiag|={max_hb:.4f}")
    if max_err < 1e-10:
        print(f"    → ALL within-block off-diagonals CANCEL perfectly ✅")
    else:
        print(f"    → max surviving |H_full| = {max_err:.6e} at {worst}")
        print(f"      H_full={worst[4]:.6f}, HA={worst[5]:.6f}, HB={worst[6]:.6f}, HA+HB+HAB={worst[7]:.6f}")

# === The big question: is HAB_offdiag = -(HA+HB)_offdiag ALWAYS? ===
print(f"\n=== Reconstruction: HAB_recon = -(HA+HB) off-diagonal ===")
HAB_recon = np.zeros((D, D))
for i in range(D):
    for j in range(D):
        if i != j:
            HAB_recon[i, j] = -(HA_ref[i, j] + HB_ref[i, j])

diff = np.abs(HAB_ref * mask_offdiag - HAB_recon * mask_offdiag).max()
print(f"  max|HAB_ref - HAB_recon|_offdiag = {diff:.6e}")

# What about the DIAGONAL? Can RDM handle diagonals correctly?
diag_ref = np.diag(HAB_ref)
diag_rdm = np.diag(decomps_ref['HAB'])  # same as reference in this run
# Actually need the RDM version
from dm_svd_embedding.hab_rdm_contract import build_hab_rdm
from dm_svd_embedding.transition_rdm import compute_transition_matrices
trans_A = compute_transition_matrices(partition, schmidt, n_occ, subspace='A', verbose=False)
trans_B = compute_transition_matrices(partition, schmidt, n_virt, subspace='B', verbose=False)

HAB_rdm_full = np.zeros((D, D))
block_offsets_dict = {na: block_offsets[na] for na in all_blocks}
build_hab_rdm(HAB_rdm_full, schmidt, block_offsets_dict, trans_A, trans_B, h1eff, h2_4d, n_occ, n_act, verbose=False)

diag_rdm_full = np.diag(HAB_rdm_full)
diag_ref_full = np.diag(HAB_ref)
diag_diff = np.abs(diag_rdm_full - diag_ref_full)
print(f"\n=== Diagonal comparison (RDM vs ref) ===")
print(f"  max|diag_RDM - diag_ref| = {diag_diff.max():.4f}")
print(f"  mean|diag_RDM - diag_ref| = {diag_diff.mean():.4f}")
print(f"  ||diag_ref|| = {np.linalg.norm(diag_ref_full):.4f}")
print(f"  ||diag_RDM|| = {np.linalg.norm(diag_rdm_full):.4f}")

# Combine: RDM diagonal + reconstructed off-diagonal
print(f"\n=== Combined H_AB: RDM diagonal + -(HA+HB) off-diagonal ===")
HAB_combined = np.copy(HAB_rdm_full)
for i in range(D):
    for j in range(D):
        if i != j:
            HAB_combined[i, j] = -(HA_ref[i, j] + HB_ref[i, j])

diff_combined = np.abs(HAB_ref - HAB_combined).max()
print(f"  max|HAB_ref - HAB_combined| = {diff_combined:.6e}")

# Now build full H_emb with this combined H_AB
H_combined = HA_ref + HB_ref + HAB_combined
ev_ref, _ = np.linalg.eigh(H_ref)
ev_combined, _ = np.linalg.eigh(H_combined)
print(f"\n  Eigenvalue comparison:")
for i in range(min(6, D)):
    diff_ev = abs(ev_ref[i] - ev_combined[i])
    print(f"    E[{i}]: ref={ev_ref[i]:.8f}, combined={ev_combined[i]:.8f}, diff={diff_ev:.6e}")

dE0 = (ev_combined[0] - ev_ref[0]) * 1000
print(f"\n  dE0 = {dE0:+.3f} mH")

print("\n=== Done ===")
