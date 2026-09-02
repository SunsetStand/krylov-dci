#!/usr/bin/env python3
"""
Test: Is H_full[αβ, γβ] = 0 for α≠γ a GENERAL property or specific to H₂O?

Check ALL blocks and ALL α≠γ pairs. If it's universal, the fix is simple.
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

# Test 1: H₂O/STO-3G (default)
# Test 2: N₂/STO-3G (different system)

systems = []

# System 1: H₂O
print("=== System 1: H₂O/STO-3G ===")
(mol, mf, cas, q_idx, backend, h1eff, h2_4d, fcivec, ci_flat,
 n_act, n_elec, n_occ, na, nb, ecore, E_fci) = _setup_h2o_system()
n_virt = n_act - n_occ

partition, full_dets = setup_partition(n_act, n_elec, n_occ, ms=0)
C_blocks = build_block_matrices(partition, ci_flat)
schmidt = compute_schmidt_decomposition(C_blocks, eps=1e-3)

H_ref, basis_ref, decomps_ref = build_h_emb(
    schmidt, partition, q_idx, backend, h1eff, h2_4d,
    n_occ, n_act, verbose=False)

# Compute HA_schmidt and HB_schmidt per block
A_orb = np.arange(n_occ)
B_orb = np.arange(n_occ, n_act)

block_offsets = {}
off = 0
for na in sorted(schmidt.keys()):
    block_offsets[na] = off
    off += schmidt[na]['r'] ** 2

for na in sorted(schmidt.keys()):
    rn = schmidt[na]['r']
    if rn < 2: continue
    bo = block_offsets[na]

    sd = schmidt[na]
    blk = partition[na]
    U = sd['U']; V = sd['V']

    # HA_schmidt
    h1_A, h2_A = _extract_subspace_integrals(h1eff, h2_4d, A_orb)
    a_dets = blk['a_dets']
    if len(a_dets) > 0:
        HA_det = _build_subspace_hamiltonian(a_dets, h1_A, h2_A, n_occ,
                                              a_dets[0][0].bit_count(), a_dets[0][1].bit_count())
        HA_schmidt = U.T @ HA_det @ U
    else:
        HA_schmidt = np.zeros((rn, rn))

    # HB_schmidt
    h1_B, h2_B = _extract_subspace_integrals(h1eff, h2_4d, B_orb)
    b_dets = blk['b_dets']
    if len(b_dets) > 0:
        HB_det = _build_subspace_hamiltonian(b_dets, h1_B, h2_B, n_virt,
                                              b_dets[0][0].bit_count(), b_dets[0][1].bit_count())
        HB_schmidt = V.T @ HB_det @ V
    else:
        HB_schmidt = np.zeros((rn, rn))

    # Check: for all α≠γ, β=δ, is H_full close to 0?
    max_cancellation_err = 0.0
    max_ha_offdiag = 0.0
    worst = None

    for alpha in range(rn):
        for gamma in range(rn):
            if alpha == gamma: continue
            ha_coupling = HA_schmidt[gamma, alpha]
            if abs(ha_coupling) < 0.01: continue  # no coupling to cancel

            for beta in range(rn):
                s = bo + alpha * rn + beta   # ket
                d = bo + gamma * rn + beta   # bra
                h_full_val = H_ref[d, s]

                err = abs(h_full_val)
                if err > max_cancellation_err:
                    max_cancellation_err = err
                    worst = (na, alpha, beta, gamma, ha_coupling, h_full_val)
                if abs(ha_coupling) > max_ha_offdiag:
                    max_ha_offdiag = abs(ha_coupling)

    print(f"  n={na} r={rn}: max|HA_offdiag|={max_ha_offdiag:.4f}, "
          f"worst cancellation: α={worst[1]},{worst[3]}, β={worst[2]} at {worst[0]}, "
          f"HA_coupling={worst[4]:.4f}, H_full={worst[5]:.6f}")

# Check also B-off-diagonal cancellation
print(f"\n  B-off-diagonal cancellation check:")
for na in sorted(schmidt.keys()):
    rn = schmidt[na]['r']
    if rn < 2: continue
    bo = block_offsets[na]

    sd = schmidt[na]
    blk = partition[na]
    V = sd['V']

    h1_B, h2_B = _extract_subspace_integrals(h1eff, h2_4d, B_orb)
    b_dets = blk['b_dets']
    if len(b_dets) > 0:
        HB_det = _build_subspace_hamiltonian(b_dets, h1_B, h2_B, n_virt,
                                              b_dets[0][0].bit_count(), b_dets[0][1].bit_count())
        HB_schmidt = V.T @ HB_det @ V
    else:
        continue

    for beta in range(rn):
        for delta in range(rn):
            if beta == delta: continue
            hb_coupling = HB_schmidt[delta, beta]
            if abs(hb_coupling) < 0.01: continue

            for alpha in range(rn):
                s = bo + alpha * rn + beta
                d = bo + alpha * rn + delta
                h_full_val = H_ref[d, s]

                if abs(h_full_val) > 0.1:
                    print(f"    B-offdiag: n={na}, α={alpha}, β={beta}→{delta}, "
                          f"HB_coupling={hb_coupling:.4f}, H_full={h_full_val:.6f}")

print(f"  All others: H_full → 0 within tolerance")

# Now: test if we can simply reconstruct H_AB_offdiag from HA and HB
print(f"\n=== Test: H_AB_offdiag = -(HA_offdiag + HB_offdiag) ===")
HAB_offdiag_recon = np.zeros_like(decomps_ref['HAB'])
HA_ref = decomps_ref['HA']
HB_ref = decomps_ref['HB']

for i in range(H_ref.shape[0]):
    for j in range(H_ref.shape[0]):
        if i == j: continue
        # The off-diagonal H_AB should cancel HA and HB contributions
        HAB_offdiag_recon[i, j] = -(HA_ref[i, j] + HB_ref[i, j])

# Compare with reference H_AB (off-diagonal only)
HAB_ref = decomps_ref['HAB']
mask = np.eye(H_ref.shape[0]) == 0
diff = np.abs(HAB_ref[mask] - HAB_offdiag_recon[mask]).max()
norm_ref = np.linalg.norm(HAB_ref * mask)
norm_recon = np.linalg.norm(HAB_offdiag_recon * mask)

print(f"  ||HAB_ref_offdiag|| = {norm_ref:.4f}")
print(f"  ||HAB_recon_offdiag|| = {norm_recon:.4f}")
print(f"  max|ref - recon| (off-diag) = {diff:.6e}")

# How much of HAB is off-diagonal vs diagonal?
HAB_diag_norm = np.linalg.norm(np.diag(HAB_ref))
print(f"  ||HAB_ref_diag|| = {HAB_diag_norm:.4f}")
print(f"  Ratio offdiag/diag = {norm_ref/HAB_diag_norm:.4f}")

print("\n=== Done ===")
