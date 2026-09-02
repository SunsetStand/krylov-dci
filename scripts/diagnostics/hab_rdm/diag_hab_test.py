#!/usr/bin/env python3
"""
Diagnostic: decompose H_AB reference and RDM into components,
compare them term-by-term to isolate the bug.
"""
import numpy as np
import sys, os
sys.path.insert(0, '/data/home/wangcx/krylov-dci')

from dm_svd_embedding.embedded_hamiltonian import (
    build_h_emb, _setup_h2o_system, _extract_subspace_integrals,
    _build_subspace_hamiltonian,
)
from dm_svd_embedding.occ_virt_partition import setup_partition, build_block_matrices
from dm_svd_embedding.density_matrix import compute_schmidt_decomposition
from dm_svd_embedding.transition_rdm import compute_transition_matrices
from dm_svd_embedding.hab_rdm_contract import (
    build_hab_rdm, _add_1e_cross_block_rdm, _add_nconserved_complementary,
    _add_pair_transfer_complementary, _add_3body_patterns, _add_1a3b_patterns,
)
from typing import Dict

# ── Setup ──
(mol, mf, cas, q_idx, backend, h1eff, h2_4d, fcivec, ci_flat,
 n_act, n_elec, n_occ, na, nb, ecore, E_fci) = _setup_h2o_system()
n_virt = n_act - n_occ
print(f"H₂O/STO-3G CAS({n_act},{n_elec}) A={n_occ} B={n_virt}")

partition, full_dets = setup_partition(n_act, n_elec, n_occ, ms=0)
C_blocks = build_block_matrices(partition, ci_flat)
schmidt = compute_schmidt_decomposition(C_blocks, eps=1e-3)

# ── Reference: full sigma-vector H^emb ──
print("\n=== Reference H^emb (sigma-vector) ===")
H_ref, basis_ref, decomps_ref = build_h_emb(
    schmidt, partition, q_idx, backend, h1eff, h2_4d,
    n_occ, n_act, verbose=True)
D = H_ref.shape[0]
H_AB_ref = decomps_ref['HAB']
HA_ref = decomps_ref['HA']
HB_ref = decomps_ref['HB']

# ── Build transition matrices ──
trans_A = compute_transition_matrices(
    partition, schmidt, n_occ, subspace='A', verbose=False)
trans_B = compute_transition_matrices(
    partition, schmidt, n_virt, subspace='B', verbose=False)

# ── Build H_AB via RDM, COMPONENT BY COMPONENT ──
# Replicate block_offsets from build_h_emb
block_offsets: Dict[int, int] = {}
offset = 0
for n_A in sorted(schmidt.keys()):
    r = schmidt[n_A]['r']
    block_offsets[n_A] = offset
    offset += r * r

assert offset == D, f"Dimension mismatch: {offset} != {D}"

# Build each component separately
H_AB_1e = np.zeros((D, D))
H_AB_nc = np.zeros((D, D))       # n-conserved 2e
H_AB_pair = np.zeros((D, D))     # pair transfer
H_AB_3a1b = np.zeros((D, D))     # 3A+1B
H_AB_1a3b = np.zeros((D, D))     # 1A+3B

all_blocks = sorted(schmidt.keys())
for n_A in all_blocks:
    if schmidt[n_A]['r'] == 0:
        continue
    _add_1e_cross_block_rdm(H_AB_1e, block_offsets, n_A, trans_A, trans_B, h1eff, n_occ, n_act, n_virt)

for n_A in all_blocks:
    if schmidt[n_A]['r'] == 0:
        continue
    os = block_offsets.get(n_A)
    if os is not None:
        _add_nconserved_complementary(H_AB_nc, os, n_A, trans_A, trans_B, h2_4d, n_occ, n_act, n_virt)

for n_A in all_blocks:
    if schmidt[n_A]['r'] == 0:
        continue
    _add_pair_transfer_complementary(H_AB_pair, block_offsets, n_A, trans_A, trans_B, h2_4d, n_occ, n_act, n_virt)

for n_A in all_blocks:
    _add_3body_patterns(H_AB_3a1b, block_offsets, n_A, trans_A, trans_B, h2_4d, n_occ, n_act, n_virt)
    _add_1a3b_patterns(H_AB_1a3b, block_offsets, n_A, trans_A, trans_B, h2_4d, n_occ, n_act, n_virt)

# Also compute full RDM H_AB
H_AB_rdm = np.zeros((D, D))
build_hab_rdm(H_AB_rdm, schmidt, block_offsets, trans_A, trans_B, h1eff, h2_4d, n_occ, n_act, verbose=False)

# ── Compare each component against reference ──
print("\n=== Component-by-component comparison ===")
H_ref_full = HA_ref + HB_ref + H_AB_ref
H_rdm_full = HA_ref + HB_ref + H_AB_rdm
print(f"max|H_full_ref - H_full_rdm| = {np.abs(H_ref_full - H_rdm_full).max():.6e}")

# Split reference H_AB into n_A-conserved vs off-diagonal blocks
# n-conserved: same n_A
H_AB_ref_nc = np.zeros((D, D))
H_AB_ref_off = np.zeros((D, D))
for n_A_src in all_blocks:
    r_src = schmidt[n_A_src]['r']
    os = block_offsets[n_A_src]
    for n_A_dst in all_blocks:
        r_dst = schmidt[n_A_dst]['r']
        od = block_offsets[n_A_dst]
        for a_src in range(r_src):
            for b_src in range(r_src):
                s = os + a_src * r_src + b_src
                for a_dst in range(r_dst):
                    for b_dst in range(r_dst):
                        d = od + a_dst * r_dst + b_dst
                        v = H_AB_ref[d, s]
                        if n_A_src == n_A_dst:
                            H_AB_ref_nc[d, s] = v
                        else:
                            H_AB_ref_off[d, s] = v

components = [
    ("1e cross",        H_AB_1e,    None),
    ("n-conserved 2e",  H_AB_nc,    H_AB_ref_nc),
    ("pair transfer",   H_AB_pair,  None),
    ("3A+1B",           H_AB_3a1b,  None),
    ("1A+3B",           H_AB_1a3b,  None),
]

H_AB_rdm_reconstructed = H_AB_1e + H_AB_nc + H_AB_pair + H_AB_3a1b + H_AB_1a3b

for name, comp, ref_comp in components:
    nc = np.linalg.norm(comp)
    if ref_comp is not None:
        nc_ref = np.linalg.norm(ref_comp)
        diff = np.abs(comp - ref_comp).max()
        print(f"  {name:20s}: ||comp||={nc:.4f}, ||ref||={nc_ref:.4f}, max|diff|={diff:.6e}")
    else:
        print(f"  {name:20s}: ||comp||={nc:.4f}")

# Compare full H_AB_rdm vs component sum
print(f"\n  H_AB_rdm vs sum of components:")
print(f"    ||H_AB_rdm|| = {np.linalg.norm(H_AB_rdm):.4f}")
print(f"    ||sum comps|| = {np.linalg.norm(H_AB_rdm_reconstructed):.4f}")
print(f"    max|rdm - sum| = {np.abs(H_AB_rdm - H_AB_rdm_reconstructed).max():.6e}")

# Compare with reference H_AB
print(f"\n  Reference H_AB decomposed:")
print(f"    ||H_AB_nc(ref)|| = {np.linalg.norm(H_AB_ref_nc):.4f}")
print(f"    ||H_AB_off(ref)|| = {np.linalg.norm(H_AB_ref_off):.4f}")

# Compare RDM n-conserved vs ref n-conserved
diff_nc = np.abs(H_AB_nc - H_AB_ref_nc).max()
print(f"\n  n-conserved: max|rdm - ref| = {diff_nc:.6e}")

# Also compare per-block contributions
print(f"\n=== Per-block H_AB_ref analysis ===")
for n_A_src in all_blocks:
    for n_A_dst in all_blocks:
        if n_A_src == n_A_dst:
            r = schmidt[n_A_src]['r']
            os = block_offsets[n_A_src]
            # Extract this block from H_AB_ref and H_AB_nc
            block_ref = np.zeros((r*r, r*r))
            block_rdm = np.zeros((r*r, r*r))
            for a_src in range(r):
                for b_src in range(r):
                    s_loc = a_src * r + b_src
                    for a_dst in range(r):
                        for b_dst in range(r):
                            d_loc = a_dst * r + b_dst
                            block_ref[d_loc, s_loc] = H_AB_ref[os + d_loc, os + s_loc]
                            block_rdm[d_loc, s_loc] = H_AB_rdm[os + d_loc, os + s_loc]
            diff_block = np.abs(block_ref - block_rdm).max()
            if diff_block > 1e-10:
                nr = np.linalg.norm(block_ref)
                nd = np.linalg.norm(block_rdm)
                print(f"  n={n_A_src}: r={r}, ||ref||={nr:.4f}, ||rdm||={nd:.4f}, "
                      f"max|diff|={diff_block:.6e}")

print("\n=== Done ===")
