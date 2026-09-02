#!/usr/bin/env python3
"""
Element-by-element comparison of H_AB between reference (sigma-vector)
and RDM (complementary operator) approaches for a single Schmidt block.

Finds the matrix element with the largest discrepancy and reports the RDM
term-by-term breakdown to understand which contributions are missing/wrong.
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
from dm_svd_embedding.transition_rdm import compute_transition_matrices

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

trans_A = compute_transition_matrices(partition, schmidt, n_occ, subspace='A', verbose=False)
trans_B = compute_transition_matrices(partition, schmidt, n_virt, subspace='B', verbose=False)

# Build block_offsets (same as build_h_emb)
block_offsets = {}
offset = 0
all_blocks = sorted(schmidt.keys())
for n_A in all_blocks:
    rn = schmidt[n_A]['r']
    block_offsets[n_A] = offset
    offset += rn * rn

# Focus on n_A=3
n_A_target = 3
sd = schmidt[n_A_target]
r = sd['r']
blk = partition[n_A_target]

U = sd['U']
V = sd['V']

# Build HA_schmidt and HB_schmidt for this block
a_dets = blk['a_dets']
aA0, bA0 = a_dets[0]
nA_alpha = aA0.bit_count()
nA_beta = bA0.bit_count()
h1_A, h2_A = _extract_subspace_integrals(h1eff, h2_4d, np.arange(n_occ))
HA_det = _build_subspace_hamiltonian(a_dets, h1_A, h2_A, n_occ, nA_alpha, nA_beta)

b_dets = blk['b_dets']
bB0, bB0b = b_dets[0]
nB_alpha = bB0.bit_count()
nB_beta = bB0b.bit_count()
h1_B, h2_B = _extract_subspace_integrals(h1eff, h2_4d, np.arange(n_occ, n_act))
HB_det = _build_subspace_hamiltonian(b_dets, h1_B, h2_B, n_virt, nB_alpha, nB_beta)

HA_schmidt = U.T @ HA_det @ U
HB_schmidt = V.T @ HB_det @ V

# ── Scan for max |H_AB_ref| ──
print(f"=== Scanning n_A=3 block (r={r}) ===")
b_off = block_offsets[n_A_target]
max_ref = -1.0
best = None
for a in range(r):
    for b in range(r):
        ib = b_off + a * r + b
        for g in range(r):
            for d in range(r):
                ik = b_off + g * r + d
                v = abs(HAB_ref[ib, ik])
                if v > max_ref:
                    max_ref = v
                    best = (a, b, g, d)

print(f"Max |H_AB_ref| = {max_ref:.8f} at ({best[0]},{best[1]},{best[2]},{best[3]})")

# ── Compute RDM n-conserved for this element ──
alpha, beta, gamma, delta = best
TA = trans_A.trans_1.get(n_A_target)
TB = trans_B.trans_1.get(n_A_target)

terms = {
    'b': 0.0, 'c': 0.0, 'd': 0.0, 'e': 0.0
}

# Term (b): (i_A,j_A|k_B,l_B)  TA[i,j] TB[k,l]
for i in range(n_occ):
    for j in range(n_occ):
        ta = TA[alpha, gamma, i, j]
        if abs(ta) < 1e-14: continue
        for k in range(n_virt):
            for l in range(n_virt):
                tb = TB[beta, delta, k, l]
                if abs(tb) < 1e-14: continue
                v = h2_4d[i, j, k + n_occ, l + n_occ]
                terms['b'] += v * ta * tb

# Term (c): (i_A,j_B|k_B,l_A)  -TA[i,l] TB[k,j]
for i in range(n_occ):
    for la in range(n_occ):
        ta = TA[alpha, gamma, i, la]
        if abs(ta) < 1e-14: continue
        for jb in range(n_virt):
            for kb in range(n_virt):
                tb = TB[beta, delta, kb, jb]
                if abs(tb) < 1e-14: continue
                v = h2_4d[i, jb + n_occ, kb + n_occ, la]
                terms['c'] -= v * ta * tb

# Term (d): (i_B,j_B|k_A,l_A)  TB[i,j] TA[k,l]
for ib in range(n_virt):
    for jb in range(n_virt):
        tb = TB[beta, delta, ib, jb]
        if abs(tb) < 1e-14: continue
        for ka in range(n_occ):
            for la in range(n_occ):
                ta = TA[alpha, gamma, ka, la]
                if abs(ta) < 1e-14: continue
                v = h2_4d[ib + n_occ, jb + n_occ, ka, la]
                terms['d'] += v * ta * tb

# Term (e): (i_B,j_A|k_A,l_B)  -TB[i,l] TA[k,j]
for ib in range(n_virt):
    for lb in range(n_virt):
        tb = TB[beta, delta, ib, lb]
        if abs(tb) < 1e-14: continue
        for ja in range(n_occ):
            for ka in range(n_occ):
                ta = TA[alpha, gamma, ka, ja]
                if abs(ta) < 1e-14: continue
                v = h2_4d[ib + n_occ, ja, ka, lb + n_occ]
                terms['e'] -= v * ta * tb

rdm_no_half = sum(terms.values())
rdm_half = rdm_no_half * 0.5

ha = HA_schmidt[gamma, alpha] if beta == delta else 0.0
hb = HB_schmidt[delta, beta] if alpha == gamma else 0.0

ib_ref = b_off + alpha * r + beta
ik_ref = b_off + gamma * r + delta
ref_total = HAB_ref[ib_ref, ik_ref]

print(f"\nElement H_AB[n={n_A_target}, ({alpha},{beta}),({gamma},{delta})]:")
print(f"  Reference H_AB:  {ref_total:.12f}")
print(f"  HA_schmidt part: {ha:.12f}")
print(f"  HB_schmidt part: {hb:.12f}")
print(f"\n  RDM breakdown (no ½):")
for tname in ['b', 'c', 'd', 'e']:
    print(f"    Term ({tname}): {terms[tname]:.12f}")
print(f"    Total (no ½):     {rdm_no_half:.12f}")
print(f"  RDM (×½):           {rdm_half:.12f}")

print(f"\n  Ratio RDM(×½)/ref:  {rdm_half/ref_total:.6f}" if abs(ref_total) > 1e-14 else "")
print(f"  Ratio RDM(no½)/ref: {rdm_no_half/ref_total:.6f}" if abs(ref_total) > 1e-14 else "")

# Also scan for max |HAB_ref - HAB_rdm| in this block using full H_AB reference
print(f"\n=== Compare full H_AB for n_A=3 block ===")
# Rebuild H_AB_rdm for just this block using the updated code
from dm_svd_embedding.hab_rdm_contract import (
    _add_nconserved_complementary, _add_nconserved_complementary_b_to_a,
)
D = HAB_ref.shape[0]
HAB_rdm_block = np.zeros((D, D))
_add_nconserved_complementary(HAB_rdm_block, b_off, n_A_target, trans_A, trans_B, h2_4d, n_occ, n_act, n_virt)
_add_nconserved_complementary_b_to_a(HAB_rdm_block, b_off, n_A_target, trans_A, trans_B, h2_4d, n_occ, n_act, n_virt)

max_diff = 0.0
worst = None
for a in range(r):
    for b in range(r):
        ib = b_off + a * r + b
        for g in range(r):
            for d in range(r):
                ik = b_off + g * r + d
                diff_val = abs(HAB_ref[ib, ik] - HAB_rdm_block[ib, ik])
                if diff_val > max_diff:
                    max_diff = diff_val
                    worst = (a, b, g, d)

print(f"Max |HAB_ref - HAB_rdm| = {max_diff:.8f} at {worst}")

# Full block norms
bref = HAB_ref[b_off:b_off+r*r, b_off:b_off+r*r]
brdm = HAB_rdm_block[b_off:b_off+r*r, b_off:b_off+r*r]
print(f"||HAB_ref_block|| = {np.linalg.norm(bref):.6f}")
print(f"||HAB_rdm_block|| = {np.linalg.norm(brdm):.6f}")

print("\n=== Done ===")
