#!/usr/bin/env python3
"""
Complete audit of the complementary operator decomposition for H_AB.

For a SPECIFIC Schmidt basis element in the n_A=3 block:
1. Compute H_AB element-by-element using the determinant expansion
   (explicit Slater-Condon) — the TRUTH reference.
2. Compare with: (a) sigma-vector reference, (b) RDM per-term decomposition.
3. Verify each of the 4 n-conserved patterns independently.
"""
import numpy as np
import sys
sys.path.insert(0, '/data/home/wangcx/krylov-dci')

from dm_svd_embedding.embedded_hamiltonian import (
    _setup_h2o_system, _extract_subspace_integrals,
    _expand_schmidt_product_to_ci_matrix,
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

# ── Focus on n_A=3, pick a specific element ──
n_A = 3
sd = schmidt[n_A]
r = sd['r']
blk = partition[n_A]
U = sd['U']
V = sd['V']

print(f"n_A={n_A}, r={r}")
print(f"dim_A={U.shape[0]}, dim_B={V.shape[0]}")

# ── Build H_AB via explicit Slater-Condon in the Schmidt basis ──
# This is the GROUND TRUTH: no operator decomposition, just direct SC rules.

alpha_strs = q_idx.alpha_strs
beta_strs = q_idx.beta_strs
n_as = len(alpha_strs)
n_bs = len(beta_strs)
alpha_to_idx = {int(s): i for i, s in enumerate(alpha_strs)}
beta_to_idx = {int(s): i for i, s in enumerate(beta_strs)}

# Full Hamiltonian in the determinant basis (using sigma_vector for efficiency)
# For each pair of Schmidt basis states, we compute:
# ⟨Ã_α|⟨B̃_β| H |Ã_γ⟩|B̃_δ⟩

# Actually, let's use a simpler approach: compute H^emb = C^T H C for this block
# by expanding all r×r Schmidt states and computing sigma-vectors.

# Expand each Schmidt state
ci_mats = []
for a in range(r):
    ci_beta = np.zeros(n_virt)  # dummy, will expand differently
    for b in range(r):
        ci_mat = _expand_schmidt_product_to_ci_matrix(
            a, b, sd, blk, n_as, n_bs, n_occ,
            alpha_strs, beta_strs, alpha_to_idx, beta_to_idx)
        ci_mats.append(ci_mat)

# Compute sigma for each
sigmas = []
for k, ci_mat in enumerate(ci_mats):
    sigma_mat = backend.sigma_full(ci_mat)
    sigmas.append(sigma_mat)

# Build H^emb for this block
H_block = np.zeros((r*r, r*r))
for i in range(r*r):
    for j in range(r*r):
        H_block[i, j] = np.sum(ci_mats[i] * sigmas[j])

# Now compute HA and HB for this block via Path C
h1_A, h2_A = _extract_subspace_integrals(h1eff, h2_4d, np.arange(n_occ))
h1_B, h2_B = _extract_subspace_integrals(h1eff, h2_4d, np.arange(n_occ, n_act))

from dm_svd_embedding.embedded_hamiltonian import _build_subspace_hamiltonian

a_dets = blk['a_dets']
aA0, bA0 = a_dets[0]
nA_a = aA0.bit_count()
nA_b = bA0.bit_count()
HA_det = _build_subspace_hamiltonian(a_dets, h1_A, h2_A, n_occ, nA_a, nA_b)
HA_schmidt = U.T @ HA_det @ U

b_dets = blk['b_dets']
bB0, bB0b = b_dets[0]
nB_a = bB0.bit_count()
nB_b = bB0b.bit_count()
HB_det = _build_subspace_hamiltonian(b_dets, h1_B, h2_B, n_virt, nB_a, nB_b)
HB_schmidt = V.T @ HB_det @ V

# H_AB from truth
HAB_truth = np.zeros((r*r, r*r))
for a_src in range(r):
    for b_src in range(r):
        ik = a_src * r + b_src
        for a_dst in range(r):
            for b_dst in range(r):
                ib = a_dst * r + b_dst
                ha = HA_schmidt[a_dst, a_src] if b_dst == b_src else 0.0
                hb = HB_schmidt[b_dst, b_src] if a_dst == a_src else 0.0
                HAB_truth[ib, ik] = H_block[ib, ik] - ha - hb

print(f"||H_block|| = {np.linalg.norm(H_block):.6f}")
print(f"||HA_schmidt|| = {np.linalg.norm(HA_schmidt):.6f}")
print(f"||HB_schmidt|| = {np.linalg.norm(HB_schmidt):.6f}")
print(f"||HAB_truth|| = {np.linalg.norm(HAB_truth):.6f}")

# ── Now compute H_AB via RDM n-conserved patterns ──
trans_A = compute_transition_matrices(partition, schmidt, n_occ, subspace='A', verbose=False)
trans_B = compute_transition_matrices(partition, schmidt, n_virt, subspace='B', verbose=False)

TA = trans_A.trans_1.get(n_A)  # (r, r, n_occ, n_occ)
TB = trans_B.trans_1.get(n_A)  # (r, r, n_virt, n_virt)

print(f"\nTA.shape = {TA.shape}, TB.shape = {TB.shape}")

# RDM: pattern by pattern
HAB_b = np.zeros((r*r, r*r))   # (A,A|B,B)
HAB_c = np.zeros((r*r, r*r))   # (A,B|B,A)
HAB_d = np.zeros((r*r, r*r))   # (B,B|A,A)
HAB_e = np.zeros((r*r, r*r))   # (B,A|A,B)

for a_dst in range(r):
    for a_src in range(r):
        for b_dst in range(r):
            for b_src in range(r):
                ib = a_dst * r + b_dst
                ik = a_src * r + b_src

                # Term (b): (i_A,j_A|k_B,l_B) → +TA[i,j] · TB[k,l]
                vb = 0.0
                for i in range(n_occ):
                    for j in range(n_occ):
                        ta = TA[a_dst, a_src, i, j]
                        if abs(ta) < 1e-14: continue
                        for k in range(n_virt):
                            for li in range(n_virt):
                                tb = TB[b_dst, b_src, k, li]
                                if abs(tb) < 1e-14: continue
                                vb += h2_4d[i, j, k + n_occ, li + n_occ] * ta * tb
                HAB_b[ib, ik] = vb

                # Term (c): (i_A,j_B|k_B,l_A) → -TA[i,l] · TB[k,j]
                vc = 0.0
                for i in range(n_occ):
                    for la in range(n_occ):
                        ta = TA[a_dst, a_src, i, la]
                        if abs(ta) < 1e-14: continue
                        for jb in range(n_virt):
                            for kb in range(n_virt):
                                tb = TB[b_dst, b_src, kb, jb]
                                if abs(tb) < 1e-14: continue
                                vc -= h2_4d[i, jb + n_occ, kb + n_occ, la] * ta * tb
                HAB_c[ib, ik] = vc

                # Term (d): (i_B,j_B|k_A,l_A) → +TB[i,j] · TA[k,l]
                vd = 0.0
                for ibb in range(n_virt):
                    for jbb in range(n_virt):
                        tb_val = TB[b_dst, b_src, ibb, jbb]
                        if abs(tb_val) < 1e-14: continue
                        for ka in range(n_occ):
                            for la in range(n_occ):
                                ta_val = TA[a_dst, a_src, ka, la]
                                if abs(ta_val) < 1e-14: continue
                                vd += h2_4d[ibb + n_occ, jbb + n_occ, ka, la] * ta_val * tb_val
                HAB_d[ib, ik] = vd

                # Term (e): (i_B,j_A|k_A,l_B) → -TB[i,l] · TA[k,j]
                ve = 0.0
                for ibb in range(n_virt):
                    for lbb in range(n_virt):
                        tb_val = TB[b_dst, b_src, ibb, lbb]
                        if abs(tb_val) < 1e-14: continue
                        for ja in range(n_occ):
                            for ka in range(n_occ):
                                ta_val = TA[a_dst, a_src, ka, ja]
                                if abs(ta_val) < 1e-14: continue
                                ve -= h2_4d[ibb + n_occ, ja, ka, lbb + n_occ] * ta_val * tb_val
                HAB_e[ib, ik] = ve

HAB_rdm = HAB_b + HAB_c + HAB_d + HAB_e

# ── Compare ──
print(f"\n=== Comparison ===")
for name, hmat in [("b (A,A|B,B)", HAB_b), ("c (A,B|B,A)", HAB_c),
                    ("d (B,B|A,A)", HAB_d), ("e (B,A|A,B)", HAB_e),
                    ("RDM total", HAB_rdm)]:
    print(f"  ||HAB_{name:20s}|| = {np.linalg.norm(hmat):.8f}")

print(f"\n  ||HAB_truth||       = {np.linalg.norm(HAB_truth):.8f}")
print(f"  ||HAB_rdm||         = {np.linalg.norm(HAB_rdm):.8f}")

# Check: does HAB_rdm = HAB_truth × 2? (missing ½ factor)
ratio_nohalf = np.linalg.norm(HAB_rdm) / np.linalg.norm(HAB_truth)
print(f"\n  Ratio ||RDM||/||truth|| (no ½) = {ratio_nohalf:.6f}")

# What about × ½?
HAB_rdm_half = HAB_rdm * 0.5
ratio_half = np.linalg.norm(HAB_rdm_half) / np.linalg.norm(HAB_truth)
print(f"  Ratio ||RDM×½||/||truth||       = {ratio_half:.6f}")

# Element-by-element max diff
diff_nohalf = np.abs(HAB_truth - HAB_rdm).max()
diff_half = np.abs(HAB_truth - HAB_rdm_half).max()
print(f"\n  max|truth - RDM|     = {diff_nohalf:.8f}")
print(f"  max|truth - RDM×½|   = {diff_half:.8f}")

# Compute the EXACT factor that minimizes the difference
# (least-squares best fit scaling factor)
num = np.sum(HAB_truth * HAB_rdm)
den = np.sum(HAB_rdm * HAB_rdm)
best_scale = num / den if den > 1e-14 else 0.0
print(f"\n  Best-fit scale RDM×α = truth: α = {best_scale:.8f}")
print(f"  (1/2 = 0.5, 2/3 ≈ 0.667, 1/3 ≈ 0.333, 2× ?)")

# Compare with the diagonal element for diagnostics
idx_diag = 3 * r + 2  # alpha=3, beta=2
print(f"\n  Diagonal element (3,2):")
print(f"    HAB_truth = {HAB_truth[idx_diag, idx_diag]:.12f}")
print(f"    HAB_rdm   = {HAB_rdm[idx_diag, idx_diag]:.12f}")
print(f"    Term (b)  = {HAB_b[idx_diag, idx_diag]:.12f}")
print(f"    Term (c)  = {HAB_c[idx_diag, idx_diag]:.12f}")
print(f"    Term (d)  = {HAB_d[idx_diag, idx_diag]:.12f}")
print(f"    Term (e)  = {HAB_e[idx_diag, idx_diag]:.12f}")

# Also check: is the truth H_AB correct by comparing with sigma-vector direct
print(f"\n=== Verification: truth vs sigma-vector ===")
# H_AB from sigma in the Schmidt basis for this block
from dm_svd_embedding.embedded_hamiltonian import build_h_emb
H_full, _, decomps = build_h_emb(schmidt, partition, q_idx, backend, h1eff, h2_4d, n_occ, n_act, verbose=False)
HAB_full = decomps['HAB']

block_offsets = {}
off = 0
for na in sorted(schmidt.keys()):
    rn = schmidt[na]['r']
    block_offsets[na] = off
    off += rn * rn
bo = block_offsets[n_A]
HAB_ref_block = HAB_full[bo:bo+r*r, bo:bo+r*r]

# Check agreement
diff_truth_ref = np.abs(HAB_truth - HAB_ref_block).max()
print(f"  max|HAB_truth - HAB_ref| = {diff_truth_ref:.2e}")
# If ≈0, truth is consistent with reference

print("\n=== Done ===")
