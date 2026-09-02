#!/usr/bin/env python3
"""
Definitive test: for ONE specific Schmidt state |Ã_0⟩|B̃_0⟩ in block n_A=3,
compute the expectation value ⟨H_AB⟩ using three independent methods:

1. Reference (sigma-vector projection)
2. Direct 4-index integral contraction (ALL 2A+2B terms, brute force)
3. RDM complementary operator decomposition (4 patterns)

If methods 1 and 2 agree but 3 doesn't, the RDM decomposition is wrong.
"""
import numpy as np
import sys, os
sys.path.insert(0, '/data/home/wangcx/krylov-dci')

from dm_svd_embedding.embedded_hamiltonian import (
    build_h_emb, _setup_h2o_system, _extract_subspace_integrals,
    _build_subspace_hamiltonian, _expand_schmidt_product_to_ci_matrix,
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

n_A = 3
sd = schmidt[n_A]
r = sd['r']
blk = partition[n_A]
U = sd['U']
V = sd['V']

# ── Pick Schmidt state |α=0, β=0⟩ ──
alpha = 0
beta = 0

alpha_strs = q_idx.alpha_strs
beta_strs = q_idx.beta_strs
n_as = len(alpha_strs)
n_bs = len(beta_strs)
alpha_to_idx = {int(s): i for i, s in enumerate(alpha_strs)}
beta_to_idx = {int(s): i for i, s in enumerate(beta_strs)}

ci_ket = _expand_schmidt_product_to_ci_matrix(
    alpha, beta, sd, blk, n_as, n_bs, n_occ,
    alpha_strs, beta_strs, alpha_to_idx, beta_to_idx)

# ── Method 1: Sigma-vector reference ──
sigma_ket = backend.sigma_full(ci_ket)
h_expect_ref = float(np.sum(ci_ket * sigma_ket))

# HA and HB expectations
h1_A, h2_A = _extract_subspace_integrals(h1eff, h2_4d, np.arange(n_occ))
h1_B, h2_B = _extract_subspace_integrals(h1eff, h2_4d, np.arange(n_occ, n_act))

a_dets = blk['a_dets']
aA0, bA0 = a_dets[0]
HA_det = _build_subspace_hamiltonian(a_dets, h1_A, h2_A, n_occ, aA0.bit_count(), bA0.bit_count())
HA_schmidt = U.T @ HA_det @ U

b_dets = blk['b_dets']
bB0, bB0b = b_dets[0]
HB_det = _build_subspace_hamiltonian(b_dets, h1_B, h2_B, n_virt, bB0.bit_count(), bB0b.bit_count())
HB_schmidt = V.T @ HB_det @ V

ha_expect = HA_schmidt[alpha, alpha]
hb_expect = HB_schmidt[beta, beta]
hab_expect_ref = h_expect_ref - ha_expect - hb_expect

print(f"=== Method 1: Sigma-vector reference ===")
print(f"  ⟨H⟩ = {h_expect_ref:.12f}")
print(f"  ⟨HA⟩ = {ha_expect:.12f}")
print(f"  ⟨HB⟩ = {hb_expect:.12f}")
print(f"  ⟨H_AB⟩ = {hab_expect_ref:.12f}")

# ── Method 2: Direct 4-index integral summation ──
# For each (p,q,r,s) in the full active space, compute:
#   ½(pq|rs) × ⟨Ψ| a†_p a†_r a_s a_q |Ψ⟩
# and classify by A/B index assignments.

# Expand |Ψ⟩ = Σ_i Σ_j U[i,0] V[j,0] |a_i⟩|b_j⟩
# Need to compute the expectation of a†_p a†_r a_s a_q over this superposition

# Precompute: for each determinant in A-space |a_i⟩ and B-space |b_j⟩
# ⟨a_i,b_j| a†_p a†_r a_s a_q |a_k,b_l⟩ = matrix element
# But this requires explicit Slater-Condon for all 4-index combinations.

# Faster approach: use the transition matrices!
# For expectation value of (pq|rs) a†_p a†_r a_s a_q:
# ⟨α,β| a†_p a†_r a_s a_q |α,β⟩
# Depends on which spaces p,q,r,s are in.

# Let me just compute directly on the Schmidt basis using the transition matrices
# for ALL 2A+2B index patterns.

trans_A = compute_transition_matrices(partition, schmidt, n_occ, subspace='A', verbose=False)
trans_B = compute_transition_matrices(partition, schmidt, n_virt, subspace='B', verbose=False)

TA = trans_A.trans_1.get(n_A)
TB = trans_B.trans_1.get(n_A)

print(f"\n=== Method 2: Full 4-index sum (brute force) ===")

# Pattern (b): (A,A|B,B)
hab_b = 0.0
count_b = 0
for i in range(n_occ):
    for j in range(n_occ):
        ta = TA[alpha, alpha, i, j]
        if abs(ta) < 1e-14: continue
        for k in range(n_virt):
            for li in range(n_virt):
                tb = TB[beta, beta, k, li]
                if abs(tb) < 1e-14: continue
                count_b += 1
                hab_b += h2_4d[i, j, k + n_occ, li + n_occ] * ta * tb

# Pattern (c): (A,B|B,A)
hab_c = 0.0
count_c = 0
for i in range(n_occ):
    for la in range(n_occ):
        ta = TA[alpha, alpha, i, la]
        if abs(ta) < 1e-14: continue
        for jb in range(n_virt):
            for kb in range(n_virt):
                tb = TB[beta, beta, kb, jb]
                if abs(tb) < 1e-14: continue
                count_c += 1
                # JW: -a†_i a_l a†_k a_j = -(a†_i a_l)_A ⊗ (a†_k a_j)_B
                hab_c -= h2_4d[i, jb + n_occ, kb + n_occ, la] * ta * tb

# Pattern (d): (B,B|A,A)
hab_d = 0.0
count_d = 0
for ib in range(n_virt):
    for jb in range(n_virt):
        tb_val = TB[beta, beta, ib, jb]
        if abs(tb_val) < 1e-14: continue
        for ka in range(n_occ):
            for la in range(n_occ):
                ta_val = TA[alpha, alpha, ka, la]
                if abs(ta_val) < 1e-14: continue
                count_d += 1
                # JW: +a†_i a_j a†_k a_l → wait, this is B-dominant
                # (i_B,j_B|k_A,l_A) → a†_i(B) a†_k(A) a_l(A) a_j(B)
                # = +(a†_k a_l)_A ⊗ (a†_i a_j)_B
                hab_d += h2_4d[ib + n_occ, jb + n_occ, ka, la] * ta_val * tb_val

# Pattern (e): (B,A|A,B)
hab_e = 0.0
count_e = 0
for ib in range(n_virt):
    for lb in range(n_virt):
        tb_val = TB[beta, beta, ib, lb]
        if abs(tb_val) < 1e-14: continue
        for ja in range(n_occ):
            for ka in range(n_occ):
                ta_val = TA[alpha, alpha, ka, ja]
                if abs(ta_val) < 1e-14: continue
                count_e += 1
                # JW: -(a†_k a_j)_A ⊗ (a†_i a_l)_B
                hab_e -= h2_4d[ib + n_occ, ja, ka, lb + n_occ] * ta_val * tb_val

hab_direct = hab_b + hab_c + hab_d + hab_e
print(f"  Non-zero terms: b={count_b}, c={count_c}, d={count_d}, e={count_e}")
print(f"  hab_b = {hab_b:.12f}")
print(f"  hab_c = {hab_c:.12f}")
print(f"  hab_d = {hab_d:.12f}")
print(f"  hab_e = {hab_e:.12f}")
print(f"  hab_direct (sum) = {hab_direct:.12f}")

# Apply ½ factor (from Hamiltonian definition)
hab_direct_half = hab_direct * 0.5
print(f"  hab_direct×½ = {hab_direct_half:.12f}")
print(f"  hab_ref = {hab_expect_ref:.12f}")

ratio_no_half = hab_direct / hab_expect_ref if abs(hab_expect_ref) > 1e-14 else float('inf')
ratio_half = hab_direct_half / hab_expect_ref if abs(hab_expect_ref) > 1e-14 else float('inf')
print(f"  Ratio direct/ref = {ratio_no_half:.6f}")
print(f"  Ratio direct×½/ref = {ratio_half:.6f}")

# ── Now check: does Method 2 × ½ = Method 1? ──
print(f"\n  diff: ref - direct = {hab_expect_ref - hab_direct:.12f}")
print(f"  diff: ref - direct×½ = {hab_expect_ref - hab_direct_half:.12f}")

# ── Additional check: what does the HAMILTONIAN itself give for
#    this expectation (using full sigma-vector)? ──
# Already computed: hab_expect_ref

# Let's try the RDM module version
from dm_svd_embedding.hab_rdm_contract import (
    _add_nconserved_complementary, _add_nconserved_complementary_b_to_a
)
D_total = 0
for na in sorted(schmidt.keys()):
    D_total += schmidt[na]['r'] ** 2

block_offsets = {}
off = 0
for na in sorted(schmidt.keys()):
    rn = schmidt[na]['r']
    block_offsets[na] = off
    off += rn * rn
bo = block_offsets[n_A]

HAB_module = np.zeros((D_total, D_total))
_add_nconserved_complementary(HAB_module, bo, n_A, trans_A, trans_B, h2_4d, n_occ, n_act, n_virt)
_add_nconserved_complementary_b_to_a(HAB_module, bo, n_A, trans_A, trans_B, h2_4d, n_occ, n_act, n_virt)

idx_diag = bo + alpha * r + beta
hab_module = HAB_module[idx_diag, idx_diag]
print(f"\n  Module RDM (with ½): {hab_module:.12f}")

# Check individual terms from module
HAB_b_only = np.zeros((D_total, D_total))
HAB_c_only = np.zeros((D_total, D_total))
# The module version applies ½ factors inside, let me verify by
# checking the per-term output

print(f"\n=== Summary ===")
print(f"  Method 1 (sigma-vector): {hab_expect_ref:.12f}")
print(f"  Method 2 (direct × ½):   {hab_direct_half:.12f}")
print(f"  Method 3 (RDM module):   {hab_module:.12f}")
print(f"  Method 2 (direct no ½):  {hab_direct:.12f}")

if abs(hab_expect_ref) > 1e-14:
    print(f"  Ratio M1/M1 = {hab_expect_ref/hab_expect_ref:.6f}")
    print(f"  Ratio M2(×½)/M1 = {hab_direct_half/hab_expect_ref:.6f}")
    print(f"  Ratio M3/M1 = {hab_module/hab_expect_ref:.6f}")
    print(f"  Ratio M2(no½)/M1 = {hab_direct/hab_expect_ref:.6f}")

print("\n=== Done ===")
