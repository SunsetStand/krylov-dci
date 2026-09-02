#!/usr/bin/env python3
"""
Root-cause test: verify that the Schmidt-basis transition matrices
correctly reproduce the 2-body Hamiltonian matrix elements.

For a specific Schmidt state pair and a specific 2e integral (p,q,r,s),
compare the matrix element computed via:
  (a) transition matrices TA × TB
  (b) direct expansion into determinants + sigma-vector

If (a) ≠ (b), the transition matrix or phase convention is wrong.
"""
import numpy as np
import sys
sys.path.insert(0, '/data/home/wangcx/krylov-dci')

from dm_svd_embedding.embedded_hamiltonian import (
    _setup_h2o_system, _extract_subspace_integrals,
    _build_subspace_hamiltonian, _expand_schmidt_product_to_ci_matrix,
)
from dm_svd_embedding.occ_virt_partition import setup_partition, build_block_matrices
from dm_svd_embedding.density_matrix import compute_schmidt_decomposition
from dm_svd_embedding.transition_rdm import compute_transition_matrices

(mol, mf, cas, q_idx, backend, h1eff, h2_4d, fcivec, ci_flat,
 n_act, n_elec, n_occ, na, nb, ecore, E_fci) = _setup_h2o_system()
n_virt = n_act - n_occ
print(f"H₂O CAS({n_act},{n_elec}) n_occ={n_occ} n_virt={n_virt}")

partition, full_dets = setup_partition(n_act, n_elec, n_occ, ms=0)
C_blocks = build_block_matrices(partition, ci_flat)
schmidt = compute_schmidt_decomposition(C_blocks, eps=1e-3)

trans_A = compute_transition_matrices(partition, schmidt, n_occ, subspace='A', verbose=False)
trans_B = compute_transition_matrices(partition, schmidt, n_virt, subspace='B', verbose=False)

alpha_strs = q_idx.alpha_strs
beta_strs = q_idx.beta_strs
n_as = len(alpha_strs)
n_bs = len(beta_strs)
alpha_to_idx = {int(s): i for i, s in enumerate(alpha_strs)}
beta_to_idx = {int(s): i for i, s in enumerate(beta_strs)}

# Focus on n_A=3, pick non-trivial Schmidt states
n_A = 3
sd = schmidt[n_A]
r = sd['r']
blk = partition[n_A]
U = sd['U']
V = sd['V']

# Pick states where TA and TB are non-trivial
TA = trans_A.trans_1.get(n_A)
TB = trans_B.trans_1.get(n_A)

# Find non-trivial element
alpha, gamma = 0, 3  # from earlier scan: TA[0,3] has norm 0.658
beta, delta = 0, 3    # TB[0,3] has norm 1.000

print(f"\nTesting state pair: n_A={n_A}, (α={alpha},β={beta})→(γ={gamma},δ={delta})")

# ── Method A: Transition matrix ──
# Compute one specific 2e term: (i=0,j=1,k=0,l=0) in B space
# (i_A=0, j_A=1 | k_B=0, l_B=0) → (0,1|3,3) since B starts at n_occ=3
i_a, j_a, k_b, l_b = 0, 1, 0, 0
integral = h2_4d[i_a, j_a, k_b + n_occ, l_b + n_occ]
print(f"\nTest integral: ({i_a},{j_a}|{k_b+n_occ},{l_b+n_occ}) = {integral:.12f}")

ta = TA[alpha, gamma, i_a, j_a]
tb = TB[beta, delta, k_b, l_b]
print(f"TA[{alpha},{gamma},{i_a},{j_a}] = {ta:.12f}")
print(f"TB[{beta},{delta},{k_b},{l_b}] = {tb:.12f}")

hab_tm = integral * ta * tb
print(f"TM: {integral:.6f} × {ta:.6f} × {tb:.6f} = {hab_tm:.12f}")

# ── Method B: Direct determinant expansion + Slater-Condon ──
ci_bra = _expand_schmidt_product_to_ci_matrix(
    alpha, beta, sd, blk, n_as, n_bs, n_occ,
    alpha_strs, beta_strs, alpha_to_idx, beta_to_idx)
ci_ket = _expand_schmidt_product_to_ci_matrix(
    gamma, delta, sd, blk, n_as, n_bs, n_occ,
    alpha_strs, beta_strs, alpha_to_idx, beta_to_idx)

# Build the specific 2-body Hamiltonian for this ONE term
# H_one_term = ½ × (ij|kl) × a†_i a†_k a_l a_j
# We need: ⟨bra| a†_i_a a†_k_b a_l_b a_j_a |ket⟩
# = Σ_{a_i,b_j,a_k,b_l} U[a_i,α] V[b_j,β] U[a_k,γ] V[b_l,δ]
#   × ⟨a_i,b_j|a†_i_a a†_k_b a_l_b a_j_a|a_k,b_l⟩

# Expand bra and ket in determinant basis
a_dets = blk['a_dets']
b_dets = blk['b_dets']
a_index = blk['a_index']
b_index = blk['b_index']

# Compute matrix element of a†_p(A)a†_r(B)a_s(B)a_q(A) between determinants
# For A-space: ⟨a_i|a†_p a_q|a_k⟩
# For B-space: ⟨b_j|a†_r a_s|b_l⟩
# Because A and B operators commute (even×even)

from dm_svd_embedding.transition_rdm import (
    _compute_det_1body_transition,
    _popcount, _create_sign, _annihilate_sign
)

# Compute A-space 1-body transition in determinant basis
nA_dets = len(a_dets)
TA_det = np.zeros((nA_dets, nA_dets, n_occ, n_occ))
for ki, (aA_k, bA_k) in enumerate(a_dets):
    T_row = np.zeros((nA_dets, n_occ, n_occ))
    for j in range(n_occ):
        if not ((aA_k >> j) & 1): continue  # must be occupied to annihilate
        phase_q, a1 = _annihilate_sign(aA_k, j)
        for p in range(n_occ):
            if (a1 >> p) & 1: continue  # must be empty to create
            if p == j:
                # diagonal: a_p† a_p = number operator
                T_row[ki, p, j] = 1.0
                continue
            phase_p, a2 = _create_sign(a1, p)
            if phase_p == 0: continue
            key = (a2, bA_k)
            i = a_index.get(key)
            if i is not None:
                T_row[i, p, j] = phase_q * phase_p

    # Beta part
    for j in range(n_occ):
        if not ((bA_k >> j) & 1): continue
        phase_q, b1 = _annihilate_sign(bA_k, j)
        for p in range(n_occ):
            if (b1 >> p) & 1: continue
            if p == j:
                T_row[ki, p, j] += 1.0
                continue
            phase_p, b2 = _create_sign(b1, p)
            if phase_p == 0: continue
            key = (aA_k, b2)
            i = a_index.get(key)
            if i is not None:
                T_row[i, p, j] += phase_q * phase_p

    TA_det[:, ki, :, :] = T_row

# Compute B-space 1-body transition（B uses n_virt orbitals, indices 0..n_virt-1
# but the DETERMINANTS use B-local bit strings starting from 0）
nB_dets = len(b_dets)
TB_det = np.zeros((nB_dets, nB_dets, n_virt, n_virt))
for ki, (aB_k, bB_k) in enumerate(b_dets):
    for j in range(n_virt):
        if not ((aB_k >> j) & 1): continue
        phase_q, a1 = _annihilate_sign(aB_k, j)
        for p in range(n_virt):
            if (a1 >> p) & 1: continue
            if p == j:
                TB_det[ki, ki, p, j] += 1.0
                continue
            phase_p, a2 = _create_sign(a1, p)
            if phase_p == 0: continue
            key = (a2, bB_k)
            i = b_index.get(key)
            if i is not None:
                TB_det[i, ki, p, j] = phase_q * phase_p

    # Beta
    for j in range(n_virt):
        if not ((bB_k >> j) & 1): continue
        phase_q, b1 = _annihilate_sign(bB_k, j)
        for p in range(n_virt):
            if (b1 >> p) & 1: continue
            if p == j:
                TB_det[ki, ki, p, j] += 1.0
                continue
            phase_p, b2 = _create_sign(b1, p)
            if phase_p == 0: continue
            key = (aB_k, b2)
            i = b_index.get(key)
            if i is not None:
                TB_det[i, ki, p, j] += phase_q * phase_p

# Transform to Schmidt basis
TA_schmidt_ij = np.zeros((r, r))
for ai in range(nA_dets):
    for ak in range(nA_dets):
        v = TA_det[ai, ak, i_a, j_a]
        if abs(v) < 1e-14: continue
        TA_schmidt_ij += U[ai, alpha] * U[ak, gamma] * v

TB_schmidt_kl = np.zeros((r, r))
for bj in range(nB_dets):
    for bl in range(nB_dets):
        v = TB_det[bj, bl, k_b, l_b]
        if abs(v) < 1e-14: continue
        TB_schmidt_kl += V[bj, beta] * V[bl, delta] * v

print(f"\nMethod B (explicit det expansion):")
print(f"TA_schmidt[{alpha},{gamma},{i_a},{j_a}] = {TA_schmidt_ij[0,0]:.12f}")
print(f"TB_schmidt[{beta},{delta},{k_b},{l_b}] = {TB_schmidt_kl[0,0]:.12f}")
# TA_schmidt_i_j is a scalar since α,γ are fixed

# Actually TA_schmidt_ij is r×r. We want element [0,0]? No...
# TA_schmidt_ij is the full r×r matrix for fixed (i,j). But for SPECIFIC α,γ,
# the contribution is the full sum: Σ U[ai,α] U[ak,γ] T_det[ai,ak,i,j]
# This is EXACTLY what TA_schmidt_ij is.

# Wait, TA_schmidt_i_j = U^T[:,α] @ T_det[:,:,i,j] @ U[:,γ]
# This is a SCALAR (not r×r).
# Let me fix: TA_schmidt_ij has shape (r,r), and TA[α,γ] = TA_schmidt_ij[α,γ]
# for THIS specific (i,j). So the value I want is the [α,γ] element.

# But I computed TA_schmidt_ij as sum_i sum_j U[ai,α] U[ak,γ] T_det[ai,ak,i,j]
# This sum over all ai,ak gives the [α,γ] element directly.
# So TA_schmidt_ij[0,0] is NOT right unless α=γ=0.

# Let me recompute properly:
ta_direct = 0.0
for ai in range(nA_dets):
    for ak in range(nA_dets):
        v = TA_det[ai, ak, i_a, j_a]
        if abs(v) < 1e-14: continue
        ta_direct += U[ai, alpha] * U[ak, gamma] * v

tb_direct = 0.0
for bj in range(nB_dets):
    for bl in range(nB_dets):
        v = TB_det[bj, bl, k_b, l_b]
        if abs(v) < 1e-14: continue
        tb_direct += V[bj, beta] * V[bl, delta] * v

print(f"\nCorrect Method B (sum over dets):")
print(f"TA_direct[{alpha},{gamma},{i_a},{j_a}] = {ta_direct:.12f}")
print(f"TB_direct[{beta},{delta},{k_b},{l_b}] = {tb_direct:.12f}")

print(f"\n=== Comparison ===")
print(f"Method A (TM): TA={ta:.12f}, TB={tb:.12f}, HAB={hab_tm:.12f}")
hab_direct = integral * ta_direct * tb_direct
print(f"Method B (det): TA={ta_direct:.12f}, TB={tb_direct:.12f}, HAB={hab_direct:.12f}")
print(f"diff TA: {abs(ta - ta_direct):.2e}")
print(f"diff TB: {abs(tb - tb_direct):.2e}")
print(f"diff HAB: {abs(hab_direct - hab_tm):.2e}")

# ── Method C: Sigma-vector (for the same single term, via full H) ──
# This is harder to isolate. Instead, check the FULL 2-body contribution
# for ALL (i,j) in A and (k,l) in B.

# Do a quick sanity check: compare TA from module vs direct for a few elements
print(f"\n=== Quick TA comparison ===")
TA_trans = trans_A.trans_1.get(n_A)
for a in range(r):
    for g in range(r):
        for i in range(n_occ):
            for j in range(n_occ):
                v_mod = TA_trans[a, g, i, j]
                # Direct:
                v_dir = 0.0
                for ai in range(nA_dets):
                    for ak in range(nA_dets):
                        vd = TA_det[ai, ak, i, j]
                        if abs(vd) > 1e-14:
                            v_dir += U[ai, a] * U[ak, g] * vd
                if abs(v_mod) > 0.01 or abs(v_dir) > 0.01:
                    diff_ij = abs(v_mod - v_dir)
                    if diff_ij > 1e-10:
                        print(f"  MISMATCH: TA[{a},{g},{i},{j}]: module={v_mod:.8f}, direct={v_dir:.8f}, diff={diff_ij:.2e}")

print("\n=== Done ===")
