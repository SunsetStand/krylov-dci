#!/usr/bin/env python3
"""
BRUTE FORCE: enumerate every (p,q,r,s) contribution to H_AB[0,0,1,0].
Track by operator type: (a) 1e pure A, (b) 1e pure B, (c) 1e cross,
(d-f) 2e cross types, (g) 2e 3-body, etc.

The GOAL: find exactly which terms give the -2.446 that we can't explain
via the 1-body RDM approach.
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

partition, full_dets = setup_partition(n_act, n_elec, n_occ, ms=0)
C_blocks = build_block_matrices(partition, ci_flat)
schmidt = compute_schmidt_decomposition(C_blocks, eps=1e-3)

n_A = 3
sd = schmidt[n_A]
r = sd['r']
blk = partition[n_A]
U = sd['U']
V = sd['V']

alpha_strs = q_idx.alpha_strs
beta_strs = q_idx.beta_strs
n_as = len(alpha_strs)
n_bs = len(beta_strs)
alpha_to_idx = {int(s): i for i, s in enumerate(alpha_strs)}
beta_to_idx = {int(s): i for i, s in enumerate(beta_strs)}

a_dets = blk['a_dets']
b_dets = blk['b_dets']
na_dets = len(a_dets)
nb_dets = len(b_dets)

# Helper: combine A-det and B-det to get full (alpha_str, beta_str)
def prod_to_full(i_a, k_b):
    a_alpha, a_beta = a_dets[i_a]
    b_alpha, b_beta = b_dets[k_b]
    return (a_alpha | (b_alpha << n_occ), a_beta | (b_beta << n_occ))

# Precompute the Slater-Condon matrix elements between ALL product states
# in the n_A=3 block
print(f"Building full H in product basis ({na_dets}*{nb_dets}={na_dets*nb_dets})...")
sys.path.insert(0, '/data/home/wangcx/krylov-dci')
from src.hamiltonian import Hamiltonian
ham = Hamiltonian(h1=h1eff, h2=h2_4d, E_nuc=0.0, E_HF=0.0)

# Expand Schmidt states to product basis
# |psi_alpha_beta> = sum_i sum_k U[i,alpha] V[k,beta] |a_i>|b_k>
psi = {}
for alpha in range(r):
    for beta_val in range(r):
        psi_ik = U[:, alpha][:, None] * V[:, beta_val][None, :]
        psi[(alpha, beta_val)] = psi_ik

# Compute sigma in product basis for |psi_{1,0}>
# sigma = H @ psi_prod
# H[ik, jl] = <a_i,b_k|H|a_j,b_l>
# sigma[i,k] = sum_{j,l} H[ik,jl] psi[j,l]

# Build H in product basis only for the needed rows
# Actually, let's just compute sigma via sigma-vector (already done in first diag)
ci_ket = _expand_schmidt_product_to_ci_matrix(
    1, 0, sd, blk, n_as, n_bs, n_occ,
    alpha_strs, beta_strs, alpha_to_idx, beta_to_idx)
sigma_ket = backend.sigma_full(ci_ket)
ci_bra = _expand_schmidt_product_to_ci_matrix(
    0, 0, sd, blk, n_as, n_bs, n_occ,
    alpha_strs, beta_strs, alpha_to_idx, beta_to_idx)

h_full_via_sigma = np.sum(ci_bra * sigma_ket)
print(f"H_full[0,0,1,0] via sigma = {h_full_via_sigma:.12f}")

# Now: enumerate every (A-det, B-det) pair contributing to H_full
# and classify each matrix element by operator type
print(f"\nEnumerating all (a_i,b_k) --> (a_j,b_l) contributions...")

# Classify contributions
from collections import defaultdict
contrib_by_type = defaultdict(float)

count = 0
for i in range(na_dets):
    ui0 = U[i, 0]
    if abs(ui0) < 1e-12: continue
    for k in range(nb_dets):
        vk0 = V[k, 0]
        if abs(vk0) < 1e-12: continue
        bra_weight = ui0 * vk0

        for j in range(na_dets):
            uj1 = U[j, 1]
            if abs(uj1) < 1e-12: continue
            for l in range(nb_dets):
                vl0 = V[l, 0]
                if abs(vl0) < 1e-12: continue
                ket_weight = uj1 * vl0

                total_weight = bra_weight * ket_weight
                if abs(total_weight) < 1e-14: continue

                # Compute <a_i,b_k|H|a_j,b_l>
                bra_full = prod_to_full(i, k)
                ket_full = prod_to_full(j, l)
                hij = ham.matrix_element(bra_full, ket_full)

                contrib_by_type['total_sc'] += total_weight * hij
                count += 1

print(f"  Total non-zero contributions: {count}")
print(f"  H_full via Slater-Condon = {contrib_by_type['total_sc']:.12f}")

# Now decompose further: classify by determinant excitation level
# between (a_i,b_k) and (a_j,b_l)
print(f"\n=== Classify by excitation between A-dets ===")

# For each (i,j) pair in A-space, check excitation level
for i in range(na_dets):
    ui0 = U[i, 0]
    if abs(ui0) < 1e-12: continue
    for j in range(na_dets):
        uj1 = U[j, 1]
        if abs(uj1) < 1e-12: continue

        # What's the excitation level between A-det i and A-det j?
        aA_i, aB_i = a_dets[i]
        aA_j, aB_j = a_dets[j]

        # Count bit differences
        xA = aA_i ^ aA_j
        xB = aB_i ^ aB_j
        exc_a = (xA.bit_count() + xB.bit_count()) // 2  # excitation level

        # Contribution from this (i,j) pair summed over all (k,l) B-det pairs
        contrib_ij = 0.0
        for k in range(nb_dets):
            vk0 = V[k, 0]
            if abs(vk0) < 1e-12: continue
            for l in range(nb_dets):
                vl0 = V[l, 0]
                if abs(vl0) < 1e-12: continue
                weight = ui0 * uj1 * vk0 * vl0
                if abs(weight) < 1e-14: continue

                bra_full = prod_to_full(i, k)
                ket_full = prod_to_full(j, l)
                hij = ham.matrix_element(bra_full, ket_full)
                contrib_ij += weight * hij

        if abs(contrib_ij) > 1e-10:
            key = f"A-exc={exc_a}"
            contrib_by_type[key] += contrib_ij

for key in sorted(contrib_by_type.keys()):
    if key.startswith('A-exc') or key == 'total_sc':
        print(f"  {key}: {contrib_by_type[key]:.12f}")

# Expected: for exc=0 (same A-det), the B-space must account for the coupling
# For exc>0 (different A-det), A-space operators contribute

# Now the KEY QUESTION: for A-exc=1 (1-body transition in A),
# we know TA[0,1]=0. So what's the contribution?
# If it's non-zero, it must come from B-space effective 1-body!

print(f"\n=== Decompose A-exc=1: pure B vs cross vs effective 1-body ===")

# For each (i,j,k,l) where A-exc(i,j)=1 (i.e., a_i and a_j differ by 1 spin-orbital),
# classify the Hamiltonian element by operator location

exc1_contrib = 0.0
exc1_by_subtype = defaultdict(float)

for i in range(na_dets):
    ui0 = U[i, 0]
    if abs(ui0) < 1e-12: continue
    aA_i, aB_i = a_dets[i]
    for j in range(na_dets):
        uj1 = U[j, 1]
        if abs(uj1) < 1e-12: continue
        aA_j, aB_j = a_dets[j]

        # Check excitation level
        xA = aA_i ^ aA_j
        xB = aB_i ^ aB_j
        exc = (xA.bit_count() + xB.bit_count()) // 2
        if exc != 1:
            continue

        for k in range(nb_dets):
            vk0 = V[k, 0]
            if abs(vk0) < 1e-12: continue
            for l in range(nb_dets):
                vl0 = V[l, 0]
                if abs(vl0) < 1e-12: continue
                weight = ui0 * uj1 * vk0 * vl0
                if abs(weight) < 1e-14: continue

                bra_full = prod_to_full(i, k)
                ket_full = prod_to_full(j, l)
                hij = ham.matrix_element(bra_full, ket_full)

                # Classify: is this from HA, HB, or H_AB?
                # If k==l: B-det is same, so B-space is just expectation
                #   hij = <a_i,b_k|HA|a_j,b_k> + <a_i,b_k|H_AB|a_j,b_k>
                #        + <a_i,b_k|HB|a_j,b_k> [0 if i!=j]
                # If k!=l: B-det differs, so both A and B operators are involved

                if k == l:
                    # Same B-det → A-space transition + 2e cross with B spectator
                    exc1_by_subtype['k=l (same B-det)'] += weight * hij
                else:
                    exc1_by_subtype['k!=l (B-det transition)'] += weight * hij

                exc1_contrib += weight * hij

print(f"  Total A-exc=1 contribution: {exc1_contrib:.12f}")
for subtype, val in sorted(exc1_by_subtype.items()):
    print(f"    {subtype}: {val:.12f}")

# For k=l case, decompose hij into HA_det + H_AB_effective
# HA_det[i,j] is known
print(f"\n=== k=l decomposition: pure HA vs H_AB effective ===")

ha_contrib = 0.0
hab_contrib = 0.0

h1_A, h2_A = _extract_subspace_integrals(h1eff, h2_4d, np.arange(n_occ))
HA_det_full = _build_subspace_hamiltonian(
    a_dets, h1_A, h2_A, n_occ,
    a_dets[0][0].bit_count(), a_dets[0][1].bit_count())

for i in range(na_dets):
    ui0 = U[i, 0]
    if abs(ui0) < 1e-12: continue
    for j in range(na_dets):
        uj1 = U[j, 1]
        if abs(uj1) < 1e-12: continue

        xA = a_dets[i][0] ^ a_dets[j][0]
        xB = a_dets[i][1] ^ a_dets[j][1]
        exc = (xA.bit_count() + xB.bit_count()) // 2
        if exc != 1:
            continue

        # HA_det contribution
        ha_ij = HA_det_full[i, j]
        v_norm_sq = 0.0
        for k in range(nb_dets):
            vk0 = V[k, 0]
            if abs(vk0) < 1e-12: continue
            v_norm_sq += vk0 * vk0

        ha_contrib += ui0 * uj1 * ha_ij * v_norm_sq

        # k=l H_AB contribution
        for k in range(nb_dets):
            vk0 = V[k, 0]
            if abs(vk0) < 1e-12: continue

            bra_full = prod_to_full(i, k)
            ket_full = prod_to_full(j, k)
            hij = ham.matrix_element(bra_full, ket_full)
            hab_contrib += ui0 * uj1 * vk0 * vk0 * (hij - ha_ij)

print(f"  HA_det contribution (projected): {ha_contrib:.12f}")
print(f"  H_AB effective (k=l, H-hA):     {hab_contrib:.12f}")
print(f"  Sum:                              {ha_contrib + hab_contrib:.12f}")
print(f"  HA_schmidt[0,1] from U^T HA U:   {np.dot(U[:,0], HA_det_full @ U[:,1]):.12f}")

# The key insight: hab_contrib should give us the -2.446!

print(f"\n=== B-space effective 1-body in A ===")
# For each orbital pair (p,q) in A space, compute the B-space 1-RDM contribution
# v_eff[p,q] = sum_{r,s in B} [2*(pq|rs) - (pr|sq)] * rho_B[r,s]
# where rho_B[r,s] = <B_0| a+_r a_s |B_0>

# 1-RDM of B-space for Schmidt state B_0
rho_B = np.zeros((n_virt, n_virt))
for k in range(nb_dets):
    vk0_sq = V[k, 0] * V[k, 0]
    if abs(vk0_sq) < 1e-14: continue
    bA, bB = b_dets[k]
    for p in range(n_virt):
        rho_B[p, p] += vk0_sq * ((bA >> p) & 1)
        rho_B[p, p] += vk0_sq * ((bB >> p) & 1)

# Off-diagonal: transitions between different B-dets
for k in range(nb_dets):
    vk0 = V[k, 0]
    if abs(vk0) < 1e-14: continue
    for l in range(nb_dets):
        if k == l: continue
        vl0 = V[l, 0]
        if abs(vl0) < 1e-14: continue

        bA_k, bB_k = b_dets[k]
        bA_l, bB_l = b_dets[l]

        # Find which spin-orbital differs
        # Alpha spin
        dA = bA_k ^ bA_l
        if dA.bit_count() == 2:  # single excitation in alpha
            # Find the orbitals
            occ_bits = bA_k & dA  # occupied in k, empty in l
            vir_bits = bA_l & dA  # occupied in l, empty in k
            if occ_bits.bit_count() == 1 and vir_bits.bit_count() == 1:
                p = occ_bits.bit_length() - 1
                q = vir_bits.bit_length() - 1
                # sign from JW: (-1)^(n_electrons between p and q)
                if p < q:
                    n_between = bin(bA_k & ((1 << q) - 1) & ~((1 << (p+1)) - 1)).count('1')
                else:
                    n_between = bin(bA_k & ((1 << p) - 1) & ~((1 << (q+1)) - 1)).count('1')
                sign = 1.0 if n_between % 2 == 0 else -1.0
                rho_B[p, q] += vk0 * vl0 * sign

        # Beta spin
        dB = bB_k ^ bB_l
        if dB.bit_count() == 2:
            occ_bits = bB_k & dB
            vir_bits = bB_l & dB
            if occ_bits.bit_count() == 1 and vir_bits.bit_count() == 1:
                p = occ_bits.bit_length() - 1
                q = vir_bits.bit_length() - 1
                if p < q:
                    n_between = bin(bB_k & ((1 << q) - 1) & ~((1 << (p+1)) - 1)).count('1')
                else:
                    n_between = bin(bB_k & ((1 << p) - 1) & ~((1 << (q+1)) - 1)).count('1')
                sign = 1.0 if n_between % 2 == 0 else -1.0
                rho_B[p, q] += vk0 * vl0 * sign

print(f"  rho_B:")
for i in range(n_virt):
    row = [f"{rho_B[i,j]:8.4f}" for j in range(n_virt)]
    print(f"    {row}")

# Effective 1-body in A from B-space 1-RDM
# Coulomb: J[p,q] = sum_{rs} (pq|rs) * rho_B[r,s]
# Exchange: K[p,q] = -sum_{rs} (pr|sq) * rho_B[r,s]  (but with antisymmetrized integral notation)

# In the chemist notation: v_eff[p,q] = sum_{r,s in B} [2*(pq|rs) - (pr|sq)] * rho_B[r,s]
# For coupled spin:
v_eff_A = np.zeros((n_occ, n_occ))
for p in range(n_occ):
    for q in range(n_occ):
        v_coul = 0.0
        v_exch = 0.0
        for r in range(n_virt):
            for s in range(n_virt):
                v_coul += h2_4d[p, q, r+n_occ, s+n_occ] * rho_B[r, s] * 2.0
                v_exch -= h2_4d[p, s+n_occ, r+n_occ, q] * rho_B[r, s]
        v_eff_A[p, q] = v_coul + v_exch

print(f"\n  Effective 1-body in A from B-space 1-RDM:")
print(f"    (Coulomb J + Exchange K, spin-summed)")
for i in range(n_occ):
    row = [f"{v_eff_A[i,j]:10.6f}" for j in range(n_occ)]
    print(f"    {row}")

# Now compute the contribution to H_full[0,0,1,0]
# v_eff_contribution = sum_{p,q} v_eff[p,q] * <A_0| a+_p a_q |A_1>
# = sum_{p,q} v_eff[p,q] * TA[0,1,p,q]

veff_contrib = np.sum(v_eff_A * TA[0, 1])  # where TA = trans_A.trans_1[n_A]
# TA[0,1] = 0! So this should be 0.

trans_A = compute_transition_matrices(partition, schmidt, n_occ, subspace='A', verbose=False)
TA_01 = trans_A.trans_1.get(n_A)[0, 1]
veff_contrib = np.sum(v_eff_A * TA_01)
print(f"\n  v_eff contribution via TA[0,1]: {veff_contrib:.12e}")
print(f"  (Expected ~0 since TA[0,1] ≈ 0)")

# BUT: this effective potential should be considered as MODIFYING the A-space
# 1-body integrals BEFORE computing HA_det. I.e.:
# h1_A_eff[p,q] = h1_A[p,q] + v_eff_A[p,q]
# Then HA_det_eff = build_from(h1_A_eff, h2_A)
# Then HA_schmidt_eff = U^T HA_det_eff U
# Then the difference HA_schmidt_eff - HA_schmidt = the mean-field correction

# Let's compute this!
print(f"\n=== Recompute HA_schmidt with B-mean-field correction ===")
h1_A_raw = h1eff[:n_occ, :n_occ].copy()
h1_A_mf = h1_A_raw + v_eff_A

from dm_svd_embedding.embedded_hamiltonian import _extract_subspace_integrals
h1_A_mf_full, _ = _extract_subspace_integrals(
    np.diag(np.diag(h1eff)) + np.diag(np.diag(h1eff)) * 0, h2_4d, np.arange(n_occ))
# Actually just replace the relevant part
h1_mf_fake = h1eff.copy()
h1_mf_fake[:n_occ, :n_occ] = h1_A_mf

# Build HA with effective 1-body
aA0, bA0 = a_dets[0]
HA_det_mf = _build_subspace_hamiltonian(
    a_dets, h1_A_mf, h2_A, n_occ, aA0.bit_count(), bA0.bit_count())
HA_schmidt_mf = U.T @ HA_det_mf @ U

print(f"  HA_schmidt[0,1] (bare h1):  {np.dot(U[:,0], HA_det_full @ U[:,1]):.12f}")
print(f"  HA_schmidt[0,1] (with mf):  {HA_schmidt_mf[0,1]:.12f}")
print(f"  Difference (mf - bare):      {HA_schmidt_mf[0,1] - np.dot(U[:,0], HA_det_full @ U[:,1]):.12f}")
print(f"  H_AB[0,0,1,0] from sigma:    {-np.dot(U[:,0], HA_det_full @ U[:,1]):.12f} (=-HA)")
print(f"  Expected if mf correction = -HA: {np.dot(U[:,0], HA_det_full @ U[:,1]) + HA_schmidt_mf[0,1]:.12f}")

# If the mean-field correction is exactly -HA, then:
# HA_mf[0,1] = 0, which would make H_full = 0 naturally!

# Check HA_schmidt_mf matrix
print(f"\n  HA_schmidt_mf:")
for i in range(r):
    row = [f"{HA_schmidt_mf[i,j]:10.4f}" for j in range(r)]
    print(f"    {row}")

print("\n=== Done ===")
