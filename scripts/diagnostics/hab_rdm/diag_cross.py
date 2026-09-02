#!/usr/bin/env python3
"""
Targeted diagnostic: WHY is H_AB[0β, 1β] ≠ 0 when TA[0,1,...] = 0?

Enumerate every 4-index integral contribution to this specific off-diagonal
element of H_AB, tracking contributions by (p,q,r,s) index classification.
"""
import numpy as np
import sys
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

# ── Build reference ──
H_ref, basis_ref, decomps_ref = build_h_emb(
    schmidt, partition, q_idx, backend, h1eff, h2_4d,
    n_occ, n_act, verbose=False)
HA_ref = decomps_ref['HA']
HB_ref = decomps_ref['HB']
HAB_ref = decomps_ref['HAB']

# Block offsets
block_offsets = {}
off = 0
for na in sorted(schmidt.keys()):
    block_offsets[na] = off
    off += schmidt[na]['r'] ** 2

# ── Focus on n_A=3 block ──
n_A = 3
sd = schmidt[n_A]
r = sd['r']
blk = partition[n_A]
U = sd['U']
V = sd['V']
bo = block_offsets[n_A]

# Compute HA_schmidt, HB_schmidt
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

# ── Transition matrices ──
trans_A = compute_transition_matrices(partition, schmidt, n_occ, subspace='A', verbose=False)
trans_B = compute_transition_matrices(partition, schmidt, n_virt, subspace='B', verbose=False)
TA = trans_A.trans_1.get(n_A)

print(f"=== n_A=3 block, r={r} ===")

# Check HA_schmidt[0,1] and TA
print(f"\n--- HA_schmidt ---")
for i in range(r):
    row = [f"{HA_schmidt[i,j]:8.4f}" for j in range(r)]
    print(f"  {row}")

print(f"\n--- TA[α,γ,i,j] check ---")
for alpha in range(r):
    for gamma in range(r):
        tnorm = np.linalg.norm(TA[alpha, gamma])
        if tnorm > 1e-10:
            print(f"  TA[{alpha},{gamma}] norm = {tnorm:.6f}")
        elif alpha != gamma:
            print(f"  TA[{alpha},{gamma}] norm = {tnorm:.2e}  ← ZERO")

# ── For the specific pair (α=0,γ=1) with β=δ ──
# Enumerate ALL determinant contributions
alpha_strs = q_idx.alpha_strs
beta_strs = q_idx.beta_strs
n_as = len(alpha_strs)
n_bs = len(beta_strs)
alpha_to_idx = {int(s): i for i, s in enumerate(alpha_strs)}
beta_to_idx = {int(s): i for i, s in enumerate(beta_strs)}

# Expand bra and ket to determinant basis
alpha_bra = 0
beta_val = 0   # fix β

ci_bra = _expand_schmidt_product_to_ci_matrix(
    alpha_bra, beta_val, sd, blk, n_as, n_bs, n_occ,
    alpha_strs, beta_strs, alpha_to_idx, beta_to_idx)

alpha_ket = 1
ci_ket = _expand_schmidt_product_to_ci_matrix(
    alpha_ket, beta_val, sd, blk, n_as, n_bs, n_occ,
    alpha_strs, beta_strs, alpha_to_idx, beta_to_idx)

# Now enumerate every (p,q,r,s) integral contribution to ⟨bra|H|ket⟩
# H = Σ_{pr} h1[p,r] a†p ar + ½ Σ_{pqrs} (pq|rs) a†p a†q as ar

# We need for each (p,q,r,s): (pq|rs) × ⟨U[0]V[β]| a†p a†q as ar |U[1]V[β]⟩
# Expand: |U[α]⟩ = Σ_i U[i,α] |a_i⟩
# So: ⟨U[0]V[β]|H_AB|U[1]V[β]⟩ = Σ_{ij} U[i,0] U[j,1] ⟨a_i V[β]|H_AB|a_j V[β]⟩

# Expand V[β] in B-det basis
b_phase = sd['b_phase'] if 'b_phase' in sd else 1.0

# Actually, let's use a MUCH simpler approach:
# Build the full H in determinant basis for this block
# and expand Schmidt states to determinant vectors

# Get Schmidt basis CI vectors for each a,b pair
schmidt_cis = {}
for a in range(r):
    for b in range(r):
        ci = _expand_schmidt_product_to_ci_matrix(
            a, b, sd, blk, n_as, n_bs, n_occ,
            alpha_strs, beta_strs, alpha_to_idx, beta_to_idx)
        schmidt_cis[(a,b)] = ci

# Also compute sigma for each
schmidt_sigmas = {}
for (a,b), ci in schmidt_cis.items():
    schmidt_sigmas[(a,b)] = backend.sigma_full(ci)

# Now H_full[αβ, γδ] = sum_{det_idx} ci_bra[det] * sigma_ket[det]
for b1 in range(r):
    for b2 in range(r):
        bra_key = (0, b1)
        ket_key = (1, b2)
        if b1 != b2:
            continue  # we want β=δ
        ci_b = schmidt_cis[bra_key]
        sigma_k = schmidt_sigmas[ket_key]
        h_val = np.sum(ci_b * sigma_k)
        ha_val = HA_schmidt[1, 0] if b1 == b2 else 0.0
        hb_val = HB_schmidt[b2, b1] if 0 == 1 else 0.0
        hab_val = h_val - ha_val - hb_val
        print(f"\n--- H_full[{0},{b1},{1},{b2}] = {h_val:.12f} ---")
        print(f"  HA_schmidt[1,0] = {ha_val:.12f}")
        print(f"  H_AB = {hab_val:.12f}")
        print(f"  HAB_ref from build_h_emb = {HAB_ref[bo + 0*r + b1, bo + 1*r + b2]:.12f}")

# ── NOW: enumerate all (p,q,r,s) with A/B index classification ──
# For each determinant pair (det_a, det_a') that U connects,
# and each (det_b, det_b') that V connects,
# compute all 4-index contributions

# Use the determinant-basis transition approach
# |U[α]⟩ = Σ_i U[i,α] |a_i⟩,  |V[β]⟩ = Σ_j V[j,β] |b_j⟩
# H_AB = all terms where at least one index in A and one in B

# Actually, simplest approach: compute H_full in Schmidt basis,
# then subtract HA⊗I and I⊗HB, enumerate what's left

# Let's compute the H_full[0β, 1β] contribution directly via determinant expansion
print(f"\n=== Enumerating all (p,q,r,s) contributions to H_AB[0,{beta_val},1,{beta_val}] ===")

# The approach: for each determinant in the A product space expansion of bra and ket,
# compute all Slater-Condon matrix elements, classify by index pattern

# U[i,α] maps from A-det index i to Schmidt index α
# So |Ã_α⟩ = Σ_i U[i,α] |a_i⟩

# H_AB in Schmidt basis:
# H_AB[α,β, γ,β] = Σ_{i,j} U[i,α] U[j,γ] *
#   [⟨a_i,b_β|H|a_j,b_β⟩ - ⟨a_i|HA|a_j⟩*δ_{β,β} - ⟨b_β|HB|b_β⟩*δ_{i,j}]
#
# Wait, that's not quite right. Let me think more carefully.

# H_full[αβ, γδ] = Σ_{i,k} U[i,α] V[β_component] ...
# Actually the Schmidt basis is |Ã_α^(n)⟩ = Σ_i U[n][i,α] |a_i^(n)⟩
# |B̃_β^(n)⟩ = Σ_j V[n][j,β] |b_j^(n)⟩

# H_full[αβ, γδ] = Σ_{i,j,k,l} U[i,α] U[j,γ] V[k,β] V[l,δ] ⟨a_i,b_k|H|a_j,b_l⟩

# H_AB = the difference between this and HA⊗I + I⊗HB

# For β=δ:
# HA⊗I contribution: HA_schmidt[α,γ] × 1 (since β=δ, overlap=1)
# I⊗HB contribution: δ_{α,γ} × HB_schmidt[β,β]

# So H_AB[αβ, γβ] = H_full[αβ, γβ] - HA_schmidt[α,γ] - HB_schmidt[β,β]*δ_{α,γ}

# The question: what 4-index terms contribute to H_AB[0β, 1β]?

# Let me expand H_full[0β, 1β] in terms of determinants:
# H_full[0β, 1β] = Σ_{i,j,k,l} U[i,0]U[j,1] V[k,β]V[l,β] ⟨a_i,b_k|H|a_j,b_l⟩

# HA_schmidt[0,1] = Σ_{i,j} U[i,0]U[j,1] ⟨a_i|HA|a_j⟩
# Note: this is ⟨a_i|HA⊗I_B|a_j⟩, i.e., the part of H where only A-space acts

# So H_AB[0β, 1β] = Σ_{i,j,k,l} U[i,0]U[j,1] V[k,β]V[l,β]
#   [⟨a_i,b_k|H|a_j,b_l⟩ - ⟨a_i|HA|a_j⟩ δ_{k,l}]

# Now, ⟨a_i,b_k|H|a_j,b_l⟩ contains:
# 1. Pure A terms: ⟨a_i|HA|a_j⟩ δ_{k,l}  → cancelled by HA⊗I subtraction
# 2. Pure B terms: ⟨b_k|HB|b_l⟩ δ_{i,j}  → for α≠γ, δ_{i,j} not necessarily zero...
#    Wait, but I⊗HB is subtracted as HB_schmidt[β,β] δ_{α,γ}, which is ZERO for α≠γ
#    So pure B terms DON'T get subtracted for off-diagonal A!
#    This means: if V[k,β]V[l,β] connects different B-dets (k≠l), and i=j (same A-det),
#    there could be a pure B term contribution!

# Hmm actually, let me reconsider. H = HA + HB + H_AB, and:
# HA = HA ⊗ I_B  (only A-space operators)
# HB = I_A ⊗ HB  (only B-space operators)
# H_AB = cross terms with ops in both spaces

# In the Schmidt basis:
# H_full = HA_schmidt ⊗ I + I ⊗ HB_schmidt + H_AB

# So H_AB = H_full - HA_schmidt⊗I - I⊗HB_schmidt
# H_AB[0β, 1β] = H_full[0β, 1β] - HA_schmidt[0,1]*δ_{ββ} - 0  (since α≠γ, the HB term is 0)
# = H_full[0β, 1β] - HA_schmidt[0,1]

# H_full[0β, 1β] = Σ_{i,j,k,l} U[i,0]U[j,1] V[k,β]V[l,β] ⟨a_i,b_k|H|a_j,b_l⟩

# ⟨a_i,b_k|H|a_j,b_l⟩ breaks into:
# 1. A-only: ⟨a_i|HA|a_j⟩ δ_{k,l}
# 2. B-only: δ_{i,j} ⟨b_k|HB|b_l⟩
# 3. Cross: terms with operators in both A and B

# So:
# H_full[0β, 1β] = Σ_{i,j} U[i,0]U[j,1] ⟨a_i|HA|a_j⟩ (Σ_k V[k,β]²)
#                 + Σ_{i,k,l} U[i,0]U[i,1] V[k,β]V[l,β] ⟨b_k|HB|b_l⟩
#                 + cross terms

# Since Σ_k V[k,β]² = 1 (normalization):
# Term 1 = HA_schmidt[0,1]

# So H_AB[0β, 1β] = H_full - HA = ...
# Term 1 exactly cancels HA_schmidt[0,1] contribution
# Term 2 = Σ_i U[i,0]U[i,1] × Σ_{k,l} V[k,β]V[l,β] ⟨b_k|HB|b_l⟩
# Term 3 = cross terms (operators in both A and B)

# So H_AB[0β, 1β] involves:
# (a) B-space-only terms weighted by U[i,0]U[i,1] ≠ 0
# (b) Cross terms (A+B operators)

# If Term 3 = 0 (TA[0,1]=0), then H_AB[0β, 1β] = Term 2,
# which is the B-space-only contribution weighted by the non-orthogonal A-space states!

# THIS IS THE KEY INSIGHT! When U[i,0]U[i,1] ≠ 0, even if there are NO A-space operators
# changing the state, the B-space-only terms contribute to the off-diagonal!

# Let me verify this numerically.

# Compute Term 2:
term2 = 0.0
for i in range(U.shape[0]):
    ui01 = U[i, 0] * U[i, 1]
    if abs(ui01) < 1e-14:
        continue
    for k in range(V.shape[0]):
        for l in range(V.shape[0]):
            vkvl = V[k, beta_val] * V[l, beta_val]
            if abs(vkvl) < 1e-14:
                continue
            term2 += ui01 * vkvl * HB_det[k, l]

term2_schmidt = np.sum(U[:, 0] * U[:, 1]) * HB_schmidt[beta_val, beta_val]
# Wait, HB_schmidt = V^T HB_det V, so V[k,β] HB_det[k,l] V[l,β] = HB_schmidt[β,β]
# and Σ_i U[i,0]U[i,1] = ⟨Ã_0|Ã_1⟩ = dot product of Schmidt vectors = 0 (orthogonal!)
# That's why Term 2 vanishes! U columns are orthonormal!

# Hmm wait, then what gives H_AB[0β, 1β]?

# Actually U columns are orthonormal by construction (SVD). So Σ_i U[i,0]U[i,1] = 0.
# But that's not the same as Σ_i U[i,0]U[i,1] × Σ_{k,l} V[k,β]V[l,β] ⟨b_k|HB|b_l⟩
# Unless V[k,β]V[l,β] ⟨b_k|HB|b_l⟩ factors out... which it doesn't!

# Let me reconsider. The expansion:
# H_full[0β, 1β] = Σ_{i,j} U[i,0]V[k,β] H_{ik,jl} U[j,1]V[l,β]
# where H_{ik,jl} = ⟨a_i,b_k|H|a_j,b_l⟩

# HA_schmidt = U^T HA_det U
# HA_schmidt[0,1] = Σ_{i,j} U[i,0] U[j,1] HA_det[i,j]

# H_full - HA⊗I:
# = Σ_{i,j,k,l} U[i,0]U[j,1] V[k,β]V[l,β] H_{ik,jl}
#   - Σ_{i,j} U[i,0]U[j,1] HA_det[i,j]
# = Σ_{i,j,k,l} U[i,0]U[j,1] V[k,β]V[l,β] H_{ik,jl}
#   - Σ_{i,j,k,l} U[i,0]U[j,1] HA_det[i,j] δ_{k,l}

# Now H_{ik,jl} can be decomposed. Let's use the full second-quantized Hamiltonian:

# H = Σ_{pr} h1[p,r] a†_p a_r + ½ Σ_{pqrs} (pq|rs) a†_p a†_q a_s a_r

# In the determinant basis (A-det i,j; B-det k,l):
# ⟨a_i,b_k|H|a_j,b_l⟩ = ⟨a_i,b_k|HA+HB+H_AB|a_j,b_l⟩

# For the pure A term: a†_p a_r with p,r ∈ A:
# ⟨a_i,b_k| a†_p a_r |a_j,b_l⟩ = ⟨a_i|a†_p a_r|a_j⟩ × ⟨b_k|b_l⟩
# = ⟨a_i|a†_p a_r|a_j⟩ × δ_{k,l}

# So HA contribution in determinant basis: HA_det[i,j] × δ_{k,l}
# The subtraction in H_AB removes exactly this!

# Pure B: a†_p a_r with p,r ∈ B:
# ⟨a_i,b_k| a†_p a_r |a_j,b_l⟩ = ⟨a_i|a_j⟩ × ⟨b_k|a†_p a_r|b_l⟩
# = δ_{i,j} × HB_det[k,l]

# In Schmidt basis:
# Σ_{i,j,k,l} U[i,0]U[j,1] V[k,β]V[l,β] × δ_{i,j} HB_det[k,l]
# = Σ_i U[i,0]U[i,1] × Σ_{k,l} V[k,β]V[l,β] HB_det[k,l]
# = (U[:,0]·U[:,1]) × HB_schmidt[β,β]

# Since U[:,0]·U[:,1] = 0 (orthonormal Schmidt vectors), this is ZERO!

# Cross terms (2A+2B): a†_p a†_q a_s a_r with 2 indices in A, 2 in B
# These are the 4 index patterns: (A,A|B,B), (A,B|B,A), (B,B|A,A), (B,A|A,B)
# Each involves TA[0,1,i,j] × TB[β,β,k,l] × integrals

# Since TA[0,1,i,j] = 0 (as claimed by user), these are ZERO!

# So if both Term 2 (pure B with A overlap) and Term 3 (cross 2e) are zero,
# what gives H_AB[0β, 1β] ≠ 0?

# OPTION 1: The 1e cross terms (a†_p(A) a_r(B)) contribute to the SAME n_A block
# via intermediate states! But these should connect n_A and n_A±1 blocks...

# OPTION 2: Pair transfer terms (n_A → n_A±2) also connect different blocks...

# OPTION 3: The issue is that H_full[0β, 1β] is computed from the sigma-vector,
# which could include contributions where the full H connects to states OUTSIDE
# the n_A=3 block in the full CAS space, and these project back.

# Actually, let me reconsider. In embedded_hamiltonian.py, how is H_full computed?
# It computes sigma of each Schmidt state and projects onto all Schmidt states.
# The Schmidt states span a subspace of the full CAS space.
# If the Schmidt basis doesn't exactly diagonalize the n_A-conserving part,
# off-diagonal terms can appear.

# But wait, the Schmidt decomposition guarantees:
# C = Σ_n U^(n) Σ^(n) V^(n)T
# The Schmidt vectors are orthonormal WITHIN each n_A block.
# BUT across blocks, they may not be orthogonal.

# Actually, the Schmidt decomposition builds U and V as orthonormal sets,
# so U[:,0]·U[:,1] = 0 within each block. The 1e and pair transfer contributions
# connect DIFFERENT blocks, so they can't explain in-block off-diagonals.

# Hmm, let me think about this differently.

# Maybe I should just compute numerically: build HA_schmidt⊗I, I⊗HB_schmidt,
# and H_full for this block, and compare term by term.

# Or better: directly compute H_AB[0β, 1β] via determinant expansion
# and track contributions.

# Let me try a brute force approach: compute H_full in the Schmidt basis
# for this block by building the full Hamiltonian in the product determinant
# basis and projecting.

print(f"\n=== Brute-force analysis: where does H_AB[0β, 1β] come from? ===")

# Build H in the product basis of A-dets × B-dets for n_A=3
na_dets = len(blk['a_dets'])
nb_dets = len(blk['b_dets'])
print(f"  n_A=3: {na_dets} A-dets × {nb_dets} B-dets = {na_dets*nb_dets} product states")

# Expand each Schmidt state explicitly
# For indexing: state (a,b) = a-th A-Schmidt vector ⊗ b-th B-Schmidt vector
# |Ã_a, B̃_b⟩ = Σ_{i,j} U[i,a] V[j,b] |a_i⟩|b_j⟩_{n_A=3}

# This means the CI matrix for |Ã_a, B̃_b⟩ is a na_dets × nb_dets matrix
# where entry (i,j) = U[i,a] * V[j,b]

# Let's compute sigma for all r×r product Schmidt states
all_sigmas = {}
for a_dst in range(r):
    for b_dst in range(r):
        ci = _expand_schmidt_product_to_ci_matrix(
            a_dst, b_dst, sd, blk, n_as, n_bs, n_occ,
            alpha_strs, beta_strs, alpha_to_idx, beta_to_idx)
        all_sigmas[(a_dst, b_dst)] = backend.sigma_full(ci)

# Also compute all CI matrices
all_cis = {}
for a in range(r):
    for b in range(r):
        all_cis[(a, b)] = _expand_schmidt_product_to_ci_matrix(
            a, b, sd, blk, n_as, n_bs, n_occ,
            alpha_strs, beta_strs, alpha_to_idx, beta_to_idx)

# Compute the key element
a_bra, b_bra = 0, 0
a_ket, b_ket = 1, 0

ci_bra_mat = all_cis[(a_bra, b_bra)]
sigma_ket_mat = all_sigmas[(a_ket, b_ket)]

# H_full expansion in determinant-basis terms:
# H_full = Σ_{i,j} U[i,a_bra]V[k,b_bra] × ((H_ik)_determinant-basis) × U[j,a_ket]V[l,b_ket]

# The sigma-vector gives us the full H acting on the ket:
# sigma_ket_mat[p,q] = Σ_{r,s} H_{pq,rs} × ci_ket_mat[r,s]
# where p,q are (alpha_str, beta_str) indices

# So H_full[a_bra,b_bra, a_ket,b_ket] = Σ_{p,q} ci_bra_mat[p,q] * sigma_ket_mat[p,q]

h_full_01 = np.sum(ci_bra_mat * sigma_ket_mat)
ha_01 = HA_schmidt[1, 0]
hb_00 = HB_schmidt[0, 0]

hab_01 = h_full_01 - ha_01  # since hb contribution is 0 for α≠γ

print(f"\n  H_full[0,0,1,0] via sigma = {h_full_01:.12f}")
print(f"  HA_schmidt[1,0]           = {ha_01:.12f}")
print(f"  H_AB[0,0,1,0]             = {hab_01:.12f}")
print(f"  HAB_ref from decomps      = {HAB_ref[bo + 0*r + 0, bo + 1*r + 0]:.12f}")

# Now decompose contributions to H_full by expanding in A-det, B-det basis
# |Ã_a, B̃_b⟩ = Σ_{i,j} U[i,a]V[j,b] |a_i⟩|b_j⟩
# H|Ã_a', B̃_b'⟩ = Σ_{i',j'} U[i',a']V[j',b'] H |a_i'⟩|b_j'⟩

# So ⟨Ã_a,B̃_b|H|Ã_a',B̃_b'⟩ = Σ_{i,j,i',j'} U[i,a]U[i',a'] V[j,b]V[j',b'] ⟨a_i,b_j|H|a_i',b_j'⟩

# These ⟨a_i,b_j|H|a_i',b_j'⟩ elements can be classified by operator pattern.
# Let me enumerate contributions by 1e vs 2e and index classification.

# Actually, the easier approach: use the sigma framework.
# For each (i,j) pair, compute sigma of |a_i⟩|b_j⟩ and project.

# Let me do this: enumerate all pairs (i,j) → (i',j') and compute the
# Hamiltonian matrix element, classifying by type.

# Actually this is getting very complex. Let me take a step back and think.

# The user said: HA_schmidt[0,1] = -2.446, HAB_ref = +2.446
# This means H_full[0,0,1,0] = HA[0,1] + HAB[0,0,1,0] = -2.446 + 2.446 = 0
# The full embedding Hamiltonian has NO coupling between these two Schmidt states!

# But the user's analysis (from yesterday) showed that TA[0,1] = 0,
# meaning no 2A+2B cross term. So they're confused about where H_AB comes from.

# The key question: if TA[0,1] = 0, what contributes to H_AB[0,0,1,0]?

# Let me compute the FACTOID term decomposition:
# H_AB = Σ_{pr, p∈A,r∈B} h1[p,r] (a†_p ⊗ a_r + h.c.)
#      + ½ Σ_{pqrs, 2∈A,2∈B} (pq|rs) a†_p a†_q a_s a_r

# For the 2e terms with 2A+2B, in the Schmidt basis:
# ⟨Ã_0,B̃_0| a†_p a†_q a_s a_r |Ã_1,B̃_0⟩
# For (A,A|B,B) pattern (p,q∈A; r,s∈B):
#   ⟨Ã_0|a†_p a_q|Ã_1⟩ × ⟨B̃_0|a†_r a_s|B̃_0⟩ = TA[0,1,p,q] × TB[0,0,r,s]

# If TA[0,1,p,q] = 0 for all p,q, then this pattern gives 0.
# Similarly for other 2A+2B patterns.

# What about 1A+3B? That would need 3 operators in B space. For n_A=3,
# 1A+3B would change n_A by 1, putting us in n_A=2 or n_A=4 block.
# But the matrix element H_AB[0,0,1,0] is WITHIN the n_A=3 block!
# So 1A+3B can't contribute to this element.

# Similarly 3A+1B: changes n_A, can't be in the same block.

# 1e cross: a†_p(A) a_r(B) — changes n_A by ±1. Can't be same block.

# Pair transfer: a†_p a†_q(A) × a_r a_s(B) — changes n_A by ±2. Can't be same block.

# So ONLY 2A+2B terms can give same-block off-diagonals!
# If TA[0,1] = 0, then H_AB[0*b, 1*b] MUST be zero.

# BUT the sigma-vector reference says it's NOT zero!

# This means either:
# 1. TA[0,1] is NOT zero (the user's calculation was wrong)
# 2. OR the reference H_AB is wrong (subtraction bug)

# Let me check TA[0,1] directly and the H_AB reference value.

print(f"\n=== Checking TA[0,1] and H_AB reference ===")
TA_norm_01 = np.linalg.norm(TA[0, 1])
print(f"  ||TA[0,1]|| = {TA_norm_01:.10e}")

# Show the actual TA values
print(f"  TA[0,1] matrix:")
for i in range(min(n_occ, 4)):
    row = [f"{TA[0,1,i,j]:.6e}" for j in range(min(n_occ, 4))]
    print(f"    {row}")

# Now compute H_AB[0,0,1,0] using the 4-index sum
hab_decomposed = 0.0
from dm_svd_embedding.hab_rdm_contract import (
    _add_nconserved_complementary,
)

# Actually let me just compute the 2A+2B 4-pattern sum for this element
b_dst = 0
b_src = 0
TA01 = TA[0, 1]  # (n_occ, n_occ)
TB00 = TB[0, 0]  # (n_virt, n_virt)

hab_b = hab_c = hab_d = hab_e = 0.0
# (b) (A,A|B,B): TA[i,j] TB[k,l]
for i in range(n_occ):
    for j in range(n_occ):
        ta = TA01[i, j]
        if abs(ta) < 1e-14: continue
        for k in range(n_virt):
            for l in range(n_virt):
                tb = TB00[k, l]
                if abs(tb) < 1e-14: continue
                hab_b += h2_4d[i, j, k+n_occ, l+n_occ] * ta * tb
# (c) (A,B|B,A): -TA[i,l] TB[k,j]
for i in range(n_occ):
    for l in range(n_occ):
        ta = TA01[i, l]
        if abs(ta) < 1e-14: continue
        for j in range(n_virt):
            for k in range(n_virt):
                tb = TB00[k, j]
                if abs(tb) < 1e-14: continue
                hab_c -= h2_4d[i, j+n_occ, k+n_occ, l] * ta * tb
# (d) (B,B|A,A): TB[i,j] TA[k,l]
for i in range(n_virt):
    for j in range(n_virt):
        tb = TB00[i, j]
        if abs(tb) < 1e-14: continue
        for k in range(n_occ):
            for l in range(n_occ):
                ta = TA01[k, l]
                if abs(ta) < 1e-14: continue
                hab_d += h2_4d[i+n_occ, j+n_occ, k, l] * ta * tb
# (e) (B,A|A,B): -TB[i,l] TA[k,j]
for i in range(n_virt):
    for l in range(n_virt):
        tb = TB00[i, l]
        if abs(tb) < 1e-14: continue
        for j in range(n_occ):
            for k in range(n_occ):
                ta = TA01[k, j]
                if abs(ta) < 1e-14: continue
                hab_e -= h2_4d[i+n_occ, j, k, l+n_occ] * ta * tb

hab_2a2b = hab_b + hab_c + hab_d + hab_e
print(f"\n  2A+2B sum for H_AB[0,0,1,0]:")
print(f"    (b) (A,A|B,B): {hab_b:.12f}")
print(f"    (c) (A,B|B,A): {hab_c:.12f}")
print(f"    (d) (B,B|A,A): {hab_d:.12f}")
print(f"    (e) (B,A|A,B): {hab_e:.12f}")
print(f"    Total (no ½):   {hab_2a2b:.12f}")
print(f"    Total × ½:      {hab_2a2b*0.5:.12f}")
print(f"    H_AB ref:       {hab_01:.12f}")

# Also check: is there a factor of 2?
print(f"\n    Ratio (2a2b×½)/ref = {hab_2a2b*0.5/hab_01:.6f}" if abs(hab_01) > 1e-14 else "")
print(f"    Ratio (2a2b)/ref   = {hab_2a2b/hab_01:.6f}" if abs(hab_01) > 1e-14 else "")

# Key insight: are we missing 1e cross terms WITHIN the block?
# In DMRG H_AB construction, there are also 1e contributions:
# h1[p,r] where p∈A, r∈B (or vice versa)
# These give: ⟨Ã_0|a†_p|Ã_1⟩ × ⟨B̃_β|a_r|B̃_β⟩
# TA.create_1[0,1,p] × TB.annihilate_1[β,β,r]

# But wait: a†_p ACTS on the A space, changing n_A! For p∈A, a†_p on a state
# with n_A=3 gives n_A=4, which is a DIFFERENT block.
# So 1e cross connects blocks, not within-block!

# Unless there's somehow a JW phase factor or something...
# No, the Jordan-Wigner string for 1e cross gives (-1)^{n_A} sign,
# but it's still connecting n_A and n_A±1 blocks.

# Hmm, so if TA[0,1] = 0 AND same-block 2e terms don't contribute,
# the only remaining possibility is that the reference subtraction is wrong.

# Let me check the subtraction more carefully.
# H_AB_ref = decomps_ref['HAB'], which should be H_full - HA⊗I - I⊗HB

# For this element: HAB_ref[bo+0, bo+r] where bo = block offset
# H_full[bo+0, bo+r] = sigma projection
# HA⊗I[bo+0, bo+r] = HA_schmidt[0,1] (since beta same, overlap=1)
# I⊗HB = 0 (since alpha≠gamma)

# So HAB = H_full - HA_schmidt[0,1]

# What if HA_schmidt is stored as A-space only without B-space identity?
# Let me check.

print(f"\n=== Verification: H = HA⊗I + I⊗HB + H_AB ===")
# Build HA⊗I + I⊗HB explicitly
D = H_ref.shape[0]
HA_vs_I = np.zeros((D, D))
# This would be built in product basis: HA_schmidt[a,g] * delta(b,d)

# Check one specific element
print(f"  H_full[{bo},{bo+r}] = {H_ref[bo, bo+r]:.12f}")
print(f"  HA[{bo},{bo+r}]    = {HA_ref[bo, bo+r]:.12f}")
print(f"  HB[{bo},{bo+r}]    = {HB_ref[bo, bo+r]:.12f}")
print(f"  HAB[{bo},{bo+r}]   = {HAB_ref[bo, bo+r]:.12f}")
print(f"  HA+HB+HAB          = {HA_ref[bo,bo+r] + HB_ref[bo,bo+r] + HAB_ref[bo,bo+r]:.12f}")

# Also check the HA_schmidt directly
print(f"\n  HA_schmidt[0,1]    = {HA_schmidt[0,1]:.12f}")
print(f"  HA_schmidt[1,0]    = {HA_schmidt[1,0]:.12f}")
print(f"  -HA_schmidt[0,1]   = {-HA_schmidt[0,1]:.12f}")
print(f"  HAB_ref[0,0,1,0]   = {HAB_ref[bo+0, bo+r]:.12f}")
print(f"  Diff HAB+HA        = {HAB_ref[bo+0, bo+r] + HA_schmidt[0,1]:.12f}")

print("\n=== Done ===")
