"""Trace specific H_AB diagonal contributions for n_A=3, alpha=0, beta=1."""
import numpy as np
import sys
sys.path.insert(0, '/data/home/wangcx/krylov-dci')

from dm_svd_embedding.occ_virt_partition import setup_partition, build_block_matrices
from dm_svd_embedding.density_matrix import compute_schmidt_decomposition
from dm_svd_embedding.embedded_hamiltonian import _setup_h2o_system, build_h_emb
from dm_svd_embedding.transition_rdm import compute_transition_matrices

(mol, mf, cas, q_idx, backend, h1eff, h2_4d, fcivec, ci_flat,
 n_act, n_elec, n_occ, na, nb, ecore, E_fci) = _setup_h2o_system()
n_virt = n_act - n_occ

partition, full_dets = setup_partition(n_act, n_elec, n_occ, ms=0)
C_blocks = build_block_matrices(partition, ci_flat)
schmidt = compute_schmidt_decomposition(C_blocks, eps=1e-3)

trans_A = compute_transition_matrices(partition, schmidt, n_occ, subspace='A', verbose=False)
trans_B = compute_transition_matrices(partition, schmidt, n_virt, subspace='B', verbose=False)

# Reference H_AB
H_ref, basis_ref, _ = build_h_emb(schmidt, partition, q_idx, backend, h1eff, h2_4d, n_occ, n_act, verbose=False)

# Compute H_A and H_B same as Path C
from dm_svd_embedding.embedded_hamiltonian import _build_subspace_hamiltonian, _extract_subspace_integrals

A_orb = np.arange(n_occ); B_orb = np.arange(n_occ, n_act)
h1_A, h2_A = _extract_subspace_integrals(h1eff, h2_4d, A_orb)
h1_B, h2_B = _extract_subspace_integrals(h1eff, h2_4d, B_orb)

# Block offsets
block_offsets = {}
offset = 0
for na in sorted(schmidt.keys()):
    rr = schmidt[na]['r']
    block_offsets[na] = offset
    offset += rr * rr

n_A = 3; sd = schmidt[n_A]; r = sd['r']
blk = partition[n_A]; a_dets = blk['a_dets']; b_dets = blk['b_dets']
U = sd['U']; V = sd['V']; n_ad = len(a_dets); n_bd = len(b_dets)

# H_A_det and H_B_det
nA_a = int(a_dets[0][0].bit_count())
nA_b = int(a_dets[0][1].bit_count())
nB_a = int(b_dets[0][0].bit_count())
nB_b = int(b_dets[0][1].bit_count())
HA_det = _build_subspace_hamiltonian(a_dets, h1_A, h2_A, n_occ, nA_a, nA_b)
HB_det = _build_subspace_hamiltonian(b_dets, h1_B, h2_B, n_virt, nB_a, nB_b)

# Now: build H_full in det basis using src.hamiltonian
sys.path.insert(0, '/data/home/wangcx/krylov-dci')
from src.hamiltonian import Hamiltonian

# Build combined H_full including cross terms
# For each (i,j) state in A⊗B:
# H_full[(i,j), (k,l)] = HA_det[i,k]*δ[j,l] + HB_det[j,l]*δ[i,k] + H_AB_cross
H_full = np.kron(HA_det, np.eye(n_bd)) + np.kron(np.eye(n_ad), HB_det)

# Now add cross terms: iterate over all A,B operator products
# This is expensive but exact
ham = Hamiltonian(h1=h1eff, h2=h2_4d, E_nuc=0.0, E_HF=0.0)

for i_idx in range(n_ad):
    aA_i, bA_i = a_dets[i_idx]
    for j_idx in range(n_bd):
        aB_j, bB_j = b_dets[j_idx]
        # Build full determinant string
        a_full_i = aA_i | (aB_j << n_occ)
        b_full_i = bA_i | (bB_j << n_occ)
        i_flat = i_idx * n_bd + j_idx
        
        # Diagonal
        H_full[i_flat, i_flat] += ham.diagonal_element(int(a_full_i), int(b_full_i))
        
        # Subtract already-counted HA and HB contributions
        # HA_det contribution
        for k_idx in range(n_ad):
            aA_k, bA_k = a_dets[k_idx]
            a_full_k = aA_k | (aB_j << n_occ)
            b_full_k = bA_k | (bB_j << n_occ)
            k_flat = k_idx * n_bd + j_idx
            # Only keep the CROSS part (subtract HA contribution)
            H_full[i_flat, k_flat] -= HA_det[i_idx, k_idx]
        
        # HB_det contribution  
        for l_idx in range(n_bd):
            aB_l, bB_l = b_dets[l_idx]
            l_flat = i_idx * n_bd + l_idx
            H_full[i_flat, l_flat] -= HB_det[j_idx, l_idx]

        # Add FULL Hamiltonian off-diagonal elements
        for k_idx in range(n_ad):
            aA_k, bA_k = a_dets[k_idx]
            for l_idx in range(n_bd):
                aB_l, bB_l = b_dets[l_idx]
                a_full_k = aA_k | (aB_l << n_occ)
                b_full_k = bA_k | (bB_l << n_occ)
                k_flat = k_idx * n_bd + l_idx
                if i_flat < k_flat:
                    hij = ham.matrix_element(
                        (int(a_full_i), int(b_full_i)),
                        (int(a_full_k), int(b_full_k)))
                    if abs(hij) > 1e-14:
                        H_full[i_flat, k_flat] += hij
                        H_full[k_flat, i_flat] += hij

# Now H_AB_det = H_full - kron(HA, I) - kron(I, HB)
# But we already subtracted HA and HB contributions
# So H_full should be exactly the cross-space part
# Wait no, we started with kron(HA, I) + kron(I, HB), then added H_full diag and off-diag,
# and subtracted HA and HB from the diag elements.
# This is messy. Let me just compute H_full properly.

# Actually, let me recompute more cleanly:
H_full2 = np.zeros((n_ad * n_bd, n_ad * n_bd))

for i_idx in range(n_ad):
    aA_i, bA_i = a_dets[i_idx]
    for j_idx in range(n_bd):
        aB_j, bB_j = b_dets[j_idx]
        a_full_i = aA_i | (aB_j << n_occ)
        b_full_i = bA_i | (bB_j << n_occ)
        i_flat = i_idx * n_bd + j_idx
        
        # Diagonal
        H_full2[i_flat, i_flat] = ham.diagonal_element(int(a_full_i), int(b_full_i))
        
        # Off-diagonal (upper triangle)
        for k_idx in range(i_idx, n_ad):
            for l_idx in range(j_idx+1 if k_idx==i_idx else 0, n_bd):
                aA_k, bA_k = a_dets[k_idx]
                aB_l, bB_l = b_dets[l_idx]
                a_full_k = aA_k | (aB_l << n_occ)
                b_full_k = bA_k | (bB_l << n_occ)
                k_flat = k_idx * n_bd + l_idx
                hij = ham.matrix_element(
                    (int(a_full_i), int(b_full_i)),
                    (int(a_full_k), int(b_full_k)))
                if abs(hij) > 1e-14:
                    H_full2[i_flat, k_flat] = hij
                    H_full2[k_flat, i_flat] = hij

# H_A+B = kron(HA, I) + kron(I, HB)
HA_plus_HB = np.kron(HA_det, np.eye(n_bd)) + np.kron(np.eye(n_ad), HB_det)

# H_AB_det = H_full2 - HA_plus_HB (cross-space only)
H_ab_det = H_full2 - HA_plus_HB

# Transform to Schmidt basis
UV = np.zeros((n_ad * n_bd, r * r))
for a in range(r):
    for b in range(r):
        col = a * r + b
        for i in range(n_ad):
            for j in range(n_bd):
                UV[i * n_bd + j, col] = U[i, a] * V[j, b]

H_ab_schmidt = UV.T @ H_ab_det @ UV

# Compare with RDM
bo = block_offsets[n_A]
H_ab_ref_block = H_ref[bo:bo+r*r, bo:bo+r*r]
# H_ab_ref = H_ref - HA_pathC - HB_pathC
# But we need HAB_ref block
HA_pathC_block = np.zeros((r*r, r*r))
HB_pathC_block = np.zeros((r*r, r*r))
HA_schmidt = U.T @ HA_det @ U
HB_schmidt = V.T @ HB_det @ V
for alpha in range(r):
    for beta in range(r):
        k = alpha * r + beta
        for gamma in range(r):
            HA_pathC_block[gamma*r+beta, k] = HA_schmidt[gamma, alpha]
        for delta in range(r):
            HB_pathC_block[alpha*r+delta, k] = HB_schmidt[delta, beta]

H_ab_ref = H_ab_ref_block - HA_pathC_block - HB_pathC_block

print(f"||H_ab_schmidt (brute force)|| = {np.linalg.norm(H_ab_schmidt):.4f}")
print(f"||H_ab_ref (sigma-vector)|| = {np.linalg.norm(H_ab_ref):.4f}")
print(f"||diff (brute vs ref)|| = {np.linalg.norm(H_ab_schmidt - H_ab_ref):.6f}")
print(f"Match: {np.linalg.norm(H_ab_schmidt - H_ab_ref) < 1e-6}")

print(f"\nDiagonal comparison:")
for a in range(r):
    for b in range(r):
        flat = a * r + b
        print(f"  (a={a},b={b}): brute={H_ab_schmidt[flat,flat]:.4f}, ref={H_ab_ref[flat,flat]:.4f}")
