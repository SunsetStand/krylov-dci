"""Brute-force H_AB in determinant basis for n_A=3 block, compare with RDM."""
import numpy as np
import sys
sys.path.insert(0, '/data/home/wangcx/krylov-dci')

from dm_svd_embedding.occ_virt_partition import setup_partition, build_block_matrices
from dm_svd_embedding.density_matrix import compute_schmidt_decomposition
from dm_svd_embedding.embedded_hamiltonian import _setup_h2o_system, build_h_emb, build_hemb_via_rdm
from dm_svd_embedding.transition_rdm import compute_transition_matrices
from dm_svd_embedding.embedded_hamiltonian import _build_subspace_hamiltonian, _extract_subspace_integrals

(mol, mf, cas, q_idx, backend, h1eff, h2_4d, fcivec, ci_flat,
 n_act, n_elec, n_occ, na, nb, ecore, E_fci) = _setup_h2o_system()
n_virt = n_act - n_occ

partition, full_dets = setup_partition(n_act, n_elec, n_occ, ms=0)
C_blocks = build_block_matrices(partition, ci_flat)
schmidt = compute_schmidt_decomposition(C_blocks, eps=1e-3)

trans_A = compute_transition_matrices(partition, schmidt, n_occ, subspace='A', verbose=False)
trans_B = compute_transition_matrices(partition, schmidt, n_virt, subspace='B', verbose=False)
H_rdm, basis_rdm, decomps = build_hemb_via_rdm(schmidt, partition, h1eff, h2_4d, n_occ, n_act, trans_A, trans_B, verbose=False)

# Focus on n_A=3 block
n_A = 3
sd = schmidt[n_A]
r = sd['r']
blk = partition[n_A]
a_dets = blk['a_dets']
b_dets = blk['b_dets']
n_ad = len(a_dets)
n_bd = len(b_dets)
print(f"n_A={n_A}: r={r}, dim_A={n_ad}, dim_B={n_bd}")

U = sd['U']  # (dim_A, r)
V = sd['V']  # (dim_B, r)

# Build H_A_det and H_B_det
A_orb = np.arange(n_occ)
B_orb = np.arange(n_occ, n_act)
h1_A, h2_A = _extract_subspace_integrals(h1eff, h2_4d, A_orb)
h1_B, h2_B = _extract_subspace_integrals(h1eff, h2_4d, B_orb)

nA_a = int(a_dets[0][0].bit_count()) if n_ad > 0 else 0
nA_b = int(a_dets[0][1].bit_count()) if n_ad > 0 else 0
nB_a = int(b_dets[0][0].bit_count()) if n_bd > 0 else 0
nB_b = int(b_dets[0][1].bit_count()) if n_bd > 0 else 0

HA_det = _build_subspace_hamiltonian(a_dets, h1_A, h2_A, n_occ, nA_a, nA_b) if n_ad > 0 else np.zeros((0,0))
HB_det = _build_subspace_hamiltonian(b_dets, h1_B, h2_B, n_virt, nB_a, nB_b) if n_bd > 0 else np.zeros((0,0))

# Build FULL Hamiltonian in A⊗B determinant basis
# H_full[i*j_dim, k*l_dim] = HA_det[i,k]*δ[j,l] + HB_det[j,l]*δ[i,k] + H_AB_det
H_full = np.kron(HA_det, np.eye(n_bd)) + np.kron(np.eye(n_ad), HB_det)
# Add A-B cross terms
for i in range(n_ad):
    for j in range(n_bd):
        a_str, b_str_a = a_dets[i]
        b_str, b_str_b = b_dets[j]
        # A-B 2e contribution
        _add_hab_det(H_full, i, j, n_ad, n_bd, a_dets, b_dets, h2_4d, n_occ, n_act)
        # A-B 1e contribution already in HA, HB via h1eff

# Now H_full_ab = H_full - kron(HA, I) - kron(I, HB) 
# This is PURELY the cross-space contribution in det basis
H_ab_det = H_full - np.kron(HA_det, np.eye(n_bd)) - np.kron(np.eye(n_ad), HB_det)

# Transform to Schmidt basis
# U⊗V is (n_ad*n_bd, r*r)
UV = np.zeros((n_ad * n_bd, r * r))
for a in range(r):
    for b in range(r):
        col = a * r + b
        for i in range(n_ad):
            for j in range(n_bd):
                UV[i * n_bd + j, col] = U[i, a] * V[j, b]

H_ab_schmidt = UV.T @ H_ab_det @ UV

# Compare with RDM H_AB for n_A=3 block
# Block offset
bo = 1  # n_A=3 offset
H_ab_rdm_block = decomps['HAB'][bo:bo+r*r, bo:bo+r*r]

print(f"\n||H_AB_det (schmidt)|| = {np.linalg.norm(H_ab_schmidt):.4f}")
print(f"||H_AB_rdm (block)|| = {np.linalg.norm(H_ab_rdm_block):.4f}")
print(f"||diff|| = {np.linalg.norm(H_ab_schmidt - H_ab_rdm_block):.4f}")

print(f"\nDiagonal comparison:")
for a in range(r):
    for b in range(r):
        flat = a * r + b
        det_v = H_ab_schmidt[flat, flat]
        rdm_v = H_ab_rdm_block[flat, flat]
        print(f"  (a={a},b={b}): det={det_v:.4f}, rdm={rdm_v:.4f}, diff={det_v-rdm_v:.4f}")

# Now: what's in H_ab_det that's NOT in RDM?
# Compute: H_ab_det - UV @ H_ab_rdm_block @ UV.T
H_ab_det_rdm = UV @ H_ab_rdm_block @ UV.T
H_diff_det = H_ab_det - H_ab_det_rdm
print(f"\n||H_diff_det|| = {np.linalg.norm(H_diff_det):.4f}")

# What are the dominant contributions in H_diff_det?
# Check the 2e part of H_ab_det
# A-B interaction in det basis is built from the 2e integrals
# Let me compute term(b) directly in det basis for comparison
def _add_hab_det(H_full, i_A, j_B, n_ad, n_bd, a_dets, b_dets, h2, n_occ, n_act):
    """Add A-B 2e cross terms to H_full in determinant basis (n_A-conserved only)."""
    from src.hamiltonian import Hamiltonian
    
    # For each pair of determinants, compute cross-space matrix elements
    # This is a simplified version - just the n_A-conserved 2e part
    pass  # Placeholder - the actual implementation is complex

# Actually, H_full already includes ALL terms (HA + HB + HAB).
# H_ab_det = H_full - HA⊗I - I⊗HB gives the cross contribution in det basis.
# We already computed this above.

# Let me instead verify that H_full is correct by diagonalizing and comparing with FCI
# (We know FCI = cas.e_tot)
ev_full = np.sort(np.linalg.eigvalsh(H_full))[:5]
ev_a = np.sort(np.linalg.eigvalsh(HA_det))[:min(5,n_ad)]
ev_b = np.sort(np.linalg.eigvalsh(HB_det))[:min(5,n_bd)]
print(f"\nH_full eigs: {ev_full}")

# H_full should be a sub-block of the full CAS Hamiltonian
# It's not the full FCI - it's only the n_A=3 block

# Let me just check: is UV^T @ H_full @ UV the same as H_ref block?
# H_ref for n_A=3 should be from the sigma-vector
H_ref_block = build_h_emb(schmidt, partition, q_idx, backend, h1eff, h2_4d, n_occ, n_act, verbose=False)[0][bo:bo+r*r, bo:bo+r*r]
H_full_schmidt = UV.T @ H_full @ UV
print(f"||H_ref_block - H_full_schmidt|| = {np.linalg.norm(H_ref_block - H_full_schmidt):.6f}")
print(f"H_ref_block ~ H_full_schmidt: {np.linalg.norm(H_ref_block - H_full_schmidt) < 1e-6}")
