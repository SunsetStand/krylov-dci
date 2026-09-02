import numpy as np
import sys
sys.path.insert(0, '/data/home/wangcx/krylov-dci')

from dm_svd_embedding.occ_virt_partition import setup_partition, build_block_matrices
from dm_svd_embedding.density_matrix import compute_schmidt_decomposition
from dm_svd_embedding.embedded_hamiltonian import _setup_h2o_system, build_h_emb, build_hemb_via_rdm
from dm_svd_embedding.transition_rdm import compute_transition_matrices

(mol, mf, cas, q_idx, backend, h1eff, h2_4d, fcivec, ci_flat,
 n_act, n_elec, n_occ, na, nb, ecore, E_fci) = _setup_h2o_system()
n_virt = n_act - n_occ

partition, full_dets = setup_partition(n_act, n_elec, n_occ, ms=0)
C_blocks = build_block_matrices(partition, ci_flat)
schmidt = compute_schmidt_decomposition(C_blocks, eps=1e-3)

H_ref, basis_ref, _ = build_h_emb(schmidt, partition, q_idx, backend, h1eff, h2_4d, n_occ, n_act, verbose=False)
trans_A = compute_transition_matrices(partition, schmidt, n_occ, subspace='A', verbose=False)
trans_B = compute_transition_matrices(partition, schmidt, n_virt, subspace='B', verbose=False)
H_rdm, basis_rdm, decomps = build_hemb_via_rdm(schmidt, partition, h1eff, h2_4d, n_occ, n_act, trans_A, trans_B, verbose=False)

HA_ref = decomps['HA']
HB_ref = decomps['HB']
HAB_ref = H_ref - HA_ref - HB_ref
HAB_rdm = decomps['HAB']

# Block offsets
block_offsets = {}
offset = 0
for na in sorted(schmidt.keys()):
    rr = schmidt[na]['r']
    block_offsets[na] = offset
    offset += rr * rr

# Focus on n_A=3 block: find the element with largest ref-rdm difference
n_A = 3
sd = schmidt[n_A]
r = sd['r']
o = block_offsets[n_A]

# Print all diagonal elements for n_A=3 block
print(f"n_A=3 block (offset={o}, r={r}):")
print(f"{'flat':>5s} {'alpha':>5s} {'beta':>5s} {'ref':>10s} {'rdm':>10s} {'diff':>10s}")
for alpha in range(r):
    for beta in range(r):
        flat = o + alpha * r + beta
        ref_v = HAB_ref[flat, flat]
        rdm_v = HAB_rdm[flat, flat]
        diff = ref_v - rdm_v
        marker = " ***" if abs(diff) > 5 else ""
        print(f"{flat:5d} {alpha:5d} {beta:5d} {ref_v:10.4f} {rdm_v:10.4f} {diff:10.4f}{marker}")

# The large diff is at alpha=0/1 with beta=1/0,3. 
# Let me check what HA and HB contribute to this element
print(f"\nDetailed for flat=2 (alpha=0,beta=1):")
flat = o + 0*r + 1
print(f"  H_ref[{flat},{flat}] = {H_ref[flat,flat]:.4f}")
print(f"  HA_ref[{flat},{flat}] = {HA_ref[flat,flat]:.4f}")
print(f"  HB_ref[{flat},{flat}] = {HB_ref[flat,flat]:.4f}")
print(f"  HA+HB = {HA_ref[flat,flat]+HB_ref[flat,flat]:.4f}")
print(f"  HAB_ref = {HAB_ref[flat,flat]:.4f}")

# Now decompose HA_ref: HA_emb[gamma,beta; alpha,beta] = HA_schmidt[gamma,alpha]
# For diagonal: HA_emb[flat(alpha,beta), flat(alpha,beta)] = HA_schmidt[alpha,alpha]
# Check if this is correct
HA_schmidt = sd['U'].T @ _build_subspace_hamiltonian_wrapper(
    partition[n_A]['a_dets'], h1eff[:n_occ,:n_occ], h2_4d[:n_occ,:n_occ,:n_occ,:n_occ],
    n_occ, int(partition[n_A]['a_dets'][0][0].bit_count()),
    int(partition[n_A]['a_dets'][0][1].bit_count()))
HB_schmidt = sd['V'].T @ _build_subspace_hamiltonian_wrapper(
    partition[n_A]['b_dets'], h1eff[n_occ:,n_occ:], h2_4d[n_occ:,n_occ:,n_occ:,n_occ:],
    n_virt, int(partition[n_A]['b_dets'][0][0].bit_count()),
    int(partition[n_A]['b_dets'][0][1].bit_count()))

# Check: for alpha=0, HA_emb[flat(0,1),flat(0,1)] should be HA_schmidt[0,0]
idx01 = 0*r + 1  # within-block index
ha_from_ref = HA_ref[o+idx01, o+idx01]
ha_from_schmidt = HA_schmidt[0, 0]
print(f"\n  HA_emb[{flat},{flat}] from ref = {ha_from_ref:.4f}")
print(f"  HA_schmidt[0,0] = {ha_from_schmidt:.4f}")
print(f"  Match: {abs(ha_from_ref - ha_from_schmidt) < 1e-6}")

hb_from_ref = HB_ref[o+idx01, o+idx01]
hb_from_schmidt = HB_schmidt[1, 1]
print(f"  HB_emb[{flat},{flat}] from ref = {hb_from_ref:.4f}")
print(f"  HB_schmidt[1,1] = {hb_from_schmidt:.4f}")
print(f"  Match: {abs(hb_from_ref - hb_from_schmidt) < 1e-6}")

# Now: if HA and HB are correct, then HAB_ref = H_ref - HA - HB should be the cross term
# Let's compute HAB cross from scratch:
# H_cross = <psi| H - H_A⊗I - I⊗H_B |psi>
# where |psi> = U[:,0] ⊗ V[:,1]
U = sd['U']
V = sd['V']
a_dets = partition[n_A]['a_dets']
b_dets = partition[n_A]['b_dets']
n_ad = len(a_dets)
n_bd = len(b_dets)

# Build psi_A = U[:,0], psi_B = V[:,1]
# H_AB_cross = <psi_A⊗psi_B| H_full - H_A⊗I - I⊗H_B |psi_A⊗psi_B>
#           = <psi_A⊗psi_B| H_full |psi_A⊗psi_B> - <psi_A|H_A|psi_A> - <psi_B|H_B|psi_B>
#           = <psi_A⊗psi_B| H_full |psi_A⊗psi_B> - HA_schmidt[0,0] - HB_schmidt[1,1]

# We need <psi_A⊗psi_B| H_full |psi_A⊗psi_B> in the determinant basis.
# This is: Σ_{i,j,k,l} U[i,0]V[j,1] U[k,0]V[l,1] <a_i,b_j| H |a_k,b_l>

# For this we need the FULL Hamiltonian in the A⊗B determinant basis for n_A=3 block only.
# We can compute this using the sigma-vector method explicitly.

# Actually, the sigma-vector method already computed this:
# H_ref[flat, flat] = <psi_A⊗psi_B| H_full |psi_A⊗psi_B>
# where psi_A⊗psi_B is the Schmidt product state.

# So: HAB_cross = H_ref[flat,flat] - HA_schmidt[0,0] - HB_schmidt[1,1]
h_cross_direct = H_ref[flat, flat] - ha_from_schmidt - hb_from_schmidt
print(f"\n  H_cross (direct) = {h_cross_direct:.4f}")
print(f"  HAB_ref = {HAB_ref[flat,flat]:.4f}")
print(f"  HAB_rdm = {HAB_rdm[flat,flat]:.4f}")

# The difference between direct cross and HAB_ref should be zero if HA, HB are
# correctly computed and subtracted
print(f"\n  Cross_diff (direct - HAB_ref) = {h_cross_direct - HAB_ref[flat,flat]:.4f}")
print(f"  RDM_diff (HAB_ref - HAB_rdm) = {HAB_ref[flat,flat] - HAB_rdm[flat,flat]:.4f}")
