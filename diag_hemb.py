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

# Reference: sigma-vector
H_ref, basis_ref, _ = build_h_emb(
    schmidt, partition, q_idx, backend, h1eff, h2_4d,
    n_occ, n_act, verbose=False)
D = H_ref.shape[0]

# RDM
trans_A = compute_transition_matrices(partition, schmidt, n_occ, subspace='A', verbose=False)
trans_B = compute_transition_matrices(partition, schmidt, n_virt, subspace='B', verbose=False)
H_rdm, basis_rdm, decomps = build_hemb_via_rdm(
    schmidt, partition, h1eff, h2_4d,
    n_occ, n_act, trans_A, trans_B, verbose=False)

# Extract H_AB from reference
HA_ref = decomps['HA']
HB_ref = decomps['HB']
HAB_ref = H_ref - HA_ref - HB_ref

# Compare H_AB norms
print(f"||H_AB_ref|| = {np.linalg.norm(HAB_ref):.4f}")
print(f"||H_AB_rdm|| = {np.linalg.norm(decomps['HAB']):.4f}")
print(f"||H_AB_diff|| = {np.linalg.norm(HAB_ref - decomps['HAB']):.4f}")

# Look at a few elements
print("\nH_AB[0:4, 0:4] (reference):")
print(HAB_ref[:4, :4])
print("\nH_AB[0:4, 0:4] (RDM):")
print(decomps['HAB'][:4, :4])

# Are they correlated?
ref_flat = HAB_ref.ravel()
rdm_flat = decomps['HAB'].ravel()
corr = np.corrcoef(ref_flat, rdm_flat)[0,1]
print(f"\nCorrelation(H_AB_ref, H_AB_rdm) = {corr:.4f}")

# Is there a simple scaling relationship?
mask = np.abs(ref_flat) > 1e-10
ratio = rdm_flat[mask] / ref_flat[mask]
print(f"Ratio stats: mean={np.mean(ratio):.4f}, std={np.std(ratio):.4f}, "
      f"min={np.min(ratio):.4f}, max={np.max(ratio):.4f}")

# Also check: if we flip sign of RDM H_AB, does it improve?
H_rdm_flipped = HA_ref + HB_ref - decomps['HAB']
ev_ref = np.sort(np.linalg.eigvalsh(H_ref))
ev_flip = np.sort(np.linalg.eigvalsh(H_rdm_flipped))
print(f"\nWith sign-flipped H_AB:")
print(f"  E[0]: ref={ev_ref[0]:.6f}, flip={ev_flip[0]:.6f}, diff={abs(ev_ref[0]-ev_flip[0]):.4e}")
