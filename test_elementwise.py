import sys, os, numpy as np
sys.path.insert(0, '/data/home/wangcx/krylov-dci')

from dm_svd_embedding.embedded_hamiltonian import _setup_h2o_system
from dm_svd_embedding.occ_virt_partition import setup_partition, build_block_matrices
from dm_svd_embedding.density_matrix import compute_schmidt_decomposition
from dm_svd_embedding.transition_rdm import compute_transition_matrices
from dm_svd_embedding.hab_rdm_contract import build_hab_rdm

(mol, mf, cas, q_idx, backend, h1eff, h2_4d, fcivec, ci_flat,
 n_act, n_elec, n_occ, na, nb, ecore, E_fci) = _setup_h2o_system()

n_virt = n_act - n_occ
partition, full_dets = setup_partition(n_act, n_elec, n_occ, ms=0)
C_blocks = build_block_matrices(partition, ci_flat)
schmidt = compute_schmidt_decomposition(C_blocks, eps=1e-3)

trans_A = compute_transition_matrices(partition, schmidt, n_occ, subspace='A', verbose=False)
trans_B = compute_transition_matrices(partition, schmidt, n_virt, subspace='B', verbose=False)

# Reference: sigma-vector
from dm_svd_embedding.embedded_hamiltonian import build_h_emb
H_ref, basis_ref, decomp_ref = build_h_emb(schmidt, partition, q_idx, backend, h1eff, h2_4d, n_occ, n_act, verbose=False)

# Reference H_AB
H_ref_AB = H_ref - decomp_ref['HA'] - decomp_ref['HB']

# RDM H_AB
D = H_ref.shape[0]
block_offsets = {}
off = 0
for n_A in sorted(schmidt.keys()):
    r = schmidt[n_A]['r']
    block_offsets[n_A] = off
    off += r*r

H_AB_rdm = np.zeros((D, D))
build_hab_rdm(H_AB_rdm, schmidt, block_offsets, trans_A, trans_B, h1eff, h2_4d, n_occ, n_act, verbose=True)

# Compare element-by-element for first 5x5
print("\n=== Element-wise comparison (first 10x10) ===")
N = min(10, D)
for i in range(N):
    for j in range(N):
        ref_v = H_ref_AB[i, j]
        rdm_v = H_AB_rdm[i, j]
        diff = abs(ref_v - rdm_v)
        if diff > 1e-3:
            print(f"  H_AB[{i},{j}]: ref={ref_v:+.6f} rdm={rdm_v:+.6f} diff={diff:.4f}")

# Also check: what are the norms of each contribution?
ref_HA = decomp_ref['HA']
ref_HB = decomp_ref['HB']
print(f"\n||H_A||={np.linalg.norm(ref_HA):.4f} ||H_B||={np.linalg.norm(ref_HB):.4f}")
print(f"||H_AB_ref||={np.linalg.norm(H_ref_AB):.4f} ||H_AB_rdm||={np.linalg.norm(H_AB_rdm):.4f}")
print(f"max|HA_diff|={np.abs(ref_HA - np.zeros_like(ref_HA)):.2e}")
print(f"Ratio rdm/ref = {np.linalg.norm(H_AB_rdm)/max(np.linalg.norm(H_ref_AB),1e-10):.4f}")
