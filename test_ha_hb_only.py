import sys, os, numpy as np
sys.path.insert(0, '/data/home/wangcx/krylov-dci')

from dm_svd_embedding.embedded_hamiltonian import _setup_h2o_system, build_h_emb
from dm_svd_embedding.occ_virt_partition import setup_partition, build_block_matrices
from dm_svd_embedding.density_matrix import compute_schmidt_decomposition

(mol, mf, cas, q_idx, backend, h1eff, h2_4d, fcivec, ci_flat,
 n_act, n_elec, n_occ, na, nb, ecore, E_fci) = _setup_h2o_system()

n_virt = n_act - n_occ
partition, full_dets = setup_partition(n_act, n_elec, n_occ, ms=0)
C_blocks = build_block_matrices(partition, ci_flat)
schmidt = compute_schmidt_decomposition(C_blocks, eps=1e-3)

# Full H^emb (reference)
H_full, _, decomp = build_h_emb(schmidt, partition, q_idx, backend, h1eff, h2_4d, n_occ, n_act, verbose=False)

# Just HA + HB
H_hab_only = decomp['HA'] + decomp['HB']

# Eigenvalues
ev_full = np.linalg.eigvalsh(H_full)
ev_hab = np.linalg.eigvalsh(H_hab_only)
ev_fci_ref = np.linalg.eigvalsh(H_full)  # full H^emb should give FCI (within Schmidt basis truncation)

print("Eigenvalue comparison (relative to ecore):")
for i in range(min(5, len(ev_full))):
    print(f"  E[{i}]: full={ev_full[i]+ecore:.6f} HA+HB_only={ev_hab[i]+ecore:.6f} diff={abs(ev_full[i]-ev_hab[i]):.4f} Ha = {abs(ev_full[i]-ev_hab[i])*1000:.1f} mH")
