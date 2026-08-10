import sys, os, numpy as np
sys.path.insert(0, '/data/home/wangcx/krylov-dci')
from dm_svd_embedding.embedded_hamiltonian import _setup_h2o_system, build_h_emb, build_hemb_via_rdm
from dm_svd_embedding.occ_virt_partition import setup_partition, build_block_matrices
from dm_svd_embedding.density_matrix import compute_schmidt_decomposition
from dm_svd_embedding.transition_rdm import compute_transition_matrices

(mol, mf, cas, q_idx, backend, h1eff, h2_4d, fcivec, ci_flat,
 n_act, n_elec, n_occ, na, nb, ecore, E_fci) = _setup_h2o_system()
n_virt = n_act - n_occ
partition, full_dets = setup_partition(n_act, n_elec, n_occ, ms=0)
C_blocks = build_block_matrices(partition, ci_flat)
schmidt = compute_schmidt_decomposition(C_blocks, eps=1e-3)
trans_A = compute_transition_matrices(partition, schmidt, n_occ, 'A', verbose=False)
trans_B = compute_transition_matrices(partition, schmidt, n_virt, 'B', verbose=False)

H_ref, _, decomp = build_h_emb(schmidt, partition, q_idx, backend, h1eff, h2_4d, n_occ, n_act, verbose=False)
H_rdm, _, _ = build_hemb_via_rdm(schmidt, partition, h1eff, h2_4d, n_occ, n_act, trans_A, trans_B, verbose=False)

ev_ref = np.linalg.eigvalsh(H_ref)
D = H_ref.shape[0]

# Try scaling H_AB by different factors
H_AB_rdm = H_rdm - decomp['HA'] - decomp['HB']
for scale in [0.5, 1.0, 2.0, 2.5, 3.0, 4.0]:
    H_test = decomp['HA'] + decomp['HB'] + scale * H_AB_rdm
    H_test = 0.5 * (H_test + H_test.T)
    ev_test = np.linalg.eigvalsh(H_test)
    mae = np.mean(np.abs(ev_ref[:5] - ev_test[:5]))
    print(f"scale={scale:.1f}: MAE(5 states)={mae:.6f} Ha  E[0] test={ev_test[0]+ecore:.6f} (ref={ev_ref[0]+ecore:.6f})")
