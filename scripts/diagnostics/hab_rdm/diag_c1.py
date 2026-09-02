import numpy as np, sys
sys.path.insert(0, '/data/home/wangcx/krylov-dci')
from dm_svd_embedding.embedded_hamiltonian import _setup_h2o_system
from dm_svd_embedding.occ_virt_partition import setup_partition, build_block_matrices
from dm_svd_embedding.density_matrix import compute_schmidt_decomposition
from dm_svd_embedding.transition_rdm import compute_transition_matrices, _compute_det_creation_explicit, _compute_det_annihilation_explicit
(mol, mf, cas, q_idx, backend, h1eff, h2_4d, fcivec, ci_flat, n_act, n_elec, n_occ, na, nb, ecore, E_fci) = _setup_h2o_system()
n_virt = n_act - n_occ
partition, full_dets = setup_partition(n_act, n_elec, n_occ, ms=0)
C_blocks = build_block_matrices(partition, ci_flat)
schmidt = compute_schmidt_decomposition(C_blocks, eps=1e-3)
trans_A = compute_transition_matrices(partition, schmidt, n_occ, subspace='A', verbose=False)

def st(T, Ud, Us):
    return np.einsum('ijp,ia,jg->agp', T, Ud, Us)

# A-side create_1: source n=2 -> dest n=3, with JW (-1)^{n_sigma}
n_A = 2
asrc = partition[n_A]['a_dets']
adst = partition[n_A+1]['a_dets']
aidx = partition[n_A+1]['a_index']
Us = schmidt[n_A]['U']; Ud = schmidt[n_A+1]['U']
T = _compute_det_creation_explicit(asrc, adst, aidx, n_occ)
# apply JW
for j, (aA_j, bA_j) in enumerate(asrc):
    T['a'][:, j, :] *= (-1)**aA_j.bit_count()
    T['b'][:, j, :] *= (-1)**bA_j.bit_count()
manual = {k: st(T[k], Ud, Us) for k in ('a','b')}
print("A-side create_1 (JW bake):")
for k in manual:
    got = trans_A.create_1_explicit[n_A][k]
    d = np.abs(got - manual[k]).max()
    print(f"  create_1[{k}] max|diff| = {d:.3e} {'OK' if d < 1e-12 else 'FAIL'}")
