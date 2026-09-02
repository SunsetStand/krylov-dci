import numpy as np
import sys
sys.path.insert(0, '/data/home/wangcx/krylov-dci')

# Monkey-patch to trace
import dm_svd_embedding.hab_rdm_contract as hrc

_orig_contract = hrc._contract_2e_nA_conserved

def traced_contract(H_AB, offset_src, n_A_val, TA, TB, h2_full, n_occ, n_act, n_virt, rA_sch):
    """Traced version that logs contributions to a specific matrix element."""
    old_HAB_diag = H_AB[offset_src, offset_src].copy() if offset_src < H_AB.shape[0] else 0
    
    _orig_contract(H_AB, offset_src, n_A_val, TA, TB, h2_full, n_occ, n_act, n_virt, rA_sch)
    
    new_HAB_diag = H_AB[offset_src, offset_src] if offset_src < H_AB.shape[0] else 0
    if abs(new_HAB_diag - old_HAB_diag) > 1e-10:
        print(f"  n_A={n_A_val}: H_AB[{offset_src},{offset_src}] changed by {new_HAB_diag - old_HAB_diag:.6f}")

hrc._contract_2e_nA_conserved = traced_contract

# Now rerun
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

trans_A = compute_transition_matrices(partition, schmidt, n_occ, subspace='A', verbose=False)
trans_B = compute_transition_matrices(partition, schmidt, n_virt, subspace='B', verbose=False)

H_rdm, basis_rdm, decomps = build_hemb_via_rdm(
    schmidt, partition, h1eff, h2_4d,
    n_occ, n_act, trans_A, trans_B, verbose=False)

print(f"\nFinal H_AB[0,0] = {decomps['HAB'][0,0]:.6e}")
print(f"Final H_AB[0,1] = {decomps['HAB'][0,1]:.6e}")
