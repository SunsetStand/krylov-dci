import numpy as np
import sys
sys.path.insert(0, '/data/home/wangcx/krylov-dci')

from dm_svd_embedding.occ_virt_partition import setup_partition, build_block_matrices
from dm_svd_embedding.density_matrix import compute_schmidt_decomposition
from dm_svd_embedding.embedded_hamiltonian import build_h_emb, build_hemb_via_rdm
from dm_svd_embedding.transition_rdm import compute_transition_matrices
from pyscf import gto, scf, ao2mo, mcscf
from pyscf.fci import cistring
from src_mf.pyscf_backend import QSpaceIndex, KDCIBackend
from src.hamiltonian import _unpack_4fold

# N2/cc-pVDZ CAS(6,6) — small enough to be fast, but with non-trivial integrals
n_act, n_elec = 6, 6
n_occ = 3

mol = gto.M(atom='N 0 0 0; N 0 0 1.1', basis='cc-pvdz', verbose=0)
mf = scf.RHF(mol); mf.kernel()
cas = mcscf.CASCI(mf, n_act, n_elec)
cas.frozen = 2
h1eff, ecore = cas.get_h1eff()
h2eff = cas.get_h2eff()
cas.kernel()
fcivec = cas.ci
ci_flat = fcivec.reshape(-1)
E_fci = cas.e_tot

na = nb = n_elec // 2
alpha_strs = cistring.gen_strings4orblist(range(n_act), na)
beta_strs = cistring.gen_strings4orblist(range(n_act), nb)
q_idx = QSpaceIndex(alpha_strs, beta_strs, n_act, (na, nb), h1eff, h2eff)
backend = KDCIBackend(q_idx)
h2_4d = _unpack_4fold(h2eff, n_act)

n_virt = n_act - n_occ
partition, full_dets = setup_partition(n_act, n_elec, n_occ, ms=0)
C_blocks = build_block_matrices(partition, ci_flat)
schmidt = compute_schmidt_decomposition(C_blocks, eps=1e-3)

# Reference
H_ref, basis_ref, _ = build_h_emb(
    schmidt, partition, q_idx, backend, h1eff, h2_4d,
    n_occ, n_act, verbose=False)
D = H_ref.shape[0]
print(f"N2 CAS({n_act},{n_elec}) D={D}")

# RDM
trans_A = compute_transition_matrices(partition, schmidt, n_occ, subspace='A', verbose=False)
trans_B = compute_transition_matrices(partition, schmidt, n_virt, subspace='B', verbose=False)
H_rdm, basis_rdm, decomps = build_hemb_via_rdm(
    schmidt, partition, h1eff, h2_4d,
    n_occ, n_act, trans_A, trans_B, verbose=False)

# Compare
HA_ref = decomps['HA']
HB_ref = decomps['HB']
HAB_ref = H_ref - HA_ref - HB_ref

print(f"||H_AB_ref|| = {np.linalg.norm(HAB_ref):.4f}")
print(f"||H_AB_rdm|| = {np.linalg.norm(decomps['HAB']):.4f}")
print(f"||H_AB_diff|| = {np.linalg.norm(HAB_ref - decomps['HAB']):.4f}")

corr = np.corrcoef(HAB_ref.ravel(), decomps['HAB'].ravel())[0,1]
print(f"Correlation = {corr:.6f}")

# Check diagonals
print(f"\nH_AB_ref diag range: [{np.min(np.diag(HAB_ref)):.4f}, {np.max(np.diag(HAB_ref)):.4f}]")
print(f"H_AB_rdm diag range: [{np.min(np.diag(decomps['HAB'])):.4f}, {np.max(np.diag(decomps['HAB'])):.4f}]")

# Eigenvalue comparison
ev_ref = np.sort(np.linalg.eigvalsh(H_ref))
ev_rdm = np.sort(np.linalg.eigvalsh(H_rdm))
for i in range(min(5, D)):
    print(f"E[{i}]: ref={ev_ref[i]:.6f}, rdm={ev_rdm[i]:.6f}, diff={abs(ev_ref[i]-ev_rdm[i]):.4e}")

# Also test: is (pp|qq) ≠ (pq|qp) for p∈A, q∈B?
print(f"\n(0,{n_occ}|0,{n_occ}) = {h2_4d[0,n_occ,0,n_occ]:.6f}")
print(f"(0,{n_occ}|{n_occ},0) = {h2_4d[0,n_occ,n_occ,0]:.6f}")
