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

# Block-by-block comparison
block_offsets = {}
offset = 0
for n_A in sorted(schmidt.keys()):
    r = schmidt[n_A]['r']
    block_offsets[n_A] = offset
    offset += r * r

for n_A in sorted(schmidt.keys()):
    r = schmidt[n_A]['r']
    if r == 0: continue
    o = block_offsets[n_A]
    sz = r * r
    ref_block = HAB_ref[o:o+sz, o:o+sz]
    rdm_block = HAB_rdm[o:o+sz, o:o+sz]
    
    norm_ref = np.linalg.norm(ref_block)
    norm_rdm = np.linalg.norm(rdm_block)
    diff_norm = np.linalg.norm(ref_block - rdm_block)
    
    diag_ref = np.diag(ref_block)
    diag_rdm = np.diag(rdm_block)
    
    print(f"n_A={n_A}: r={r}, size={sz}")
    print(f"  ||H_AB_ref||={norm_ref:.2f}, ||H_AB_rdm||={norm_rdm:.2f}, diff={diff_norm:.2f}")
    print(f"  ref diag: [{diag_ref.min():.2f}, {diag_ref.max():.2f}] mean={diag_ref.mean():.2f}")
    print(f"  rdm diag: [{diag_rdm.min():.2f}, {diag_rdm.max():.2f}] mean={diag_rdm.mean():.2f}")

# Manual term (b) for n_A=3
print("\n=== Detailed term (b) for n_A=3 ===")
n_A = 3
sd = schmidt[n_A]
r = sd['r']
TA = trans_A.trans_1[n_A]
TB = trans_B.trans_1[n_A]

for alpha in range(r):
    for beta in range(r):
        val = 0.0
        for i in range(n_occ):
            for j in range(n_occ):
                ta = TA[alpha, alpha, i, j]
                if abs(ta) < 1e-10: continue
                for k in range(n_virt):
                    for l in range(n_virt):
                        v = h2_4d[i, j, k + n_occ, l + n_occ]
                        if abs(v) < 1e-10: continue
                        tb = TB[beta, beta, k, l]
                        val += ta * v * tb
        flat = block_offsets[n_A] + alpha * r + beta
        ref_val = HAB_ref[flat, flat]
        rdm_val = HAB_rdm[flat, flat]
        print(f"  (a={alpha},b={beta}): term(b)={val:.4f}, ref={ref_val:.4f}, rdm={rdm_val:.4f}")
