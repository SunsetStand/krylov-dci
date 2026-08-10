import sys, os, numpy as np
sys.path.insert(0, '/data/home/wangcx/krylov-dci')

from dm_svd_embedding.embedded_hamiltonian import _setup_h2o_system, build_h_emb
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

# Reference H^emb
H_ref, _, decomp = build_h_emb(schmidt, partition, q_idx, backend, h1eff, h2_4d, n_occ, n_act, verbose=False)
H_ref_AB = H_ref - decomp['HA'] - decomp['HB']
D = H_ref.shape[0]

# Direct 1-RDM computation and 1e contribution
block_offsets = {}
off = 0
for n_A in sorted(schmidt.keys()):
    r = schmidt[n_A]['r']
    block_offsets[n_A] = off
    off += r*r

# Build 1-RDM in Schmidt product basis and contract
H_1e_rdm = np.zeros((D, D))

for n_A in sorted(schmidt.keys()):
    sd = schmidt[n_A]
    r = sd['r']
    if r == 0:
        continue
    os = block_offsets[n_A]
    
    TA1 = trans_A.trans_1.get(n_A)  # (r, r, n_occ, n_occ)
    TB1 = trans_B.trans_1.get(n_A)  # (r, r, n_virt, n_virt)
    
    # 1e: Σ_{p,q} h_pq a_p† a_q
    # A-A: δ_{βδ} · TA1[α,γ,p,q] · h[p,q]
    # B-B: δ_{αγ} · TB1[β,δ,p,q] · h[p+n_occ,q+n_occ]
    # A-B: create_A[p,α,γ] · annihilate_B[q,β,δ] · h[p,q+n_occ]
    # B-A: annihilate_A[p,α,γ] · create_B[q,β,δ] · h[p+n_occ,q]
    
    for a_dst in range(r):
        for a_src in range(r):
            for b_dst in range(r):
                for b_src in range(r):
                    val = 0.0
                    # A-A
                    for p in range(n_occ):
                        for qq in range(n_occ):
                            if abs(TA1[a_dst, a_src, p, qq]) > 1e-14:
                                if b_dst == b_src:
                                    val += h1eff[p, qq] * TA1[a_dst, a_src, p, qq]
                    # B-B
                    for p_b in range(n_virt):
                        for q_b in range(n_virt):
                            if abs(TB1[b_dst, b_src, p_b, q_b]) > 1e-14:
                                if a_dst == a_src:
                                    val += h1eff[p_b + n_occ, q_b + n_occ] * TB1[b_dst, b_src, p_b, q_b]
                    
                    # A-B cross
                    cre_A = trans_A.create_1.get(n_A)
                    ann_B = trans_B.annihilate_1.get(n_A)
                    if cre_A is not None and ann_B is not None:
                        for p in range(n_occ):
                            ta_c = cre_A[a_dst, a_src, p]
                            if abs(ta_c) < 1e-14: continue
                            for q_b in range(n_virt):
                                tb_a = ann_B[b_dst, b_src, q_b]
                                if abs(tb_a) < 1e-14: continue
                                val += h1eff[p, q_b + n_occ] * ta_c * tb_a
                    
                    # B-A cross
                    ann_A = trans_A.annihilate_1.get(n_A)
                    cre_B = trans_B.create_1.get(n_A)
                    if ann_A is not None and cre_B is not None:
                        for qq in range(n_occ):
                            ta_a = ann_A[a_dst, a_src, qq]
                            if abs(ta_a) < 1e-14: continue
                            for p_b in range(n_virt):
                                tb_c = cre_B[b_dst, b_src, p_b]
                                if abs(tb_c) < 1e-14: continue
                                val += h1eff[p_b + n_occ, qq] * ta_a * tb_c
                    
                    if abs(val) > 1e-14:
                        ss = os + a_src * r + b_src
                        dd = os + a_dst * r + b_dst
                        H_1e_rdm[dd, ss] += val

# Compare: reference 1e part = ?
# The reference H includes both 1e and 2e. Let me just compare norms.
H_1e_ref = np.zeros((D, D))
# Actually, I can't easily separate 1e from 2e in the reference.
# Let me just print the norm comparison.

print(f"||H_1e_rdm|| = {np.linalg.norm(H_1e_rdm):.4f}")
print(f"||H_ref|| = {np.linalg.norm(H_ref):.4f}")
print(f"||HA|| = {np.linalg.norm(decomp['HA']):.4f}, ||HB|| = {np.linalg.norm(decomp['HB']):.4f}")

# Diagonalize HA+HB+H_1e and compare
H_test = decomp['HA'] + decomp['HB'] + H_1e_rdm
ev_test = np.linalg.eigvalsh(H_test)
ev_ref = np.linalg.eigvalsh(H_ref)
print("\nWith HA+HB+H_1e_rdm only (no 2e cross):")
for i in range(min(5, D)):
    print(f"  E[{i}]: ref={ev_ref[i]+ecore:.6f} test={ev_test[i]+ecore:.6f} diff={abs(ev_ref[i]-ev_test[i]):.4f} Ha")
