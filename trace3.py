import numpy as np
import sys
sys.path.insert(0, '/data/home/wangcx/krylov-dci')

from dm_svd_embedding.occ_virt_partition import setup_partition, build_block_matrices
from dm_svd_embedding.density_matrix import compute_schmidt_decomposition
from dm_svd_embedding.embedded_hamiltonian import _setup_h2o_system
from dm_svd_embedding.transition_rdm import compute_transition_matrices
import dm_svd_embedding.hab_rdm_contract as hrc

(mol, mf, cas, q_idx, backend, h1eff, h2_4d, fcivec, ci_flat,
 n_act, n_elec, n_occ, na, nb, ecore, E_fci) = _setup_h2o_system()
n_virt = n_act - n_occ

partition, full_dets = setup_partition(n_act, n_elec, n_occ, ms=0)
C_blocks = build_block_matrices(partition, ci_flat)
schmidt = compute_schmidt_decomposition(C_blocks, eps=1e-3)

trans_A = compute_transition_matrices(partition, schmidt, n_occ, subspace='A', verbose=False)
trans_B = compute_transition_matrices(partition, schmidt, n_virt, subspace='B', verbose=False)

# Build H_AB manually, tracing the n_A-conserved loop
D = 70
H_AB = np.zeros((D, D))

block_offsets = {2: 0, 3: 1, 4: 17, 5: 53, 6: 69}

# Manually call the contraction for n_A=2 and trace
n_A = 2
os = 0
rA_sch = 1
TA = trans_A.trans_1[n_A]
TB = trans_B.trans_1[n_A]

print(f"n_A={n_A}: r={rA_sch}, TA={TA.shape}, TB={TB.shape}")

count_total = 0
count_pass = 0
count_nonzero_integ = 0
count_diag = 0
count_offdiag = 0

for p in range(n_act):
    p_in_A = p < n_occ
    for q in range(n_act):
        q_in_A = q < n_occ
        for r in range(n_act):
            r_in_A = r < n_occ
            for s in range(n_act):
                s_in_A = s < n_occ
                count_total += 1
                
                n_in_A = p_in_A + q_in_A + r_in_A + s_in_A
                if n_in_A != 2:
                    continue
                n_create_A = p_in_A + q_in_A
                n_annih_A = r_in_A + s_in_A
                if n_create_A != 1 or n_annih_A != 1:
                    continue
                count_pass += 1
                
                integ = h2_4d[p, q, r, s]
                if abs(integ) < 1e-14:
                    continue
                count_nonzero_integ += 1
                
                # Compute JW sign (FIXED version)
                cur = n_A
                jw = 1
                # a_r (integral idx s)
                if s_in_A: cur -= 1
                else: jw *= hrc._jw(cur)
                # a_s (integral idx r)
                if r_in_A: cur -= 1
                else: jw *= hrc._jw(cur)
                # a_q†
                if q_in_A: cur += 1
                else: jw *= hrc._jw(cur)
                # a_p†
                if p_in_A: cur += 1
                else: jw *= hrc._jw(cur)
                
                p_sub = p if p_in_A else p - n_occ
                q_sub = q if q_in_A else q - n_occ
                r_sub = r if r_in_A else r - n_occ
                s_sub = s if s_in_A else s - n_occ
                
                # Explicit creation/annihilation
                a_create, a_annih, b_create, b_annih = [], [], [], []
                for op_type, in_A, sub_idx in [
                    ('c', p_in_A, p_sub), ('c', q_in_A, q_sub),
                    ('a', s_in_A, s_sub), ('a', r_in_A, r_sub)]:
                    if in_A:
                        (a_create if op_type=='c' else a_annih).append(sub_idx)
                    else:
                        (b_create if op_type=='c' else b_annih).append(sub_idx)
                
                if len(a_create)!=1 or len(a_annih)!=1:
                    continue
                ac, aa = a_create[0], a_annih[0]
                bc, ba = b_create[0], b_annih[0]
                
                ta_slice = TA[:, :, ac, aa]
                tb_slice = TB[:, :, bc, ba]
                
                factor = 0.5 * integ * jw
                
                for a_dst in range(rA_sch):
                    for a_src in range(rA_sch):
                        ta = ta_slice[a_dst, a_src]
                        if abs(ta) < 1e-14: continue
                        for b_dst in range(rA_sch):
                            for b_src in range(rA_sch):
                                tb = tb_slice[b_dst, b_src]
                                if abs(tb) < 1e-14: continue
                                v = factor * ta * tb
                                if abs(v) < 1e-14: continue
                                s_idx = os + a_src * rA_sch + b_src
                                d_idx = os + a_dst * rA_sch + b_dst
                                H_AB[d_idx, s_idx] += v
                                if d_idx == s_idx:
                                    count_diag += 1
                                else:
                                    count_offdiag += 1
                                if count_diag + count_offdiag <= 5:
                                    print(f"  p={p} q={q} r={r} s={s}: "
                                          f"H_AB[{d_idx},{s_idx}] += {v:.6f} "
                                          f"(integ={integ:.6f}, jw={jw}, ta={ta:.6f}, tb={tb:.6f})")

print(f"\nTotal: {count_total}, pass filter: {count_pass}, "
      f"nonzero integ: {count_nonzero_integ}")
print(f"Added: {count_diag} diag + {count_offdiag} offdiag = {count_diag + count_offdiag}")
print(f"H_AB[0,0] = {H_AB[0,0]:.10e}")
