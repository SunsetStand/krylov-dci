import numpy as np
import sys
sys.path.insert(0, '/data/home/wangcx/krylov-dci')

from dm_svd_embedding.occ_virt_partition import setup_partition, build_block_matrices
from dm_svd_embedding.density_matrix import compute_schmidt_decomposition
from dm_svd_embedding.transition_rdm import compute_transition_matrices
from pyscf import gto, scf, mcscf
from pyscf.fci import cistring
from src_mf.pyscf_backend import QSpaceIndex, KDCIBackend
from src.hamiltonian import _unpack_4fold
import dm_svd_embedding.hab_rdm_contract as hrc

n_act, n_elec = 6, 6
n_occ = 3
n_virt = n_act - n_occ

mol = gto.M(atom='N 0 0 0; N 0 0 1.1', basis='cc-pvdz', verbose=0)
mf = scf.RHF(mol); mf.kernel()
cas = mcscf.CASCI(mf, n_act, n_elec)
cas.frozen = 2
h1eff, ecore = cas.get_h1eff()
h2eff = cas.get_h2eff()
cas.kernel()
fcivec = cas.ci
ci_flat = fcivec.reshape(-1)
h2_4d = _unpack_4fold(h2eff, n_act)

partition, full_dets = setup_partition(n_act, n_elec, n_occ, ms=0)
C_blocks = build_block_matrices(partition, ci_flat)
schmidt = compute_schmidt_decomposition(C_blocks, eps=1e-3)

trans_A = compute_transition_matrices(partition, schmidt, n_occ, subspace='A', verbose=False)
trans_B = compute_transition_matrices(partition, schmidt, n_virt, subspace='B', verbose=False)

# Manually trace one block: n_A=3
n_A = 3
sd = schmidt[n_A]
r = sd['r']
print(f"n_A={n_A}: r={r}, blocks={sorted(schmidt.keys())}")
TA = trans_A.trans_1.get(n_A)
TB = trans_B.trans_1.get(n_A)
print(f"TA trans_1 shape: {TA.shape}, TB trans_1 shape: {TB.shape}")

# Check non-zero diagonal elements of TA and TB
print(f"\nTA non-zero diagonals:")
for a in range(r):
    for p in range(n_occ):
        v = TA[a, a, p, p]
        if abs(v) > 1e-6:
            print(f"  TA[{a},{a},{p},{p}] = {v:.6f}")

print(f"TB non-zero diagonals:")
for a in range(r):
    for p in range(n_virt):
        v = TB[a, a, p, p]
        if abs(v) > 1e-6:
            print(f"  TB[{a},{a},{p},{p}] = {v:.6f}")

# Now run the contraction with tracing
offset_src = 0  # We'll track this
count_pass = 0
count_nonzero_integ = 0
count_contrib = 0

for p in range(n_act):
    p_in_A = p < n_occ
    for q in range(n_act):
        q_in_A = q < n_occ
        for r_loop in range(n_act):
            r_in_A = r_loop < n_occ
            for s_loop in range(n_act):
                s_in_A = s_loop < n_occ
                
                n_in_A = p_in_A + q_in_A + r_in_A + s_in_A
                if n_in_A != 2: continue
                n_create_A = p_in_A + q_in_A
                n_annih_A = r_in_A + s_in_A
                if n_create_A != 1 or n_annih_A != 1: continue
                count_pass += 1
                
                integ = h2_4d[p, q, r_loop, s_loop]
                if abs(integ) < 1e-14: continue
                count_nonzero_integ += 1
                
                # JW (FIXED version)
                cur = n_A
                jw = 1
                if s_in_A: cur -= 1
                else: jw *= hrc._jw(cur)
                if r_in_A: cur -= 1
                else: jw *= hrc._jw(cur)
                if q_in_A: cur += 1
                else: jw *= hrc._jw(cur)
                if p_in_A: cur += 1
                else: jw *= hrc._jw(cur)
                
                p_sub = p if p_in_A else p - n_occ
                q_sub = q if q_in_A else q - n_occ
                r_sub = r_loop if r_in_A else r_loop - n_occ
                s_sub = s_loop if s_in_A else s_loop - n_occ
                
                a_create, a_annih, b_create, b_annih = [], [], [], []
                for ot, ia, si in [('c',p_in_A,p_sub),('c',q_in_A,q_sub),('a',s_in_A,s_sub),('a',r_in_A,r_sub)]:
                    if ia:
                        (a_create if ot=='c' else a_annih).append(si)
                    else:
                        (b_create if ot=='c' else b_annih).append(si)
                if len(a_create)!=1 or len(a_annih)!=1: continue
                ac, aa = a_create[0], a_annih[0]
                bc, ba = b_create[0], b_annih[0]
                
                ta_slice = TA[:, :, ac, aa]
                tb_slice = TB[:, :, bc, ba]
                
                factor = 0.5 * integ * jw
                
                for a_dst in range(r):
                    for a_src in range(r):
                        ta = ta_slice[a_dst, a_src]
                        if abs(ta) < 1e-14: continue
                        for b_dst in range(r):
                            for b_src in range(r):
                                tb = tb_slice[b_dst, b_src]
                                if abs(tb) < 1e-14: continue
                                v = factor * ta * tb
                                if abs(v) < 1e-14: continue
                                count_contrib += 1
                                is_diag = (a_dst==a_src and b_dst==b_src)
                                if count_contrib <= 10 or is_diag:
                                    print(f"  p={p}({p_sub}) q={q}({q_sub}) r={r_loop}({r_sub}) s={s_loop}({s_sub}): "
                                          f"H[{a_dst},{b_dst}|{a_src},{b_src}] += {v:.6f} "
                                          f"(integ={integ:.6f} jw={jw} ta={ta:.6f} tb={tb:.6f}) {'DIAG' if is_diag else ''}")

print(f"\nPass filter: {count_pass}, nonzero integ: {count_nonzero_integ}, contributions: {count_contrib}")

# Explicitly check: mean-field (pp|qq) for p=0(A), q=0(B)
# This should have p=r=0(A), q=s=0+n_occ=3(B)
pA, qB = 0, 0
direct_idx = (pA, qB+n_occ, pA, qB+n_occ)
exch_idx = (pA, qB+n_occ, qB+n_occ, pA)
print(f"\n(pp|qq) = h2[{pA},{qB+n_occ},{pA},{qB+n_occ}] = {h2_4d[direct_idx]:.6f}")
print(f"(pq|qp) = h2[{pA},{qB+n_occ},{qB+n_occ},{pA}] = {h2_4d[exch_idx]:.6f}")
