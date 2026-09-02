import numpy as np
import sys
sys.path.insert(0, '/data/home/wangcx/krylov-dci')

from dm_svd_embedding.occ_virt_partition import setup_partition, build_block_matrices
from dm_svd_embedding.density_matrix import compute_schmidt_decomposition
from dm_svd_embedding.embedded_hamiltonian import _setup_h2o_system
from dm_svd_embedding.transition_rdm import compute_transition_matrices

(mol, mf, cas, q_idx, backend, h1eff, h2_4d, fcivec, ci_flat,
 n_act, n_elec, n_occ, na, nb, ecore, E_fci) = _setup_h2o_system()
n_virt = n_act - n_occ

partition, full_dets = setup_partition(n_act, n_elec, n_occ, ms=0)
C_blocks = build_block_matrices(partition, ci_flat)
schmidt = compute_schmidt_decomposition(C_blocks, eps=1e-3)

trans_A = compute_transition_matrices(partition, schmidt, n_occ, subspace='A', verbose=False)
trans_B = compute_transition_matrices(partition, schmidt, n_virt, subspace='B', verbose=False)

# Check transition matrix diagonals
for n_A in schmidt:
    sd = schmidt[n_A]
    r = sd['r']
    if r == 0: continue
    
    if n_A in trans_A.trans_1:
        TA = trans_A.trans_1[n_A]
        print(f"n_A={n_A}: TA.trans_1 shape={TA.shape}")
        for a in range(min(3, r)):
            for p in range(n_occ):
                v = TA[a, a, p, p]
                if abs(v) > 1e-10:
                    print(f"  TA[{a},{a},{p},{p}] = {v:.6f}")
        # Also check a few off-diagonal
        if r > 1:
            v = TA[0, 1, 0, 0]
            if abs(v) > 1e-10:
                print(f"  TA[0,1,0,0] = {v:.6f}")
    
    if n_A in trans_B.trans_1:
        TB = trans_B.trans_1[n_A]
        print(f"n_A={n_A}: TB.trans_1 shape={TB.shape}")
        for a in range(min(3, r)):
            for p in range(n_virt):
                v = TB[a, a, p, p]
                if abs(v) > 1e-10:
                    print(f"  TB[{a},{a},{p},{p}] = {v:.6f}")

# Now trace one specific mean-field term:
# (p=0∈A, q=0∈B, r=0∈A, s=0∈B) with (00|00) integral
print(f"\nIntegral (0,0,0,0) = {h2_4d[0,0,0,0]:.6f}")
print(f"Integral (0,{n_occ},{n_occ},0) = {h2_4d[0,n_occ,n_occ,0]:.6f} (p∈A,q∈B,r∈B,s∈A)")

# Check: for n_A block that exists, does the mean-field term get included?
for n_A in sorted(schmidt.keys()):
    sd = schmidt[n_A]
    if sd['r'] == 0: continue
    if n_A not in trans_A.trans_1: continue
    TA = trans_A.trans_1[n_A]
    TB = trans_B.trans_1[n_A]
    
    # Mean-field: p∈A, q∈B, s∈B, r∈A (same orbitals p=r, q=s)
    # This is the n_A-conserved pattern
    for p_A in range(n_occ):
        for q_B in range(n_virt):
            integ = h2_4d[p_A, q_B + n_occ, q_B + n_occ, p_A]  # (pq|qp) = (pp|qq)... wait
            # Actually mean-field is (pp|qq) = h2_full[p, p, q, q]
            integ_mf = h2_4d[p_A, p_A, q_B + n_occ, q_B + n_occ]
            if abs(integ_mf) > 1e-10:
                print(f"n_A={n_A}: (p={p_A},p={p_A}|q={q_B},q={q_B}) = {integ_mf:.6f}")
                break
        break
    break
