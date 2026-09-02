import numpy as np, sys
sys.path.insert(0, '/data/home/wangcx/krylov-dci')
from dm_svd_embedding.embedded_hamiltonian import (_setup_h2o_system, _expand_schmidt_product_to_ci_matrix)
from dm_svd_embedding.occ_virt_partition import setup_partition, build_block_matrices
from dm_svd_embedding.density_matrix import compute_schmidt_decomposition
from dm_svd_embedding.transition_rdm import compute_transition_matrices, _compute_det_creation_explicit, _compute_det_create1_annih2_explicit
from dm_svd_embedding.hab_rdm_contract import _add_1a3b_patterns
(mol, mf, cas, q_idx, backend, h1eff, h2_4d, fcivec, ci_flat, n_act, n_elec, n_occ, na, nb, ecore, E_fci) = _setup_h2o_system()
n_virt = n_act - n_occ
partition, full_dets = setup_partition(n_act, n_elec, n_occ, ms=0)
C_blocks = build_block_matrices(partition, ci_flat)
schmidt = compute_schmidt_decomposition(C_blocks, eps=1e-3)
trans_A = compute_transition_matrices(partition, schmidt, n_occ, subspace='A', verbose=False)
trans_B = compute_transition_matrices(partition, schmidt, n_virt, subspace='B', verbose=False)

# build block offsets
bo = {}; off = 0
for nA in sorted(schmidt.keys()):
    bo[nA] = off; off += schmidt[nA]['r']**2
D = off

# code R13
R13 = np.zeros((D, D))
for nA in sorted(schmidt.keys()):
    if schmidt[nA]['r'] == 0: continue
    _add_1a3b_patterns(R13, bo, nA, trans_A, trans_B, h2_4d, n_occ, n_act, n_virt)

# manual R13: for n+1 (n_A=2 -> 3), 1a3b = A create_1, B c1a2
n_A = 2
asrc = partition[n_A]['a_dets']; adst = partition[n_A+1]['a_dets']; aidx = partition[n_A+1]['a_index']
Us = schmidt[n_A]['U']; Ud = schmidt[n_A+1]['U']
c1_A_raw = _compute_det_creation_explicit(asrc, adst, aidx, n_occ)
for j,(aA_j,bA_j) in enumerate(asrc):
    c1_A_raw['a'][:,j,:] *= (-1)**aA_j.bit_count()
    c1_A_raw['b'][:,j,:] *= (-1)**bA_j.bit_count()
c1_A_s = {k: np.einsum('ijp,ia,jg->agp', c1_A_raw[k], Ud, Us) for k in ('a','b')}

bsrc = partition[n_A]['b_dets']; bdst = partition[n_A+1]['b_dets']; bidx = partition[n_A+1]['b_index']
Vs = schmidt[n_A]['V']; Vd = schmidt[n_A+1]['V']
c1a2_B_raw = _compute_det_create1_annih2_explicit(bsrc, bdst, bidx, n_virt)
c1a2_B_s = {k: np.einsum('ijpqr,ia,jg->agpqr', c1a2_B_raw[k], Vd, Vs) for k in c1a2_B_raw}

r_src = schmidt[n_A]['r']; r_dst = schmidt[n_A+1]['r']
os = bo[n_A]; od = bo[n_A+1]
manual = np.zeros((D, D))
jw_a = {}
for spin in ('a','b'):
    pass
# A-side create JW is baked into c1_A_s already. extra -1 for qA same-spin.
for integ_kind, pairs in (('pA', [('a','aaa'),('a','bba'),('b','aab'),('b','bbb')]),
                          ('qA', [('a','aaa'),('b','aba'),('a','bab'),('b','bbb')])):
    for spin, combo in pairs:
        extra = -1.0 if (combo in ('aaa','bbb') and integ_kind=='qA') else 1.0
        TA = c1_A_s[spin]; TB = c1a2_B_s[combo]
        for a_dst in range(r_dst):
            for a_src in range(r_src):
                for b_dst in range(r_dst):
                    for b_src in range(r_src):
                        val = 0.0
                        for x in range(n_virt):
                            for y in range(n_virt):
                                for z in range(n_virt):
                                    tb = TB[b_dst,b_src,x,y,z]
                                    if abs(tb)<1e-14: continue
                                    for a in range(n_occ):
                                        ta = TA[a_dst,a_src,a]
                                        if abs(ta)<1e-14: continue
                                        v = h2_4d[a, z+n_occ, x+n_occ, y+n_occ] if integ_kind=='pA' else h2_4d[x+n_occ, z+n_occ, a, y+n_occ]
                                        val += 0.5*v*ta*tb*extra
                        if abs(val)>1e-14:
                            manual[od + a_dst*r_dst + b_dst, os + a_src*r_src + b_src] += val

# compare manual vs code for 2->3 block
slr = slice(od, od+r_dst*r_dst); slc = slice(os, os+r_src*r_src)
d = np.abs(manual[slr,slc] - R13[slr,slc]).max()
print(f"2->3 1a3b manual vs code: max|diff| = {d:.3e}")
print(f"manual[12,0]={manual[od+12, os+0]:+.6f}  code[12,0]={R13[od+12, os+0]:+.6f}")
