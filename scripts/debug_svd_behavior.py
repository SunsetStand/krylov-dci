import sys; sys.path.insert(0,'src')
import numpy as np
from pyscf import gto, scf, ao2mo
from hamiltonian import Hamiltonian, _unpack_4fold
from determinants import generate_determinants_ms
from partitioning import partition_cas, compute_reference_energy
from krylov import compute_A, build_H_QP, generate_layer_0
from svd_compression import build_weighted_coupling

mol=gto.M(atom='N 0 0 0; N 0 0 2.20',basis='cc-pVDZ',verbose=0)
mf=scf.RHF(mol);mf.kernel()
N_CORE=3;N_ACT=8;N_ELEC=8
mo=mf.mo_coeff[:,N_CORE:N_CORE+N_ACT]
h1_ao=mol.intor_symmetric('int1e_kin')+mol.intor_symmetric('int1e_nuc')
h2_act=_unpack_4fold(ao2mo.incore.full(mol.intor('int2e'),mo),N_ACT)
h1_act=mo.T@h1_ao@mo
ham=Hamiltonian(h1=h1_act,h2=h2_act,E_nuc=0.0,E_HF=mf.e_tot)
dets=generate_determinants_ms(N_ACT,N_ELEC,ms=0)
M=len(dets)

for label,sub_n,sub_elec in [('CAS(4,4)',4,4),('CAS(6,6)',6,6)]:
    p_idx,q_idx=partition_cas(N_ACT,N_ELEC,n_active_orb=sub_n,n_active_elec=sub_elec)
    p_dets=[dets[i] for i in p_idx]; q_dets=[dets[i] for i in q_idx]
    N=len(p_dets); Mq=len(q_dets)
    E0=compute_reference_energy(ham,dets,p_idx)
    diag_H_QQ=np.array([ham.diagonal_element(a,b) for a,b in q_dets])
    A_diag=compute_A(E0,diag_H_QQ)
    H_QP_mat=build_H_QP(ham,p_dets,q_dets)
    layer0=generate_layer_0(H_QP_mat,A_diag)
    T=build_weighted_coupling(layer0,A_diag)
    U,s,Vt=np.linalg.svd(T,full_matrices=False)
    ratios=s/s[0]
    print(f'{label}: P={N}, Q={Mq}')
    r8=[f'{r:.4f}' for r in ratios[:8]]
    print(f'  sigma/sigma1: {r8}...')
    print(f'  >1e-3: {np.sum(ratios>1e-3)}, >1e-2: {np.sum(ratios>1e-2)}, >1e-1: {np.sum(ratios>1e-1)}')
    print()

hf_a=(1<<4)-1;hf_b=(1<<4)-1;E_hf=ham.diagonal_element(hf_a,hf_b)
for label,de in [('EW 1.0Ha',1.0),('EW 2.0Ha',2.0)]:
    p_idx=[];q_idx=[]
    for i,(a,b) in enumerate(dets):
        (p_idx if abs(ham.diagonal_element(a,b)-E_hf)<de else q_idx).append(i)
    p_idx=np.array(p_idx);q_idx=np.array(q_idx)
    p_dets=[dets[i] for i in p_idx];q_dets=[dets[i] for i in q_idx]
    N=len(p_dets);Mq=len(q_dets)
    if Mq==0: continue
    E0=compute_reference_energy(ham,dets,p_idx)
    diag_H_QQ=np.array([ham.diagonal_element(a,b) for a,b in q_dets])
    A_diag=compute_A(E0,diag_H_QQ)
    H_QP_mat=build_H_QP(ham,p_dets,q_dets)
    layer0=generate_layer_0(H_QP_mat,A_diag)
    T=build_weighted_coupling(layer0,A_diag)
    U,s,Vt=np.linalg.svd(T,full_matrices=False)
    ratios=s/s[0]
    print(f'{label} at R=2.20: P={N}, Q={Mq}')
    r8=[f'{r:.4f}' for r in ratios[:8]]
    print(f'  sigma/sigma1: {r8}...')
    print(f'  >1e-3: {np.sum(ratios>1e-3)}, >1e-2: {np.sum(ratios>1e-2)}, >1e-1: {np.sum(ratios>1e-1)}')
    print(f'  last ratio: {ratios[-1]:.4e}')
