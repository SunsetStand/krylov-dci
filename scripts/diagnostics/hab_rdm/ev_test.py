import numpy as np
import sys; sys.path.insert(0,'/data/home/wangcx/krylov-dci')
from dm_svd_embedding.embedded_hamiltonian import _setup_h2o_system,build_h_emb,build_hemb_via_rdm
from dm_svd_embedding.occ_virt_partition import setup_partition,build_block_matrices
from dm_svd_embedding.density_matrix import compute_schmidt_decomposition
from dm_svd_embedding.transition_rdm import compute_transition_matrices

_,_,_,_,_,_,_,ci_flat,n_act,n_elec,n_occ,_,_,_,_ = _setup_h2o_system()
n_virt=n_act-n_occ
q_idx=_setup_h2o_system()[2]
backend=_setup_h2o_system()[3]
h1eff=_setup_h2o_system()[4]
h2_4d=_setup_h2o_system()[5]
partition,_ = setup_partition(n_act,n_elec,n_occ,ms=0)
C_blocks=build_block_matrices(partition,ci_flat)
schmidt=compute_schmidt_decomposition(C_blocks,eps=1e-3)
H_ref,_,_=build_h_emb(schmidt,partition,q_idx,backend,h1eff,h2_4d,n_occ,n_act,verbose=False)
trans_A=compute_transition_matrices(partition,schmidt,n_occ,subspace='A',verbose=False)
trans_B=compute_transition_matrices(partition,schmidt,n_virt,subspace='B',verbose=False)
H_rdm,_,decomps=build_hemb_via_rdm(schmidt,partition,h1eff,h2_4d,n_occ,n_act,trans_A,trans_B,verbose=False)
HA,HB,HAB_rdm=decomps['HA'],decomps['HB'],decomps['HAB']
H_HAHB=HA+HB
ev_hahb=np.sort(np.linalg.eigvalsh(H_HAHB))
ev_ref=np.sort(np.linalg.eigvalsh(H_ref))
ev_rdm=np.sort(np.linalg.eigvalsh(H_rdm))
print('Eigenvalues (Ha):')
for i in range(5):
    print(f'  E[{i}]: HA+HB={ev_hahb[i]:.4f}, ref={ev_ref[i]:.4f}, rdm={ev_rdm[i]:.4f}')
    d_ref=ev_ref[i]-ev_hahb[i]
    d_rdm=ev_rdm[i]-ev_hahb[i]
    print(f'         HAB shift: ref={d_ref*1000:+.0f} mH, rdm={d_rdm*1000:+.0f} mH')
