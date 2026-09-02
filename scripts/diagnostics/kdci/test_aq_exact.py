#!/usr/bin/env python3
"""Test: A_q = 1/(E_DMRG[k] - H_D), per-state, m=0, with excited states."""
import sys,numpy as np,itertools
sys.path.insert(0,'/data/home/wangcx/krylov-dci')
from src_mf import QSpaceIndex, KDCIBackend
from src.effective_h import build_effective_H, diagonalize_effective_H
from src.determinants import hf_determinant, bit_positions
from src.hamiltonian import Hamiltonian
from pyscf import gto,scf,ao2mo
from pyscf.fci import cistring,direct_spin1

N_CORE,N_ACT,P_N,NROOTS=2,10,400,6
mol=gto.M(atom='N 0 0 0; N 0 0 1.1',basis='cc-pVDZ',verbose=0)
mf=scf.RHF(mol).run(verbose=0)
no=list(range(N_CORE,N_CORE+N_ACT)); norb=mf.mo_coeff.shape[1]
h1m=mf.mo_coeff.T@mf.get_hcore()@mf.mo_coeff
e2=ao2mo.full(mol.intor('int2e'),mf.mo_coeff,compact=False).reshape(norb,norb,norb,norb)
h1a=h1m[np.ix_(no,no)]; era=e2[np.ix_(no,no,no,no)]
ne=(mol.nelec[0]-N_CORE,mol.nelec[1]-N_CORE)
q_idx=QSpaceIndex(cistring.gen_strings4orblist(range(N_ACT),ne[0]),
                  cistring.gen_strings4orblist(range(N_ACT),ne[1]),
                  N_ACT,ne,h1a,era)
be=KDCIBackend(q_idx)

ef,_=direct_spin1.FCI().kernel(h1a,era,N_ACT,ne,nroots=NROOTS,verbose=0)
e_dmrg=[float(e) for e in np.atleast_1d(ef)[:NROOTS]]

h2_4d=ao2mo.restore('s1',era,N_ACT).reshape(N_ACT,N_ACT,N_ACT,N_ACT)
hm=Hamiltonian(h1=h1a,h2=h2_4d)
hfa,hfb=hf_determinant(*ne); ao=bit_positions(hfa); bo=bit_positions(hfb)
av=[p for p in range(N_ACT) if p not in ao]; bv=[p for p in range(N_ACT) if p not in bo]
E_HF=hm.matrix_element((hfa,hfb),(hfa,hfb)); sc=[]
for i in ao:
    for a in av:
        d=(hfa^(1<<i)|(1<<a),hfb); hij=hm.matrix_element(d,(hfa,hfb)); de=E_HF-hm.matrix_element(d,d)
        if abs(de)>1e-12: sc.append((d,-hij*hij/de))
for i1,i2 in itertools.combinations(ao,2):
    for a1,a2 in itertools.combinations(av,2):
        d=(hfa^(1<<i1)^(1<<i2)|(1<<a1)|(1<<a2),hfb)
        hij=hm.matrix_element(d,(hfa,hfb)); de=E_HF-hm.matrix_element(d,d)
        if abs(de)>1e-12: sc.append((d,-hij*hij/de))
for i in ao:
    for j in bo:
        for a in av:
            for b in bv:
                d=(hfa^(1<<i)|(1<<a),hfb^(1<<j)|(1<<b))
                hij=hm.matrix_element(d,(hfa,hfb)); de=E_HF-hm.matrix_element(d,d)
                if abs(de)>1e-12: sc.append((d,-hij*hij/de))
sc.sort(key=lambda x:x[1],reverse=True); pd=[(hfa,hfb)]
seen=set(); seen.add((int(hfa),int(hfb)))
for det,_ in sc:
    key=(int(det[0]),int(det[1]))
    if key not in seen: seen.add(key); pd.append(det)
    if len(pd)>=P_N: break

H_PP=np.zeros((len(pd),len(pd)))
for i in range(len(pd)):
    for j in range(len(pd)): H_PP[i,j]=hm.matrix_element(pd[i],pd[j])
H_PP=0.5*(H_PP+H_PP.T)
E0_vals,_=np.linalg.eigh(H_PP); E0_vals=E0_vals[:NROOTS]
H_QP=be.build_hqp(pd,verbose=False)

print("A_q = 1/(E_DMRG[k] - H_D), per-state, m=0")
print("{:>6} {:>13} {:>13} {:>12}".format("State","E0(k)","E_DMRG(k)","dE/mH"))
print("-"*48)
for k in range(NROOTS):
    E_exact_k = e_dmrg[k]
    basis,_=be.build_basis(H_QP,E_exact_k,verbose=False)
    H_QQ_t,_=be.build_projected_blocks(basis,pd,H_QP=H_QP,verbose=False)
    H_PQ_t=H_QP.T@basis
    H_eff=build_effective_H(H_PP,H_PQ_t,H_QQ_t,E_exact_k,delta=0.0)
    ev_all,_=diagonalize_effective_H(H_eff,n_states=k+1)
    dE=(ev_all[k]-e_dmrg[k])*1000
    print("  S{:<4} {:>13.8f} {:>13.8f} {:>+12.1f}".format(k,E0_vals[k],E_exact_k,dE), flush=True)
print("DONE")
