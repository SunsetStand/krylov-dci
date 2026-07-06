#!/usr/bin/env python3
"""Diagnose: why does m=1 degrade even with exact E0?"""
import sys,numpy as np,itertools
sys.path.insert(0,'/data/home/wangcx/krylov-dci')
from src_mf import QSpaceIndex, KDCIBackend
from src.effective_h import build_effective_H, diagonalize_effective_H
from src.determinants import hf_determinant, bit_positions
from src.hamiltonian import Hamiltonian
from pyscf import gto,scf,ao2mo
from pyscf.fci import cistring,direct_spin1

N_CORE,N_ACT,P_N=2,10,400
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
ef,_=direct_spin1.FCI().kernel(h1a,era,N_ACT,ne,nroots=1,verbose=0)
e_exact=float(np.atleast_1d(ef)[0])
h2_4d=ao2mo.restore('s1',era,N_ACT).reshape(N_ACT,N_ACT,N_ACT,N_ACT)
hm=Hamiltonian(h1=h1a,h2=h2_4d)
hfa,hfb=hf_determinant(*ne); ao=bit_positions(hfa); bo=bit_positions(hfb)
av=[p for p in range(N_ACT) if p not in ao]; bv=[p for p in range(N_ACT) if p not in bo]
E_HF=hm.matrix_element((hfa,hfb),(hfa,hfb))
sc=[]
for i in ao:
    for a in av:
        d=(hfa^(1<<i)|(1<<a),hfb); hij=hm.matrix_element(d,(hfa,hfb))
        de=E_HF-hm.matrix_element(d,d)
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
sc.sort(key=lambda x:x[1],reverse=True)
pd=[(hfa,hfb)]
seen=set()
seen.add((int(hfa),int(hfb)))
for det,_ in sc:
    key=(int(det[0]),int(det[1]))
    if key not in seen:
        seen.add(key); pd.append(det)
    if len(pd)>=P_N: break
print(f"P-space: {len(pd)} dets", flush=True)

H_PP=np.zeros((len(pd),len(pd)))
for i in range(len(pd)):
    for j in range(len(pd)): H_PP[i,j]=hm.matrix_element(pd[i],pd[j])
H_PP=0.5*(H_PP+H_PP.T)
E0_vals,_=np.linalg.eigh(H_PP); E0_P=float(E0_vals[0])
H_QP=be.build_hqp(pd,verbose=False)
print(f"E0_P={E0_P:.8f}, off {(e_exact-E0_P)*1000:.0f} mH", flush=True)

# Compare m=0 vs m=1 correction matrices
for label,E0 in [("E0_P",E0_P),("E_exact",e_exact)]:
    print(f"\n--- {label} ---", flush=True)
    basis0,_=be.build_basis(H_QP,E0,verbose=False)
    d0=basis0.shape[1]
    H_QQ0,_=be.build_projected_blocks(basis0,pd,H_QP=H_QP,verbose=False)
    H_PQ0=H_QP.T@basis0
    H_eff0=build_effective_H(H_PP,H_PQ0,H_QQ0,E0,delta=0.0)
    ev0,_=diagonalize_effective_H(H_eff0,n_states=1)
    corr0=H_PQ0@np.linalg.inv(E0*np.eye(d0)-H_QQ0)@H_PQ0.T
    print(f"  m=0: dE={(ev0[0]-e_exact)*1000:+.1f} mH, |corr|_avg={np.mean(np.abs(corr0)):.4f}", flush=True)

    print("  propagating...", flush=True)
    basis1,_=be.propagate_basis(basis0,E0,delta=0.0,verbose=False)
    d1=basis1.shape[1]
    H_QQ1,_=be.build_projected_blocks(basis1,pd,H_QP=H_QP,verbose=False)
    H_PQ1=H_QP.T@basis1
    H_eff1=build_effective_H(H_PP,H_PQ1,H_QQ1,E0,delta=0.0)
    ev1,_=diagonalize_effective_H(H_eff1,n_states=1)
    corr1=H_PQ1@np.linalg.inv(E0*np.eye(d1)-H_QQ1)@H_PQ1.T
    print(f"  m=1: dE={(ev1[0]-e_exact)*1000:+.1f} mH, |corr|_avg={np.mean(np.abs(corr1)):.4f}", flush=True)
    print(f"  |corr1-corr0|_max: {np.max(np.abs(corr1-corr0)):.4f}", flush=True)
print("\nDONE")
