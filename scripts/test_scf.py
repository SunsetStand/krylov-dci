#!/usr/bin/env python3
"""Self-consistent Krylov-dCI: NO damping, pure E_cur = E_new."""
import sys,numpy as np,itertools,time
sys.path.insert(0,'/data/home/wangcx/krylov-dci')
from src_mf import QSpaceIndex, KDCIBackend
from src.effective_h import build_effective_H, diagonalize_effective_H
from src.determinants import hf_determinant, bit_positions
from src.hamiltonian import Hamiltonian
from pyscf import gto,scf,ao2mo
from pyscf.fci import cistring,direct_spin1

N_CORE,N_ACT,P_N,NROOTS=2,10,400,3  # just 3 states for speed
THR=1e-8; MAX_ITER=20

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
print("H_QP ready. NO DAMPING.", flush=True)

results=[]
for k in range(NROOTS):
    E_cur=E0_vals[k]
    print("\n--- State %d ---" % k, flush=True)
    print("  iter  E_cur(Ha)        dE(mH)   wall  basis_d", flush=True)
    for it in range(MAX_ITER):
        tl=time.perf_counter()
        basis,_=be.build_basis(H_QP,E_cur,verbose=False)
        H_QQ_t,_=be.build_projected_blocks(basis,pd,H_QP=H_QP,verbose=False)
        H_PQ_t=H_QP.T@basis
        H_eff=build_effective_H(H_PP,H_PQ_t,H_QQ_t,E_cur,delta=0.0)
        ev_all,_=diagonalize_effective_H(H_eff,n_states=k+1)
        E_new=float(ev_all[k])
        wall=time.perf_counter()-tl
        dE=(E_new-e_dmrg[k])*1000
        print("  %4d  %.10f  %+8.2f  %5.1f  %4d"%(it,E_cur,dE,wall,basis.shape[1]),flush=True)
        if abs(E_new-E_cur)<THR:
            print("  Converged!",flush=True)
            break
        E_cur=E_new  # NO DAMPING
    results.append({'state':k,'E':E_cur,'n_iter':it+1,'dE_mH':(E_cur-e_dmrg[k])*1000})

print("\n"+"="*60)
print("SCF Results (NO DAMPING)")
for r in results:
    print("  S%d: dE=%+.1f mH, n_iter=%d"%(r['state'],r['dE_mH'],r['n_iter']))
print("DONE")
