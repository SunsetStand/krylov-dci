#!/usr/bin/env python3
"""Test Neumann expansion H^eff vs matrix-inverse H^eff for N2/cc-pVDZ."""
import sys,numpy as np,itertools
sys.path.insert(0,'/data/home/wangcx/krylov-dci')
from src_mf import QSpaceIndex, KDCIBackend
from src.effective_h import build_effective_H, build_effective_H_neumann, diagonalize_effective_H
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

ef,_=direct_spin1.FCI().kernel(h1a,era,N_ACT,ne,nroots=1,verbose=0); e_exact=float(np.atleast_1d(ef)[0])
h2_4d=ao2mo.restore('s1',era,N_ACT).reshape(N_ACT,N_ACT,N_ACT,N_ACT)
hm=Hamiltonian(h1=h1a,h2=h2_4d)
hfa,hfb=hf_determinant(*ne); ao=bit_positions(hfa); bo=bit_positions(hfb)
av=[p for p in range(N_ACT) if p not in ao]; bv=[p for p in range(N_ACT) if p not in bo]
E_HF=hm.matrix_element((hfa,hfb),(hfa,hfb))
sc=[]
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
sc.sort(key=lambda x:x[1],reverse=True)
pd=[(hfa,hfb)]
seen=set(); seen.add((int(hfa),int(hfb)))
for det,_ in sc:
    key=(int(det[0]),int(det[1]))
    if key not in seen: seen.add(key); pd.append(det)
    if len(pd)>=P_N: break
print(f"P={len(pd)}", flush=True)

H_PP=np.zeros((len(pd),len(pd)))
for i in range(len(pd)):
    for j in range(len(pd)): H_PP[i,j]=hm.matrix_element(pd[i],pd[j])
H_PP=0.5*(H_PP+H_PP.T)
E0_vals,_=np.linalg.eigh(H_PP); E0_P=float(E0_vals[0])
H_QP=be.build_hqp(pd,verbose=False)
print(f"E0_P={E0_P:.8f}, off {(e_exact-E0_P)*1000:.0f} mH", flush=True)

# Test both E0_P and E_exact, m=0 and m=1
for label,E0 in [("E0_P",E0_P),("E_exact",e_exact)]:
    print(f"\n=== {label} ===", flush=True)
    basis0,_=be.build_basis(H_QP,E0,verbose=False)
    H_QQ0,_=be.build_projected_blocks(basis0,pd,H_QP=H_QP,verbose=False)

    # Old method: matrix inverse
    H_PQ_t=H_QP.T@basis0
    H_eff_old=build_effective_H(H_PP,H_PQ_t,H_QQ0,E0,delta=0.0)
    ev_old,_=diagonalize_effective_H(H_eff_old,n_states=1)
    print(f"  old m=0: dE={(ev_old[0]-e_exact)*1000:+.1f} mH", flush=True)

    # Neumann m=0
    H_eff_n0=build_effective_H_neumann(H_PP,H_PQ_t,H_QQ0,basis0,q_idx.hdiag,H_QP,E0,m_order=0)
    ev_n0,_=diagonalize_effective_H(H_eff_n0,n_states=1)
    print(f"  neu m=0: dE={(ev_n0[0]-e_exact)*1000:+.1f} mH", flush=True)

    # Propagate to m=1
    basis1,_=be.propagate_basis(basis0,E0,delta=0.0,verbose=False)
    H_QQ1,_=be.build_projected_blocks(basis1,pd,H_QP=H_QP,verbose=False)

    # Old method: matrix inverse with 800 basis vectors
    H_PQ1=H_QP.T@basis1
    H_eff_old1=build_effective_H(H_PP,H_PQ1,H_QQ1,E0,delta=0.0)
    ev_old1,_=diagonalize_effective_H(H_eff_old1,n_states=1)
    print(f"  old m=1: dE={(ev_old1[0]-e_exact)*1000:+.1f} mH", flush=True)

    # Neumann m=0,1 (using m=1 basis)
    H_eff_n1=build_effective_H_neumann(H_PP,H_PQ1,H_QQ1,basis1,q_idx.hdiag,H_QP,E0,m_order=1)
    ev_n1,_=diagonalize_effective_H(H_eff_n1,n_states=1)
    print(f"  neu m=1: dE={(ev_n1[0]-e_exact)*1000:+.1f} mH", flush=True)

    # Neumann m=0 (using m=1 basis but order 0)
    H_eff_n1m0=build_effective_H_neumann(H_PP,H_PQ1,H_QQ1,basis1,q_idx.hdiag,H_QP,E0,m_order=0)
    ev_n1m0,_=diagonalize_effective_H(H_eff_n1m0,n_states=1)
    print(f"  neu m=1 basis, order 0: dE={(ev_n1m0[0]-e_exact)*1000:+.1f} mH", flush=True)

print("\nDONE")
