#!/usr/bin/env python3
"""Diagnostic: characterize FCI excited states of N2 CAS(10,10) — spin & excitation character."""
import numpy as np
from pyscf import gto, scf, ao2mo, fci
from pyscf.fci import cistring, direct_spin1, spin_op

N_ACT=10; N_CORE=2; NROOTS=6; R=1.1; ne=(5,5)
mol = gto.M(atom=f'N 0 0 0; N 0 0 {R}', basis='cc-pVDZ', verbose=0)
mf = scf.RHF(mol).run(verbose=0)
na_o=list(range(N_CORE,N_CORE+N_ACT))
norb=mf.mo_coeff.shape[1]
h1=mf.mo_coeff.T@mf.get_hcore()@mf.mo_coeff
eri=ao2mo.full(mol.intor('int2e'),mf.mo_coeff,compact=False).reshape([norb]*4)
h1a=h1[np.ix_(na_o,na_o)]; era=eri[np.ix_(na_o,na_o,na_o,na_o)]

cis=direct_spin1.FCI()
e,c=cis.kernel(h1a,era,N_ACT,ne,nroots=NROOTS,verbose=0)
print(f"{'St':>3} {'E(Ha)':>16} {'exc(mH)':>9} {'exc(eV)':>8} {'<S^2>':>7} {'2S+1':>5}  top-dets")
as_=cistring.gen_strings4orblist(range(N_ACT),ne[0])
bs_=cistring.gen_strings4orblist(range(N_ACT),ne[1])
na,nb=len(as_),len(bs_)
for i in range(NROOTS):
    ci=c[i]
    ss=spin_op.spin_square(ci,N_ACT,ne)[0]
    mult=np.sqrt(4*ss+1)  # 2S+1 = sqrt(4S(S+1)+1)
    exc=(e[i]-e[0])*1000
    # top determinants
    flat=ci.reshape(-1)
    idx=np.argsort(-flat**2)[:3]
    tops=[]
    for k in idx:
        ia,ib=divmod(k,nb)
        abits=bin(as_[ia])[2:].zfill(N_ACT)[::-1]
        bbits=bin(bs_[ib])[2:].zfill(N_ACT)[::-1]
        tops.append(f"|{abits}|{bbits}|={flat[k]:+.3f}")
    print(f"{i:>3} {e[i]:>16.8f} {exc:>9.1f} {exc*27.2114/1000:>8.2f} {ss:>7.3f} {mult:>5.2f}  "+"  ".join(tops))

# HF determinant reference
hf_a=int('1'*ne[0],2); print(f"\nHF det (alpha closed) = |{bin(hf_a)[2:].zfill(N_ACT)[::-1]}| occ first {ne[0]} orbs")
print("Note: bits left->right = active orbital 0..9 (orb0=lowest). '1'=occupied.")
