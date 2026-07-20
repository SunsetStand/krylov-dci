#!/usr/bin/env python3
"""Discriminator: is 4000 dets enough for the S1 triplet? Rayleigh quotient of truncated FCI vectors."""
import numpy as np
from pyscf import gto, scf, ao2mo
from pyscf.fci import cistring, direct_spin1

N_ACT=10; N_CORE=2; NROOTS=6; R=1.1; ne=(5,5)
mol=gto.M(atom=f'N 0 0 0; N 0 0 {R}',basis='cc-pVDZ',verbose=0)
mf=scf.RHF(mol).run(verbose=0)
na_o=list(range(N_CORE,N_CORE+N_ACT)); norb=mf.mo_coeff.shape[1]
h1=mf.mo_coeff.T@mf.get_hcore()@mf.mo_coeff
eri=ao2mo.full(mol.intor('int2e'),mf.mo_coeff,compact=False).reshape([norb]*4)
h1a=h1[np.ix_(na_o,na_o)]; era=eri[np.ix_(na_o,na_o,na_o,na_o)]
cis=direct_spin1.FCI(); e,c=cis.kernel(h1a,era,N_ACT,ne,nroots=NROOTS,verbose=0)
h2e=cis.absorb_h1e(h1a,era,N_ACT,ne,0.5)
def rq(vec):
    hv=cis.contract_2e(h2e,vec.reshape(c[0].shape),N_ACT,ne).ravel()
    v=vec.ravel(); return v@hv/(v@v)
Ns=[200,500,1000,2000,4000,8000,16000]
print(f"FCI: S0={e[0]:.6f}  S1={e[1]:.6f} (triplet)  S3={e[3]:.6f} (singlet)")
print(f"\n{'N':>6} | {'S0 RQ':>12} {'ΔS0mH':>8} | {'S1 RQ':>12} {'ΔS1mH':>8} | {'cumw_S1':>9}")
for st,lbl in [(0,'S0'),(1,'S1')]:
    pass
for N in Ns:
    row=f"{N:>6} |"
    for st in (0,1):
        flat=c[st].ravel(); idx=np.argsort(-flat**2)
        keep=idx[:N]; tr=np.zeros_like(flat); tr[keep]=flat[keep]
        cumw=np.sum(tr**2)/np.sum(flat**2)
        E=rq(tr); dE=(E-e[st])*1000
        row+=f" {E:>12.6f} {dE:>+8.1f} |" if st==0 else f" {E:>12.6f} {dE:>+8.1f} | {cumw:>9.5f}"
    print(row)
print("\nRQ = <ψ_trunc|H|ψ_trunc>/<ψ_trunc|ψ_trunc>, upper bound to true energy.")
print("If S1 RQ approaches FCI S1 at N~4000 -> 4000 dets suffice, problem is SELECTION not capacity.")
