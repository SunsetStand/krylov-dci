#!/usr/bin/env python3
"""Step 1: Is a CIS eigenvector a good initial guess for the FCI S1 triplet?
Build CIS space (HF + all singles) in determinant basis, diagonalize,
report energy / <S^2> / overlap with FCI roots."""
import sys, numpy as np
from numpy.linalg import eigh
sys.path.insert(0, '/data/home/wangcx/krylov-dci')
from pyscf import gto, scf, ao2mo
from pyscf.fci import cistring, direct_spin1, spin_op
from src.determinants import hf_determinant, bit_positions
from src.hamiltonian import Hamiltonian

N_ACT=10; N_CORE=2; NROOTS=6; R=1.1; ne=(5,5)
mol=gto.M(atom=f'N 0 0 0; N 0 0 {R}',basis='cc-pVDZ',verbose=0)
mf=scf.RHF(mol).run(verbose=0)
na_o=list(range(N_CORE,N_CORE+N_ACT)); norb=mf.mo_coeff.shape[1]
h1=mf.mo_coeff.T@mf.get_hcore()@mf.mo_coeff
eri=ao2mo.full(mol.intor('int2e'),mf.mo_coeff,compact=False).reshape([norb]*4)
h1a=h1[np.ix_(na_o,na_o)]; era=eri[np.ix_(na_o,na_o,na_o,na_o)]

# FCI reference (ground truth for identification)
cis_f=direct_spin1.FCI(); e_fci,c_fci=cis_f.kernel(h1a,era,N_ACT,ne,nroots=NROOTS,verbose=0)
as_=cistring.gen_strings4orblist(range(N_ACT),ne[0])
bs_=cistring.gen_strings4orblist(range(N_ACT),ne[1])
na,nb=len(as_),len(bs_)
aidx={int(s):i for i,s in enumerate(as_)}; bidx={int(s):i for i,s in enumerate(bs_)}

# Hamiltonian (Slater-Condon)
h2_4d=ao2mo.restore('s1',era,N_ACT).reshape([N_ACT]*4)
ham=Hamiltonian(h1=h1a,h2=h2_4d,E_nuc=0.0,E_HF=0.0)
hf_a,hf_b=hf_determinant(*ne)
occ_a=bit_positions(hf_a); occ_b=bit_positions(hf_b)
virt_a=[p for p in range(N_ACT) if p not in occ_a]
virt_b=[p for p in range(N_ACT) if p not in occ_b]

# CIS determinant space: HF + all single excitations
cis_dets=[(hf_a,hf_b)]
for i in occ_a:
    for a in virt_a: cis_dets.append((hf_a^(1<<i)|(1<<a), hf_b))
for i in occ_b:
    for a in virt_b: cis_dets.append((hf_a, hf_b^(1<<i)|(1<<a)))
ncis=len(cis_dets)
H=np.zeros((ncis,ncis))
for i in range(ncis):
    for j in range(i,ncis):
        v=ham.matrix_element(cis_dets[i],cis_dets[j]); H[i,j]=v; H[j,i]=v
Ecis,Ccis=eigh(H)

def embed(vec):
    full=np.zeros((na,nb))
    for k,(da,db) in enumerate(cis_dets):
        full[aidx[int(da)],bidx[int(db)]]+=vec[k]
    return full

print(f"CIS space: {ncis} dets (HF + {ncis-1} singles)")
print(f"FCI: S0={e_fci[0]:.6f} S1={e_fci[1]:.6f}(T) S2={e_fci[2]:.6f}(T) S3={e_fci[3]:.6f}(S)")
print(f"\n{'CISst':>5} {'E(Ha)':>13} {'exc_eV':>7} {'<S^2>':>6} | overlap with FCI S0..S5")
for k in range(min(7,ncis)):
    fv=embed(Ccis[:,k]); nrm=np.linalg.norm(fv)
    if nrm>0: fv/=nrm
    ss=spin_op.spin_square(fv,N_ACT,ne)[0]
    ovs=[abs(np.vdot(fv.ravel(),c_fci[s].ravel())) for s in range(NROOTS)]
    ovstr=" ".join(f"{o:.3f}" for o in ovs)
    print(f"{k:>5} {Ecis[k]:>13.6f} {(Ecis[k]-Ecis[0])*27.2114:>7.2f} {ss:>6.2f} | {ovstr}")
print("\nGoal: find a CIS excited root with high overlap to FCI S1 -> good initial guess.")
