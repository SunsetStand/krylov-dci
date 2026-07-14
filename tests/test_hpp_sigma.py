"""Correctness test: build_hpp_sigma / extend_hpp_sigma  vs  matrix_element reference.
Real N2 CAS(10,10) backend. Run: python tests/test_hpp_sigma.py   (exit 0 = pass)
"""
import sys, itertools
import numpy as np
sys.path.insert(0, '/data/home/wangcx/krylov-dci')
from pyscf import gto, scf, ao2mo
from pyscf.fci import cistring, direct_spin1
from src_mf import QSpaceIndex, KDCIBackend
from src_mf.pspace_ops import build_hpp_sigma, extend_hpp_sigma
from src.determinants import hf_determinant, bit_positions
from src.hamiltonian import Hamiltonian

N_ACT=10; N_CORE=2; R=1.1; ne=(5,5)
mol=gto.M(atom=f'N 0 0 0; N 0 0 {R}',basis='cc-pVDZ',verbose=0)
mf=scf.RHF(mol).run(verbose=0)
na_o=list(range(N_CORE,N_CORE+N_ACT)); norb=mf.mo_coeff.shape[1]
h1=mf.mo_coeff.T@mf.get_hcore()@mf.mo_coeff
eri=ao2mo.full(mol.intor('int2e'),mf.mo_coeff,compact=False).reshape([norb]*4)
h1a=h1[np.ix_(na_o,na_o)]; era=eri[np.ix_(na_o,na_o,na_o,na_o)]
as_=cistring.gen_strings4orblist(range(N_ACT),ne[0]); bs_=cistring.gen_strings4orblist(range(N_ACT),ne[1])
na,nb=len(as_),len(bs_)
aidx={int(s):i for i,s in enumerate(as_)}; bidx={int(s):i for i,s in enumerate(bs_)}
q_idx=QSpaceIndex(as_,bs_,N_ACT,ne,h1a,era); backend=KDCIBackend(q_idx)
h2_4d=ao2mo.restore('s1',era,N_ACT).reshape([N_ACT]*4)
ham=Hamiltonian(h1=h1a,h2=h2_4d,E_nuc=0.0,E_HF=0.0)

# reference builders (Slater-Condon)
def ref_build(dets):
    n=len(dets); H=np.zeros((n,n))
    for i in range(n):
        for j in range(i,n):
            v=ham.matrix_element(dets[i],dets[j]); H[i,j]=v; H[j,i]=v
    return H

hf_a,hf_b=hf_determinant(*ne)
ao=bit_positions(hf_a); bo=bit_positions(hf_b)
av=[p for p in range(N_ACT) if p not in ao]; bv=[p for p in range(N_ACT) if p not in bo]
dets=[(hf_a,hf_b)]
for i in ao:
    for a in av: dets.append((hf_a^(1<<i)|(1<<a),hf_b))
for i in bo:
    for a in bv: dets.append((hf_a,hf_b^(1<<i)|(1<<a)))
for i1,i2 in itertools.combinations(ao,2):     # a few doubles
    for a1,a2 in itertools.combinations(av,2):
        dets.append((hf_a^(1<<i1)^(1<<i2)|(1<<a1)|(1<<a2),hf_b))
        if len(dets)>=90: break
    if len(dets)>=90: break

fails=[]
# 1) full build
H_ref=ref_build(dets)
H_new=build_hpp_sigma(dets,backend,aidx,bidx,na,nb)
d1=np.max(np.abs(H_ref-H_new))
if d1>1e-9: fails.append(f"build_hpp_sigma max|Δ|={d1:.2e}")
print(f"build: max|Δ|={d1:.2e}  (H range [{H_ref.min():.3f},{H_ref.max():.3f}])")

# 2) incremental extend: seed 40, add 25, add 25
seed=dets[:40]; add1=dets[40:65]; add2=dets[65:90]
H0=build_hpp_sigma(seed,backend,aidx,bidx,na,nb)
H1=extend_hpp_sigma(H0,seed,add1,backend,aidx,bidx,na,nb)
H2=extend_hpp_sigma(H1,seed+add1,add2,backend,aidx,bidx,na,nb)
H2ref=ref_build(seed+add1+add2)
d2=np.max(np.abs(H2-H2ref))
if d2>1e-9: fails.append(f"extend_hpp_sigma max|Δ|={d2:.2e}")
print(f"extend(chained): max|Δ|={d2:.2e}")

# 3) eigenvalues match (what actually matters downstream)
ev_ref=np.linalg.eigvalsh(H2ref)[:6]; ev_new=np.linalg.eigvalsh(H2)[:6]
d3=np.max(np.abs(ev_ref-ev_new))
if d3>1e-9: fails.append(f"eigenvalue max|Δ|={d3:.2e}")
print(f"eig(lowest6): max|Δ|={d3:.2e}")

if fails:
    for f in fails: print("FAIL:", f)
    sys.exit(1)
print("PASS: sigma-based H_PP build/extend match Slater-Condon reference to <1e-9.")
