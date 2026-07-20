#!/usr/bin/env python3
"""Step 2: CIS-seeded, overlap-tracked per-state P-space selection for the S1 triplet.
Tests whether a CIS initial guess + single-excitation seed + state tracking builds a
P-space whose target eigenvalue converges to FCI S1 (capacity test said right 4000 dets -> +0.2 mH)."""
import sys, time, itertools, numpy as np
from numpy.linalg import eigh
sys.path.insert(0,'/data/home/wangcx/krylov-dci')
from pyscf import gto, scf, ao2mo
from pyscf.fci import cistring, direct_spin1, spin_op
from src_mf import QSpaceIndex, KDCIBackend
from src.determinants import hf_determinant, bit_positions
from src.hamiltonian import Hamiltonian

N_ACT=10; N_CORE=2; NROOTS=6; R=1.1; ne=(5,5); BATCH=200; P_MAX=2000
mol=gto.M(atom=f'N 0 0 0; N 0 0 {R}',basis='cc-pVDZ',verbose=0)
mf=scf.RHF(mol).run(verbose=0)
na_o=list(range(N_CORE,N_CORE+N_ACT)); norb=mf.mo_coeff.shape[1]
h1=mf.mo_coeff.T@mf.get_hcore()@mf.mo_coeff
eri=ao2mo.full(mol.intor('int2e'),mf.mo_coeff,compact=False).reshape([norb]*4)
h1a=h1[np.ix_(na_o,na_o)]; era=eri[np.ix_(na_o,na_o,na_o,na_o)]
as_=cistring.gen_strings4orblist(range(N_ACT),ne[0]); bs_=cistring.gen_strings4orblist(range(N_ACT),ne[1])
na,nb=len(as_),len(bs_); M_all=na*nb
aidx={int(s):i for i,s in enumerate(as_)}; bidx={int(s):i for i,s in enumerate(bs_)}
q_idx=QSpaceIndex(as_,bs_,N_ACT,ne,h1a,era); backend=KDCIBackend(q_idx); hdiag=q_idx.hdiag

cisf=direct_spin1.FCI(); e_fci,c_fci=cisf.kernel(h1a,era,N_ACT,ne,nroots=NROOTS,verbose=0)
E_S1=e_fci[1]; print(f"FCI S0={e_fci[0]:.6f}  S1(triplet)={E_S1:.6f}  (exc {(E_S1-e_fci[0])*1000:.1f} mH)")

h2_4d=ao2mo.restore('s1',era,N_ACT).reshape([N_ACT]*4)
ham=Hamiltonian(h1=h1a,h2=h2_4d,E_nuc=0.0,E_HF=0.0)
hf_a,hf_b=hf_determinant(*ne)
occ_a=bit_positions(hf_a); occ_b=bit_positions(hf_b)
va=[p for p in range(N_ACT) if p not in occ_a]; vb=[p for p in range(N_ACT) if p not in occ_b]
full_dets=[(int(a),int(b)) for a in as_ for b in bs_]
det_to_full={d:i for i,d in enumerate(full_dets)}

def flat_idx(d): return aidx[int(d[0])]*nb+bidx[int(d[1])]

# ---- CIS space: HF + all singles ----
cis_dets=[(hf_a,hf_b)]
for i in occ_a:
    for a in va: cis_dets.append((hf_a^(1<<i)|(1<<a),hf_b))
for i in occ_b:
    for a in vb: cis_dets.append((hf_a,hf_b^(1<<i)|(1<<a)))
Hc=np.zeros((len(cis_dets),)*2)
for i in range(len(cis_dets)):
    for j in range(i,len(cis_dets)):
        v=ham.matrix_element(cis_dets[i],cis_dets[j]); Hc[i,j]=v; Hc[j,i]=v
Ec,Cc=eigh(Hc)
# pick CIS root with max overlap to FCI S1 (validation-time identification)
def embed_cis(vec):
    f=np.zeros(M_all)
    for k,d in enumerate(cis_dets): f[flat_idx(d)]+=vec[k]
    return f
best_k,best_ov=None,-1
for k in range(1,len(cis_dets)):
    fv=embed_cis(Cc[:,k]); fv/=np.linalg.norm(fv)
    ov=abs(np.dot(fv,c_fci[1].ravel()))
    if ov>best_ov: best_ov,best_k=ov,k
print(f"CIS guess: root {best_k}, overlap to FCI S1 = {best_ov:.3f}")

# ---- seed P-space = HF + all singles ----
p_dets=list(cis_dets)
p_full_idx=[det_to_full[d] for d in p_dets]; p_set=set(p_full_idx)
def build_hpp(dets):
    n=len(dets); H=np.zeros((n,n))
    for i in range(n):
        for j in range(i,n):
            v=ham.matrix_element(dets[i],dets[j]); H[i,j]=v; H[j,i]=v
    return H
def extend_hpp(Ho,od,nd):
    No=len(od); m=len(nd); Hn=np.zeros((No+m,No+m)); Hn[:No,:No]=Ho
    for il,dn in enumerate(nd):
        r=No+il
        for j in range(No): v=ham.matrix_element(dn,od[j]); Hn[r,j]=v; Hn[j,r]=v
        for jl in range(il+1):
            c=No+jl; v=ham.matrix_element(dn,nd[jl]); Hn[r,c]=v; Hn[c,r]=v
    return Hn
H_PP=build_hpp(p_dets)

# initial target vector (flat, on P) from CIS guess
psi_prev=embed_cis(Cc[:,best_k]); psi_prev/=np.linalg.norm(psi_prev)

print(f"\nseed P={len(p_dets)} (HF+singles)")
print(f"{'iter':>4} {'P':>6} {'k*':>3} {'E_tgt':>13} {'dE_S1_mH':>10} {'<S2>':>5} {'ovlp_track':>10} {'ovlpFCI':>8}")
print("-"*70)
it=0
while len(p_dets)<P_MAX:
    E_P,C_P=eigh(H_PP)
    # state tracking: root whose embedded vector overlaps psi_prev most
    best=-1; kstar=1
    for k in range(min(len(E_P),12)):
        f=np.zeros(M_all)
        for l,gi in enumerate(p_full_idx): f[gi]=C_P[l,k]
        ov=abs(np.dot(f,psi_prev))
        if ov>best: best,kstar=ov,k
    # target vector
    psi=np.zeros(M_all)
    for l,gi in enumerate(p_full_idx): psi[gi]=C_P[l,kstar]
    psi/=np.linalg.norm(psi)
    E_ref=E_P[kstar]
    ss=spin_op.spin_square(psi.reshape(na,nb),N_ACT,ne)[0]
    ov_fci=abs(np.dot(psi,c_fci[1].ravel()))
    dE=(E_P[kstar]-E_S1)*1000
    print(f"{it:>4} {len(p_dets):>6} {kstar:>3} {E_P[kstar]:>13.6f} {dE:>+10.2f} {ss:>5.2f} {best:>10.3f} {ov_fci:>8.3f}",flush=True)
    # score Q dets: |sigma|^2/|E_ref-hdiag|
    sig=backend.sigma(psi)
    w=np.zeros(M_all); asig=np.abs(sig)
    for qi in range(M_all):
        if qi in p_set: continue
        c2=asig[qi]**2
        if c2<1e-24: continue
        den=abs(E_ref-hdiag[qi]); den=max(den,1e-8); w[qi]=c2/den
    cands=[(qi,w[qi]) for qi in range(M_all) if qi not in p_set and w[qi]>0]
    cands.sort(key=lambda x:x[1],reverse=True)
    add=[c[0] for c in cands[:BATCH]]
    nd=[full_dets[qi] for qi in add]
    H_PP=extend_hpp(H_PP,p_dets,nd)
    p_dets.extend(nd); p_full_idx.extend(add); p_set.update(add)
    psi_prev=psi; it+=1

E_P,C_P=eigh(H_PP)
best=-1;kstar=1
for k in range(min(len(E_P),12)):
    f=np.zeros(M_all)
    for l,gi in enumerate(p_full_idx): f[gi]=C_P[l,k]
    ov=abs(np.dot(f,psi_prev))
    if ov>best:best,kstar=ov,k
print(f"\nFINAL P={len(p_dets)}: E_tgt={E_P[kstar]:.6f} dE_S1={(E_P[kstar]-E_S1)*1000:+.2f} mH")
print(f"(compare: OLD ground-state-seeded run stuck at dE_S1 ~ +636 mH)")
