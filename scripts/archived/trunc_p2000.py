import sys,os,time,json,itertools,gc,numpy as np
from numpy.linalg import eigh,svd,norm
sys.path.insert(0,"/data/home/wangcx/krylov-dci")
from src_mf import QSpaceIndex,KDCIBackend,KDCISparse
from src_mf.pspace_ops import embed_pspace_vec,build_pmask,score_and_select,build_hpp_sigma,extend_hpp_sigma
from src.effective_h import build_effective_H,diagonalize_effective_H
from src.determinants import hf_determinant,bit_positions
from src.hamiltonian import Hamiltonian
from pyscf import gto,scf,ao2mo; from pyscf.fci import cistring,direct_spin1,spin_op

T=800;M=1;B=200;NA=10;NC=2;NR=6;R=1.1;ne=(5,5)
t0=time.perf_counter()
mol=gto.M(atom=f"N 0 0 0; N 0 0 {R}",basis="cc-pVDZ",verbose=0,spin=0)
mf=scf.RHF(mol).run(verbose=0)
cas=list(range(NC,NC+NA));mc=mf.CASSCF(NA,sum(ne));mc.fix_spin_(ss=0)
mo=mc.sort_mo(cas,base=0)
h1=mo.T@mf.get_hcore()@mo;h1=h1[NC:NC+NA,NC:NC+NA]
era=ao2mo.kernel(mol,mo[:,NC:NC+NA],aosym="s4");h1a=h1.copy();h1b=h1.copy()
as_=cistring.gen_strings4orblist(range(NA),ne[0]);bs_=cistring.gen_strings4orblist(range(NA),ne[1])
na,nb=len(as_),len(bs_);M_all=na*nb
aidx={int(s):i for i,s in enumerate(as_)};bidx={int(s):i for i,s in enumerate(bs_)}
q_idx=QSpaceIndex(as_,bs_,NA,ne,h1a,era);backend=KDCIBackend(q_idx);kdci=KDCISparse(q_idx)
hdiag=q_idx.hdiag
ef,_=direct_spin1.FCI().kernel(h1a,era,NA,ne,nroots=NR,verbose=0)
e_fci=[float(e) for e in np.atleast_1d(ef)[:NR]]
print(f"FCI S0={e_fci[0]:.8f}",flush=True)
h2=ao2mo.restore("s1",era,NA).reshape(NA,NA,NA,NA)
ham=Hamiltonian(h1=h1a,h2=h2,E_nuc=0.0,E_HF=0.0)
hf_a,hf_b=hf_determinant(*ne);ao=bit_positions(hf_a);bo=bit_positions(hf_b)
av=[p for p in range(NA) if p not in ao];bv=[p for p in range(NA) if p not in bo]
E_HF=ham.matrix_element((hf_a,hf_b),(hf_a,hf_b))
fd=[(int(a),int(b)) for a in as_ for b in bs_];dtf={d:i for i,d in enumerate(fd)}
P_INIT=200;init=[(hf_a,hf_b)]
sgls=[]
for i in ao:
    for a in av:sgls.append((hf_a^(1<<i)|(1<<a),hf_b))
for i in bo:
    for a in bv:sgls.append((hf_a,hf_b^(1<<i)|(1<<a)))
for d in sgls:
    if d not in init:init.append(d)
sc=[]
for i1,i2 in itertools.combinations(ao,2):
    for a1,a2 in itertools.combinations(av,2):
        d=(hf_a^(1<<i1)^(1<<i2)|(1<<a1)|(1<<a2),hf_b)
        hij=ham.matrix_element(d,(hf_a,hf_b));den=E_HF-ham.matrix_element(d,d)
        if abs(den)>1e-12:sc.append((d,-hij*hij/den))
for i1,i2 in itertools.combinations(bo,2):
    for a1,a2 in itertools.combinations(bv,2):
        d=(hf_a,hf_b^(1<<i1)^(1<<i2)|(1<<a1)|(1<<a2))
        hij=ham.matrix_element(d,(hf_a,hf_b));den=E_HF-ham.matrix_element(d,d)
        if abs(den)>1e-12:sc.append((d,-hij*hij/den))
for i in ao:
    for j in bo:
        for a in av:
            for b in bv:
                d=(hf_a^(1<<i)|(1<<a),hf_b^(1<<j)|(1<<b))
                hij=ham.matrix_element(d,(hf_a,hf_b));den=E_HF-ham.matrix_element(d,d)
                if abs(den)>1e-12:sc.append((d,-hij*hij/den))
ts=None;n_singles=len(init)-1
for d,scr in sc:
    if len(init)>=P_INIT:
        if ts is not None and abs(scr-ts)>1e-12:break
    if d not in init:init.append(d)
    if len(init)==P_INIT:ts=scr
print(f"Seed P={len(init)} (singles={n_singles} doubles={len(init)-1-n_singles})",flush=True)
pd=list(init);pfi=[dtf[d] for d in pd];ps=set(pfi)
H_PP=build_hpp_sigma(pd,backend,aidx,bidx,na,nb)
N_p=len(pd);SR=list(range(min(NR,5)))
while N_p<2000:
    E_P,C_P=eigh(H_PP);sigmas=[]
    for sk in range(min(len(SR),N_p)):
        k=SR[sk];vec=embed_pspace_vec(C_P[:,k],pfi,M_all)
        sigmas.append((E_P[k],backend.sigma(vec)))
    pm=build_pmask(ps,M_all);sel,_,_=score_and_select(sigmas,hdiag,pm,B)
    ng=[int(qi) for qi in sel];nd=[fd[qi] for qi in ng]
    H_PP=extend_hpp_sigma(H_PP,pd,nd,backend,aidx,bidx,na,nb)
    pd.extend(nd);pfi.extend(ng);ps.update(ng);N_p=len(pd)
    print(f"  P={N_p}",flush=True)
print(f"Krylov at P={N_p}",flush=True)
pis=set()
for pa,pb in pd:
    idx=q_idx.flat_index(int(pa),int(pb))
    if idx is not None and idx>=0:pis.add(idx)
E0_vals,_=eigh(H_PP);E0=E0_vals[0]
A_q=np.where(np.abs(E0-hdiag)>1e-10,1.0/(E0-hdiag),0.0)
fp="/data/home/wangcx/krylov-dci/tmp/p2000_build.dat"
T=np.memmap(fp,dtype="float64",mode="w+",shape=(M_all,N_p))
for p in range(N_p):
    pa,pb=int(pd[p][0]),int(pd[p][1])
    ia=q_idx._alpha_idx.get(pa);ib=q_idx._beta_idx.get(pb)
    if ia is None or ib is None:continue
    ci=np.zeros((na,nb));ci[ia,ib]=1.0
    sf=backend.sigma_full(ci).reshape(-1)
    for q in pis:sf[q]=0.0;A_half=np.sqrt(np.abs(A_q))
    T[:,p]=A_half*sf
T.flush();U0,s0r,_=svd(T,full_matrices=False)
d0=int(np.sum(s0r>=1e-3*s0r[0]));U0=U0[:,:d0];s0=s0r[:d0]/s0r[0]
del T;gc.collect();os.unlink(fp)
print(f"  m=0: d={d0}",flush=True)
fpp="/data/home/wangcx/krylov-dci/tmp/p2000_prop.dat"
Tp=np.memmap(fpp,dtype="float64",mode="w+",shape=(M_all,d0))
for k in range(d0):
    sk=backend.sigma_full(U0[:,k].reshape(na,nb)).reshape(-1)
    rs=sk-hdiag*U0[:,k]
    for q in pis:rs[q]=0.0
    Tp[:,k]=A_half*rs;A_half=np.sqrt(np.abs(A_q))
Tp.flush();Un,spr,_=svd(Tp,full_matrices=False)
dn=int(np.sum(spr>=1e-3*spr[0]));Ui=Un[:,:dn];sp=spr[:dn]/spr[0]
Ui-=U0@(U0.T@Ui)
nrm=np.sqrt(np.sum(Ui**2,axis=0));val=nrm>1e-12
Ui=Ui[:,val];dn=int(np.sum(val));sp=sp[val]
U1=np.hstack([U0,Ui]);d1=U1.shape[1]
sc=np.concatenate([s0,sp[:d1-d0]])
del Tp;gc.collect();os.unlink(fpp)
print(f"  m=1: d={d1}",flush=True)
si=np.argsort(-sc[:d1]);U1=U1[:,si];sc=sc[si];d1=U1.shape[1]
# Zero P rows
for q in pis:U1[q,:]=0.0
print(f"Sigma {d1} cols...",flush=True)
SIG=np.zeros((M_all,d1))
for k in range(d1):
    SIG[:,k]=backend.sigma_full(U1[:,k].reshape(na,nb)).reshape(-1)
    if (k+1)%max(1,d1//5)==0:print(f"  {k+1}/{d1}",flush=True)
def peff(Hpp,Hpk,Hkk,erefs,nroots):
    ev_out=np.zeros(len(erefs))
    for k,Ek in enumerate(erefs[:nroots]):
        evk=np.asarray(diagonalize_effective_H(build_effective_H(Hpp,Hpk,Hkk,float(Ek),delta=0.0),n_states=nroots)[0])
        ev_out[k]=evk[int(np.argmin(np.abs(evk-Ek)))]
    return ev_out
pf=kdci.q_idx.p_indices(pd);pv=pf>=0;pff=pf[pv];Er=E0_vals[:NR]
# Baseline
Hkk=U1.T@SIG;Hkk=0.5*(Hkk+Hkk.T);Hpk=np.zeros((N_p,d1));Hpk[pv,:]=SIG[pff,:]
ev_full=peff(H_PP,Hpk,Hkk,Er,NR)
print(f"\nBaseline m=1:",flush=True)
for k in range(NR):print(f"  S{k}: {(ev_full[k]-e_fci[k])*1000:+.1f} mH",flush=True)
thrs=[1e-3,5e-3,1e-2,5e-2,1e-1,2e-1,5e-1]
for thr in thrs:
    r=int(np.sum(sc>=thr))
    if r==0:continue
    Ur=U1[:,:r];SIGr=SIG[:,:r]
    Hkkr=Ur.T@SIGr;Hkkr=0.5*(Hkkr+Hkkr.T)
    Hpkr=np.zeros((N_p,r));Hpkr[pv,:]=SIGr[pff,:]
    ev=peff(H_PP,Hpkr,Hkkr,Er,NR)
    dE=[(ev[k]-e_fci[k])*1000 for k in range(min(NR,len(ev)))]
    cp=100*(1-r/d1)
    print(f"  {thr:.0e} {r:>5} {cp:>6.1f}% {dE[0]:>+9.1f} {dE[1]:>+9.1f} {dE[2]:>+9.1f} {dE[3]:>+9.1f}",flush=True)
print(f"\nDone: {time.perf_counter()-t0:.0f}s",flush=True)
