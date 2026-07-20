import sys,os,time,json,itertools,gc
import numpy as np
from numpy.linalg import eigh,svd,norm
sys.path.insert(0,"/data/home/wangcx/krylov-dci")
from src_mf import QSpaceIndex,KDCIBackend,KDCISparse
from src_mf.pspace_ops import embed_pspace_vec,build_pmask,score_and_select,build_hpp_sigma,extend_hpp_sigma
from src.effective_h import build_effective_H,diagonalize_effective_H
from src.determinants import hf_determinant,bit_positions
from src.hamiltonian import Hamiltonian
from pyscf import gto,scf,ao2mo; from pyscf.fci import cistring,direct_spin1,spin_op

TARGET_P=800;M_MAX=1;BATCH=200;N_ACT=10;N_CORE=2;NROOTS=6;R=1.1;ne=(5,5)
t0=time.perf_counter()
mol=gto.M(atom=f"N 0 0 0; N 0 0 {R}",basis="cc-pVDZ",verbose=0,spin=0)
mf=scf.RHF(mol).run(verbose=0)
cas_list=list(range(N_CORE,N_CORE+N_ACT))
mc=mf.CASSCF(N_ACT,sum(ne));mc.fix_spin_(ss=0)
mo=mc.sort_mo(cas_list,base=0)
h1=mo.T@mf.get_hcore()@mo;h1=h1[N_CORE:N_CORE+N_ACT,N_CORE:N_CORE+N_ACT]
era=ao2mo.kernel(mol,mo[:,N_CORE:N_CORE+N_ACT],aosym="s4")
h1a=h1.copy();h1b=h1.copy()
as_=cistring.gen_strings4orblist(range(N_ACT),ne[0])
bs_=cistring.gen_strings4orblist(range(N_ACT),ne[1])
na,nb=len(as_),len(bs_);M_all=na*nb
aidx={int(s):i for i,s in enumerate(as_)}
bidx={int(s):i for i,s in enumerate(bs_)}
q_idx=QSpaceIndex(as_,bs_,N_ACT,ne,h1a,era)
backend=KDCIBackend(q_idx);kdci_sparse=KDCISparse(q_idx)
hdiag=q_idx.hdiag
ef,_=direct_spin1.FCI().kernel(h1a,era,N_ACT,ne,nroots=NROOTS,verbose=0)
e_fci=[float(e) for e in np.atleast_1d(ef)[:NROOTS]]
print(f"FCI S0={e_fci[0]:.8f}",flush=True)

h2_4d=ao2mo.restore("s1",era,N_ACT).reshape(N_ACT,N_ACT,N_ACT,N_ACT)
ham=Hamiltonian(h1=h1a,h2=h2_4d,E_nuc=0.0,E_HF=0.0)
hf_a,hf_b=hf_determinant(*ne)
ao=bit_positions(hf_a);bo=bit_positions(hf_b)
av,bv=[p for p in range(N_ACT) if p not in ao],[p for p in range(N_ACT) if p not in bo]
E_HF=ham.matrix_element((hf_a,hf_b),(hf_a,hf_b))
full_dets=[(int(a),int(b)) for a in as_ for b in bs_]
det_to_full={d:i for i,d in enumerate(full_dets)}

# CIS seed
P_INIT=200
init_dets=[(hf_a,hf_b)]
singles=[]
for i in ao:
    for a in av:singles.append((hf_a^(1<<i)|(1<<a),hf_b))
for i in bo:
    for a in bv:singles.append((hf_a,hf_b^(1<<i)|(1<<a)))
for d in singles:
    if d not in init_dets:init_dets.append(d)
scores=[]
for i1,i2 in itertools.combinations(ao,2):
    for a1,a2 in itertools.combinations(av,2):
        d=(hf_a^(1<<i1)^(1<<i2)|(1<<a1)|(1<<a2),hf_b)
        hij=ham.matrix_element(d,(hf_a,hf_b));den=E_HF-ham.matrix_element(d,d)
        if abs(den)>1e-12:scores.append((d,-hij*hij/den))
for i1,i2 in itertools.combinations(bo,2):
    for a1,a2 in itertools.combinations(bv,2):
        d=(hf_a,hf_b^(1<<i1)^(1<<i2)|(1<<a1)|(1<<a2))
        hij=ham.matrix_element(d,(hf_a,hf_b));den=E_HF-ham.matrix_element(d,d)
        if abs(den)>1e-12:scores.append((d,-hij*hij/den))
for i in ao:
    for j in bo:
        for a in av:
            for b in bv:
                d=(hf_a^(1<<i)|(1<<a),hf_b^(1<<j)|(1<<b))
                hij=ham.matrix_element(d,(hf_a,hf_b));den=E_HF-ham.matrix_element(d,d)
                if abs(den)>1e-12:scores.append((d,-hij*hij/den))
scores.sort(key=lambda x:x[1],reverse=True)
for d,_ in scores:
    if len(init_dets)>=P_INIT:break
    if d not in init_dets:init_dets.append(d)
print(f"Seed P={len(init_dets)}",flush=True)

# Iterative P
p_dets=list(init_dets);p_full_idx=[det_to_full[d] for d in p_dets];p_set=set(p_full_idx)
H_PP=build_hpp_sigma(p_dets,backend,aidx,bidx,na,nb)
N_p=len(p_dets);SCORING_ROOTS=list(range(min(NROOTS,5)))
while N_p<TARGET_P:
    E_P,C_P=eigh(H_PP)
    sigmas=[]
    for sk in range(min(len(SCORING_ROOTS),N_p)):
        k=SCORING_ROOTS[sk]
        vec=embed_pspace_vec(C_P[:,k],p_full_idx,M_all)
        sigmas.append((E_P[k],backend.sigma(vec)))
    p_mask=build_pmask(p_set,M_all)
    sel,max_w,_=score_and_select(sigmas,hdiag,p_mask,BATCH)
    new_gi=[int(qi) for qi in sel]
    new_dets=[full_dets[qi] for qi in new_gi]
    H_PP=extend_hpp_sigma(H_PP,p_dets,new_dets,backend,aidx,bidx,na,nb)
    p_dets.extend(new_dets);p_full_idx.extend(new_gi);p_set.update(new_gi)
    N_p=len(p_dets)
    print(f"  P={N_p}",flush=True)

print(f"Krylov at P={N_p}",flush=True)
p_idx_set=set()
for pa,pb in p_dets:
    idx=q_idx.flat_index(int(pa),int(pb))
    if idx is not None and idx>=0:p_idx_set.add(idx)
E0_vals,E0_vecs=eigh(H_PP);E0=E0_vals[0]
dE0_bare=(E0-e_fci[0])*1000
print(f"  E0={E0:.8f} dE0(bare)={dE0_bare:+.1f} mH",flush=True)

A_q=np.where(np.abs(E0-hdiag)>1e-10,1.0/(E0-hdiag),0.0)
# build
fpath=f"/data/home/wangcx/krylov-dci/tmp/clean_build.dat"
T=np.memmap(fpath,dtype="float64",mode="w+",shape=(M_all,N_p))
for p in range(N_p):
    pa,pb=int(p_dets[p][0]),int(p_dets[p][1])
    ia=q_idx._alpha_idx.get(pa);ib=q_idx._beta_idx.get(pb)
    if ia is None or ib is None:continue
    ci=np.zeros((na,nb));ci[ia,ib]=1.0
    sigma_flat=backend.sigma_full(ci).reshape(-1)
    for q in p_idx_set:sigma_flat[q]=0.0
    T[:,p]=A_q*sigma_flat
T.flush()
U0,s0_raw,_=svd(T,full_matrices=False)
d_0=int(np.sum(s0_raw>=1e-3*s0_raw[0]))
U0=U0[:,:d_0];s0=s0_raw[:d_0]/s0_raw[0]
del T;gc.collect();os.unlink(fpath)
print(f"  m=0: d={d_0}",flush=True)

# propagate
fpath_p=f"/data/home/wangcx/krylov-dci/tmp/clean_prop.dat"
Tp=np.memmap(fpath_p,dtype="float64",mode="w+",shape=(M_all,d_0))
for k in range(d_0):
    sig_k=backend.sigma_full(U0[:,k].reshape(na,nb)).reshape(-1)
    residual=sig_k-hdiag*U0[:,k]
    Tp[:,k]=A_q*residual
Tp.flush()
Un,sp_raw,_=svd(Tp,full_matrices=False)
d_new=int(np.sum(sp_raw>=1e-3*sp_raw[0]))
Ui=Un[:,:d_new];sp=sp_raw[:d_new]/sp_raw[0]
Ui-=U0@(U0.T@Ui)
nrm=np.sqrt(np.sum(Ui**2,axis=0));val=nrm>1e-12
Ui=Ui[:,val];d_new=int(np.sum(val));sp=sp[val]
U1=np.hstack([U0,Ui]);d1=U1.shape[1]
sigma_comb=np.concatenate([s0,sp[:d1-d_0]])
del Tp;gc.collect();os.unlink(fpath_p)
print(f"  m=1: d={d1}",flush=True)

# Sort
# Zero P-space rows in U1 for pure Q-space Krylov basis
for q in p_idx_set: U1[q, :] = 0.0
sort_idx=np.argsort(-sigma_comb[:d1])
U1=U1[:,sort_idx];sigma_comb=sigma_comb[sort_idx];d1=U1.shape[1]

# Sigma
print(f"Sigma {d1} cols...",flush=True)
SIG=np.zeros((M_all,d1))
for k in range(d1):
    SIG[:,k]=backend.sigma_full(U1[:,k].reshape(na,nb)).reshape(-1)
    if (k+1)%max(1,d1//5)==0:print(f"  {k+1}/{d1}",flush=True)

# Per-state eff
def perstate_eff(Hpp,Hpk,Hkk,erefs,nroots):
    ev_out=np.zeros(len(erefs))
    for k,Ek in enumerate(erefs[:nroots]):
        evk=np.asarray(diagonalize_effective_H(build_effective_H(Hpp,Hpk,Hkk,float(Ek),delta=0.0),n_states=nroots)[0])
        ev_out[k]=evk[int(np.argmin(np.abs(evk-Ek)))]
    return ev_out

p_flat=kdci_sparse.q_idx.p_indices(p_dets)
p_valid=p_flat>=0;p_f=p_flat[p_valid]
E_refs=E0_vals[:NROOTS]

# Baseline
Hkk=U1.T@SIG;Hkk=0.5*(Hkk+Hkk.T)
Hpk=np.zeros((N_p,d1));Hpk[p_valid,:]=SIG[p_f,:]
ev_full=perstate_eff(H_PP,Hpk,Hkk,E_refs,NROOTS)
print(f"\nBaseline m=1:",flush=True)
for k in range(NROOTS):
    print(f"  S{k}: {(ev_full[k]-e_fci[k])*1000:+.1f} mH",flush=True)

# Truncation
thrs=[1e-3,5e-3,1e-2,5e-2,1e-1,2e-1,5e-1]
print(f"\n  thr      r  compr%      dE0       S1       S2       S3  (mH)")
for thr in thrs:
    r=int(np.sum(sigma_comb>=thr))
    if r==0:continue
    Ur=U1[:,:r];SIGr=SIG[:,:r]
    Hkk_r=Ur.T@SIGr;Hkk_r=0.5*(Hkk_r+Hkk_r.T)
    Hpk_r=np.zeros((N_p,r));Hpk_r[p_valid,:]=SIGr[p_f,:]
    ev=perstate_eff(H_PP,Hpk_r,Hkk_r,E_refs,NROOTS)
    dE=[(ev[k]-e_fci[k])*1000 for k in range(min(NROOTS,len(ev)))]
    compr=100*(1-r/d1)
    print(f"  {thr:.0e} {r:>5} {compr:>6.1f}% {dE[0]:>+9.1f} {dE[1]:>+9.1f} {dE[2]:>+9.1f} {dE[3]:>+9.1f}",flush=True)

# Save
outdir=f"{os.path.dirname(os.path.abspath(__file__))}/../checkpoints_phaseA"
os.makedirs(outdir,exist_ok=True)
results=[{"thr":float(thr),"r":int(np.sum(sigma_comb>=thr)),"d0":int(d1),
          "compr_pct":100*(1-np.sum(sigma_comb>=thr)/d1),
          "dE_mH":[float((ev[k]-e_fci[k])*1000) for k in range(NROOTS)]}
         for thr in thrs]
with open(f"{outdir}/cas10_trunc_dE_P800_m1_cis.json","w") as f:
    json.dump({"config":{"target_p":TARGET_P,"d_full":int(d1),"e_fci":e_fci},
               "sigma":[float(s) for s in sigma_comb[:d1]],
               "results":results},f,indent=2)
print(f"\nDone: {time.perf_counter()-t0:.0f}s",flush=True)
