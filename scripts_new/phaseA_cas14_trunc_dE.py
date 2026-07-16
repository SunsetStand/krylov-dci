#!/usr/bin/env python3
"""
CAS(14,10) SVD Truncation + Energy Error (dE) — v6
= = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

VERBATIM adaptation of phaseA_cas10_svd.py (CAS10-SVD pipeline):
  1. H_PP → E0 = lowest eigenvalue, per-state E_refs
  2. build_basis_mf → T=A²·H_QP(SVD) → U_0 (m=0)
  3. propagate_basis_mf → m=1,2,... (adds Krylov vectors via residual propagation)
  4. build_blocks → full H_KK, H_PK from cache
  5. per-state H_eff via build_effective_H + diagonalize_effective_H
  6. NEW: SVD truncation dE sweep at each m

Key change from v5: ADDED propagate_basis_mf following CAS10-SVD logic.
CAS10 at P=200 m=0→dE0=+6 mH; CAS14 P=800 needs m>0 for excited-state convergence.
"""
import sys, os, time, json, argparse, itertools, gc
import numpy as np
from numpy.linalg import svd, eigh

PROJECT_ROOT = '/data/home/wangcx/krylov-dci'
sys.path.insert(0, PROJECT_ROOT)

from src_mf import QSpaceIndex, KDCIBackend
from src_mf.pspace_ops import build_hpp_sigma
from src.effective_h import build_effective_H, diagonalize_effective_H
from src.determinants import hf_determinant, bit_positions
from src.hamiltonian import Hamiltonian
from pyscf import gto, scf, ao2mo
from pyscf.fci import cistring

# ═══════════════════════════════════════════════════════════════
args = argparse.ArgumentParser()
args.add_argument('--target-p', type=int, default=1600)
args.add_argument('--m-max', type=int, default=3, help='Krylov propagation steps (m=0 is build_basis_mf only)')
args = args.parse_args()

TARGET_P = args.target_p; M_MAX = args.m_max
SVD_THR = 1e-3
THRESHOLDS = [1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 3e-2, 5e-2, 1e-1, 2e-1, 5e-1]
N_ACT = 14; N_CORE = 2; R = 1.1; ne = (5, 5); NROOTS = 6

E_FCI = np.array([-62.410924132579,-62.280783729436,-62.280783729436,
                  -62.228221943409,-62.228221943409,-62.225780103936])

print("="*70)
print(f"CAS({N_ACT},{sum(ne)}) SVD Truncation → dE  P={TARGET_P}  m_max={M_MAX}")
print("Pipeline: H_PP→build_basis_mf(m=0)→propagate_basis_mf(m>0)→blocks→per-state H_eff")
print("="*70, flush=True)

# ═══════════════════════════════════════════════════════════════
# Setup + HFPT2 pool (identical to CAS10-SVD / phaseA_cas14_svd_scan.py)
# ═══════════════════════════════════════════════════════════════
t0=time.perf_counter()
mol=gto.M(atom=f'N 0 0 0; N 0 0 {R}',basis='cc-pVDZ',verbose=0)
mf=scf.RHF(mol).run(verbose=0)
na_o=list(range(N_CORE,N_CORE+N_ACT)); norb=mf.mo_coeff.shape[1]
h1_mo=mf.mo_coeff.T@mf.get_hcore()@mf.mo_coeff
eri=ao2mo.full(mol.intor('int2e'),mf.mo_coeff,compact=False).reshape([norb]*4)
h1a=h1_mo[np.ix_(na_o,na_o)]; era=eri[np.ix_(na_o,na_o,na_o,na_o)]
as_=cistring.gen_strings4orblist(range(N_ACT),ne[0])
bs_=cistring.gen_strings4orblist(range(N_ACT),ne[1])
na,nb=len(as_),len(bs_); M_all=na*nb
q_idx=QSpaceIndex(as_,bs_,N_ACT,ne,h1a,era); backend=KDCIBackend(q_idx)
hdiag=np.array([q_idx.hdiag[qi] for qi in range(M_all)])
print(f"  M_all={M_all:,}  setup={time.perf_counter()-t0:.0f}s",flush=True)

h2_4d=ao2mo.restore('s1',era,N_ACT).reshape([N_ACT]*4)
ham=Hamiltonian(h1=h1a,h2=h2_4d,E_nuc=0.0,E_HF=0.0)
hf_a,hf_b=hf_determinant(*ne)
ao_b=bit_positions(hf_a);bo_b=bit_positions(hf_b)
av=[p for p in range(N_ACT) if p not in ao_b];bv=[p for p in range(N_ACT) if p not in bo_b]
def gen_hfpt2():
    sc=[]; E_HF=ham.matrix_element((hf_a,hf_b),(hf_a,hf_b))
    for i in ao_b:
        for a in av:
            d=(hf_a^(1<<i)|(1<<a),hf_b);hij=ham.matrix_element(d,(hf_a,hf_b))
            den=E_HF-ham.matrix_element(d,d)
            if abs(den)>1e-12:sc.append((d,-hij*hij/den))
    for i in bo_b:
        for a in bv:
            d=(hf_a,hf_b^(1<<i)|(1<<a));hij=ham.matrix_element(d,(hf_a,hf_b))
            den=E_HF-ham.matrix_element(d,d)
            if abs(den)>1e-12:sc.append((d,-hij*hij/den))
    for i1,i2 in itertools.combinations(ao_b,2):
        for a1,a2 in itertools.combinations(av,2):
            d=(hf_a^(1<<i1)^(1<<i2)|(1<<a1)|(1<<a2),hf_b);hij=ham.matrix_element(d,(hf_a,hf_b))
            den=E_HF-ham.matrix_element(d,d)
            if abs(den)>1e-12:sc.append((d,-hij*hij/den))
    for i1,i2 in itertools.combinations(bo_b,2):
        for a1,a2 in itertools.combinations(bv,2):
            d=(hf_a,hf_b^(1<<i1)^(1<<i2)|(1<<a1)|(1<<a2));hij=ham.matrix_element(d,(hf_a,hf_b))
            den=E_HF-ham.matrix_element(d,d)
            if abs(den)>1e-12:sc.append((d,-hij*hij/den))
    for i in ao_b:
        for j in bo_b:
            for a in av:
                for b in bv:
                    d=(hf_a^(1<<i)|(1<<a),hf_b^(1<<j)|(1<<b));hij=ham.matrix_element(d,(hf_a,hf_b))
                    den=E_HF-ham.matrix_element(d,d)
                    if abs(den)>1e-12:sc.append((d,-hij*hij/den))
    sc.sort(key=lambda x:x[1],reverse=True);return sc
sc=gen_hfpt2()
pool=[(hf_a,hf_b)]
for d,s in sc:
    if d not in pool:pool.append(d)
    if len(pool)>=TARGET_P:break
p_dets=pool[:TARGET_P];print(f"  Pool: {len(p_dets)} dets",flush=True)

# P-space flat indices
p_flat_arr=np.array([q_idx.flat_index(int(pa),int(pb)) for pa,pb in p_dets])
p_idx_set=set(int(x) for x in p_flat_arr if x>=0)
p_valid=p_flat_arr>=0;p_f=p_flat_arr[p_valid]

# ═══════════════════════════════════════════════════════════════
# 1. H_PP → E0, per-state E_refs
# ═══════════════════════════════════════════════════════════════
print(f"\nBuilding H_PP...",flush=True);t_h=time.perf_counter()
H_PP=build_hpp_sigma(p_dets,backend,q_idx._alpha_idx,q_idx._beta_idx,na,nb)
E_vals_HPP,_=eigh(H_PP);E0=float(E_vals_HPP[0]);E_refs=E_vals_HPP[:NROOTS]
print(f"  lowest(H_PP)={E0:.10f} (dE_bare={(E0-E_FCI[0])*1000:+.1f} mH)  H_PP:{time.perf_counter()-t_h:.0f}s",flush=True)

# ═══════════════════════════════════════════════════════════════
# 2. build_basis_mf (VERBATIM from CAS10-SVD)
# ═══════════════════════════════════════════════════════════════
print(f"\n[build_basis_mf] T=A²·H_QP, E0={E0:.10f}...",flush=True)
A_q=np.where(np.abs(E0-hdiag)>1e-10,1.0/(E0-hdiag),0.0);A_half=np.sqrt(np.abs(A_q))
tmpdir=f'{PROJECT_ROOT}/tmp';os.makedirs(tmpdir,exist_ok=True)
fpath=f'{tmpdir}/cas14_svd_T_P{TARGET_P}.dat'
T=np.memmap(fpath,dtype='float64',mode='w+',shape=(M_all,TARGET_P),order='F')
t_b=time.perf_counter()
for p in range(TARGET_P):
    pa,pb=int(p_dets[p][0]),int(p_dets[p][1])
    ia=q_idx._alpha_idx.get(pa);ib=q_idx._beta_idx.get(pb)
    if ia is None or ib is None:continue
    ci=np.zeros((na,nb));ci[ia,ib]=1.0;sig=backend.sigma_full(ci).reshape(-1)
    for q in p_idx_set:sig[q]=0.0;T[:,p]=A_half*sig
    if(p+1)%max(1,TARGET_P//10)==0:
        e=time.perf_counter()-t_b;print(f"  col {p+1}/{TARGET_P} ({e:.0f}s)",flush=True)
T.flush();t_build=time.perf_counter()-t_b
print(f"  T:{t_build:.0f}s",flush=True)

t_s=time.perf_counter();print(f"  SVD({M_all},{TARGET_P})...",flush=True)
U_full,sigma,_=svd(T,full_matrices=False)
t_svd=time.perf_counter()-t_s;smax=sigma[0]
print(f"  SVD:{t_svd:.0f}s σ₁={smax:.4f} σ_min/σ₁={sigma[-1]/smax:.6f}",flush=True)
try:del T;gc.collect();os.unlink(fpath)
except:pass
d0=int(np.sum(sigma>=SVD_THR*max(1.0,smax)))
U_0=np.ascontiguousarray(U_full[:,:d0])
print(f"  kept {d0}/{TARGET_P} @ thr={SVD_THR}",flush=True)

# ═══════════════════════════════════════════════════════════════
# 3. Krylov propagation (VERBATIM from CAS10-SVD propagate_basis_mf)
# ═══════════════════════════════════════════════════════════════
def propagate_basis_mf(U_basis,A_q,E0,p_set,tag):
    A_h2=np.sqrt(np.abs(A_q));M_dim,d_old=U_basis.shape
    if d_old==0:return U_basis.copy(),d_old
    tmp=f'{tmpdir}/cas14_prop_d{d_old}_{tag}.dat'
    T2=np.memmap(tmp,dtype='float64',mode='w+',shape=(M_dim,d_old),order='F')
    t0b=time.perf_counter()
    print(f"    [propagate] d={d_old}...",flush=True)
    for k in range(d_old):
        bk=U_basis[:,k];sig=backend.sigma_full(bk.reshape(na,nb)).reshape(-1)
        res=sig-hdiag*bk
        for q in p_set:res[q]=0.0
        T2[:,k]=A_h2*res
        if(k+1)%max(1,d_old//5)==0:
            print(f"      col {k+1}/{d_old} ({time.perf_counter()-t0b:.0f}s)",flush=True)
    T2.flush()
    t_mgs=time.perf_counter()
    print(f"    [MGS→SVD] orthogonalizing vs existing...",flush=True)
    for k in range(d_old):
        col=np.array(T2[:,k]);col-=U_basis@(U_basis.T@col);T2[:,k]=col
    T2.flush();print(f"    MGS:{time.perf_counter()-t_mgs:.0f}s",flush=True)
    t_s2=time.perf_counter()
    print(f"    SVD({M_dim},{d_old})...",flush=True)
    Us,sigma2,_=svd(T2,full_matrices=False)
    s2max=sigma2[0] if len(sigma2)>0 else 0;mask=sigma2>=SVD_THR*max(1.0,s2max)
    U_trunc=Us[:,mask];n_keep=int(np.sum(mask))
    print(f"    SVD:{time.perf_counter()-t_s2:.0f}s {d_old}→{n_keep} kept",flush=True)
    try:del T2;gc.collect();os.unlink(tmp)
    except:pass
    basis=list(U_basis[:,j] for j in range(d_old));new=0
    for k in range(U_trunc.shape[1]):
        v=U_trunc[:,k].copy();v-=U_basis@(U_basis.T@v)
        for b in basis[d_old:]:v-=np.dot(b,v)*b
        nrm=np.linalg.norm(v)
        if nrm>1e-10:v/=nrm;basis.append(v);new+=1
    return np.column_stack(basis) if len(basis)>d_old else U_basis, len(basis)

# ═══════════════════════════════════════════════════════════════
# 4. build_blocks + per-state H_eff (VERBATIM from CAS10-SVD)
# ═══════════════════════════════════════════════════════════════
def build_blocks(U_basis):
    d=U_basis.shape[1];Np=TARGET_P
    if d==0:return np.zeros((0,0)),np.zeros((Np,0))
    print(f"    [blocks] d={d}...",flush=True);t0b=time.perf_counter()
    H_KK=np.zeros((d,d));H_PK=np.zeros((Np,d))
    for k in range(d):
        sk=backend.sigma_full(U_basis[:,k].reshape(na,nb)).reshape(-1)
        H_KK[:,k]=U_basis.T@sk;H_PK[p_valid,k]=sk[p_f]
        if(k+1)%max(1,d//5)==0:print(f"      {k+1}/{d} ({time.perf_counter()-t0b:.0f}s)",flush=True)
    H_KK=0.5*(H_KK+H_KK.T)
    print(f"    [blocks] {time.perf_counter()-t0b:.0f}s",flush=True)
    return H_KK,H_PK

def perstate_eff(H_PP,H_PK,H_KK,E_refs,nroots):
    ev_out=np.zeros(len(E_refs))
    for k,Ek in enumerate(E_refs):
        evk=np.asarray(diagonalize_effective_H(
            build_effective_H(H_PP,H_PK,H_KK,float(Ek),delta=0.0),n_states=nroots)[0])
        ev_out[k]=evk[int(np.argmin(np.abs(evk-Ek)))]
    return ev_out

# ═══════════════════════════════════════════════════════════════
# Run krylov pipeline
# ═══════════════════════════════════════════════════════════════
U_m=U_0;d_m=d0;krylov_ev={};krylov_d={}
H_KK_cur,H_PK_cur=build_blocks(U_m)
ev=perstate_eff(H_PP,H_PK_cur,H_KK_cur,E_refs[:NROOTS],NROOTS)
de=[(ev[k]-E_FCI[k])*1000 for k in range(min(NROOTS,len(ev)))]
krylov_ev[0]=de;krylov_d[0]=d_m
print(f"\n  m=0: d={d_m} dE0={de[0]:+.1f}  S1={de[1]:+.1f}  S2={de[2]:+.1f} mH",flush=True)
for m in range(1,M_MAX+1):
    U_m,d_m=propagate_basis_mf(U_m,A_q,E0,p_idx_set,f"m{m}")
    if d_m==krylov_d[m-1]:
        print(f"    m={m}: no new directions, stopping",flush=True)
        krylov_ev[m]=krylov_ev[m-1];krylov_d[m]=d_m;break
    H_KK_cur,H_PK_cur=build_blocks(U_m)
    ev=perstate_eff(H_PP,H_PK_cur,H_KK_cur,E_refs[:NROOTS],NROOTS)
    de=[(ev[k]-E_FCI[k])*1000 for k in range(min(NROOTS,len(ev)))]
    krylov_ev[m]=de;krylov_d[m]=d_m
    print(f"  m={m}: d={d_m} dE0={de[0]:+.1f}  S1={de[1]:+.1f}  S2={de[2]:+.1f} mH",flush=True)

# ═══════════════════════════════════════════════════════════════
# 5. SVD truncation dE sweep (on full basis from last m)
# ═══════════════════════════════════════════════════════════════
# Use the U_0 basis for truncation (SVD vectors from build_basis_mf).
# Pre-compute sigma vectors for the reference basis U_0.
n_cols=d0;print(f"\nSigma pass {n_cols} cols for truncation sweep...",flush=True)
t_sp=time.perf_counter();sigs=[None]*n_cols
for k in range(n_cols):
    sigs[k]=backend.sigma_full(U_0[:,k].reshape(na,nb)).reshape(-1)
    if(k+1)%max(1,n_cols//10)==0:
        e=time.perf_counter()-t_sp;print(f"  col {k+1}/{n_cols} ({e:.0f}s)",flush=True)
print(f"  sigma pass:{time.perf_counter()-t_sp:.0f}s",flush=True)

results=[]
print(f"\n{'='*60}")
print(f"Testing {len(THRESHOLDS)} SVD thresholds (per-state H_eff, full basis)...")
print(f"{'='*60}",flush=True)
for thr in THRESHOLDS:
    r=int(np.sum(sigma>=thr*smax))
    if r==0 or r>n_cols:r=min(r if r>0 else 0,n_cols)
    if r==0:results.append({'thr':thr,'r_svd':0});print(f"  thr={thr:.0e}: r=0");continue
    t0e=time.perf_counter()
    U_r=U_0[:,:r];H_KK_r=np.zeros((r,r));H_PK_r=np.zeros((TARGET_P,r))
    for k in range(r):
        sk=sigs[k];H_KK_r[:,k]=U_r.T@sk;H_PK_r[p_valid,k]=sk[p_f]
    H_KK_r=0.5*(H_KK_r+H_KK_r.T)
    ev=perstate_eff(H_PP,H_PK_r,H_KK_r,E_refs[:NROOTS],NROOTS)
    de=[(ev[k]-E_FCI[k])*1000 for k in range(min(NROOTS,len(ev)))]
    compr=(1-r/TARGET_P)*100;max_ex=max(abs(x) for x in de[1:]) if len(de)>1 else 0
    te=time.perf_counter()-t0e
    print(f"  thr={thr:.0e}: r={r}  compr={compr:.1f}%  dE0={de[0]:+.1f}  S1={de[1]:+.1f}  S2={de[2]:+.1f} mH ({te:.0f}s)",flush=True)
    results.append({'thr':thr,'r_svd':r,'compr':compr,'dE0_mH':float(de[0]),'max_dE_ex_mH':float(max_ex),'dE':[float(x) for x in de]})

# ═══════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print(f"CAS(14,10) SVD Truncation  P={TARGET_P}  m_max={M_MAX}")
print(f"Krylov convergence: m=0 d={krylov_d[0]} dE0={krylov_ev[0][0]:+.1f}")
for m,de in krylov_ev.items():
    if m==0:continue
    print(f"  m={m}: d={krylov_d[m]} dE0={de[0]:+.1f} S1={de[1]:+.1f} S2={de[2]:+.1f} mH")
print(f"FCI: S0={E_FCI[0]:.12f}  S1={E_FCI[1]:.12f}")
print(f"{'='*80}")
print(f"{'thr':>8}  {'r_svd':>6}  {'compr%':>7}  {'dE0/mH':>9}  {'S1/mH':>8}  {'S2/mH':>8}  {'S3/mH':>8}  {'max_ex':>8}")
print("-"*80)
for r in results:
    de=r.get('dE')
    if de is None:print(f"{r['thr']:>8.0e}  {r['r_svd']:>6}  —");continue
    d1=de[1] if len(de)>1 else 0;d2=de[2] if len(de)>2 else 0;d3=de[3] if len(de)>3 else 0
    print(f"{r['thr']:>8.0e}  {r['r_svd']:>6}  {r['compr']:>6.1f}%  {r['dE0_mH']:>+9.1f}  {d1:>+8.1f}  {d2:>+8.1f}  {d3:>+8.1f}  {r['max_dE_ex_mH']:>+8.1f}")

outdir=os.path.join(PROJECT_ROOT,'checkpoints_phaseA');os.makedirs(outdir,exist_ok=True)
fname=f'{outdir}/cas14_truncation_dE_P{TARGET_P}.json'
with open(fname,'w') as f:
    json.dump({'config':{'P':TARGET_P,'cas':N_ACT,'M_all':M_all,'thresholds':THRESHOLDS,
        'svd_thr':SVD_THR,'m_max':M_MAX,'E_FCI':E_FCI.tolist(),'d0':d0},
        'krylov':{str(m):{'d':krylov_d[m],'dE_mH':[float(x) for x in krylov_ev[m][:NROOTS]]}
                  for m in krylov_d},
        'sigma_full':sigma.tolist(),'results':results},f,indent=2)
print(f"\nSaved: {fname}\nDone.")
