import sys; sys.path.insert(0,'src')
import numpy as np
from pyscf import gto, scf, mcscf, ao2mo
from pyscf.fci import selected_ci, cistring
from pyscf.fci.direct_nosym import FCI
from hamiltonian import Hamiltonian, _unpack_4fold
from krylov import compute_A, modified_gram_schmidt
from svd_compression import build_weighted_coupling, svd_truncate
from effective_h import self_consistent_iteration, build_effective_H, diagonalize_effective_H

mol=gto.M(atom='N 0 0 0; N 0 0 1.10',basis='cc-pVDZ',verbose=0)
mf=scf.RHF(mol);mf.kernel()
mycas=mcscf.CASCI(mf,10,10);mycas.frozen=2;mycas.verbose=0;mycas.kernel()
E_ref=mycas.e_tot
mo=mycas.mo_coeff[:,2:12]
h1_ao=mol.intor_symmetric('int1e_kin')+mol.intor_symmetric('int1e_nuc')
eri_ao=mol.intor('int2e')
h1=mo.T@h1_ao@mo;h2=_unpack_4fold(ao2mo.incore.full(eri_ao,mo),10)
s=FCI();s.verbose=0;E_act,_=s.kernel(h1,h2,10,(5,5),ecore=0.0);ecore=E_ref-E_act
ham=Hamiltonian(h1=h1,h2=h2,E_nuc=0.0,E_HF=mf.e_tot)

# PT2 P=50
hfa=(1<<5)-1;hfb=(1<<5)-1;E_hf=ham.diagonal_element(hfa,hfb)
aoc=list(range(5));boc=list(range(5))
a_str=sum(1<<p for p in aoc);b_str=sum(1<<p for p in boc)
av=[p for p in range(10) if p not in aoc]
bv=[p for p in range(10) if p not in boc]
cands={(a_str,b_str)}
for i in aoc:
    for a in av: cands.add(((a_str^(1<<i))|(1<<a),b_str))
for i in boc:
    for a in bv: cands.add((a_str,(b_str^(1<<i))|(1<<a)))
if len(aoc)>=2:
    for ii,i in enumerate(aoc):
        for j in aoc[ii+1:]:
            for ia,va in enumerate(av):
                for vb in av[ia+1:]:
                    cands.add((a_str^(1<<i)^(1<<j)|(1<<va)|(1<<vb),b_str))
if len(boc)>=2:
    for ii,i in enumerate(boc):
        for j in boc[ii+1:]:
            for ia,va in enumerate(bv):
                for vb in bv[ia+1:]:
                    cands.add((a_str,b_str^(1<<i)^(1<<j)|(1<<va)|(1<<vb)))
for i in aoc:
    for j in boc:
        for va in av:
            for vb in bv:
                cands.add(((a_str^(1<<i))|(1<<va),(b_str^(1<<j))|(1<<vb)))
cands=list(cands)

scores=[(1e10,(hfa,hfb))]
for d in cands:
    if d==(hfa,hfb): continue
    hij=ham.matrix_element((hfa,hfb),d)
    if abs(hij)<1e-14: continue
    ei=ham.diagonal_element(*d);de=E_hf-ei
    ss=1e9 if abs(de)<1e-14 else abs(hij*hij/de)
    if ss>1e-5: scores.append((ss,d))
scores.sort(key=lambda x:x[0],reverse=True)
p_dets=[d for _,d in scores[:50]]

# Q-neighborhood + H_QP (simplified, inline)
pa_s={d[0] for d in p_dets};pb_s={d[1] for d in p_dets}
qa_s, qb_s=set(),set()
for pa,pb in p_dets:
    aocc=[i for i in range(10) if (pa>>i)&1];bocc=[i for i in range(10) if (pb>>i)&1]
    avir=[i for i in range(10) if i not in aocc];bvir=[i for i in range(10) if i not in bocc]
    nao,nbo=len(aocc),len(bocc)
    for i in aocc:
        for v in avir: qa_s.add((pa^(1<<i))|(1<<v));qb_s.add(pb)
    for i in bocc:
        for v in bvir: qa_s.add(pa);qb_s.add((pb^(1<<i))|(1<<v))
    if nao>=2:
        for ii,i in enumerate(aocc):
            for j in aocc[ii+1:]:
                for ia,va in enumerate(avir):
                    for vb in avir[ia+1:]:
                        qa_s.add(pa^(1<<i)^(1<<j)|(1<<va)|(1<<vb));qb_s.add(pb)
    if nbo>=2:
        for ii,i in enumerate(bocc):
            for j in bocc[ii+1:]:
                for ia,va in enumerate(bvir):
                    for vb in bvir[ia+1:]:
                        qa_s.add(pa);qb_s.add(pb^(1<<i)^(1<<j)|(1<<va)|(1<<vb))
    for i in aocc:
        for j in bocc:
            for va in avir:
                for vb in bvir:
                    qa_s.add((pa^(1<<i))|(1<<va));qb_s.add((pb^(1<<j))|(1<<vb))
qa_s.update(pa_s);qb_s.update(pb_s)
qa=sorted(qa_s);qb=sorted(qb_s)

# H_QP sparse
nb_q=len(qb);N=len(p_dets);M=len(qa)*nb_q
H_QP=np.zeros((M,N))
qam={s:i for i,s in enumerate(qa)};qbm={s:i for i,s in enumerate(qb)}
for p_idx,(pa,pb) in enumerate(p_dets):
    aocc=[i for i in range(10) if (pa>>i)&1];bocc=[i for i in range(10) if (pb>>i)&1]
    avir=[i for i in range(10) if i not in aocc];bvir=[i for i in range(10) if i not in bocc]
    nao,nbo=len(aocc),len(bocc)
    conn=[]
    for i in aocc:
        for v in avir: conn.append(((pa^(1<<i))|(1<<v),pb))
    for i in bocc:
        for v in bvir: conn.append((pa,(pb^(1<<i))|(1<<v)))
    if nao>=2:
        for ii,i in enumerate(aocc):
            for j in aocc[ii+1:]:
                for ia,va in enumerate(avir):
                    for vb in avir[ia+1:]:
                        conn.append((pa^(1<<i)^(1<<j)|(1<<va)|(1<<vb),pb))
    if nbo>=2:
        for ii,i in enumerate(bocc):
            for j in bocc[ii+1:]:
                for ia,va in enumerate(bvir):
                    for vb in bvir[ia+1:]:
                        conn.append((pa,pb^(1<<i)^(1<<j)|(1<<va)|(1<<vb)))
    for i in aocc:
        for j in bocc:
            for va in avir:
                for vb in bvir:
                    conn.append(((pa^(1<<i))|(1<<va),(pb^(1<<j))|(1<<vb)))
    for qa_str,qb_str in conn:
        ia=qam.get(qa_str);ib=qbm.get(qb_str)
        if ia is not None and ib is not None:
            if qa_str in pa_s and qb_str in pb_s: continue
            hij=ham.matrix_element((pa,pb),(qa_str,qb_str))
            if abs(hij)>1e-14: H_QP[ia*nb_q+ib,p_idx]=hij

# H_PP
H_PP=np.zeros((N,N))
for i in range(N):
    for j in range(N): H_PP[i,j]=ham.matrix_element(p_dets[i],p_dets[j])
E0=np.linalg.eigh(H_PP)[0][0]

# SVD
eri_p=ao2mo.restore(1,ao2mo.incore.full(eri_ao,mo),10)
ci_s=(np.asarray(qa,dtype=np.int64),np.asarray(qb,dtype=np.int64))
hdiag=selected_ci.make_hdiag(h1,eri_p,ci_s,10,(5,5))
A_diag=1.0/np.maximum(np.abs(E0-hdiag),1e-12)
L0=H_QP*A_diag[:,np.newaxis]
T=build_weighted_coupling(L0,A_diag)
U,_,_=svd_truncate(T,1e-3)
U,_=modified_gram_schmidt(U,np.zeros((M,0)))
H_QQ_t=(U*hdiag[:,None]).T@U;H_QQ_t=0.5*(H_QQ_t+H_QQ_t.T)
H_PQ_t=(U.T@H_QP).T

# Compare 3 modes
delta_exact = E_ref - E0 - ecore
print(f'E0 = {E0:.10f}')
print(f'Δ_exact (from CASCI ref) = {delta_exact:.10f} Ha = {delta_exact*1000:.1f} mH')
print()

# Fixed delta (use CASCI)
# Fixed delta (use CASCI)
H_eff_fd = build_effective_H(H_PP,H_PQ_t,H_QQ_t,E0,delta_exact)
eigvals_fd, _ = diagonalize_effective_H(H_eff_fd)
E_fd = eigvals_fd[0]
dE_fd=((E_fd+ecore)-E_ref)*1000
print(f'Fixed-Δ:  E={E_fd+ecore:.10f}, dE={dE_fd:.1f} mH')
print()

# Delta = 0 (no correction)
# Delta = 0
H_eff_d0 = build_effective_H(H_PP,H_PQ_t,H_QQ_t,E0,0.0)
eigvals_d0, _ = diagonalize_effective_H(H_eff_d0)
E_d0 = eigvals_d0[0]
dE_d0=((E_d0+ecore)-E_ref)*1000
print(f'Δ=0:      E={E_d0+ecore:.10f}, dE={dE_d0:.1f} mH')
print()

# SCF self-consistent
result=self_consistent_iteration(H_PP,H_PQ_t,H_QQ_t,E0,verbose=True)
E_scf=float(result['E_final'])
dE_scf=((E_scf+ecore)-E_ref)*1000
print(f'SCF:      E={E_scf+ecore:.10f}, dE={dE_scf:.1f} mH, {result["n_iter"]} iters')
print(f'  delta_initial=0, delta_final={result["delta_final"]:.10f} Ha')
print()

# Compare delta values
print(f'Summary:')
print(f'  Fixed-Δ   dE = {dE_fd:.1f} mH')
print(f'  Δ=0        dE = {dE_d0:.1f} mH')
print(f'  SCF        dE = {dE_scf:.1f} mH ({result["n_iter"]} iters)')
print(f'  SCF vs fixed-Δ difference: {abs(dE_scf-dE_fd):.2f} mH')
print(f'  SCF vs Δ=0   difference: {abs(dE_scf-dE_d0):.2f} mH')
