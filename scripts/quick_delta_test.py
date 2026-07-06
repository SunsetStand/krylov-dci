#!/usr/bin/env python3
"""Quick delta-B test: state 0 only, m=0,1, delta=0 vs exact."""
import sys, numpy as np, itertools
sys.path.insert(0, '/data/home/wangcx/krylov-dci')
from src_mf import QSpaceIndex, KDCIBackend
from src.effective_h import build_effective_H, diagonalize_effective_H
from src.determinants import hf_determinant, bit_positions
from src.hamiltonian import Hamiltonian
from pyscf import gto, scf, ao2mo
from pyscf.fci import cistring, direct_spin1

N_CORE, N_ACT, P_N = 2, 10, 400
mol = gto.M(atom='N 0 0 0; N 0 0 1.1', basis='cc-pVDZ', verbose=0)
mf = scf.RHF(mol).run(verbose=0)
norbs = list(range(N_CORE, N_CORE+N_ACT))
norb = mf.mo_coeff.shape[1]
h1m = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
e2 = ao2mo.full(mol.intor('int2e'), mf.mo_coeff, compact=False).reshape(norb,norb,norb,norb)
h1a = h1m[np.ix_(norbs, norbs)]
era = e2[np.ix_(norbs, norbs, norbs, norbs)]
ne = (mol.nelec[0]-N_CORE, mol.nelec[1]-N_CORE)
q_idx = QSpaceIndex(cistring.gen_strings4orblist(range(N_ACT),ne[0]),
                    cistring.gen_strings4orblist(range(N_ACT),ne[1]),
                    N_ACT, ne, h1a, era)
be = KDCIBackend(q_idx)

ef, _ = direct_spin1.FCI().kernel(h1a, era, N_ACT, ne, nroots=2, verbose=0)
e_dmrg = [float(e) for e in np.atleast_1d(ef)[:2]]
print(f"DMRG-CI E0={e_dmrg[0]:.8f}")

h2_4d = ao2mo.restore('s1', era, N_ACT).reshape(N_ACT,N_ACT,N_ACT,N_ACT)
hm = Hamiltonian(h1=h1a, h2=h2_4d)
hfa, hfb = hf_determinant(*ne)
ao=bit_positions(hfa); bo=bit_positions(hfb)
av=[p for p in range(N_ACT) if p not in ao]; bv=[p for p in range(N_ACT) if p not in bo]
E_HF = hm.matrix_element((hfa,hfb),(hfa,hfb))
sc=[]
for i in ao:
    for a in av:
        d=(hfa^(1<<i)|(1<<a),hfb); hij=hm.matrix_element(d,(hfa,hfb))
        de=E_HF-hm.matrix_element(d,d)
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
for det,_ in sc:
    if det not in pd: pd.append(det)
    if len(pd)>=P_N: break

H_PP=np.zeros((P_N,P_N))
for i in range(P_N):
    for j in range(P_N): H_PP[i,j]=hm.matrix_element(pd[i],pd[j])
H_PP=0.5*(H_PP+H_PP.T)
E0_vals,_=np.linalg.eigh(H_PP); E0_vals=E0_vals[:2]
dk = e_dmrg[0]-E0_vals[0]
print(f"E0={E0_vals[0]:.8f}, delta_exact={dk*1000:.1f} mH")

print("Building H_QP...")
H_QP = be.build_hqp(pd, verbose=False)
print(f"H_QP: {H_QP.shape}")

for label, delta_val in [("delta=0", 0.0), ("delta=exact", dk)]:
    print(f"\n--- {label} ---")
    basis, d0 = be.build_basis(H_QP, E0_vals[0], verbose=False)
    H_QQ_t, _ = be.build_projected_blocks(basis, pd, H_QP=H_QP, verbose=False)
    H_PQ_t = H_QP.T @ basis
    H_eff = build_effective_H(H_PP, H_PQ_t, H_QQ_t, E0_vals[0], delta=0.0)
    ev, _ = diagonalize_effective_H(H_eff, n_states=1)
    dEm0 = (ev[0]-e_dmrg[0])*1000
    print(f"  m=0: d={d0}, dE={dEm0:+.1f} mH")

    basis, d1 = be.propagate_basis(basis, E0_vals[0], delta=delta_val, verbose=False)
    H_QQ_t, _ = be.build_projected_blocks(basis, pd, H_QP=H_QP, verbose=False)
    H_PQ_t = H_QP.T @ basis
    H_eff = build_effective_H(H_PP, H_PQ_t, H_QQ_t, E0_vals[0], delta=0.0)
    ev, _ = diagonalize_effective_H(H_eff, n_states=1)
    dEm1 = (ev[0]-e_dmrg[0])*1000
    print(f"  m=1: d={d1}, dE={dEm1:+.1f} mH")

print("\nDONE")
