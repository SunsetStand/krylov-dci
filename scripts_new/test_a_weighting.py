#!/usr/bin/env python3
"""Test: A-weighting impact on SVD + Bloch H^eff at P=200."""
import numpy as np, sys, time
from numpy.linalg import eigh
sys.path.insert(0, '/data/home/wangcx/krylov-dci')
from src_mf.pyscf_backend import QSpaceIndex, KDCIBackend
from src.effective_h import build_effective_H, diagonalize_effective_H
from src.hamiltonian import Hamiltonian
from src.determinants import hf_determinant, bit_positions
from pyscf import gto, scf, ao2mo
from pyscf.fci import cistring, direct_spin1
import itertools

N_CORE, N_ACT, P_INIT = 2, 10, 200

mol = gto.M(atom='N 0 0 0; N 0 0 1.1', basis='cc-pVDZ', verbose=0)
mf = scf.RHF(mol).run(verbose=0)
na_o = list(range(N_CORE, N_CORE+N_ACT))
norb = mf.mo_coeff.shape[1]
h1_mo = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
eri_mo = ao2mo.full(mol.intor('int2e'), mf.mo_coeff, compact=False).reshape(norb,norb,norb,norb)
h1a = h1_mo[np.ix_(na_o,na_o)]
era = eri_mo[np.ix_(na_o,na_o,na_o,na_o)]
ne = (mol.nelec[0]-N_CORE, mol.nelec[1]-N_CORE)
as_ = cistring.gen_strings4orblist(range(N_ACT), ne[0])
bs_ = cistring.gen_strings4orblist(range(N_ACT), ne[1])
q_idx = QSpaceIndex(as_, bs_, N_ACT, ne, h1a, era)
backend = KDCIBackend(q_idx)

ef, _ = direct_spin1.FCI().kernel(h1a, era, N_ACT, ne, nroots=1, verbose=0)
E_FCI = float(ef)

h2_4d = ao2mo.restore('s1', era, N_ACT).reshape(N_ACT,N_ACT,N_ACT,N_ACT)
ham = Hamiltonian(h1=h1a, h2=h2_4d, E_nuc=0.0, E_HF=0.0)
hf_a, hf_b = hf_determinant(*ne)
ao = bit_positions(hf_a); bo = bit_positions(hf_b)
av = [p for p in range(N_ACT) if p not in ao]
bv = [p for p in range(N_ACT) if p not in bo]

E_HF = ham.matrix_element((hf_a,hf_b),(hf_a,hf_b))
scores = []
for i in ao:
    for a in av:
        d = (hf_a^(1<<i)|(1<<a), hf_b)
        hij = ham.matrix_element(d,(hf_a,hf_b)); den = E_HF - ham.matrix_element(d,d)
        if abs(den)>1e-12: scores.append((d,-hij*hij/den))
for i in bo:
    for a in bv:
        d = (hf_a, hf_b^(1<<i)|(1<<a))
        hij = ham.matrix_element(d,(hf_a,hf_b)); den = E_HF - ham.matrix_element(d,d)
        if abs(den)>1e-12: scores.append((d,-hij*hij/den))
scores.sort(key=lambda x: x[1], reverse=True)
p_dets = [(hf_a,hf_b)]
for det,_ in scores:
    if det not in p_dets: p_dets.append(det)
    if len(p_dets)>=P_INIT: break
N = len(p_dets)
print(f"P={N} dets, M={q_idx.M}")

# All αβ double excitations (most important)
for i in ao:
    for j in bo:
        for a in av:
            for b in bv:
                d = (hf_a^(1<<i)|(1<<a), hf_b^(1<<j)|(1<<b))
                hij = ham.matrix_element(d,(hf_a,hf_b)); den = E_HF - ham.matrix_element(d,d)
                if abs(den)>1e-12: scores.append((d,-hij*hij/den))

# αα and ββ doubles
for i1,i2 in itertools.combinations(ao,2):
    for a1,a2 in itertools.combinations(av,2):
        d = (hf_a^(1<<i1)^(1<<i2)|(1<<a1)|(1<<a2), hf_b)
        hij = ham.matrix_element(d,(hf_a,hf_b)); den = E_HF - ham.matrix_element(d,d)
        if abs(den)>1e-12: scores.append((d,-hij*hij/den))
for i1,i2 in itertools.combinations(bo,2):
    for a1,a2 in itertools.combinations(bv,2):
        d = (hf_a, hf_b^(1<<i1)^(1<<i2)|(1<<a1)|(1<<a2))
        hij = ham.matrix_element(d,(hf_a,hf_b)); den = E_HF - ham.matrix_element(d,d)
        if abs(den)>1e-12: scores.append((d,-hij*hij/den))

scores.sort(key=lambda x: x[1], reverse=True)
p_dets = [(hf_a,hf_b)]
for det,_ in scores:
    if det not in p_dets: p_dets.append(det)
    if len(p_dets)>=P_INIT: break
N = len(p_dets)
print(f"After adding doubles: P={N} dets")

H_PP = np.zeros((N,N))
for i in range(N):
    for j in range(N): H_PP[i,j] = ham.matrix_element(p_dets[i],p_dets[j])
H_PP = 0.5*(H_PP+H_PP.T)
E0 = eigh(H_PP)[0][0]
print(f"E0 = {E0:.8f}  dE_bare = {(E0-E_FCI)*1000:.3f} mH")

H_QP = backend.build_hqp(p_dets, verbose=False)

denom = E0 - q_idx.hdiag
A_q = np.where(np.abs(denom)>1e-10, 1.0/denom, 0.0)
A_sqrt = np.sqrt(np.abs(A_q))

L0 = H_QP * A_q[:, np.newaxis]

results = {}
svd_thr = 1e-3
for label, T in [('no-weight', L0), ('A2-weight', A_q[:,np.newaxis]*L0), ('Ahalf-weight', A_sqrt[:,np.newaxis]*L0)]:
    t0 = time.perf_counter()
    U, s, _ = np.linalg.svd(T, full_matrices=False)
    keep = s > svd_thr * max(1.0, s[0])
    d = int(np.sum(keep))
    basis = U[:, keep]
    H_QQ_t, H_PQ_t = backend.build_projected_blocks(basis, p_dets, H_QP=H_QP, verbose=False)
    H_eff = build_effective_H(H_PP, H_PQ_t, H_QQ_t, E0, delta=0.0)
    E_bloch = float(diagonalize_effective_H(H_eff, n_states=1)[0][0])
    dE = (E_bloch - E_FCI)*1000
    results[label] = (d, dE, s[:5], time.perf_counter()-t0)

print()
print(f"{'Weighting':<15} {'d_keep':<8} {'dE0 (mH)':<15} {'top-5 sigma':<45} {'time':<8}")
print('-'*100)
for label in ['no-weight', 'A2-weight', 'Ahalf-weight']:
    d, dE, s5, t = results[label]
    sigma_str = ' '.join(f'{sv:.4f}' for sv in s5)
    print(f'{label:<15} {d:<8} {dE:+.6f}       {sigma_str:<45} {t:.1f}s')
print()
print(f"E_FCI = {E_FCI:.12f}")

# Also test: different SVD thresholds
print(f"\n--- SVD threshold scan (A2-weight) ---")
print(f"{'theta':<10} {'d_keep':<8} {'dE0 (mH)':<15} {'top-5 sigma'}")
print('-'*60)
for theta in [1e-2, 1e-3, 1e-4, 1e-5, 0.0]:
    T = A_q[:,np.newaxis]*L0
    U, s, _ = np.linalg.svd(T, full_matrices=False)
    if theta == 0.0:
        keep = np.ones(len(s), dtype=bool)
    else:
        keep = s > theta * max(1.0, s[0])
    d = int(np.sum(keep))
    basis = U[:, keep]
    H_QQ_t, H_PQ_t = backend.build_projected_blocks(basis, p_dets, H_QP=H_QP, verbose=False)
    H_eff = build_effective_H(H_PP, H_PQ_t, H_QQ_t, E0, delta=0.0)
    E_bloch = float(diagonalize_effective_H(H_eff, n_states=1)[0][0])
    dE = (E_bloch - E_FCI)*1000
    sigma_str = ' '.join(f'{sv:.4f}' for sv in s[:5])
    print(f'{theta:<10.0e} {d:<8} {dE:+.6f}       {sigma_str}')
