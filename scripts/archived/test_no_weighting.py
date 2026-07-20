import numpy as np, sys, time
sys.path.insert(0, '/data/home/wangcx/krylov-dci')
from src_mf import QSpaceIndex, KDCIBackend
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
h1a = h1_mo[np.ix_(na_o,na_o)]; era = eri_mo[np.ix_(na_o,na_o,na_o,na_o)]
ne = (mol.nelec[0]-N_CORE, mol.nelec[1]-N_CORE)
as_ = cistring.gen_strings4orblist(range(N_ACT), ne[0])
bs_ = cistring.gen_strings4orblist(range(N_ACT), ne[1])
q_idx = QSpaceIndex(as_, bs_, N_ACT, ne, h1a, era)
backend = KDCIBackend(q_idx)
ef = float(direct_spin1.FCI().kernel(h1a, era, N_ACT, ne, nroots=1, verbose=0)[0])

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

H_PP = np.zeros((N,N))
for i in range(N):
    for j in range(N): H_PP[i,j] = ham.matrix_element(p_dets[i],p_dets[j])
H_PP = 0.5*(H_PP+H_PP.T); E0 = np.linalg.eigh(H_PP)[0][0]

dE_bare_mH = (E0 - ef) * 1000
print(f"P={N}  E_bare dE={dE_bare_mH:.1f} mH")

H_QP = backend.build_hqp(p_dets, verbose=False)

t0 = time.perf_counter()
basis, d0 = backend.build_basis(H_QP, E0, verbose=False)
H_QQ_t, H_PQ_t = backend.build_projected_blocks(basis, p_dets, H_QP=H_QP, verbose=False)
H_eff = build_effective_H(H_PP, H_PQ_t, H_QQ_t, E0, delta=0.0)
E_bloch = float(diagonalize_effective_H(H_eff, n_states=1)[0][0])
dE = (E_bloch - ef)*1000
print(f"m=0: d={d0}  dE0={dE:+.6f} mH  wall={time.perf_counter()-t0:.0f}s")

p_indices = backend.q_idx.p_indices(p_dets)
p_indices = p_indices[p_indices >= 0]
t0 = time.perf_counter()
basis, d1 = backend.propagate_basis(basis, E0, p_indices=p_indices, verbose=False)
H_QQ_t, H_PQ_t = backend.build_projected_blocks(basis, p_dets, H_QP=H_QP, verbose=False)
H_eff = build_effective_H(H_PP, H_PQ_t, H_QQ_t, E0, delta=0.0)
E_bloch = float(diagonalize_effective_H(H_eff, n_states=1)[0][0])
dE = (E_bloch - ef)*1000
print(f"m=1: d={d1}  dE0={dE:+.6f} mH  wall={time.perf_counter()-t0:.0f}s")

print(f"FCI = {ef:.12f}")
