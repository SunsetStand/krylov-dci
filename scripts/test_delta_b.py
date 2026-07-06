#!/usr/bin/env python3
"""
Phase 18 — Delta-B propagation test: per-state Krylov, m=0,1.
Compares delta=0 vs delta=exact on N2/cc-pVDZ CAS(10,10) P=400.

Delta in propagation:
  B = H_QQ - D_QQ - delta*I  (was: B = H_QQ - D_QQ)
  A_q = 1/(E0_k - H_D[qq]) unchanged

Runs: per-state, dense backend, m=0,1.
"""
import sys, os, time, json, itertools
import numpy as np
from numpy.linalg import eigh

PROJECT_ROOT = '/data/home/wangcx/krylov-dci'
sys.path.insert(0, PROJECT_ROOT)
from src_mf import QSpaceIndex, KDCIBackend
from src.effective_h import build_effective_H, diagonalize_effective_H
from src.determinants import hf_determinant, bit_positions
from src.hamiltonian import Hamiltonian
from pyscf import gto, scf, ao2mo
from pyscf.fci import cistring, direct_spin1

N_CORE, N_ACT = 2, 10
NROOTS, M_MAX, P_TARGET = 2, 1, 400  # quick test: first 3 states

# ── Build system ──
print("Building N2/cc-pVDZ CAS(10,10)...", flush=True)
mol = gto.M(atom='N 0 0 0; N 0 0 1.1', basis='cc-pVDZ', verbose=0)
mf = scf.RHF(mol).run(verbose=0)
n_act_orbs = list(range(N_CORE, N_CORE+N_ACT))
norb = mf.mo_coeff.shape[1]
h1e_mo = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
e2 = ao2mo.full(mol.intor('int2e'), mf.mo_coeff, compact=False)
eri_mo = e2.reshape(norb,norb,norb,norb)
h1a = h1e_mo[np.ix_(n_act_orbs, n_act_orbs)]
era = eri_mo[np.ix_(n_act_orbs, n_act_orbs, n_act_orbs, n_act_orbs)]
ne = (mol.nelec[0]-N_CORE, mol.nelec[1]-N_CORE)
as_ = cistring.gen_strings4orblist(range(N_ACT), ne[0])
bs_ = cistring.gen_strings4orblist(range(N_ACT), ne[1])
q_idx = QSpaceIndex(as_, bs_, N_ACT, ne, h1a, era)
backend = KDCIBackend(q_idx)

# ── DMRG-CI ──
ef, _ = direct_spin1.FCI().kernel(h1a, era, N_ACT, ne, nroots=NROOTS, verbose=0)
e_dmrg = [float(e) for e in np.atleast_1d(ef)[:NROOTS]]
print(f"DMRG-CI E0 = {e_dmrg[0]:.8f}", flush=True)

# ── P-space HFPT2 ──
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
        d=(hf_a^(1<<i)|(1<<a),hf_b); hij=ham.matrix_element(d,(hf_a,hf_b))
        de=E_HF-ham.matrix_element(d,d)
        if abs(de)>1e-12: scores.append((d,-hij*hij/de))
for i in bo:
    for a in bv:
        d=(hf_a,hf_b^(1<<i)|(1<<a)); hij=ham.matrix_element(d,(hf_a,hf_b))
        de=E_HF-ham.matrix_element(d,d)
        if abs(de)>1e-12: scores.append((d,-hij*hij/de))
for i1,i2 in itertools.combinations(ao,2):
    for a1,a2 in itertools.combinations(av,2):
        d=(hf_a^(1<<i1)^(1<<i2)|(1<<a1)|(1<<a2),hf_b)
        hij=ham.matrix_element(d,(hf_a,hf_b)); de=E_HF-ham.matrix_element(d,d)
        if abs(de)>1e-12: scores.append((d,-hij*hij/de))
for i1,i2 in itertools.combinations(bo,2):
    for a1,a2 in itertools.combinations(bv,2):
        d=(hf_a,hf_b^(1<<i1)^(1<<i2)|(1<<a1)|(1<<a2))
        hij=ham.matrix_element(d,(hf_a,hf_b)); de=E_HF-ham.matrix_element(d,d)
        if abs(de)>1e-12: scores.append((d,-hij*hij/de))
for i in ao:
    for j in bo:
        for a in av:
            for b in bv:
                d=(hf_a^(1<<i)|(1<<a),hf_b^(1<<j)|(1<<b))
                hij=ham.matrix_element(d,(hf_a,hf_b)); de=E_HF-ham.matrix_element(d,d)
                if abs(de)>1e-12: scores.append((d,-hij*hij/de))
scores.sort(key=lambda x:x[1],reverse=True)
p_dets=[(hf_a,hf_b)]
for det,_ in scores:
    if det not in p_dets: p_dets.append(det)
    if len(p_dets)>=P_TARGET: break
N=len(p_dets)
print(f"  P={N}", flush=True)

# ── H_PP ──
H_PP=np.zeros((N,N))
for i in range(N):
    for j in range(N): H_PP[i,j]=ham.matrix_element(p_dets[i],p_dets[j])
H_PP=0.5*(H_PP+H_PP.T)
E0_vals,_=eigh(H_PP); E0_vals=E0_vals[:NROOTS]
delta_vals = np.array([e_dmrg[k] - E0_vals[k] for k in range(NROOTS)])
print(f"  E0(P) = {E0_vals[0]:.8f}, deltas (mH): {[f'{d*1000:.0f}' for d in delta_vals]}", flush=True)

# ── H_QP ──
print("Building H_QP...", flush=True)
H_QP = backend.build_hqp(p_dets, verbose=False)
print("  done", flush=True)

# ── Test: per-state, delta=0 vs delta=exact ──
def run_per_state(delta_mode):
    """delta_mode: 'zero' or 'exact'"""
    results = {}
    for k in range(NROOTS):
        E0_k = E0_vals[k]
        dk = 0.0 if delta_mode == 'zero' else delta_vals[k]
        results_k = []

        basis = None
        for m in range(M_MAX + 1):
            tl = time.perf_counter()
            if m == 0:
                basis, d = backend.build_basis(H_QP, E0_k, verbose=False)
            else:
                basis, d = backend.propagate_basis(basis, E0_k, delta=dk, verbose=False)

            H_QQ_t, _ = backend.build_projected_blocks(basis, p_dets, H_QP=H_QP, verbose=False)
            H_PQ_t = H_QP.T @ basis

            # In H^eff, we always use delta=0 in resolvent (consistent with current approach)
            # The delta only enters the Krylov propagation
            H_eff = build_effective_H(H_PP, H_PQ_t, H_QQ_t, E0_k, delta=0.0)
            ev_all, _ = diagonalize_effective_H(H_eff, n_states=k+1)
            ev_k = float(ev_all[k])
            dE = (ev_k - e_dmrg[k]) * 1000
            wall = time.perf_counter() - tl
            results_k.append({'m': m, 'd': d, 'dE_mH': dE, 'wall': wall})
        results[k] = results_k
    return results

print("\n=== delta=0 ===", flush=True)
r0 = run_per_state('zero')
for k in range(NROOTS):
    print(f"  S{k}: m=0 dE={r0[k][0]['dE_mH']:+.1f}, m=1 dE={r0[k][1]['dE_mH']:+.1f}", flush=True)

print("\n=== delta=exact ===", flush=True)
rX = run_per_state('exact')
for k in range(NROOTS):
    print(f"  S{k}: m=0 dE={rX[k][0]['dE_mH']:+.1f}, m=1 dE={rX[k][1]['dE_mH']:+.1f}", flush=True)

print("\n=== COMPARISON ===")
print(f"{'State':>6} {'d0 m=0':>10} {'dX m=0':>10} {'d0 m=1':>10} {'dX m=1':>10}")
for k in range(NROOTS):
    print(f"  S{k:<4} {r0[k][0]['dE_mH']:>+10.1f} {rX[k][0]['dE_mH']:>+10.1f} "
          f"{r0[k][1]['dE_mH']:>+10.1f} {rX[k][1]['dE_mH']:>+10.1f}")

print("\nDone.")
