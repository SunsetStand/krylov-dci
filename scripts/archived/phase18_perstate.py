#!/usr/bin/env python3
"""
Phase 18 — Per-State Krylov Basis: state-specific H^eff, delta=0.

Each state k gets its OWN Krylov basis built at E0_vals[k]:
  - m=0: basis = MGS(A_k * H_QP), A_k = 1/(E0_k - H_D[qq])
  - m≥1: propagate with A_k-weighted resolvent

This is the correct approach: the Krylov subspace must represent
(E0_k - H_QQ)^{-1} for each state individually, not a single
ground-state resolvent for all states.

Bug fixes (vs old Phase 18):
  🔧 Fix 1: H_PQ_t = H_QP.T @ basis  (P-P contamination)
  🔧 Fix 2: n_states=k+1, take ev[k]  (wrong root)
  (No P-projection — that was harmful.)

N2/cc-pVDZ, N_CORE=2, P=400 HFPT2, delta=0.
"""
import sys, os, time, json, itertools
import numpy as np
from numpy.linalg import eigh

PROJECT_ROOT = '/data/home/wangcx/krylov-dci'
sys.path.insert(0, PROJECT_ROOT)
from src_mf.pyscf_backend import QSpaceIndex, KDCIBackend
from src.effective_h import build_effective_H, diagonalize_effective_H
from src.determinants import hf_determinant, bit_positions
from src.hamiltonian import Hamiltonian
from pyscf import gto, scf, ao2mo
from pyscf.fci import cistring, direct_spin1

N_CORE, N_ACT, BOND_LENGTH = 2, 10, 1.10
NROOTS, M_MAX, P_TARGET = 6, 1, 400  # M_MAX=1 for fast per-state test

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
M = q_idx.M
print(f"  {ne[0]}a+{ne[1]}b in {N_ACT} orbs, M={M:,}", flush=True)

# ── DMRG-CI ──
print("DMRG-CI reference...", flush=True)
ef, _ = direct_spin1.FCI().kernel(h1a, era, N_ACT, ne, nroots=NROOTS, verbose=0)
e_dmrg = [float(e) for e in np.atleast_1d(ef)[:NROOTS]]
print(f"  E0 = {e_dmrg[0]:.8f} Ha", flush=True)

# ── P-space HFPT2 ──
print(f"P={P_TARGET} via HFPT2...", flush=True)
t0 = time.perf_counter()
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
print(f"  N={N} ({time.perf_counter()-t0:.0f}s)", flush=True)

# ── H_PP ──
H_PP=np.zeros((N,N))
for i in range(N):
    for j in range(N): H_PP[i,j]=ham.matrix_element(p_dets[i],p_dets[j])
H_PP=0.5*(H_PP+H_PP.T)
E0_vals,_=eigh(H_PP); E0_vals=E0_vals[:NROOTS]
dE0_P=(E0_vals[0]-e_dmrg[0])*1000
print(f"  E0(P)={E0_vals[0]:.8f} Ha, P-only dE0={dE0_P:+.3f} mH", flush=True)
for k in range(NROOTS):
    print(f"  E0^(state {k}) = {E0_vals[k]:.8f} Ha  "
          f"(delta={e_dmrg[k]-E0_vals[k]:+.6f} Ha)", flush=True)

# ── H_QP ──
print("Building H_QP...", flush=True)
t0=time.perf_counter(); H_QP=backend.build_hqp(p_dets,verbose=False)
print(f"  done in {time.perf_counter()-t0:.0f}s", flush=True)

# ── Stage C reference ──
stgC = { # m, dE0_mH, [S1..S5]_mH
    0: (-0.689, [241.5, 311.2, 293.2, 283.7, 332.1]),
    1: (+5.309, [240.4, 310.4, 292.7, 283.0, 330.1]),
    2: (+3.451, [240.4, 310.4, 292.5, 283.0, 330.0]),
}

# ── Per-state Krylov-dCI ──
print(f"\n{'='*90}")
print("Per-State Krylov Basis: each state k builds its own Krylov subspace")
print(f"at E0_vals[k], propagates independently, delta=0.")
print(f"{'='*90}")

all_results = {}  # state_k -> list of {m, d, dE, ev}

for k in range(NROOTS):
    E0_k = E0_vals[k]
    print(f"\n--- State {k} (E0={E0_k:.8f} Ha) ---")
    print(f"  {'m':>3} {'d_basis':>8} {'E/mH':>13} {'dE vs exact':>12} {'wall':>6}")
    print(f"  {'-'*3} {'-'*8} {'-'*13} {'-'*12} {'-'*6}")
    results_k = []
    basis = None

    for m in range(M_MAX + 1):
        tl = time.perf_counter()

        # Build per-state Krylov basis at E0_k
        if m == 0:
            basis, d = backend.build_basis(H_QP, E0_k, verbose=False)
        else:
            basis, d = backend.propagate_basis(basis, E0_k, verbose=False)

        # Projected blocks  (Fix 1: H_PQ_t from H_QP^T)
        H_QQ_t, _ = backend.build_projected_blocks(
            basis, p_dets, H_QP=H_QP, verbose=False)
        H_PQ_t = H_QP.T @ basis

        # State-specific H^eff for state k  (Fix 2: n_states=k+1, take ev[k])
        H_eff = build_effective_H(H_PP, H_PQ_t, H_QQ_t, E0_k, delta=0.0)
        ev_all, _ = diagonalize_effective_H(H_eff, n_states=k+1)
        ev_k = float(ev_all[k])

        wall = time.perf_counter() - tl
        dE = (ev_k - e_dmrg[k]) * 1000

        print(f"  {m:>3} {d:>8} {ev_k:>13.8f} {dE:>+12.3f} {wall:>6.0f}s")

        results_k.append({'m': m, 'd': d, 'E': ev_k, 'dE_mH': dE, 'wall': wall})

    all_results[k] = results_k

# ── Summary tables ──
print(f"\n{'='*90}")
print("Cross-State Summary: dE vs DMRG-CI (mH)")
print(f"{'='*90}")

# Table 1: per-m comparison
print(f"\n  Per m-layer, all states:")
print(f"  {'m':>3}  {'S0':>8} {'S1':>8} {'S2':>8} {'S3':>8} {'S4':>8} {'S5':>8}")
for m in range(M_MAX + 1):
    vals = [f"{all_results[k][m]['dE_mH']:+8.1f}" for k in range(NROOTS)]
    print(f"  {m:>3}  {' '.join(vals)}")

# Table 2: excited states only, vs Stage C
print(f"\n  Per m-layer, |dE| for excited states: New vs Stage C")
for m in range(M_MAX + 1):
    new_ex = [all_results[k][m]['dE_mH'] for k in range(1,NROOTS)]
    if m in stgC:
        old_ex = stgC[m][1]
        print(f"  m={m}: New {[f'{x:+.0f}' for x in new_ex]}  "
              f"Old {[f'{x:+.1f}' for x in old_ex]}")
    else:
        print(f"  m={m}: New {[f'{x:+.0f}' for x in new_ex]}")

# Table 3: absolute errors comparison
print(f"\n  Max(|dE|) across excited states:")
print(f"  {'m':>3}  {'New max':>10}  {'Old max':>10}")
for m in range(M_MAX + 1):
    new_max = max(abs(all_results[k][m]['dE_mH']) for k in range(1,NROOTS))
    old_max = max(abs(x) for x in stgC.get(m,(0,[0]*5))[1]) if m in stgC else float('nan')
    print(f"  {m:>3}  {new_max:>10.1f}  {old_max:>10.1f}")

# ── Save ──
out = {
    'method': 'per_state_krylov_basis_delta_zero',
    'P': P_TARGET, 'N': N, 'M': M, 'N_CORE': N_CORE,
    'dE0_P_mH': float(dE0_P),
    'e_dmrg': e_dmrg,
    'E0_vals': [float(e) for e in E0_vals],
    'results': {str(k): all_results[k] for k in range(NROOTS)},
    'stageC_ref': stgC,
}
outdir = os.path.join(PROJECT_ROOT, 'checkpoints_phase18_perstate')
os.makedirs(outdir, exist_ok=True)
with open(os.path.join(outdir, 'P0400_perstate.json'), 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nSaved. Done.")
