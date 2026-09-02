#!/usr/bin/env python3
"""
Phase 18b: N2/cc-pVDZ CAS(10,10) P=400, state-specific H^eff.

Clean run — no checkpoint loading. Compares new backend
(SVD on A^2*H_QP, propagate with A*H_O', state-specific resolvent)
against old Stage C results (delta=0, single-resolvent H^eff).

P-space: HFPT2 selection with full SD excitation manifold.
"""

import sys, os, time, json, itertools
import numpy as np
from numpy.linalg import eigh

sys.path.insert(0, '/data/home/wangcx/krylov-dci')

from src_mf.pyscf_backend import QSpaceIndex, KDCIBackend
from src.effective_h import build_effective_H, diagonalize_effective_H
from src.determinants import hf_determinant, bit_positions
from src.hamiltonian import Hamiltonian
from pyscf import gto, scf, ao2mo
from pyscf.fci import cistring, direct_spin1, selected_ci

N_CORE = 2; N_ACT = 10; NROOTS = 6; P_TARGET = 400; M_MAX = 3

# ── Build system ──
print("Building N2/cc-pVDZ CAS(10,10)...", flush=True)
mol = gto.M(atom='N 0 0 0; N 0 0 1.1', basis='cc-pVDZ', verbose=0)
mf = scf.RHF(mol).run(verbose=0)

n_act_orbs = list(range(N_CORE, N_CORE + N_ACT))
norb = mf.mo_coeff.shape[1]
h1e_mo = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
eri_ao = mol.intor('int2e')
eri_mo = ao2mo.full(eri_ao, mf.mo_coeff, compact=False).reshape(norb,norb,norb,norb)
h1e_act = h1e_mo[np.ix_(n_act_orbs, n_act_orbs)]
eri_act = eri_mo[np.ix_(n_act_orbs, n_act_orbs, n_act_orbs, n_act_orbs)]
nelec = (mol.nelec[0] - N_CORE, mol.nelec[1] - N_CORE)

alpha_strs = cistring.gen_strings4orblist(range(N_ACT), nelec[0])
beta_strs = cistring.gen_strings4orblist(range(N_ACT), nelec[1])

q_idx = QSpaceIndex(alpha_strs, beta_strs, N_ACT, nelec, h1e_act, eri_act)
backend = KDCIBackend(q_idx)
M = q_idx.M
print(f"  {nelec[0]}a+{nelec[1]}b in {N_ACT} orbs, M={M:,}", flush=True)

# ── DMRG-CI reference ──
print("Computing DMRG-CI reference...", flush=True)
e_fci, _ = direct_spin1.FCI().kernel(h1e_act, eri_act, N_ACT, nelec,
                                      nroots=NROOTS, verbose=0)
e_dmrg = [float(e) for e in np.atleast_1d(e_fci)[:NROOTS]]
print(f"  DMRG-CI E0 = {e_dmrg[0]:.8f} Ha", flush=True)

# ── P-space: HFPT2 with full SD manifold ──
print(f"Selecting P={P_TARGET} via HFPT2...", flush=True)
h2_4d = ao2mo.restore('s1', eri_act, N_ACT).reshape(N_ACT,N_ACT,N_ACT,N_ACT)
ham = Hamiltonian(h1=h1e_act, h2=h2_4d, E_nuc=0.0, E_HF=0.0)
hf_a, hf_b = hf_determinant(*nelec)
a_occ = bit_positions(hf_a); b_occ = bit_positions(hf_b)
av = [p for p in range(N_ACT) if p not in a_occ]
bv = [p for p in range(N_ACT) if p not in b_occ]
E_HF = ham.matrix_element((hf_a, hf_b), (hf_a, hf_b))

scores = []
# Singles
for i in a_occ:
    for a in av:
        det = (hf_a^(1<<i)|(1<<a), hf_b)
        hij = ham.matrix_element(det, (hf_a, hf_b))
        d = E_HF - ham.matrix_element(det, det)
        if abs(d) > 1e-12: scores.append((det, -hij*hij/d))
for i in b_occ:
    for a in bv:
        det = (hf_a, hf_b^(1<<i)|(1<<a))
        hij = ham.matrix_element(det, (hf_a, hf_b))
        d = E_HF - ham.matrix_element(det, det)
        if abs(d) > 1e-12: scores.append((det, -hij*hij/d))
# Same-spin doubles
for i1,i2 in itertools.combinations(a_occ, 2):
    for a1,a2 in itertools.combinations(av, 2):
        det = (hf_a^(1<<i1)^(1<<i2)|(1<<a1)|(1<<a2), hf_b)
        hij = ham.matrix_element(det, (hf_a, hf_b))
        d = E_HF - ham.matrix_element(det, det)
        if abs(d) > 1e-12: scores.append((det, -hij*hij/d))
for i1,i2 in itertools.combinations(b_occ, 2):
    for a1,a2 in itertools.combinations(bv, 2):
        det = (hf_a, hf_b^(1<<i1)^(1<<i2)|(1<<a1)|(1<<a2))
        hij = ham.matrix_element(det, (hf_a, hf_b))
        d = E_HF - ham.matrix_element(det, det)
        if abs(d) > 1e-12: scores.append((det, -hij*hij/d))
# Alpha-beta doubles
for i in a_occ:
    for j in b_occ:
        for a in av:
            for b in bv:
                det = (hf_a^(1<<i)|(1<<a), hf_b^(1<<j)|(1<<b))
                hij = ham.matrix_element(det, (hf_a, hf_b))
                d = E_HF - ham.matrix_element(det, det)
                if abs(d) > 1e-12: scores.append((det, -hij*hij/d))

scores.sort(key=lambda x: x[1], reverse=True)
p_dets = [(hf_a, hf_b)]
for det, _ in scores:
    if det not in p_dets: p_dets.append(det)
    if len(p_dets) >= P_TARGET: break
N = len(p_dets)
print(f"  N = {N} determinants", flush=True)

# ── Build H_PP ──
H_PP = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        H_PP[i,j] = ham.matrix_element(p_dets[i], p_dets[j])
H_PP = 0.5 * (H_PP + H_PP.T)

E0_vals, _ = eigh(H_PP)
E0_vals = E0_vals[:NROOTS]
E0_P = float(E0_vals[0])
delta_vals = np.array([e_dmrg[k] - E0_vals[k] for k in range(NROOTS)])
dE0_P_mH = (E0_P - e_dmrg[0]) * 1000

print(f"  E0(P) = {E0_P:.8f} Ha, P-only dE0 = {dE0_P_mH:+.3f} mH")
print(f"  Delta per state (unused): {[f'{d*1000:.1f}' for d in delta_vals]}",
      flush=True)

# ── Build H_QP ──
print("Building H_QP...", flush=True)
t0 = time.perf_counter()
H_QP = backend.build_hqp(p_dets, verbose=False)
print(f"  H_QP built in {time.perf_counter()-t0:.0f}s", flush=True)

# ── Krylov layers ──
print(f"Running Krylov-dCI m=0..{M_MAX}...", flush=True)
results = []
E0_ground = E0_vals[0]

for m in range(M_MAX + 1):
    tl = time.perf_counter()

    if m == 0:
        basis, d = backend.build_basis(H_QP, E0_ground, verbose=False)
        d_layers = [d]
    else:
        basis, d = backend.propagate_basis(basis, E0_ground, verbose=False)
        d_layers = results[-1]['d_layers'] + [d]

    H_QQ_t, H_PQ_t = backend.build_projected_blocks(
        basis, p_dets, H_QP=H_QP, verbose=False)

    # State-specific H^eff
    ev_total = []
    for k in range(NROOTS):
        H_eff_k = build_effective_H(H_PP, H_PQ_t, H_QQ_t,
                                     E0_vals[k], delta=0.0)
        ev_k, _ = diagonalize_effective_H(H_eff_k, n_states=1)
        ev_total.append(float(ev_k[0]))

    tel = time.perf_counter() - tl
    dE0 = (ev_total[0] - e_dmrg[0]) * 1000
    ex = [(ev_total[s] - e_dmrg[s]) * 1000 for s in range(1, NROOTS)]
    ex_str = '  '.join(f'S{s+1}:{ex[s]:+.0f}' for s in range(len(ex)))

    print(f"  m={m}: d={d} layers={d_layers}, dE0={dE0:+.3f} mH, "
          f"{ex_str}, wall={tel:.0f}s", flush=True)

    results.append({
        'm': m, 'd_basis': d, 'd_layers': list(d_layers),
        'dE0_mH': dE0, 'ev_total': ev_total,
        'ex_dE_mH': ex, 'wall_s': tel,
    })

# ── Comparison ──
old_stageC = [
    {'m':0,'dE0_mH':-0.689,'S1':+241.5,'S2':+311.2,'S3':+293.2,'S4':+283.7,'S5':+332.1},
    {'m':1,'dE0_mH':+5.309,'S1':+240.4,'S2':+310.4,'S3':+292.7,'S4':+283.0,'S5':+330.1},
    {'m':2,'dE0_mH':+3.451,'S1':+240.4,'S2':+310.4,'S3':+292.5,'S4':+283.0,'S5':+330.0},
    {'m':3,'dE0_mH':+3.357,'S1':+240.4,'S2':+310.4,'S3':+292.4,'S4':+283.0,'S5':+329.9},
]

print(f"\n{'='*70}")
print(f"Comparison: New (state-specific) vs Old (Stage C, delta=0)")
print(f"{'m':>3} {'NS dE0':>10} {'OS dE0':>10} {'NS S1':>8} {'OS S1':>8} "
      f"{'NS S2':>8} {'OS S2':>8}")
print('-' * 62)
for r, ro in zip(results, old_stageC):
    print(f"{r['m']:>3} {r['dE0_mH']:>+10.3f} {ro['dE0_mH']:>+10.3f} "
          f"{r['ex_dE_mH'][0]:>+8.0f} {ro['S1']:>+8.1f} "
          f"{r['ex_dE_mH'][1]:>+8.0f} {ro['S2']:>+8.1f}")

# ── Save ──
out = {
    'P': P_TARGET, 'N': N, 'M': M,
    'dE0_P_ref_mH': float(dE0_P_mH),
    'e_dmrg': e_dmrg,
    'E0_vals': [float(e) for e in E0_vals],
    'delta_vals': [float(d) for d in delta_vals],
    'results': results,
    'old_stageC': old_stageC,
}
outdir = '/data/home/wangcx/krylov-dci/checkpoints_phase18b'
os.makedirs(outdir, exist_ok=True)
with open(os.path.join(outdir, 'P0400_clean.json'), 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nSaved. Done.")
