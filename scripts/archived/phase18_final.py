#!/usr/bin/env python3
"""
Phase 18: N2/cc-pVDZ CAS(10,10) P=400, state-specific H^eff (delta=0).

Compares new backend against old Stage C results.
State-specific: H^eff for state k uses E0^(k) from H_PP eigenvalues.
Delta = 0 for fair comparison with Stage C.
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
from pyscf.fci import cistring, direct_spin1

N_CORE = 2; N_ACT = 10; NROOTS = 6; P_TARGET = 400; M_MAX = 3

print("="*60)
print("Phase 18: State-specific H^eff, delta=0")
print(f"N2/cc-pVDZ CAS(10,10) P={P_TARGET} m=0..{M_MAX}")
print("="*60)

# --- Build system ---
print("\nBuilding N2/cc-pVDZ CAS(10,10)...", flush=True)
mol = gto.M(atom='N 0 0 0; N 0 0 1.1', basis='cc-pVDZ', verbose=0)
mf = scf.RHF(mol).run(verbose=0)
na_o = list(range(N_CORE, N_CORE + N_ACT))
norb = mf.mo_coeff.shape[1]
h1_mo = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
eri_mo = ao2mo.full(mol.intor('int2e'), mf.mo_coeff, compact=False)
eri_mo = eri_mo.reshape(norb, norb, norb, norb)
h1a = h1_mo[np.ix_(na_o, na_o)]
era = eri_mo[np.ix_(na_o, na_o, na_o, na_o)]
ne = (mol.nelec[0] - N_CORE, mol.nelec[1] - N_CORE)
as_ = cistring.gen_strings4orblist(range(N_ACT), ne[0])
bs_ = cistring.gen_strings4orblist(range(N_ACT), ne[1])
q_idx = QSpaceIndex(as_, bs_, N_ACT, ne, h1a, era)
backend = KDCIBackend(q_idx)
M = q_idx.M
print(f"  {ne[0]}a+{ne[1]}b in {N_ACT} orbs, M={M:,}")

# --- DMRG-CI reference ---
print("Computing DMRG-CI reference...", flush=True)
ef, _ = direct_spin1.FCI().kernel(h1a, era, N_ACT, ne, nroots=NROOTS, verbose=0)
e_dmrg = [float(e) for e in np.atleast_1d(ef)[:NROOTS]]
print(f"  DMRG-CI E0 = {e_dmrg[0]:.8f} Ha")

# --- P-space: HFPT2 full SD ---
print(f"Selecting P={P_TARGET} via HFPT2 (full SD)...", flush=True)
h2_4d = ao2mo.restore('s1', era, N_ACT).reshape(N_ACT, N_ACT, N_ACT, N_ACT)
ham = Hamiltonian(h1=h1a, h2=h2_4d, E_nuc=0.0, E_HF=0.0)
hf_a, hf_b = hf_determinant(*ne)
ao = bit_positions(hf_a); bo = bit_positions(hf_b)
av = [p for p in range(N_ACT) if p not in ao]
bv = [p for p in range(N_ACT) if p not in bo]
E_HF = ham.matrix_element((hf_a, hf_b), (hf_a, hf_b))

scores = []
# alpha singles
for i in ao:
    for a in av:
        d = (hf_a ^ (1<<i) | (1<<a), hf_b)
        hij = ham.matrix_element(d, (hf_a, hf_b))
        den = E_HF - ham.matrix_element(d, d)
        if abs(den) > 1e-12: scores.append((d, -hij*hij/den))
# beta singles
for i in bo:
    for a in bv:
        d = (hf_a, hf_b ^ (1<<i) | (1<<a))
        hij = ham.matrix_element(d, (hf_a, hf_b))
        den = E_HF - ham.matrix_element(d, d)
        if abs(den) > 1e-12: scores.append((d, -hij*hij/den))
# aa doubles
for i1, i2 in itertools.combinations(ao, 2):
    for a1, a2 in itertools.combinations(av, 2):
        d = (hf_a ^ (1<<i1) ^ (1<<i2) | (1<<a1) | (1<<a2), hf_b)
        hij = ham.matrix_element(d, (hf_a, hf_b))
        den = E_HF - ham.matrix_element(d, d)
        if abs(den) > 1e-12: scores.append((d, -hij*hij/den))
# bb doubles
for i1, i2 in itertools.combinations(bo, 2):
    for a1, a2 in itertools.combinations(bv, 2):
        d = (hf_a, hf_b ^ (1<<i1) ^ (1<<i2) | (1<<a1) | (1<<a2))
        hij = ham.matrix_element(d, (hf_a, hf_b))
        den = E_HF - ham.matrix_element(d, d)
        if abs(den) > 1e-12: scores.append((d, -hij*hij/den))
# ab doubles
for i in ao:
    for j in bo:
        for a in av:
            for b in bv:
                d = (hf_a ^ (1<<i) | (1<<a), hf_b ^ (1<<j) | (1<<b))
                hij = ham.matrix_element(d, (hf_a, hf_b))
                den = E_HF - ham.matrix_element(d, d)
                if abs(den) > 1e-12: scores.append((d, -hij*hij/den))
scores.sort(key=lambda x: x[1], reverse=True)

p_dets = [(hf_a, hf_b)]
for det, _ in scores:
    if det not in p_dets:
        p_dets.append(det)
    if len(p_dets) >= P_TARGET:
        break
N = len(p_dets)
print(f"  N = {N} determinants (from {len(scores)} unique SD excitations)")

# --- H_PP ---
H_PP = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        H_PP[i, j] = ham.matrix_element(p_dets[i], p_dets[j])
H_PP = 0.5 * (H_PP + H_PP.T)
E0_vals, _ = eigh(H_PP)
E0_vals = E0_vals[:NROOTS]
E0_P = float(E0_vals[0])
dE0_P = (E0_P - e_dmrg[0]) * 1000
print(f"  E0(P) = {E0_P:.8f} Ha, P-only dE0 = {dE0_P:+.3f} mH")
for k in range(NROOTS):
    print(f"  E0^(state {k}) = {E0_vals[k]:.8f} Ha")

# --- H_QP ---
print("Building H_QP...", flush=True)
t0 = time.perf_counter()
H_QP = backend.build_hqp(p_dets, verbose=False)
print(f"  done in {time.perf_counter()-t0:.0f}s")

# --- Krylov layers ---
print(f"\n{'m':>3} {'d_basis':>8} {'dE0/mH':>10} {'S1':>8} {'S2':>8} "
      f"{'S3':>8} {'S4':>8} {'S5':>8} {'wall':>6}")
print("-" * 75)
results = []

for m in range(M_MAX + 1):
    tl = time.perf_counter()
    if m == 0:
        basis, d = backend.build_basis(H_QP, E0_vals[0], verbose=False)
        d_layers = [d]
    else:
        basis, d = backend.propagate_basis(basis, E0_vals[0], verbose=False)
        d_layers = results[-1]['d_layers'] + [d]

    H_QQ_t, H_PQ_t = backend.build_projected_blocks(
        basis, p_dets, H_QP=H_QP, verbose=False)

    # State-specific H^eff with delta=0
    ev_total = []
    for k in range(NROOTS):
        H_eff_k = build_effective_H(
            H_PP, H_PQ_t, H_QQ_t, E0_vals[k], delta=0.0)
        ev_k, _ = diagonalize_effective_H(H_eff_k, n_states=1)
        ev_total.append(float(ev_k[0]))

    tel = time.perf_counter() - tl
    dE0 = (ev_total[0] - e_dmrg[0]) * 1000
    ex = [(ev_total[i] - e_dmrg[i]) * 1000 for i in range(1, NROOTS)]

    print(f"{m:>3} {d:>8} {dE0:>+10.3f} "
          + "".join(f"{ex[i]:>+8.0f}" for i in range(5))
          + f" {tel:>6.0f}", flush=True)

    results.append({
        'm': m, 'd_basis': d, 'd_layers': list(d_layers),
        'dE0_mH': dE0, 'ev_total': ev_total,
        'ex_dE_mH': ex, 'wall_s': tel,
    })

# --- Comparison with Stage C ---
old = [
    {'m':0,'dE0':-0.689,'ex':[241.5,311.2,293.2,283.7,332.1]},
    {'m':1,'dE0':+5.309,'ex':[240.4,310.4,292.7,283.0,330.1]},
    {'m':2,'dE0':+3.451,'ex':[240.4,310.4,292.5,283.0,330.0]},
    {'m':3,'dE0':+3.357,'ex':[240.4,310.4,292.4,283.0,329.9]},
]
print(f"\n{'='*70}")
print("Comparison: New vs Old Stage C")
print(f"{'m':>3} {'NS dE0':>10} {'OS dE0':>10} {'NS S1':>8} {'OS S1':>8}")
print("-" * 48)
for r, ro in zip(results, old):
    print(f"{r['m']:>3} {r['dE0_mH']:>+10.3f} {ro['dE0']:>+10.3f} "
          f"{r['ex_dE_mH'][0]:>+8.0f} {ro['ex'][0]:>+8.1f}")

# Save
out = {
    'P': P_TARGET, 'N': N, 'M': M,
    'dE0_P_ref_mH': float(dE0_P),
    'e_dmrg': e_dmrg, 'E0_vals': [float(e) for e in E0_vals],
    'results': results, 'old_stageC': old,
}
d = '/data/home/wangcx/krylov-dci/checkpoints_phase18b'
os.makedirs(d, exist_ok=True)
with open(d + '/P0400_clean.json', 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nSaved. Total wall: {time.perf_counter()-t0:.0f}s")
