#!/usr/bin/env python3
"""
Phase 18 — BUG-FIXED: state-specific H^eff, delta=0.

Fixes vs phase18b/phase18_final:
  🔧 Fix 1: H_PQ_t = H_QP.T @ basis  (was: sigma_all[p_flat,:] → P-P pollution)
  🔧 Fix 2: n_states=k+1, take ev[k]   (was: n_states=1 always → wrong root)

Setup: N2/cc-pVDZ, N_CORE=2 (matching effective Stage C), P=400 HFPT2,
       delta=0, m=0..3.
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

N_CORE = 2; N_ACT = 10; BOND_LENGTH = 1.10
NROOTS = 6; M_MAX = 3; P_TARGET = 400

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
for i in range(1, NROOTS):
    print(f"  DMRG-CI E{i} = {e_dmrg[i]:.8f} Ha  "
          f"({1000*(e_dmrg[i]-e_dmrg[0]):.1f} mH)", flush=True)

# ── P-space: HFPT2 with full SD manifold ──
print(f"\nSelecting P={P_TARGET} via HFPT2...", flush=True)
t0 = time.perf_counter()
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
print(f"  N = {N} determinants ({time.perf_counter()-t0:.0f}s)", flush=True)

# ── Build H_PP ──
H_PP = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        H_PP[i,j] = ham.matrix_element(p_dets[i], p_dets[j])
H_PP = 0.5 * (H_PP + H_PP.T)

E0_vals, _ = eigh(H_PP)
E0_vals = E0_vals[:NROOTS]
E0_P = float(E0_vals[0])
dE0_P_mH = (E0_P - e_dmrg[0]) * 1000

print(f"  E0(P)           = {E0_P:.8f} Ha")
print(f"  P-only dE0      = {dE0_P_mH:+.3f} mH")
print(f"  E0(P) per state (Ha) = {[f'{e:.5f}' for e in E0_vals[:NROOTS]]}")
print(f"  delta = 0.0 for all states", flush=True)

# ── Build H_QP ──
print("Building H_QP via contract_2e backend...", flush=True)
t0 = time.perf_counter()
H_QP = backend.build_hqp(p_dets, verbose=False)
print(f"  H_QP built in {time.perf_counter()-t0:.0f}s", flush=True)

# ── Stage C reference results (same P-space size, HFPT2, N_CORE=2) ──
old_stageC = [
    {'m': 0, 'd_basis': 400, 'dE0_mH': -0.689,
     'ex': [241.5, 311.2, 293.2, 283.7, 332.1]},
    {'m': 1, 'd_basis': 800, 'dE0_mH': +5.309,
     'ex': [240.4, 310.4, 292.7, 283.0, 330.1]},
    {'m': 2, 'd_basis': 1200, 'dE0_mH': +3.451,
     'ex': [240.4, 310.4, 292.5, 283.0, 330.0]},
    {'m': 3, 'd_basis': 1600, 'dE0_mH': +3.357,
     'ex': [240.4, 310.4, 292.4, 283.0, 329.9]},
]

# ── Run Krylov-dCI layers ──
print(f"\n--- Krylov-dCI layers (m=0..{M_MAX}, delta=0) ---")
print(f"{'m':>3} {'d_basis':>8} {'d_layers':>22} {'dE0/mH':>10} "
      f"{'S1':>8} {'S2':>8} {'S3':>8} {'S4':>8} {'S5':>8} {'wall':>6}")
print("-" * 90)
results = []
E0_ground = E0_vals[0]

for m in range(M_MAX + 1):
    t_layer = time.perf_counter()

    # Step A: Build Krylov basis
    if m == 0:
        basis, d_total = backend.build_basis(H_QP, E0_ground, verbose=False)
        d_layers = [d_total]
    else:
        basis, d_total = backend.propagate_basis(basis, E0_ground, verbose=False)
        d_layers = results[-1]['d_layers'] + [d_total]

    # 🔧 Fix 3: Project out P-space components from basis.
    # After propagation, basis vectors contain P-admixture (H couples Q↔P).
    # The effective-Hamiltonian theory requires Q̃ ⊂ Q — the compressed
    # basis must lie entirely in Q-space. Zero P-rows, then re-orthonormalize.
    p_flat = q_idx.p_indices(p_dets)
    p_valid_mask = p_flat >= 0
    p_rows = p_flat[p_valid_mask]
    if len(p_rows) > 0:
        basis[p_rows, :] = 0.0
        Q_r, R_r = np.linalg.qr(basis)
        r_keep = np.sum(np.abs(np.diag(R_r)) > 1e-10)
        basis = Q_r[:, :r_keep]
        d_total = basis.shape[1]

    # Step B: Projected blocks
    # 🔧 Fix 1: H_PQ_t = H_QP^T @ basis — avoids P-P contamination.
    #           (basis already P-projected, so sigma_all approach would
    #            also be correct now, but H_QP^T is cheaper.)
    H_QQ_t, _ = backend.build_projected_blocks(
        basis, p_dets, H_QP=H_QP, verbose=False)
    H_PQ_t = H_QP.T @ basis

    # Step C: State-specific H^eff, delta=0
    # 🔧 Fix 2: n_states=k+1, take ev[k] — correct state indexing
    ev_total = []
    for k in range(NROOTS):
        H_eff_k = build_effective_H(H_PP, H_PQ_t, H_QQ_t,
                                     E0_vals[k], delta=0.0)
        ev_all_k, _ = diagonalize_effective_H(H_eff_k, n_states=k+1)
        ev_total.append(float(ev_all_k[k]))

    t_elapsed = time.perf_counter() - t_layer

    dE0_mH = (ev_total[0] - e_dmrg[0]) * 1000
    ex_dE = [(ev_total[i] - e_dmrg[i]) * 1000 for i in range(1, NROOTS)]

    ex_str = '  '.join(f'{ex_dE[s]:+8.0f}' for s in range(len(ex_dE)))
    print(f"{m:>3} {d_total:>8} {str(d_layers):>22} {dE0_mH:>+10.3f}  "
          f"{ex_str}  {t_elapsed:>5.0f}s", flush=True)

    results.append({
        'm': m, 'd_basis': d_total, 'd_layers': list(d_layers),
        'dE0_mH': dE0_mH, 'ev_total': ev_total,
        'ex_dE_mH': ex_dE, 'wall_s': t_elapsed,
    })

# ── Comparison ──
print(f"\n{'='*90}")
print(f"Comparison: FIXED Phase 18 (delta=0)  vs  Old Stage C (delta=0)")
print(f"{'m':>3} {'New dE0':>10} {'Old dE0':>10} {'New S1':>8} "
      f"{'Old S1':>8} {'New S2':>8} {'Old S2':>8} {'New S3':>8} {'Old S3':>8}")
print("-" * 80)
for r, ro in zip(results, old_stageC):
    print(f"{r['m']:>3} {r['dE0_mH']:>+10.3f} {ro['dE0_mH']:>+10.3f} "
          f"{r['ex_dE_mH'][0]:>+8.0f} {ro['ex'][0]:>+8.1f} "
          f"{r['ex_dE_mH'][1]:>+8.0f} {ro['ex'][1]:>+8.1f} "
          f"{r['ex_dE_mH'][2]:>+8.0f} {ro['ex'][2]:>+8.1f}")

# ── Per-state detail (m=0) ──
print(f"\n{'='*90}")
print(f"Per-State detail (m=0)")
old_m0 = old_stageC[0]['ex']
print(f"{'State':>6} {'New dE/mH':>11} {'Old dE/mH':>11} {'Diff':>10}")
print("-" * 42)
for s in range(len(results[0]['ex_dE_mH'])):
    diff = results[0]['ex_dE_mH'][s] - old_m0[s]
    print(f"  S{s+1}    {results[0]['ex_dE_mH'][s]:>+11.1f}  "
          f"{old_m0[s]:>+11.1f}  {diff:>+10.1f}")

# ── Save ──
out = {
    'bugfix': 'FIX1_H_PQ_t_via_H_QP_T_dot_basis__FIX2_n_states_kplus1',
    'delta_mode': 'zero',
    'P': P_TARGET, 'N': N, 'M': M, 'N_CORE': N_CORE,
    'dE0_P_ref_mH': float(dE0_P_mH),
    'e_dmrg_total': e_dmrg,
    'E0_vals': [float(e) for e in E0_vals],
    'results': results,
    'old_stageC': old_stageC,
}
outdir = os.path.join(PROJECT_ROOT, 'checkpoints_phase18_fixed')
os.makedirs(outdir, exist_ok=True)
outpath = os.path.join(outdir, 'P0400_fixed.json')
with open(outpath, 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nSaved to {outpath}")
print("Done.")
