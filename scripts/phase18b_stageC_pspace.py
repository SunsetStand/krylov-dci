#!/usr/bin/env python3
"""
Phase 18b: m-convergence with state-specific H^eff, using Stage C P-space.

Loads P=400 P-space from Stage C checkpoint, runs with the new
Krylov-dCI backend (SVD on A^2*H_QP, propagate with A*H_O'*B,
A-weighted SVD truncation, state-specific resolvent).
"""

import sys, os, time, json
import numpy as np
from numpy.linalg import eigh

PROJECT_ROOT = '/data/home/wangcx/krylov-dci'
sys.path.insert(0, PROJECT_ROOT)

from src_mf.pyscf_backend import QSpaceIndex, KDCIBackend
from src.effective_h import build_effective_H, diagonalize_effective_H
from pyscf import gto, scf, ao2mo
from pyscf.fci import cistring, direct_spin1
from scipy import sparse

N_CORE = 3; N_ACT = 10; BOND_LENGTH = 1.10
NROOTS = 6; M_MAX = 3; P_TARGET = 400

# ── Build N2 CAS(10,10) system ──
print("Building N2/cc-pVDZ CAS(10,10)...", flush=True)
mol = gto.M(atom='N 0 0 0; N 0 0 1.1', basis='cc-pVDZ', verbose=0)
mf = scf.RHF(mol).run(verbose=0)
n_core_orbs = list(range(N_CORE))
n_act_orbs = list(range(N_CORE, N_CORE + N_ACT))
norb = mf.mo_coeff.shape[1]
h1e_mo = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
eri_ao = mol.intor('int2e')
eri_mo = ao2mo.full(eri_ao, mf.mo_coeff, compact=False)
eri_mo = eri_mo.reshape(norb, norb, norb, norb)
h1e_act = h1e_mo[np.ix_(n_act_orbs, n_act_orbs)]
eri_act = eri_mo[np.ix_(n_act_orbs, n_act_orbs, n_act_orbs, n_act_orbs)]
nelec = (mol.nelec[0] - N_CORE, mol.nelec[1] - N_CORE)
alpha_strs = cistring.gen_strings4orblist(range(N_ACT), nelec[0])
beta_strs = cistring.gen_strings4orblist(range(N_ACT), nelec[1])

q_idx = QSpaceIndex(alpha_strs, beta_strs, N_ACT, nelec, h1e_act, eri_act)
backend = KDCIBackend(q_idx)
M = q_idx.M
print(f"  Active: {nelec[0]}a+{nelec[1]}b in {N_ACT} orbs, M={M:,}", flush=True)

# ── DMRG-CI reference ──
print("Computing DMRG-CI reference...", flush=True)
e_fci, _ = direct_spin1.FCI().kernel(h1e_act, eri_act, N_ACT, nelec,
                                      nroots=NROOTS, verbose=0)
e_dmrg = [float(e) for e in np.atleast_1d(e_fci)[:NROOTS]]
print(f"  DMRG-CI E0 = {e_dmrg[0]:.8f} Ha", flush=True)

# ── Load Stage C P-space ──
print(f"\nLoading Stage C P={P_TARGET}...", flush=True)
ps = np.load(os.path.join(PROJECT_ROOT, 'checkpoints_stageC',
                          'P0400', 'p_space.npz'), allow_pickle=True)
p_alpha = [int(a) for a in ps['p_alpha']]
p_beta = [int(b) for b in ps['p_beta']]
p_dets = list(zip(p_alpha, p_beta))
N = len(p_dets)
print(f"  N = {N} determinants", flush=True)

pb = np.load(os.path.join(PROJECT_ROOT, 'checkpoints_stageC',
                          'P0400', 'p_blocks.npz'), allow_pickle=True)
H_PP = pb['H_PP']
E0_stageC = float(pb['E0_P'])
print(f"  E0(P)_stageC = {E0_stageC:.8f} Ha", flush=True)

# ── Build H_QP via new backend ──
print("Building H_QP via contract_2e backend...", flush=True)
t0 = time.perf_counter()
H_QP_new = backend.build_hqp(p_dets, verbose=False)
print(f"  H_QP built in {time.perf_counter()-t0:.0f}s", flush=True)

# ── Verify H_PP and E0 ──
E0_vals, _ = eigh(H_PP)
E0_vals = E0_vals[:NROOTS]
E0_P = float(E0_vals[0])
delta_vals = np.array([e_dmrg[k] - E0_vals[k] for k in range(NROOTS)])
dE0_P_mH = (E0_P - e_dmrg[0]) * 1000

print(f"  E0(P)           = {E0_P:.8f} Ha")
print(f"  P-only dE0      = {dE0_P_mH:+.3f} mH")
print(f"  E0(P) per state = {[f'{e:.4f}' for e in E0_vals[:NROOTS]]}")
print(f"  Delta per state (mH) = {[f'{d*1000:.1f}' for d in delta_vals]}",
      flush=True)

# ── Run Krylov-dCI layers ──
print(f"\nRunning Krylov-dCI layers m=0..{M_MAX}...", flush=True)
results = []
E0_ground = E0_vals[0]

for m in range(M_MAX + 1):
    t_layer = time.perf_counter()

    # Build basis
    if m == 0:
        basis, d_total = backend.build_basis(H_QP_new, E0_ground, verbose=False)
        d_layers = [d_total]
    else:
        basis, d_total = backend.propagate_basis(basis, E0_ground, verbose=False)
        d_layers = results[-1]['d_layers'] + [d_total]

    # Build projected blocks
    H_QQ_t, H_PQ_t = backend.build_projected_blocks(
        basis, p_dets, H_QP=H_QP_new, verbose=False)

    # State-specific effective Hamiltonians
    ev_total = []
    for k in range(NROOTS):
        E0_k = E0_vals[k]
        delta_k = delta_vals[k]
        H_eff_k = build_effective_H(H_PP, H_PQ_t, H_QQ_t, E0_k, delta=delta_k)
        ev_k, _ = diagonalize_effective_H(H_eff_k, n_states=1)
        ev_total.append(float(ev_k[0]))

    t_elapsed = time.perf_counter() - t_layer

    dE0_mH = (ev_total[0] - e_dmrg[0]) * 1000
    ex_dE = [(ev_total[i] - e_dmrg[i]) * 1000 for i in range(1, NROOTS)]

    ex_str = '  '.join(f'S{s+1}:{ex_dE[s]:+.0f}'
                        for s in range(len(ex_dE)))
    print(f"  m={m}: d={d_total} layers={d_layers}, "
          f"dE0={dE0_mH:+.3f} mH, {ex_str}, wall={t_elapsed:.0f}s",
          flush=True)

    results.append({
        'm': m, 'd_basis': d_total, 'd_layers': list(d_layers),
        'dE0_mH': dE0_mH, 'ev_total': ev_total,
        'ex_dE_mH': ex_dE, 'wall_s': t_elapsed,
    })

# ── Comparison with old Stage C ──
old_stageC = [
    {'m': 0, 'd_basis': 400, 'dE0_mH': -0.689},
    {'m': 1, 'd_basis': 800, 'dE0_mH': +5.309},
    {'m': 2, 'd_basis': 1200, 'dE0_mH': +3.451},
    {'m': 3, 'd_basis': 1600, 'dE0_mH': +3.357},
]

print(f"\n{'='*70}")
print(f"Comparison: New (state-specific H^eff) vs Old (Stage C)")
print(f"{'m':>3}  {'d_basis':>8}  {'New dE0/mH':>12}  {'Old dE0/mH':>12}  {'Diff':>10}")
print(f"{'-'*3}  {'-'*8}  {'-'*12}  {'-'*12}  {'-'*10}")
for r, ro in zip(results, old_stageC):
    diff = r['dE0_mH'] - ro['dE0_mH']
    print(f"{r['m']:>3}  {r['d_basis']:>8}  {r['dE0_mH']:>+12.3f}  "
          f"{ro['dE0_mH']:>+12.3f}  {diff:>+10.3f}")

# ── Per-state comparison ──
print(f"\n{'='*70}")
print(f"Per-State Errors (m=0): New vs Old Stage C")
old_m0_ex = [241.5, 311.2, 293.2, 283.7, 332.1]  # from Stage C P=400 m=0
print(f"{'State':>6}  {'New dE/mH':>11}  {'Old dE/mH':>11}")
print(f"{'------':>6}  {'----------':>11}  {'----------':>11}")
for s in range(len(results[0]['ex_dE_mH'])):
    print(f"  S{s+1}    {results[0]['ex_dE_mH'][s]:>+11.1f}  {old_m0_ex[s]:>+11.1f}")

# ── Save ──
out = {
    'P': P_TARGET, 'N': N, 'M': M,
    'dE0_P_ref_mH': float(dE0_P_mH),
    'E0_stageC': float(E0_stageC),
    'e_dmrg_total': e_dmrg,
    'results': results,
    'old_stageC': old_stageC,
}
outdir = os.path.join(PROJECT_ROOT, 'checkpoints_phase18b')
os.makedirs(outdir, exist_ok=True)
outpath = os.path.join(outdir, 'P0400_new_vs_old.json')
with open(outpath, 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nSaved to {outpath}")
print("Done.")
