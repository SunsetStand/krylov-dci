#!/usr/bin/env python3
"""
Step 2: Bloch H^eff batch evaluation over P-space convergence checkpoints.

Loads checkpoints from Step 1, computes per-state m=0 Bloch H^eff for
each P size, and compares:
  - bare H_PP energies (dCI-like)
  - Bloch H^eff energies (our method)

The gap between these two curves is the Bloch correction contribution.
If Bloch converges faster than bare, the method has unique value.

System: N2/cc-pVDZ CAS(10,10)
"""
import sys, os, time, json
import numpy as np
from numpy.linalg import eigh

sys.path.insert(0, '/data/home/wangcx/krylov-dci')
from src_mf.pyscf_backend import QSpaceIndex, KDCIBackend
from src.effective_h import build_effective_H, diagonalize_effective_H
from src.determinants import hf_determinant
from src.hamiltonian import Hamiltonian
from pyscf import gto, scf, ao2mo
from pyscf.fci import cistring, direct_spin1

# ── Parameters ────────────────────────────────────────────────────
N_CORE = 2
N_ACT = 10
NROOTS = 6
DELTA = 0.0                    # no level shift for clean comparison
P_TARGETS = [200, 400, 800, 1200, 1600, 2000]
INDIR = '/data/home/wangcx/krylov-dci/checkpoints_pspace'
OUTDIR = '/data/home/wangcx/krylov-dci/checkpoints_pspace'
# ───────────────────────────────────────────────────────────────────

print("=" * 64)
print("Step 2: Bloch H^eff Batch Evaluation")
print(f"N2/cc-pVDZ CAS({N_ACT},{N_ACT})  delta={DELTA}  nroots={NROOTS}")
print(f"P sizes: {P_TARGETS}")
print("=" * 64)

# ── Build system ──────────────────────────────────────────────────
print("\n[1] Building N2/cc-pVDZ CAS(10,10)...", flush=True)
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
print(f"  M={M:,}", flush=True)

# ── DMRG-CI reference ─────────────────────────────────────────────
print("[2] Computing DMRG-CI reference...", flush=True)
ef, _ = direct_spin1.FCI().kernel(h1a, era, N_ACT, ne,
                                   nroots=NROOTS, verbose=0)
e_dmrg = [float(e) for e in np.atleast_1d(ef)[:NROOTS]]
print(f"  E_FCI = {e_dmrg[0]:.8f} Ha", flush=True)

# ── Prepare Hamiltonian ───────────────────────────────────────────
h2_4d = ao2mo.restore('s1', era, N_ACT).reshape(N_ACT, N_ACT, N_ACT, N_ACT)
ham = Hamiltonian(h1=h1a, h2=h2_4d, E_nuc=0.0, E_HF=0.0)

# ── Process each checkpoint ───────────────────────────────────────
results = {}
wall_total0 = time.perf_counter()

for p_size in P_TARGETS:
    fname = f"{INDIR}/step1_P{p_size:04d}.json"
    if not os.path.exists(fname):
        print(f"\n  ⚠ Skipping P={p_size}: checkpoint not found", flush=True)
        continue

    print(f"\n[P={p_size}] Loading checkpoint...", flush=True)
    with open(fname) as f:
        ckpt = json.load(f)

    p_dets = [(int(a), int(b)) for a, b in ckpt['p_dets']]
    E_bare = ckpt['E_bare']
    N = len(p_dets)

    t0 = time.perf_counter()

    # 2a. Build H_PP
    print(f"  Building H_PP ({N}×{N})...", flush=True)
    H_PP = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            H_PP[i, j] = ham.matrix_element(p_dets[i], p_dets[j])
    H_PP = 0.5 * (H_PP + H_PP.T)
    E0_vals, _ = eigh(H_PP)
    E0_vals = E0_vals[:NROOTS]

    t_hpp = time.perf_counter() - t0

    # 2b. Build H_QP (sparse)
    print(f"  Building H_QP (Q={M-N:,} × P={N})...", flush=True)
    t1 = time.perf_counter()
    H_QP = backend.build_hqp(p_dets, verbose=False)
    t_hqp = time.perf_counter() - t1

    # 2c. Per-state Bloch correction (m=0, delta=0)
    print(f"  Computing per-state Bloch H^eff...", flush=True)
    E_bloch = []
    H_QQ_diag = q_idx.hdiag

    for k in range(NROOTS):
        E0_k = E0_vals[k]

        # A_q = (E0_k - H_QQ)^(-1), diagonal approximation (m=0)
        A_q_diag = 1.0 / (E0_k + DELTA - H_QQ_diag)
        # Cap near-singular values
        A_q_diag = np.clip(A_q_diag, -1e10, 1e10)

        # Correction = H_QP^T · diag(A_q) · H_QP
        # H_QP is Q×P in dense (from build_hqp returns dense)
        if isinstance(H_QP, np.ndarray):
            weighted = H_QP * A_q_diag[:, np.newaxis]  # Q×P
        else:
            # sparse
            weighted = H_QP.multiply(A_q_diag[:, np.newaxis])

        correction = H_QP.T @ weighted  # P×P
        H_eff = H_PP + correction
        H_eff = 0.5 * (H_eff + H_eff.T)

        ev, _ = eigh(H_eff)
        E_bloch.append(float(ev[k]))

    t_heff = time.perf_counter() - t1

    # 2d. Compute errors
    dE_bare = [(E_bare[i] - e_dmrg[i]) * 1000 for i in range(NROOTS)]
    dE_bloch = [(E_bloch[i] - e_dmrg[i]) * 1000 for i in range(NROOTS)]
    improvement = [dE_bare[i] - dE_bloch[i] for i in range(NROOTS)]

    wall_this = time.perf_counter() - t0

    results[p_size] = {
        'P': p_size, 'N': N, 'M': M,
        'E_bare': [float(e) for e in E_bare],
        'E_bloch': E_bloch,
        'e_dmrg': e_dmrg,
        'dE_bare_mH': dE_bare,
        'dE_bloch_mH': dE_bloch,
        'improvement_mH': [float(imp) for imp in improvement],
        'timing': {
            't_hpp': t_hpp,
            't_hqp': t_hqp,
            't_heff': t_heff,
            't_total': wall_this,
        },
        'nroots': NROOTS,
        'delta': DELTA,
    }

    # Save individual result
    outf = f"{OUTDIR}/step2_P{p_size:04d}.json"
    with open(outf, 'w') as f:
        json.dump(results[p_size], f, indent=2)

    # Print summary table
    print(f"\n  {'root':>5} {'FCI (Ha)':>14} {'bare dE(mH)':>14} "
          f"{'Bloch dE(mH)':>14} {'improve(mH)':>13}", flush=True)
    print(f"  {'-'*60}", flush=True)
    for k in range(NROOTS):
        print(f"  {k:>5} {e_dmrg[k]:>14.8f} {dE_bare[k]:>+14.3f} "
              f"{dE_bloch[k]:>+14.3f} {improvement[k]:>+13.3f}", flush=True)

    print(f"  ─ H_PP: {t_hpp:.1f}s  H_QP: {t_hqp:.1f}s  "
          f"H^eff: {t_heff:.1f}s  total: {wall_this:.1f}s", flush=True)

# ── Global summary ────────────────────────────────────────────────
wall_total = time.perf_counter() - wall_total0

print(f"\n{'='*70}")
print("P-space Convergence Summary")
print(f"{'P':>6}", end="")
for k in range(NROOTS):
    print(f" {'bare_dE'+str(k):>11}", end="")
print(f" {'bloch_dE0':>11}")
print("-" * 70)

for p_size in P_TARGETS:
    if p_size not in results:
        continue
    r = results[p_size]
    print(f"{p_size:>6}", end="")
    for k in range(NROOTS):
        print(f" {r['dE_bare_mH'][k]:>+11.3f}", end="")
    print(f" {r['dE_bloch_mH'][0]:>+11.3f}", flush=True)

print(f"\nBloch Correction Impact (improvement = bare_error - bloch_error)")
print(f"{'P':>6}", end="")
for k in range(NROOTS):
    print(f" {'Δ_err'+str(k):>11}", end="")
print(f" {'Δ_err_max':>11}")
print("-" * 70)

for p_size in P_TARGETS:
    if p_size not in results:
        continue
    r = results[p_size]
    print(f"{p_size:>6}", end="")
    for k in range(NROOTS):
        print(f" {r['improvement_mH'][k]:>+11.3f}", end="")
    max_imp = max(abs(v) for v in r['improvement_mH'])
    print(f" {max_imp:>11.3f}", flush=True)

# Save master summary
summary = {
    'system': 'N2/cc-pVDZ CAS(10,10)',
    'method': 'm=0 per-state Bloch H^eff',
    'delta': DELTA,
    'nroots': NROOTS,
    'wall_total_s': wall_total,
    'results': {str(k): v for k, v in results.items()},
}
with open(f"{OUTDIR}/summary.json", 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n{'='*64}")
print(f"Step 2 complete. {wall_total:.0f}s total wall.")
print(f"Summary saved to {OUTDIR}/summary.json")
print(f"{'='*64}")
