#!/usr/bin/env python3
"""
Step 3: Iterative P + Krylov build_basis + per-state Bloch H^eff.

Combines the best of both approaches:
  1. Iterative P selection (validated: 0.34 mH ground state at P=2000)
  2. build_basis (Krylov+SVD compression, handles Q near-degeneracies)
  3. Per-state Bloch (state-specific E₀ and Δ=0)

Reads shared-P checkpoints from Step 1, runs per-state Krylov Bloch for
each root k using E0_k = E0_vals[k] (from rebuilt H_PP).

Comparison targets:
  - Two-step m=0 diagonal resolvent (Step 2 shared): ground 0.34, excited 624-680
  - Phase 18 Stage C P=400 + build_basis m=0: excited 240-332 mH
  - This combined approach: should beat both for excited states

System: N2/cc-pVDZ CAS(10,10)
"""
import sys, os, time, json, argparse
import numpy as np
from numpy.linalg import eigh

sys.path.insert(0, '/data/home/wangcx/krylov-dci')
from src_mf.pyscf_backend import QSpaceIndex, KDCIBackend
from src.effective_h import build_effective_H, diagonalize_effective_H
from src.hamiltonian import Hamiltonian
from pyscf import gto, scf, ao2mo
from pyscf.fci import cistring, direct_spin1

N_CORE = 2
N_ACT = 10
NROOTS = 6
DELTA = 0.0
P_TARGETS = [200, 400, 800, 1200, 1600, 2000]

parser = argparse.ArgumentParser()
parser.add_argument('--indir', default='/data/home/wangcx/krylov-dci/checkpoints_pspace')
args = parser.parse_args()

INPDIR = args.indir
OUTDIR = INPDIR + '_krylov'
os.makedirs(OUTDIR, exist_ok=True)

print("=" * 64)
print("Step 3: Iterative P + Krylov build_basis + per-state Bloch")
print("N2/cc-pVDZ CAS({},{})  delta={}  nroots={}".format(N_ACT, N_ACT, DELTA, NROOTS))
print("P sizes: {}".format(P_TARGETS))
print("Input: {}".format(INPDIR))
print("Output: {}".format(OUTDIR))
print("=" * 64, flush=True)

# ── Build system ──
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
print("  M={:,}".format(M), flush=True)

# ── DMRG-CI reference ──
print("[2] Computing DMRG-CI reference...", flush=True)
ef, _ = direct_spin1.FCI().kernel(h1a, era, N_ACT, ne,
                                   nroots=NROOTS, verbose=0)
e_dmrg = [float(e) for e in np.atleast_1d(ef)[:NROOTS]]
for k, e in enumerate(e_dmrg):
    print("  root {}: {:.8f} Ha".format(k, e), flush=True)

h2_4d = ao2mo.restore('s1', era, N_ACT).reshape(N_ACT, N_ACT, N_ACT, N_ACT)
ham = Hamiltonian(h1=h1a, h2=h2_4d, E_nuc=0.0, E_HF=0.0)

# ═══════════════════════════════════════════════════════════════════
wall_total0 = time.perf_counter()
all_results = {}

for p_size in P_TARGETS:
    fname = "{}/step1_P{:04d}.json".format(INPDIR, p_size)
    if not os.path.exists(fname):
        print("\n  SKIP P={}: checkpoint not found".format(p_size))
        continue

    print("\n" + "=" * 60)
    print("  P = {}".format(p_size))
    print("=" * 60, flush=True)

    with open(fname) as f:
        ckpt = json.load(f)

    p_dets = [(int(a), int(b)) for a, b in ckpt['p_dets']]
    N = len(p_dets)

    t0 = time.perf_counter()

    # ── Build H_PP ──
    print("  Building H_PP ({}x{})...".format(N, N), flush=True)
    H_PP = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            H_PP[i, j] = ham.matrix_element(p_dets[i], p_dets[j])
    H_PP = 0.5 * (H_PP + H_PP.T)
    E0_vals, C_P = eigh(H_PP)
    E0_vals = E0_vals[:NROOTS]
    C_P = C_P[:, :NROOTS]  # bare eigenvectors for overlap tracking

    dE_bare = [(E0_vals[i] - e_dmrg[i]) * 1000 for i in range(NROOTS)]
    print("  Bare dE0 = {:+.1f} mH".format(dE_bare[0]), flush=True)

    # ── Build H_QP ──
    print("  Building H_QP (Q={:,} x P={})...".format(M-N, N), flush=True)
    t_qp = time.perf_counter()
    H_QP = backend.build_hqp(p_dets, verbose=False)
    t_hqp = time.perf_counter() - t_qp
    print("  H_QP built in {:.1f}s".format(t_hqp), flush=True)

    # ── Per-state Krylov build_basis + Bloch ──
    E_bloch = []
    d_basis_list = []

    for k in range(NROOTS):
        E0_k = E0_vals[k]
        print("  --- root {} (E0 = {:.6f} Ha) ---".format(k, E0_k), flush=True)
        t_k0 = time.perf_counter()

        # build_basis with root-specific E0
        basis, d_basis = backend.build_basis(H_QP, E0_k, verbose=False)
        d_basis_list.append(d_basis)

        # Project H_QP into compressed basis
        H_QQ_t, H_PQ_t = backend.build_projected_blocks(
            basis, p_dets, H_QP=H_QP, verbose=False)

        # Build effective H with delta=0 (non-self-consistent)
        H_eff_k = build_effective_H(H_PP, H_PQ_t, H_QQ_t, E0_k, delta=DELTA)

        # Diagonalize fully, track by overlap with bare eigenvector
        ev_all, C_eff = diagonalize_effective_H(H_eff_k)
        v_k_bare = C_P[:, k]
        overlaps = [abs(float(np.dot(C_eff[:, m], v_k_bare))) for m in range(len(ev_all))]
        m_star = int(np.argmax(overlaps))
        E_bloch_k = float(ev_all[m_star])

        E_bloch.append(E_bloch_k)

        dE_b = (E_bloch_k - e_dmrg[k]) * 1000
        print("    d_basis={}  Bloch dE={:+.3f} mH  {:.1f}s".format(
            d_basis, dE_b, time.perf_counter() - t_k0), flush=True)

    t_total = time.perf_counter() - t0

    dE_bloch = [(E_bloch[i] - e_dmrg[i]) * 1000 for i in range(NROOTS)]
    improvement = [dE_bare[i] - dE_bloch[i] for i in range(NROOTS)]

    all_results[p_size] = {
        'P': p_size, 'N': N, 'M': M,
        'E_bare': [float(e) for e in E0_vals],
        'E_bloch': E_bloch,
        'e_dmrg': e_dmrg,
        'dE_bare_mH': dE_bare,
        'dE_bloch_mH': dE_bloch,
        'improvement_mH': improvement,
        'd_basis': d_basis_list,
        'timing': {'t_hqp': t_hqp, 't_total': t_total},
    }

    # Per-P table
    print("\n  {:>5} {:>14} {:>14} {:>14} {:>13} {:>10}".format(
        "root", "FCI (Ha)", "bare dE(mH)", "Bloch dE(mH)", "improve(mH)", "d_basis"))
    print("  " + "-" * 70)
    for k in range(NROOTS):
        print("  {:>5} {:>14.8f} {:>+14.3f} {:>+14.3f} {:>+13.3f} {:>10}".format(
            k, e_dmrg[k], dE_bare[k], dE_bloch[k], improvement[k], d_basis_list[k]),
            flush=True)

    print("  Total: {:.1f}s (H_QP: {:.1f}s)".format(t_total, t_hqp), flush=True)

# ═══════════════ Summary ═══════════════
wall_total = time.perf_counter() - wall_total0

print("\n" + "=" * 75)
print("Krylov + Per-State Bloch H^eff Convergence Summary")
print("Method: iterative P + build_basis (SVD on A^2 H_QP) + per-state delta=0")
print("=" * 75)

# Header
print("{:>6}".format("P"), end="")
for k in range(NROOTS):
    print(" {:>10}".format("bare"+str(k)), end="")
for k in range(NROOTS):
    print(" {:>10}".format("Bloch"+str(k)), end="")
print()

print("-" * 75)
for p_size in P_TARGETS:
    if p_size not in all_results:
        continue
    r = all_results[p_size]
    print("{:>6}".format(p_size), end="")
    for k in range(NROOTS):
        print(" {:>+10.1f}".format(r['dE_bare_mH'][k]), end="")
    for k in range(NROOTS):
        print(" {:>+10.1f}".format(r['dE_bloch_mH'][k]), end="")
    print()

print()
print("Improvement (bare - Bloch):")
print("{:>6}".format("P"), end="")
for k in range(NROOTS):
    print(" {:>10}".format("Delta"+str(k)), end="")
print()

for p_size in P_TARGETS:
    if p_size not in all_results:
        continue
    r = all_results[p_size]
    print("{:>6}".format(p_size), end="")
    for k in range(NROOTS):
        print(" {:>+10.1f}".format(r['improvement_mH'][k]), end="")
    print()

# Save
summary = {
    'system': 'N2/cc-pVDZ CAS(10,10)',
    'method': 'iterative P + build_basis + per-state Bloch (delta=0)',
    'nroots': NROOTS, 'delta': DELTA,
    'wall_total_s': wall_total,
    'results': {str(k): v for k, v in all_results.items()},
}
with open("{}/summary.json".format(OUTDIR), 'w') as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 64)
print("Step 3 complete. {:.0f}s total wall.".format(wall_total))
print("Summary: {}/summary.json".format(OUTDIR))
print("=" * 64)
