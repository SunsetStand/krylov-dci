#!/usr/bin/env python3
"""
Step 2 (fixed): Bloch H^eff batch evaluation over P-space convergence checkpoints.

Key fixes vs original:
  1. dE_bare now uses E0_vals from freshly-rebuilt H_PP (guarantees consistency)
  2. Cross-checks checkpoint E_bare vs rebuilt H_PP eigenvalues
  3. Supports per-state mode (reads root{k}/step1_P*.json)

Modes:
  --mode shared    : read from INPDIR/step1_P*.json
  --mode per_state : read from INPDIR/root{k}/step1_P*.json

System: N2/cc-pVDZ CAS(10,10)
"""
import sys, os, time, json, argparse
import numpy as np
from numpy.linalg import eigh

sys.path.insert(0, '/data/home/wangcx/krylov-dci')
from src_mf.pyscf_backend import QSpaceIndex, KDCIBackend
from src.effective_h import build_effective_H, diagonalize_effective_H
from src.determinants import hf_determinant
from src.hamiltonian import Hamiltonian
from pyscf import gto, scf, ao2mo
from pyscf.fci import cistring, direct_spin1

N_CORE = 2
N_ACT = 10
NROOTS = 6
DELTA = 0.0
P_TARGETS = [200, 400, 800, 1200, 1600, 2000]

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['shared', 'per_state'], default='shared',
                    help='Evaluation mode')
parser.add_argument('--indir', default='/data/home/wangcx/krylov-dci/checkpoints_pspace')
args = parser.parse_args()

MODE = args.mode
INPDIR = args.indir if MODE == 'shared' else args.indir + '_perstate'
OUTDIR = INPDIR  # save results alongside checkpoints

print("=" * 64)
print("Step 2: Bloch H^eff Batch Evaluation (mode={})".format(MODE))
print("N2/cc-pVDZ CAS({},{})  delta={}  nroots={}".format(N_ACT, N_ACT, DELTA, NROOTS))
print("P sizes: {}".format(P_TARGETS))
print("Input: {}".format(INPDIR))
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
print("  E_FCI = {:.8f} Ha".format(e_dmrg[0]), flush=True)

h2_4d = ao2mo.restore('s1', era, N_ACT).reshape(N_ACT, N_ACT, N_ACT, N_ACT)
ham = Hamiltonian(h1=h1a, h2=h2_4d, E_nuc=0.0, E_HF=0.0)


def get_checkpoint_path(p_size):
    if MODE == 'shared':
        return "{}/step1_P{:04d}.json".format(INPDIR, p_size)
    else:
        # per_state: checkpoints are in root{k}/ dirs
        # We'll build a dict of {root: path} per P size
        return None  # handled differently


def process_per_state():
    """Process per-state P_k checkpoints. Each root k has its own P_k."""
    all_results = {}  # root_k -> {P: result}

    for k in range(NROOTS):
        print("\n" + "=" * 60)
        print("  Root {} P-space Bloch evaluation".format(k))
        print("=" * 60, flush=True)

        k_results = {}
        state_dir = "{}/root{}".format(INPDIR, k)

        for p_size in P_TARGETS:
            fname = "{}/step1_P{:04d}.json".format(state_dir, p_size)
            if not os.path.exists(fname):
                print("  SKIP P={}: checkpoint not found".format(p_size))
                continue

            t0 = time.perf_counter()

            with open(fname) as f:
                ckpt = json.load(f)

            p_dets = [(int(a), int(b)) for a, b in ckpt['p_dets']]
            E_bare_ckpt = ckpt['E_bare']
            N_p = len(p_dets)

            # Rebuild H_PP from scratch
            H_PP = np.zeros((N_p, N_p))
            for i in range(N_p):
                for j in range(N_p):
                    H_PP[i, j] = ham.matrix_element(p_dets[i], p_dets[j])
            H_PP = 0.5 * (H_PP + H_PP.T)
            E0_vals, _ = eigh(H_PP)
            E0_vals = E0_vals[:NROOTS]

            # Cross-check checkpoint E_bare vs rebuilt E0_vals
            max_diff = max(abs(E_bare_ckpt[i] - E0_vals[i]) for i in range(NROOTS))
            if max_diff > 1e-8:
                print("  WARNING P={}: ckpt-rebuilt max|diff| = {:.3e} Ha".format(
                    p_size, max_diff))

            # Build H_QP
            H_QP = backend.build_hqp(p_dets, verbose=False)

            # Bloch correction for THIS root k
            H_QQ_diag = q_idx.hdiag
            E0_k = E0_vals[k]

            A_q_diag = 1.0 / (E0_k + DELTA - H_QQ_diag)
            A_q_diag = np.clip(A_q_diag, -1e10, 1e10)

            if isinstance(H_QP, np.ndarray):
                weighted = H_QP * A_q_diag[:, np.newaxis]
            else:
                weighted = H_QP.multiply(A_q_diag[:, np.newaxis])

            correction = H_QP.T @ weighted
            H_eff = H_PP + correction
            H_eff = 0.5 * (H_eff + H_eff.T)

            ev, _ = eigh(H_eff)
            E_bloch_k = float(ev[k])

            dE_bare = (E0_vals[k] - e_dmrg[k]) * 1000
            dE_bloch = (E_bloch_k - e_dmrg[k]) * 1000

            wall_this = time.perf_counter() - t0

            k_results[p_size] = {
                'P': p_size, 'N': N_p, 'M': M,
                'E0_bare': float(E0_vals[k]),
                'E_bloch': E_bloch_k,
                'e_dmrg': e_dmrg[k],
                'dE_bare_mH': dE_bare,
                'dE_bloch_mH': dE_bloch,
                'improvement_mH': dE_bare - dE_bloch,
                'timing': {'t_total': wall_this},
            }

            print("  P={:4d}  bare={:+.3f}  Bloch={:+.3f}  improve={:+.3f} mH  {:.1f}s".format(
                p_size, dE_bare, dE_bloch, dE_bare - dE_bloch, wall_this), flush=True)

        all_results['root{}'.format(k)] = k_results

    # Per-state summary
    print("\n" + "=" * 70)
    print("Per-State P-space Convergence Summary")
    print("       {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
        "root0", "root1", "root2", "root3", "root4", "root5"))
    print("       {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
        "bare Bloch", "bare Bloch", "bare Bloch", "bare Bloch", "bare Bloch", "bare Bloch"))
    print("-" * 70)

    for p_size in P_TARGETS:
        row = "P={:4d}".format(p_size)
        for k in range(NROOTS):
            rk = all_results.get('root{}'.format(k), {})
            if p_size in rk:
                r = rk[p_size]
                row += " {:>4.1f} {:>4.1f}".format(
                    r['dE_bare_mH'], r['dE_bloch_mH'])
            else:
                row += " {:>10}".format("---")
        print(row, flush=True)

    return all_results


def process_shared():
    """Process shared P checkpoints with per-state Bloch correction."""
    results = {}

    for p_size in P_TARGETS:
        fname = "{}/step1_P{:04d}.json".format(INPDIR, p_size)
        if not os.path.exists(fname):
            print("\n  SKIP P={}: checkpoint not found".format(p_size))
            continue

        print("\n[P={}] Loading checkpoint...".format(p_size), flush=True)
        with open(fname) as f:
            ckpt = json.load(f)

        p_dets = [(int(a), int(b)) for a, b in ckpt['p_dets']]
        E_bare_ckpt = ckpt['E_bare']
        N = len(p_dets)

        t0 = time.perf_counter()

        # Rebuild H_PP from scratch (guarantees no incremental-matrix artifacts)
        print("  Building H_PP ({}x{})...".format(N, N), flush=True)
        H_PP = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                H_PP[i, j] = ham.matrix_element(p_dets[i], p_dets[j])
        H_PP = 0.5 * (H_PP + H_PP.T)
        E0_vals, _ = eigh(H_PP)
        E0_vals = E0_vals[:NROOTS]

        # DIAGNOSTIC: cross-check checkpoint E_bare vs rebuilt
        max_diff = max(abs(E_bare_ckpt[i] - E0_vals[i]) for i in range(NROOTS))
        if max_diff > 1e-8:
            print("  ** DIAG: ckpt E_bare vs rebuilt H_PP differ! max|diff| = {:.3e} Ha".format(
                max_diff))
            for i in range(NROOTS):
                print("    root{}: ckpt={:.10f}  rebuilt={:.10f}  diff={:.3e}".format(
                    i, E_bare_ckpt[i], E0_vals[i], E_bare_ckpt[i] - E0_vals[i]))
        else:
            print("  DIAG OK: ckpt E_bare == rebuilt H_PP (max|diff| < 1e-8)", flush=True)

        t_hpp = time.perf_counter() - t0

        # Build H_QP
        print("  Building H_QP (Q={:,} x P={})...".format(M-N, N), flush=True)
        t1 = time.perf_counter()
        H_QP = backend.build_hqp(p_dets, verbose=False)
        t_hqp = time.perf_counter() - t1

        # Per-state Bloch correction (m=0, delta=0)
        print("  Computing per-state Bloch H^eff...", flush=True)
        E_bloch = []
        H_QQ_diag = q_idx.hdiag

        for k in range(NROOTS):
            E0_k = E0_vals[k]
            A_q_diag = 1.0 / (E0_k + DELTA - H_QQ_diag)
            A_q_diag = np.clip(A_q_diag, -1e10, 1e10)

            if isinstance(H_QP, np.ndarray):
                weighted = H_QP * A_q_diag[:, np.newaxis]
            else:
                weighted = H_QP.multiply(A_q_diag[:, np.newaxis])

            correction = H_QP.T @ weighted
            H_eff = H_PP + correction
            H_eff = 0.5 * (H_eff + H_eff.T)
            ev, _ = eigh(H_eff)
            E_bloch.append(float(ev[k]))

        t_heff = time.perf_counter() - t1

        # Compute errors using rebuilt E0_vals (not checkpoint) for consistency
        dE_bare = [(E0_vals[i] - e_dmrg[i]) * 1000 for i in range(NROOTS)]
        dE_bloch = [(E_bloch[i] - e_dmrg[i]) * 1000 for i in range(NROOTS)]
        improvement = [dE_bare[i] - dE_bloch[i] for i in range(NROOTS)]

        wall_this = time.perf_counter() - t0

        results[p_size] = {
            'P': p_size, 'N': N, 'M': M,
            'E_bare': [float(e) for e in E0_vals],
            'E_bloch': E_bloch,
            'e_dmrg': e_dmrg,
            'dE_bare_mH': dE_bare,
            'dE_bloch_mH': dE_bloch,
            'improvement_mH': [float(imp) for imp in improvement],
            'timing': {'t_hpp': t_hpp, 't_hqp': t_hqp,
                       't_heff': t_heff, 't_total': wall_this},
            'nroots': NROOTS, 'delta': DELTA,
        }

        # Save individual result
        outf = "{}/step2_P{:04d}.json".format(OUTDIR, p_size)
        with open(outf, 'w') as f:
            json.dump(results[p_size], f, indent=2)

        # Print per-P table
        print("\n  {:>5} {:>14} {:>14} {:>14} {:>13}".format(
            "root", "FCI (Ha)", "bare dE(mH)", "Bloch dE(mH)", "improve(mH)"))
        print("  " + "-" * 60)
        for k in range(NROOTS):
            print("  {:>5} {:>14.8f} {:>+14.3f} {:>+14.3f} {:>+13.3f}".format(
                k, e_dmrg[k], dE_bare[k], dE_bloch[k], improvement[k]), flush=True)

        print("  H_PP: {:.1f}s  H_QP: {:.1f}s  H^eff: {:.1f}s  total: {:.1f}s".format(
            t_hpp, t_hqp, t_heff, wall_this), flush=True)

    return results


# ═══════════════ Main ═══════════════
wall_total0 = time.perf_counter()

if MODE == 'shared':
    all_results = process_shared()
else:
    all_results = process_per_state()

# ── Global Summary ──
print("\n" + "=" * 70)
print("P-space Convergence Summary (using REBUILT H_PP eigenvalues)")
print("{:>6}".format("P"), end="")
for k in range(NROOTS):
    print(" {:>11}".format("bare_dE"+str(k)), end="")
print(" {:>11}".format("bloch_dE0"))
print("-" * 70)

for p_size in P_TARGETS:
    if MODE == 'shared':
        if p_size not in all_results:
            continue
        r = all_results[p_size]
        print("{:>6}".format(p_size), end="")
        for k in range(NROOTS):
            print(" {:>+11.3f}".format(r['dE_bare_mH'][k]), end="")
        print(" {:>+11.3f}".format(r['dE_bloch_mH'][0]), flush=True)
    else:
        # per_state: show per-root bare and Bloch
        row = "{:>6}".format(p_size)
        for k in range(NROOTS):
            rk = all_results.get('root{}'.format(k), {})
            if p_size in rk:
                row += " {:>+11.3f}".format(rk[p_size]['dE_bare_mH'])
            else:
                row += " {:>11}".format("---")
        print(row, flush=True)

print("\nBloch Correction Impact (improvement = bare - Bloch)")
print("{:>6}".format("P"), end="")
n_cols = NROOTS if MODE == 'shared' else NROOTS
for k in range(n_cols):
    print(" {:>11}".format("Delta_err"+str(k)), end="")
print("\n" + "-" * 70)

for p_size in P_TARGETS:
    if MODE == 'shared':
        if p_size not in all_results:
            continue
        r = all_results[p_size]
        print("{:>6}".format(p_size), end="")
        for k in range(NROOTS):
            print(" {:>+11.3f}".format(r['improvement_mH'][k]), end="")
        print()
    else:
        row = "{:>6}".format(p_size)
        for k in range(NROOTS):
            rk = all_results.get('root{}'.format(k), {})
            if p_size in rk:
                row += " {:>+11.3f}".format(rk[p_size]['improvement_mH'])
            else:
                row += " {:>11}".format("---")
        print(row, flush=True)

# Save summary
summary = {
    'system': 'N2/cc-pVDZ CAS(10,10)',
    'method': 'm=0 per-state Bloch H^eff',
    'mode': MODE,
    'delta': DELTA,
    'nroots': NROOTS,
    'wall_total_s': time.perf_counter() - wall_total0,
}
if MODE == 'shared':
    summary['results'] = {str(k): v for k, v in all_results.items()}
else:
    summary['results'] = {
        'root{}'.format(k): {str(ps): v for ps, v in rk.items()}
        for k, rk in all_results.items() if rk
    }

with open("{}/summary.json".format(OUTDIR), 'w') as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 64)
print("Step 2 complete. {:.0f}s total wall.".format(summary['wall_total_s']))
print("Summary saved to {}/summary.json".format(OUTDIR))
print("=" * 64)
