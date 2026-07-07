#!/usr/bin/env python3
"""
Strong-correlation N2 bond scan: Extended P-space strategies.

For R = 1.5, 1.8, 2.2 (where P=2000 doesn't reach chemical accuracy):

Strategy A — Continue iterative selection from P=2000:
  → Load existing P=2000 checkpoint
  → Continue iterative σ-vector selection: P=2000→4000→6000→8000
  → m=0 Bloch H^eff at each P

Strategy B — Fresh start with larger HFPT2 seed + large batch:
  → P_init = 1000 (HFPT2, max available)
  → batch = 500 (larger steps)
  → Iterative: P=1000→2000→4000→6000→8000
  → m=0 Bloch H^eff at each P

System: N2/cc-pVDZ CAS(10,10)
"""
import sys, os, time, json, argparse
import numpy as np
from numpy.linalg import eigh

sys.path.insert(0, '/data/home/wangcx/krylov-dci')
from src_mf.pyscf_backend import QSpaceIndex, KDCIBackend
from src.hamiltonian import Hamiltonian
from src.determinants import hf_determinant, bit_positions
from pyscf import gto, scf, ao2mo
from pyscf.fci import cistring, direct_spin1
import itertools

# ── Parameters ──
N_CORE = 2
N_ACT = 10
NROOTS = 6
DELTA = 0.0

# Strategy A: continue from P=2000
STRAT_A = {
    'P_start': 2000,
    'P_max': 8000,
    'batch': 500,
    'P_targets': [2000, 3000, 4000, 5000, 6000, 7000, 8000],
    'label': 'continue',
}

# Strategy B: fresh with large seed
STRAT_B = {
    'P_init': 1000,
    'P_max': 8000,
    'batch': 500,
    'P_targets': [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000],
    'label': 'fresh_large',
}

STRONG_R = [1.5, 1.8, 2.2]  # bond lengths to attack

parser = argparse.ArgumentParser()
parser.add_argument('--strategy', choices=['A', 'B', 'both'], default='both')
parser.add_argument('--outdir', default='/data/home/wangcx/krylov-dci/checkpoints_bondscan_ext')
parser.add_argument('--bondscan_dir', default='/data/home/wangcx/krylov-dci/checkpoints_bondscan')
args = parser.parse_args()

OUTDIR = args.outdir
BONDSCAN_DIR = args.bondscan_dir

strategies = []
if args.strategy in ('A', 'both'):
    strategies.append(('A', STRAT_A))
if args.strategy in ('B', 'both'):
    strategies.append(('B', STRAT_B))

print("=" * 72)
print("Strong-Correlation N₂: Extended P-space Strategies")
print("Bond lengths: {} Å".format(STRONG_R))
print("Strategies: {}".format([s[0] for s in strategies]))
print("Output: {}".format(OUTDIR))
print("=" * 72, flush=True)

for R in STRONG_R:
    print("\n" + "▓" * 66)
    print("▓  R = {:.1f} Å".format(R))
    print("▓" * 66, flush=True)

    R_dir_bondscan = "{}/R{:.1f}".format(BONDSCAN_DIR, R)
    if not os.path.exists(R_dir_bondscan):
        print("  SKIP: no bondscan checkpoint for R={:.1f}".format(R))
        continue

    t_geo0 = time.perf_counter()

    # ── Build system ──
    print("\n  [1] Building N2/cc-pVDZ CAS(10,10) at R={:.1f}...".format(R), flush=True)
    mol = gto.M(atom='N 0 0 0; N 0 0 {:.1f}'.format(R), basis='cc-pVDZ', verbose=0)
    mf = scf.RHF(mol)
    try:
        mf.kernel(verbose=0)
    except Exception:
        mf = scf.newton(mf)
        mf.kernel(verbose=0)
    if not mf.converged:
        print("  ERROR: RHF failed. Skipping R={:.1f}".format(R))
        continue

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
    hdiag_full = q_idx.hdiag
    print("  M={:,}".format(M), flush=True)

    # ── DMRG-CI reference ──
    print("  [2] DMRG-CI reference...", flush=True)
    ef, _ = direct_spin1.FCI().kernel(h1a, era, N_ACT, ne,
                                       nroots=NROOTS, max_cycle=200, verbose=0)
    e_dmrg = [float(e) for e in np.atleast_1d(ef)[:NROOTS]]
    print("    S0 = {:.8f} Ha  S1 exc = {:.1f} eV".format(
        e_dmrg[0], (e_dmrg[1]-e_dmrg[0])*27.2114), flush=True)

    # ── Hamiltonian ──
    h2_4d = ao2mo.restore('s1', era, N_ACT).reshape(N_ACT, N_ACT, N_ACT, N_ACT)
    ham = Hamiltonian(h1=h1a, h2=h2_4d, E_nuc=0.0, E_HF=0.0)

    full_dets = [(int(as_[ai]), int(bs_[bi])) for ai in range(len(as_)) for bi in range(len(bs_))]
    det_to_full = {d: i for i, d in enumerate(full_dets)}

    # ── HF determinants ──
    hf_a, hf_b = hf_determinant(*ne)
    ao = bit_positions(hf_a)
    bo = bit_positions(hf_b)
    av = [p for p in range(N_ACT) if p not in ao]
    bv = [p for p in range(N_ACT) if p not in bo]

    # ── Helper: HFPT2 scoring ──
    def gen_hfpt2_scores():
        scores = []
        E_HF = ham.matrix_element((hf_a, hf_b), (hf_a, hf_b))
        for i in ao:
            for a in av:
                d = (hf_a ^ (1 << i) | (1 << a), hf_b)
                hij = ham.matrix_element(d, (hf_a, hf_b))
                den = E_HF - ham.matrix_element(d, d)
                if abs(den) > 1e-12: scores.append((d, -hij * hij / den))
        for i in bo:
            for a in bv:
                d = (hf_a, hf_b ^ (1 << i) | (1 << a))
                hij = ham.matrix_element(d, (hf_a, hf_b))
                den = E_HF - ham.matrix_element(d, d)
                if abs(den) > 1e-12: scores.append((d, -hij * hij / den))
        for i1, i2 in itertools.combinations(ao, 2):
            for a1, a2 in itertools.combinations(av, 2):
                d = (hf_a ^ (1 << i1) ^ (1 << i2) | (1 << a1) | (1 << a2), hf_b)
                hij = ham.matrix_element(d, (hf_a, hf_b))
                den = E_HF - ham.matrix_element(d, d)
                if abs(den) > 1e-12: scores.append((d, -hij * hij / den))
        for i1, i2 in itertools.combinations(bo, 2):
            for a1, a2 in itertools.combinations(bv, 2):
                d = (hf_a, hf_b ^ (1 << i1) ^ (1 << i2) | (1 << a1) | (1 << a2))
                hij = ham.matrix_element(d, (hf_a, hf_b))
                den = E_HF - ham.matrix_element(d, d)
                if abs(den) > 1e-12: scores.append((d, -hij * hij / den))
        for i in ao:
            for j in bo:
                for a in av:
                    for b in bv:
                        d = (hf_a ^ (1 << i) | (1 << a), hf_b ^ (1 << j) | (1 << b))
                        hij = ham.matrix_element(d, (hf_a, hf_b))
                        den = E_HF - ham.matrix_element(d, d)
                        if abs(den) > 1e-12: scores.append((d, -hij * hij / den))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    # ── Helper functions ──
    def build_hpp(dets):
        n = len(dets)
        H = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                v = ham.matrix_element(dets[i], dets[j])
                H[i, j] = v; H[j, i] = v
        return H

    def extend_hpp(H_old, old_dets, new_dets):
        N_old = len(old_dets); n_add = len(new_dets)
        H_new = np.zeros((N_old + n_add, N_old + n_add))
        H_new[:N_old, :N_old] = H_old
        for i_local, det_new in enumerate(new_dets):
            row = N_old + i_local
            for j in range(N_old):
                val = ham.matrix_element(det_new, old_dets[j])
                H_new[row, j] = val; H_new[j, row] = val
            for j_local in range(i_local + 1):
                col = N_old + j_local
                val = ham.matrix_element(det_new, new_dets[j_local])
                H_new[row, col] = val; H_new[col, row] = val
        return H_new

    def score_q_space(sigmas, p_set):
        weights = np.zeros(M)
        for E_ref, sigma_k in sigmas:
            abs_sigma = np.abs(sigma_k)
            for qi in range(M):
                if qi not in p_set:
                    c2 = abs_sigma[qi] ** 2
                    if c2 < 1e-24: continue
                    denom = max(abs(E_ref - hdiag_full[qi]), 1e-8)
                    weights[qi] += c2 / denom
        return weights

    def bloch_eval(dets, Hpp, e_dmrg):
        """m=0 Bloch H^eff for ground state only."""
        N = len(dets)
        E0_vals, _ = eigh(Hpp)
        E0 = E0_vals[0]

        H_QP = backend.build_hqp(dets, verbose=False)
        A_q_diag = 1.0 / (E0 + DELTA - hdiag_full)
        A_q_diag = np.clip(A_q_diag, -1e10, 1e10)

        if hasattr(H_QP, 'multiply'):
            weighted = H_QP.multiply(A_q_diag[:, np.newaxis])
            correction = H_QP.T @ weighted
        else:
            weighted = H_QP * A_q_diag[:, np.newaxis]
            correction = H_QP.T @ weighted

        H_eff = Hpp + correction
        H_eff = 0.5 * (H_eff + H_eff.T)
        ev, _ = eigh(H_eff)
        E_bloch = float(ev[0])
        dE_bare = (E0 - e_dmrg[0]) * 1000
        dE_bloch = (E_bloch - e_dmrg[0]) * 1000
        return dE_bare, dE_bloch, E0, E_bloch

    # ═══════════════════════════════════════════════════════
    # Strategy A: Continue from P=2000 checkpoint
    # ═══════════════════════════════════════════════════════
    if 'A' in [s[0] for s in strategies]:
        strat = STRAT_A
        print("\n  ── Strategy A: Continue iterative selection from P=2000 ──", flush=True)

        save_dir = "{}/R{:.1f}_A".format(OUTDIR, R)
        os.makedirs(save_dir, exist_ok=True)

        # Load P=2000 checkpoint from bondscan
        ckpt_file = "{}/step1_P2000.json".format(R_dir_bondscan)
        with open(ckpt_file) as f:
            ckpt = json.load(f)
        p_dets = [(int(a), int(b)) for a, b in ckpt['p_dets']]
        p_full_indices = ckpt['p_full_indices']
        p_set = set(p_full_indices)
        N_p = len(p_dets)
        assert N_p == 2000, f"Expected 2000 dets, got {N_p}"

        # Rebuild H_PP
        H_PP = build_hpp(p_dets)
        dE_b, dE_bl, _, _ = bloch_eval(p_dets, H_PP, e_dmrg)
        print("  P=2000 base: bare={:+.2f} Bloch={:+.2f} mH".format(dE_b, dE_bl), flush=True)

        results_A = {2000: {'dE_bare_mH': dE_b, 'dE_bloch_mH': dE_bl}}

        # Continue iterative selection
        B = strat['batch']
        print("  {:>5} {:>10} {:>14} {:>14}".format("iter", "P", "bare(mH)", "Bloch(mH)"))
        print("  " + "-" * 50, flush=True)

        iter_num = 0
        while N_p < strat['P_max']:
            iter_num += 1
            t_iter = time.perf_counter()

            E_P, C_P = eigh(H_PP)
            n_sigma = min(5, N_p)
            sigmas = []
            for sk in range(n_sigma):
                vec_full = np.zeros(M)
                for local_i, gi in enumerate(p_full_indices):
                    vec_full[gi] = C_P[local_i, sk]
                sigma_k = backend.sigma(vec_full)
                sigmas.append((E_P[sk], sigma_k))

            weights = score_q_space(sigmas, p_set)
            candidates = [(qi, float(weights[qi])) for qi in range(M)
                          if qi not in p_set and weights[qi] > 0]
            candidates.sort(key=lambda x: x[1], reverse=True)

            n_add = min(B, len(candidates))
            if n_add == 0:
                print("  No more determinants. Stopping.", flush=True)
                break

            new_global = [c[0] for c in candidates[:n_add]]
            new_dets = [full_dets[qi] for qi in new_global]

            H_PP = extend_hpp(H_PP, p_dets, new_dets)
            p_dets.extend(new_dets)
            p_full_indices.extend(new_global)
            p_set.update(new_global)
            N_p = len(p_dets)

            dE_bare, dE_bloch, _, _ = bloch_eval(p_dets, H_PP, e_dmrg)
            t_elapsed = time.perf_counter() - t_iter

            for pt in strat['P_targets']:
                if N_p >= pt and pt not in results_A:
                    results_A[pt] = {'dE_bare_mH': dE_bare, 'dE_bloch_mH': dE_bloch}

            print("  {:>5} {:>10} {:>+14.2f} {:>+14.3f}  ({:.0f}s)".format(
                iter_num, N_p, dE_bare, dE_bloch, t_elapsed), flush=True)

        # Save
        with open("{}/summary.json".format(save_dir), 'w') as f:
            json.dump({
                'strategy': 'A_continue', 'R': R, 'results': results_A
            }, f, indent=2)

        print("  Strategy A summary for R={:.1f}:".format(R))
        for pt in sorted(results_A.keys()):
            r = results_A[pt]
            print("    P={:>5}: bare={:+.2f}  Bloch={:+.3f} mH".format(
                pt, r['dE_bare_mH'], r['dE_bloch_mH']), flush=True)

    # ═══════════════════════════════════════════════════════
    # Strategy B: Fresh start with large HFPT2 seed + large batch
    # ═══════════════════════════════════════════════════════
    if 'B' in [s[0] for s in strategies]:
        strat = STRAT_B
        print("\n  ── Strategy B: Large HFPT2 seed (P_init={}) + batch={} ──".format(
            strat['P_init'], strat['batch']), flush=True)

        save_dir = "{}/R{:.1f}_B".format(OUTDIR, R)
        os.makedirs(save_dir, exist_ok=True)

        # Fresh HFPT2 P_init
        scores = gen_hfpt2_scores()
        init_dets = [(hf_a, hf_b)]
        for det, _ in scores:
            if det not in init_dets:
                init_dets.append(det)
            if len(init_dets) >= strat['P_init']:
                break
        P_actual = len(init_dets)
        print("  HFPT2 P_init: {} determinants (from {} SD)".format(P_actual, len(scores)), flush=True)

        init_full_indices = [det_to_full[d] for d in init_dets]
        H_PP = build_hpp(init_dets)
        p_dets = list(init_dets)
        p_full_indices = list(init_full_indices)
        p_set = set(init_full_indices)
        N_p = len(p_dets)

        dE_b, dE_bl, _, _ = bloch_eval(p_dets, H_PP, e_dmrg)
        print("  P={} init: bare={:+.2f} Bloch={:+.2f} mH".format(N_p, dE_b, dE_bl), flush=True)

        results_B = {N_p: {'dE_bare_mH': dE_b, 'dE_bloch_mH': dE_bl}}

        # Iterative from here
        B = strat['batch']
        print("  {:>5} {:>10} {:>14} {:>14}".format("iter", "P", "bare(mH)", "Bloch(mH)"))
        print("  " + "-" * 50, flush=True)

        iter_num = 0
        while N_p < strat['P_max']:
            iter_num += 1
            t_iter = time.perf_counter()

            E_P, C_P = eigh(H_PP)
            n_sigma = min(5, N_p)
            sigmas = []
            for sk in range(n_sigma):
                vec_full = np.zeros(M)
                for local_i, gi in enumerate(p_full_indices):
                    vec_full[gi] = C_P[local_i, sk]
                sigma_k = backend.sigma(vec_full)
                sigmas.append((E_P[sk], sigma_k))

            weights = score_q_space(sigmas, p_set)
            candidates = [(qi, float(weights[qi])) for qi in range(M)
                          if qi not in p_set and weights[qi] > 0]
            candidates.sort(key=lambda x: x[1], reverse=True)

            n_add = min(B, len(candidates))
            if n_add == 0:
                print("  No more determinants. Stopping.", flush=True)
                break

            new_global = [c[0] for c in candidates[:n_add]]
            new_dets = [full_dets[qi] for qi in new_global]

            H_PP = extend_hpp(H_PP, p_dets, new_dets)
            p_dets.extend(new_dets)
            p_full_indices.extend(new_global)
            p_set.update(new_global)
            N_p = len(p_dets)

            dE_bare, dE_bloch, _, _ = bloch_eval(p_dets, H_PP, e_dmrg)
            t_elapsed = time.perf_counter() - t_iter

            for pt in strat['P_targets']:
                if N_p >= pt and pt not in results_B:
                    results_B[pt] = {'dE_bare_mH': dE_bare, 'dE_bloch_mH': dE_bloch}

            print("  {:>5} {:>10} {:>+14.2f} {:>+14.3f}  ({:.0f}s)".format(
                iter_num, N_p, dE_bare, dE_bloch, t_elapsed), flush=True)

        # Save
        with open("{}/summary.json".format(save_dir), 'w') as f:
            json.dump({
                'strategy': 'B_fresh_large', 'R': R, 'results': results_B
            }, f, indent=2)

        print("  Strategy B summary for R={:.1f}:".format(R))
        for pt in sorted(results_B.keys()):
            r = results_B[pt]
            print("    P={:>5}: bare={:+.2f}  Bloch={:+.3f} mH".format(
                pt, r['dE_bare_mH'], r['dE_bloch_mH']), flush=True)

    t_geo = time.perf_counter() - t_geo0
    print("\n  R={:.1f} strategies done. {:.0f}s wall.".format(R, t_geo), flush=True)

print("\n" + "=" * 72)
print("Strong-correlation extended P-space complete.")
print("Results: {}".format(OUTDIR))
print("=" * 72)
