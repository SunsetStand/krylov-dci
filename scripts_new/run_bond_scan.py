#!/usr/bin/env python3
"""
N2 Bond Length Scan: P-space convergence + Bloch H^eff across geometries.

For each bond length R in [0.8, 0.9, 1.0, 1.1, 1.3, 1.5, 1.8, 2.2]:
  1. DMRG-CI reference (6 roots) — exact FCI in CAS(10,10)
  2. Iterative P-space selection (shared mode, P_init=200 → P_max=2000)
  3. m=0 Bloch H^eff at each P checkpoint (200, 400, 800, 1200, 1600, 2000)

Goal: determine minimum P for chemical accuracy (1.6 mH) at each geometry.

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
P_INIT = 200
P_MAX = 2000
BATCH_SIZE = 200
P_TARGETS = [200, 400, 800, 1200, 1600, 2000]
DELTA = 0.0

BOND_LENGTHS = [0.8, 0.9, 1.0, 1.1, 1.3, 1.5, 1.8, 2.2]

parser = argparse.ArgumentParser()
parser.add_argument('--outdir', default='/data/home/wangcx/krylov-dci/checkpoints_bondscan')
parser.add_argument('--start', type=int, default=0, help='Start index in BOND_LENGTHS')
parser.add_argument('--end', type=int, default=None, help='End index (exclusive)')
args = parser.parse_args()

OUTDIR = args.outdir
os.makedirs(OUTDIR, exist_ok=True)
end_idx = args.end if args.end is not None else len(BOND_LENGTHS)
R_list = BOND_LENGTHS[args.start:end_idx]

print("=" * 64)
print("N₂ Bond Length Scan: P-space + Bloch H^eff Convergence")
print("Basis: cc-pVDZ  CAS(10,10)  P={}->{}  Batch={}  Delta={}".format(
    P_INIT, P_MAX, BATCH_SIZE, DELTA))
print("Bond lengths: {} Å".format(R_list))
print("Output: {}".format(OUTDIR))
print("=" * 64, flush=True)

all_summaries = {}

for iR, R in enumerate(R_list):
    print("\n" + "▓" * 60)
    print("▓  R = {:.1f} Å  ({}/{})".format(R, iR + 1, len(R_list)))
    print("▓" * 60, flush=True)

    R_dir = "{}/R{:.1f}".format(OUTDIR, R)
    os.makedirs(R_dir, exist_ok=True)

    t_geo0 = time.perf_counter()

    # ── Build system ──
    print("\n  [1] Building N2/cc-pVDZ CAS(10,10) at R={:.1f}...".format(R), flush=True)
    mol = gto.M(atom='N 0 0 0; N 0 0 {:.1f}'.format(R), basis='cc-pVDZ', verbose=0)
    mf = scf.RHF(mol)
    try:
        mf.kernel(verbose=0)
    except Exception:
        print("  WARNING: RHF not converged at R={:.1f}, trying DM...".format(R))
        mf = scf.newton(mf)
        mf.kernel(verbose=0)

    if not mf.converged:
        print("  ERROR: RHF failed to converge at R={:.1f}. Skipping.".format(R))
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
    print("  M={:,}  E_HF={:.8f}".format(M, mf.e_tot), flush=True)

    # ── DMRG-CI reference ──
    print("  [2] DMRG-CI reference...", flush=True)
    ef, _ = direct_spin1.FCI().kernel(h1a, era, N_ACT, ne,
                                       nroots=NROOTS, max_cycle=200, verbose=0)
    e_dmrg = [float(e) for e in np.atleast_1d(ef)[:NROOTS]]
    for k, e in enumerate(e_dmrg):
        de = (e - e_dmrg[0]) * 27.2114  # eV
        print("    root {}: {:.8f} Ha  ({:+.2f} eV)".format(k, e, de), flush=True)

    # ── Hamiltonian ──
    h2_4d = ao2mo.restore('s1', era, N_ACT).reshape(N_ACT, N_ACT, N_ACT, N_ACT)
    ham = Hamiltonian(h1=h1a, h2=h2_4d, E_nuc=0.0, E_HF=0.0)

    # ── Full determinant list ──
    full_dets = [(int(as_[ai]), int(bs_[bi])) for ai in range(len(as_)) for bi in range(len(bs_))]
    det_to_full = {d: i for i, d in enumerate(full_dets)}

    # ── HFPT2 initial P ──
    hf_a, hf_b = hf_determinant(*ne)
    ao = bit_positions(hf_a)
    bo = bit_positions(hf_b)
    av = [p for p in range(N_ACT) if p not in ao]
    bv = [p for p in range(N_ACT) if p not in bo]

    def gen_hfpt2_scores():
        scores = []
        for i in ao:
            for a in av:
                d = (hf_a ^ (1 << i) | (1 << a), hf_b)
                hij = ham.matrix_element(d, (hf_a, hf_b))
                den = ham.matrix_element((hf_a, hf_b), (hf_a, hf_b)) - ham.matrix_element(d, d)
                if abs(den) > 1e-12:
                    scores.append((d, -hij * hij / den))
        for i in bo:
            for a in bv:
                d = (hf_a, hf_b ^ (1 << i) | (1 << a))
                hij = ham.matrix_element(d, (hf_a, hf_b))
                den = ham.matrix_element((hf_a, hf_b), (hf_a, hf_b)) - ham.matrix_element(d, d)
                if abs(den) > 1e-12:
                    scores.append((d, -hij * hij / den))
        for i1, i2 in itertools.combinations(ao, 2):
            for a1, a2 in itertools.combinations(av, 2):
                d = (hf_a ^ (1 << i1) ^ (1 << i2) | (1 << a1) | (1 << a2), hf_b)
                hij = ham.matrix_element(d, (hf_a, hf_b))
                den = ham.matrix_element((hf_a, hf_b), (hf_a, hf_b)) - ham.matrix_element(d, d)
                if abs(den) > 1e-12:
                    scores.append((d, -hij * hij / den))
        for i1, i2 in itertools.combinations(bo, 2):
            for a1, a2 in itertools.combinations(bv, 2):
                d = (hf_a, hf_b ^ (1 << i1) ^ (1 << i2) | (1 << a1) | (1 << a2))
                hij = ham.matrix_element(d, (hf_a, hf_b))
                den = ham.matrix_element((hf_a, hf_b), (hf_a, hf_b)) - ham.matrix_element(d, d)
                if abs(den) > 1e-12:
                    scores.append((d, -hij * hij / den))
        for i in ao:
            for j in bo:
                for a in av:
                    for b in bv:
                        d = (hf_a ^ (1 << i) | (1 << a), hf_b ^ (1 << j) | (1 << b))
                        hij = ham.matrix_element(d, (hf_a, hf_b))
                        den = ham.matrix_element((hf_a, hf_b), (hf_a, hf_b)) - ham.matrix_element(d, d)
                        if abs(den) > 1e-12:
                            scores.append((d, -hij * hij / den))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    scores = gen_hfpt2_scores()
    init_dets = [(hf_a, hf_b)]
    for det, _ in scores:
        if det not in init_dets:
            init_dets.append(det)
        if len(init_dets) >= P_INIT:
            break
    print("  HFPT2 initial P: {} determinants".format(len(init_dets)), flush=True)

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
                    if c2 < 1e-24:
                        continue
                    denom = max(abs(E_ref - hdiag_full[qi]), 1e-8)
                    weights[qi] += c2 / denom
        return weights

    # ── Iterative P selection ──
    print("  [3] Iterative P-space selection...", flush=True)
    init_full_indices = [det_to_full[d] for d in init_dets]
    H_PP = build_hpp(init_dets)
    p_dets = list(init_dets)
    p_full_indices = list(init_full_indices)
    p_set = set(init_full_indices)
    N_p = len(p_dets)
    checkpoints_p = {}

    print("  {:>5} {:>10} {:>14} {:>14}".format("iter", "P", "E0_bare(Ha)", "dE0(mH)"))
    print("  " + "-" * 50, flush=True)

    iter_num = 0
    while N_p < P_MAX:
        E_P, C_P = eigh(H_PP)
        n_sigma = min(5, N_p)
        sigmas = []
        for sk in range(n_sigma):
            vec_full = np.zeros(M)
            for local_i, global_i in enumerate(p_full_indices):
                vec_full[global_i] = C_P[local_i, sk]
            sigma_k = backend.sigma(vec_full)
            sigmas.append((E_P[sk], sigma_k))

        weights = score_q_space(sigmas, p_set)
        q_candidates = [(qi, float(weights[qi])) for qi in range(M)
                        if qi not in p_set and weights[qi] > 0]
        q_candidates.sort(key=lambda x: x[1], reverse=True)

        n_add = min(BATCH_SIZE, len(q_candidates))
        if n_add == 0:
            print("  No more determinants. Stopping.", flush=True)
            break

        new_global = [q_c[0] for q_c in q_candidates[:n_add]]
        new_dets = [full_dets[qi] for qi in new_global]

        H_PP = extend_hpp(H_PP, p_dets, new_dets)
        p_dets.extend(new_dets)
        p_full_indices.extend(new_global)
        p_set.update(new_global)
        N_p = len(p_dets)

        dE0_mH = (E_P[0] - e_dmrg[0]) * 1000
        iter_num += 1

        if iter_num <= 3 or iter_num % 5 == 0:
            print("  {:>5} {:>10} {:>14.8f} {:>+14.3f}".format(
                iter_num, N_p, E_P[0], dE0_mH), flush=True)

        for p_target in P_TARGETS:
            if N_p >= p_target and p_target not in checkpoints_p:
                dets_p = p_dets[:p_target]
                idx_p = p_full_indices[:p_target]
                H_exact = H_PP[:p_target, :p_target]
                H_exact = 0.5 * (H_exact + H_exact.T)
                E_exact, _ = eigh(H_exact)

                checkpoints_p[p_target] = {
                    'P': p_target,
                    'p_dets': [(int(a), int(b)) for a, b in dets_p],
                    'p_full_indices': idx_p,
                    'E_bare': [float(e) for e in E_exact[:NROOTS]],
                    'dE0_bare_mH': float((E_exact[0] - e_dmrg[0]) * 1000),
                }
                fname = "{}/step1_P{:04d}.json".format(R_dir, p_target)
                with open(fname, 'w') as f:
                    json.dump(checkpoints_p[p_target], f, indent=2)

    # ── m=0 Bloch H^eff at each checkpoint ──
    print("\n  [4] m=0 Bloch H^eff at checkpoints...", flush=True)
    bloch_results = {}
    H_QQ_diag = hdiag_full  # diagonal of H_QQ

    for p_size in P_TARGETS:
        if p_size not in checkpoints_p:
            continue
        ckpt = checkpoints_p[p_size]
        ckpt_dets = [(int(a), int(b)) for a, b in ckpt['p_dets']]
        N = len(ckpt_dets)

        # Rebuild H_PP
        Hpp = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                Hpp[i, j] = ham.matrix_element(ckpt_dets[i], ckpt_dets[j])
        Hpp = 0.5 * (Hpp + Hpp.T)
        E0_vals, C_P = eigh(Hpp)
        E0_vals = E0_vals[:NROOTS]
        C_P = C_P[:, :NROOTS]

        # Build H_QP
        H_QP = backend.build_hqp(ckpt_dets, verbose=False)

        # Per-state diagonal-resolvent Bloch correction (same as step2)
        E_bloch = []
        for k in range(NROOTS):
            E0_k = E0_vals[k]
            A_q_diag = 1.0 / (E0_k + DELTA - H_QQ_diag)
            A_q_diag = np.clip(A_q_diag, -1e10, 1e10)

            if hasattr(H_QP, 'multiply'):  # sparse
                weighted = H_QP.multiply(A_q_diag[:, np.newaxis])
                correction = H_QP.T @ weighted
            else:  # dense
                weighted = H_QP * A_q_diag[:, np.newaxis]
                correction = H_QP.T @ weighted

            H_eff = Hpp + correction
            H_eff = 0.5 * (H_eff + H_eff.T)
            ev_all, C_eff = eigh(H_eff)

            # Overlap tracking for correct root identification
            v_k_bare = C_P[:, k]
            overlaps = [abs(float(np.dot(C_eff[:, m], v_k_bare))) for m in range(len(ev_all))]
            m_star = int(np.argmax(overlaps))
            E_bloch.append(float(ev_all[m_star]))

        dE_bloch = [(E_bloch[i] - e_dmrg[i]) * 1000 for i in range(NROOTS)]
        dE_bare = [(E0_vals[i] - e_dmrg[i]) * 1000 for i in range(NROOTS)]

        bloch_results[p_size] = {
            'E_bloch': E_bloch,
            'E_bare': [float(e) for e in E0_vals],
            'dE_bare_mH': dE_bare,
            'dE_bloch_mH': dE_bloch,
        }

        fname = "{}/step2_P{:04d}.json".format(R_dir, p_size)
        with open(fname, 'w') as f:
            json.dump(bloch_results[p_size], f, indent=2)
        print("    P={:4d}: S0 bare={:+.2f} mH  bloch={:+.3f} mH  |  S1 bloch={:+.1f} mH".format(
            p_size, dE_bare[0], dE_bloch[0], dE_bloch[1]), flush=True)

    t_geo = time.perf_counter() - t_geo0

    # ── Summarize this geometry ──
    summary_R = {
        'R': R,
        'M': M,
        'E_HF': float(mf.e_tot),
        'e_dmrg': e_dmrg,
        'bloch_by_P': {str(p): bloch_results[p] for p in P_TARGETS if p in bloch_results},
        'timing_s': t_geo,
    }
    all_summaries[str(R)] = summary_R

    with open("{}/summary.json".format(R_dir), 'w') as f:
        json.dump(summary_R, f, indent=2)

    print("\n  R={:.1f} done. {:.0f}s wall.".format(R, t_geo), flush=True)

# ═══════════════ Cross-geometry summary ═══════════════
print("\n" + "=" * 64)
print("Cross-Geometry Convergence Summary")
print("=" * 64)

# Table: ground state Bloch error at each (R, P)
print("\n  Ground state |ΔE_Bloch| (mH) vs DMRG-CI:")
header = "  {:>5}".format("R(Å)")
for p in P_TARGETS:
    header += " {:>8}".format("P="+str(p))
print(header)
print("  " + "-" * (6 + 9 * len(P_TARGETS)))

for R in BOND_LENGTHS:
    key = str(R)
    if key not in all_summaries:
        continue
    s = all_summaries[key]
    line = "  {:>5.1f}".format(R)
    for p in P_TARGETS:
        pk = str(p)
        if pk in s.get('bloch_by_P', {}):
            dE = abs(s['bloch_by_P'][pk]['dE_bloch_mH'][0])
            marker = " ✓" if dE <= 1.6 else ""
            line += " {:>7.2f}{}".format(dE, marker)
        else:
            line += " {:>8}".format("—")
    print(line)

# Excited states (S1) summary
print("\n  Excited state S₁ |ΔE_Bloch| (mH):")
header = "  {:>5}".format("R(Å)")
for p in P_TARGETS:
    header += " {:>8}".format("P="+str(p))
print(header)
print("  " + "-" * (6 + 9 * len(P_TARGETS)))

for R in BOND_LENGTHS:
    key = str(R)
    if key not in all_summaries:
        continue
    s = all_summaries[key]
    line = "  {:>5.1f}".format(R)
    for p in P_TARGETS:
        pk = str(p)
        if pk in s.get('bloch_by_P', {}):
            dE = abs(s['bloch_by_P'][pk]['dE_bloch_mH'][1])
            line += " {:>8.1f}".format(dE)
        else:
            line += " {:>8}".format("—")
    print(line)

# Save global summary
with open("{}/global_summary.json".format(OUTDIR), 'w') as f:
    json.dump({
        'system': 'N2/cc-pVDZ CAS(10,10) bond scan',
        'P_init': P_INIT, 'P_max': P_MAX, 'batch': BATCH_SIZE,
        'bond_lengths': BOND_LENGTHS,
        'P_targets': P_TARGETS,
        'method': 'iterative shared P + m=0 per-state Bloch H^eff',
        'summaries': all_summaries,
    }, f, indent=2)

print("\n" + "=" * 64)
print("Bond scan complete. Results: {}".format(OUTDIR))
print("=" * 64)
