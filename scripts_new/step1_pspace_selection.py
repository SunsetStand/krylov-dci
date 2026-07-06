#!/usr/bin/env python3
"""
Step 1 (fixed): Iterative P-space selection — shared or per-state.

Modes:
  --mode shared   : Σ_k |σ_k|² scoring → single P for all roots (original)
  --mode per_state: |σ_k|² scoring → separate P_k for each root k

In shared mode, saves correct sub-matrix E_bare for each checkpoint P size.
Fix: checkpoint E_bare is computed from the exact p_target×p_target submatrix,
not from the larger H_PP. Step2 will also recompute from scratch and cross-check.

System: N2/cc-pVDZ CAS(10,10)
Reference: DMRG-CI = exact FCI in this CAS
"""
import sys, os, time, json, itertools, argparse
import numpy as np
from numpy.linalg import eigh

sys.path.insert(0, '/data/home/wangcx/krylov-dci')
from src_mf.pyscf_backend import QSpaceIndex, KDCIBackend
from src.determinants import hf_determinant, bit_positions
from src.hamiltonian import Hamiltonian
from pyscf import gto, scf, ao2mo
from pyscf.fci import cistring, direct_spin1

N_CORE = 2
N_ACT = 10
NROOTS = 6
P_INIT = 200
BATCH_SIZE = 200
P_TARGETS = [200, 400, 800, 1200, 1600, 2000]
P_MAX = max(P_TARGETS)

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['shared', 'per_state'], default='shared',
                    help='Selection mode: shared P (default) or per-state P_k')
parser.add_argument('--outdir', default='/data/home/wangcx/krylov-dci/checkpoints_pspace')
args = parser.parse_args()

MODE = args.mode
OUTDIR = args.outdir if MODE == 'shared' else args.outdir + '_perstate'
os.makedirs(OUTDIR, exist_ok=True)

print("=" * 64)
print("Step 1: Iterative P-space Selection (mode={})".format(MODE))
print("N2/cc-pVDZ CAS({},{})  P_init={} -> P_max={}".format(N_ACT, N_ACT, P_INIT, P_MAX))
print("batch={}  nroots={}  checkpoints={}".format(BATCH_SIZE, NROOTS, P_TARGETS))
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
hdiag_full = q_idx.hdiag
print("  {}a+{}b in {} orbs, M={:,}".format(ne[0], ne[1], N_ACT, M), flush=True)

# ── DMRG-CI reference ──
print("[2] Computing DMRG-CI reference...", flush=True)
ef, _ = direct_spin1.FCI().kernel(h1a, era, N_ACT, ne,
                                   nroots=NROOTS, verbose=0)
e_dmrg = [float(e) for e in np.atleast_1d(ef)[:NROOTS]]
for k, e in enumerate(e_dmrg):
    print("  root {}: {:.8f} Ha".format(k, e), flush=True)

# ── Hamiltonian ──
h2_4d = ao2mo.restore('s1', era, N_ACT).reshape(N_ACT, N_ACT, N_ACT, N_ACT)
ham = Hamiltonian(h1=h1a, h2=h2_4d, E_nuc=0.0, E_HF=0.0)
hf_a, hf_b = hf_determinant(*ne)
ao = bit_positions(hf_a)
bo = bit_positions(hf_b)
av = [p for p in range(N_ACT) if p not in ao]
bv = [p for p in range(N_ACT) if p not in bo]

full_dets = []
for ai, a_str in enumerate(as_):
    for bi, b_str in enumerate(bs_):
        full_dets.append((int(a_str), int(b_str)))
assert len(full_dets) == M

# ── HFPT2 initial P ──
print("\n[3] Initial P via HFPT2 (target {})...".format(P_INIT), flush=True)
E_HF = ham.matrix_element((hf_a, hf_b), (hf_a, hf_b))

def gen_hfpt2_scores():
    scores = []
    for i in ao:
        for a in av:
            d = (hf_a ^ (1 << i) | (1 << a), hf_b)
            hij = ham.matrix_element(d, (hf_a, hf_b))
            den = E_HF - ham.matrix_element(d, d)
            if abs(den) > 1e-12:
                scores.append((d, -hij * hij / den))
    for i in bo:
        for a in bv:
            d = (hf_a, hf_b ^ (1 << i) | (1 << a))
            hij = ham.matrix_element(d, (hf_a, hf_b))
            den = E_HF - ham.matrix_element(d, d)
            if abs(den) > 1e-12:
                scores.append((d, -hij * hij / den))
    for i1, i2 in itertools.combinations(ao, 2):
        for a1, a2 in itertools.combinations(av, 2):
            d = (hf_a ^ (1 << i1) ^ (1 << i2) | (1 << a1) | (1 << a2), hf_b)
            hij = ham.matrix_element(d, (hf_a, hf_b))
            den = E_HF - ham.matrix_element(d, d)
            if abs(den) > 1e-12:
                scores.append((d, -hij * hij / den))
    for i1, i2 in itertools.combinations(bo, 2):
        for a1, a2 in itertools.combinations(bv, 2):
            d = (hf_a, hf_b ^ (1 << i1) ^ (1 << i2) | (1 << a1) | (1 << a2))
            hij = ham.matrix_element(d, (hf_a, hf_b))
            den = E_HF - ham.matrix_element(d, d)
            if abs(den) > 1e-12:
                scores.append((d, -hij * hij / den))
    for i in ao:
        for j in bo:
            for a in av:
                for b in bv:
                    d = (hf_a ^ (1 << i) | (1 << a),
                         hf_b ^ (1 << j) | (1 << b))
                    hij = ham.matrix_element(d, (hf_a, hf_b))
                    den = E_HF - ham.matrix_element(d, d)
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
print("  Initial P = {} determinants (from {} SD excitations)".format(
    len(init_dets), len(scores)))

det_to_full = {d: i for i, d in enumerate(full_dets)}


def build_hpp(dets):
    n = len(dets)
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            v = ham.matrix_element(dets[i], dets[j])
            H[i, j] = v
            H[j, i] = v
    return H


def extend_hpp(H_old, old_dets, new_dets):
    N_old = len(old_dets)
    n_add = len(new_dets)
    H_new = np.zeros((N_old + n_add, N_old + n_add))
    H_new[:N_old, :N_old] = H_old
    for i_local, det_new in enumerate(new_dets):
        row = N_old + i_local
        for j in range(N_old):
            val = ham.matrix_element(det_new, old_dets[j])
            H_new[row, j] = val
            H_new[j, row] = val
        for j_local in range(i_local + 1):
            col = N_old + j_local
            val = ham.matrix_element(det_new, new_dets[j_local])
            H_new[row, col] = val
            H_new[col, row] = val
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


def iterative_selection(p_dets_init, p_set_init, H_PP_init,
                        sigmas_for_scoring, mode_label, save_dir):
    p_dets = list(p_dets_init)
    p_full_indices = list(p_set_init)
    p_set = set(p_set_init)
    H_PP = H_PP_init.copy()
    N_p = len(p_dets)
    checkpoints = {}

    print("\n[{}] Iterative selection: P={} -> P={}".format(
        mode_label, N_p, P_MAX), flush=True)
    print("{:>4} {:>7} {:>14} {:>14} {:>12} {:>8}".format(
        "iter", "P_size", "E0_bare", "dE0_bare(mH)", "max_w", "wall(s)"))
    print("-" * 64, flush=True)

    iter_num = 0
    while N_p < P_MAX:
        t_iter = time.perf_counter()

        E_P, C_P = eigh(H_PP)

        n_roots_used = len(sigmas_for_scoring)
        n_sigma = min(n_roots_used, N_p)
        sigmas = []
        for sk in range(n_sigma):
            k = sigmas_for_scoring[sk]
            vec_full = np.zeros(M)
            for local_i, global_i in enumerate(p_full_indices):
                vec_full[global_i] = C_P[local_i, k]
            sigma_k = backend.sigma(vec_full)
            sigmas.append((E_P[k], sigma_k))

        weights = score_q_space(sigmas, p_set)

        q_candidates = [(qi, float(weights[qi])) for qi in range(M)
                        if qi not in p_set and weights[qi] > 0]
        q_candidates.sort(key=lambda x: x[1], reverse=True)

        n_add = min(BATCH_SIZE, len(q_candidates))
        if n_add == 0:
            print("  No more determinants. Stopping.", flush=True)
            break

        max_w = q_candidates[0][1]
        new_global_indices = [q_c[0] for q_c in q_candidates[:n_add]]
        new_dets = [full_dets[qi] for qi in new_global_indices]

        H_PP = extend_hpp(H_PP, p_dets, new_dets)
        p_dets.extend(new_dets)
        p_full_indices.extend(new_global_indices)
        p_set.update(new_global_indices)
        N_p = len(p_dets)

        dE0_mH = (E_P[0] - e_dmrg[0]) * 1000
        t_elapsed = time.perf_counter() - t_iter
        iter_num += 1

        print("{:>4} {:>7} {:>14.8f} {:>+14.3f} {:>12.3e} {:>8.1f}".format(
            iter_num, N_p, E_P[0], dE0_mH, max_w, t_elapsed), flush=True)

        for p_target in P_TARGETS:
            if N_p >= p_target and p_target not in checkpoints:
                dets_p = p_dets[:p_target]
                idx_p = p_full_indices[:p_target]

                # KEY FIX: diagonalize exact p_target x p_target submatrix
                H_exact = H_PP[:p_target, :p_target]
                H_exact = 0.5 * (H_exact + H_exact.T)
                E_exact, _ = eigh(H_exact)

                checkpoints[p_target] = {
                    'P': p_target,
                    'p_dets': [(int(a), int(b)) for a, b in dets_p],
                    'p_full_indices': [int(i) for i in idx_p],
                    'E_bare': [float(e) for e in E_exact[:NROOTS]],
                    'dE0_bare_mH': float((E_exact[0] - e_dmrg[0]) * 1000),
                    'iter_num': iter_num,
                    'mode': mode_label,
                }
                fname = "{}/step1_P{:04d}.json".format(save_dir, p_target)
                with open(fname, 'w') as f:
                    json.dump(checkpoints[p_target], f, indent=2)
                print("    OK saved {} (E0_bare={:.8f}, dE0={:.3f} mH)".format(
                    fname, E_exact[0], (E_exact[0]-e_dmrg[0])*1000), flush=True)

    return p_dets, p_full_indices, H_PP, checkpoints


# ═══════════════ Main ═══════════════
wall_total = time.perf_counter()

if MODE == 'shared':
    init_full_indices = [det_to_full[d] for d in init_dets]
    H_PP_init = build_hpp(init_dets)

    p_dets, p_idx, H_PP, checkpoints = iterative_selection(
        init_dets, init_full_indices, H_PP_init,
        sigmas_for_scoring=list(range(min(NROOTS, 5))),
        mode_label='shared', save_dir=OUTDIR
    )
else:
    for k in range(NROOTS):
        print("\n" + "-" * 60)
        print("  >>> Building P-space for root {} <<<".format(k))
        print("-" * 60, flush=True)

        init_full_indices = [det_to_full[d] for d in init_dets]
        H_PP_init = build_hpp(init_dets)

        state_outdir = "{}/root{}".format(OUTDIR, k)
        os.makedirs(state_outdir, exist_ok=True)

        p_dets_k, p_idx_k, H_PP_k, ck_k = iterative_selection(
            init_dets, init_full_indices, H_PP_init,
            sigmas_for_scoring=[k],
            mode_label='root_{}'.format(k), save_dir=state_outdir
        )

        for p_target in P_TARGETS:
            if p_target in ck_k:
                ck_k[p_target]['root'] = k
                ck_k[p_target]['mode'] = 'per_state'

wall_total = time.perf_counter() - wall_total
print("\n" + "=" * 64)
print("Step 1 ({}) complete. {:.0f}s wall.".format(MODE, wall_total))
print("Output: {}".format(OUTDIR))
print("=" * 64)
