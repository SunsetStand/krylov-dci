#!/usr/bin/env python3
"""
Phase A — Per-State P-space Selection for S1 (first excited state)

Scoring:  |<det| σ_S1 >|² / |E_ref - H_det,det|
  where σ_S1 = H · ψ_S1^{(P)}  (sigma vector for P-space S1 wavefunction)
  and   ψ_S1^{(P)} = C_P[:, 1]  (2nd eigenvector of H_PP)

Two variants of E_ref:
  --eref fci   : E_ref = FCI exact S1 energy (fixed)
  --eref hpp   : E_ref = 2nd eigenvalue of H_PP (updates as P expands)

Krylov: T = A²·H_QP (build_basis) / T = A²·H_O'·U (propagate)
  with A_q = 1/(E_ref - H_qq), m_max=1

Convergence: monitor 2nd eigenvalue of H^eff (should → FCI S1 energy)

Usage:
    python phaseA_cas10_perstate_s1.py --P 400,600,800,1000,2000 --eref fci
    python phaseA_cas10_perstate_s1.py --P 400,600,800,1000,2000 --eref hpp
"""
import sys, os, time, json, argparse, itertools, gc
import numpy as np
from numpy.linalg import eigh, svd, norm

PROJECT_ROOT = '/data/home/wangcx/krylov-dci'
sys.path.insert(0, PROJECT_ROOT)

from src_mf import QSpaceIndex, KDCIBackend, KDCISparse
from src.effective_h import build_effective_H, diagonalize_effective_H
from src.determinants import hf_determinant, bit_positions
from src.hamiltonian import Hamiltonian
from pyscf import gto, scf, ao2mo
from pyscf.fci import cistring, direct_spin1

# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════
args = argparse.ArgumentParser()
args.add_argument('--P', type=str, default='400,600,800,1000,2000,4000',
                  help='P-space checkpoints (comma-separated)')
args.add_argument('--m-max', type=int, default=1, help='Krylov depth (default 1)')
args.add_argument('--svd-threshold', type=float, default=1e-3)
args.add_argument('--batch', type=int, default=200)
args.add_argument('--eref', type=str, default='fci', choices=['fci', 'hpp'],
                  help='E_ref source: fci (exact S1) or hpp (2nd eigenvalue of H_PP)')
args.add_argument('--tag', type=str, default='s1')
args = args.parse_args()

P_CHECKPOINTS = sorted([int(x) for x in args.P.split(',')])
M_MAX = args.m_max; SVD_THR = args.svd_threshold; BATCH = args.batch
TAG = args.tag; P_MAX = max(P_CHECKPOINTS)
TARGET_STATE = 1  # S1 = first excited state
EREFF_FCI = (args.eref == 'fci')

N_ACT = 10; N_CORE = 2; NROOTS = 6; R = 1.1; ne = (5, 5)
print("=" * 70)
print(f"Phase A — Per-State S{TARGET_STATE}  Matrix-Free Krylov m=0..{M_MAX}  E_ref={'FCI' if EREFF_FCI else 'H_PP'}")
print(f"N2/cc-pVDZ R={R}  checkpoints={P_CHECKPOINTS}")
print(f"SVD thr={SVD_THR}  m_max={M_MAX}  batch={BATCH}")
print("=" * 70, flush=True)

t0 = time.perf_counter()
mol = gto.M(atom=f'N 0 0 0; N 0 0 {R}', basis='cc-pVDZ', verbose=0)
mf = scf.RHF(mol).run(verbose=0)
na_o = list(range(N_CORE, N_CORE + N_ACT))
norb = mf.mo_coeff.shape[1]
h1_mo = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
eri_4d = ao2mo.full(mol.intor('int2e'), mf.mo_coeff, compact=False)
eri_4d = eri_4d.reshape(norb, norb, norb, norb)
h1a = h1_mo[np.ix_(na_o, na_o)]; era = eri_4d[np.ix_(na_o, na_o, na_o, na_o)]
as_ = cistring.gen_strings4orblist(range(N_ACT), ne[0])
bs_ = cistring.gen_strings4orblist(range(N_ACT), ne[1])
na, nb = len(as_), len(bs_); M_all = na * nb
q_idx = QSpaceIndex(as_, bs_, N_ACT, ne, h1a, era)
backend = KDCIBackend(q_idx); kdci_sparse = KDCISparse(q_idx)
hdiag = q_idx.hdiag

print("  FCI reference...", flush=True)
ef, _ = direct_spin1.FCI().kernel(h1a, era, N_ACT, ne, nroots=NROOTS, verbose=0)
e_fci = [float(e) for e in np.atleast_1d(ef)[:NROOTS]]
for i in range(NROOTS):
    exc = f"  ({(e_fci[i] - e_fci[0]) * 1000:.1f} mH)" if i > 0 else ""
    print(f"    S{i}: {e_fci[i]:.12f} Ha{exc}")
E_FCI_S1 = e_fci[TARGET_STATE]
print(f"  Target: S{TARGET_STATE} = {E_FCI_S1:.12f} Ha  "
      f"(excitation { (E_FCI_S1 - e_fci[0]) * 1000:.1f} mH)")

h2_4d = ao2mo.restore('s1', era, N_ACT).reshape(N_ACT, N_ACT, N_ACT, N_ACT)
ham = Hamiltonian(h1=h1a, h2=h2_4d, E_nuc=0.0, E_HF=0.0)
hf_a, hf_b = hf_determinant(*ne)
E_HF = ham.matrix_element((hf_a, hf_b), (hf_a, hf_b))
full_dets = [(int(a), int(b)) for a in as_ for b in bs_]
det_to_full = {d: i for i, d in enumerate(full_dets)}
print(f"  CAS({N_ACT},{sum(ne)}): M={M_all:,}  ({time.perf_counter() - t0:.0f}s)\n")


# ═══════════════════════════════════════════════════════════════
# HFPT2 initial P space
# ═══════════════════════════════════════════════════════════════
ao = bit_positions(hf_a); bo = bit_positions(hf_b)
av = [p for p in range(N_ACT) if p not in ao]
bv = [p for p in range(N_ACT) if p not in bo]

def gen_hfpt2_scores():
    sc = []
    for i in ao:
        for a in av:
            d = (hf_a ^ (1 << i) | (1 << a), hf_b)
            hij = ham.matrix_element(d, (hf_a, hf_b)); den = E_HF - ham.matrix_element(d, d)
            if abs(den) > 1e-12: sc.append((d, -hij * hij / den))
    for i in bo:
        for a in bv:
            d = (hf_a, hf_b ^ (1 << i) | (1 << a))
            hij = ham.matrix_element(d, (hf_a, hf_b)); den = E_HF - ham.matrix_element(d, d)
            if abs(den) > 1e-12: sc.append((d, -hij * hij / den))
    for i1, i2 in itertools.combinations(ao, 2):
        for a1, a2 in itertools.combinations(av, 2):
            d = (hf_a ^ (1 << i1) ^ (1 << i2) | (1 << a1) | (1 << a2), hf_b)
            hij = ham.matrix_element(d, (hf_a, hf_b)); den = E_HF - ham.matrix_element(d, d)
            if abs(den) > 1e-12: sc.append((d, -hij * hij / den))
    for i1, i2 in itertools.combinations(bo, 2):
        for a1, a2 in itertools.combinations(bv, 2):
            d = (hf_a, hf_b ^ (1 << i1) ^ (1 << i2) | (1 << a1) | (1 << a2))
            hij = ham.matrix_element(d, (hf_a, hf_b)); den = E_HF - ham.matrix_element(d, d)
            if abs(den) > 1e-12: sc.append((d, -hij * hij / den))
    for i in ao:
        for j in bo:
            for a in av:
                for b in bv:
                    d = (hf_a ^ (1 << i) | (1 << a), hf_b ^ (1 << j) | (1 << b))
                    hij = ham.matrix_element(d, (hf_a, hf_b)); den = E_HF - ham.matrix_element(d, d)
                    if abs(den) > 1e-12: sc.append((d, -hij * hij / den))
    sc.sort(key=lambda x: x[1], reverse=True)
    return sc

P_INIT = 200
scores = gen_hfpt2_scores()
init_dets = [(hf_a, hf_b)]
for d, _ in scores:
    if d not in init_dets: init_dets.append(d)
    if len(init_dets) >= P_INIT: break
print(f"  HFPT2 initial P={len(init_dets)}\n")


# ═══════════════════════════════════════════════════════════════
# H_PP builders
# ═══════════════════════════════════════════════════════════════
def build_hpp(dets):
    n = len(dets); H = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            v = ham.matrix_element(dets[i], dets[j]); H[i, j] = v; H[j, i] = v
    return H

def extend_hpp(H_old, old_dets, new_dets):
    No = len(old_dets); na = len(new_dets); Hn = np.zeros((No + na, No + na))
    Hn[:No, :No] = H_old
    for il, dn in enumerate(new_dets):
        r = No + il
        for j in range(No):
            v = ham.matrix_element(dn, old_dets[j]); Hn[r, j] = v; Hn[j, r] = v
        for jl in range(il + 1):
            c = No + jl; v = ham.matrix_element(dn, new_dets[jl]); Hn[r, c] = v; Hn[c, r] = v
    return Hn


# ═══════════════════════════════════════════════════════════════
# Matrix-Free build_basis: T = A² · H_QP
#   A_q = 1/(E_ref - H_qq)  ← per-state E_ref
# ═══════════════════════════════════════════════════════════════
def build_basis_mf(p_dets, E_ref, tag=""):
    """T = A² · H_QP columns, stream to memmap, SVD → U.
    A_q = 1/(E_ref - hdiag), centred at target state energy.
    """
    N = len(p_dets)
    denom = E_ref - hdiag
    A_q = np.where(np.abs(denom) > 1e-10, 1.0 / denom, 0.0)
    A_half = np.sqrt(np.abs(A_q))

    tmpdir = f'{PROJECT_ROOT}/tmp'; os.makedirs(tmpdir, exist_ok=True)
    fpath = f'{tmpdir}/phaseA_ps1_L0_N{N}_{TAG}_{tag}_pid{os.getpid()}.dat'
    T = np.memmap(fpath, dtype='float64', mode='w+', shape=(M_all, N))

    p_idx_set = set()
    for pa, pb in p_dets:
        idx = q_idx.flat_index(int(pa), int(pb))
        if idx is not None and idx >= 0: p_idx_set.add(idx)

    t0 = time.perf_counter()
    print(f"    [build_basis] T=A²·H_QP, N={N} cols, E_ref={E_ref:.8f} → memmap...", flush=True)
    for p in range(N):
        pa, pb = int(p_dets[p][0]), int(p_dets[p][1])
        ia = q_idx._alpha_idx.get(pa); ib = q_idx._beta_idx.get(pb)
        if ia is None or ib is None: continue
        ci_unit = np.zeros((na, nb)); ci_unit[ia, ib] = 1.0
        sigma_flat = backend.sigma_full(ci_unit).reshape(-1)
        for q in p_idx_set: sigma_flat[q] = 0.0
        T[:, p] = A_q * sigma_flat  # T = A*H_QP (A1, matches src_mf)

        if (p + 1) % max(1, N // 5) == 0:
            print(f"      col {p+1}/{N} ({time.perf_counter()-t0:.0f}s)", flush=True)
    T.flush()

    t_svd = time.perf_counter()
    print(f"    SVD({M_all},{N})...", flush=True)
    U, sigma, Vt = svd(T, full_matrices=False)
    smax = sigma[0] if len(sigma) > 0 else 0
    mask = sigma >= SVD_THR * max(1.0, smax)
    d = int(np.sum(mask))
    U_ret = U[:, mask]; sig_ret = sigma[mask]
    e = time.perf_counter() - t_svd
    ratios = ", ".join(f"{s/smax:.4f}" for s in sigma[:min(8, len(sigma))])
    print(f"    SVD done: {e:.0f}s, {N}→d={d} (σ/σ₁=[{ratios}])", flush=True)

    try: del T; gc.collect(); os.unlink(fpath)
    except: pass
    return U_ret, d, A_q


# ═══════════════════════════════════════════════════════════════
# Matrix-Free propagate_basis: T = A² · H_O' · U
# ═══════════════════════════════════════════════════════════════
def propagate_basis_mf(U_basis, A_q, p_idx_set, tag=""):
    """Propagate with per-state A_q."""
    A_half = np.sqrt(np.abs(A_q))
    M_dim, d_old = U_basis.shape
    if d_old == 0: return U_basis.copy(), d_old

    tmpdir = f'{PROJECT_ROOT}/tmp'; os.makedirs(tmpdir, exist_ok=True)
    fpath = f'{tmpdir}/phaseA_ps1_prop_d{d_old}_{TAG}_{tag}_pid{os.getpid()}.dat'
    T = np.memmap(fpath, dtype='float64', mode='w+', shape=(M_dim, d_old))

    t0 = time.perf_counter()
    print(f"    [propagate] d={d_old}, T=A²·H_O'·U → memmap...", flush=True)
    for k in range(d_old):
        b_k = U_basis[:, k]
        sigma_k = backend.sigma_full(b_k.reshape(na, nb)).reshape(-1)
        residual = sigma_k - hdiag * b_k
        for q in p_idx_set: residual[q] = 0.0
        T[:, k] = A_q * residual  # T = A*residual (A1, matches src_mf)

        if (k + 1) % max(1, d_old // 5) == 0:
            print(f"      col {k+1}/{d_old} ({time.perf_counter()-t0:.0f}s)", flush=True)
    T.flush()

    t_svd = time.perf_counter()
    print(f"    SVD({M_dim},{d_old})...", flush=True)
    U_svd, sigma, Vt = svd(T, full_matrices=False)
    smax = sigma[0] if len(sigma) > 0 else 0
    mask = sigma >= SVD_THR * max(1.0, smax)
    n_keep = int(np.sum(mask))
    U_trunc = U_svd[:, mask]
    e = time.perf_counter() - t_svd
    print(f"    SVD done: {e:.0f}s, {d_old}→{n_keep} kept", flush=True)

    try: del T; gc.collect(); os.unlink(fpath)
    except: pass

    # MGS
    basis_list = [U_basis[:, j] for j in range(d_old)]
    new_count = 0
    for k in range(U_trunc.shape[1]):
        v = U_trunc[:, k].copy()
        for b in basis_list:
            v -= np.dot(b, v) * b
        nrm = np.linalg.norm(v)
        if nrm > 1e-10:
            v /= nrm
            basis_list.append(v)
            new_count += 1
    U_new = np.column_stack(basis_list)
    d_new = len(basis_list)
    print(f"    MGS: d_old={d_old} + {new_count} new = {d_new}", flush=True)
    return U_new, d_new


# ═══════════════════════════════════════════════════════════════
# Per-state Krylov pipeline
# ═══════════════════════════════════════════════════════════════
def krylov_mf_pipeline_perstate(H_PP, p_dets, E_ref, p_idx_set_mf, tag=""):
    """build_basis + propagate, H_eff at each m. Monitor S1 (2nd eigenvalue)."""
    results = []

    U_0, d_0, A_q = build_basis_mf(p_dets, E_ref, tag)

    def build_blocks(U_basis, p_dets):
        d = U_basis.shape[1]; Np = len(p_dets)
        if d == 0: return np.zeros((0, 0)), np.zeros((Np, 0))
        t0 = time.perf_counter()
        print(f"    [blocks] d={d}...", flush=True)
        H_KK = np.zeros((d, d)); H_PK = np.zeros((Np, d))
        p_flat = kdci_sparse.q_idx.p_indices(p_dets)
        p_valid = p_flat >= 0; p_f = p_flat[p_valid]
        for k in range(d):
            ci_k = U_basis[:, k].reshape(na, nb)
            sk = backend.sigma_full(ci_k).reshape(-1)
            H_KK[:, k] = U_basis.T @ sk
            H_PK[p_valid, k] = sk[p_f]
            if (k + 1) % max(1, d // 5) == 0:
                print(f"      {k+1}/{d} ({time.perf_counter()-t0:.0f}s)", flush=True)
        H_KK = 0.5 * (H_KK + H_KK.T)
        print(f"    [blocks] done: {time.perf_counter()-t0:.0f}s", flush=True)
        return H_KK, H_PK

    U_m = U_0; d_m = d_0
    H_KK, H_PK = build_blocks(U_m, p_dets)
    H_eff = build_effective_H(H_PP, H_PK, H_KK, E_ref, delta=0.0)
    ev = diagonalize_effective_H(H_eff, n_states=NROOTS)[0]
    # Monitor: 2nd eigenvalue (index 1) and closest to E_ref
    dE_s1 = (ev[TARGET_STATE] - e_fci[TARGET_STATE]) * 1000
    idx_closest = np.argmin(np.abs(ev - E_ref))
    results.append({'d': d_m, 'ev': ev, 'dE_s1': dE_s1, 'idx_closest': idx_closest,
                    'e_closest': ev[idx_closest], 'U': U_m})
    print(f"    m=0: d={d_m}, ev[S1]={ev[TARGET_STATE]:.12f} "
          f"dE_S1={dE_s1:+.3f} mH  (closest: ev[{idx_closest}]={ev[idx_closest]:.12f})",
          flush=True)

    for m in range(1, M_MAX + 1):
        U_m, d_m = propagate_basis_mf(U_m, A_q, p_idx_set_mf, f"{tag}_m{m}")
        if d_m == results[-1]['d']:
            print(f"    m={m}: no new directions, stopping", flush=True)
            results.append(results[-1])
            break

        H_KK, H_PK = build_blocks(U_m, p_dets)
        H_eff = build_effective_H(H_PP, H_PK, H_KK, E_ref, delta=0.0)
        ev = diagonalize_effective_H(H_eff, n_states=NROOTS)[0]
        dE_s1 = (ev[TARGET_STATE] - e_fci[TARGET_STATE]) * 1000
        ddE = dE_s1 - results[-1]['dE_s1']
        idx_closest = np.argmin(np.abs(ev - E_ref))
        results.append({'d': d_m, 'ev': ev, 'dE_s1': dE_s1, 'idx_closest': idx_closest,
                        'e_closest': ev[idx_closest], 'U': U_m})
        print(f"    m={m}: d={d_m}, ev[S1]={ev[TARGET_STATE]:.12f} "
              f"dE_S1={dE_s1:+.3f} mH (Δ={ddE:+.1f})", flush=True)

    return results, A_q


# ═══════════════════════════════════════════════════════════════
# Checkpoint evaluation (per-state)
# ═══════════════════════════════════════════════════════════════
def eval_checkpoint(p_dets, p_full_idx, H_PP_sub, p_target, it_num, E_ref):
    N = len(p_dets)
    p_idx_set = set()
    for pa, pb in p_dets:
        idx = q_idx.flat_index(int(pa), int(pb))
        if idx is not None and idx >= 0: p_idx_set.add(idx)

    E_P_vals, _ = eigh(H_PP_sub)
    E0 = E_P_vals[0]; E_s1_hpp = E_P_vals[TARGET_STATE]
    label = 'FCI' if EREFF_FCI else 'H_PP'
    print(f"  P={N}, E_ref={E_ref:.8f} ({label})", flush=True)
    print(f"  H_PP: E0={E0:.8f}  S1={E_s1_hpp:.8f}  "
          f"(dE_S1(HPP)={(E_s1_hpp - e_fci[TARGET_STATE]) * 1000:+.1f} mH)",
          flush=True)

    tag = f"P{p_target}_i{it_num}"
    kr_results, A_q = krylov_mf_pipeline_perstate(H_PP_sub, p_dets, E_ref, p_idx_set, tag)

    m_last = min(M_MAX, len(kr_results) - 1)
    r_last = kr_results[m_last]
    print(f"  Summary P={p_target}: d(m=0)={kr_results[0]['d']}  "
          f"dE_S1(m=0)={kr_results[0]['dE_s1']:+.1f}  "
          f"dE_S1(m={m_last})={r_last['dE_s1']:+.1f} mH  "
          f"d(m={m_last})={r_last['d']}", flush=True)
    ev_str = ", ".join(f"{r_last['ev'][i]:.8f}" for i in range(min(6, NROOTS)))
    print(f"  H^eff ev[0..{min(5, NROOTS - 1)}] = [{ev_str}]", flush=True)

    return {
        'P': p_target, 'N': N, 'iter': it_num,
        'E_ref': float(E_ref), 'eref_label': label,
        'E0_HPP': float(E0), 'S1_HPP': float(E_s1_hpp),
        'krylov': {m: {'d': kr['d'],
                       'dE_S1_mH': float(kr['dE_s1']),
                       'ev_S1': float(kr['ev'][TARGET_STATE]),
                       'e_closest': float(kr['e_closest']),
                       'idx_closest': int(kr['idx_closest'])}
                   for m, kr in enumerate(kr_results)},
    }


# ═══════════════════════════════════════════════════════════════
# Main: iterative per-state P expansion
# ═══════════════════════════════════════════════════════════════
p_dets = list(init_dets)
p_full_idx = [det_to_full[d] for d in p_dets]
p_set = set(p_full_idx)
H_PP = build_hpp(p_dets)
N_p = len(p_dets)
all_results = {}

print(f"Iterative per-state (S{TARGET_STATE}) P expansion: {N_p} → {P_MAX}")
print(f"E_ref mode: {'FCI (fixed)' if EREFF_FCI else 'H_PP (dynamic)'}")
hdr = f"{'iter':>4} {'P':>6} {'E_S1(HPP)':>14} {'dE_S1_mH':>10} {'E_ref':>14} {'max_w':>10} {'wall':>8}"
print(hdr)
print("-" * 72, flush=True)

total_t0 = time.perf_counter()
it = 0

while N_p < P_MAX:
    t_it = time.perf_counter()
    E_P, C_P = eigh(H_PP)

    # Determine E_ref
    if EREFF_FCI:
        E_ref = E_FCI_S1
    else:
        E_ref = E_P[TARGET_STATE]

    # ── Per-state scoring (S1 only) ──
    vec_s1 = np.zeros(M_all)
    for li, gi in enumerate(p_full_idx):
        vec_s1[gi] = C_P[li, TARGET_STATE]
    sigma_s1 = backend.sigma(vec_s1)

    weights = np.zeros(M_all)
    abs_sigma = np.abs(sigma_s1)
    for qi in range(M_all):
        if qi in p_set: continue
        c2 = abs_sigma[qi] ** 2
        if c2 < 1e-24: continue
        denom = abs(E_ref - hdiag[qi])
        if denom < 1e-8: denom = 1e-8
        weights[qi] = c2 / denom

    cands = [(qi, float(weights[qi])) for qi in range(M_all)
             if qi not in p_set and weights[qi] > 0]
    cands.sort(key=lambda x: x[1], reverse=True)
    n_add = min(BATCH, len(cands))
    max_w = cands[0][1] if cands else 0

    new_gi = [c[0] for c in cands[:n_add]]
    new_dets = [full_dets[qi] for qi in new_gi]
    H_PP = extend_hpp(H_PP, p_dets, new_dets)
    p_dets.extend(new_dets); p_full_idx.extend(new_gi); p_set.update(new_gi)
    N_p = len(p_dets)

    dE_s1 = (E_P[TARGET_STATE] - e_fci[TARGET_STATE]) * 1000
    print(f"{it:>4} {N_p:>6} {E_P[TARGET_STATE]:>14.8f} {dE_s1:>+10.3f} "
          f"{E_ref:>14.8f} {max_w:>10.3e} {time.perf_counter()-t_it:>8.1f}",
          flush=True)
    it += 1

    for pt in P_CHECKPOINTS:
        if N_p >= pt and pt not in all_results:
            eref_str = 'FCI' if EREFF_FCI else f'H_PP={E_ref:.8f}'
            print(f"\n  ══ Checkpoint P={pt} (E_ref={eref_str}) ══", flush=True)
            all_results[pt] = eval_checkpoint(
                p_dets[:pt], p_full_idx[:pt], H_PP[:pt, :pt], pt, it, E_ref)

# ── Final Summary ──
print(f"\n{'='*70}")
print(f"Phase A Per-State S{TARGET_STATE} Complete ({time.perf_counter()-total_t0:.0f}s)")
print(f"E_ref mode: {'FCI' if EREFF_FCI else 'H_PP'}")
print(f"FCI ref: S0={e_fci[0]:.12f}  S1={e_fci[TARGET_STATE]:.12f}")
print(f"{'='*70}")

print(f"\n{'P':>6} {'N':>6} {'d(m=0)':>7} ", end="")
for m in range(M_MAX + 1):
    print(f"{'dE_S1(m='+str(m)+')':>14} ", end="")
print(f"{'ev[S1](m_last)':>16}")
print("-" * (30 + 32 * (M_MAX + 1)))
for pt in P_CHECKPOINTS:
    r = all_results[pt]
    print(f"{pt:>6} {r['N']:>6} {r['krylov'][0]['d']:>7} ", end="")
    for m in range(M_MAX + 1):
        if m < len(r['krylov']):
            print(f"{r['krylov'][m]['dE_S1_mH']:>+14.1f} ", end="")
        else:
            print(f"{'---':>14} ", end="")
    m_last = min(M_MAX, len(r['krylov']) - 1)
    print(f"{r['krylov'][m_last]['ev_S1']:>16.12f}")

# Save
outdir = os.path.join(PROJECT_ROOT, 'checkpoints_phaseA')
os.makedirs(outdir, exist_ok=True)
fname = f'{outdir}/phaseA_perstate_S{TARGET_STATE}_eref{args.eref}_m{M_MAX}_svd{SVD_THR}_{TAG}.json'
with open(fname, 'w') as f:
    json.dump({
        'config': {'cas': N_ACT, 'n_core': N_CORE, 'P': P_CHECKPOINTS,
                   'm_max': M_MAX, 'svd_threshold': SVD_THR, 'M': M_all,
                   'e_fci': e_fci, 'tag': TAG, 'target_state': TARGET_STATE,
                   'eref_mode': args.eref, 'batch': BATCH},
        'results': {str(k): v for k, v in all_results.items()},
    }, f, indent=2)
print(f"\nSaved: {fname}")
print("Done.")
