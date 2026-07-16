#!/usr/bin/env python3
"""
CAS(14,10) SVD Truncation → Energy Error Analysis (optimized)

Builds T = A²·H_QP once, does full SVD, then tests multiple thresholds
by truncating U and building H_eff for each.

Usage:
    python phaseA_cas14_truncation_error.py --target-p 1600
"""
import sys, os, time, json, argparse, itertools, gc
import numpy as np
from numpy.linalg import svd, eigh

PROJECT_ROOT = '/data/home/wangcx/krylov-dci'
sys.path.insert(0, PROJECT_ROOT)

from src_mf import QSpaceIndex, KDCIBackend
from src.determinants import hf_determinant, bit_positions
from src.hamiltonian import Hamiltonian
from pyscf import gto, scf, ao2mo
from pyscf.fci import cistring

args = argparse.ArgumentParser()
args.add_argument('--target-p', type=int, default=1600)
args = args.parse_args()

TARGET_P = args.target_p
# Dense threshold grid
THRESHOLDS = [1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 3e-2, 5e-2, 1e-1, 2e-1, 5e-1]
N_ACT = 14; N_CORE = 2; R = 1.1; ne = (5, 5); NROOTS = 6

# FCI reference (Job 15258)
E_FCI = np.array([
    -62.410924132579, -62.280783729436, -62.280783729436,
    -62.228221943409, -62.228221943409, -62.225780103936,
])

print("=" * 70)
print(f"CAS(14,10) SVD Truncation → dE  P={TARGET_P}")
print(f"Thresholds: {THRESHOLDS}")
print("=" * 70, flush=True)

# ── Build system ──
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
backend = KDCIBackend(q_idx)
hdiag = np.array([q_idx.hdiag[qi] for qi in range(M_all)])
h2_4d = ao2mo.restore('s1', era, N_ACT).reshape(N_ACT, N_ACT, N_ACT, N_ACT)
ham = Hamiltonian(h1=h1a, h2=h2_4d, E_nuc=0.0, E_HF=0.0)
hf_a, hf_b = hf_determinant(*ne)
det_list = [(int(a), int(b)) for a in as_ for b in bs_]
det_to_idx = {d: i for i, d in enumerate(det_list)}
print(f"  M_all={M_all:,}  setup={time.perf_counter()-t0:.0f}s")

# ── HFPT2 pool ──
ao = bit_positions(hf_a); bo = bit_positions(hf_b)
av = [p for p in range(N_ACT) if p not in ao]
bv = [p for p in range(N_ACT) if p not in bo]
E_HF = ham.matrix_element((hf_a, hf_b), (hf_a, hf_b))

def gen_hfpt2_scores():
    sc = []
    for i in ao:
        for a in av:
            d = (hf_a ^ (1 << i) | (1 << a), hf_b)
            hij = ham.matrix_element(d, (hf_a, hf_b))
            den = E_HF - ham.matrix_element(d, d)
            if abs(den) > 1e-12: sc.append((d, -hij*hij/den))
    for i in bo:
        for a in bv:
            d = (hf_a, hf_b ^ (1 << i) | (1 << a))
            hij = ham.matrix_element(d, (hf_a, hf_b))
            den = E_HF - ham.matrix_element(d, d)
            if abs(den) > 1e-12: sc.append((d, -hij*hij/den))
    for i1, i2 in itertools.combinations(ao, 2):
        for a1, a2 in itertools.combinations(av, 2):
            d = (hf_a ^ (1 << i1) ^ (1 << i2) | (1 << a1) | (1 << a2), hf_b)
            hij = ham.matrix_element(d, (hf_a, hf_b))
            den = E_HF - ham.matrix_element(d, d)
            if abs(den) > 1e-12: sc.append((d, -hij*hij/den))
    for i1, i2 in itertools.combinations(bo, 2):
        for a1, a2 in itertools.combinations(bv, 2):
            d = (hf_a, hf_b ^ (1 << i1) ^ (1 << i2) | (1 << a1) | (1 << a2))
            hij = ham.matrix_element(d, (hf_a, hf_b))
            den = E_HF - ham.matrix_element(d, d)
            if abs(den) > 1e-12: sc.append((d, -hij*hij/den))
    for i in ao:
        for j in bo:
            for a in av:
                for b in bv:
                    d = (hf_a ^ (1 << i) | (1 << a), hf_b ^ (1 << j) | (1 << b))
                    hij = ham.matrix_element(d, (hf_a, hf_b))
                    den = E_HF - ham.matrix_element(d, d)
                    if abs(den) > 1e-12: sc.append((d, -hij*hij/den))
    sc.sort(key=lambda x: x[1], reverse=True)
    return sc

print("HFPT2 pool...", flush=True)
scores = gen_hfpt2_scores()
pool = [(hf_a, hf_b)]
for d, s in scores:
    if d not in pool: pool.append(d)
    if len(pool) >= TARGET_P: break
p_dets = pool[:TARGET_P]
p_idx_set = set(det_to_idx[(int(pa), int(pb))] for pa, pb in p_dets)
print(f"  Pool: {len(p_dets)} dets")

# ── Build T once ──
tmpdir = f'{PROJECT_ROOT}/tmp'; os.makedirs(tmpdir, exist_ok=True)
E_ref = np.mean([hdiag[q] for q in p_idx_set])
denom = E_ref - hdiag
A_half = np.sqrt(np.abs(np.where(np.abs(denom) > 1e-10, 1.0/denom, 0.0)))

fpath = f'{tmpdir}/cas14_trunc_P{TARGET_P}.dat'
T = np.memmap(fpath, dtype='float64', mode='w+', shape=(M_all, TARGET_P))
t_b = time.perf_counter()
print(f"\nBuilding T ({M_all:,} × {TARGET_P})...", flush=True)
for col in range(TARGET_P):
    pa, pb = int(p_dets[col][0]), int(p_dets[col][1])
    ia = q_idx._alpha_idx.get(pa); ib = q_idx._beta_idx.get(pb)
    if ia is None or ib is None: continue
    ci_unit = np.zeros((na, nb)); ci_unit[ia, ib] = 1.0
    sigma_flat = backend.sigma_full(ci_unit).reshape(-1)
    for q in p_idx_set: sigma_flat[q] = 0.0
    T[:, col] = A_half * sigma_flat
    if (col+1) % max(1, TARGET_P//10) == 0:
        e = time.perf_counter()-t_b
        print(f"  col {col+1}/{TARGET_P} ({e:.0f}s, ETA {e/(col+1)*TARGET_P-e:.0f}s)", flush=True)
T.flush()
t_build = time.perf_counter() - t_b
print(f"  T built: {t_build:.0f}s ({t_build/TARGET_P:.1f}s/col)")

# ── Full SVD once ──
print(f"SVD({M_all}, {TARGET_P})...", flush=True)
t_s = time.perf_counter()
U, sigma, Vt = svd(T, full_matrices=False)
t_svd = time.perf_counter() - t_s
smax = sigma[0]
print(f"  SVD: {t_svd:.0f}s  σ₁={smax:.4f}  σ_min/σ₁={sigma[-1]/smax:.6f}")

# ── Clean T ──
try: del T; gc.collect(); os.unlink(fpath)
except: pass

# ── Precompute H_PP ──
print(f"Building H_PP ({TARGET_P}×{TARGET_P})...", flush=True)
H_PP = np.zeros((TARGET_P, TARGET_P))
for i in range(TARGET_P):
    for j in range(i, TARGET_P):
        v = ham.matrix_element(p_dets[i], p_dets[j])
        H_PP[i, j] = v; H_PP[j, i] = v

# ── For each threshold: truncate → build H_eff → dE ──
results = []

def build_heff_and_diag(U_trunc_r):
    """Build H_eff with truncated basis, return eigenvalues."""
    r = U_trunc_r.shape[1]
    if r == 0: return np.array([])

    # MGS orthonormalize
    basis = []
    for k in range(r):
        v = U_trunc_r[:, k].copy()
        for b in basis: v -= np.dot(b, v) * b
        nrm = np.linalg.norm(v)
        if nrm > 1e-10: basis.append(v / nrm)
    d = len(basis)
    if d == 0: return np.array([])
    U_orth = np.column_stack(basis)

    # H_QQ: d×d
    H_QQ = np.zeros((d, d))
    for i in range(d):
        ci_i = U_orth[:, i].reshape(na, nb)
        sig_i = backend.sigma_full(ci_i).reshape(-1)
        for j in range(i, d):
            v = np.dot(U_orth[:, j], sig_i)
            H_QQ[i, j] = v; H_QQ[j, i] = v

    # H_PQ: Np × d
    H_PQ = np.zeros((TARGET_P, d))
    for i in range(d):
        ci_i = U_orth[:, i].reshape(na, nb)
        sig_i = backend.sigma_full(ci_i).reshape(-1)
        for pj, gj in enumerate(p_idx_set):
            H_PQ[pj, i] = sig_i[gj]

    # H_eff = H_PP + H_PQ @ diag(1/(E_ref - H_QQ_diag)) @ H_PQ^T
    diag_QQ = np.diag(H_QQ)
    H_eff = H_PP.copy()
    for k in range(d):
        dk = E_ref - diag_QQ[k]
        if abs(dk) > 1e-10:
            col = H_PQ[:, k]
            H_eff += (1.0 / dk) * np.outer(col, col)

    ev, _ = eigh(H_eff)
    return ev

print(f"\n{'='*60}")
print(f"Testing {len(THRESHOLDS)} thresholds...")
print(f"{'='*60}")

for thr in THRESHOLDS:
    mask = sigma >= thr * smax
    r = int(np.sum(mask))
    if r == 0:
        results.append({'thr': thr, 'r_svd': 0, 'd_mgs': 0, 'dE': None})
        print(f"  thr={thr:.0e}: r=0 (all truncated!)")
        continue

    U_r = U[:, mask]
    t0_e = time.perf_counter()
    ev = build_heff_and_diag(U_r)
    t_e = time.perf_counter() - t0_e

    d = len(ev)
    n_ev = min(NROOTS, d)
    dE = [(ev[i] - E_FCI[i]) * 1000 for i in range(n_ev)] if d > 0 else []

    compr = (1 - d / TARGET_P) * 100 if d > 0 else 100
    dE0 = dE[0] if len(dE) > 0 else float('nan')
    max_dE_ex = max(abs(x) for x in dE[1:]) if len(dE) > 1 else 0

    print(f"  thr={thr:.0e}: SVD {TARGET_P}→{r}  MGS {r}→{d}  "
          f"compr={compr:.1f}%  dE0={dE0:+.1f} mH  "
          f"max|dE_ex|={max_dE_ex:.1f} mH  ({t_e:.0f}s)")

    results.append({
        'thr': thr, 'r_svd': r, 'd_mgs': d,
        'E0': float(ev[0]) if d > 0 else None,
        'dE0_mH': float(dE0), 'max_dE_ex_mH': float(max_dE_ex),
        'dE': [float(x) for x in dE],
    })

# ── Summary ──
print(f"\n{'='*80}")
print(f"CAS(14,10) SVD Truncation → dE  (P={TARGET_P}, M={M_all:,})")
print(f"FCI: S0={E_FCI[0]:.12f}  S1={E_FCI[1]:.12f}")
print(f"{'='*80}")
hdr = f"{'thr':>8}  {'r_svd':>6}  {'d':>6}  {'compr%':>7}  {'dE0/mH':>9}  {'S1/mH':>8}  {'S2/mH':>8}  {'S3/mH':>8}  {'max_ex':>8}"
print(hdr)
print("-" * 90)
for r in results:
    thr = r['thr']; compr = (1 - r['d_mgs']/TARGET_P)*100
    dE = r.get('dE') or []
    dE1 = dE[1] if len(dE) > 1 else float('nan')
    dE2 = dE[2] if len(dE) > 2 else float('nan')
    dE3 = dE[3] if len(dE) > 3 else float('nan')
    print(f"{thr:>8.0e}  {r['r_svd']:>6}  {r['d_mgs']:>6}  {compr:>6.1f}%  "
          f"{r['dE0_mH']:>+9.1f}  {dE1:>+8.1f}  {dE2:>+8.1f}  {dE3:>+8.1f}  "
          f"{r['max_dE_ex_mH']:>+8.1f}")

# Save
outdir = os.path.join(PROJECT_ROOT, 'checkpoints_phaseA')
os.makedirs(outdir, exist_ok=True)
fname = f'{outdir}/cas14_truncation_dE_P{TARGET_P}.json'
with open(fname, 'w') as f:
    json.dump({
        'config': {'P': TARGET_P, 'cas': N_ACT, 'M_all': M_all,
                   'thresholds': THRESHOLDS,
                   'E_FCI': E_FCI.tolist()},
        'sigma_full': sigma.tolist(),
        'results': results,
    }, f, indent=2)
print(f"\nSaved: {fname}")
print("Done.")
