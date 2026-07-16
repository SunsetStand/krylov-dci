#!/usr/bin/env python3
"""
CAS(14,10) SVD Truncation → Energy Error (dE) Analysis — v5
= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

This script follows the CAS(10,10)-SVD pipeline from phaseA_cas10_svd.py VERBATIM.
The only differences are CAS(14,10) instead of CAS(10,10) and a fixed P (no iterative
expansion). After the standard build_basis_mf → build_blocks → perstate H_eff,
we add a per-SVD-threshold dE sweep: at each threshold r we slice the SVD basis U,
slice H_QQ/H_PQ accordingly, and compute per-state H_eff at the same E_refs.
"""
import sys, os, time, json, argparse, itertools, gc
import numpy as np
from numpy.linalg import svd, eigh

PROJECT_ROOT = '/data/home/wangcx/krylov-dci'
sys.path.insert(0, PROJECT_ROOT)

from src_mf import QSpaceIndex, KDCIBackend
from src_mf.pspace_ops import build_hpp_sigma
from src.effective_h import build_effective_H, diagonalize_effective_H
from src.determinants import hf_determinant, bit_positions
from src.hamiltonian import Hamiltonian
from pyscf import gto, scf, ao2mo
from pyscf.fci import cistring

# ═══════════════════════════════════════════════════════════════
args = argparse.ArgumentParser()
args.add_argument('--target-p', type=int, default=1600)
args = args.parse_args()

TARGET_P = args.target_p
SVD_THR = 1e-3  # for build_basis_mf truncation
THRESHOLDS = [1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 3e-2, 5e-2, 1e-1, 2e-1, 5e-1]
N_ACT = 14; N_CORE = 2; R = 1.1; ne = (5, 5); NROOTS = 6

E_FCI = np.array([  # Job 15258
    -62.410924132579, -62.280783729436, -62.280783729436,
    -62.228221943409, -62.228221943409, -62.225780103936,
])

print("=" * 70)
print(f"CAS({N_ACT},{sum(ne)}) SVD Truncation → dE  P={TARGET_P}  N₂/cc-pVDZ R={R}")
print(f"Pipeline: build_hpp_sigma → build_basis_mf(T=A²·H_QP) → SVD → blocks → per-state H_eff")
print(f"{'='*70}", flush=True)

# ═══════════════════════════════════════════════════════════════
# 1. System setup (same as phaseA_cas14_svd_scan.py)
# ═══════════════════════════════════════════════════════════════
t0 = time.perf_counter()
mol = gto.M(atom=f'N 0 0 0; N 0 0 {R}', basis='cc-pVDZ', verbose=0)
mf = scf.RHF(mol).run(verbose=0)
na_o = list(range(N_CORE, N_CORE + N_ACT))
norb = mf.mo_coeff.shape[1]
h1_mo = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
eri_4d = ao2mo.full(mol.intor('int2e'), mf.mo_coeff, compact=False)
eri_4d = eri_4d.reshape(norb, norb, norb, norb)
h1a = h1_mo[np.ix_(na_o, na_o)]
era = eri_4d[np.ix_(na_o, na_o, na_o, na_o)]

as_ = cistring.gen_strings4orblist(range(N_ACT), ne[0])
bs_ = cistring.gen_strings4orblist(range(N_ACT), ne[1])
na, nb = len(as_), len(bs_)
M_all = na * nb
q_idx = QSpaceIndex(as_, bs_, N_ACT, ne, h1a, era)
backend = KDCIBackend(q_idx)
hdiag = np.array([q_idx.hdiag[qi] for qi in range(M_all)])
print(f"  M_all = {M_all:,}  setup = {time.perf_counter()-t0:.0f}s", flush=True)

# ═══════════════════════════════════════════════════════════════
# 2. HFPT2 pool (same as phaseA_cas14_svd_scan.py)
# ═══════════════════════════════════════════════════════════════
h2_4d = ao2mo.restore('s1', era, N_ACT).reshape(N_ACT, N_ACT, N_ACT, N_ACT)
ham = Hamiltonian(h1=h1a, h2=h2_4d, E_nuc=0.0, E_HF=0.0)
hf_a, hf_b = hf_determinant(*ne)
E_HF = ham.matrix_element((hf_a, hf_b), (hf_a, hf_b))
det_list = [(int(a), int(b)) for a in as_ for b in bs_]

ao_b = bit_positions(hf_a); bo_b = bit_positions(hf_b)
av = [p for p in range(N_ACT) if p not in ao_b]
bv = [p for p in range(N_ACT) if p not in bo_b]
def gen_hfpt2_scores():
    sc = []
    for i in ao_b:
        for a in av:
            d = (hf_a ^ (1 << i) | (1 << a), hf_b)
            hij = ham.matrix_element(d, (hf_a, hf_b))
            den = E_HF - ham.matrix_element(d, d)
            if abs(den) > 1e-12: sc.append((d, -hij*hij/den))
    for i in bo_b:
        for a in bv:
            d = (hf_a, hf_b ^ (1 << i) | (1 << a))
            hij = ham.matrix_element(d, (hf_a, hf_b))
            den = E_HF - ham.matrix_element(d, d)
            if abs(den) > 1e-12: sc.append((d, -hij*hij/den))
    for i1, i2 in itertools.combinations(ao_b, 2):
        for a1, a2 in itertools.combinations(av, 2):
            d = (hf_a ^ (1 << i1) ^ (1 << i2) | (1 << a1) | (1 << a2), hf_b)
            hij = ham.matrix_element(d, (hf_a, hf_b))
            den = E_HF - ham.matrix_element(d, d)
            if abs(den) > 1e-12: sc.append((d, -hij*hij/den))
    for i1, i2 in itertools.combinations(bo_b, 2):
        for a1, a2 in itertools.combinations(bv, 2):
            d = (hf_a, hf_b ^ (1 << i1) ^ (1 << i2) | (1 << a1) | (1 << a2))
            hij = ham.matrix_element(d, (hf_a, hf_b))
            den = E_HF - ham.matrix_element(d, d)
            if abs(den) > 1e-12: sc.append((d, -hij*hij/den))
    for i in ao_b:
        for j in bo_b:
            for a in av:
                for b in bv:
                    d = (hf_a ^ (1 << i) | (1 << a), hf_b ^ (1 << j) | (1 << b))
                    hij = ham.matrix_element(d, (hf_a, hf_b))
                    den = E_HF - ham.matrix_element(d, d)
                    if abs(den) > 1e-12: sc.append((d, -hij*hij/den))
    sc.sort(key=lambda x: x[1], reverse=True)
    return sc

print("HFPT2 pool...", flush=True)
sc = gen_hfpt2_scores()
pool_dets = [(hf_a, hf_b)]
for d, s in sc:
    if d not in pool_dets: pool_dets.append(d)
    if len(pool_dets) >= TARGET_P: break
p_dets = pool_dets[:TARGET_P]
print(f"  Pool: {len(p_dets)} dets", flush=True)

# ═══════════════════════════════════════════════════════════════
# 3. H_PP first, then E0 = lowest eigenvalue → feeds build_basis_mf
#    (Matching CAS10-SVD pipeline order exactly)
# ═══════════════════════════════════════════════════════════════
print(f"\nBuilding H_PP ({TARGET_P}×{TARGET_P}) via build_hpp_sigma...", flush=True)
t_h = time.perf_counter()
H_PP = build_hpp_sigma(p_dets, backend, q_idx._alpha_idx, q_idx._beta_idx, na, nb)
t_hpp = time.perf_counter() - t_h
print(f"  H_PP: {t_hpp:.0f}s", flush=True)

E_vals_HPP, E_vecs_HPP = eigh(H_PP)
E0 = float(E_vals_HPP[0])
E_refs = E_vals_HPP[:NROOTS]  # per-state references for Bloch resolvent
nlab = min(NROOTS, TARGET_P)
print(f"  lowest(H_PP) = {E0:.10f} (vs FCI[0]={E_FCI[0]:.10f}, dE={E0-E_FCI[0]:+.6f} Ha)")
for k in range(nlab):
    print(f"    S{k}: E={E_refs[k]:.10f}  dE_bare={(E_refs[k]-E_FCI[k])*1000:+.1f} mH")

# ═══════════════════════════════════════════════════════════════
# 4. build_basis_mf: T = A² · H_QP → SVD → U
#    (copied VERBATIM from phaseA_cas10_svd.py build_basis_mf)
# ═══════════════════════════════════════════════════════════════
print(f"\n[build_basis] T=A²·H_QP, E0={E0:.10f}...", flush=True)
A_q = np.where(np.abs(E0 - hdiag) > 1e-10, 1.0 / (E0 - hdiag), 0.0)
A_half = np.sqrt(np.abs(A_q))

tmpdir = f'{PROJECT_ROOT}/tmp'; os.makedirs(tmpdir, exist_ok=True)
fpath = f'{tmpdir}/cas14_svd_trunc_P{TARGET_P}.dat'
T = np.memmap(fpath, dtype='float64', mode='w+', shape=(M_all, TARGET_P), order='F')

p_idx_set = set()
for pa, pb in p_dets:
    idx = q_idx.flat_index(int(pa), int(pb))
    if idx is not None and idx >= 0: p_idx_set.add(idx)

t_b = time.perf_counter()
for p in range(TARGET_P):
    pa, pb = int(p_dets[p][0]), int(p_dets[p][1])
    ia = q_idx._alpha_idx.get(pa); ib = q_idx._beta_idx.get(pb)
    if ia is None or ib is None: continue
    ci_unit = np.zeros((na, nb)); ci_unit[ia, ib] = 1.0
    sigma_flat = backend.sigma_full(ci_unit).reshape(-1)
    for q in p_idx_set: sigma_flat[q] = 0.0
    T[:, p] = A_half * sigma_flat
    if (p+1) % max(1, TARGET_P//10) == 0:
        e = time.perf_counter() - t_b
        print(f"  col {p+1}/{TARGET_P} ({e:.0f}s, ETA {e/(p+1)*TARGET_P-e:.0f}s)", flush=True)
T.flush()
t_build = time.perf_counter() - t_b
print(f"  T built: {t_build:.0f}s ({t_build/TARGET_P:.1f}s/col)", flush=True)

# SVD (keeping ALL singular vectors for later threshold sweep)
print(f"  SVD({M_all},{TARGET_P}) full...", flush=True)
t_s = time.perf_counter()
U_full, sigma, Vt = svd(T, full_matrices=False)
t_svd = time.perf_counter() - t_s
smax = sigma[0]
print(f"  SVD done: {t_svd:.0f}s  σ₁={smax:.4f}  σ_min/σ₁={sigma[-1]/smax:.6f}", flush=True)
try: del T; gc.collect(); os.unlink(fpath)
except: pass

# SVD-truncate at default threshold for the non-truncated reference
d0 = int(np.sum(sigma >= SVD_THR * max(1.0, smax)))
U_0 = np.ascontiguousarray(U_full[:, :d0])
print(f"  SVD_trunc{d0}  kept {d0}/{TARGET_P} @ thr={SVD_THR}", flush=True)

# ═══════════════════════════════════════════════════════════════
# 5. build_blocks: H_KK (d×d) + H_PK (P×d)
#    (copied VERBATIM from phaseA_cas10_svd.py build_blocks)
# ═══════════════════════════════════════════════════════════════
def build_blocks(U_basis, Np_trunc):
    """H_KK = U_basis^T * H * U_basis,  H_PK rows = P-space slice of H*U.
    Returns full (d,d) H_KK and (Np,d) H_PK (all P rows, not just valid)."""
    d = U_basis.shape[1]; Np = min(Np_trunc, TARGET_P)
    if d == 0: return np.zeros((0,0)), np.zeros((Np,0))
    print(f"    [blocks] d={d}...", flush=True)
    t0_b = time.perf_counter()
    H_KK = np.zeros((d, d))
    H_PK = np.zeros((Np, d))
    p_flat = np.array([q_idx.flat_index(int(pa), int(pb)) for pa, pb in p_dets[:Np]])
    p_valid = p_flat >= 0; p_f = p_flat[p_valid]
    for k in range(d):
        ci_k = U_basis[:, k].reshape(na, nb)
        sk = backend.sigma_full(ci_k).reshape(-1)
        H_KK[:, k] = U_basis.T @ sk
        H_PK[p_valid, k] = sk[p_f]
        if (k+1) % max(1, d//5) == 0:
            print(f"      {k+1}/{d} ({time.perf_counter()-t0_b:.0f}s)", flush=True)
    H_KK = 0.5 * (H_KK + H_KK.T)
    print(f"    [blocks] done: {time.perf_counter()-t0_b:.0f}s", flush=True)
    return H_KK, H_PK

H_KK_0, H_PK_0 = build_blocks(U_0, TARGET_P)
d0_actual = H_KK_0.shape[0]

# Per-state H_eff at default SVD threshold (the reference)
print(f"\nPer-state H_eff (d={d0_actual})...", flush=True)
ev_ref = np.zeros(min(NROOTS, TARGET_P))
for k in range(len(ev_ref)):
    H_eff_k = build_effective_H(H_PP, H_PK_0, H_KK_0, float(E_refs[k]), delta=0.0)
    evk = np.asarray(diagonalize_effective_H(H_eff_k, n_states=NROOTS)[0])
    ev_ref[k] = evk[int(np.argmin(np.abs(evk - E_refs[k])))]
dE0_ref = (ev_ref[0] - E_FCI[0]) * 1000
print(f"  dE0(d={d0_actual}) = {dE0_ref:+.1f} mH", flush=True)
for k in range(1, len(ev_ref)):
    print(f"    S{k}: dE={(ev_ref[k]-E_FCI[k])*1000:+.1f} mH", flush=True)

# ═══════════════════════════════════════════════════════════════
# 6. Truncation-dE sweep ← THIS IS THE ONLY NEW PIECE BEYOND CAS10-SVD
# ═══════════════════════════════════════════════════════════════
# Pre-compute sigma vectors for the full U_0 basis (needed for H_QQ at each r)
n_cols = d0_actual
sigs_full = [None] * n_cols
print(f"\nSigma pass over {n_cols} SVD columns (for truncation sweep)...", flush=True)
t_sp = time.perf_counter()
for k in range(n_cols):
    sk = backend.sigma_full(U_0[:, k].reshape(na, nb)).reshape(-1)
    sigs_full[k] = sk
    if (k+1) % max(1, n_cols//10) == 0:
        e = time.perf_counter() - t_sp
        print(f"  col {k+1}/{n_cols} ({e:.0f}s, ETA {e/(k+1)*n_cols-e:.0f}s)", flush=True)
t_sigma_pass = time.perf_counter() - t_sp
print(f"  sigma pass done: {t_sigma_pass:.0f}s", flush=True)

# Per-threshold sweep: slice U, build H_KK/H_PK from cached sigma vectors, per-state H_eff
results = []
print(f"\n{'='*60}")
print(f"Testing {len(THRESHOLDS)} SVD thresholds (per-state H_eff, full H_KK)...")
print(f"{'='*60}", flush=True)
for thr in THRESHOLDS:
    r = int(np.sum(sigma >= thr * smax))
    if r == 0:
        results.append({'thr': thr, 'r_svd': 0, 'dE': None, 'dE0': None, 'max_ex': None})
        print(f"  thr={thr:.0e}: r=0 (all truncated!)")
        continue
    if r > n_cols: r = n_cols
    t0_e = time.perf_counter()

    # Build H_KK (r×r) from cached sigma vectors
    U_r = U_0[:, :r]
    H_KK_r = np.zeros((r, r))
    p_flat = np.array([q_idx.flat_index(int(pa), int(pb)) for pa, pb in p_dets])
    p_valid = p_flat >= 0; p_f = p_flat[p_valid]
    H_PK_r = np.zeros((TARGET_P, r))
    for k in range(r):
        sk = sigs_full[k]
        H_KK_r[:, k] = U_r.T @ sk
        H_PK_r[p_valid, k] = sk[p_f]
    H_KK_r = 0.5 * (H_KK_r + H_KK_r.T)

    # Per-state H_eff
    n_st = min(NROOTS, TARGET_P)
    ev = np.zeros(n_st)
    for k in range(n_st):
        H_eff_k = build_effective_H(H_PP, H_PK_r, H_KK_r, float(E_refs[k]), delta=0.0)
        evk = np.asarray(diagonalize_effective_H(H_eff_k, n_states=NROOTS)[0])
        ev[k] = evk[int(np.argmin(np.abs(evk - E_refs[k])))]

    t_e = time.perf_counter() - t0_e
    dE = [(ev[i] - E_FCI[i]) * 1000 for i in range(n_st)]
    dE0 = dE[0]; max_ex = max(abs(x) for x in dE[1:]) if n_st > 1 else 0.0
    compr = (1 - r / TARGET_P) * 100
    print(f"  thr={thr:.0e}: r={r}  compr={compr:.1f}%  dE0={dE0:+.1f} mH  "
          f"max|dE_ex|={max_ex:.1f} mH  ({t_e:.0f}s)", flush=True)
    results.append({'thr': thr, 'r_svd': r, 'compr': compr,
                    'dE0_mH': float(dE0), 'max_dE_ex_mH': float(max_ex),
                    'dE': [float(x) for x in dE]})

# ═══════════════════════════════════════════════════════════════
# 7. Summary
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print(f"CAS(14,10) SVD Truncation → dE  (P={TARGET_P}, M={M_all:,})")
print(f"Reference d={d0_actual} (SVD thr={SVD_THR}): dE0={dE0_ref:+.1f} mH")
print(f"FCI: S0={E_FCI[0]:.12f}  S1={E_FCI[1]:.12f}")
print(f"{'='*80}")
hdr = f"{'thr':>8}  {'r_svd':>6}  {'compr%':>7}  {'dE0/mH':>9}  {'S1/mH':>8}  {'S2/mH':>8}  {'S3/mH':>8}  {'max_ex':>8}"
print(hdr); print("-" * 80)
for r in results:
    dE = r.get('dE')
    if dE is None:
        print(f"{r['thr']:>8.0e}  {r['r_svd']:>6}  —")
        continue
    dE1 = dE[1] if len(dE) > 1 else float('nan')
    dE2 = dE[2] if len(dE) > 2 else float('nan')
    dE3 = dE[3] if len(dE) > 3 else float('nan')
    print(f"{r['thr']:>8.0e}  {r['r_svd']:>6}  {r['compr']:>6.1f}%  {r['dE0_mH']:>+9.1f}  "
          f"{dE1:>+8.1f}  {dE2:>+8.1f}  {dE3:>+8.1f}  {r['max_dE_ex_mH']:>+8.1f}")

# Save
outdir = os.path.join(PROJECT_ROOT, 'checkpoints_phaseA')
os.makedirs(outdir, exist_ok=True)
fname = f'{outdir}/cas14_truncation_dE_P{TARGET_P}.json'
with open(fname, 'w') as f:
    json.dump({
        'config': {'P': TARGET_P, 'cas': N_ACT, 'M_all': M_all,
                   'thresholds': THRESHOLDS, 'svd_thr': SVD_THR,
                   'E_FCI': E_FCI.tolist(), 'd0': d0_actual,
                   'dE0_ref_mH': float(dE0_ref)},
        'sigma_full': sigma.tolist(),
        'results': results,
        'timing': {'t_build_s': round(t_build, 1), 't_svd_s': round(t_svd, 1),
                   't_sigma_pass_s': round(t_sigma_pass, 1)},
    }, f, indent=2)
print(f"\nSaved: {fname}")
print("Done.")
