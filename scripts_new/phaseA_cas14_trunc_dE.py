#!/usr/bin/env python3
"""
CAS(14,10) SVD Truncation → Energy Error (dE) Analysis

基于 phaseA_cas14_svd_scan.py（Job 15269 成功版）的最小增量扩展：
- Setup / HFPT2 pool / T build / SVD 与 svd_scan 完全一致（逐行复用）
- 新增：SVD 后的 dE 评估段
  * H_PP 用 src_mf.pspace_ops.build_hpp_sigma（C 级 sigma，7/14 优化路径）
  * 对最松阈值下保留的列做一次 sigma pass，得 diag(H_QQ) 与 H_PQ
    （各阈值保留列是嵌套子集，只需一次 pass，按列切片复用）
  * 每个阈值：H_eff = H_PP + H_PQ · diag(1/(E_ref−diag_QQ)) · H_PQ^T → eigh → dE vs FCI
- 修正原 trunc 脚本两处问题：
  * 不再用 O(P²) 纯 Python matrix_element 循环建 H_PP
  * H_PQ 行序用有序 p_flat（与 p_dets/H_PP 对齐），不再迭代 set

注：SVD 的 U 列本身正交归一，截断子集仍正交 → 无需 MGS 再正交化。

Usage:
    python phaseA_cas14_trunc_dE.py --target-p 1600
"""
import sys, os, time, json, argparse, itertools, gc
import numpy as np
from numpy.linalg import svd, eigh

PROJECT_ROOT = '/data/home/wangcx/krylov-dci'
sys.path.insert(0, PROJECT_ROOT)

from src_mf import QSpaceIndex, KDCIBackend
from src_mf.pspace_ops import build_hpp_sigma
from src.determinants import hf_determinant, bit_positions
from src.hamiltonian import Hamiltonian
from pyscf import gto, scf, ao2mo
from pyscf.fci import cistring

# ═══════════════════════════════════════════════════════════════
args = argparse.ArgumentParser()
args.add_argument('--target-p', type=int, default=1600)
args = args.parse_args()

TARGET_P = args.target_p
THRESHOLDS = [1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 3e-2, 5e-2, 1e-1, 2e-1, 5e-1]
N_ACT = 14; N_CORE = 2; R = 1.1; ne = (5, 5); NROOTS = 6

# FCI reference (Job 15258)
E_FCI = np.array([
    -62.410924132579, -62.280783729436, -62.280783729436,
    -62.228221943409, -62.228221943409, -62.225780103936,
])

print("=" * 70)
print(f"CAS({N_ACT},{sum(ne)}) SVD Truncation → dE  P={TARGET_P}  N₂/cc-pVDZ R={R}")
print(f"Thresholds: {THRESHOLDS}")
print("=" * 70, flush=True)

# ═══════════════════════════════════════════════════════════════
# Build system —— 与 phaseA_cas14_svd_scan.py 相同
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
# HFPT2 pool —— 与 phaseA_cas14_svd_scan.py 相同
# ═══════════════════════════════════════════════════════════════
h2_4d = ao2mo.restore('s1', era, N_ACT).reshape(N_ACT, N_ACT, N_ACT, N_ACT)
ham = Hamiltonian(h1=h1a, h2=h2_4d, E_nuc=0.0, E_HF=0.0)
hf_a, hf_b = hf_determinant(*ne)
ao = bit_positions(hf_a); bo = bit_positions(hf_b)
av = [p for p in range(N_ACT) if p not in ao]
bv = [p for p in range(N_ACT) if p not in bo]
E_HF = ham.matrix_element((hf_a, hf_b), (hf_a, hf_b))
det_list = [(int(a), int(b)) for a in as_ for b in bs_]
det_to_idx = {d: i for i, d in enumerate(det_list)}

def gen_hfpt2_scores():
    sc = []
    for i in ao:
        for a in av:
            d = (hf_a ^ (1 << i) | (1 << a), hf_b)
            hij = ham.matrix_element(d, (hf_a, hf_b))
            den = E_HF - ham.matrix_element(d, d)
            if abs(den) > 1e-12: sc.append((d, -hij * hij / den))
    for i in bo:
        for a in bv:
            d = (hf_a, hf_b ^ (1 << i) | (1 << a))
            hij = ham.matrix_element(d, (hf_a, hf_b))
            den = E_HF - ham.matrix_element(d, d)
            if abs(den) > 1e-12: sc.append((d, -hij * hij / den))
    for i1, i2 in itertools.combinations(ao, 2):
        for a1, a2 in itertools.combinations(av, 2):
            d = (hf_a ^ (1 << i1) ^ (1 << i2) | (1 << a1) | (1 << a2), hf_b)
            hij = ham.matrix_element(d, (hf_a, hf_b))
            den = E_HF - ham.matrix_element(d, d)
            if abs(den) > 1e-12: sc.append((d, -hij * hij / den))
    for i1, i2 in itertools.combinations(bo, 2):
        for a1, a2 in itertools.combinations(bv, 2):
            d = (hf_a, hf_b ^ (1 << i1) ^ (1 << i2) | (1 << a1) | (1 << a2))
            hij = ham.matrix_element(d, (hf_a, hf_b))
            den = E_HF - ham.matrix_element(d, d)
            if abs(den) > 1e-12: sc.append((d, -hij * hij / den))
    for i in ao:
        for j in bo:
            for a in av:
                for b in bv:
                    d = (hf_a ^ (1 << i) | (1 << a), hf_b ^ (1 << j) | (1 << b))
                    hij = ham.matrix_element(d, (hf_a, hf_b))
                    den = E_HF - ham.matrix_element(d, d)
                    if abs(den) > 1e-12: sc.append((d, -hij * hij / den))
    sc.sort(key=lambda x: x[1], reverse=True)
    return sc

print("HFPT2 pool...", flush=True)
scores = gen_hfpt2_scores()
pool_dets = [(hf_a, hf_b)]
for d, s in scores:
    if d not in pool_dets:
        pool_dets.append(d)
    if len(pool_dets) >= TARGET_P:
        break
p_dets = pool_dets[:TARGET_P]
print(f"  Pool: {len(p_dets)} dets", flush=True)

# 有序 p_flat：与 p_dets / H_PP 行序严格对齐（修正原 set 迭代隐患）
p_flat = np.array([det_to_idx[(int(pa), int(pb))] for pa, pb in p_dets])
p_idx_set = set(int(x) for x in p_flat)

# ═══════════════════════════════════════════════════════════════
# Build T + SVD —— 与 phaseA_cas14_svd_scan.py record_svd_spectrum 相同
# ═══════════════════════════════════════════════════════════════
tmpdir = f'{PROJECT_ROOT}/tmp'
os.makedirs(tmpdir, exist_ok=True)
E_ref = np.mean([hdiag[q] for q in p_idx_set])
denom = E_ref - hdiag
A_q = np.where(np.abs(denom) > 1e-10, 1.0 / denom, 0.0)
A_half = np.sqrt(np.abs(A_q))

fpath = f'{tmpdir}/cas14_trunc_P{TARGET_P}.dat'
T = np.memmap(fpath, dtype='float64', mode='w+', order='F', shape=(M_all, TARGET_P))
t_b = time.perf_counter()
print(f"\nBuilding T ({M_all:,} × {TARGET_P})...", flush=True)
for col in range(TARGET_P):
    pa, pb = int(p_dets[col][0]), int(p_dets[col][1])
    ia = q_idx._alpha_idx.get(pa); ib = q_idx._beta_idx.get(pb)
    if ia is None or ib is None:
        continue
    ci_unit = np.zeros((na, nb)); ci_unit[ia, ib] = 1.0
    sigma_flat = backend.sigma_full(ci_unit).reshape(-1)
    for q in p_idx_set:
        sigma_flat[q] = 0.0
    T[:, col] = A_half * sigma_flat
    if (col + 1) % max(1, TARGET_P // 10) == 0:
        e = time.perf_counter() - t_b
        print(f"  col {col+1}/{TARGET_P} ({e:.0f}s, ETA {e/(col+1)*TARGET_P-e:.0f}s)", flush=True)
T.flush()
t_build = time.perf_counter() - t_b
print(f"  T built: {t_build:.0f}s ({t_build/TARGET_P:.1f}s/col)", flush=True)

print(f"SVD({M_all}, {TARGET_P})...", flush=True)
t_s = time.perf_counter()
U, sigma, Vt = svd(T, full_matrices=False)
t_svd = time.perf_counter() - t_s
smax = sigma[0]
print(f"  SVD: {t_svd:.0f}s  σ₁={smax:.4f}  σ_min/σ₁={sigma[-1]/smax:.6f}", flush=True)

try:
    del T; gc.collect(); os.unlink(fpath)
except Exception:
    pass

# ═══════════════════════════════════════════════════════════════
# NEW: dE evaluation stage
# ═══════════════════════════════════════════════════════════════
# 1) H_PP via C-level sigma (pspace_ops, 7/14 优化路径)
print(f"\nBuilding H_PP ({TARGET_P}×{TARGET_P}) via build_hpp_sigma...", flush=True)
t_h = time.perf_counter()
H_PP = build_hpp_sigma(p_dets, backend, q_idx._alpha_idx, q_idx._beta_idx, na, nb)
E0 = eigh(H_PP)[0][0]
print(f"  lowest(H_PP) = {eigh(H_PP)[0][0]:.10f} (vs FCI[0]={E_FCI[0]:.10f})",flush=True)
print(f"  H_PP: {time.perf_counter()-t_h:.0f}s", flush=True)
thr_min = min(THRESHOLDS)
r0 = int(np.sum(sigma >= thr_min * smax))
print(f"Sigma pass over r0={r0} kept columns (thr_min={thr_min:.0e})...", flush=True)
diag_qq = np.zeros(r0)
HPQ = np.zeros((TARGET_P, r0))
t_p = time.perf_counter()
for k in range(r0):
    u = np.ascontiguousarray(U[:, k])
    sig = backend.sigma_full(u.reshape(na, nb)).reshape(-1)
    diag_qq[k] = float(np.dot(u, sig))
    HPQ[:, k] = sig[p_flat]
    if (k + 1) % max(1, r0 // 10) == 0:
        e = time.perf_counter() - t_p
        print(f"  col {k+1}/{r0} ({e:.0f}s, ETA {e/(k+1)*r0-e:.0f}s)", flush=True)
print(f"  sigma pass: {time.perf_counter()-t_p:.0f}s", flush=True)

# 3) 每个阈值：切片 → H_eff → eigh → dE
results = []
print(f"\n{'='*60}")
print(f"Testing {len(THRESHOLDS)} thresholds...")
print(f"{'='*60}", flush=True)
for thr in THRESHOLDS:
    r = int(np.sum(sigma >= thr * smax))
    if r == 0:
        results.append({'thr': thr, 'r_svd': 0, 'dE': None})
        print(f"  thr={thr:.0e}: r=0 (all truncated!)")
        continue
    t0_e = time.perf_counter()
    dk = E0 - diag_qq[:r]
    w = np.where(np.abs(dk) > 1e-10, 1.0 / dk, 0.0)
    H_eff = H_PP + (HPQ[:, :r] * w) @ HPQ[:, :r].T
    ev = eigh(H_eff)[0]
    t_e = time.perf_counter() - t0_e

    n_ev = min(NROOTS, len(ev))
    dE = [(ev[i] - E_FCI[i]) * 1000 for i in range(n_ev)]
    dE0 = dE[0]
    max_dE_ex = max(abs(x) for x in dE[1:]) if len(dE) > 1 else 0.0
    compr = (1 - r / TARGET_P) * 100
    print(f"  thr={thr:.0e}: r={r}  compr={compr:.1f}%  dE0={dE0:+.1f} mH  "
          f"max|dE_ex|={max_dE_ex:.1f} mH  ({t_e:.0f}s)", flush=True)
    results.append({
        'thr': thr, 'r_svd': r,
        'E0': float(ev[0]),
        'dE0_mH': float(dE0), 'max_dE_ex_mH': float(max_dE_ex),
        'dE': [float(x) for x in dE],
    })

# ═══════════════════════════════════════════════════════════════
# Summary + save
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print(f"CAS(14,10) SVD Truncation → dE  (P={TARGET_P}, M={M_all:,})")
print(f"FCI: S0={E_FCI[0]:.12f}  S1={E_FCI[1]:.12f}")
print(f"{'='*80}")
print(f"{'thr':>8}  {'r_svd':>6}  {'compr%':>7}  {'dE0/mH':>9}  {'S1/mH':>8}  {'S2/mH':>8}  {'S3/mH':>8}  {'max_ex':>8}")
print("-" * 80)
for r in results:
    dE = r.get('dE') or []
    if not dE:
        print(f"{r['thr']:>8.0e}  {r['r_svd']:>6}  —")
        continue
    dE1 = dE[1] if len(dE) > 1 else float('nan')
    dE2 = dE[2] if len(dE) > 2 else float('nan')
    dE3 = dE[3] if len(dE) > 3 else float('nan')
    compr = (1 - r['r_svd'] / TARGET_P) * 100
    print(f"{r['thr']:>8.0e}  {r['r_svd']:>6}  {compr:>6.1f}%  {r['dE0_mH']:>+9.1f}  "
          f"{dE1:>+8.1f}  {dE2:>+8.1f}  {dE3:>+8.1f}  {r['max_dE_ex_mH']:>+8.1f}")

outdir = os.path.join(PROJECT_ROOT, 'checkpoints_phaseA')
os.makedirs(outdir, exist_ok=True)
fname = f'{outdir}/cas14_truncation_dE_P{TARGET_P}.json'
with open(fname, 'w') as f:
    json.dump({
        'config': {'P': TARGET_P, 'cas': N_ACT, 'M_all': M_all,
                   'thresholds': THRESHOLDS, 'E_FCI': E_FCI.tolist(),
                   'system': 'N2/cc-pVDZ', 'r0_sigma_pass': int(r0)},
        'sigma_full': sigma.tolist(),
        'results': results,
        'timing': {'t_build_s': round(t_build, 1), 't_svd_s': round(t_svd, 1)},
    }, f, indent=2)
print(f"\nSaved: {fname}")
print("Done.")
