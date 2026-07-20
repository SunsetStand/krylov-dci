#!/usr/bin/env python3
"""
CAS(14,10) SVD Spectrum P-Scan — 探索大 P 下 SVD 奇异值是否出现衰减

核心问题：P/M_all 比例增大后，build_basis T=A²·H_QP 的 SVD 谱是否从平坦变为可截断？

实验设计：
- N₂/cc-pVDZ, CAS(14,10), M = 4,008,004
- P checkpoints: 200, 400, 800, 1600, 3200
- 每个 checkpoint: build_basis → 记录完整 SVD 奇异值谱
- HFPT2 排序生成 P 空间（避免 FCI reference 的昂贵计算）

Usage:
    python phaseA_cas14_svd_scan.py --P 200,400,800,1600,3200
    sbatch phaseA_cas14_svd_scan.slurm
"""
import sys, os, time, json, argparse, itertools, gc
import numpy as np
from numpy.linalg import svd

PROJECT_ROOT = '/data/home/wangcx/krylov-dci'
sys.path.insert(0, PROJECT_ROOT)

from src_mf import QSpaceIndex, KDCIBackend
from src.determinants import hf_determinant, bit_positions
from src.hamiltonian import Hamiltonian
from pyscf import gto, scf, ao2mo
from pyscf.fci import cistring

# ═══════════════════════════════════════════════════════════════
args = argparse.ArgumentParser()
args.add_argument('--P', type=str, default='200,400,800,1600,3200',
                  help='P checkpoints (comma-separated)')
args.add_argument('--svd-threshold', type=float, default=1e-3,
                  help='SVD truncation threshold for stats')
args = args.parse_args()

P_CHECKPOINTS = sorted([int(x) for x in args.P.split(',')])
SVD_THR = args.svd_threshold; P_MAX = max(P_CHECKPOINTS)
N_ACT = 14; N_CORE = 2; R = 1.1; ne = (5, 5)

print("=" * 70)
print(f"CAS({N_ACT},{sum(ne)}) SVD Spectrum P-Scan — N₂/cc-pVDZ R={R}")
print(f"P checkpoints: {P_CHECKPOINTS}")
print(f"SVD threshold for stats: {SVD_THR}")
print(f"M = C({N_ACT},5)² = {2002**2:,}")
print("=" * 70, flush=True)

# ═══════════════════════════════════════════════════════════════
# Build system
# ═══════════════════════════════════════════════════════════════
t0 = time.perf_counter()
mol = gto.M(atom=f'N 0 0 0; N 0 0 {R}', basis='cc-pVDZ', verbose=0)
mf = scf.RHF(mol).run(verbose=0)
na_o = list(range(N_CORE, N_CORE + N_ACT))
norb = mf.mo_coeff.shape[1]
print(f"  Total MOs: {norb}, active: {N_ACT} (orbitals {N_CORE}-{N_CORE+N_ACT-1})")

h1_mo = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
eri_4d = ao2mo.full(mol.intor('int2e'), mf.mo_coeff, compact=False)
eri_4d = eri_4d.reshape(norb, norb, norb, norb)
h1a = h1_mo[np.ix_(na_o, na_o)]
era = eri_4d[np.ix_(na_o, na_o, na_o, na_o)]

as_ = cistring.gen_strings4orblist(range(N_ACT), ne[0])
bs_ = cistring.gen_strings4orblist(range(N_ACT), ne[1])
na, nb = len(as_), len(bs_)
M_all = na * nb
print(f"  α-strings: {na}, β-strings: {nb}, M_all = {M_all:,}")

q_idx = QSpaceIndex(as_, bs_, N_ACT, ne, h1a, era)
backend = KDCIBackend(q_idx)
hdiag = np.array([q_idx.hdiag[qi] for qi in range(M_all)])
print(f"  Setup: {time.perf_counter() - t0:.0f}s\n")

# ═══════════════════════════════════════════════════════════════
# HFPT2 scoring — generate sorted determinant pool
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
print(f"  E_HF = {E_HF:.8f} Ha")

def gen_hfpt2_scores():
    """Generate HFPT2 scores: |H_{det,HF}|² / (E_HF - H_{det,det})."""
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
    # Doubles
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

print("  Generating HFPT2 scores...", flush=True)
t_pt = time.perf_counter()
scores = gen_hfpt2_scores()
print(f"  {len(scores)} HFPT2 candidates ({time.perf_counter() - t_pt:.0f}s)")

# Build P-space pool: HF det + top HFPT2
pool_dets = [(hf_a, hf_b)]
for d, s in scores:
    if d not in pool_dets:
        pool_dets.append(d)
    if len(pool_dets) >= P_MAX:
        break
print(f"  P-space pool: {len(pool_dets)} determinants\n")

# ═══════════════════════════════════════════════════════════════
# SVD spectrum recorder
# ═══════════════════════════════════════════════════════════════
tmpdir = f'{PROJECT_ROOT}/tmp'
os.makedirs(tmpdir, exist_ok=True)

def record_svd_spectrum(p_dets, tag):
    """build_basis T=A²·H_QP → SVD → return full spectrum, timing, stats."""
    N = len(p_dets)
    p_idx_set = set()
    for pa, pb in p_dets:
        idx = det_to_idx.get((int(pa), int(pb)))
        if idx is not None and idx >= 0:
            p_idx_set.add(idx)

    # Energy denominator: use average H_qq as E_ref
    E_ref = np.mean([hdiag[q] for q in p_idx_set])

    denom = E_ref - hdiag
    A_q = np.where(np.abs(denom) > 1e-10, 1.0 / denom, 0.0)
    A_half = np.sqrt(np.abs(A_q))

    fpath = f'{tmpdir}/cas14_svd_P{N}_{tag}.dat'
    T = np.memmap(fpath, dtype='float64', mode='w+', shape=(M_all, N))

    t_build = time.perf_counter()
    print(f"    Building T ({M_all:,} × {N})...", flush=True)
    for col in range(N):
        pa, pb = int(p_dets[col][0]), int(p_dets[col][1])
        ia = q_idx._alpha_idx.get(pa)
        ib = q_idx._beta_idx.get(pb)
        if ia is None or ib is None:
            continue
        ci_unit = np.zeros((na, nb))
        ci_unit[ia, ib] = 1.0
        sigma_flat = backend.sigma_full(ci_unit).reshape(-1)
        for q in p_idx_set:
            sigma_flat[q] = 0.0
        T[:, col] = A_q * sigma_flat  # T = A*H_QP (A1)

        if (col + 1) % max(1, N // 10) == 0:
            elapsed = time.perf_counter() - t_build
            eta = elapsed / (col + 1) * N - elapsed
            print(f"      col {col+1}/{N}  ({elapsed:.0f}s, ETA {eta:.0f}s)", flush=True)
    T.flush()
    t_build = time.perf_counter() - t_build
    print(f"    T built: {t_build:.0f}s  ({t_build/N:.1f}s/col)", flush=True)

    # SVD
    t_svd = time.perf_counter()
    print(f"    SVD({M_all:,}, {N})...", flush=True)
    U, sigma, Vt = svd(T, full_matrices=False)
    t_svd = time.perf_counter() - t_svd
    print(f"    SVD done: {t_svd:.0f}s", flush=True)

    # Stats
    smax = sigma[0] if len(sigma) > 0 else 0
    rel_sigma = sigma / max(smax, 1e-16)
    n_kept = int(np.sum(rel_sigma >= SVD_THR))

    # Log first/last few singular values
    n_show = min(10, len(sigma))
    head = ", ".join(f"{sigma[i]/smax:.6f}" for i in range(n_show))
    tail = ", ".join(f"{sigma[-n_show+i]/smax:.6f}" for i in range(n_show)) if N > n_show else ""

    print(f"    σ/σ₁  first {n_show}: [{head}]")
    if tail:
        print(f"    σ/σ₁  last  {n_show}: [{tail}]")
    print(f"    kept @ thr={SVD_THR}: {n_kept}/{N} ({100*n_kept/N:.1f}%)")

    # Cleanup
    try:
        del T; gc.collect(); os.unlink(fpath)
    except:
        pass

    return {
        'P': N,
        'sigma': sigma.tolist(),
        'sigma_rel': rel_sigma.tolist(),
        'n_kept': n_kept,
        't_build_s': round(t_build, 1),
        't_svd_s': round(t_svd, 1),
    }


# ═══════════════════════════════════════════════════════════════
# Main: scan P checkpoints
# ═══════════════════════════════════════════════════════════════
results = {}
total_t0 = time.perf_counter()

for P in P_CHECKPOINTS:
    print(f"\n{'─'*60}")
    print(f"  Checkpoint P={P}  (P/M_all = {P/M_all*100:.4f}%)")
    print(f"{'─'*60}", flush=True)
    p_dets = pool_dets[:P]
    results[f"P{P}"] = record_svd_spectrum(p_dets, f"p{P}")

# ═══════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"Summary: CAS(14,10) SVD Spectrum vs P")
print(f"M_all = {M_all:,}  (C({N_ACT},5)²)")
print(f"{'='*70}")
print(f"{'P':>6}  {'P/M%':>8}  {'σ₁':>14}  {'σ_min/σ₁':>10}  {'σ_N/σ₁':>10}  {'kept%':>7}  {'t_build':>8}  {'t_svd':>8}")
print("-" * 85)
for P in P_CHECKPOINTS:
    r = results[f"P{P}"]
    sig = np.array(r['sigma'])
    s1 = sig[0]
    ratio_end = sig[-1] / s1 if len(sig) > 0 else 0
    # MP prediction: σ_min/σ_max ≈ (1-√γ)/(1+√γ), γ = P/M_all
    gamma = P / M_all
    mp_pred = (1 - np.sqrt(gamma)) / (1 + np.sqrt(gamma)) if gamma < 1 else 0
    print(f"{P:>6}  {gamma*100:>8.4f}  {s1:>14.6f}  {ratio_end:>10.6f}  "
          f"{ratio_end:>10.6f}  {r['n_kept']/P*100:>6.1f}%  "
          f"{r['t_build_s']:>7.0f}s  {r['t_svd_s']:>6.0f}s  "
          f"  MP pred σ_min/σ₁≈{mp_pred:.6f}")

# ═══════════════════════════════════════════════════════════════
# Save results
# ═══════════════════════════════════════════════════════════════
outdir = os.path.join(PROJECT_ROOT, 'checkpoints_phaseA')
os.makedirs(outdir, exist_ok=True)
fname = f'{outdir}/cas14_svd_spectrum_P{P_MAX}.json'
with open(fname, 'w') as f:
    json.dump({
        'config': {
            'cas': N_ACT, 'n_core': N_CORE, 'ne': list(ne),
            'M_all': M_all, 'P_checkpoints': P_CHECKPOINTS,
            'svd_threshold': SVD_THR, 'system': 'N2/cc-pVDZ',
        },
        'results': results,
    }, f, indent=2)
print(f"\nSaved: {fname}")
print(f"Total wall time: {time.perf_counter() - total_t0:.0f}s")
print("Done.")
