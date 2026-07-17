#!/usr/bin/env python3
"""
Truncation sweep at P=800, m=1 with CIS+state-average P-space.
Minimal modification of phaseA_cas10_trunc_dE.py:
  - M_MAX=1, TARGET_P=800, CIS seed, state-average scoring
  - Truncation sweep on FULL m=1 Krylov basis (not just m=0)
"""
import sys, os, time, json, itertools, gc
import numpy as np
from numpy.linalg import eigh, svd, norm

PROJECT_ROOT = '/data/home/wangcx/krylov-dci'
sys.path.insert(0, PROJECT_ROOT)

from src_mf import QSpaceIndex, KDCIBackend, KDCISparse
from src_mf.pspace_ops import embed_pspace_vec, build_pmask, score_and_select, build_hpp_sigma, extend_hpp_sigma
from src.effective_h import build_effective_H, diagonalize_effective_H
from src.determinants import hf_determinant, bit_positions
from src.hamiltonian import Hamiltonian
from pyscf import gto, scf, ao2mo
from pyscf.fci import cistring, direct_spin1, spin_op

TARGET_P = 800
M_MAX = 1
SVD_THR = 1e-3
BATCH = 200
TAG = 'cas10_trunc_m1_cis'
P_CHECKPOINTS = [TARGET_P]
P_MAX = TARGET_P
N_ACT = 10; N_CORE = 2; NROOTS = 6; R = 1.1; ne = (5, 5)

t0 = time.perf_counter()
print("=" * 70)
print(f"CIS-Seeded Truncation Sweep: P={TARGET_P}, m_max={M_MAX}, CAS(10,10)")
print("=" * 70, flush=True)

# ── PySCF setup ──
mol = gto.M(atom=f'N 0 0 0; N 0 0 {R}', basis='cc-pVDZ', verbose=0, spin=0)
mf = scf.RHF(mol).run(verbose=0)
ncore = N_CORE; ncas = N_ACT; ne_cas = sum(ne)
cas_list = list(range(ncore, ncore + ncas))
mc = mf.CASSCF(ncas, ne_cas)
mc.fix_spin_(ss=0)
mo_coeff = mc.sort_mo(cas_list, base=0)

h1 = mo_coeff.T @ mf.get_hcore() @ mo_coeff; h1 = h1[ncore:ncore+ncas, ncore:ncore+ncas]
era = ao2mo.kernel(mol, mo_coeff[:, ncore:ncore+ncas], aosym='s4')
h1a = h1.copy(); h1b = h1.copy()
backend = KDCIBackend(h1a, h1b, era, N_ACT, ne, verbose=0)
kdci_sparse = KDCISparse(h1a, h1b, era, N_ACT, ne)
q_idx = kdci_sparse.q_idx
na, nb = backend.na, backend.nb
aidx = {int(a): i for i, a in enumerate(backend.as_)}; bidx = {int(b): i for i, b in enumerate(backend.bs_)}
as_ = backend.as_; bs_ = backend.bs_
hdiag = kdci_sparse.hdiag; M_all = len(as_) * len(bs_)

print(f"  CAS({N_ACT},{ne_cas}): M={M_all:,}  ({time.perf_counter()-t0:.0f}s)", flush=True)

# FCI reference
ef, _ = direct_spin1.FCI().kernel(h1a, era, N_ACT, ne, nroots=NROOTS, verbose=0)
e_fci = [float(e) for e in np.atleast_1d(ef)[:NROOTS]]
for i in range(NROOTS):
    exc = f"  ({(e_fci[i]-e_fci[0])*1000:.1f} mH)" if i > 0 else ""
    print(f"    S{i}: {e_fci[i]:.12f} Ha{exc}", flush=True)

h2_4d = ao2mo.restore('s1', era, N_ACT).reshape(N_ACT, N_ACT, N_ACT, N_ACT)
ham = Hamiltonian(h1=h1a, h2=h2_4d, E_nuc=0.0, E_HF=0.0)
hf_a, hf_b = hf_determinant(*ne)
ao = bit_positions(hf_a); bo = bit_positions(hf_b)
av, bv = [p for p in range(N_ACT) if p not in ao], [p for p in range(N_ACT) if p not in bo]
E_HF = ham.matrix_element((hf_a, hf_b), (hf_a, hf_b))
full_dets = [(int(a), int(b)) for a in as_ for b in bs_]
det_to_full = {d: i for i, d in enumerate(full_dets)}

def s2_of_pvec(cvec_p, p_dets):
    full = np.zeros((na, nb))
    for l, d in enumerate(p_dets):
        full[aidx[int(d[0])], bidx[int(d[1])]] += cvec_p[l]
    nrm = np.linalg.norm(full)
    if nrm > 0: full /= nrm
    return spin_op.spin_square(full, N_ACT, ne)[0]

# ── CIS-seeded P-space initialization ──
P_INIT = 200
init_dets = [(hf_a, hf_b)]
# Force ALL singles into seed (Brillouin fix)
singles = []
for i in ao:
    for a in av: singles.append((hf_a ^ (1<<i) | (1<<a), hf_b))
for i in bo:
    for a in bv: singles.append((hf_a, hf_b ^ (1<<i) | (1<<a)))
for d in singles:
    if d not in init_dets: init_dets.append(d)
n_singles = len(init_dets) - 1

# Top HFPT2 doubles to fill to P_INIT
scores = []
for i1,i2 in itertools.combinations(ao,2):
    for a1,a2 in itertools.combinations(av,2):
        d=(hf_a^(1<<i1)^(1<<i2)|(1<<a1)|(1<<a2), hf_b)
        hij=ham.matrix_element(d,(hf_a,hf_b)); den=E_HF-ham.matrix_element(d,d)
        if abs(den)>1e-12: scores.append((d,-hij*hij/den))
for i1,i2 in itertools.combinations(bo,2):
    for a1,a2 in itertools.combinations(bv,2):
        d=(hf_a, hf_b^(1<<i1)^(1<<i2)|(1<<a1)|(1<<a2))
        hij=ham.matrix_element(d,(hf_a,hf_b)); den=E_HF-ham.matrix_element(d,d)
        if abs(den)>1e-12: scores.append((d,-hij*hij/den))
for i in ao:
    for j in bo:
        for a in av:
            for b in bv:
                d=(hf_a^(1<<i)|(1<<a), hf_b^(1<<j)|(1<<b))
                hij=ham.matrix_element(d,(hf_a,hf_b)); den=E_HF-ham.matrix_element(d,d)
                if abs(den)>1e-12: scores.append((d,-hij*hij/den))
scores.sort(key=lambda x: x[1], reverse=True)
for d,_ in scores:
    if len(init_dets) >= P_INIT: break
    if d not in init_dets: init_dets.append(d)
n_doubles = len(init_dets)-1-n_singles
print(f"  Seed P={len(init_dets)} (HF + {n_singles} singles + {n_doubles} HFPT2 doubles)\n", flush=True)

# ── Matrix-Free build/propagate ──
def build_basis_mf(p_dets, E0, tag=""):
    N = len(p_dets)
    A_q = np.where(np.abs(E0 - hdiag) > 1e-10, 1.0/(E0 - hdiag), 0.0)
    tmpdir = f'{PROJECT_ROOT}/tmp'; os.makedirs(tmpdir, exist_ok=True)
    fpath = f'{tmpdir}/trunc_m1_build_N{N}_{tag}.dat'
    T = np.memmap(fpath, dtype='float64', mode='w+', shape=(M_all, N))
    p_idx_set = set()
    for pa, pb in p_dets:
        idx = q_idx.flat_index(int(pa), int(pb))
        if idx is not None and idx >= 0: p_idx_set.add(idx)
    t_b = time.perf_counter()
    print(f"    [build_basis] T=A*H_QP, N={N} cols...", flush=True)
    for p in range(N):
        pa, pb = int(p_dets[p][0]), int(p_dets[p][1])
        ia = q_idx._alpha_idx.get(pa); ib = q_idx._beta_idx.get(pb)
        if ia is None or ib is None: continue
        ci_unit = np.zeros((na, nb)); ci_unit[ia, ib] = 1.0
        sigma_flat = backend.sigma_full(ci_unit).reshape(-1)
        for q in p_idx_set: sigma_flat[q] = 0.0
        T[:, p] = A_q * sigma_flat  # T = A*H_QP (A1)
        if (p+1) % max(1, N//5) == 0:
            print(f"      col {p+1}/{N} ({time.perf_counter()-t_b:.0f}s)", flush=True)
    T.flush()
    t_svd = time.perf_counter()
    print(f"    SVD({M_all},{N})...", flush=True)
    U, sigma, _ = svd(T, full_matrices=False)
    smax = sigma[0] if len(sigma)>0 else 0
    mask = sigma >= SVD_THR * max(1.0, smax)
    d = int(np.sum(mask))
    e = time.perf_counter()-t_svd
    ratios = ", ".join(f"{s/smax:.4f}" for s in sigma[:min(8,len(sigma))])
    print(f"    SVD done: {e:.0f}s, {N}->d={d} (sigma/sigma1=[{ratios}])", flush=True)
    try: del T; gc.collect(); os.unlink(fpath)
    except: pass
    return U[:, mask], d, A_q, sigma[mask]

def propagate_basis_mf(U_basis, A_q, E0, tag=""):
    M_dim, d_old = U_basis.shape
    if d_old == 0: return U_basis.copy(), d_old
    tmpdir = f'{PROJECT_ROOT}/tmp'; os.makedirs(tmpdir, exist_ok=True)
    fpath = f'{tmpdir}/trunc_m1_prop_d{d_old}_{tag}.dat'
    T = np.memmap(fpath, dtype='float64', mode='w+', shape=(M_dim, d_old))
    t_p = time.perf_counter()
    print(f"    [propagate] d={d_old}, T=A*H_O'*U...", flush=True)
    sigma_cache = np.zeros((M_dim, d_old))
    for k in range(d_old):
        b_k = U_basis[:, k]
        sigma_k = backend.sigma_full(b_k.reshape(na, nb)).reshape(-1)
        residual = sigma_k - hdiag * b_k
        T[:, k] = A_q * residual  # T = A*residual (A1)
        sigma_cache[:, k] = sigma_k
        if (k+1) % max(1, d_old//5) == 0:
            print(f"      col {k+1}/{d_old} ({time.perf_counter()-t_p:.0f}s)", flush=True)
    T.flush()
    t_svd = time.perf_counter()
    print(f"    SVD({M_dim},{d_old})...", flush=True)
    U_new, sigma_new, _ = svd(T, full_matrices=False)
    smax_new = sigma_new[0] if len(sigma_new)>0 else 0
    mask_new = sigma_new >= SVD_THR * max(1.0, smax_new)
    U_incr = U_new[:, mask_new]; d_new = int(np.sum(mask_new))
    e = time.perf_counter()-t_svd
    print(f"    SVD done: {e:.0f}s, {d_old}->d_new={d_new}", flush=True)
    # MGS against existing basis
    from scipy.linalg import block_diag
    if d_new > 0:
        U_incr -= U_basis @ (U_basis.T @ U_incr)
        norms = np.sqrt(np.sum(U_incr**2, axis=0))
        valid = norms > 1e-12
        U_incr = U_incr[:, valid]; d_new = int(np.sum(valid))
    U_full = np.hstack([U_basis, U_incr]) if d_new > 0 else U_basis
    print(f"    MGS done: d_full={U_full.shape[1]}", flush=True)
    try: del T; gc.collect(); os.unlink(fpath)
    except: pass
    return U_full, d_old + d_new, sigma_cache, sigma_new[mask_new]

# ── P-space and H_PP ──
p_dets = list(init_dets)
p_full_idx = [det_to_full[d] for d in p_dets]
p_set = set(p_full_idx)
H_PP = build_hpp_sigma(p_dets, backend, aidx, bidx, na, nb)
N_p = len(p_dets)
SCORING_ROOTS = list(range(min(NROOTS, 5)))
print(f"Iterative P: {N_p} -> {P_MAX}", flush=True)

it = 0
while N_p < P_MAX:
    t_it = time.perf_counter()
    E_P, C_P = eigh(H_PP); E0_cur = E_P[0]
    sigmas = []
    ns = min(len(SCORING_ROOTS), N_p)
    for sk in range(ns):
        k = SCORING_ROOTS[sk]
        vec = embed_pspace_vec(C_P[:, k], p_full_idx, M_all)
        sigmas.append((E_P[k], backend.sigma(vec)))
    p_mask = build_pmask(p_set, M_all)
    sel, max_w, weights = score_and_select(sigmas, hdiag, p_mask, BATCH)
    new_gi = [int(qi) for qi in sel]
    new_dets = [full_dets[qi] for qi in new_gi]
    H_PP = extend_hpp_sigma(H_PP, p_dets, new_dets, backend, aidx, bidx, na, nb)
    p_dets.extend(new_dets); p_full_idx.extend(new_gi); p_set.update(new_gi)
    N_p = len(p_dets)
    dE0 = (E0_cur-e_fci[0])*1000
    print(f"  iter{it:>3}: P={N_p:>5} E0={E0_cur:>14.8f} dE0={dE0:>+7.2f} mH  wall={time.perf_counter()-t_it:>6.1f}s", flush=True)
    it += 1

# ── At P=800: build_basis (m=0) + propagate (m=1) ──
print(f"\n=== Krylov at P={len(p_dets)} ===", flush=True)
p_idx_set = set()
for pa, pb in p_dets:
    idx = q_idx.flat_index(int(pa), int(pb))
    if idx is not None and idx >= 0: p_idx_set.add(idx)
E0_vals, _ = eigh(H_PP); E0 = E0_vals[0]
print(f"  E0={E0:.8f}, dE0(bare)={(E0-e_fci[0])*1000:+.1f} mH", flush=True)

# m=0
U_0, d_0, A_q, sigma_0 = build_basis_mf(p_dets, E0, f"P{TARGET_P}")
print(f"  m=0: d={d_0}", flush=True)

# m=1
U_1, d_1, sigma_cache_prop, sigma_1 = propagate_basis_mf(U_0, A_q, E0, f"P{TARGET_P}")
print(f"  m=1: d_full={d_1}", flush=True)

# Compute sigma vectors for FULL m=1 Krylov basis
print(f"\n  Computing sigma vectors for full m=1 basis ({d_1} cols)...", flush=True)
SIG = np.zeros((M_all, d_1))
t_sig = time.perf_counter()
for k in range(d_1):
    SIG[:, k] = backend.sigma_full(U_1[:, k].reshape(na, nb)).reshape(-1)
    if (k+1) % max(1, d_1//5) == 0:
        print(f"    {k+1}/{d_1} ({time.perf_counter()-t_sig:.0f}s)", flush=True)
print(f"  Sigma done: {time.perf_counter()-t_sig:.0f}s", flush=True)

# Need singular values for the FULL m=1 basis (for truncation ordering)
# Use sigma_0 and sigma_1 to approximate combined ordering
# Actually, the m=1 basis columns from build are in σ_0 order, then propagate columns
# For truncation, we sort ALL columns by their singular value magnitude
# Since U_1 = [U_0_kept, U_1_new], the σ ordering is:
#   First d_0 entries: σ_0 (build SVD singular values)
#   Next (d_1 - d_0) entries: σ_1 (propagate SVD singular values)
sigma_combined = np.concatenate([sigma_0, sigma_1[:d_1 - d_0]])
# Normalize
smax = np.max(sigma_combined)
sigma_norm = sigma_combined / smax if smax > 0 else np.ones(d_1)

# Re-sort basis columns by singular value (descending) for truncation
sort_idx = np.argsort(-sigma_norm)
U_1_sorted = U_1[:, sort_idx]
SIG_sorted = SIG[:, sort_idx]
sigma_sorted = sigma_norm[sort_idx]

# ── H_PP at P=800, per-state E_refs ──
p_flat = kdci_sparse.q_idx.p_indices(p_dets)
p_valid = p_flat >= 0; p_f = p_flat[p_valid]
Np = len(p_dets)
E_vals, _ = eigh(H_PP)
E_refs = E_vals[:NROOTS]

def perstate_eff_eigvals(Hpp, Hpk, Hkk, erefs, nroots):
    ev_list = []
    for k in range(min(nroots, len(erefs))):
        H_eff = build_effective_H(Hpp, Hpk, Hkk, erefs[k], delta_shift=0.0)
        ev = diagonalize_effective_H(H_eff, erefs[k])
        ev_list.append(ev)
    return ev_list

# ── Truncation sweep ──
THRESHOLDS = [1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 2e-1, 5e-1]
print(f"\n{'='*70}")
print(f"SVD Truncation Sweep (m=1 basis, P={TARGET_P}, d_full={d_1})")
print(f"{'='*70}")
print(f"  {'thr':>8} {'r':>5} {'compr%':>8} {'dE0':>9} {'S1':>9} {'S2':>9} {'S3':>9}  (mH)")
print("  " + "-"*56, flush=True)
trunc_results = []
for thr in THRESHOLDS:
    r = int(np.sum(sigma_sorted >= thr))
    if r == 0:
        print(f"  {thr:>8.0e} {0:>5}  (skip)", flush=True); continue
    U_r = U_1_sorted[:, :r]; SIG_r = SIG_sorted[:, :r]
    H_KK_r = U_r.T @ SIG_r
    H_KK_r = 0.5 * (H_KK_r + H_KK_r.T)
    H_PK_r = np.zeros((Np, r))
    H_PK_r[p_valid, :] = SIG_r[p_f, :]
    ev = perstate_eff_eigvals(H_PP, H_PK_r, H_KK_r, E_refs, NROOTS)
    dE = [(ev[k] - e_fci[k])*1000 for k in range(min(NROOTS, len(ev)))]
    compr = 100.0*(1.0 - r/d_1)
    print(f"  {thr:>8.0e} {r:>5} {compr:>7.1f}% {dE[0]:>+9.3f} {dE[1]:>+9.3f} {dE[2]:>+9.3f} {dE[3]:>+9.3f}", flush=True)
    trunc_results.append({'thr': float(thr), 'r': r, 'd0': int(d_1),
                          'compr_pct': float(compr),
                          'dE_mH': [float(x) for x in dE[:NROOTS]]})

# ── Save ──
outdir = f'{PROJECT_ROOT}/checkpoints_phaseA'
os.makedirs(outdir, exist_ok=True)
outpath = f'{outdir}/cas10_trunc_dE_P{TARGET_P}_m1_cis.json'
with open(outpath, 'w') as f:
    json.dump({'config': {'target_p': TARGET_P, 'm_max': M_MAX,
                          'svd_threshold': SVD_THR, 'd_full': int(d_1), 'd_m0': int(d_0),
                          'smax': float(smax), 'e_fci': e_fci, 'tag': TAG},
               'sigma': [float(s) for s in sigma_sorted],
               'results': trunc_results}, f, indent=2)
print(f"\nSaved: {outpath}", flush=True)

# ── Summary ──
print(f"\n{'='*70}")
print(f"Summary: P={TARGET_P}, d_m0={d_0}, d_m1={d_1}")
print(f"{'='*70}")
print(f"  {'thr':>8} {'r':>5} {'compr%':>8} {'dE0':>9} {'dE1':>9} {'dE2':>9}")
for t in trunc_results:
    print(f"  {t['thr']:>8.0e} {t['r']:>5} {t['compr_pct']:>7.1f}% " +
          " ".join(f"{t['dE_mH'][k]:>+9.1f}" for k in range(min(3, len(t['dE_mH'])))))
print(f"\nTotal: {time.perf_counter()-t0:.0f}s")
print("Done.")
