#!/usr/bin/env python3
"""
CIS-seeded m=1 SVD truncation sweep at P=800.
Uses the EXACT v10_sacis pipeline. Only adds truncation after Krylov construction.
"""
import sys, os, time, json, itertools, gc
import numpy as np
from numpy.linalg import eigh, svd

PROJECT_ROOT = '/data/home/wangcx/krylov-dci'
sys.path.insert(0, PROJECT_ROOT)

from src_mf import QSpaceIndex, KDCIBackend, KDCISparse
from src_mf.pspace_ops import embed_pspace_vec, build_pmask, score_and_select, build_hpp_sigma, extend_hpp_sigma
from src.effective_h import build_effective_H, diagonalize_effective_H, self_consistent_iteration
from src.determinants import hf_determinant, bit_positions
from src.hamiltonian import Hamiltonian
from pyscf import gto, scf, ao2mo
from pyscf.fci import cistring, direct_spin1, spin_op

TARGET_P = 800; M_MAX = 1; SVD_THR = 1e-3; BATCH = 200
N_ACT = 10; N_CORE = 2; NROOTS = 6; R = 1.1; ne = (5, 5)

total_t0 = time.perf_counter()
print("=" * 70)
print(f"CIS-Seeded m=1 Truncation Sweep: P={TARGET_P}, CAS(10,10)")
print("=" * 70, flush=True)

# ─── PySCF setup ───
mol = gto.M(atom=f'N 0 0 0; N 0 0 {R}', basis='cc-pVDZ', verbose=0, spin=0)
mf = scf.RHF(mol).run(verbose=0)
ncore = N_CORE; ncas = N_ACT; ne_cas = sum(ne)
cas_list = list(range(ncore, ncore + ncas))
mc = mf.CASSCF(ncas, ne_cas); mc.fix_spin_(ss=0)
mo_coeff = mc.sort_mo(cas_list, base=0)
h1 = mo_coeff.T @ mf.get_hcore() @ mo_coeff
h1 = h1[ncore:ncore+ncas, ncore:ncore+ncas]
era = ao2mo.kernel(mol, mo_coeff[:, ncore:ncore+ncas], aosym='s4')
h1a = h1.copy(); h1b = h1.copy()

as_ = cistring.gen_strings4orblist(range(N_ACT), ne[0])
bs_ = cistring.gen_strings4orblist(range(N_ACT), ne[1])
na, nb = len(as_), len(bs_); M_all = na * nb
aidx = {int(s): i for i, s in enumerate(as_)}
bidx = {int(s): i for i, s in enumerate(bs_)}
q_idx = QSpaceIndex(as_, bs_, N_ACT, ne, h1a, era)
backend = KDCIBackend(q_idx); kdci_sparse = KDCISparse(q_idx)
hdiag = q_idx.hdiag
full_dets = [(int(a), int(b)) for a in as_ for b in bs_]
det_to_full = {d: i for i, d in enumerate(full_dets)}

# FCI
ef, _ = direct_spin1.FCI().kernel(h1a, era, N_ACT, ne, nroots=NROOTS, verbose=0)
e_fci = [float(e) for e in np.atleast_1d(ef)[:NROOTS]]
for i in range(NROOTS):
    exc = f"  ({(e_fci[i]-e_fci[0])*1000:.1f} mH)" if i > 0 else ""
    print(f"  FCI S{i}: {e_fci[i]:.12f} Ha{exc}", flush=True)

h2_4d = ao2mo.restore('s1', era, N_ACT).reshape(N_ACT, N_ACT, N_ACT, N_ACT)
ham = Hamiltonian(h1=h1a, h2=h2_4d, E_nuc=0.0, E_HF=0.0)
hf_a, hf_b = hf_determinant(*ne)
ao = bit_positions(hf_a); bo = bit_positions(hf_b)
av, bv = [p for p in range(N_ACT) if p not in ao], [p for p in range(N_ACT) if p not in bo]
E_HF = ham.matrix_element((hf_a, hf_b), (hf_a, hf_b))

def s2_of_pvec(cvec_p, p_dets):
    full = np.zeros((na, nb))
    for l, d in enumerate(p_dets):
        full[aidx[int(d[0])], bidx[int(d[1])]] += cvec_p[l]
    nrm = np.linalg.norm(full)
    if nrm > 0: full /= nrm
    return spin_op.spin_square(full, N_ACT, ne)[0]

# ─── CIS-seeded P-space (EXACTLY matching v10_sacis) ───
P_INIT = 200
init_dets = [(hf_a, hf_b)]
singles = []
for i in ao:
    for a in av: singles.append((hf_a ^ (1<<i) | (1<<a), hf_b))
for i in bo:
    for a in bv: singles.append((hf_a, hf_b ^ (1<<i) | (1<<a)))
for d in singles:
    if d not in init_dets: init_dets.append(d)
n_singles = len(init_dets) - 1

# Top HFPT2 doubles
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

# ─── Iterative P-space expansion (state-average, v10_sacis style) ───
p_dets = list(init_dets)
p_full_idx = [det_to_full[d] for d in p_dets]
p_set = set(p_full_idx)
H_PP = build_hpp_sigma(p_dets, backend, aidx, bidx, na, nb)
N_p = len(p_dets)
SCORING_ROOTS = list(range(min(NROOTS, 5)))

print(f"Iterative P: {N_p} → {TARGET_P}", flush=True)
it = 0
while N_p < TARGET_P:
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
    print(f"  iter{it:>3}: P={N_p:>5} dE0(bare)={dE0:>+7.1f} mH  wall={time.perf_counter()-t_it:>5.1f}s", flush=True)
    it += 1

# ─── Krylov: build_basis + propagate (A¹ weight, matching v10_sacis) ───
print(f"\n=== Krylov at P={len(p_dets)} ===", flush=True)
p_idx_set = set()
for pa, pb in p_dets:
    idx = q_idx.flat_index(int(pa), int(pb))
    if idx is not None and idx >= 0: p_idx_set.add(idx)
E0_vals, _ = eigh(H_PP); E0 = E0_vals[0]
print(f"  E0={E0:.8f}, dE0(bare)={(E0-e_fci[0])*1000:+.1f} mH", flush=True)

# build_basis_mf: T = A · H_QP → SVD
A_q = np.where(np.abs(E0 - hdiag) > 1e-10, 1.0/(E0 - hdiag), 0.0)
tmpdir = f'{PROJECT_ROOT}/tmp'; os.makedirs(tmpdir, exist_ok=True)
fpath = f'{tmpdir}/sacis_trunc_build.dat'
T = np.memmap(fpath, dtype='float64', mode='w+', shape=(M_all, len(p_dets)))
print(f"  [build_basis] T=A*H_QP, N={len(p_dets)} cols...", flush=True)
t_b = time.perf_counter()
for p in range(len(p_dets)):
    pa, pb = int(p_dets[p][0]), int(p_dets[p][1])
    ia = q_idx._alpha_idx.get(pa); ib = q_idx._beta_idx.get(pb)
    if ia is None or ib is None: continue
    ci_unit = np.zeros((na, nb)); ci_unit[ia, ib] = 1.0
    sigma_flat = backend.sigma_full(ci_unit).reshape(-1)
    for q in p_idx_set: sigma_flat[q] = 0.0
    T[:, p] = A_q * sigma_flat  # T = A * H_QP (A1)
    if (p+1) % max(1, len(p_dets)//5) == 0:
        print(f"    col {p+1}/{len(p_dets)} ({time.perf_counter()-t_b:.0f}s)", flush=True)
T.flush()
print(f"  SVD({M_all},{len(p_dets)})...", flush=True)
U_0, sigma_0_raw, _ = svd(T, full_matrices=False)
smax0 = sigma_0_raw[0]
mask0 = sigma_0_raw >= SVD_THR * smax0
U_0 = U_0[:, mask0]; d_0 = int(np.sum(mask0))
sigma_0 = (sigma_0_raw[mask0] / smax0).astype(float)
print(f"  m=0: d={d_0}, sigmas normalized to [1.0, {sigma_0[-1]:.4f}]", flush=True)
try: del T; gc.collect(); os.unlink(fpath)
except: pass

# propagate: residual = H*b_k - D*b_k, T = A * residual → SVD → MGS
fpath_p = f'{tmpdir}/sacis_trunc_prop.dat'
T_p = np.memmap(fpath_p, dtype='float64', mode='w+', shape=(M_all, d_0))
print(f"  [propagate] d={d_0}...", flush=True)
t_p = time.perf_counter()
for k in range(d_0):
    sigma_k = backend.sigma_full(U_0[:, k].reshape(na, nb)).reshape(-1)
    residual = sigma_k - hdiag * U_0[:, k]
    T_p[:, k] = A_q * residual  # T = A * residual (A1)
    if (k+1) % max(1, d_0//5) == 0:
        print(f"    col {k+1}/{d_0} ({time.perf_counter()-t_p:.0f}s)", flush=True)
T_p.flush()
print(f"  SVD({M_all},{d_0})...", flush=True)
U_new, sigma_prop_raw, _ = svd(T_p, full_matrices=False)
smax_p = sigma_prop_raw[0]
mask_p = sigma_prop_raw >= SVD_THR * smax_p
U_incr = U_new[:, mask_p]; d_new = int(np.sum(mask_p))
sigma_prop = (sigma_prop_raw[mask_p] / smax_p).astype(float)
# MGS against existing basis
U_incr -= U_0 @ (U_0.T @ U_incr)
norms = np.sqrt(np.sum(U_incr**2, axis=0))
valid = norms > 1e-12
U_incr = U_incr[:, valid]; d_new = int(np.sum(valid))
sigma_prop = sigma_prop[valid]
U_1 = np.hstack([U_0, U_incr]); d_1 = U_1.shape[1]
sigma_combined = np.concatenate([sigma_0, sigma_prop[:d_1 - len(sigma_0)]])
print(f"  m=1: d_full={d_1}, sigma range=[{sigma_combined.min():.4f}, {sigma_combined.max():.4f}]", flush=True)
try: del T_p; gc.collect(); os.unlink(fpath_p)
except: pass

# Sort basis by descending sigma
sort_idx = np.argsort(-sigma_combined[:d_1])
U_1 = U_1[:, sort_idx]
sigma_combined = sigma_combined[sort_idx]
d_1 = U_1.shape[1]

# ─── Compute sigma vectors and do truncation sweep ───
print(f"\n  Computing sigma vectors for m=1 basis ({d_1} cols)...", flush=True)
SIG = np.zeros((M_all, d_1))
t_s = time.perf_counter()
for k in range(d_1):
    SIG[:, k] = backend.sigma_full(U_1[:, k].reshape(na, nb)).reshape(-1)
    if (k+1) % max(1, d_1//5) == 0:
        print(f"    {k+1}/{d_1} ({time.perf_counter()-t_s:.0f}s)", flush=True)
print(f"  Sigma done: {time.perf_counter()-t_s:.0f}s", flush=True)

p_flat = kdci_sparse.q_idx.p_indices(p_dets)
p_valid = p_flat >= 0; p_f = p_flat[p_valid]
Np = len(p_dets)
E_refs = E0_vals[:NROOTS]

def perstate_eff_eigvals(Hpp, Hpk, Hkk, erefs, nroots):
    ev_out = np.zeros(len(erefs))
    for k, Ek in enumerate(erefs[:nroots]):
        evk = np.asarray(diagonalize_effective_H(
            build_effective_H(Hpp, Hpk, Hkk, float(Ek), delta=0.0),
            n_states=nroots)[0])
        ev_out[k] = evk[int(np.argmin(np.abs(evk - Ek)))]
    return ev_out

THRESHOLDS = [1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 2e-1, 5e-1]
print(f"\n{'='*70}")
print(f"SVD Truncation Sweep (m=1, P={TARGET_P}, d={d_1})")
print(f"{'='*70}")
print(f"  {'thr':>8} {'r':>5} {'compr%':>8} {'dE0':>9} {'S1':>9} {'S2':>9} {'S3':>9}  (mH)")
print("  " + "-" * 70, flush=True)

trunc_results = []
for thr in THRESHOLDS:
    r = int(np.sum(sigma_combined[:d_1] >= thr))
    if r == 0:
        print(f"  {thr:>8.0e} {0:>5}  (skip)", flush=True); continue
    U_r = U_1[:, :r]; SIG_r = SIG[:, :r]
    H_KK_r = U_r.T @ SIG_r; H_KK_r = 0.5*(H_KK_r + H_KK_r.T)
    H_PK_r = np.zeros((Np, r))
    H_PK_r[p_valid, :] = SIG_r[p_f, :]
    ev = perstate_eff_eigvals(H_PP, H_PK_r, H_KK_r, E_refs, NROOTS)
    dE = [(ev[k] - e_fci[k])*1000 for k in range(min(NROOTS, len(ev)))]
    compr = 100.0*(1.0 - r/d_1)
    print(f"  {thr:>8.0e} {r:>5} {compr:>7.1f}% {dE[0]:>+9.3f} {dE[1]:>+9.3f} {dE[2]:>+9.3f} {dE[3]:>+9.3f}", flush=True)
    trunc_results.append({'thr': float(thr), 'r': r, 'd0': int(d_1),
                          'compr_pct': float(compr),
                          'dE_mH': [float(x) for x in dE[:NROOTS]]})

# Save
outdir = f'{PROJECT_ROOT}/checkpoints_phaseA'
os.makedirs(outdir, exist_ok=True)
outpath = f'{outdir}/cas10_trunc_dE_P800_m1_cis.json'
with open(outpath, 'w') as f:
    json.dump({'config': {'target_p': TARGET_P, 'm_max': M_MAX, 'd_full': int(d_1),
                          'e_fci': e_fci, 'tag': 'sacis_m1_trunc'},
               'sigma': [float(s) for s in sigma_combined[:d_1]],
               'results': trunc_results}, f, indent=2)
print(f"\nSaved: {outpath}", flush=True)

# Summary
print(f"\n{'='*70}")
print(f"Summary: P={TARGET_P}, m=1, d_full={d_1}")
for t in trunc_results:
    print(f"  thr={t['thr']:.0e}: r={t['r']}/{d_1} ({t['compr_pct']:.1f}% compr)  " +
          "  ".join(f"S{k}={t['dE_mH'][k]:+.1f}" for k in range(min(4, len(t['dE_mH'])))))
print(f"\nTotal: {time.perf_counter()-total_t0:.0f}s")
