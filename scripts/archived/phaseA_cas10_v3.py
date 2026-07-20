#!/usr/bin/env python3
"""
Phase A v3 — CAS(10,10) Iterative-P + MGS + SVD + Krylov

Pipeline:
  1. Iterative P-space selection (σ-vector scoring via KDCIBackend.sigma)
  2. At each checkpoint:
     a. build_basis_streaming: MGS + save raw vectors to memmap
     b. SVD on raw vectors → d_svd (weighted, with threshold)
     c. build_projected_blocks_sparse from SVD-compressed basis
     d. m=0 effective H
     e. m=1 Krylov propagation
     f. Compare d_mgs vs d_svd

Key design choices:
  - Uses proven KDCISparse methods for MGS and projected blocks
  - SVD operates on raw (pre-MGS) vectors saved during streaming
  - Iterative P selection reuses step1 logic (σ-vector scoring)

Usage:
    python phaseA_cas10_v3.py --P 200,500,1000,2000 --svd-threshold 1e-3
"""
import sys, os, time, json, argparse, itertools, gc
import numpy as np
from numpy.linalg import eigh, svd, norm

PROJECT_ROOT = '/data/home/wangcx/krylov-dci'
sys.path.insert(0, PROJECT_ROOT)

from src_mf import QSpaceIndex, KDCIBackend, KDCISparse
from src_mf.sparse_vector import SparseQVector
from src.effective_h import build_effective_H, diagonalize_effective_H
from src.determinants import hf_determinant, bit_positions
from src.hamiltonian import Hamiltonian
from pyscf import gto, scf, ao2mo
from pyscf.fci import cistring, direct_spin1

# ═══════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--P', type=str, default='200,500,1000,2000')
    p.add_argument('--svd-threshold', type=float, default=1e-3)
    p.add_argument('--m-max', type=int, default=1)
    p.add_argument('--batch', type=int, default=200)
    p.add_argument('--tag', type=str, default='v3')
    return p.parse_args()

args = parse_args()
P_CHECKPOINTS = sorted([int(x) for x in args.P.split(',')])
SVD_THR = args.svd_threshold; M_MAX = args.m_max; BATCH = args.batch; TAG = args.tag
P_MAX = max(P_CHECKPOINTS)

# ═══════════════════════════════════════════════════════════════
# System: N2/cc-pVDZ CAS(10,10)
# ═══════════════════════════════════════════════════════════════
N_ACT = 10; N_CORE = 2; NROOTS = 6; R = 1.1; ne = (5, 5)

print("=" * 70)
print(f"Phase A v3 — CAS({N_ACT},{sum(ne)})  Iterative-P + MGS + SVD + Krylov")
print(f"N2/cc-pVDZ R={R}  checkpoints={P_CHECKPOINTS}")
print(f"svd_thr={SVD_THR}  m_max={M_MAX}  batch={BATCH}")
print("=" * 70, flush=True)

t0 = time.perf_counter()
mol = gto.M(atom=f'N 0 0 0; N 0 0 {R}', basis='cc-pVDZ', verbose=0)
mf = scf.RHF(mol).run(verbose=0)
na_o = list(range(N_CORE, N_CORE+N_ACT))
norb = mf.mo_coeff.shape[1]
h1_mo = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
eri_4d = ao2mo.full(mol.intor('int2e'), mf.mo_coeff, compact=False)
eri_4d = eri_4d.reshape(norb, norb, norb, norb)
h1a = h1_mo[np.ix_(na_o, na_o)]; era = eri_4d[np.ix_(na_o, na_o, na_o, na_o)]
as_ = cistring.gen_strings4orblist(range(N_ACT), ne[0])
bs_ = cistring.gen_strings4orblist(range(N_ACT), ne[1])
na, nb = len(as_), len(bs_); M = na*nb
q_idx = QSpaceIndex(as_, bs_, N_ACT, ne, h1a, era)
backend = KDCIBackend(q_idx)
kdci_sparse = KDCISparse(q_idx)
hdiag = q_idx.hdiag

# FCI reference
print("  FCI reference...", flush=True)
ef, _ = direct_spin1.FCI().kernel(h1a, era, N_ACT, ne, nroots=NROOTS, verbose=0)
e_fci = [float(e) for e in np.atleast_1d(ef)[:NROOTS]]
for i in range(NROOTS):
    exc = f"  ({(e_fci[i]-e_fci[0])*1000:.1f} mH)" if i > 0 else "  (ground)"
    print(f"    S{i}: {e_fci[i]:.12f} Ha{exc}")

# Hamiltonian & full det list
h2_4d = ao2mo.restore('s1', era, N_ACT).reshape(N_ACT, N_ACT, N_ACT, N_ACT)
ham = Hamiltonian(h1=h1a, h2=h2_4d, E_nuc=0.0, E_HF=0.0)
hf_a, hf_b = hf_determinant(*ne)
ao = bit_positions(hf_a); bo = bit_positions(hf_b)
av, bv = [p for p in range(N_ACT) if p not in ao], [p for p in range(N_ACT) if p not in bo]
E_HF = ham.matrix_element((hf_a, hf_b), (hf_a, hf_b))
full_dets = [(int(a), int(b)) for a in as_ for b in bs_]
det_to_full = {d: i for i, d in enumerate(full_dets)}
print(f"  CAS({N_ACT},{sum(ne)}): M={M:,}  ({time.perf_counter()-t0:.0f}s)\n")

# ═══════════════════════════════════════════════════════════════
# HFPT2 initial P
# ═══════════════════════════════════════════════════════════════
def gen_hfpt2_scores():
    sc = []
    for i in ao:
        for a in av:
            d = (hf_a^(1<<i)|(1<<a), hf_b)
            hij=ham.matrix_element(d,(hf_a,hf_b))
            den=E_HF-ham.matrix_element(d,d)
            if abs(den)>1e-12: sc.append((d,-hij*hij/den))
    for i in bo:
        for a in bv:
            d = (hf_a, hf_b^(1<<i)|(1<<a))
            hij=ham.matrix_element(d,(hf_a,hf_b))
            den=E_HF-ham.matrix_element(d,d)
            if abs(den)>1e-12: sc.append((d,-hij*hij/den))
    for i1,i2 in itertools.combinations(ao,2):
        for a1,a2 in itertools.combinations(av,2):
            d=(hf_a^(1<<i1)^(1<<i2)|(1<<a1)|(1<<a2), hf_b)
            hij=ham.matrix_element(d,(hf_a,hf_b))
            den=E_HF-ham.matrix_element(d,d)
            if abs(den)>1e-12: sc.append((d,-hij*hij/den))
    for i1,i2 in itertools.combinations(bo,2):
        for a1,a2 in itertools.combinations(bv,2):
            d=(hf_a, hf_b^(1<<i1)^(1<<i2)|(1<<a1)|(1<<a2))
            hij=ham.matrix_element(d,(hf_a,hf_b))
            den=E_HF-ham.matrix_element(d,d)
            if abs(den)>1e-12: sc.append((d,-hij*hij/den))
    for i in ao:
        for j in bo:
            for a in av:
                for b in bv:
                    d=(hf_a^(1<<i)|(1<<a), hf_b^(1<<j)|(1<<b))
                    hij=ham.matrix_element(d,(hf_a,hf_b))
                    den=E_HF-ham.matrix_element(d,d)
                    if abs(den)>1e-12: sc.append((d,-hij*hij/den))
    sc.sort(key=lambda x: x[1], reverse=True)
    return sc

P_INIT = P_CHECKPOINTS[0]
scores = gen_hfpt2_scores()
init_dets = [(hf_a, hf_b)]
for d, _ in scores:
    if d not in init_dets: init_dets.append(d)
    if len(init_dets) >= P_INIT: break
print(f"  HFPT2 initial P={len(init_dets)}\n")

# H_PP builders
def build_hpp(dets):
    n=len(dets); H=np.zeros((n,n))
    for i in range(n):
        for j in range(i,n):
            v=ham.matrix_element(dets[i],dets[j]); H[i,j]=v; H[j,i]=v
    return H

def extend_hpp(H_old, old_dets, new_dets):
    No=len(old_dets); na=len(new_dets); Hn=np.zeros((No+na,No+na))
    Hn[:No,:No]=H_old
    for il,dn in enumerate(new_dets):
        r=No+il
        for j in range(No):
            v=ham.matrix_element(dn,old_dets[j]); Hn[r,j]=v; Hn[j,r]=v
        for jl in range(il+1):
            c=No+jl; v=ham.matrix_element(dn,new_dets[jl]); Hn[r,c]=v; Hn[c,r]=v
    return Hn


# ═══════════════════════════════════════════════════════════════
# build_basis WITH raw vector saving
# ═══════════════════════════════════════════════════════════════
def build_basis_with_raw(p_dets, E0, tag=""):
    """Streaming MGS + save raw vectors to memmap.

    Returns: (basis_sparse, d_mgs, V_raw_path)
      basis_sparse: list of SparseQVector (orthonormal)
      d_mgs: number of basis vectors
      V_raw_path: path to memmap of raw vectors (M, N)
    """
    N = len(p_dets)
    denom = E0 - hdiag
    A_q = np.where(np.abs(denom) > 1e-10, 1.0 / denom, 0.0)

    tmpdir = f'{PROJECT_ROOT}/tmp'; os.makedirs(tmpdir, exist_ok=True)
    fpath = f'{tmpdir}/phaseA_v3_raw_{tag}_N{N}.dat'
    V_raw = np.memmap(fpath, dtype='float64', mode='w+', shape=(M, N))

    # P indices for zeroing
    p_idx_set = set()
    for pa, pb in p_dets:
        idx = q_idx.flat_index(int(pa), int(pb))
        if idx is not None and idx >= 0: p_idx_set.add(idx)

    t0 = time.perf_counter()
    print(f"    [stream] N={N} columns → MGS + raw memmap...", flush=True)

    basis = []
    for p in range(N):
        pa, pb = int(p_dets[p][0]), int(p_dets[p][1])
        ia = q_idx._alpha_idx.get(pa); ib = q_idx._beta_idx.get(pb)
        if ia is None or ib is None: continue

        ci_unit = np.zeros((na, nb)); ci_unit[ia, ib] = 1.0
        sigma_mat = backend.sigma_full(ci_unit)
        sigma_flat = sigma_mat.reshape(-1)

        # Zero P-space components
        for q in p_idx_set: sigma_flat[q] = 0.0

        # Save raw (weighted) vector
        V_raw[:, p] = A_q * sigma_flat

        # MGS: build sparse weighted vector, orthonormalize
        w_p = SparseQVector()
        nnz = np.where(np.abs(sigma_flat) > 1e-14)[0]
        for q in nnz:
            if q in p_idx_set: continue
            val = A_q[q] * sigma_flat[q]
            if abs(val) > 1e-14:
                a_s = int(q_idx.alpha_strs[q // nb])
                b_s = int(q_idx.beta_strs[q % nb])
                w_p[(a_s, b_s)] = float(val)

        for b in basis:
            proj = b.dot(w_p)
            w_p.add_scaled(b, alpha=-proj)

        nrm = w_p.norm()
        if nrm > 1e-10:
            w_p.scale(1.0 / nrm)
            basis.append(w_p)

        if (p+1) % max(1, N//5) == 0:
            e = time.perf_counter()-t0
            print(f"      col {p+1}/{N}, basis={len(basis)} ({e:.0f}s)", flush=True)

    V_raw.flush()
    d = len(basis)
    e = time.perf_counter()-t0
    nnz_tot = sum(b.nnz() for b in basis)
    print(f"    [stream] done: {N}→{d} vectors, {nnz_tot} nnz, {e:.0f}s", flush=True)
    return basis, d, V_raw, A_q, fpath


# ═══════════════════════════════════════════════════════════════
# SVD on raw vectors
# ═══════════════════════════════════════════════════════════════
def svd_compress(V_raw, A_q, threshold):
    """T = A^{1/2}·V_raw → SVD → truncated U."""
    M_dim, K = V_raw.shape
    t0 = time.perf_counter()
    sqrt_A = np.sqrt(np.abs(A_q))
    T = V_raw * sqrt_A[:, np.newaxis]
    print(f"    [SVD] SVD({M_dim},{K})...", flush=True)
    U, sigma, Vt = svd(T, full_matrices=False)
    smax = sigma[0] if len(sigma)>0 else 0
    if smax<1e-15: return np.zeros((M_dim,0)), np.array([]), 0
    mask = sigma >= threshold*smax
    d = np.sum(mask)
    U_r = U[:, mask]; sig_r = sigma[mask]
    e = time.perf_counter()-t0
    ratios = ", ".join(f"{s/smax:.4f}" for s in sigma[:min(8,len(sigma))])
    print(f"    [SVD] done: {e:.0f}s, K={K}→d_svd={d} (σ/σ₁=[{ratios}])", flush=True)
    return U_r, sig_r, d


# ═══════════════════════════════════════════════════════════════
# Build projected blocks from dense SVD basis
# ═══════════════════════════════════════════════════════════════
def projected_blocks_from_dense(U_dense, p_dets):
    """From dense (M,d) basis: H_QQ_tilde, H_PQ_tilde."""
    d = U_dense.shape[1]; Np = len(p_dets)
    if d == 0: return np.zeros((0,0)), np.zeros((Np,0))
    t0 = time.perf_counter()
    print(f"    [proj] Projecting d={d} dense vectors...", flush=True)

    H_QQ_t = np.zeros((d, d)); H_PQ_t = np.zeros((Np, d))
    p_flat = kdci_sparse.q_idx.p_indices(p_dets)
    p_valid = p_flat >= 0; p_f = p_flat[p_valid]

    for k in range(d):
        ci_k = U_dense[:, k].reshape(na, nb)
        sigma_k = backend.sigma_full(ci_k).reshape(-1)
        H_QQ_t[:, k] = U_dense.T @ sigma_k
        H_PQ_t[p_valid, k] = sigma_k[p_f]
        if (k+1) % max(1, d//5) == 0:
            print(f"      basis {k+1}/{d} ({time.perf_counter()-t0:.0f}s)", flush=True)

    H_QQ_t = 0.5*(H_QQ_t + H_QQ_t.T)
    e = time.perf_counter()-t0
    print(f"    [proj] done: {e:.0f}s", flush=True)
    return H_QQ_t, H_PQ_t


# ═══════════════════════════════════════════════════════════════
# m=0, m=1
# ═══════════════════════════════════════════════════════════════
def krylov_m0(H_PP, H_PQ_t, H_QQ_t, E0):
    return diagonalize_effective_H(
        build_effective_H(H_PP, H_PQ_t, H_QQ_t, E0, delta=0.0),
        n_states=NROOTS)[0]

def krylov_m1(H_PP, H_PQ_t, H_QQ_t, E0, U_comp, p_dets):
    d0 = U_comp.shape[1]
    if d0 == 0: return krylov_m0(H_PP, H_PQ_t, H_QQ_t, E0), 0
    t0 = time.perf_counter()
    A_q = np.where(np.abs(E0-hdiag)>1e-10, 1.0/(E0-hdiag), 0.0)

    new_raw = np.zeros((M, d0))
    for k in range(d0):
        sk = backend.sigma_full(U_comp[:,k].reshape(na,nb)).reshape(-1)
        new_raw[:, k] = A_q * sk

    new_orth = np.zeros((M, d0)); retained = []
    for k in range(d0):
        v = new_raw[:, k].copy()
        for j in range(d0): v -= np.dot(U_comp[:,j], v)*U_comp[:,j]
        for r in retained: v -= np.dot(new_orth[:,r], v)*new_orth[:,r]
        nrm = norm(v)
        if nrm > 1e-10: new_orth[:,k]=v/nrm; retained.append(k)

    d1 = len(retained)
    if d1 == 0:
        print(f"    [m=1] all dependent → =m=0", flush=True)
        return krylov_m0(H_PP, H_PQ_t, H_QQ_t, E0), 0

    U_exp = np.hstack([U_comp, new_orth[:,:d1]])
    H_QQ_e, H_PQ_e = projected_blocks_from_dense(U_exp, p_dets)
    ev = krylov_m0(H_PP, H_PQ_e, H_QQ_e, E0)
    print(f"    [m=1] d={d0}+{d1}, {time.perf_counter()-t0:.0f}s", flush=True)
    return ev, d1


# ═══════════════════════════════════════════════════════════════
# Checkpoint evaluation
# ═══════════════════════════════════════════════════════════════
def eval_checkpoint(p_dets, p_full_idx, H_PP_sub, p_target, it_num):
    N = len(p_dets)
    E0_vals, _ = eigh(H_PP_sub); E0 = E0_vals[0]
    dE0_bare = (E0 - e_fci[0])*1000
    print(f"  P={N}, E0={E0:.8f}, dE0(bare)={dE0_bare:+.3f} mH", flush=True)

    # Step a: build_basis with raw save
    tag = f"P{p_target}_i{it_num}"
    basis_sp, d_mgs, V_raw, A_q, raw_path = build_basis_with_raw(p_dets, E0, tag)

    # Step b: SVD on raw
    U_svd, sigma_svd, d_svd = svd_compress(V_raw, A_q, SVD_THR)

    try: del V_raw; gc.collect(); os.unlink(raw_path)
    except: pass

    compression = f"{d_svd}/{d_mgs}" if d_mgs > 0 else "0/0"
    print(f"  MGS→SVD: d_mgs={d_mgs} → d_svd={d_svd} ({compression})", flush=True)

    # Step c: MGS sparse basis → projected blocks (reference, known-good)
    print("  ── m=0 (MGS sparse) ──", flush=True)
    t_mgs = time.perf_counter()
    H_QQ_mgs, H_PQ_mgs = kdci_sparse.build_projected_blocks_sparse(basis_sp, p_dets, verbose=False)
    ev_mgs = krylov_m0(H_PP_sub, H_PQ_mgs, H_QQ_mgs, E0)
    wall_mgs_bloch = time.perf_counter()-t_mgs
    dE_mgs = [(ev_mgs[k]-e_fci[k])*1000 for k in range(min(NROOTS,len(ev_mgs)))]
    for k in range(min(NROOTS,len(ev_mgs))):
        print(f"    S{k}: E={ev_mgs[k]:.12f}  dE={dE_mgs[k]:+8.1f} mH  [MGS]", flush=True)

    # Step d: SVD dense basis → projected blocks (compare)
    print("  ── m=0 (SVD dense) ──", flush=True)
    t_svd_b = time.perf_counter()
    H_QQ_svd, H_PQ_svd = projected_blocks_from_dense(U_svd, p_dets)
    ev_svd = krylov_m0(H_PP_sub, H_PQ_svd, H_QQ_svd, E0)
    wall_svd_b = time.perf_counter()-t_svd_b
    dE_svd = [(ev_svd[k]-e_fci[k])*1000 for k in range(min(NROOTS,len(ev_svd)))]
    for k in range(min(NROOTS,len(ev_svd))):
        dd = dE_svd[k] - dE_mgs[k]
        print(f"    S{k}: E={ev_svd[k]:.12f}  dE={dE_svd[k]:+8.1f} mH  (d={dd:+.1f} vs MGS)", flush=True)

    # Use MGS result as primary (known-good)
    ev_m0 = ev_mgs; dE_m0 = dE_mgs; wall_m0 = wall_mgs_bloch

    # Step e: m=1 (uses SVD-compressed basis as starting point)
    ev_m1, dE_m1, wall_m1, d1 = None, None, None, 0
    if M_MAX >= 1:
        print("  ── m=1 ──", flush=True)
        t_m1 = time.perf_counter()
        ev_m1, d1 = krylov_m1(H_PP_sub, H_PQ_mgs, H_QQ_mgs, E0,
                              U_svd, p_dets)
        wall_m1 = time.perf_counter()-t_m1
        dE_m1 = [(ev_m1[k]-e_fci[k])*1000 for k in range(min(NROOTS,len(ev_m1)))]
        for k in range(min(NROOTS,len(ev_m1))):
            ddE = dE_m1[k]-dE_m0[k]
            print(f"    S{k}: E={ev_m1[k]:.12f}  dE={dE_m1[k]:+8.1f} mH  (Δ={ddE:+.1f})",
                  flush=True)

    ex_de = [abs(dE_m0[k]) for k in range(1,min(NROOTS,len(ev_m0)))]
    print(f"  Summary: d_mgs={d_mgs} d_svd={d_svd} dE0={dE_m0[0]:+.1f} mH  "
          f"max|dE_ex|={max(ex_de):.0f} mH\n", flush=True)

    sigs = [float(s) for s in sigma_svd[:min(20,len(sigma_svd))]]
    smax = sigs[0] if sigs else 0

    return {
        'P': p_target, 'N': N, 'iter': it_num,
        'd_mgs': d_mgs, 'd_svd': d_svd,
        'E0': float(E0), 'dE0_bare_mH': float(dE0_bare),
        'sigma_max': smax,
        'sigma_ratios': [s/smax if smax>0 else 0 for s in sigs],
        'm0': {'E': [float(e) for e in ev_m0[:NROOTS]],
               'dE_mH': dE_m0[:NROOTS], 'wall_s': wall_m0},
        'm1': ({'E': [float(e) for e in ev_m1[:NROOTS]],
                'dE_mH': dE_m1[:NROOTS], 'wall_s': wall_m1, 'd1': d1}
               if ev_m1 is not None else None),
    }


# ═══════════════════════════════════════════════════════════════
# Main: iterative P expansion
# ═══════════════════════════════════════════════════════════════
p_dets = list(init_dets)
p_full_idx = [det_to_full[d] for d in p_dets]
p_set = set(p_full_idx)
H_PP = build_hpp(p_dets)
N_p = len(p_dets)
SCORING_ROOTS = list(range(min(NROOTS, 5)))
all_results = {}

print(f"Iterative P: {N_p} → {P_MAX}")
print(f"{'iter':>4} {'P':>6} {'E0_bare':>14} {'dE0_mH':>10} {'max_w':>10} {'wall':>8}")
print("-"*56, flush=True)

total_t0 = time.perf_counter()
it = 0

while N_p < P_MAX:
    t_it = time.perf_counter()
    E_P, C_P = eigh(H_PP)
    E0_cur = E_P[0]

    # σ-vector scoring
    sigmas = []
    ns = min(len(SCORING_ROOTS), N_p)
    for sk in range(ns):
        k = SCORING_ROOTS[sk]
        vec = np.zeros(M)
        for li, gi in enumerate(p_full_idx): vec[gi] = C_P[li, k]
        sigma_k = backend.sigma(vec)
        sigmas.append((E_P[k], sigma_k))

    weights = np.zeros(M)
    for E_ref, sk in sigmas:
        abs_s = np.abs(sk)
        for qi in range(M):
            if qi in p_set: continue
            c2 = abs_s[qi]**2
            if c2 < 1e-24: continue
            weights[qi] += c2 / max(abs(E_ref-hdiag[qi]), 1e-8)

    cands = [(qi, float(weights[qi])) for qi in range(M)
             if qi not in p_set and weights[qi] > 0]
    cands.sort(key=lambda x: x[1], reverse=True)
    n_add = min(BATCH, len(cands))
    max_w = cands[0][1] if cands else 0

    new_gi = [c[0] for c in cands[:n_add]]
    new_dets = [full_dets[qi] for qi in new_gi]
    H_PP = extend_hpp(H_PP, p_dets, new_dets)
    p_dets.extend(new_dets); p_full_idx.extend(new_gi); p_set.update(new_gi)
    N_p = len(p_dets)

    dE0 = (E0_cur-e_fci[0])*1000
    print(f"{it:>4} {N_p:>6} {E0_cur:>14.8f} {dE0:>+10.3f} {max_w:>10.3e} "
          f"{time.perf_counter()-t_it:>8.1f}", flush=True)
    it += 1

    # Checkpoint evaluation
    for pt in P_CHECKPOINTS:
        if N_p >= pt and pt not in all_results:
            print(f"\n  ══ Checkpoint P={pt} ══", flush=True)
            all_results[pt] = eval_checkpoint(
                p_dets[:pt], p_full_idx[:pt], H_PP[:pt,:pt], pt, it)

# ── Summary ──
print(f"\n{'='*70}")
print(f"Phase A v3 Complete: {time.perf_counter()-total_t0:.0f}s")
print(f"{'='*70}")
print(f"\n{'P':>6} {'N':>6} {'d_mgs':>7} {'d_svd':>7} {'dE0_m0':>10} "
      f"{'dE0_m1':>10} {'max|dE_ex|':>12}")
print("-"*58)
for pt in P_CHECKPOINTS:
    r = all_results[pt]
    d0 = r['m0']['dE_mH'][0]
    d1 = r['m1']['dE_mH'][0] if r['m1'] else float('nan')
    mx = max(abs(r['m0']['dE_mH'][k]) for k in range(1,min(NROOTS,len(r['m0']['dE_mH']))))
    print(f"{pt:>6} {r['N']:>6} {r['d_mgs']:>7} {r['d_svd']:>7} {d0:>+10.1f} "
          f"{d1:>+10.1f} {mx:>12.0f}")

# Save
outdir = os.path.join(PROJECT_ROOT, 'checkpoints_phaseA')
os.makedirs(outdir, exist_ok=True)
with open(f'{outdir}/phaseA_v3_svd{SVD_THR}_{TAG}.json','w') as f:
    json.dump({
        'config': {'cas':N_ACT,'n_core':N_CORE,'P':P_CHECKPOINTS,
                   'svd_threshold':SVD_THR,'m_max':M_MAX,'M':M,
                   'e_fci':e_fci,'tag':TAG},
        'results': {str(k):v for k,v in all_results.items()},
    }, f, indent=2)
print(f"\nSaved: {outdir}/phaseA_v3_svd{SVD_THR}_{TAG}.json")
print("Done.")
