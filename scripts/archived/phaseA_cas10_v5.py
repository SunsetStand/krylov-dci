#!/usr/bin/env python3
"""
Phase A v5 — CAS(10,10) Krylov Basis Expansion (NOT Neumann series)

Correct m-expansion:
  m=0: K_0 (build_basis+SVD) → H_eff = H_PP + H_PK·(E₀I-H_KK)⁻¹·H_KP
  m=1: propagate K_0 → ΔK → K_1=[K_0|ΔK] → exact resolvent in expanded basis
  m=2: propagate ΔK → ΔK' → K_2=[K_1|ΔK'] → ...
  
K-space optimization (P1):
  - Pre-compute H_KK (from projected blocks)
  - Propagation: v_perp = H·w_k - K·H_KK[:,k]  (subtract K-projection using H_KK)
  - Avoids inner-product loop with all existing basis vectors

Usage:
    python phaseA_cas10_v5.py --P 200,500,1000,2000 --svd-threshold 1e-3 --m-max 3
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
    p.add_argument('--m-max', type=int, default=3)
    p.add_argument('--batch', type=int, default=200)
    p.add_argument('--tag', type=str, default='v5')
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
print(f"Phase A v5 — CAS({N_ACT},{sum(ne)})  Krylov Basis Expansion m=0..{M_MAX}")
print(f"N2/cc-pVDZ R={R}  checkpoints={P_CHECKPOINTS}  svd_thr={SVD_THR}")
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

# Hamiltonian & HF
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
# HFPT2 initial P (reverse=True FIXED)
# ═══════════════════════════════════════════════════════════════
def gen_hfpt2_scores():
    sc = []
    for i in ao:
        for a in av:
            d=(hf_a^(1<<i)|(1<<a),hf_b)
            hij=ham.matrix_element(d,(hf_a,hf_b));den=E_HF-ham.matrix_element(d,d)
            if abs(den)>1e-12:sc.append((d,-hij*hij/den))
    for i in bo:
        for a in bv:
            d=(hf_a,hf_b^(1<<i)|(1<<a))
            hij=ham.matrix_element(d,(hf_a,hf_b));den=E_HF-ham.matrix_element(d,d)
            if abs(den)>1e-12:sc.append((d,-hij*hij/den))
    for i1,i2 in itertools.combinations(ao,2):
        for a1,a2 in itertools.combinations(av,2):
            d=(hf_a^(1<<i1)^(1<<i2)|(1<<a1)|(1<<a2),hf_b)
            hij=ham.matrix_element(d,(hf_a,hf_b));den=E_HF-ham.matrix_element(d,d)
            if abs(den)>1e-12:sc.append((d,-hij*hij/den))
    for i1,i2 in itertools.combinations(bo,2):
        for a1,a2 in itertools.combinations(bv,2):
            d=(hf_a,hf_b^(1<<i1)^(1<<i2)|(1<<a1)|(1<<a2))
            hij=ham.matrix_element(d,(hf_a,hf_b));den=E_HF-ham.matrix_element(d,d)
            if abs(den)>1e-12:sc.append((d,-hij*hij/den))
    for i in ao:
        for j in bo:
            for a in av:
                for b in bv:
                    d=(hf_a^(1<<i)|(1<<a),hf_b^(1<<j)|(1<<b))
                    hij=ham.matrix_element(d,(hf_a,hf_b));den=E_HF-ham.matrix_element(d,d)
                    if abs(den)>1e-12:sc.append((d,-hij*hij/den))
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
    n=len(dets);H=np.zeros((n,n))
    for i in range(n):
        for j in range(i,n):
            v=ham.matrix_element(dets[i],dets[j]);H[i,j]=v;H[j,i]=v
    return H

def extend_hpp(H_old, old_dets, new_dets):
    No=len(old_dets);na=len(new_dets);Hn=np.zeros((No+na,No+na))
    Hn[:No,:No]=H_old
    for il,dn in enumerate(new_dets):
        r=No+il
        for j in range(No):
            v=ham.matrix_element(dn,old_dets[j]);Hn[r,j]=v;Hn[j,r]=v
        for jl in range(il+1):
            c=No+jl;v=ham.matrix_element(dn,new_dets[jl]);Hn[r,c]=v;Hn[c,r]=v
    return Hn


# ═══════════════════════════════════════════════════════════════
# build_basis with raw save (streaming MGS + memmap)
# ═══════════════════════════════════════════════════════════════
def build_basis_with_raw(p_dets, E0, tag=""):
    N = len(p_dets)
    A_q = np.where(np.abs(E0 - hdiag) > 1e-10, 1.0 / (E0 - hdiag), 0.0)
    tmpdir = f'{PROJECT_ROOT}/tmp'; os.makedirs(tmpdir, exist_ok=True)
    fpath = f'{tmpdir}/phaseA_v5_raw_{tag}_N{N}.dat'
    V_raw = np.memmap(fpath, dtype='float64', mode='w+', shape=(M, N))
    p_idx_set = set()
    for pa, pb in p_dets:
        idx = q_idx.flat_index(int(pa), int(pb))
        if idx is not None and idx >= 0: p_idx_set.add(idx)
    t0 = time.perf_counter()
    print(f"    [stream] N={N} → MGS + raw memmap...", flush=True)
    basis = []
    for p in range(N):
        pa, pb = int(p_dets[p][0]), int(p_dets[p][1])
        ia = q_idx._alpha_idx.get(pa); ib = q_idx._beta_idx.get(pb)
        if ia is None or ib is None: continue
        ci_unit = np.zeros((na, nb)); ci_unit[ia, ib] = 1.0
        sigma_flat = backend.sigma_full(ci_unit).reshape(-1)
        for q in p_idx_set: sigma_flat[q] = 0.0
        V_raw[:, p] = A_q * sigma_flat
        w_p = SparseQVector()
        nnz = np.where(np.abs(sigma_flat) > 1e-14)[0]
        for q in nnz:
            if q in p_idx_set: continue
            val = A_q[q] * sigma_flat[q]
            if abs(val) > 1e-14:
                a_s = int(q_idx.alpha_strs[q // nb])
                b_s = int(q_idx.beta_strs[q % nb])
                w_p[(a_s, b_s)] = float(val)
        for b in basis: w_p.add_scaled(b, alpha=-b.dot(w_p))
        nrm = w_p.norm()
        if nrm > 1e-10:
            w_p.scale(1.0 / nrm)
            basis.append(w_p)
        if (p+1) % max(1, N//5) == 0:
            print(f"      col {p+1}/{N}, basis={len(basis)} "
                  f"({time.perf_counter()-t0:.0f}s)", flush=True)
    V_raw.flush()
    d = len(basis); e = time.perf_counter()-t0
    nnz_tot = sum(b.nnz() for b in basis)
    print(f"    [stream] done: {N}→{d} vectors, {nnz_tot} nnz, {e:.0f}s", flush=True)
    return basis, d, V_raw, A_q, fpath


# ═══════════════════════════════════════════════════════════════
# SVD on raw vectors
# ═══════════════════════════════════════════════════════════════
def svd_compress(V_raw, A_q, threshold):
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
    print(f"    [SVD] done: {e:.0f}s, {K}→d_svd={d} (σ/σ₁=[{ratios}])", flush=True)
    return U_r, sig_r, d


# ═══════════════════════════════════════════════════════════════
# Build H_KK, H_PK from dense basis
# ═══════════════════════════════════════════════════════════════
def build_blocks(U_basis, p_dets):
    """H_KK[j,k]=⟨w_k|H|w_j⟩, H_PK[p,k]=⟨p|H|w_k⟩."""
    d = U_basis.shape[1]; Np = len(p_dets)
    if d == 0: return np.zeros((0,0)), np.zeros((Np,0))
    t0 = time.perf_counter()
    print(f"    [blocks] d={d} vectors...", flush=True)
    H_KK = np.zeros((d, d)); H_PK = np.zeros((Np, d))
    p_flat = kdci_sparse.q_idx.p_indices(p_dets)
    p_valid = p_flat >= 0; p_f = p_flat[p_valid]
    for k in range(d):
        ci_k = U_basis[:, k].reshape(na, nb)
        sk = backend.sigma_full(ci_k).reshape(-1)
        H_KK[:, k] = U_basis.T @ sk
        H_PK[p_valid, k] = sk[p_f]
        if (k+1) % max(1, d//5) == 0:
            print(f"      {k+1}/{d} ({time.perf_counter()-t0:.0f}s)", flush=True)
    H_KK = 0.5*(H_KK + H_KK.T)
    print(f"    [blocks] done: {time.perf_counter()-t0:.0f}s", flush=True)
    return H_KK, H_PK


# ═══════════════════════════════════════════════════════════════
# Krylov propagation (K-space optimized)
# ═══════════════════════════════════════════════════════════════
def propagate_layer(U_full):
    """Standard Krylov-Arnoldi: propagate through H.

    A_q weighting is captured by m=0 basis. m>0 finds new H-iterated directions.
    Operator: H (full Hamiltonian in Q-space).
    """
    d_total = U_full.shape[1]
    if d_total == 0: return np.zeros((M, 0)), 0
    t0 = time.perf_counter()
    print(f"    [propagate] d={d_total} → (H only, no A_q) ...", flush=True)

    new_vecs = []
    max_new = min(d_total, 50)  # cap new vectors per level
    for k in range(d_total):
        w_k = U_full[:, k]
        # v = H @ w_k (standard Krylov)
        v = backend.sigma_full(w_k.reshape(na, nb)).reshape(-1)

        # MGS: orthonormalize against U_full
        for j in range(d_total):
            v -= np.dot(U_full[:, j], v) * U_full[:, j]
        # MGS: against previously retained new vectors
        for w_prev in new_vecs:
            v -= np.dot(w_prev, v) * w_prev

        nrm = norm(v)
        if nrm > 1e-5:
            v /= nrm
            new_vecs.append(v)
            if len(new_vecs) >= max_new:
                break

        if (k+1) % max(1, d_total//5) == 0:
            print(f"      col {k+1}/{d_total}, new={len(new_vecs)} "
                  f"({time.perf_counter()-t0:.0f}s)", flush=True)

    d_new = len(new_vecs)
    U_new = np.column_stack(new_vecs) if d_new > 0 else np.zeros((M, 0))
    e = time.perf_counter()-t0
    print(f"    [propagate] done: {d_total} → d_new={d_new}, {e:.0f}s", flush=True)
    return U_new, d_new


# ═══════════════════════════════════════════════════════════════
# Effective H from expanded basis
# ═══════════════════════════════════════════════════════════════
def extend_blocks(H_KK_old, H_PK_old, U_old, U_new, p_dets):
    """Extend H_KK and H_PK for U_old→[U_old|U_new] without recomputing old.

    Keeps H_KK_old, H_PK_old intact. Only computes new rows/columns.
    """
    d0 = U_old.shape[1]; d1 = U_new.shape[1]; Np = len(p_dets)
    if d1 == 0: return H_KK_old.copy(), H_PK_old.copy()
    t0 = time.perf_counter()
    print(f"    [extend] d={d0}+{d1} (incremental)...", flush=True)

    H_KK = np.zeros((d0+d1, d0+d1))
    H_KK[:d0, :d0] = H_KK_old
    H_PK = np.zeros((Np, d0+d1))
    H_PK[:, :d0] = H_PK_old

    p_flat = kdci_sparse.q_idx.p_indices(p_dets)
    p_valid = p_flat >= 0; p_f = p_flat[p_valid]

    # Compute new columns: H·u for each new vector
    for k in range(d1):
        u_k = U_new[:, k]
        sk = backend.sigma_full(u_k.reshape(na, nb)).reshape(-1)
        # Old-new coupling: U_old^T @ H @ u_k
        H_KK[:d0, d0+k] = U_old.T @ sk
        H_KK[d0+k, :d0] = H_KK[:d0, d0+k]  # symmetric
        # New-new coupling: U_new^T @ H @ u_k
        H_KK[d0:, d0+k] = U_new.T @ sk
        # P-K coupling
        H_PK[p_valid, d0+k] = sk[p_f]

        if (k+1) % max(1, d1//5) == 0:
            print(f"      new {k+1}/{d1} ({time.perf_counter()-t0:.0f}s)", flush=True)

    H_KK = 0.5*(H_KK + H_KK.T)
    e = time.perf_counter()-t0
    print(f"    [extend] done: {e:.0f}s", flush=True)
    return H_KK, H_PK


def build_heff_from_basis(H_PP, U_basis, p_dets, E0, nroots, delta=0.0):
    """H_eff = H_PP + H_PK · (E₀·I - H_KK)⁻¹ · H_KP.

    Builds H_KK, H_PK from U_basis and computes exact resolvent inverse.
    """
    H_KK, H_PK = build_blocks(U_basis, p_dets)
    ev, _ = diagonalize_effective_H(
        build_effective_H(H_PP, H_PK, H_KK, E0, delta=delta),
        n_states=nroots)
    return ev, H_KK, H_PK


# ═══════════════════════════════════════════════════════════════
# Full Krylov expansion m=0..M_MAX
# ═══════════════════════════════════════════════════════════════
def krylov_expansion(H_PP, U_0, p_dets, E0, A_q, nroots, m_max):
    """Krylov basis expansion: (A_q·B)^m · K_0, exact resolvent at each m."""
    results = []
    t_total = time.perf_counter()

    # m=0: use U_0 directly
    U_m = U_0
    d_m = U_0.shape[1]
    ev_m, H_KK_m, H_PK_m = build_heff_from_basis(H_PP, U_m, p_dets, E0, nroots, delta=0.0)
    dE_m = [(ev_m[k]-e_fci[k])*1000 for k in range(min(nroots, len(ev_m)))]
    results.append({'ev': ev_m, 'd': d_m, 'dE': dE_m})
    dt = time.perf_counter()-t_total
    print(f"    m=0: d={d_m}, dE0={dE_m[0]:+.1f} mH  ({dt:.0f}s)", flush=True)

    for m in range(1, m_max + 1):
        # Propagate
        U_base = U_m  # save before expansion
        U_new, d_new = propagate_layer(U_m)

        if d_new == 0:
            print(f"    m={m}: no new vectors (Krylov exhausted)", flush=True)
            results.append({'ev': ev_m, 'd': d_m, 'dE': dE_m})
            continue

        # Extend H_KK, H_PK incrementally (keep K_0 H_KK intact!)
        H_KK_m, H_PK_m = extend_blocks(
            H_KK_m, H_PK_m, U_base, U_new, p_dets)
        U_m = np.hstack([U_base, U_new])
        d_m = U_m.shape[1]

        ev_m, _ = diagonalize_effective_H(
            build_effective_H(H_PP, H_PK_m, H_KK_m, E0, delta=0.0),
            n_states=nroots)
        dE_m = [(ev_m[k]-e_fci[k])*1000 for k in range(min(nroots, len(ev_m)))]

        ddE = dE_m[0] - results[-1]['dE'][0]
        dt = time.perf_counter()-t_total
        print(f"    m={m}: d={d_m} (+{d_new}), dE0={dE_m[0]:+.1f} mH  "
              f"(Δ={ddE:+.1f})  ({dt:.0f}s)", flush=True)

        results.append({'ev': ev_m, 'd': d_m, 'dE': dE_m})

    wall = time.perf_counter()-t_total
    print(f"    Krylov wall: {wall:.0f}s (m_max={m_max})", flush=True)
    return results


# ═══════════════════════════════════════════════════════════════
# Checkpoint evaluation
# ═══════════════════════════════════════════════════════════════
def eval_checkpoint(p_dets, p_full_idx, H_PP_sub, p_target, it_num):
    N = len(p_dets)
    E0_vals, _ = eigh(H_PP_sub); E0 = E0_vals[0]
    dE0_bare = (E0 - e_fci[0])*1000
    print(f"  P={N}, E0={E0:.8f}, dE0(bare)={dE0_bare:+.3f} mH", flush=True)

    # Step 1: build_basis + raw → SVD
    tag = f"P{p_target}_i{it_num}"
    basis_sp, d_mgs, V_raw, A_q, raw_path = build_basis_with_raw(p_dets, E0, tag)
    U_svd, sigma_svd, d_svd = svd_compress(V_raw, A_q, SVD_THR)
    try: del V_raw; gc.collect(); os.unlink(raw_path)
    except: pass

    compression = f"{d_svd}/{d_mgs}" if d_mgs > 0 else "0/0"
    print(f"  MGS→SVD: d_mgs={d_mgs} → d_svd={d_svd} ({compression})", flush=True)

    # Step 2: MGS reference (sparse, known-good)
    H_KK_mgs, H_PK_mgs = kdci_sparse.build_projected_blocks_sparse(
        basis_sp, p_dets, verbose=False)
    ev_mgs_ref = diagonalize_effective_H(
        build_effective_H(H_PP_sub, H_PK_mgs, H_KK_mgs, E0, delta=0.0),
        n_states=NROOTS)[0]
    dE_mgs_ref = (ev_mgs_ref[0] - e_fci[0])*1000

    # Step 3: Full Krylov expansion (m=0..M_MAX) from SVD basis
    print(f"\n  ── Krylov expansion m=0..{M_MAX} ──", flush=True)
    kr_results = krylov_expansion(
        H_PP_sub, U_svd, p_dets, E0, A_q, NROOTS, M_MAX)

    # Print detailed results
    print(f"\n  Results:", flush=True)
    for m, kr in enumerate(kr_results):
        dE = kr['dE']
        tag_mgs = f"  (vs MGS ref: Δ={dE[0]-dE_mgs_ref:+.1f})" if m == 0 else \
                  f"  (Δm={dE[0]-kr_results[m-1]['dE'][0]:+.1f})"
        print(f"    m={m}: d={kr['d']}, dE0={dE[0]:+.1f} mH{tag_mgs}", flush=True)
        if m <= 1:
            for k in range(1, min(4, NROOTS)):
                print(f"      S{k}: dE={dE[k]:+.1f} mH", flush=True)

    # Summary
    ex_de = [abs(kr_results[0]['dE'][k]) for k in range(1,min(NROOTS,len(kr_results[0]['dE'])))]
    print(f"\n  Summary P={p_target}: d_mgs={d_mgs} d_svd={d_svd} "
          f"dE0(m=0)={kr_results[0]['dE'][0]:+.1f} "
          f"dE0(m={M_MAX})={kr_results[M_MAX]['dE'][0]:+.1f} mH  "
          f"max|dE_ex|={max(ex_de):.0f} mH\n", flush=True)

    sigs = [float(s) for s in sigma_svd[:min(20,len(sigma_svd))]]
    smax = sigs[0] if sigs else 0

    return {
        'P': p_target, 'N': N, 'iter': it_num,
        'd_mgs': d_mgs, 'd_svd': d_svd,
        'E0': float(E0), 'dE0_bare_mH': float(dE0_bare),
        'sigma_max': smax,
        'sigma_ratios': [s/smax if smax>0 else 0 for s in sigs],
        'dE_mgs_ref_mH': dE_mgs_ref,
        'krylov': {m: {'d': kr['d'],
                       'dE_mH': kr['dE'][:NROOTS]}
                   for m, kr in enumerate(kr_results)},
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

    for pt in P_CHECKPOINTS:
        if N_p >= pt and pt not in all_results:
            print(f"\n  ══ Checkpoint P={pt} ══", flush=True)
            all_results[pt] = eval_checkpoint(
                p_dets[:pt], p_full_idx[:pt], H_PP[:pt,:pt], pt, it)

# ── Final Summary ──
print(f"\n{'='*70}")
print(f"Phase A v5 Complete: {time.perf_counter()-total_t0:.0f}s")
print(f"{'='*70}")

# m-convergence table
print(f"\n{'P':>6} {'N':>6} {'d_svd':>7} ", end="")
for m in range(M_MAX+1):
    print(f"{'m='+str(m):>10} ", end="")
print(f"{'mgs_ref':>10}")
print("-"*(30+12*(M_MAX+2)))
for pt in P_CHECKPOINTS:
    r = all_results[pt]
    print(f"{pt:>6} {r['N']:>6} {r['d_svd']:>7} ", end="")
    for m in range(M_MAX+1):
        print(f"{r['krylov'][m]['dE_mH'][0]:>+10.1f} ", end="")
    print(f"{r['dE_mgs_ref_mH']:>+10.1f}")

# Save
outdir = os.path.join(PROJECT_ROOT, 'checkpoints_phaseA')
os.makedirs(outdir, exist_ok=True)
with open(f'{outdir}/phaseA_v5_m{M_MAX}_svd{SVD_THR}_{TAG}.json','w') as f:
    json.dump({
        'config': {'cas':N_ACT,'n_core':N_CORE,'P':P_CHECKPOINTS,
                   'svd_threshold':SVD_THR,'m_max':M_MAX,'M':M,
                   'e_fci':e_fci,'tag':TAG},
        'results': {str(k):v for k,v in all_results.items()},
    }, f, indent=2)
print(f"\nSaved: {outdir}/phaseA_v5_m{M_MAX}_svd{SVD_THR}_{TAG}.json")
print("Done.")
