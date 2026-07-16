#!/usr/bin/env python3
"""
Phase A v4 — CAS(10,10) Full P1 Proposal: m=0,1,2,3 + K-space optimization

Key improvements over v3:
  - m=0,1,2,3 Krylov propagation (full Neumann series)
  - Pre-reduce B into K basis: B_K = K^T · B · K  (d×d, avoids M-dim ops)
  - Pre-compute A_K = K^T · A · K  (d×d)
  - Full Neumann series in compressed K-space
  - B_K threshold truncation (sort by magnitude, keep top fraction)
  - Iterative P-space selection + MGS + SVD

Pipeline per checkpoint:
  1. Iterative P selection (σ-vector scoring)
  2. build_basis_streaming (MGS) + save raw vectors → SVD compression
  3. Compute H_PK = H_PQ_tilde, H_KK = H_QQ_tilde (from SVD basis)
  4. Pre-compute D_K, A_K, B_K in K-space (d×d)
  5. m=0,1,2,3 Neumann series → H_eff corrections
  6. Record dE vs FCI, d_basis(P), m-convergence

Usage:
    python phaseA_cas10_v4.py --P 200,500,1000,2000 --svd-threshold 1e-3 --m-max 3
"""
import sys, os, time, json, argparse, itertools, gc
import numpy as np
from numpy.linalg import eigh, svd, norm
from scipy.linalg import sqrtm

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
    p.add_argument('--b-threshold', type=float, default=0.0,
                   help='B_K truncation (fraction of max|B|), 0=no truncation')
    p.add_argument('--tag', type=str, default='v4')
    return p.parse_args()

args = parse_args()
P_CHECKPOINTS = sorted([int(x) for x in args.P.split(',')])
SVD_THR = args.svd_threshold; M_MAX = args.m_max; BATCH = args.batch
B_THR = args.b_threshold; TAG = args.tag
P_MAX = max(P_CHECKPOINTS)

# ═══════════════════════════════════════════════════════════════
# System: N2/cc-pVDZ CAS(10,10)
# ═══════════════════════════════════════════════════════════════
N_ACT = 10; N_CORE = 2; NROOTS = 6; R = 1.1; ne = (5, 5)

print("=" * 70)
print(f"Phase A v4 — CAS({N_ACT},{sum(ne)})  Full P1: m=0..{M_MAX} + K-space opt")
print(f"N2/cc-pVDZ R={R}  checkpoints={P_CHECKPOINTS}")
print(f"svd_thr={SVD_THR}  m_max={M_MAX}  batch={BATCH}  b_thr={B_THR}")
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
# HFPT2 initial P (FIXED: reverse=True)
# ═══════════════════════════════════════════════════════════════
def gen_hfpt2_scores():
    sc = []
    for i in ao:
        for a in av:
            d=(hf_a^(1<<i)|(1<<a), hf_b)
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
    fpath = f'{tmpdir}/phaseA_v4_raw_{tag}_N{N}.dat'
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
        # MGS
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
            w_p.add_scaled(b, alpha=-b.dot(w_p))
        nrm = w_p.norm()
        if nrm > 1e-10:
            w_p.scale(1.0 / nrm)
            basis.append(w_p)
        if (p+1) % max(1, N//5) == 0:
            print(f"      col {p+1}/{N}, basis={len(basis)} "
                  f"({time.perf_counter()-t0:.0f}s)", flush=True)
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
# Build H_PK and H_KK from dense SVD basis
# ═══════════════════════════════════════════════════════════════
def build_blocks_from_svd(U_svd, p_dets):
    """H_KK[j,k]=⟨w_k|H|w_j⟩, H_PK[p,k]=⟨p|H|w_k⟩."""
    d = U_svd.shape[1]; Np = len(p_dets)
    if d == 0: return np.zeros((0,0)), np.zeros((Np,0))
    t0 = time.perf_counter()
    print(f"    [blocks] Building from d={d} SVD vectors...", flush=True)
    H_KK = np.zeros((d, d)); H_PK = np.zeros((Np, d))
    p_flat = kdci_sparse.q_idx.p_indices(p_dets)
    p_valid = p_flat >= 0; p_f = p_flat[p_valid]
    for k in range(d):
        ci_k = U_svd[:, k].reshape(na, nb)
        sk = backend.sigma_full(ci_k).reshape(-1)
        H_KK[:, k] = U_svd.T @ sk
        H_PK[p_valid, k] = sk[p_f]
        if (k+1) % max(1, d//5) == 0:
            print(f"      basis {k+1}/{d} ({time.perf_counter()-t0:.0f}s)", flush=True)
    H_KK = 0.5*(H_KK + H_KK.T)
    e = time.perf_counter()-t0
    print(f"    [blocks] done: {e:.0f}s", flush=True)
    return H_KK, H_PK


# ═══════════════════════════════════════════════════════════════
# P1 optimizations: K-space pre-computations
# ═══════════════════════════════════════════════════════════════
def compute_k_space_matrices(U_svd, A_q, H_KK, E0, delta, b_threshold):
    """Pre-compute A_K, D_K, B_K in compressed K-space.

    D_K[j,k]  = Σ_q K[q,j] · hdiag[q] · K[q,k]      (d×d)
    A_K[j,k]  = Σ_q K[q,j] · A_q[q] · K[q,k]         (d×d)
    B_K       = H_KK - D_K - Δ·I                      (d×d)

    If b_threshold > 0: truncate B_K by keeping top fraction of |entries|.
    """
    d = U_svd.shape[1]
    if d == 0: return np.zeros((0,0)), np.zeros((0,0)), np.zeros((0,0))

    t0 = time.perf_counter()

    # D_K = U^T · diag(hdiag) · U
    D_K = (U_svd * hdiag[:, np.newaxis]).T @ U_svd

    # A_K = U^T · diag(A_q) · U
    A_K = (U_svd * A_q[:, np.newaxis]).T @ U_svd

    # B_K = H_KK - D_K - delta*I
    B_K = H_KK - D_K - delta * np.eye(d)

    # Symmetrize
    D_K = 0.5*(D_K + D_K.T)
    A_K = 0.5*(A_K + A_K.T)
    B_K = 0.5*(B_K + B_K.T)

    # B_K threshold truncation
    if b_threshold > 0:
        abs_B = np.abs(B_K)
        max_abs = np.max(abs_B)
        if max_abs > 1e-15:
            mask = abs_B >= b_threshold * max_abs
            n_kept = np.sum(mask)
            B_K_trunc = B_K * mask
            n_total = d*d
            print(f"    [K-opt] B_K truncated: {n_kept}/{n_total} entries "
                  f"({100*n_kept/n_total:.1f}%), thr={b_threshold}", flush=True)
            B_K = B_K_trunc

    e = time.perf_counter()-t0
    print(f"    [K-opt] A_K, D_K, B_K: d={d}, {e:.0f}s", flush=True)
    return A_K, D_K, B_K


# ═══════════════════════════════════════════════════════════════
# Krylov propagation in K-space (Neumann series)
# ═══════════════════════════════════════════════════════════════
def krylov_neumann_kspace(H_PP, H_PK, A_K, B_K, E0, delta, m_max, nroots):
    """Neumann series in compressed K-space.

    H_P^eff(m) = H_PP + H_PK · A_K^{1/2} · Σ_{k=0}^{m} (A_K·B_K)^k · A_K^{1/2} · H_PK^T

    Where all matrices are d×d (or N×d for H_PK).

    Returns: list of eigenvalues for m=0..m_max
    """
    d = A_K.shape[0]
    if d == 0:
        ev0, _ = diagonalize_effective_H(H_PP, n_states=nroots)
        return [ev0[:nroots] for _ in range(m_max+1)]

    # A_K^{1/2}: matrix square root of A_K
    try:
        A_half = sqrtm(A_K)
    except:
        # Fallback: eigenvalues may be negative → use eigendecomposition
        w, V = eigh(A_K)
        w_pos = np.maximum(w, 0)
        A_half = V @ np.diag(np.sqrt(w_pos)) @ V.T

    A_half = np.real(A_half)  # Ensure real

    # Base term: H_PK · A_K^{1/2} (N×d)
    base_right = H_PK @ A_half  # N×d × d×d = N×d

    results = []
    term_k = np.eye(d)  # Σ start: I

    for m in range(m_max + 1):
        # Σ_{k=0}^{m} (A_K·B_K)^k
        if m > 0:
            # Add next term: (A_K·B_K) · term_{k-1}
            term_k = (A_K @ B_K) @ term_k + np.eye(d)
            # Actually: Σ_{k=0}^{m} = I + AB + (AB)^2 + ... + (AB)^m
            # term_0 = I
            # term_1 = I + AB = term_0 + AB
            # term_2 = I + AB + ABAB = term_1 + ABAB
            # Need to accumulate properly

        # Recompute full sum for each m (avoids accumulation errors)
        sum_k = np.eye(d)
        power = np.eye(d)
        for k in range(1, m + 1):
            power = (A_K @ B_K) @ power  # (AB)^k
            sum_k = sum_k + power         # I + AB + (AB)^2 + ...

        # H_eff = H_PP + base_right @ sum_k @ base_right^T
        middle = A_half @ sum_k @ A_half  # d×d
        correction = H_PK @ middle @ H_PK.T  # N×N
        H_eff = H_PP + correction
        H_eff = 0.5 * (H_eff + H_eff.T)

        ev, _ = diagonalize_effective_H(H_eff, n_states=nroots)
        results.append(ev[:nroots])

    return results


# ═══════════════════════════════════════════════════════════════
# Checkpoint evaluation
# ═══════════════════════════════════════════════════════════════
def eval_checkpoint(p_dets, p_full_idx, H_PP_sub, p_target, it_num):
    N = len(p_dets)
    E0_vals, _ = eigh(H_PP_sub); E0 = E0_vals[0]
    dE0_bare = (E0 - e_fci[0])*1000
    print(f"  P={N}, E0={E0:.8f}, dE0(bare)={dE0_bare:+.3f} mH", flush=True)

    # Step 1: build_basis + raw save
    tag = f"P{p_target}_i{it_num}"
    basis_sp, d_mgs, V_raw, A_q_full, raw_path = build_basis_with_raw(p_dets, E0, tag)
    A_q = A_q_full  # for K-space

    # Step 2: SVD on raw
    U_svd, sigma_svd, d_svd = svd_compress(V_raw, A_q, SVD_THR)
    try: del V_raw; gc.collect(); os.unlink(raw_path)
    except: pass

    compression = f"{d_svd}/{d_mgs}" if d_mgs > 0 else "0/0"
    print(f"  MGS→SVD: d_mgs={d_mgs} → d_svd={d_svd} ({compression})", flush=True)

    # Step 3: Build H_KK, H_PK from SVD basis
    H_KK, H_PK = build_blocks_from_svd(U_svd, p_dets)

    # Step 3.5: Also compute MGS-sparse reference (for validation)
    H_KK_mgs, H_PK_mgs = kdci_sparse.build_projected_blocks_sparse(
        basis_sp, p_dets, verbose=False)
    ev_mgs_ref = diagonalize_effective_H(
        build_effective_H(H_PP_sub, H_PK_mgs, H_KK_mgs, E0, delta=0.0),
        n_states=NROOTS)[0]
    dE_mgs_ref = [(ev_mgs_ref[k]-e_fci[k])*1000 for k in range(min(NROOTS,len(ev_mgs_ref)))]

    # Step 4: K-space pre-computations (P1 optimizations)
    delta = 0.0
    A_K, D_K, B_K = compute_k_space_matrices(U_svd, A_q, H_KK, E0, delta, B_THR)

    # Step 5: Krylov Neumann series m=0..M_MAX
    print(f"\n  ── Krylov m=0..{M_MAX} (K-space Neumann) ──", flush=True)
    t_krylov = time.perf_counter()
    ev_all_m = krylov_neumann_kspace(
        H_PP_sub, H_PK, A_K, B_K, E0, delta, M_MAX, NROOTS)
    wall_krylov = time.perf_counter()-t_krylov

    results_m = {}
    for m in range(M_MAX + 1):
        ev = ev_all_m[m]
        dE = [(ev[k]-e_fci[k])*1000 for k in range(min(NROOTS, len(ev)))]
        ddE_mgs = dE[0] - dE_mgs_ref[0]
        results_m[m] = {'E': [float(e) for e in ev[:NROOTS]],
                         'dE_mH': dE[:NROOTS]}

        s0_str = f"dE0={dE[0]:+8.1f} mH"
        if m == 0:
            s0_str += f"  (vs MGS ref: Δ={ddE_mgs:+.1f})"
        else:
            ddE_prev = dE[0] - results_m[m-1]['dE_mH'][0]
            s0_str += f"  (Δm={ddE_prev:+.1f})"
        print(f"    m={m}: {s0_str}", flush=True)

        if m <= 1:  # print first few roots for small m
            for k in range(1, min(4, NROOTS)):
                print(f"      S{k}: dE={dE[k]:+8.1f} mH", flush=True)

    print(f"  Krylov wall: {wall_krylov:.0f}s (M_MAX={M_MAX}, d={d_svd})", flush=True)

    # Summary
    ex_de = [abs(results_m[0]['dE_mH'][k]) for k in range(1,min(NROOTS,len(results_m[0]['dE_mH'])))]
    print(f"  Summary P={p_target}: d_mgs={d_mgs} d_svd={d_svd} "
          f"dE0(m=0)={results_m[0]['dE_mH'][0]:+.1f} "
          f"dE0(m={M_MAX})={results_m[M_MAX]['dE_mH'][0]:+.1f} mH  "
          f"max|dE_ex|={max(ex_de):.0f} mH\n", flush=True)

    sigs = [float(s) for s in sigma_svd[:min(20,len(sigma_svd))]]
    smax = sigs[0] if sigs else 0

    return {
        'P': p_target, 'N': N, 'iter': it_num,
        'd_mgs': d_mgs, 'd_svd': d_svd,
        'E0': float(E0), 'dE0_bare_mH': float(dE0_bare),
        'sigma_max': smax,
        'sigma_ratios': [s/smax if smax>0 else 0 for s in sigs],
        'mgs_ref_dE0_mH': dE_mgs_ref[0],
        'results': results_m,
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
print(f"Phase A v4 Complete: {time.perf_counter()-total_t0:.0f}s")
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
        print(f"{r['results'][m]['dE_mH'][0]:>+10.1f} ", end="")
    print(f"{r['mgs_ref_dE0_mH']:>+10.1f}")

# Save
outdir = os.path.join(PROJECT_ROOT, 'checkpoints_phaseA')
os.makedirs(outdir, exist_ok=True)
with open(f'{outdir}/phaseA_v4_m{M_MAX}_svd{SVD_THR}_{TAG}.json','w') as f:
    json.dump({
        'config': {'cas':N_ACT,'n_core':N_CORE,'P':P_CHECKPOINTS,
                   'svd_threshold':SVD_THR,'m_max':M_MAX,'M':M,
                   'e_fci':e_fci,'tag':TAG,'b_threshold':B_THR},
        'results': {str(k):v for k,v in all_results.items()},
    }, f, indent=2)
print(f"\nSaved: {outdir}/phaseA_v4_m{M_MAX}_svd{SVD_THR}_{TAG}.json")
print("Done.")
