#!/usr/bin/env python3
"""
Phase A v8 — Matrix-Free Krylov matching kdci_dense.py exactly

Key: T = A²·X in BOTH build_basis and propagate_basis, SVD after each.

build_basis:  V = A·H_QP (stream), T = A·V = A²·H_QP → SVD → U
propagate:    residual = H·b_k - D·b_k, X = A·residual,
              T = A·X = A²·residual → SVD → MGS
H_KK/H_PK:    from kdci_sparse (sparse projected blocks)

Usage:
    python phaseA_cas10_v8.py --P 400,600,800,1000,2000,4000 --m-max 3
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
from pyscf.fci import cistring, direct_spin1, spin_op

args = argparse.ArgumentParser()
args.add_argument('--P', type=str, default='400,600,800,1000,2000,4000')
args.add_argument('--m-max', type=int, default=3)
args.add_argument('--svd-threshold', type=float, default=1e-3)
args.add_argument('--batch', type=int, default=200)
args.add_argument('--tag', type=str, default='v10sacis')
args = args.parse_args()

P_CHECKPOINTS = sorted([int(x) for x in args.P.split(',')])
M_MAX = args.m_max; SVD_THR = args.svd_threshold; BATCH = args.batch
TAG = args.tag; P_MAX = max(P_CHECKPOINTS)

N_ACT = 10; N_CORE = 2; NROOTS = 6; R = 1.1; ne = (5, 5)
print("=" * 70)
print(f"Phase A v8 — CAS({N_ACT},{sum(ne)})  Matrix-Free Krylov m=0..{M_MAX}")
print(f"N2/cc-pVDZ R={R}  checkpoints={P_CHECKPOINTS}")
print(f"SVD thr={SVD_THR}  m_max={M_MAX}  batch={BATCH}")
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
na, nb = len(as_), len(bs_); M_all = na*nb
aidx = {int(s): i for i, s in enumerate(as_)}
bidx = {int(s): i for i, s in enumerate(bs_)}
q_idx = QSpaceIndex(as_, bs_, N_ACT, ne, h1a, era)
backend = KDCIBackend(q_idx); kdci_sparse = KDCISparse(q_idx)
hdiag = q_idx.hdiag

print("  FCI reference...", flush=True)
ef, _ = direct_spin1.FCI().kernel(h1a, era, N_ACT, ne, nroots=NROOTS, verbose=0)
e_fci = [float(e) for e in np.atleast_1d(ef)[:NROOTS]]
for i in range(NROOTS):
    exc = f"  ({(e_fci[i]-e_fci[0])*1000:.1f} mH)" if i > 0 else ""
    print(f"    S{i}: {e_fci[i]:.12f} Ha{exc}")

h2_4d = ao2mo.restore('s1', era, N_ACT).reshape(N_ACT, N_ACT, N_ACT, N_ACT)
ham = Hamiltonian(h1=h1a, h2=h2_4d, E_nuc=0.0, E_HF=0.0)
hf_a, hf_b = hf_determinant(*ne)
ao = bit_positions(hf_a); bo = bit_positions(hf_b)
av, bv = [p for p in range(N_ACT) if p not in ao], [p for p in range(N_ACT) if p not in bo]
E_HF = ham.matrix_element((hf_a, hf_b), (hf_a, hf_b))
full_dets = [(int(a), int(b)) for a in as_ for b in bs_]
det_to_full = {d: i for i, d in enumerate(full_dets)}
print(f"  CAS({N_ACT},{sum(ne)}): M={M_all:,}  ({time.perf_counter()-t0:.0f}s)\n")

def s2_of_pvec(cvec_p, p_dets):
    """<S^2> of a P-space eigenvector (embed into full na x nb civec)."""
    full = np.zeros((na, nb))
    for l, d in enumerate(p_dets):
        full[aidx[int(d[0])], bidx[int(d[1])]] += cvec_p[l]
    nrm = np.linalg.norm(full)
    if nrm > 0: full /= nrm
    return spin_op.spin_square(full, N_ACT, ne)[0]

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

P_INIT = 200
scores = gen_hfpt2_scores()
init_dets = [(hf_a, hf_b)]
# FIX (2026-07-13): force ALL single excitations into the seed.
# Brillouin's theorem zeroes <HF|H|singles>, so HFPT2 never selects singles,
# but excited states (esp. triplets) are single-excitation dominated. Without
# singles in the seed the excited roots of H_PP are wrong (dE ~+600 mH).
singles = []
for i in ao:
    for a in av: singles.append((hf_a ^ (1 << i) | (1 << a), hf_b))
for i in bo:
    for a in bv: singles.append((hf_a, hf_b ^ (1 << i) | (1 << a)))
for d in singles:
    if d not in init_dets: init_dets.append(d)
n_singles = len(init_dets) - 1
for d, _ in scores:  # fill remainder with top HFPT2 doubles
    if len(init_dets) >= P_INIT: break
    if d not in init_dets: init_dets.append(d)
n_doubles = len(init_dets) - 1 - n_singles
print(f"  Seed P={len(init_dets)} (HF + {n_singles} singles + {n_doubles} HFPT2 doubles)\n")

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
# Matrix-Free build_basis: T = A²·H_QP (matching kdci_dense.py)
# ═══════════════════════════════════════════════════════════════
def build_basis_mf(p_dets, E0, tag=""):
    """T = A² · H_QP columns, stream to memmap, SVD → U.

    Exactly matches KDCIBackend.build_basis:
      L0 = H_QP * A_q (column-wise)
      T  = A_q * L0 = A² * H_QP
      SVD(T) → U_trunc
    """
    N = len(p_dets)
    A_q = np.where(np.abs(E0 - hdiag) > 1e-10, 1.0 / (E0 - hdiag), 0.0)
    A_half = np.sqrt(np.abs(A_q))

    tmpdir = f'{PROJECT_ROOT}/tmp'; os.makedirs(tmpdir, exist_ok=True)
    fpath = f'{tmpdir}/phaseA_v8_L0_N{N}_{tag}.dat'
    T = np.memmap(fpath, dtype='float64', mode='w+', shape=(M_all, N))

    p_idx_set = set()
    for pa, pb in p_dets:
        idx = q_idx.flat_index(int(pa), int(pb))
        if idx is not None and idx >= 0: p_idx_set.add(idx)

    t0 = time.perf_counter()
    print(f"    [build_basis] T=A²·H_QP, N={N} cols → memmap...", flush=True)
    for p in range(N):
        pa, pb = int(p_dets[p][0]), int(p_dets[p][1])
        ia = q_idx._alpha_idx.get(pa); ib = q_idx._beta_idx.get(pb)
        if ia is None or ib is None: continue
        ci_unit = np.zeros((na, nb)); ci_unit[ia, ib] = 1.0
        sigma_flat = backend.sigma_full(ci_unit).reshape(-1)
        for q in p_idx_set: sigma_flat[q] = 0.0
        # L0 = A * H_QP
        T[:, p] = A_half * sigma_flat  # T = A^(1/2) = A² * H_QP
        
        if (p+1) % max(1, N//5) == 0:
            print(f"      col {p+1}/{N} ({time.perf_counter()-t0:.0f}s)", flush=True)
    T.flush()

    # SVD
    t_svd = time.perf_counter()
    print(f"    SVD({M_all},{N})...", flush=True)
    U, sigma, Vt = svd(T, full_matrices=False)
    smax = sigma[0] if len(sigma)>0 else 0
    mask = sigma >= SVD_THR * max(1.0, smax)
    d = int(np.sum(mask))
    U_ret = U[:, mask]; sig_ret = sigma[mask]
    e = time.perf_counter()-t_svd
    ratios = ", ".join(f"{s/smax:.4f}" for s in sigma[:min(8,len(sigma))])
    print(f"    SVD done: {e:.0f}s, {N}→d={d} (σ/σ₁=[{ratios}])", flush=True)

    try: del T; gc.collect(); os.unlink(fpath)
    except: pass
    return U_ret, d, A_q


# ═══════════════════════════════════════════════════════════════
# Matrix-Free propagate_basis: matching kdci_dense.propagate_basis
# ═══════════════════════════════════════════════════════════════
def propagate_basis_mf(U_basis, A_q, E0, p_idx_set, tag=""):
    """Propagate: X = A·H_O'·U, T = A·X = A²·H_O'·U → SVD → MGS.

    Exactly matches KDCIBackend.propagate_basis:
      residual = H·b_k - D·b_k  (H_O')
      x_k = A * residual
      T = A * X = A² * H_O' * U
      SVD(T), MGS against existing basis
    """
    A_half = np.sqrt(np.abs(A_q))
    M_dim, d_old = U_basis.shape
    if d_old == 0: return U_basis.copy(), d_old

    tmpdir = f'{PROJECT_ROOT}/tmp'; os.makedirs(tmpdir, exist_ok=True)
    fpath = f'{tmpdir}/phaseA_v8_prop_d{d_old}_{tag}.dat'
    T = np.memmap(fpath, dtype='float64', mode='w+', shape=(M_dim, d_old))

    t0 = time.perf_counter()
    print(f"    [propagate] d={d_old}, T=A²·H_O'·U → memmap...", flush=True)
    for k in range(d_old):
        b_k = U_basis[:, k]
        sigma_k = backend.sigma_full(b_k.reshape(na, nb)).reshape(-1)
        # H_O' * b_k = H*b_k - D*b_k
        residual = sigma_k - hdiag * b_k
        for q in p_idx_set: residual[q] = 0.0  # CRITICAL: zero P-space
        # X = A * residual
        T[:, k] = A_half * residual  # T = A^(1/2) = A² * H_O' * b_k
        
        if (k+1) % max(1, d_old//5) == 0:
            print(f"      col {k+1}/{d_old} ({time.perf_counter()-t0:.0f}s)", flush=True)
    T.flush()

    # SVD
    t_svd = time.perf_counter()
    print(f"    SVD({M_dim},{d_old})...", flush=True)
    U_svd, sigma, Vt = svd(T, full_matrices=False)
    smax = sigma[0] if len(sigma)>0 else 0
    mask = sigma >= SVD_THR * max(1.0, smax)
    n_keep = int(np.sum(mask))
    U_trunc = U_svd[:, mask]
    e = time.perf_counter()-t_svd
    print(f"    SVD done: {e:.0f}s, {d_old}→{n_keep} kept", flush=True)

    try: del T; gc.collect(); os.unlink(fpath)
    except: pass

    # MGS against existing basis
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
# Full Krylov pipeline (matrix-free): m=0..M_MAX
# ═══════════════════════════════════════════════════════════════
def krylov_mf_pipeline(H_PP, p_dets, E0, p_idx_set_mf, tag=""):
    """build_basis_mf + propagate_basis_mf, H_eff at each m."""
    results = []

    # m=0: build_basis_mf
    U_0, d_0, A_q = build_basis_mf(p_dets, E0, tag)

    def build_blocks(U_basis, p_dets):
        d = U_basis.shape[1]; Np = len(p_dets)
        if d == 0: return np.zeros((0,0)), np.zeros((Np,0))
        t0 = time.perf_counter()
        print(f"    [blocks] d={d}...", flush=True)
        H_KK = np.zeros((d,d)); H_PK = np.zeros((Np,d))
        p_flat = kdci_sparse.q_idx.p_indices(p_dets)
        p_valid = p_flat >= 0; p_f = p_flat[p_valid]
        for k in range(d):
            ci_k = U_basis[:,k].reshape(na,nb)
            sk = backend.sigma_full(ci_k).reshape(-1)
            H_KK[:,k] = U_basis.T @ sk
            H_PK[p_valid,k] = sk[p_f]
            if (k+1) % max(1,d//5)==0:
                print(f"      {k+1}/{d} ({time.perf_counter()-t0:.0f}s)", flush=True)
        H_KK = 0.5*(H_KK + H_KK.T)
        print(f"    [blocks] done: {time.perf_counter()-t0:.0f}s", flush=True)
        return H_KK, H_PK

    U_m = U_0; d_m = d_0
    H_KK, H_PK = build_blocks(U_m, p_dets)
    ev = diagonalize_effective_H(
        build_effective_H(H_PP, H_PK, H_KK, E0, delta=0.0),
        n_states=NROOTS)[0]
    dE = [(ev[k]-e_fci[k])*1000 for k in range(min(NROOTS, len(ev)))]
    results.append({'d': d_m, 'dE': dE, 'U': U_m})
    print(f"    m=0: d={d_m}, dE0={dE[0]:+.3f} mH", flush=True)
    for k in range(1, min(4, NROOTS)):
        print(f"      S{k}: dE={dE[k]:+.1f} mH", flush=True)

    for m in range(1, M_MAX+1):
        U_m, d_m = propagate_basis_mf(U_m, A_q, E0, p_idx_set_mf, f"{tag}_m{m}")
        if d_m == results[-1]['d']:
            print(f"    m={m}: no new directions, stopping", flush=True)
            results.append(results[-1])
            break

        H_KK, H_PK = build_blocks(U_m, p_dets)
        ev = diagonalize_effective_H(
            build_effective_H(H_PP, H_PK, H_KK, E0, delta=0.0),
            n_states=NROOTS)[0]
        dE = [(ev[k]-e_fci[k])*1000 for k in range(min(NROOTS, len(ev)))]
        ddE = dE[0] - results[-1]['dE'][0]
        results.append({'d': d_m, 'dE': dE, 'U': U_m})
        print(f"    m={m}: d={d_m}, dE0={dE[0]:+.3f} mH (Δ={ddE:+.1f})", flush=True)

    return results, A_q


# ═══════════════════════════════════════════════════════════════
# Checkpoint evaluation
# ═══════════════════════════════════════════════════════════════
def eval_checkpoint(p_dets, p_full_idx, H_PP_sub, p_target, it_num):
    N = len(p_dets)
    p_idx_set = set()
    for pa, pb in p_dets:
        idx = q_idx.flat_index(int(pa), int(pb))
        if idx is not None and idx >= 0: p_idx_set.add(idx)
    E0_vals, E0_vecs = eigh(H_PP_sub); E0 = E0_vals[0]
    dE0_bare = (E0 - e_fci[0])*1000
    nlab = min(NROOTS, H_PP_sub.shape[0])
    s2s = [float(s2_of_pvec(E0_vecs[:, k], p_dets)) for k in range(nlab)]
    bares = [(E0_vals[k]-e_fci[k])*1000 for k in range(nlab)]
    print(f"  P={N}, E0={E0:.8f}, dE0(bare)={dE0_bare:+.3f} mH", flush=True)
    print("    [H_PP bare] " + "  ".join(
        f"S{k}:dE={bares[k]:+.0f},<S2>={s2s[k]:.2f}" for k in range(nlab)), flush=True)

    tag = f"P{p_target}_i{it_num}"
    kr_results, A_q = krylov_mf_pipeline(H_PP_sub, p_dets, E0, p_idx_set, tag)

    ex_de = [abs(kr_results[0]['dE'][k]) for k in range(1,min(NROOTS,len(kr_results[0]['dE'])))]
    m_last = min(M_MAX, len(kr_results)-1)
    print(f"  Summary P={p_target}: d(m=0)={kr_results[0]['d']} "
          f"dE0(m=0)={kr_results[0]['dE'][0]:+.1f} "
          f"dE0(m={m_last})={kr_results[m_last]['dE'][0]:+.1f} mH  "
          f"max|dE_ex|={max(ex_de):.0f} mH\n", flush=True)

    return {
        'P': p_target, 'N': N, 'iter': it_num,
        'E0': float(E0), 'dE0_bare_mH': float(dE0_bare),
        's2': s2s, 'dE_bare_mH': bares,
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

    sigmas = []
    ns = min(len(SCORING_ROOTS), N_p)
    for sk in range(ns):
        k = SCORING_ROOTS[sk]
        vec = np.zeros(M_all)
        for li, gi in enumerate(p_full_idx): vec[gi] = C_P[li, k]
        sigma_k = backend.sigma(vec)
        sigmas.append((E_P[k], sigma_k))

    weights = np.zeros(M_all)
    for E_ref, sk in sigmas:
        abs_s = np.abs(sk)
        for qi in range(M_all):
            if qi in p_set: continue
            c2 = abs_s[qi]**2
            if c2 < 1e-24: continue
            weights[qi] += c2 / max(abs(E_ref-hdiag[qi]), 1e-8)

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

    dE0 = (E0_cur-e_fci[0])*1000
    print(f"{it:>4} {N_p:>6} {E0_cur:>14.8f} {dE0:>+10.3f} {max_w:>10.3e} "
          f"{time.perf_counter()-t_it:>8.1f}", flush=True)
    it += 1

    for pt in P_CHECKPOINTS:
        if N_p >= pt and pt not in all_results:
            print(f"\n  ══ Checkpoint P={pt} ══", flush=True)
            all_results[pt] = eval_checkpoint(p_dets[:pt], p_full_idx[:pt], H_PP[:pt,:pt], pt, it)

# ── Final Summary ──
print(f"\n{'='*70}")
print(f"Phase A v8 Complete: {time.perf_counter()-total_t0:.0f}s")
print(f"{'='*70}")

print(f"\n{'P':>6} {'N':>6} ", end="")
for m in range(M_MAX+1):
    print(f"{'m='+str(m):>10} ", end="")
print(f"{'d(m=0)':>7}")
print("-"*(30+12*(M_MAX+1)))
for pt in P_CHECKPOINTS:
    r = all_results[pt]
    print(f"{pt:>6} {r['N']:>6} ", end="")
    for m in range(M_MAX+1):
        if m < len(r['krylov']):
            print(f"{r['krylov'][m]['dE_mH'][0]:>+10.1f} ", end="")
        else:
            print(f"{'---':>10} ", end="")
    print(f"{r['krylov'][0]['d']:>7}")

# ── Per-root (state-average) summary at final m, with <S^2> labels ──
print(f"\nPer-root dE (mH) at m={M_MAX}  [FCI: " +
      " ".join(f"S{k}={e_fci[k]:.4f}" for k in range(NROOTS)) + "]")
print(f"{'P':>6} " + " ".join(f"{'S'+str(k):>9}" for k in range(NROOTS)))
print("-"*(7+10*NROOTS))
for pt in P_CHECKPOINTS:
    r = all_results[pt]
    m_last = min(M_MAX, len(r['krylov'])-1)
    de = r['krylov'][m_last]['dE_mH']
    print(f"{pt:>6} " + " ".join(f"{de[k]:>+9.1f}" for k in range(min(NROOTS,len(de)))))
    s2 = r.get('s2', [])
    print("  <S2> " + " ".join(f"{(s2[k] if k<len(s2) else -1):>9.2f}" for k in range(NROOTS)))

# Save
outdir = os.path.join(PROJECT_ROOT, 'checkpoints_phaseA')
os.makedirs(outdir, exist_ok=True)
with open(f'{outdir}/phaseA_v8_m{M_MAX}_svd{SVD_THR}_{TAG}.json','w') as f:
    json.dump({
        'config': {'cas':N_ACT,'n_core':N_CORE,'P':P_CHECKPOINTS,
                   'm_max':M_MAX,'svd_threshold':SVD_THR,'M':M_all,
                   'e_fci':e_fci,'tag':TAG},
        'results': {str(k):v for k,v in all_results.items()},
    }, f, indent=2)
print(f"\nSaved: {outdir}/phaseA_v8_m{M_MAX}_svd{SVD_THR}_{TAG}.json")
print("Done.")
