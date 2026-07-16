#!/usr/bin/env python3
"""
Phase A v2 — CAS(10,10) Matrix-Free + Iterative-P + MGS + SVD + Krylov

Pipeline per P-checkpoint (P=200,500,1000,2000):
  1. Iterative P-space selection (σ-vector scoring, from step1)
  2. Build H_PP
  3. Matrix-free: raw Krylov vectors v_p[q] = A_q · ⟨q|H|p_i⟩ (NO MGS yet)
  4. MGS: orthonormalize raw vectors, detect linear dependence → d_MGS
  5. SVD: weighted SVD on raw vectors → truncated basis → d_SVD ≤ d_MGS
  6. Build projected blocks (H_QQ_tilde, H_PQ_tilde) from SVD-compressed basis
  7. m=0 effective H → dE vs FCI
  8. m=1 Krylov propagation (one B-step)
  9. Compare d_MGS vs d_SVD (key metric for A2)

Usage:
    python phaseA_cas10_v2.py --P 200,500,1000,2000 --svd-threshold 1e-3 --m-max 1
"""
import sys, os, time, json, argparse, itertools, gc
import numpy as np
from numpy.linalg import eigh, svd, norm

PROJECT_ROOT = '/data/home/wangcx/krylov-dci'
sys.path.insert(0, PROJECT_ROOT)

from src_mf import QSpaceIndex, KDCIBackend
from src.effective_h import build_effective_H, diagonalize_effective_H
from src.determinants import hf_determinant, bit_positions
from src.hamiltonian import Hamiltonian
from pyscf import gto, scf, ao2mo
from pyscf.fci import cistring, direct_spin1

# ═══════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--P', type=str, default='200,500,1000,2000',
                   help='Comma-separated P-checkpoints')
    p.add_argument('--svd-threshold', type=float, default=1e-3,
                   help='SVD truncation (fraction of sigma_max)')
    p.add_argument('--m-max', type=int, default=1)
    p.add_argument('--batch', type=int, default=200,
                   help='Iterative expansion batch size')
    p.add_argument('--tag', type=str, default='v2')
    return p.parse_args()

args = parse_args()
P_CHECKPOINTS = sorted([int(x) for x in args.P.split(',')])
SVD_THR = args.svd_threshold
M_MAX = args.m_max
BATCH = args.batch
TAG = args.tag
P_MAX = max(P_CHECKPOINTS)

# ═══════════════════════════════════════════════════════════════
# System: N2/cc-pVDZ CAS(10,10)
# ═══════════════════════════════════════════════════════════════
N_ACT = 10; N_CORE = 2; NROOTS = 6; R = 1.1; ne = (5, 5)

print("=" * 70)
print(f"Phase A v2 — CAS({N_ACT},{sum(ne)})  Iterative-P + MGS + SVD + Krylov")
print(f"N2/cc-pVDZ R={R}  nroots={NROOTS}  checkpoints={P_CHECKPOINTS}")
print(f"svd_thr={SVD_THR}  m_max={M_MAX}  batch={BATCH}")
print("=" * 70, flush=True)

t_build = time.perf_counter()
mol = gto.M(atom=f'N 0 0 0; N 0 0 {R}', basis='cc-pVDZ', verbose=0)
mf = scf.RHF(mol).run(verbose=0)
n_act_orbs = list(range(N_CORE, N_CORE + N_ACT))
norb = mf.mo_coeff.shape[1]
h1e_mo = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
e2 = ao2mo.full(mol.intor('int2e'), mf.mo_coeff, compact=False)
eri_mo = e2.reshape(norb, norb, norb, norb)
h1a = h1e_mo[np.ix_(n_act_orbs, n_act_orbs)]
era = eri_mo[np.ix_(n_act_orbs, n_act_orbs, n_act_orbs, n_act_orbs)]
as_ = cistring.gen_strings4orblist(range(N_ACT), ne[0])
bs_ = cistring.gen_strings4orblist(range(N_ACT), ne[1])
na, nb = len(as_), len(bs_); M = na * nb
q_idx = QSpaceIndex(as_, bs_, N_ACT, ne, h1a, era)
backend = KDCIBackend(q_idx)
hdiag = q_idx.hdiag

# FCI reference
print("  FCI reference...", flush=True)
ef, _ = direct_spin1.FCI().kernel(h1a, era, N_ACT, ne, nroots=NROOTS, verbose=0)
e_fci = [float(e) for e in np.atleast_1d(ef)[:NROOTS]]
for i in range(NROOTS):
    exc = f"  ({(e_fci[i]-e_fci[0])*1000:.1f} mH)" if i > 0 else "  (ground)"
    print(f"    S{i}: {e_fci[i]:.12f} Ha{exc}")

# Hamiltonian & HF reference
h2_4d = ao2mo.restore('s1', era, N_ACT).reshape(N_ACT, N_ACT, N_ACT, N_ACT)
ham = Hamiltonian(h1=h1a, h2=h2_4d, E_nuc=0.0, E_HF=0.0)
hf_a, hf_b = hf_determinant(*ne)
ao = bit_positions(hf_a); bo = bit_positions(hf_b)
av = [p for p in range(N_ACT) if p not in ao]
bv = [p for p in range(N_ACT) if p not in bo]
E_HF = ham.matrix_element((hf_a, hf_b), (hf_a, hf_b))

# Full determinant list (Q = full CAS)
full_dets = [(int(a), int(b)) for a in as_ for b in bs_]
det_to_full = {d: i for i, d in enumerate(full_dets)}
assert len(full_dets) == M

print(f"  CAS({N_ACT},{sum(ne)}): M={M:,}  ({time.perf_counter()-t_build:.0f}s)\n")

# ═══════════════════════════════════════════════════════════════
# σ = H · v  (PySCF C-level)
# ═══════════════════════════════════════════════════════════════
def sigma_flat(v_dense):
    """σ = H|v⟩ via KDCIBackend.sigma_full (proven)."""
    ci_mat = v_dense.reshape(na, nb)
    sigma_mat = backend.sigma_full(ci_mat)
    return sigma_mat.reshape(-1)


# ═══════════════════════════════════════════════════════════════
# HFPT2 initial P
# ═══════════════════════════════════════════════════════════════
def gen_hfpt2_scores():
    scores = []
    for i in ao:
        for a in av:
            d = (hf_a ^ (1 << i) | (1 << a), hf_b)
            hij = ham.matrix_element(d, (hf_a, hf_b))
            den = E_HF - ham.matrix_element(d, d)
            if abs(den) > 1e-12: scores.append((d, -hij*hij/den))
    for i in bo:
        for a in bv:
            d = (hf_a, hf_b ^ (1 << i) | (1 << a))
            hij = ham.matrix_element(d, (hf_a, hf_b))
            den = E_HF - ham.matrix_element(d, d)
            if abs(den) > 1e-12: scores.append((d, -hij*hij/den))
    for i1, i2 in itertools.combinations(ao, 2):
        for a1, a2 in itertools.combinations(av, 2):
            d = (hf_a ^ (1 << i1) ^ (1 << i2) | (1 << a1) | (1 << a2), hf_b)
            hij = ham.matrix_element(d, (hf_a, hf_b))
            den = E_HF - ham.matrix_element(d, d)
            if abs(den) > 1e-12: scores.append((d, -hij*hij/den))
    for i1, i2 in itertools.combinations(bo, 2):
        for a1, a2 in itertools.combinations(bv, 2):
            d = (hf_a, hf_b ^ (1 << i1) ^ (1 << i2) | (1 << a1) | (1 << a2))
            hij = ham.matrix_element(d, (hf_a, hf_b))
            den = E_HF - ham.matrix_element(d, d)
            if abs(den) > 1e-12: scores.append((d, -hij*hij/den))
    for i in ao:
        for j in bo:
            for a in av:
                for b in bv:
                    d = (hf_a ^ (1 << i) | (1 << a), hf_b ^ (1 << j) | (1 << b))
                    hij = ham.matrix_element(d, (hf_a, hf_b))
                    den = E_HF - ham.matrix_element(d, d)
                    if abs(den) > 1e-12: scores.append((d, -hij*hij/den))
    scores.sort(key=lambda x: x[1])
    return scores

scores = gen_hfpt2_scores()
P_INIT = P_CHECKPOINTS[0]
init_dets = [(hf_a, hf_b)]
for det, _ in scores:
    if det not in init_dets: init_dets.append(det)
    if len(init_dets) >= P_INIT: break
print(f"  HFPT2 initial P={len(init_dets)}\n")


# ═══════════════════════════════════════════════════════════════
# H_PP construction
# ═══════════════════════════════════════════════════════════════
def build_hpp(dets):
    n = len(dets)
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            v = ham.matrix_element(dets[i], dets[j])
            H[i, j] = v; H[j, i] = v
    return H

def extend_hpp(H_old, old_dets, new_dets):
    N_old = len(old_dets); n_add = len(new_dets)
    H_new = np.zeros((N_old + n_add, N_old + n_add))
    H_new[:N_old, :N_old] = H_old
    for i_local, det_new in enumerate(new_dets):
        row = N_old + i_local
        for j in range(N_old):
            val = ham.matrix_element(det_new, old_dets[j])
            H_new[row, j] = val; H_new[j, row] = val
        for j_local in range(i_local + 1):
            col = N_old + j_local
            val = ham.matrix_element(det_new, new_dets[j_local])
            H_new[row, col] = val; H_new[col, row] = val
    return H_new


# ═══════════════════════════════════════════════════════════════
# Step 3: Build raw Krylov vectors (matrix-free, NO MGS)
# ═══════════════════════════════════════════════════════════════
def build_raw_vectors(p_dets, p_full_indices, E0, tag=""):
    """Build V_raw[q, p] = A_q · ⟨q|H|p_i⟩ for all P dets. Memmap to disk."""
    N = len(p_dets)
    denom = E0 - hdiag
    mask = np.abs(denom) > 1e-10
    A_q = np.zeros(M); A_q[mask] = 1.0 / denom[mask]
    p_set = set(pi for pi in p_full_indices if pi >= 0)

    tmpdir = f'{PROJECT_ROOT}/tmp'; os.makedirs(tmpdir, exist_ok=True)
    fpath = f'{tmpdir}/phaseA_v2_raw_N{N}_{tag}.dat'
    V_raw = np.memmap(fpath, dtype='float64', mode='w+', shape=(M, N))

    t0 = time.perf_counter()
    print(f"    [raw] Building {N} raw vectors → {fpath} ...", flush=True)
    for p in range(N):
        idx = q_idx.flat_index(int(p_dets[p][0]), int(p_dets[p][1]))
        if idx is None or idx < 0: continue
        unit = np.zeros(M); unit[idx] = 1.0
        sp = sigma_flat(unit)
        for q in p_set: sp[q] = 0.0  # zero P-space
        V_raw[:, p] = A_q * sp
        if (p+1) % max(1, N//5) == 0:
            e = time.perf_counter()-t0
            print(f"      col {p+1}/{N}  ({e:.0f}s, ETA {e/(p+1)*(N-p-1):.0f}s)", flush=True)
    V_raw.flush()
    e = time.perf_counter()-t0
    print(f"    [raw] done: {e:.0f}s ({e/N:.2f}s/col)", flush=True)
    return V_raw, A_q, fpath


# ═══════════════════════════════════════════════════════════════
# Step 4: MGS on raw vectors → detect linear dependence
# ═══════════════════════════════════════════════════════════════
def mgs_orthonormalize(V_raw, lindep_thr=1e-10):
    """MGS on columns of V_raw. Returns orthonormal basis + retained indices."""
    M_dim, K = V_raw.shape
    retained = []
    for k in range(K):
        v = V_raw[:, k].copy()
        for r in retained:
            v -= np.dot(V_raw[:, r], v) * V_raw[:, r]
        nrm = norm(v)
        if nrm > lindep_thr:
            V_raw[:, k] = v / nrm  # in-place normalize
            retained.append(k)
    d_mgs = len(retained)
    return V_raw[:, retained], d_mgs


# ═══════════════════════════════════════════════════════════════
# Step 5: Weighted SVD on raw vectors → compress
# ═══════════════════════════════════════════════════════════════
def svd_compress_raw(V_raw, A_q, threshold):
    """T = A^{1/2} · V_raw → SVD → truncated U. Returns (U_comp, sigma, d_svd)."""
    M_dim, K = V_raw.shape
    t0 = time.perf_counter()
    sqrt_A = np.sqrt(np.abs(A_q))
    T = V_raw * sqrt_A[:, np.newaxis]
    print(f"    [SVD] SVD({M_dim}, {K})...", flush=True)
    U, sigma, Vt = svd(T, full_matrices=False)
    sigma_max = sigma[0] if len(sigma) > 0 else 0.0
    if sigma_max < 1e-15: return np.zeros((M_dim, 0)), np.array([]), 0
    mask = sigma >= threshold * sigma_max
    d_svd = np.sum(mask)
    U_r = U[:, mask]; sig_r = sigma[mask]
    e = time.perf_counter()-t0
    rstr = ", ".join(f"{s/sigma_max:.4f}" for s in sigma[:min(8, len(sigma))])
    print(f"    [SVD] done: {e:.0f}s, K={K} → d_svd={d_svd} (σ/σ₁ = [{rstr}])", flush=True)
    return U_r, sig_r, d_svd


# ═══════════════════════════════════════════════════════════════
# Step 6: Projected blocks from SVD-compressed basis
# ═══════════════════════════════════════════════════════════════
def build_projected_blocks(U_comp, p_full_indices):
    """H_QQ_tilde[j,k]=⟨w_k|H|w_j⟩, H_PQ_tilde[p,k]=⟨p|H|w_k⟩."""
    d = U_comp.shape[1]; N = len(p_full_indices)
    if d == 0: return np.zeros((0, 0)), np.zeros((N, 0))
    t0 = time.perf_counter()
    print(f"    [proj] Projecting d={d} basis vectors...", flush=True)
    H_QQ_t = np.zeros((d, d)); H_PQ_t = np.zeros((N, d))
    p_idx = np.array(p_full_indices); p_valid = p_idx >= 0; p_flat = p_idx[p_valid]
    for k in range(d):
        sk = sigma_flat(U_comp[:, k])
        H_QQ_t[:, k] = U_comp.T @ sk
        H_PQ_t[p_valid, k] = sk[p_flat]
        if (k+1) % max(1, d//5) == 0:
            print(f"      basis {k+1}/{d} ({time.perf_counter()-t0:.0f}s)", flush=True)
    H_QQ_t = 0.5*(H_QQ_t + H_QQ_t.T)
    e = time.perf_counter()-t0
    print(f"    [proj] done: {e:.0f}s", flush=True)
    return H_QQ_t, H_PQ_t


# ═══════════════════════════════════════════════════════════════
# Step 7-8: m=0 and m=1 H^eff
# ═══════════════════════════════════════════════════════════════
def krylov_m0(H_PP, H_PQ_t, H_QQ_t, E0):
    H_eff = build_effective_H(H_PP, H_PQ_t, H_QQ_t, E0, delta=0.0)
    ev, _ = diagonalize_effective_H(H_eff, n_states=NROOTS)
    return ev


def krylov_m1(H_PP, H_PQ_t, H_QQ_t, E0, U_comp, p_full_indices):
    """One B-step: for each m=0 basis vector, apply H → A_q filter → MGS → expand."""
    d0 = U_comp.shape[1]
    if d0 == 0: return krylov_m0(H_PP, H_PQ_t, H_QQ_t, E0), 0
    t0 = time.perf_counter()
    denom = E0 - hdiag
    A_q = np.where(np.abs(denom) > 1e-10, 1.0/denom, 0.0)

    new_raw = np.zeros((M, d0))
    for k in range(d0):
        sk = sigma_flat(U_comp[:, k])
        new_raw[:, k] = A_q * sk

    # MGS against U_comp
    new_orth = np.zeros((M, d0)); retained = []
    for k in range(d0):
        v = new_raw[:, k].copy()
        for j in range(d0): v -= np.dot(U_comp[:, j], v) * U_comp[:, j]
        for r in retained: v -= np.dot(new_orth[:, r], v) * new_orth[:, r]
        nrm = norm(v)
        if nrm > 1e-10:
            new_orth[:, k] = v/nrm; retained.append(k)
    d1 = len(retained)
    if d1 == 0:
        print(f"    [m=1] all linear dependent → same as m=0", flush=True)
        return krylov_m0(H_PP, H_PQ_t, H_QQ_t, E0), 0

    U_exp = np.hstack([U_comp, new_orth[:, :d1]])
    H_QQ_e, H_PQ_e = build_projected_blocks(U_exp, p_full_indices)
    ev = krylov_m0(H_PP, H_PQ_e, H_QQ_e, E0)
    print(f"    [m=1] d0={d0}+d1={d1} → d={d0+d1}, {time.perf_counter()-t0:.0f}s", flush=True)
    return ev, d1


# ═══════════════════════════════════════════════════════════════
# Iterative P-space expansion + checkpoint evaluation
# ═══════════════════════════════════════════════════════════════
def iterative_pspace_and_eval(init_dets):
    """Iteratively expand P while evaluating at checkpoints."""
    p_dets = list(init_dets)
    p_full_idx = [det_to_full[d] for d in p_dets]
    p_set = set(p_full_idx)
    H_PP = build_hpp(p_dets)
    N_p = len(p_dets)
    results = {}

    SCORING_ROOTS = list(range(min(NROOTS, 5)))  # use first 5 roots for scoring

    print(f"Iterative P: {N_p} → {P_MAX}")
    print(f"{'iter':>4} {'P':>6} {'E0_bare':>14} {'dE0_mH':>10} {'max_w':>10} {'wall':>8}")
    print("-" * 56, flush=True)

    it = 0
    while N_p < P_MAX:
        t_it = time.perf_counter()

        # Diagonalize H_PP
        E_P, C_P = eigh(H_PP)
        E0_cur = E_P[0]

        # σ-vector scoring for iterative expansion
        sigmas = []
        n_score = min(len(SCORING_ROOTS), N_p)
        for sk in range(n_score):
            k = SCORING_ROOTS[sk]
            vec_full = np.zeros(M)
            for li, gi in enumerate(p_full_idx): vec_full[gi] = C_P[li, k]
            sigma_k = sigma_flat(vec_full)
            sigmas.append((E_P[k], sigma_k))

        # Score Q-space
        weights = np.zeros(M)
        for E_ref, sk in sigmas:
            abs_s = np.abs(sk)
            for qi in range(M):
                if qi in p_set: continue
                c2 = abs_s[qi]**2
                if c2 < 1e-24: continue
                denom = max(abs(E_ref - hdiag[qi]), 1e-8)
                weights[qi] += c2 / denom

        candidates = [(qi, float(weights[qi])) for qi in range(M)
                      if qi not in p_set and weights[qi] > 0]
        candidates.sort(key=lambda x: x[1], reverse=True)
        n_add = min(BATCH, len(candidates))
        max_w = candidates[0][1] if candidates else 0.0

        new_gi = [c[0] for c in candidates[:n_add]]
        new_dets = [full_dets[qi] for qi in new_gi]
        H_PP = extend_hpp(H_PP, p_dets, new_dets)
        p_dets.extend(new_dets); p_full_idx.extend(new_gi); p_set.update(new_gi)
        N_p = len(p_dets)

        dE0 = (E0_cur - e_fci[0]) * 1000
        print(f"{it:>4} {N_p:>6} {E0_cur:>14.8f} {dE0:>+10.3f} {max_w:>10.3e} "
              f"{time.perf_counter()-t_it:>8.1f}", flush=True)
        it += 1

        # ── Checkpoint evaluation ──
        for p_target in P_CHECKPOINTS:
            if N_p >= p_target and p_target not in results:
                print(f"\n  ══ Checkpoint P={p_target} ══", flush=True)
                cp = evaluate_checkpoint(p_dets[:p_target], p_full_idx[:p_target],
                                         H_PP[:p_target, :p_target], p_target, it)
                results[p_target] = cp

    return results


def evaluate_checkpoint(p_dets, p_full_idx, H_PP_sub, p_target, it_num):
    """Full pipeline at a P checkpoint: raw → MGS → SVD → H^eff → m=0,1."""
    N = len(p_dets)
    E0_vals, _ = eigh(H_PP_sub); E0 = E0_vals[0]
    dE0_bare = (E0 - e_fci[0]) * 1000
    print(f"  P={N}, E0={E0:.8f}, dE0(bare)={dE0_bare:+.3f} mH", flush=True)

    # Step 3: Raw Krylov vectors
    tag_p = f"P{p_target}"
    V_raw, A_q, raw_path = build_raw_vectors(p_dets, p_full_idx, E0, tag_p)

    # Step 4: MGS
    t_mgs = time.perf_counter()
    V_raw_copy = V_raw.copy()  # MGS modifies in-place
    U_mgs, d_mgs = mgs_orthonormalize(V_raw_copy)
    wall_mgs = time.perf_counter() - t_mgs
    print(f"  [MGS] d_mgs={d_mgs}/{N}  ({wall_mgs:.0f}s)", flush=True)

    # Step 5: Weighted SVD on raw vectors
    U_svd, sigma_svd, d_svd = svd_compress_raw(V_raw, A_q, SVD_THR)

    # Cleanup raw vectors
    try: del V_raw, V_raw_copy; gc.collect(); os.unlink(raw_path)
    except: pass

    print(f"  [SVD] d_mgs={d_mgs} → d_svd={d_svd} (σ_thr={SVD_THR})", flush=True)

    # Step 6: Projected blocks from SVD basis
    H_QQ_t, H_PQ_t = build_projected_blocks(U_svd, p_full_idx)

    # Step 7: m=0
    print(f"\n  ── m=0 ──", flush=True)
    t_m0 = time.perf_counter()
    ev_m0 = krylov_m0(H_PP_sub, H_PQ_t, H_QQ_t, E0)
    wall_m0 = time.perf_counter() - t_m0
    dE_m0 = [(ev_m0[k] - e_fci[k]) * 1000 for k in range(min(NROOTS, len(ev_m0)))]
    for k in range(min(NROOTS, len(ev_m0))):
        print(f"    S{k}: E={ev_m0[k]:.12f}  dE={dE_m0[k]:+8.1f} mH", flush=True)

    # Step 8: m=1 (if requested)
    ev_m1, dE_m1, wall_m1, d1 = None, None, None, 0
    if M_MAX >= 1:
        print(f"\n  ── m=1 ──", flush=True)
        t_m1 = time.perf_counter()
        ev_m1, d1 = krylov_m1(H_PP_sub, H_PQ_t, H_QQ_t, E0, U_svd, p_full_idx)
        wall_m1 = time.perf_counter() - t_m1
        dE_m1 = [(ev_m1[k] - e_fci[k]) * 1000 for k in range(min(NROOTS, len(ev_m1)))]
        for k in range(min(NROOTS, len(ev_m1))):
            ddE = dE_m1[k] - dE_m0[k]
            print(f"    S{k}: E={ev_m1[k]:.12f}  dE={dE_m1[k]:+8.1f} mH  "
                  f"(Δ={ddE:+.1f})", flush=True)

    # Summary
    ex_de = [abs(dE_m0[k]) for k in range(1, min(NROOTS, len(ev_m0)))]
    print(f"\n  Summary P={p_target}: d_mgs={d_mgs} d_svd={d_svd} "
          f"dE0={dE_m0[0]:+.1f} mH  max|dE_ex|={max(ex_de):.0f} mH\n", flush=True)

    sigma_vals = [float(s) for s in sigma_svd[:min(20, len(sigma_svd))]]
    sigma_max = sigma_vals[0] if sigma_vals else 0.0

    return {
        'P': p_target, 'N': N, 'iter': it_num,
        'd_mgs': d_mgs, 'd_svd': d_svd,
        'E0': float(E0), 'dE0_bare_mH': float(dE0_bare),
        'sigma_max': sigma_max,
        'sigma_ratios': [s/sigma_max if sigma_max > 0 else 0 for s in sigma_vals],
        'wall_mgs_s': wall_mgs,
        'm0': {'E': [float(e) for e in ev_m0[:NROOTS]],
               'dE_mH': [float(d) for d in dE_m0[:NROOTS]],
               'wall_s': wall_m0},
        'm1': ({'E': [float(e) for e in ev_m1[:NROOTS]],
                'dE_mH': [float(d) for d in dE_m1[:NROOTS]],
                'wall_s': wall_m1, 'd1': d1}
               if ev_m1 is not None else None),
    }


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════
total_t0 = time.perf_counter()
all_results = iterative_pspace_and_eval(init_dets)

print(f"\n{'='*70}")
print(f"Phase A v2 Complete: {time.perf_counter()-total_t0:.0f}s")
print(f"{'='*70}")

# ── Final table ──
print(f"\n{'P':>6} {'N':>6} {'d_mgs':>7} {'d_svd':>7} {'dE0_m0':>10} "
      f"{'dE0_m1':>10} {'max|dE_ex|':>12}")
print("-" * 58)
for pt in P_CHECKPOINTS:
    r = all_results[pt]
    d0 = r['m0']['dE_mH'][0]
    d1 = r['m1']['dE_mH'][0] if r['m1'] else float('nan')
    mx = max(abs(r['m0']['dE_mH'][k]) for k in range(1, min(NROOTS, len(r['m0']['dE_mH']))))
    print(f"{pt:>6} {r['N']:>6} {r['d_mgs']:>7} {r['d_svd']:>7} {d0:>+10.1f} "
          f"{d1:>+10.1f} {mx:>12.0f}")

# ── Save ──
outdir = os.path.join(PROJECT_ROOT, 'checkpoints_phaseA')
os.makedirs(outdir, exist_ok=True)
fname = f"phaseA_v2_svd{SVD_THR}_{TAG}.json"
with open(os.path.join(outdir, fname), 'w') as f:
    json.dump({
        'config': {'cas': N_ACT, 'n_core': N_CORE, 'P': P_CHECKPOINTS,
                   'svd_threshold': SVD_THR, 'm_max': M_MAX, 'M': M,
                   'e_fci': e_fci, 'tag': TAG},
        'results': {str(k): v for k, v in all_results.items()},
    }, f, indent=2)
print(f"\nSaved: {outdir}/{fname}")
print("Done.")
