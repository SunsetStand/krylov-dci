#!/usr/bin/env python3
"""
Phase A — CAS(10,10) Matrix-Free + Krylov + MGS + SVD Pipeline

Post-meeting proposal §4 Phase A:
  A1: Iterative P expansion to P=10000 on N₂ CAS(10,10)
  A2: Plot d_basis(P) — verify SVD compression at large P
  A3: m-convergence at P = 2000, 4000, 6000, 8000

Pipeline per P-checkpoint:
  1. HFPT2 P-space selection
  2. Build H_PP
  3. Matrix-free: raw Krylov vectors (σ=H|p_i⟩ × A_q, NO MGS)
  4. Weighted SVD: T = A^{1/2} · V_raw → truncated U (d_basis)
  5. Build projected blocks H_QQ_tilde, H_PQ_tilde from SVD-compressed basis
  6. m=0 effective H → dE vs FCI
  7. m=1 Krylov propagation (one B-step)
  8. Save results

P-checkpoints: 200, 500, 1000, 2000, 4000, (8000 optional)

Usage:
    python phaseA_cas10.py --P 200,500,1000,2000,4000
"""
import sys, os, time, json, argparse, itertools, gc
import numpy as np
from numpy.linalg import eigh, svd, norm

PROJECT_ROOT = '/data/home/wangcx/krylov-dci'
sys.path.insert(0, PROJECT_ROOT)

from src_mf import QSpaceIndex
from src.effective_h import build_effective_H, diagonalize_effective_H
from src.determinants import hf_determinant, bit_positions
from src.hamiltonian import Hamiltonian
from pyscf import gto, scf, ao2mo
from pyscf.fci import cistring, direct_spin1
from pyscf.fci import selected_ci

# ═══════════════════════════════════════════════════════════════
# Parse args
# ═══════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--P', type=str, default='200,500,1000,2000,4000',
                   help='Comma-separated P-checkpoints')
    p.add_argument('--svd-threshold', type=float, default=1e-3,
                   help='SVD truncation threshold (fraction of sigma_max)')
    p.add_argument('--svd-threshold-fixed', type=float, default=0.0,
                   help='Absolute SVD threshold override (0=disabled)')
    p.add_argument('--no-svd', action='store_true',
                   help='Skip SVD, use full MGS basis (for comparison)')
    p.add_argument('--m-max', type=int, default=1,
                   help='Max Krylov layers (0=m0 only, 1=m0+m1, ...)')
    p.add_argument('--tag', type=str, default='',
                   help='Output tag')
    return p.parse_args()

args = parse_args()
P_CHECKPOINTS = [int(x) for x in args.P.split(',')]
SVD_THRESHOLD = args.svd_threshold
SVD_THRESHOLD_FIXED = args.svd_threshold_fixed
NO_SVD = args.no_svd
M_MAX = args.m_max
TAG = args.tag

# ═══════════════════════════════════════════════════════════════
# Build system: N2/cc-pVDZ CAS(10,10)
# ═══════════════════════════════════════════════════════════════
N_ACT = 10
N_CORE = 2  # freeze 2 core orbitals (N2 has 14e, freeze 4e)
NROOTS = 6
R = 1.1
ne = (5, 5)  # 5α + 5β = 10e in 10 orbitals

print("=" * 70)
print(f"Phase A — CAS({N_ACT},{sum(ne)}) Matrix-Free+Krylov+MGS+SVD")
print(f"N2/cc-pVDZ R={R}  N_CORE={N_CORE}  nroots={NROOTS}")
print(f"Checkpoints: P={P_CHECKPOINTS}")
print(f"SVD threshold={SVD_THRESHOLD}  no_svd={NO_SVD}  m_max={M_MAX}")
print("=" * 70)

t0 = time.perf_counter()
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

na = len(as_); nb = len(bs_)
M = na * nb
print(f"  CAS({N_ACT},{sum(ne)}): M={M:,}")

# FCI reference
print("  FCI reference...", flush=True)
ef, _ = direct_spin1.FCI().kernel(h1a, era, N_ACT, ne, nroots=NROOTS, verbose=0)
e_fci = [float(e) for e in np.atleast_1d(ef)[:NROOTS]]
for i in range(NROOTS):
    dE = e_fci[i] - e_fci[0]
    exc = f"  ({dE*1000:.1f} mH)" if i > 0 else "  (ground)"
    print(f"    S{i}: {e_fci[i]:.12f} Ha{exc}")

# Hamiltonian
h2_4d = ao2mo.restore('s1', era, N_ACT).reshape(N_ACT, N_ACT, N_ACT, N_ACT)
ham = Hamiltonian(h1=h1a, h2=h2_4d, E_nuc=0.0, E_HF=0.0)

# QSpaceIndex for sigma operations
q_idx = QSpaceIndex(as_, bs_, N_ACT, ne, h1a, era)

# HF determinant
hf_a, hf_b = hf_determinant(*ne)
ao = bit_positions(hf_a); bo = bit_positions(hf_b)
av = [p for p in range(N_ACT) if p not in ao]
bv = [p for p in range(N_ACT) if p not in bo]
E_HF = ham.matrix_element((hf_a, hf_b), (hf_a, hf_b))

# H_QQ diagonal
hdiag = q_idx.hdiag

# P indices (flat) for fast lookup
def p_flat_indices(p_dets):
    """Get flat indices of P-determinants in Q-space."""
    idx = q_idx.p_indices(p_dets)
    return idx[idx >= 0]

# ── Helper: σ-vector (H · v) in Q-space using PySCF contract_2e ──
def sigma_on_q(v_flat):
    """σ = H · v using selected_ci.contract_2e (C-level).

    v_flat: dense numpy array of length M, coefficients over Q-space.
    Returns: dense numpy array of length M, σ = H|v⟩.
    """
    na_q, nb_q = na, nb  # Q uses full CI strings
    ci_mat = v_flat.reshape(na_q, nb_q)
    sigma = selected_ci.contract_2e(era, ci_mat, as_, bs_,
                                    norb=N_ACT, nelec=ne)
    sigma += selected_ci.contract_1e(h1a, ci_mat, as_, bs_,
                                     norb=N_ACT, nelec=ne)
    return sigma.reshape(-1)


# ═══════════════════════════════════════════════════════════════
# HFPT2 P-space selector
# ═══════════════════════════════════════════════════════════════
def select_pspace_hfpt2(max_P):
    """Select top P determinants via HFPT2 energy weighting."""
    scores = []
    # singles α
    for i in ao:
        for a in av:
            d = (hf_a ^ (1 << i) | (1 << a), hf_b)
            hij = ham.matrix_element(d, (hf_a, hf_b))
            de = E_HF - ham.matrix_element(d, d)
            if abs(de) > 1e-12: scores.append((d, -hij * hij / de))
    # singles β
    for i in bo:
        for a in bv:
            d = (hf_a, hf_b ^ (1 << i) | (1 << a))
            hij = ham.matrix_element(d, (hf_a, hf_b))
            de = E_HF - ham.matrix_element(d, d)
            if abs(de) > 1e-12: scores.append((d, -hij * hij / de))
    # doubles αα
    for i1, i2 in itertools.combinations(ao, 2):
        for a1, a2 in itertools.combinations(av, 2):
            d = (hf_a ^ (1 << i1) ^ (1 << i2) | (1 << a1) | (1 << a2), hf_b)
            hij = ham.matrix_element(d, (hf_a, hf_b))
            de = E_HF - ham.matrix_element(d, d)
            if abs(de) > 1e-12: scores.append((d, -hij * hij / de))
    # doubles ββ
    for i1, i2 in itertools.combinations(bo, 2):
        for a1, a2 in itertools.combinations(bv, 2):
            d = (hf_a, hf_b ^ (1 << i1) ^ (1 << i2) | (1 << a1) | (1 << a2))
            hij = ham.matrix_element(d, (hf_a, hf_b))
            de = E_HF - ham.matrix_element(d, d)
            if abs(de) > 1e-12: scores.append((d, -hij * hij / de))
    # doubles αβ
    for i in ao:
        for j in bo:
            for a in av:
                for b in bv:
                    d = (hf_a ^ (1 << i) | (1 << a), hf_b ^ (1 << j) | (1 << b))
                    hij = ham.matrix_element(d, (hf_a, hf_b))
                    de = E_HF - ham.matrix_element(d, d)
                    if abs(de) > 1e-12: scores.append((d, -hij * hij / de))
    scores.sort(key=lambda x: x[1], reverse=True)
    p_dets = [(hf_a, hf_b)]
    seen = {p_dets[0]}
    for det, _ in scores:
        if det not in seen:
            seen.add(det)
            p_dets.append(det)
        if len(p_dets) >= max_P:
            break
    return p_dets


# ═══════════════════════════════════════════════════════════════
# Step 3: Build raw Krylov vectors (matrix-free, no MGS)
# ═══════════════════════════════════════════════════════════════
def build_raw_krylov_vectors(p_dets, E0):
    """Build raw Krylov vectors V_raw[q, p] = A_q[q] · ⟨q|H|p_i⟩.

    For each P-det p_i:
      1. Build unit CI vector at p_i
      2. σ = H|p_i⟩ (via contract_2e)
      3. Zero out P-space components
      4. Multiply by A_q diagonal resolvent

    Returns:
        V_raw: (M, N) dense array, stored as memmap for large N.
        sigma_path: path to memmap file (for cleanup).
    """
    N = len(p_dets)
    denom = E0 - hdiag
    mask = np.abs(denom) > 1e-10
    A_q = np.zeros(M)
    A_q[mask] = 1.0 / denom[mask]

    # P indices for zeroing
    p_idx_set = set(p_flat_indices(p_dets))

    tmpdir = '/data/home/wangcx/krylov-dci/tmp'
    os.makedirs(tmpdir, exist_ok=True)
    sigma_path = f'{tmpdir}/phaseA_raw_N{N}.dat'
    V_raw = np.memmap(sigma_path, dtype='float64', mode='w+', shape=(M, N))

    t0 = time.perf_counter()
    print(f"    Building raw vectors: {N} columns...", flush=True)

    for p in range(N):
        pa, pb = int(p_dets[p][0]), int(p_dets[p][1])
        idx = q_idx.flat_index(pa, pb)
        if idx is None or idx < 0:
            continue

        unit = np.zeros(M)
        unit[idx] = 1.0
        sigma_p = sigma_on_q(unit)

        # Zero P-space components
        for q in p_idx_set:
            sigma_p[q] = 0.0

        # Apply A_q weighting
        V_raw[:, p] = A_q * sigma_p

        if (p + 1) % max(1, N // 5) == 0:
            elapsed = time.perf_counter() - t0
            rate = (p + 1) / elapsed
            eta = (N - p - 1) / rate
            print(f"      col {p+1}/{N}  "
                  f"({elapsed:.0f}s, ETA {eta:.0f}s)", flush=True)

    V_raw.flush()
    elapsed = time.perf_counter() - t0
    print(f"    Raw vectors built: {elapsed:.0f}s ({elapsed/N:.2f}s/col)", flush=True)
    return V_raw, A_q, sigma_path


# ═══════════════════════════════════════════════════════════════
# Step 4: Weighted SVD compression
# ═══════════════════════════════════════════════════════════════
def svd_compress_raw(V_raw, A_q, threshold, threshold_fixed, tag=""):
    """Weighted SVD: T = A^{1/2} · V_raw → SVD → truncated U.

    Returns:
        U_compressed: (M, d) orthonormal basis
        sigma: (d,) retained singular values
        d: number of retained vectors
    """
    M, N = V_raw.shape
    t0 = time.perf_counter()

    # Weighted matrix: T[q, p] = sqrt(|A_q|) * V_raw[q, p]
    sqrt_A = np.sqrt(np.abs(A_q))
    T = V_raw * sqrt_A[:, np.newaxis]

    # Economy SVD
    print(f"    SVD({M}, {N})...", flush=True)
    U, sigma, Vt = svd(T, full_matrices=False)

    sigma_max = sigma[0] if len(sigma) > 0 else 0.0
    if sigma_max < 1e-15:
        return np.zeros((M, 0)), np.array([]), 0

    # Truncation
    if threshold_fixed > 0:
        mask = sigma >= threshold_fixed
    else:
        mask = sigma >= threshold * sigma_max
    d = np.sum(mask)
    U_retained = U[:, mask]
    sigma_retained = sigma[mask]

    elapsed = time.perf_counter() - t0
    sigma_str = ", ".join(f"{s/sigma_max:.4f}" for s in sigma[:min(8, len(sigma))])
    print(f"    SVD done: {elapsed:.0f}s, N={N} → d={d} "
          f"(σ/σ_max = [{sigma_str}])", flush=True)

    return U_retained, sigma_retained, d


# ═══════════════════════════════════════════════════════════════
# Step 5: Build projected blocks (dense, from SVD-compressed basis)
# ═══════════════════════════════════════════════════════════════
def build_projected_blocks_dense(U_compressed, p_dets):
    """Build H_QQ_tilde and H_PQ_tilde from dense SVD-compressed basis.

    U_compressed: (M, d) orthonormal basis, columns are in Q-det space.
    """
    d = U_compressed.shape[1]
    N = len(p_dets)
    if d == 0:
        return np.zeros((0, 0)), np.zeros((N, 0))

    t0 = time.perf_counter()
    print(f"    Projecting d={d} basis vectors (dense σ)...", flush=True)

    H_QQ_tilde = np.zeros((d, d))
    H_PQ_tilde = np.zeros((N, d))

    p_flat = p_flat_indices(p_dets)
    p_valid = p_flat >= 0

    for k in range(d):
        v_k = U_compressed[:, k]
        sigma_k = sigma_on_q(v_k)

        # H_QQ_tilde[j, k] = ⟨w_j|H|w_k⟩ = w_j^T · σ_k (σ_k = H|w_k⟩)
        H_QQ_tilde[:, k] = U_compressed.T @ sigma_k

        # H_PQ_tilde[p, k] = ⟨p|H|w_k⟩ (extract P-determinant components from σ_k)
        H_PQ_tilde[p_valid, k] = sigma_k[p_flat]

        if (k + 1) % max(1, d // 5) == 0:
            print(f"      basis {k+1}/{d} "
                  f"({time.perf_counter()-t0:.0f}s)", flush=True)

    H_QQ_tilde = 0.5 * (H_QQ_tilde + H_QQ_tilde.T)
    elapsed = time.perf_counter() - t0
    print(f"    Projection done: {elapsed:.0f}s", flush=True)
    return H_QQ_tilde, H_PQ_tilde


# ═══════════════════════════════════════════════════════════════
# Step 6-7: Effective H and Krylov propagation
# ═══════════════════════════════════════════════════════════════
def build_krylov_m0(H_PP, H_PQ_tilde, H_QQ_tilde, E0, p_dets, U_compressed, nroots):
    """m=0: diagonal resolvent effective H."""
    ev, evec = diagonalize_effective_H(
        build_effective_H(H_PP, H_PQ_tilde, H_QQ_tilde, E0, delta=0.0),
        n_states=nroots)
    return ev, evec, H_PQ_tilde, H_QQ_tilde


def build_krylov_m1(H_PP, H_PQ_tilde, H_QQ_tilde, E0, p_dets, U_compressed, nroots):
    """m=1: one B-step Krylov propagation.

    B = H_QQ_tilde (off-diagonal in compressed basis; A is diagonal in original Q space)
    New basis vector: w'_k = (AB) · w_k = A · B · w_k

    But in our compressed basis, B is already projected.
    For m=1, we need to apply H (full Q-space) once to each basis vector,
    then re-orthonormalize via MGS against the m=0 basis.

    Simplified: directly apply H on each U column, orthonormalize against
    original basis via MGS, build expanded projected blocks.
    """
    d0 = U_compressed.shape[1]
    if d0 == 0:
        return build_krylov_m0(H_PP, H_PQ_tilde, H_QQ_tilde, E0, p_dets, U_compressed, nroots)

    t0 = time.perf_counter()
    print(f"    Krylov m=1: propagating d0={d0} vectors...", flush=True)

    denom = E0 - hdiag
    A_q = np.where(np.abs(denom) > 1e-10, 1.0 / denom, 0.0)

    # For each basis vector w_k, compute new vector: A · H · w_k
    new_raw = np.zeros((M, d0))
    for k in range(d0):
        w_k = U_compressed[:, k]
        sigma_k = sigma_on_q(w_k)
        # Remove projection onto existing basis (B-step in compressed space)
        # B · w_k in original Q-space = H_QQ · w_k - D_QQ · w_k (minus diagonal)
        # But we use full σ and let MGS handle orthogonality
        new_raw[:, k] = A_q * sigma_k

    # MGS: orthonormalize against existing U_compressed
    new_orth = np.zeros((M, d0))
    retained = []
    for k in range(d0):
        v = new_raw[:, k].copy()
        # Against original basis
        for j in range(d0):
            v -= np.dot(U_compressed[:, j], v) * U_compressed[:, j]
        # Against previously accepted new vectors
        for r_idx in retained:
            v -= np.dot(new_orth[:, r_idx], v) * new_orth[:, r_idx]
        nrm = norm(v)
        if nrm > 1e-10:
            new_orth[:, k] = v / nrm
            retained.append(k)

    d1 = len(retained)
    if d1 == 0:
        print(f"    m=1: all new vectors linearly dependent → same as m=0")
        return build_krylov_m0(H_PP, H_PQ_tilde, H_QQ_tilde, E0, p_dets, U_compressed, nroots)

    new_basis = new_orth[:, :d1]
    # Build expanded projected blocks
    d_expanded = d0 + d1
    U_expanded = np.hstack([U_compressed, new_basis])

    H_QQ_exp, H_PQ_exp = build_projected_blocks_dense(U_expanded, p_dets)

    ev, evec = diagonalize_effective_H(
        build_effective_H(H_PP, H_PQ_exp, H_QQ_exp, E0, delta=0.0),
        n_states=nroots)

    elapsed = time.perf_counter() - t0
    print(f"    m=1 done: d0={d0} + d1={d1} = {d0+d1}, {elapsed:.0f}s", flush=True)
    return ev, evec, H_PQ_exp, H_QQ_exp


# ═══════════════════════════════════════════════════════════════
# Main: iterate over P checkpoints
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("Phase A Pipeline")
print(f"{'='*70}")

all_results = {}
total_t0 = time.perf_counter()

for P_target in P_CHECKPOINTS:
    print(f"\n{'─'*70}")
    print(f"P = {P_target}")
    print(f"{'─'*70}")

    # Step 1: HFPT2 P-space
    t1 = time.perf_counter()
    p_dets = select_pspace_hfpt2(P_target)
    N = len(p_dets)
    print(f"  HFPT2 P-space: N={N} ({time.perf_counter()-t1:.0f}s)")

    # Step 2: Build H_PP
    t1 = time.perf_counter()
    H_PP = np.zeros((N, N))
    for i in range(N):
        H_PP[i, i] = ham.matrix_element(p_dets[i], p_dets[i])
        for j in range(i + 1, N):
            H_PP[i, j] = ham.matrix_element(p_dets[i], p_dets[j])
            H_PP[j, i] = H_PP[i, j]
    E0_vals, _ = eigh(H_PP)
    E0 = E0_vals[0]
    dE0_P = (E0 - e_fci[0]) * 1000
    print(f"  H_PP: E0={E0:.8f}, dE0(P-only)={dE0_P:+.3f} mH "
          f"({time.perf_counter()-t1:.0f}s)")

    # For P > 2000, SVD of full (M,N) is expensive. Use Gram trick.
    use_gram_trick = (N > 2000 and not NO_SVD)
    if use_gram_trick:
        print(f"  N={N}>2000 → using Gram-matrix SVD trick")

    # Step 3: Raw Krylov vectors
    t1 = time.perf_counter()
    V_raw, A_q, sigma_path = build_raw_krylov_vectors(p_dets, E0)

    # Step 4: SVD compression
    threshold = SVD_THRESHOLD
    threshold_fixed = SVD_THRESHOLD_FIXED if SVD_THRESHOLD_FIXED > 0 else 0.0

    if NO_SVD:
        # Skip SVD: use full MGS rather than SVD (comparison)
        # For now just set d = N (no compression)
        print(f"    --no-svd: skipping compression, using all N={N} vectors")
        # Use raw vectors directly (after energy weighting, no MGS needed for SVD-baseline)
        # Actually need orthonormal basis. Simple: QR of weighted raw vectors.
        from numpy.linalg import qr
        sqrt_A = np.sqrt(np.abs(A_q))
        T = V_raw * sqrt_A[:, np.newaxis]
        U_compressed, _ = qr(T, mode='reduced')
        sigma = np.ones(min(M, N))
        d = U_compressed.shape[1]
    else:
        U_compressed, sigma, d = svd_compress_raw(
            V_raw, A_q, threshold, threshold_fixed, TAG)

    # Cleanup raw vectors
    try:
        del V_raw; gc.collect()
        os.unlink(sigma_path)
    except OSError:
        pass

    d_basis = d
    print(f"  d_basis(P={N}) = {d_basis}")

    # Step 5: Build projected blocks
    H_QQ_tilde, H_PQ_tilde = build_projected_blocks_dense(U_compressed, p_dets)

    # Step 6: m=0 effective H
    print(f"\n  ── Krylov m=0 ──")
    t_m0 = time.perf_counter()
    ev_m0, evec_m0, _, _ = build_krylov_m0(
        H_PP, H_PQ_tilde, H_QQ_tilde, E0, p_dets, U_compressed, NROOTS)
    wall_m0 = time.perf_counter() - t_m0
    dE_m0 = [(ev_m0[k] - e_fci[k]) * 1000 for k in range(min(NROOTS, len(ev_m0)))]
    print(f"  m=0 wall={wall_m0:.0f}s")
    for k in range(min(NROOTS, len(ev_m0))):
        print(f"    S{k}: E={ev_m0[k]:.12f}  dE={dE_m0[k]:+8.1f} mH")

    # Step 7: m=1 Krylov propagation (if M_MAX >= 1)
    ev_m1, dE_m1, wall_m1 = None, None, None
    d1 = 0
    if M_MAX >= 1:
        print(f"\n  ── Krylov m=1 ──")
        t_m1 = time.perf_counter()
        ev_m1, evec_m1, H_PQ_m1, H_QQ_m1 = build_krylov_m1(
            H_PP, H_PQ_tilde, H_QQ_tilde, E0, p_dets, U_compressed, NROOTS)
        wall_m1 = time.perf_counter() - t_m1
        d1 = H_QQ_m1.shape[0] - d_basis
        dE_m1 = [(ev_m1[k] - e_fci[k]) * 1000 for k in range(min(NROOTS, len(ev_m1)))]
        print(f"  m=1 wall={wall_m1:.0f}s")
        for k in range(min(NROOTS, len(ev_m1))):
            ddE = dE_m1[k] - dE_m0[k] if dE_m0[k] is not None else 0
            print(f"    S{k}: E={ev_m1[k]:.12f}  dE={dE_m1[k]:+8.1f} mH  "
                  f"(Δm={ddE:+.1f})")

    # Record results
    all_results[P_target] = {
        'P': P_target,
        'N': N,
        'd_basis': d_basis,
        'd1': d1,
        'E0': float(E0),
        'dE0_P_mH': float(dE0_P),
        'sigma_vals': [float(s) for s in sigma[:min(20, len(sigma))]],
        'sigma_max': float(sigma[0]) if len(sigma) > 0 else 0.0,
        'sigma_ratios': [float(s / sigma[0]) if sigma[0] > 0 else 0.0
                         for s in sigma[:min(20, len(sigma))]],
        'm0': {
            'E': [float(e) for e in ev_m0[:NROOTS]],
            'dE_mH': [float(d) for d in dE_m0[:NROOTS]],
            'wall_s': wall_m0,
        },
        'm1': {
            'E': [float(e) for e in ev_m1[:NROOTS]] if ev_m1 is not None else None,
            'dE_mH': [float(d) for d in dE_m1[:NROOTS]] if dE_m1 is not None else None,
            'wall_s': wall_m1,
        } if M_MAX >= 1 else None,
    }

    # Summary for this P
    ex_de_m0 = [abs(dE_m0[k]) for k in range(1, min(NROOTS, len(ev_m0)))]
    print(f"\n  Summary P={P_target}: d_basis={d_basis}/{N}, "
          f"dE0={dE_m0[0]:+.1f} mH, "
          f"excited |dE|={[f'{x:.0f}' for x in ex_de_m0]} mH")

print(f"\n{'='*70}")
print(f"Phase A Complete: total wall={time.perf_counter()-total_t0:.0f}s")
print(f"{'='*70}")

# ── Overall Summary ──
print(f"\n{'P':>6} {'N':>6} {'d_basis':>8} {'dE0_mH':>10} "
      f"{'dE0_m1':>10} {'max|dE_ex|':>12}")
print("-" * 56)
for P_target in P_CHECKPOINTS:
    r = all_results[P_target]
    dE0_m0 = r['m0']['dE_mH'][0]
    dE0_m1 = r['m1']['dE_mH'][0] if r['m1'] else float('nan')
    max_ex = max(abs(r['m0']['dE_mH'][k]) for k in range(1, min(NROOTS, len(r['m0']['dE_mH']))))
    print(f"{P_target:>6} {r['N']:>6} {r['d_basis']:>8} {dE0_m0:>+10.1f} "
          f"{dE0_m1:>+10.1f} {max_ex:>12.0f}")

# ── Save ──
outdir = os.path.join(PROJECT_ROOT, 'checkpoints_phaseA')
os.makedirs(outdir, exist_ok=True)
suffix = f"_{TAG}" if TAG else ""
fname = f"phaseA_cas10_svd{SVD_THRESHOLD}{suffix}.json"
outpath = os.path.join(outdir, fname)
with open(outpath, 'w') as f:
    json.dump({
        'config': {
            'cas': N_ACT,
            'n_core': N_CORE,
            'P': P_CHECKPOINTS,
            'svd_threshold': SVD_THRESHOLD,
            'no_svd': NO_SVD,
            'm_max': M_MAX,
            'M': M,
            'e_fci': e_fci,
        },
        'results': {str(k): v for k, v in all_results.items()},
    }, f, indent=2)
print(f"\nSaved: {outpath}")
print("Done.")
