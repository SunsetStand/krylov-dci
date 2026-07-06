#!/usr/bin/env python3
"""
Step 1: Iterative P-space selection via multi-reference perturbation importance.

Algorithm:
  P_0 ← HFPT2 SD selection (~200 determinants)
  Loop:
    1. Diagonalize H_PP → {E_k, |Ψ_k⟩}
    2. For each of first nroots eigenstates, compute σ_k = H|Ψ_k⟩ (C-level)
    3. Score each Q det: w_a = Σ_k |σ_k[a]|² / |E_k - H_aa|
    4. Add top-BATCH_SIZE Q dets to P
    5. Save checkpoint at target P sizes

H_PP is built incrementally (O(P·batch) per iter) for efficiency.

System: N2/cc-pVDZ CAS(10,10)
Reference: DMRG-CI = exact FCI in this CAS
"""
import sys, os, time, json, itertools
import numpy as np
from numpy.linalg import eigh

sys.path.insert(0, '/data/home/wangcx/krylov-dci')
from src_mf.pyscf_backend import QSpaceIndex, KDCIBackend
from src.determinants import hf_determinant, bit_positions
from src.hamiltonian import Hamiltonian
from pyscf import gto, scf, ao2mo
from pyscf.fci import cistring, direct_spin1

# ── Parameters ────────────────────────────────────────────────────
N_CORE = 2
N_ACT = 10
NROOTS = 6
P_INIT = 200
BATCH_SIZE = 200
P_TARGETS = [200, 400, 800, 1200, 1600, 2000]
P_MAX = max(P_TARGETS)
MAX_NROOTS_SIGMA = 5        # use at most 5 eigenstates for σ-vectors
# ───────────────────────────────────────────────────────────────────

OUTDIR = '/data/home/wangcx/krylov-dci/checkpoints_pspace'
os.makedirs(OUTDIR, exist_ok=True)

print("=" * 64)
print("Step 1: Iterative P-space Selection")
print(f"N2/cc-pVDZ CAS({N_ACT},{N_ACT})  P_init={P_INIT} → P_max={P_MAX}")
print(f"batch={BATCH_SIZE}  nroots={NROOTS}  checkpoints={P_TARGETS}")
print("=" * 64)

# ── Build system ──────────────────────────────────────────────────
print("\n[1] Building N2/cc-pVDZ CAS(10,10)...", flush=True)
mol = gto.M(atom='N 0 0 0; N 0 0 1.1', basis='cc-pVDZ', verbose=0)
mf = scf.RHF(mol).run(verbose=0)
na_o = list(range(N_CORE, N_CORE + N_ACT))
norb = mf.mo_coeff.shape[1]
h1_mo = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
eri_mo = ao2mo.full(mol.intor('int2e'), mf.mo_coeff, compact=False)
eri_mo = eri_mo.reshape(norb, norb, norb, norb)
h1a = h1_mo[np.ix_(na_o, na_o)]
era = eri_mo[np.ix_(na_o, na_o, na_o, na_o)]
ne = (mol.nelec[0] - N_CORE, mol.nelec[1] - N_CORE)
as_ = cistring.gen_strings4orblist(range(N_ACT), ne[0])
bs_ = cistring.gen_strings4orblist(range(N_ACT), ne[1])
q_idx = QSpaceIndex(as_, bs_, N_ACT, ne, h1a, era)
backend = KDCIBackend(q_idx)
M = q_idx.M
hdiag_full = q_idx.hdiag  # shape (M,)
print(f"  {ne[0]}a+{ne[1]}b in {N_ACT} orbs, M={M:,}", flush=True)

# ── DMRG-CI reference ─────────────────────────────────────────────
print("[2] Computing DMRG-CI reference...", flush=True)
ef, _ = direct_spin1.FCI().kernel(h1a, era, N_ACT, ne,
                                   nroots=NROOTS, verbose=0)
e_dmrg = [float(e) for e in np.atleast_1d(ef)[:NROOTS]]
for k, e in enumerate(e_dmrg):
    print(f"  root {k}: {e:.8f} Ha", flush=True)

# ── Hamiltonian + Full determinant list ───────────────────────────
h2_4d = ao2mo.restore('s1', era, N_ACT).reshape(N_ACT, N_ACT, N_ACT, N_ACT)
ham = Hamiltonian(h1=h1a, h2=h2_4d, E_nuc=0.0, E_HF=0.0)
hf_a, hf_b = hf_determinant(*ne)
ao = bit_positions(hf_a)
bo = bit_positions(hf_b)
av = [p for p in range(N_ACT) if p not in ao]
bv = [p for p in range(N_ACT) if p not in bo]

# Full determinant list (α-str × β-str ordering, matches q_idx)
full_dets = []
for ai, a_str in enumerate(as_):
    for bi, b_str in enumerate(bs_):
        full_dets.append((int(a_str), int(b_str)))
assert len(full_dets) == M

# ── Step 1a: Initial P via HFPT2 SD ───────────────────────────────
print(f"\n[3] Initial P via HFPT2 (target {P_INIT})...", flush=True)
E_HF = ham.matrix_element((hf_a, hf_b), (hf_a, hf_b))

def gen_hfpt2_scores(hf_alpha, hf_beta, ao_list, bo_list, av_list, bv_list):
    """Generate PT2 importance scores for all SD excitations from HF."""
    scores = []
    # alpha singles
    for i in ao_list:
        for a in av_list:
            d = (hf_alpha ^ (1 << i) | (1 << a), hf_beta)
            hij = ham.matrix_element(d, (hf_alpha, hf_beta))
            den = E_HF - ham.matrix_element(d, d)
            if abs(den) > 1e-12:
                scores.append((d, -hij * hij / den))
    # beta singles
    for i in bo_list:
        for a in bv_list:
            d = (hf_alpha, hf_beta ^ (1 << i) | (1 << a))
            hij = ham.matrix_element(d, (hf_alpha, hf_beta))
            den = E_HF - ham.matrix_element(d, d)
            if abs(den) > 1e-12:
                scores.append((d, -hij * hij / den))
    # aa doubles
    for i1, i2 in itertools.combinations(ao_list, 2):
        for a1, a2 in itertools.combinations(av_list, 2):
            d = (hf_alpha ^ (1 << i1) ^ (1 << i2) | (1 << a1) | (1 << a2),
                 hf_beta)
            hij = ham.matrix_element(d, (hf_alpha, hf_beta))
            den = E_HF - ham.matrix_element(d, d)
            if abs(den) > 1e-12:
                scores.append((d, -hij * hij / den))
    # bb doubles
    for i1, i2 in itertools.combinations(bo_list, 2):
        for a1, a2 in itertools.combinations(bv_list, 2):
            d = (hf_alpha,
                 hf_beta ^ (1 << i1) ^ (1 << i2) | (1 << a1) | (1 << a2))
            hij = ham.matrix_element(d, (hf_alpha, hf_beta))
            den = E_HF - ham.matrix_element(d, d)
            if abs(den) > 1e-12:
                scores.append((d, -hij * hij / den))
    # ab doubles
    for i in ao_list:
        for j in bo_list:
            for a in av_list:
                for b in bv_list:
                    d = (hf_alpha ^ (1 << i) | (1 << a),
                         hf_beta ^ (1 << j) | (1 << b))
                    hij = ham.matrix_element(d, (hf_alpha, hf_beta))
                    den = E_HF - ham.matrix_element(d, d)
                    if abs(den) > 1e-12:
                        scores.append((d, -hij * hij / den))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores

scores = gen_hfpt2_scores(hf_a, hf_b, ao, bo, av, bv)

p_dets = [(hf_a, hf_b)]
for det, _ in scores:
    if det not in p_dets:
        p_dets.append(det)
    if len(p_dets) >= P_INIT:
        break
N_p = len(p_dets)
print(f"  Initial P = {N_p} determinants (from {len(scores)} SD excitations)",
      flush=True)

# Det → full index mapping
det_to_full = {d: i for i, d in enumerate(full_dets)}
p_full_indices = [det_to_full[d] for d in p_dets]

# Initial H_PP
print(f"  Building initial H_PP ({N_p}×{N_p})...", flush=True)
H_PP = np.zeros((N_p, N_p))
for i in range(N_p):
    for j in range(i, N_p):
        H_PP[i, j] = ham.matrix_element(p_dets[i], p_dets[j])
        H_PP[j, i] = H_PP[i, j]

# ── Step 1b: Iterative selection ──────────────────────────────────
print(f"\n[4] Iterative selection: P={N_p} → P={P_MAX}", flush=True)
print(f"{'iter':>4} {'P_size':>7} {'E0_bare':>14} {'dE0_bare(mH)':>14} "
      f"{'max_w':>12} {'wall(s)':>8}")
print("-" * 64, flush=True)

checkpoints = {}
wall_total = time.perf_counter()
p_set = set(p_full_indices)  # for fast Q lookup

iter_num = 0
while N_p < P_MAX:
    t_iter = time.perf_counter()

    # 4a. Diagonalize current H_PP
    E_P, C_P = eigh(H_PP)
    E_bare = E_P[:NROOTS]

    # 4b. Build σ-vectors for first nroots eigenstates via C-level backend
    n_sigma = min(MAX_NROOTS_SIGMA, N_p)
    sigmas = []  # list of (E_ref, sigma_array_m)
    for k in range(n_sigma):
        # Embed eigenvector in full M-dim space
        vec_full = np.zeros(M)
        for local_i, global_i in enumerate(p_full_indices):
            vec_full[global_i] = C_P[local_i, k]
        sigma_k = backend.sigma(vec_full)  # H|Ψ_k⟩
        sigmas.append((E_P[k], sigma_k))

    # 4c. Score Q-space determinants
    # Weights: w_a = Σ_k |σ_k[a]|² / max(|E_k - H_aa|, 1e-8)
    weights = np.zeros(M)
    for E_ref, sigma_k in sigmas:
        abs_sigma = np.abs(sigma_k)
        # Only score Q-space entries
        # Vectorized: for each Q det q_i, coupling = |σ[q_i]|²
        for qi in range(M):
            if qi not in p_set:
                c2 = abs_sigma[qi] ** 2
                if c2 < 1e-24:
                    continue
                denom = max(abs(E_ref - hdiag_full[qi]), 1e-8)
                weights[qi] += c2 / denom

    # 4d. Select top BATCH_SIZE new from Q (not in P already)
    q_candidates = [(qi, float(weights[qi])) for qi in range(M)
                    if qi not in p_set and weights[qi] > 0]
    q_candidates.sort(key=lambda x: x[1], reverse=True)

    n_add = min(BATCH_SIZE, len(q_candidates))
    if n_add == 0:
        print("  No more determinants with non-zero weight. Stopping.",
              flush=True)
        break

    max_w = q_candidates[0][1]

    new_global_indices = [q_c[0] for q_c in q_candidates[:n_add]]
    new_dets = [full_dets[qi] for qi in new_global_indices]

    # 4e. Incrementally extend H_PP (O(P_old · batch) instead of O(P²))
    N_old = N_p
    N_new = N_old + n_add
    H_PP_new = np.zeros((N_new, N_new))
    H_PP_new[:N_old, :N_old] = H_PP
    for i_local, det_new in enumerate(new_dets):
        row = N_old + i_local
        # vs old P determinants
        for j in range(N_old):
            val = ham.matrix_element(det_new, p_dets[j])
            H_PP_new[row, j] = val
            H_PP_new[j, row] = val
        # vs other new determinants
        for j_local in range(i_local + 1):
            col = N_old + j_local
            val = ham.matrix_element(det_new, new_dets[j_local])
            H_PP_new[row, col] = val
            H_PP_new[col, row] = val
    H_PP = H_PP_new

    # Update state
    p_dets.extend(new_dets)
    p_full_indices.extend(new_global_indices)
    p_set.update(new_global_indices)
    N_p = N_new

    dE0_mH = (E_bare[0] - e_dmrg[0]) * 1000
    t_elapsed = time.perf_counter() - t_iter
    iter_num += 1

    print(f"{iter_num:>4} {N_p:>7} {E_bare[0]:>14.8f} {dE0_mH:>+14.3f} "
          f"{max_w:>12.3e} {t_elapsed:>8.1f}", flush=True)

    # 4f. Save checkpoints at target P sizes
    for p_target in P_TARGETS:
        if N_p >= p_target and p_target not in checkpoints:
            # Use exactly p_target determinants
            dets_p = p_dets[:p_target]
            idx_p = p_full_indices[:p_target]
            checkpoints[p_target] = {
                'P': p_target,
                'p_dets': [(int(a), int(b)) for a, b in dets_p],
                'p_full_indices': [int(i) for i in idx_p],
                'E_bare': [float(e) for e in E_bare],
                'dE0_bare_mH': float(dE0_mH),
                'iter_num': iter_num,
            }
            fname = f"{OUTDIR}/step1_P{p_target:04d}.json"
            # Also save the exact H_PP submatrix for verification
            H_PP_exact = H_PP[:p_target, :p_target]
            checkpoints[p_target]['H_PP_diag'] = [float(H_PP_exact[i,i])
                                                   for i in range(p_target)]
            with open(fname, 'w') as f:
                json.dump(checkpoints[p_target], f, indent=2)
            print(f"    ✓ saved {fname}", flush=True)

wall_total = time.perf_counter() - wall_total

# ── Summary ───────────────────────────────────────────────────────
print(f"\n{'='*64}")
print(f"Step 1 complete. {iter_num} iterations, {wall_total:.0f}s wall")
print(f"Checkpoints: {sorted(checkpoints.keys())}")
print(f"Next: python scripts/step2_bloch_benchmark.py")
print(f"{'='*64}")
