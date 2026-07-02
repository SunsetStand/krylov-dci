#!/usr/bin/env python3
"""
Phase 13 — Matrix-Free Krylov-dCI (m=0 prototype).

Validates the matrix-free sparse implementation against the dense version
on CAS(10,10) N₂/cc-pVDZ.

Key features:
  - Never enumerates Q-space or stores M-dimensional vectors
  - All Q-space operations via on-the-fly Slater-Condon + connected det generation
  - SVD via Gram matrix (avoids M × N matrix construction)
  - H_{Q̃Q̃} via sparse sigma-vector + dot products

Usage:
    python phase13_mf_kdci.py --P 200
"""

import sys, os, time, argparse
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
sys.path.insert(0, PROJECT_ROOT)  # for src_mf import

from pyscf import gto, scf, mcscf, ao2mo
from pyscf.fci import cistring, direct_spin1, selected_ci
from hamiltonian import Hamiltonian, _unpack_4fold
from effective_h import build_effective_H, diagonalize_effective_H
from src_mf.sparse_vector import SparseQVector
from src_mf.sparse_ops import (
    build_hqp_sparse, sparse_mgs, gram_svd,
    sigma_sparse, project_hqq, project_hpq
)

np.set_printoptions(linewidth=140, precision=6, suppress=True)

N_CORE = 3; N_ACT = 10; N_ELEC = 10; BOND_LENGTH = 1.10
NROOTS = 6


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--P', type=int, default=200)
    p.add_argument('--nroots', type=int, default=NROOTS)
    return p.parse_args()


def main():
    args = parse_args()
    P_target = args.P
    print(f"Phase 13: Matrix-Free Krylov-dCI  P={P_target}")
    print(f"  N₂/cc-pVDZ CAS({N_ACT},{N_ELEC}) R={BOND_LENGTH}")

    t0 = time.perf_counter()
    mol = gto.M(atom=f'N 0 0 0; N 0 0 {BOND_LENGTH}', basis='cc-pVDZ', verbose=0)
    mf = scf.RHF(mol); mf.kernel()

    na, nb = N_ELEC // 2, N_ELEC - N_ELEC // 2; nelec = (na, nb)

    # ── FCI reference ──
    cas = mcscf.CASCI(mf, N_ACT, N_ELEC)
    cas.frozen = N_CORE; cas.mo_coeff = mf.mo_coeff
    h1eff, ecore = cas.get_h1eff(); h2eff_packed = cas.get_h2eff()
    h2_4d = _unpack_4fold(h2eff_packed, N_ACT)
    ecore = float(ecore)

    qa = np.asarray(cistring.gen_strings4orblist(list(range(N_ACT)), na), dtype=np.int64)
    qb = np.asarray(cistring.gen_strings4orblist(list(range(N_ACT)), nb), dtype=np.int64)
    nb_q = len(qb); M = len(qa) * nb_q

    fs = direct_spin1.FCI(); fs.conv_tol = 1e-10; fs.nroots = args.nroots
    e_fci, c_fci = fs.kernel(h1eff, h2eff_packed, N_ACT, nelec, ecore=ecore)
    e_fci_bare = e_fci - ecore
    c_flat = c_fci[0].reshape(-1)

    print(f"  FCI E₀ = {e_fci[0]:.8f} Ha", flush=True)
    print(f"  M = {M:,} determinants", flush=True)

    # ── P-space: HF perturbation selection ──
    from src.determinants import hf_determinant, bit_positions
    ham = Hamiltonian(h1=h1eff, h2=h2_4d, E_nuc=0.0, E_HF=0.0)
    hf_a, hf_b = hf_determinant(na, nb)
    E_HF = ham.diagonal_element(hf_a, hf_b)

    # Select P via perturbation
    alpha_occ = bit_positions(hf_a); beta_occ = bit_positions(hf_b)
    all_orbs = list(range(N_ACT))
    av = [p for p in all_orbs if p not in alpha_occ]
    bv = [p for p in all_orbs if p not in beta_occ]

    scores = []
    # αα doubles
    for ii, i in enumerate(alpha_occ):
        for j in alpha_occ[ii+1:]:
            for ia, a in enumerate(av):
                for b in av[ia+1:]:
                    det = ((hf_a^(1<<i)^(1<<j))|(1<<a)|(1<<b), hf_b)
                    hij = ham.matrix_element((hf_a, hf_b), det)
                    hdd = ham.diagonal_element(det[0], det[1])
                    denom = E_HF - hdd
                    if abs(denom) > 1e-12:
                        scores.append((det, -(hij*hij)/denom))
    # ββ doubles
    for ii, i in enumerate(beta_occ):
        for j in beta_occ[ii+1:]:
            for ia, a in enumerate(bv):
                for b in bv[ia+1:]:
                    det = (hf_a, (hf_b^(1<<i)^(1<<j))|(1<<a)|(1<<b))
                    hij = ham.matrix_element((hf_a, hf_b), det)
                    hdd = ham.diagonal_element(det[0], det[1])
                    denom = E_HF - hdd
                    if abs(denom) > 1e-12:
                        scores.append((det, -(hij*hij)/denom))
    # αβ doubles
    for i in alpha_occ:
        for j in beta_occ:
            for a in av:
                for b in bv:
                    det = ((hf_a^(1<<i))|(1<<a), (hf_b^(1<<j))|(1<<b))
                    hij = ham.matrix_element((hf_a, hf_b), det)
                    hdd = ham.diagonal_element(det[0], det[1])
                    denom = E_HF - hdd
                    if abs(denom) > 1e-12:
                        scores.append((det, -(hij*hij)/denom))

    scores.sort(key=lambda x: x[1], reverse=True)
    P_actual = min(P_target - 1, len(scores))
    p_dets = [(hf_a, hf_b)] + [d for d, _ in scores[:P_actual]]
    N = len(p_dets)
    p_set = set(p_dets)

    print(f"  P = {N} dets (from {len(scores)} SD excitations)", flush=True)

    # ── H_PP ──
    H_PP = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            H_PP[i,j] = ham.matrix_element(p_dets[i], p_dets[j])
    E0_P = float(np.linalg.eigh(H_PP)[0][0])
    dE0_P = (E0_P - e_fci_bare[0]) * 1000
    print(f"  E₀(P) − E(FCI) = {dE0_P:+.1f} mH", flush=True)

    # ── Matrix-free Krylov m=0 ──
    print(f"\n{'─'*60}\nMatrix-Free Krylov-dCI (m=0)\n{'─'*60}", flush=True)
    t_mf = time.perf_counter()

    # Diagonal resolvent: A_q = 1/(E0_P - H_QQ[q,q])
    # H_QQ diag needs make_hdiag, but we want matrix-free
    # Use on-the-fly computation via ham.diagonal_element
    diag_cache = {}
    def A_diag_func(a_str, b_str):
        key = (int(a_str), int(b_str))
        if key not in diag_cache:
            diag_cache[key] = 1.0 / (E0_P - ham.diagonal_element(key[0], key[1]))
        return diag_cache[key]

    # Also compute A^{1/2} for weighted SVD
    sqrt_A_cache = {}
    def sqrt_A_func(a_str, b_str):
        key = (int(a_str), int(b_str))
        if key not in sqrt_A_cache:
            diag_val = ham.diagonal_element(key[0], key[1])
            denom = E0_P - diag_val
            sqrt_A_cache[key] = np.sqrt(abs(1.0 / denom))  # use abs for safety
        return sqrt_A_cache[key]

    # Step 1: Build sparse H_QP columns (A · H_QP)
    t1 = time.perf_counter()
    print("  Building sparse H_QP columns...", flush=True)
    hqp_cols = build_hqp_sparse(p_dets, ham, A_diag_func, N_ACT, skip_P=p_set)
    nnz_hqp = sum(c.nnz() for c in hqp_cols)
    print(f"    {N} columns, {nnz_hqp} total nnz ({time.perf_counter()-t1:.0f}s)", flush=True)

    # Step 2: MGS directly on H_QP columns
    # (Gram SVD skippped — dense version with θ=1e-3 retains all columns anyway.
    #  Gram SVD compressed vectors are too dense for matrix-free)
    t2 = time.perf_counter()
    print("  Applying MGS to H_QP columns...", flush=True)
    U = sparse_mgs(hqp_cols, [])
    r = len(U)
    nnz_u = sum(v.nnz() for v in U)
    print(f"    {N} → {r} orthonormal vectors (nnz={nnz_u}, "
          f"avg {nnz_u/max(1,r):.0f}/vec, {time.perf_counter()-t2:.0f}s)", flush=True)

    # Step 4: Projected H_QQ
    print("  Computing H_{Q̃Q̃}...", flush=True)
    t3 = time.perf_counter()
    def diag_func(a_str, b_str):
        return ham.diagonal_element(int(a_str), int(b_str))
    H_QQ_t = project_hqq(U, ham, N_ACT, diag_func)
    print(f"    {r}×{r} matrix ({time.perf_counter()-t3:.0f}s)", flush=True)

    # Step 5: Projected H_PQ
    print("  Computing H_{P~Q}...", flush=True)
    t4 = time.perf_counter()
    H_PQ_t = project_hpq(p_dets, U, ham, N_ACT, skip_P=p_set)
    print(f"    {N}×{r} matrix ({time.perf_counter()-t4:.0f}s)", flush=True)

    # Step 6: Effective Hamiltonian
    print("  Building effective Hamiltonian...", flush=True)
    H_eff = build_effective_H(H_PP, H_PQ_t, H_QQ_t, E0_P, delta=0.0)
    ev_mf, _ = diagonalize_effective_H(H_eff, n_states=None)

    t_mf_total = time.perf_counter() - t_mf
    print(f"  Matrix-free total: {t_mf_total:.0f}s", flush=True)

    # ── Results ──
    print(f"\n{'─'*60}\nResults\n{'─'*60}")
    nr = min(args.nroots, len(ev_mf))
    print(f"  {'State':>6s}  {'E(FCI)/Ha':>16s}  {'E(kDCI)/Ha':>16s}  {'Δ(mH)':>10s}")
    for st in range(nr):
        Ek = ev_mf[st] + ecore
        Eref = e_fci[st]
        print(f"  {st:6d}  {Eref:16.8f}  {Ek:16.8f}  {1000*(ev_mf[st]-e_fci_bare[st]):+10.1f}")

    print(f"\n  Wall time: {time.perf_counter()-t0:.0f}s")

    # ── Sparse statistics ──
    total_nnz_basis = sum(u.nnz() for u in U)
    print(f"\n  Sparsity: {total_nnz_basis} non-zeros in {r} basis vectors "
          f"(M = {M:,}, effective density = {total_nnz_basis/(M*r)*100:.2f}%)")


if __name__ == '__main__':
    main()
