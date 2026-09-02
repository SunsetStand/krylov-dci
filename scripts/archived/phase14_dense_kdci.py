#!/usr/bin/env python3
"""
Phase 14 — Dense Krylov-dCI using PySCF C-level contract_2e backend.

Replaces matrix-free sparse operations (Python-level Slater-Condon rules)
with dense numpy vectors + PySCF libfci C-level calls:
  - H_2e via selected_ci.contract_2e  (libfci.SCIcontract_2e_aaaa/bbaa)
  - H_1e via direct_spin1.contract_1e
  - H_diag via selected_ci.make_hdiag

Key optimizations over Phase 13 (matrix-free sparse):
  1. H_QP: N calls to sigma_full (C-level 1e+2e) vs N×n_exc Python SC calls
  2. MGS: dense numpy BLAS vs SparseQVector dict ops
  3. Projected blocks: one sigma_full per basis vector (gives both blocks)
     vs streaming Python double-sums through sparse dicts

Usage:
    python phase14_dense_kdci.py --P 200 --system N2
    python phase14_dense_kdci.py --P 50 --system H2O --benchmark
"""

import sys, os, time, argparse
import numpy as np
from numpy.linalg import eigh

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from pyscf import gto, scf, mcscf, ao2mo
from pyscf.fci import cistring, direct_spin1

from hamiltonian import Hamiltonian, _unpack_4fold
from effective_h import build_effective_H, diagonalize_effective_H
from src_mf.pyscf_backend import QSpaceIndex, KDCIBackend

np.set_printoptions(linewidth=140, precision=6, suppress=True)


SYSTEMS = {
    'H2O': {
        'atom': 'O 0 0 0; H 1.0 0 0; H -0.2774 0.9605 0',
        'basis': 'sto-3g',
        'n_cas': 4, 'n_elec': 4, 'n_core': 1,
    },
    'N2': {
        'atom': 'N 0 0 0; N 0 0 1.10',
        'basis': 'cc-pVDZ',
        'n_cas': 10, 'n_elec': 10, 'n_core': 3,
    },
    'C2': {
        'atom': 'C 0 0 0; C 0 0 1.243',
        'basis': 'cc-pVDZ',
        'n_cas': 6, 'n_elec': 6, 'n_core': 2,
    },
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--system', default='N2', choices=list(SYSTEMS.keys()))
    p.add_argument('--P', type=int, default=200)
    p.add_argument('--nroots', type=int, default=6)
    p.add_argument('--benchmark', action='store_true',
                   help='Run timing comparison with Phase 13 sparse approach')
    return p.parse_args()


def select_p_perturbative(ham, n_act, na, nb, P_target):
    """P-space selection: HF perturbation theory, doubles only."""
    from src.determinants import hf_determinant, bit_positions

    hf_a, hf_b = hf_determinant(na, nb)
    E_HF = ham.diagonal_element(hf_a, hf_b)
    all_orbs = list(range(n_act))
    alpha_occ = bit_positions(hf_a); beta_occ = bit_positions(hf_b)
    av = [p for p in all_orbs if p not in alpha_occ]
    bv = [p for p in all_orbs if p not in beta_occ]

    scores = []
    # αα doubles
    for ii, i in enumerate(alpha_occ):
        for j in alpha_occ[ii + 1:]:
            for ia, a in enumerate(av):
                for b in av[ia + 1:]:
                    det = ((hf_a ^ (1 << i) ^ (1 << j)) | (1 << a) | (1 << b), hf_b)
                    hij = ham.matrix_element((hf_a, hf_b), det)
                    hdd = ham.diagonal_element(det[0], det[1])
                    denom = E_HF - hdd
                    if abs(denom) > 1e-12:
                        scores.append((det, -(hij * hij) / denom))
    # ββ doubles
    for ii, i in enumerate(beta_occ):
        for j in beta_occ[ii + 1:]:
            for ia, a in enumerate(bv):
                for b in bv[ia + 1:]:
                    det = (hf_a, (hf_b ^ (1 << i) ^ (1 << j)) | (1 << a) | (1 << b))
                    hij = ham.matrix_element((hf_a, hf_b), det)
                    hdd = ham.diagonal_element(det[0], det[1])
                    denom = E_HF - hdd
                    if abs(denom) > 1e-12:
                        scores.append((det, -(hij * hij) / denom))
    # αβ doubles
    for i in alpha_occ:
        for j in beta_occ:
            for a in av:
                for b in bv:
                    det = ((hf_a ^ (1 << i)) | (1 << a), (hf_b ^ (1 << j)) | (1 << b))
                    hij = ham.matrix_element((hf_a, hf_b), det)
                    hdd = ham.diagonal_element(det[0], det[1])
                    denom = E_HF - hdd
                    if abs(denom) > 1e-12:
                        scores.append((det, -(hij * hij) / denom))

    scores.sort(key=lambda x: x[1], reverse=True)
    P_actual = min(P_target - 1, len(scores))
    p_dets = [(hf_a, hf_b)] + [d for d, _ in scores[:P_actual]]
    return p_dets


def build_H_PP(ham, p_dets):
    """Build H_PP (N×N). Bulk diag via C-level make_hdiag, off-diag in Python."""
    N = len(p_dets)
    H_PP = np.zeros((N, N))
    diag_bulk = ham.diagonal_elements_bulk(p_dets)
    np.fill_diagonal(H_PP, diag_bulk)
    for i in range(N):
        for j in range(i + 1, N):
            h_ij = ham.matrix_element(p_dets[i], p_dets[j])
            H_PP[i, j] = h_ij; H_PP[j, i] = h_ij
    return H_PP


def main():
    args = parse_args()
    spec = SYSTEMS[args.system]

    print(f"Phase 14: Dense Krylov-dCI (PySCF C-level backend)")
    print(f"  System: {args.system}  {spec['basis']}  "
          f"CAS({spec['n_cas']},{spec['n_elec']})")
    print(f"  P = {args.P}  nroots = {args.nroots}\n")

    t0 = time.perf_counter()

    # ── RHF + CASCI ──
    mol = gto.M(atom=spec['atom'], basis=spec['basis'], verbose=0)
    mf = scf.RHF(mol); mf.kernel()
    n_act = spec['n_cas']; n_elec = spec['n_elec']
    na = n_elec // 2; nb = n_elec - na; nelec = (na, nb)

    cas = mcscf.CASCI(mf, n_act, n_elec)
    cas.frozen = spec['n_core']; cas.mo_coeff = mf.mo_coeff
    h1eff, ecore = cas.get_h1eff(); h2eff_packed = cas.get_h2eff()
    h2_4d = _unpack_4fold(h2eff_packed, n_act)
    ecore = float(ecore)

    # ── Q-space index + C-level link tables ──
    t_idx = time.perf_counter()
    qa = np.array(cistring.gen_strings4orblist(list(range(n_act)), na), dtype=np.int64)
    qb = np.array(cistring.gen_strings4orblist(list(range(n_act)), nb), dtype=np.int64)
    q_idx = QSpaceIndex(qa, qb, n_act, nelec, h1eff, h2eff_packed)
    M = q_idx.M
    print(f"  Q-space: M = {M:,} ({len(qa)} α × {len(qb)} β) "
          f"[{time.perf_counter()-t_idx:.1f}s]", flush=True)

    # ── FCI reference ──
    t_fci = time.perf_counter()
    fs = direct_spin1.FCI(); fs.conv_tol = 1e-10; fs.nroots = args.nroots
    e_fci, c_fci = fs.kernel(h1eff, h2eff_packed, n_act, nelec, ecore=ecore)
    e_fci_bare = e_fci - ecore
    print(f"  FCI E₀ = {e_fci[0]:.8f} Ha [{time.perf_counter()-t_fci:.1f}s]",
          flush=True)

    # ── P-space selection ──
    t_psel = time.perf_counter()
    ham = Hamiltonian(h1=h1eff, h2=h2_4d, E_nuc=0.0, E_HF=0.0)
    p_dets = select_p_perturbative(ham, n_act, na, nb, args.P)
    N = len(p_dets); p_set = set(p_dets)
    H_PP = build_H_PP(ham, p_dets)
    E0_P = float(eigh(H_PP)[0][0])
    print(f"  P = {N} dets  E₀(P)−E(FCI) = {(E0_P-e_fci_bare[0])*1000:+.1f} mH "
          f"[{time.perf_counter()-t_psel:.1f}s]", flush=True)

    # ═══════════════════════════════════════════════════════════════
    # Dense Krylov-dCI (m=0) — PySCF C-level backend
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'─'*60}")
    print(f"Dense Krylov-dCI (m=0) — PySCF C-level backend")
    print(f"{'─'*60}", flush=True)

    backend = KDCIBackend(q_idx)
    t_kdci = time.perf_counter()

    # Step 1: H_QP via sigma_full (C-level, combined 1e+2e)
    t1 = time.perf_counter()
    print("  [1/4] H_QP via selected_ci.contract_2e (C-level, absorbed 1e+2e)...", flush=True)
    H_QP = backend.build_hqp(p_dets, verbose=True)
    dt1 = time.perf_counter() - t1
    print(f"    H_QP ({M},{N}) done in {dt1:.0f}s", flush=True)

    # Step 2: MGS on A-weighted H_QP
    t2 = time.perf_counter()
    print("  [2/4] MGS on A-weighted H_QP...", flush=True)
    basis, d = backend.build_basis(H_QP, E0_P, verbose=True)
    dt2 = time.perf_counter() - t2
    print(f"    Basis ({M},{d}) done in {dt2:.0f}s", flush=True)

    # Step 3: Projected blocks (H_{Q̃Q̃} + H_{PQ̃} via sigma_full)
    t3 = time.perf_counter()
    print("  [3/4] Projected Hamiltonian blocks...", flush=True)
    H_QQ_tilde, H_PQ_tilde = backend.build_projected_blocks(
        basis, p_dets, verbose=True)
    dt3 = time.perf_counter() - t3
    print(f"    H_QQ_tilde ({d},{d})  H_PQ_tilde ({N},{d})  in {dt3:.0f}s",
          flush=True)

    # Step 4: Effective Hamiltonian
    t4 = time.perf_counter()
    print("  [4/4] Effective Hamiltonian...", flush=True)
    H_eff = build_effective_H(H_PP, H_PQ_tilde, H_QQ_tilde, E0_P, delta=0.0)
    ev_kdci, ev_all = diagonalize_effective_H(H_eff, n_states=None)
    dt4 = time.perf_counter() - t4

    dt_kdci = time.perf_counter() - t_kdci
    print(f"\n  Krylov-dCI wall: {dt_kdci:.0f}s "
          f"({dt1:.0f}s H_QP, {dt2:.0f}s MGS, {dt3:.0f}s proj, {dt4:.1f}s effH)",
          flush=True)

    # ── Results ──
    print(f"\n{'─'*60}\nResults\n{'─'*60}")
    nr = min(args.nroots, len(ev_kdci))
    print(f"  {'State':>6s}  {'E(FCI)/Ha':>16s}  {'E(kDCI)/Ha':>16s}  {'Δ(mH)':>10s}")
    for st in range(nr):
        Ek = ev_kdci[st] + ecore
        Eref = e_fci[st]
        dmH = (ev_kdci[st] - e_fci_bare[st]) * 1000
        print(f"  {st:6d}  {Eref:16.8f}  {Ek:16.8f}  {dmH:+10.2f}")

    # ── Summary ──
    dt_total = time.perf_counter() - t0
    print(f"\n  Total wall: {dt_total:.0f}s  d = {d} (from N = {N})")
    mem_MB = (M * 8  +  M * d * 8  +  M * N * 8) / 1e6
    print(f"  Memory: ~{mem_MB:.0f} MB (vec + basis + H_QP)")

    # ── Benchmark vs Phase 13 sparse (if requested) ───────────────
    if args.benchmark:
        print(f"\n{'─'*60}\nBenchmark: Phase 14 (Dense) vs Phase 13 (Sparse)")
        print(f"{'─'*60}", flush=True)

        from src_mf.sparse_vector import SparseQVector
        from src_mf.sparse_ops import build_hqp_sparse, sparse_mgs, project_hqq, project_hpq

        def A_func(a_str, b_str):
            return 1.0 / (E0_P - ham.diagonal_element(int(a_str), int(b_str)))

        # Sparse H_QP
        t_s1 = time.perf_counter()
        cols_s = build_hqp_sparse(p_dets, ham, A_func, n_act, skip_P=p_set)
        dt_s1 = time.perf_counter() - t_s1

        # Sparse MGS
        t_s2 = time.perf_counter()
        U_s = sparse_mgs(cols_s, [])
        d_s = len(U_s)
        dt_s2 = time.perf_counter() - t_s2

        dt_s3 = dt_s4 = 0.0; ev_s = []
        if d_s > 0:
            t_s3 = time.perf_counter()
            def df(a, b): return ham.diagonal_element(int(a), int(b))
            H_QQ_s = project_hqq(U_s, ham, n_act, df)
            dt_s3 = time.perf_counter() - t_s3

            t_s4 = time.perf_counter()
            H_PQ_s = project_hpq(p_dets, U_s, ham, n_act, skip_P=p_set)
            dt_s4 = time.perf_counter() - t_s4

            H_eff_s = build_effective_H(H_PP, H_PQ_s, H_QQ_s, E0_P, delta=0.0)
            ev_s, _ = diagonalize_effective_H(H_eff_s, n_states=None)

        dt_sparse = dt_s1 + dt_s2 + dt_s3 + dt_s4
        print(f"\n  {'Step':<25s}  {'Dense':>10s}  {'Sparse':>10s}  {'Speedup':>10s}")
        print(f"  {'─'*25}  {'─'*10}  {'─'*10}  {'─'*10}")
        for name, dt_d, dt_s in [
            ("H_QP construction", dt1, dt_s1),
            ("MGS", dt2, dt_s2),
            ("H_{Q̃Q̃} projection", dt3, dt_s3),
            ("H_{PQ̃} projection", 0.0, dt_s4),
        ]:
            dd = dt_d if dt_d > 0 else 0.001
            sp = dt_s / dd
            print(f"  {name:<25s}  {str(round(dt_d,1)):>10s}  {str(round(dt_s,1)):>10s}  {sp:9.1f}×")

        print(f"  {'─'*25}  {'─'*10}  {'─'*10}  {'─'*10}")
        sp_tot = dt_sparse / max(dt_kdci, 0.001)
        print(f"  {'Total kDCI':<25s}  {str(round(dt_kdci,1)):>10s}  {str(round(dt_sparse,1)):>10s}  {sp_tot:9.1f}×")

        if d_s > 0 and d > 0 and len(ev_s) > 0:
            print(f"\n  Accuracy: d_dense={d}, d_sparse={d_s}")
            nr = min(args.nroots, len(ev_kdci), len(ev_s))
            for st in range(nr):
                diff = abs(ev_kdci[st] - ev_s[st]) * 1e9
                print(f"    State {st}: dense={ev_kdci[st]+ecore:.8f}  "
                      f"sparse={ev_s[st]+ecore:.8f}  diff={diff:.1f} nH")


if __name__ == '__main__':
    main()
