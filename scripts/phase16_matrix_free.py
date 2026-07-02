#!/usr/bin/env python3
"""
Phase 16 — Matrix-Free Krylov-dCI (C-level sigma, sparse storage).

NEVER stores M-dimensional dense vectors persistently:
  - build_basis_streaming: N contract_2e calls, extract sparse, stream MGS
  - build_projected_blocks_sparse: d contract_2e calls, sparse-dense dots

Only temporary dense objects: one (na,nb) CI matrix per contract_2e call.
Persistent: d SparseQVector objects (dict-based, only non-zero entries).

Memory scaling:
  CAS(10,10): ~177 MB (basis) + 0.5 MB (temp) = ~178 MB
  CAS(14,14): ~1.7 GB (basis) + 94 MB (temp) = ~1.8 GB
  CAS(16,16): ~17.6 GB (basis) + 1.3 GB (temp) = ~19 GB

Usage:
    python phase16_matrix_free.py --system N2 --P 200
    python phase16_matrix_free.py --system N2 --P 200 --dense-compare
"""

import sys, os, time, argparse
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from pyscf import gto, scf, mcscf
from pyscf.fci import cistring, direct_spin1

from hamiltonian import Hamiltonian, _unpack_4fold
from effective_h import build_effective_H, diagonalize_effective_H
from src_mf.pyscf_backend import QSpaceIndex, KDCIBackend

np.set_printoptions(linewidth=140, precision=6, suppress=True)


SYSTEMS = {
    'H2O': {
        'atom': 'O 0 0 0; H 1.0 0 0; H -0.2774 0.9605 0',
        'basis': 'sto-3g', 'n_cas': 4, 'n_elec': 4, 'n_core': 1,
    },
    'N2': {
        'atom': 'N 0 0 0; N 0 0 1.10',
        'basis': 'cc-pVDZ', 'n_cas': 10, 'n_elec': 10, 'n_core': 3,
    },
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--system', default='N2', choices=list(SYSTEMS.keys()))
    p.add_argument('--P', type=int, default=200)
    p.add_argument('--nroots', type=int, default=6)
    p.add_argument('--dense-compare', action='store_true',
                   help='Compare sparse vs dense results and timings')
    return p.parse_args()


def select_p_perturbative(ham, n_act, na, nb, P_target):
    from src.determinants import hf_determinant, bit_positions
    hf_a, hf_b = hf_determinant(na, nb)
    E_HF = ham.diagonal_element(hf_a, hf_b)
    all_orbs = list(range(n_act))
    alpha_occ = bit_positions(hf_a); beta_occ = bit_positions(hf_b)
    av = [p for p in all_orbs if p not in alpha_occ]
    bv = [p for p in all_orbs if p not in beta_occ]
    scores = []
    for ii, i in enumerate(alpha_occ):
        for j in alpha_occ[ii+1:]:
            for ia, a in enumerate(av):
                for b in av[ia+1:]:
                    det = ((hf_a^(1<<i)^(1<<j))|(1<<a)|(1<<b), hf_b)
                    hij = ham.matrix_element((hf_a,hf_b),det)
                    hdd = ham.diagonal_element(det[0],det[1])
                    d = E_HF-hdd
                    if abs(d)>1e-12: scores.append((det,-(hij*hij)/d))
    for ii, i in enumerate(beta_occ):
        for j in beta_occ[ii+1:]:
            for ia, a in enumerate(bv):
                for b in bv[ia+1:]:
                    det = (hf_a,(hf_b^(1<<i)^(1<<j))|(1<<a)|(1<<b))
                    hij = ham.matrix_element((hf_a,hf_b),det)
                    hdd = ham.diagonal_element(det[0],det[1])
                    d = E_HF-hdd
                    if abs(d)>1e-12: scores.append((det,-(hij*hij)/d))
    for i in alpha_occ:
        for j in beta_occ:
            for a in av:
                for b in bv:
                    det = ((hf_a^(1<<i))|(1<<a),(hf_b^(1<<j))|(1<<b))
                    hij = ham.matrix_element((hf_a,hf_b),det)
                    hdd = ham.diagonal_element(det[0],det[1])
                    d = E_HF-hdd
                    if abs(d)>1e-12: scores.append((det,-(hij*hij)/d))
    scores.sort(key=lambda x: x[1], reverse=True)
    P_actual = min(P_target-1, len(scores))
    return [(hf_a,hf_b)] + [d for d,_ in scores[:P_actual]]


def build_H_PP(ham, p_dets):
    N = len(p_dets)
    H_PP = np.zeros((N,N))
    diag_bulk = ham.diagonal_elements_bulk(p_dets)
    np.fill_diagonal(H_PP, diag_bulk)
    for i in range(N):
        for j in range(i+1,N):
            h = ham.matrix_element(p_dets[i],p_dets[j])
            H_PP[i,j]=h; H_PP[j,i]=h
    return H_PP


def main():
    args = parse_args()
    spec = SYSTEMS[args.system]

    print(f"Phase 16: Matrix-Free Krylov-dCI")
    print(f"  {args.system} {spec['basis']} CAS({spec['n_cas']},{spec['n_elec']}) "
          f"P={args.P}  nroots={args.nroots}\n")

    t00 = time.perf_counter()

    # ── Setup ──
    mol = gto.M(atom=spec['atom'], basis=spec['basis'], verbose=0)
    mf = scf.RHF(mol); mf.kernel()
    n_act, n_elec = spec['n_cas'], spec['n_elec']
    na = n_elec // 2; nb = n_elec - na; nelec = (na, nb)

    cas = mcscf.CASCI(mf, n_act, n_elec)
    cas.frozen = spec['n_core']; cas.mo_coeff = mf.mo_coeff
    h1eff, ecore = cas.get_h1eff(); h2eff = cas.get_h2eff()
    h2_4d = _unpack_4fold(h2eff, n_act); ecore = float(ecore)

    qa = np.array(cistring.gen_strings4orblist(range(n_act), na), dtype=np.int64)
    qb = np.array(cistring.gen_strings4orblist(range(n_act), nb), dtype=np.int64)
    q_idx = QSpaceIndex(qa, qb, n_act, nelec, h1eff, h2eff)
    M = q_idx.M

    fs = direct_spin1.FCI(); fs.conv_tol = 1e-10; fs.nroots = args.nroots
    e_fci, _ = fs.kernel(h1eff, h2eff, n_act, nelec, ecore=ecore)
    e_fci_bare = e_fci - ecore
    print(f"  FCI E₀ = {e_fci[0]:.8f} Ha  M = {M:,}\n")

    ham = Hamiltonian(h1=h1eff, h2=h2_4d, E_nuc=0.0, E_HF=0.0)
    p_dets = select_p_perturbative(ham, n_act, na, nb, args.P)
    N = len(p_dets)
    H_PP = build_H_PP(ham, p_dets)
    E0_P = float(np.linalg.eigh(H_PP)[0][0])
    print(f"  P = {N} dets  E₀(P)−E(FCI) = {(E0_P-e_fci_bare[0])*1000:+.1f} mH\n")

    backend = KDCIBackend(q_idx)

    # ═══════════════════════════════════════════════════════════════
    # Matrix-Free Krylov-dCI (m=0)
    # ═══════════════════════════════════════════════════════════════
    print(f"{'─'*60}")
    print(f"Matrix-Free Krylov-dCI (m=0)")
    print(f"{'─'*60}")

    t_kdci = time.perf_counter()

    # Step 1: Streaming build basis (no H_QP stored)
    t1 = time.perf_counter()
    print("  [1/3] Building basis via streaming MGS...", flush=True)
    basis_sparse, d = backend.build_basis_streaming(
        p_dets, E0_P, verbose=True)
    dt1 = time.perf_counter() - t1
    total_nnz = sum(b.nnz() for b in basis_sparse)
    print(f"    Basis: {N} → {d} vectors, {total_nnz} nnz, {dt1:.0f}s",
          flush=True)

    # Step 2: Sparse projection
    t2 = time.perf_counter()
    print("  [2/3] Sparse projected Hamiltonian blocks...", flush=True)
    H_QQ_tilde, H_PQ_tilde = backend.build_projected_blocks_sparse(
        basis_sparse, p_dets, verbose=True)
    dt2 = time.perf_counter() - t2
    print(f"    H_QQ_tilde ({d},{d})  H_PQ_tilde ({N},{d})  {dt2:.0f}s",
          flush=True)

    # Step 3: Effective Hamiltonian
    t3 = time.perf_counter()
    print("  [3/3] Effective Hamiltonian...", flush=True)
    H_eff = build_effective_H(H_PP, H_PQ_tilde, H_QQ_tilde, E0_P, delta=0.0)
    ev_mf, _ = diagonalize_effective_H(H_eff, n_states=None)
    dt3 = time.perf_counter() - t3

    dt_kdci = time.perf_counter() - t_kdci
    print(f"\n  Matrix-free wall: {dt_kdci:.0f}s "
          f"({dt1:.0f}s basis, {dt2:.0f}s proj, {dt3:.1f}s effH)",
          flush=True)

    # ── Results ──
    print(f"\n{'─'*60}\nResults\n{'─'*60}")
    nr = min(args.nroots, len(ev_mf))
    print(f"  {'State':>6s}  {'E(FCI)/Ha':>16s}  {'E(kDCI)/Ha':>16s}  {'Δ(mH)':>10s}")
    for st in range(nr):
        Ek = ev_mf[st] + ecore
        Eref = e_fci[st]
        dmH = (ev_mf[st] - e_fci_bare[st]) * 1000
        print(f"  {st:6d}  {Eref:16.8f}  {Ek:16.8f}  {dmH:+10.2f}")

    # ── Memory usage ──
    mem_basis = total_nnz * 88 / 1e6  # ~88 bytes per dict entry (Python overhead)
    mem_temp = M * 8 / 1e6           # one dense CI matrix
    print(f"\n  Memory estimate: ~{mem_basis:.0f} MB (basis) + "
          f"~{mem_temp:.1f} MB (temp)")

    dt_total = time.perf_counter() - t00
    print(f"  Total wall: {dt_total:.0f}s")

    # ── Dense comparison (optional) ──
    if args.dense_compare:
        print(f"\n{'─'*60}")
        print(f"Dense comparison (Phase 15)")
        print(f"{'─'*60}")
        t_dense = time.perf_counter()

        backend2 = KDCIBackend(q_idx)
        H_QP = backend2.build_hqp(p_dets, verbose=False)
        basis_dense, dd = backend2.build_basis(H_QP, E0_P, verbose=False)
        H_QQ_d, H_PQ_d = backend2.build_projected_blocks(
            basis_dense, p_dets, verbose=False)
        H_eff_d = build_effective_H(H_PP, H_PQ_d, H_QQ_d, E0_P, delta=0.0)
        ev_dense, _ = diagonalize_effective_H(H_eff_d, n_states=None)
        dt_dense = time.perf_counter() - t_dense

        print(f"  Dense wall: {dt_dense:.0f}s  d={dd}")
        print(f"  Sparse wall: {dt_kdci:.0f}s  d={d}")
        nr = min(args.nroots, len(ev_mf), len(ev_dense))
        print(f"  {'State':>6s}  {'Sparse/Ha':>16s}  {'Dense/Ha':>16s}  {'diff/nH':>10s}")
        for st in range(nr):
            diff = abs(ev_mf[st] - ev_dense[st]) * 1e9
            print(f"  {st:6d}  {ev_mf[st]+ecore:16.8f}  "
                  f"{ev_dense[st]+ecore:16.8f}  {diff:10.1f}")


if __name__ == '__main__':
    main()
