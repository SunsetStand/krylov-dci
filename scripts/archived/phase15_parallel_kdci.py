#!/usr/bin/env python3
"""
Phase 15 — Parallel + Vectorized Krylov-dCI projection.

Optimizations over Phase 14:
  1. Vectorized H_{Q̃Q̃}: basis^T @ sigma_all (single BLAS matmul)
     replaces d² Python-level np.dot calls
  2. Parallel sigma: ThreadPoolExecutor parallelizes d independent
     contract_2e calls (C-level libfci, releases GIL)

Usage:
    python phase15_parallel_kdci.py --system N2 --P 200 --workers 8
    python phase15_parallel_kdci.py --system H2O --P 50 --scan-workers
"""

import sys, os, time, argparse
import numpy as np
from numpy.linalg import eigh

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
    p.add_argument('--workers', type=int, default=8,
                   help='Number of threads for projection (default: 8)')
    p.add_argument('--scan-workers', action='store_true',
                   help='Scan workers=1,2,4,8,16,32 and report timings')
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


def run_kdci(q_idx, ham, p_dets, nroots, n_workers, label):
    """Run KDCI with given worker count, return (ev, timings)."""
    backend = KDCIBackend(q_idx)
    N = len(p_dets)
    
    t0 = time.perf_counter()
    H_QP = backend.build_hqp(p_dets, verbose=False)
    dt_hqp = time.perf_counter() - t0
    
    H_PP = build_H_PP(ham, p_dets)
    E0_P = float(eigh(H_PP)[0][0])
    
    t0 = time.perf_counter()
    basis, d = backend.build_basis(H_QP, E0_P, verbose=False)
    dt_basis = time.perf_counter() - t0
    
    t0 = time.perf_counter()
    H_QQ_t, H_PQ_t = backend.build_projected_blocks(
        basis, p_dets, n_workers=n_workers, verbose=False)
    dt_proj = time.perf_counter() - t0
    
    t0 = time.perf_counter()
    H_eff = build_effective_H(H_PP, H_PQ_t, H_QQ_t, E0_P, delta=0.0)
    ev, _ = diagonalize_effective_H(H_eff, n_states=None)
    dt_effh = time.perf_counter() - t0
    
    print(f"  {label:>12s}: H_QP={dt_hqp:.1f}s  MGS={dt_basis:.1f}s  "
          f"proj={dt_proj:.1f}s  effH={dt_effh:.1f}s  "
          f"total={dt_hqp+dt_basis+dt_proj+dt_effh:.1f}s  "
          f"d={d}  E₀={ev[0]:.8f}")
    
    return ev, {
        'hqp': dt_hqp, 'basis': dt_basis, 'proj': dt_proj,
        'effh': dt_effh, 'd': d
    }


def main():
    args = parse_args()
    spec = SYSTEMS[args.system]

    print(f"Phase 15: Parallel Krylov-dCI")
    print(f"  {args.system} {spec['basis']} CAS({spec['n_cas']},{spec['n_elec']}) "
          f"P={args.P}\n")

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
    print(f"  P = {len(p_dets)} dets")

    if args.scan_workers:
        # ── Worker scan ──
        print(f"\n{'─'*70}")
        print(f"Worker scan")
        print(f"{'─'*70}")
        workers_list = [1, 2, 4, 8, 16, 32]
        timings = {}
        for nw in workers_list:
            ev, t = run_kdci(q_idx, ham, p_dets, args.nroots, nw, f"w={nw:2d}")
            timings[nw] = t

        print(f"\n{'─'*70}")
        print(f"Speedup summary (vs serial projection)")
        print(f"{'─'*70}")
        base = timings[1]['proj']
        for nw in workers_list:
            sp = base / max(timings[nw]['proj'], 0.01)
            bar = '█' * int(sp * 2)
            print(f"  workers={nw:2d}: proj={timings[nw]['proj']:.1f}s  "
                  f"speedup={sp:.1f}×  {bar}")
    else:
        # ── Single run with specified workers ──
        ev, t = run_kdci(q_idx, ham, p_dets, args.nroots, args.workers,
                         f"w={args.workers}")

    print(f"\n  Total wall: {time.perf_counter()-t00:.0f}s")


if __name__ == '__main__':
    main()
