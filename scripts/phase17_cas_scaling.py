#!/usr/bin/env python3
"""
Phase 17 — CAS Scaling Benchmark: find the practical limit of matrix-free Krylov-dCI.

Tests N₂/cc-pVDZ with CAS (10,10) → (12,12) → (14,14) → (16,16).
For each CAS: FCI (or DMRG-CI if too large) reference, KDCI m=0 P=200,
reports timing, memory, and accuracy.

Key resource: 772 GB RAM on amd-cpu node.

Usage:
    python phase17_cas_scaling.py
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--P', type=int, default=200)
    p.add_argument('--nroots', type=int, default=3)
    p.add_argument('--max-cas', type=int, default=14,
                   help='Max CAS n_act to test (default: 14)')
    return p.parse_args()


def setup_n2_cas(n_act, n_elec):
    """Set up N₂/cc-pVDZ CASCI with specified active space."""
    mol = gto.M(atom='N 0 0 0; N 0 0 1.10', basis='cc-pVDZ', verbose=0)
    mf = scf.RHF(mol)
    mf.kernel()

    n_total_orb = mol.nao  # 28 for cc-pVDZ
    n_core = (n_total_orb - n_act) // 2  # doubly occupied core orbitals
    if n_core < 0:
        raise ValueError(f"n_act={n_act} exceeds total orbitals {n_total_orb}")

    cas = mcscf.CASCI(mf, n_act, n_elec)
    cas.frozen = n_core
    cas.mo_coeff = mf.mo_coeff
    h1eff, ecore = cas.get_h1eff()
    h2eff = cas.get_h2eff()
    return h1eff, h2eff, float(ecore), n_act, n_core


def select_p_perturbative(ham, n_act, na, nb, P_target):
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
        for j in alpha_occ[ii+1:]:
            for ia, a in enumerate(av):
                for b in av[ia+1:]:
                    det = ((hf_a^(1<<i)^(1<<j))|(1<<a)|(1<<b), hf_b)
                    hij = ham.matrix_element((hf_a,hf_b),det)
                    hdd = ham.diagonal_element(det[0],det[1])
                    d = E_HF - hdd
                    if abs(d) > 1e-12: scores.append((det, -(hij*hij)/d))
    # ββ doubles
    for ii, i in enumerate(beta_occ):
        for j in beta_occ[ii+1:]:
            for ia, a in enumerate(bv):
                for b in bv[ia+1:]:
                    det = (hf_a, (hf_b^(1<<i)^(1<<j))|(1<<a)|(1<<b))
                    hij = ham.matrix_element((hf_a,hf_b),det)
                    hdd = ham.diagonal_element(det[0],det[1])
                    d = E_HF - hdd
                    if abs(d) > 1e-12: scores.append((det, -(hij*hij)/d))
    # αβ doubles
    for i in alpha_occ:
        for j in beta_occ:
            for a in av:
                for b in bv:
                    det = ((hf_a^(1<<i))|(1<<a), (hf_b^(1<<j))|(1<<b))
                    hij = ham.matrix_element((hf_a,hf_b),det)
                    hdd = ham.diagonal_element(det[0],det[1])
                    d = E_HF - hdd
                    if abs(d) > 1e-12: scores.append((det, -(hij*hij)/d))

    scores.sort(key=lambda x: x[1], reverse=True)
    P_actual = min(P_target - 1, len(scores))
    return [(hf_a, hf_b)] + [d for d, _ in scores[:P_actual]]


def build_H_PP(ham, p_dets):
    N = len(p_dets)
    H_PP = np.zeros((N, N))
    diag_bulk = ham.diagonal_elements_bulk(p_dets)
    np.fill_diagonal(H_PP, diag_bulk)
    for i in range(N):
        for j in range(i + 1, N):
            h = ham.matrix_element(p_dets[i], p_dets[j])
            H_PP[i, j] = h; H_PP[j, i] = h
    return H_PP


def run_cas(n_act, n_elec, P_target, nroots, max_fci_M=1_000_000):
    """Run KDCI on one CAS size. Returns results dict."""
    print(f"\n{'='*60}")
    print(f"CAS({n_act},{n_elec}) — N₂/cc-pVDZ")
    print(f"{'='*60}")

    t_start = time.perf_counter()

    # ── Setup ──
    h1eff, h2eff, ecore, n_act, n_core = setup_n2_cas(n_act, n_elec)
    h2_4d = _unpack_4fold(h2eff, n_act)
    na = n_elec // 2; nb = n_elec - na; nelec = (na, nb)

    qa = np.array(cistring.gen_strings4orblist(range(n_act), na), dtype=np.int64)
    qb = np.array(cistring.gen_strings4orblist(range(n_act), nb), dtype=np.int64)
    q_idx = QSpaceIndex(qa, qb, n_act, nelec, h1eff, h2eff)
    M = q_idx.M
    print(f"  M = {M:,} ({len(qa):,} α × {len(qb):,} β), "
          f"n_core = {n_core}")

    # ── FCI reference (if feasible) ──
    if M <= max_fci_M:
        fs = direct_spin1.FCI()
        fs.conv_tol = 1e-10; fs.nroots = nroots
        t_fci = time.perf_counter()
        e_ref, _ = fs.kernel(h1eff, h2eff, n_act, nelec, ecore=ecore)
        dt_fci = time.perf_counter() - t_fci
        ref_method = "FCI"
        print(f"  FCI E₀ = {e_ref[0]:.10f} Ha ({dt_fci:.0f}s)")
    else:
        # Use P-space energy as upper-bound reference
        e_ref = None
        ref_method = "N/A (M too large)"
        print(f"  FCI not feasible (M={M:,} > {max_fci_M:,})")

    # ── P-space selection ──
    ham = Hamiltonian(h1=h1eff, h2=h2_4d, E_nuc=0.0, E_HF=0.0)
    t_psel = time.perf_counter()
    p_dets = select_p_perturbative(ham, n_act, na, nb, P_target)
    N = len(p_dets)
    dt_psel = time.perf_counter() - t_psel
    print(f"  P = {N} dets ({dt_psel:.1f}s)")

    H_PP = build_H_PP(ham, p_dets)
    E0_P = float(np.linalg.eigh(H_PP)[0][0])

    if e_ref is not None:
        e_fci_bare = e_ref - ecore
        print(f"  E₀(P) − E(FCI) = {(E0_P - e_fci_bare[0]) * 1000:+.1f} mH")

    # ── Krylov-dCI (matrix-free sparse) ──
    backend = KDCIBackend(q_idx)
    t_kdci = time.perf_counter()

    # Streaming basis
    t1 = time.perf_counter()
    basis_sparse, d = backend.build_basis_streaming(p_dets, E0_P, verbose=False)
    dt_basis = time.perf_counter() - t1

    # Sparse projection
    t2 = time.perf_counter()
    H_QQ_t, H_PQ_t = backend.build_projected_blocks_sparse(
        basis_sparse, p_dets, verbose=False)
    dt_proj = time.perf_counter() - t2

    # Effective Hamiltonian
    H_eff = build_effective_H(H_PP, H_PQ_t, H_QQ_t, E0_P, delta=0.0)
    ev_mf, _ = diagonalize_effective_H(H_eff, n_states=None)
    dt_kdci = time.perf_counter() - t_kdci

    total_nnz = sum(b.nnz() for b in basis_sparse)
    mem_basis = total_nnz * 88 / 1e6  # ~88 bytes per dict entry
    mem_temp = M * 8 / 1e6

    # ── Results ──
    total_wall = time.perf_counter() - t_start
    print(f"\n  KDCI wall: {dt_kdci:.0f}s "
          f"(basis={dt_basis:.0f}s, proj={dt_proj:.0f}s)")
    print(f"  d = {d}  avg_nnz = {total_nnz/max(d,1):.0f}")
    print(f"  Memory: ~{mem_basis:.0f} MB (basis) + ~{mem_temp:.1f} MB (temp)")

    if e_ref is not None:
        nr = min(nroots, len(ev_mf))
        print(f"  {'State':>6s}  {'E(ref)/Ha':>16s}  {'E(kDCI)/Ha':>16s}  {'Δ(mH)':>10s}")
        for st in range(nr):
            Ek = ev_mf[st] + ecore
            dmH = (ev_mf[st] - e_fci_bare[st]) * 1000
            print(f"  {st:6d}  {e_ref[st]:16.8f}  {Ek:16.8f}  {dmH:+10.2f}")
    else:
        print(f"  E₀(kDCI) = {ev_mf[0] + ecore:.10f} Ha")

    print(f"  Total wall: {total_wall:.0f}s")

    return {
        'n_act': n_act, 'M': M, 'd': d, 'N': N,
        'dt_basis': dt_basis, 'dt_proj': dt_proj,
        'dt_kdci': dt_kdci, 'total_wall': total_wall,
        'mem_basis': mem_basis, 'mem_temp': mem_temp,
        'avg_nnz': total_nnz / max(d, 1),
        'ref_method': ref_method,
        'ev': ev_mf, 'ecore': ecore,
    }


def main():
    args = parse_args()

    print("Phase 17: CAS Scaling Benchmark — N₂/cc-pVDZ")
    print(f"  P = {args.P}, nroots = {args.nroots}\n")

    cas_list = [(10, 10), (12, 12)]
    if args.max_cas >= 14:
        cas_list.append((14, 14))
    if args.max_cas >= 16:
        cas_list.append((16, 16))

    results = []
    for n_act, n_elec in cas_list:
        try:
            r = run_cas(n_act, n_elec, args.P, args.nroots)
            results.append(r)
        except Exception as e:
            print(f"  ✗ CAS({n_act},{n_elec}) FAILED: {e}")

    # ── Summary table ──
    print(f"\n{'='*80}")
    print("Summary")
    print(f"{'='*80}")
    print(f"  {'CAS':>10s}  {'M':>12s}  {'d':>6s}  "
          f"{'Basis':>8s}  {'Proj':>8s}  {'Total':>8s}  "
          f"{'Memory':>10s}  {'avg_nnz':>8s}  {'ΔE₀(mH)':>10s}")
    print(f"  {'-'*10}  {'-'*12}  {'-'*6}  "
          f"{'-'*8}  {'-'*8}  {'-'*8}  "
          f"{'-'*10}  {'-'*8}  {'-'*10}")
    for r in results:
        dmH = ""
        if r.get('e_ref') is not None:
            dmH = f"{(r['ev'][0] - (r['e_ref'][0] - r['ecore'])) * 1000:+.1f}"
        print(f"  ({r['n_act']},{r['n_act']})  {r['M']:>12,d}  {r['d']:>6d}  "
              f"{r['dt_basis']:>7.0f}s  {r['dt_proj']:>7.0f}s  "
              f"{r['dt_kdci']:>7.0f}s  "
              f"{r['mem_basis']+r['mem_temp']:>9.0f} MB  "
              f"{r['avg_nnz']:>8.0f}  "
              f"{dmH:>10s}")

    print(f"\n  Node: {os.uname().nodename}")
    print(f"  Total wall: {sum(r['total_wall'] for r in results):.0f}s")


if __name__ == '__main__':
    main()
