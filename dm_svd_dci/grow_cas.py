#!/usr/bin/env python3
"""
Growing active-space DCI: sequentially enlarge the CAS and run dmSVD + Neumann Heff.

Simplified approach — each round is an independent dmSVD calculation
on a progressively larger active space. Observes energy convergence
as the active space approaches the full CAS.

Round 0:  CASCI(n_occ_A + n_orb_B0) → dmSVD → Neumann Heff → E₀
Round 1:  CASCI(n_occ_A + n_orb_B0 + n_orb_Bt) → dmSVD → Neumann → E₁
Round 2:  CASCI(...) → ... → E₂
...
Round N:  CASCI(n_active_total) = FCI reference

Configurable: n_occ_A, n_orb_B0, n_orb_Bt, n_core, p_blocks, svd_eps, k_max.

Usage:
    python dm_svd_dci/grow_cas.py \
        --atom 'N 0 0 0; N 0 0 1.098' --basis cc-pVDZ \
        --n-active 10 --n-alpha 5 --n-beta 5 --n-core 2 \
        --n-occ-A 6 --n-orb-B0 1 --n-orb-Bt 1 \
        --p-blocks 10 --svd-eps 1e-3 --k-max 1 \
        --n-workers 16 --output-dir ./results/grow_n2
"""

import sys, os, time, json, argparse
import numpy as np
from numpy.linalg import eigh
from typing import Dict, List, Tuple, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def run_single_round(
    mf,
    mol,
    n_core: int,
    active_orb_indices: np.ndarray,
    n_active_elec: Tuple[int, int],
    n_occ_A: int,
    p_blocks: List[int],
    svd_eps: float,
    k_max: int,
    n_workers: int,
    scheme: str = 'A',
    verbose: bool = True,
) -> Dict:
    """Run one round of dmSVD + Neumann Heff.

    Args:
        mf:           PySCF RHF object.
        mol:          PySCF Mole object.
        n_core:       Frozen core orbitals.
        active_orb_indices: Which orbitals (global index) form the active space.
        n_active_elec: (na, nb) active electrons.
        n_occ_A:      A-space orbitals for occ-virt partition.
        p_blocks:     P-space n-blocks.
        svd_eps:      SVD truncation threshold.
        k_max:        Neumann order (0 or 1).
        n_workers:    Parallel sigma workers.
        scheme:       'A' (full H^emb) or 'B' (direct blocks).
        verbose:      Print diagnostics.

    Returns:
        dict with E_fci, E_bare, E_neumann, timing, schmidt_metrics, etc.
    """
    from pyscf import mcscf
    from pyscf.fci import cistring
    from src_mf.pyscf_backend import QSpaceIndex, KDCIBackend
    from src.hamiltonian import _unpack_4fold
    from dm_svd_embedding.occ_virt_partition import (
        setup_partition, build_block_matrices)
    from dm_svd_embedding.density_matrix import (
        compute_schmidt_decomposition, compute_compression_metrics)
    from dm_svd_dci.schmidt_partition import partition_schmidt_basis
    from dm_svd_dci.qspace_partition import (
        partition_qspace_by_n, extract_q_blocks_scheme_a,
        extract_q_blocks_scheme_b)
    from dm_svd_dci.neumann_effective_ham import (
        build_effective_hamiltonian_neumann)

    t0 = time.perf_counter()
    n_act = len(active_orb_indices)
    n_elec_total = sum(n_active_elec)
    na, nb = n_active_elec

    # ── Build CASCI in the active subspace ──
    # Reorder MOs so active subspace is contiguous after core
    n_total_mo = mol.nao_nr()
    core_indices = list(range(n_core))
    # active_orb_indices are global (core-relative): we need to add n_core offset
    active_global = [n_core + i for i in active_orb_indices]
    used = set(core_indices) | set(active_global)
    rest_indices = [i for i in range(n_total_mo) if i not in used]
    new_order = core_indices + active_global + rest_indices

    mo_new = mf.mo_coeff[:, new_order]
    mf2 = mf.copy()
    mf2.mo_coeff = mo_new

    cas = mcscf.CASCI(mf2, n_act, n_elec_total)
    cas.frozen = n_core
    cas.kernel()

    h1eff, ecore = cas.get_h1eff()
    h2eff = cas.get_h2eff()
    fcivec = cas.ci
    E_fci = cas.e_tot

    alpha_strs = cistring.gen_strings4orblist(range(n_act), na)
    beta_strs = cistring.gen_strings4orblist(range(n_act), nb)

    q_idx = QSpaceIndex(alpha_strs, beta_strs, n_act, n_active_elec, h1eff, h2eff)
    backend = KDCIBackend(q_idx)
    h2_4d = _unpack_4fold(h2eff, n_act)

    if verbose:
        print(f"\n  CASCI({n_act},{n_elec_total}): E={E_fci:.12f} Ha, "
              f"M={q_idx.M:,} dets, {time.perf_counter()-t0:.1f}s", flush=True)

    # ── dmSVD ──
    t_svd = time.perf_counter()
    partition, full_dets = setup_partition(n_act, n_elec_total, n_occ_A, ms=0)
    ci_flat = fcivec.reshape(-1)
    C_blocks = build_block_matrices(partition, ci_flat)
    schmidt = compute_schmidt_decomposition(C_blocks, eps=svd_eps)
    metrics = compute_compression_metrics(schmidt, C_blocks, ci_flat)

    D_schmidt = sum(sd.get('r', 0)**2 for sd in schmidt.values())

    if verbose:
        print(f"    dmSVD: r_total={metrics['r_total']}, D={D_schmidt}, "
              f"compression={metrics['compression_ratio']:.4%}, "
              f"{time.perf_counter()-t_svd:.1f}s", flush=True)

    # ── Schmidt partition + Q-space subdivision ──
    t_part = time.perf_counter()
    part = partition_schmidt_basis(schmidt, p_blocks=p_blocks)
    q_partition = partition_qspace_by_n(part, schmidt, p_blocks=p_blocks)

    if verbose:
        print(f"    Partition: |P|={part['p_dim']}, |Q|={part['q_dim']}, "
              f"active Q_n: {q_partition['active_q']}")

    if scheme == 'A':
        from dm_svd_dci._legacy_pipeline import build_hemb_parallel
        H_emb, basis_info, hemb_norms = build_hemb_parallel(
            schmidt, partition, q_idx, backend,
            h1_full=h1eff, h2_full=h2_4d,
            n_occ=n_occ_A, n_act=n_act,
            n_workers=n_workers, verbose=verbose)
        if D_schmidt > 0:
            H_emb += ecore * np.eye(D_schmidt)
        q_blocks_data = extract_q_blocks_scheme_a(
            H_emb, part, q_partition, p_blocks=p_blocks, verbose=verbose)
    else:
        q_blocks_data = extract_q_blocks_scheme_b(
            schmidt, partition, part, q_partition,
            p_blocks, backend, n_occ_A, n_act,
            n_workers=n_workers, ecore=float(ecore), verbose=verbose)

    H_PP = q_blocks_data['H_PP']
    H_PQ = q_blocks_data['H_PQ']
    H_QQ_blocks = q_blocks_data['H_QQ_blocks']
    H_QQ_diag = q_blocks_data['H_QQ_diag']

    if verbose:
        print(f"    Blocks: H_PP {H_PP.shape}, "
              f"H_PQ: {{{', '.join(f'{n}:{m.shape}' for n,m in H_PQ.items())}}}, "
              f"{time.perf_counter()-t_part:.1f}s", flush=True)

    # ── Bare H_PP energy ──
    E_bare = eigh(H_PP)[0][0] if H_PP.shape[0] > 0 else 0.0

    # ── Neumann Heff (handle empty Q) ──
    active_n = sorted(H_PQ.keys())
    if len(active_n) == 0 or all(H_PQ[n].shape[1] == 0 for n in active_n):
        if verbose:
            print(f"    ⚠ Q-space empty → E_neumann = E_bare")
        E_neumann = E_bare
    else:
        res = build_effective_hamiltonian_neumann(
            H_PP, H_PQ, H_QQ_blocks, H_QQ_diag,
            E_bare, delta=0.0, k_max=k_max, verbose=verbose)
        H_eff = res['H_eff']
        E_neumann = eigh(H_eff)[0][0] if H_eff.shape[0] > 0 else 0.0

    elapsed = time.perf_counter() - t0

    dE_bare = (E_bare - E_fci) * 1000
    dE_neumann = (E_neumann - E_fci) * 1000

    if verbose:
        print(f"    E(FCI)      = {E_fci:.12f} Ha")
        print(f"    E(bare H_PP) = {E_bare:.12f} Ha  (ΔE={dE_bare:+.3f} mH)")
        print(f"    E(Neumann)   = {E_neumann:.12f} Ha  (ΔE={dE_neumann:+.3f} mH)")
        print(f"    Total: {elapsed:.1f}s", flush=True)

    return {
        'n_act': n_act,
        'active_orbs': [int(x) for x in active_orb_indices],
        'n_occ_A': n_occ_A,
        'E_fci': float(E_fci),
        'E_bare': float(E_bare),
        'E_neumann': float(E_neumann),
        'dE_bare_mH': float(dE_bare),
        'dE_neumann_mH': float(dE_neumann),
        'D_schmidt': D_schmidt,
        'r_total': metrics['r_total'],
        'compression_ratio': float(metrics['compression_ratio']),
        'P_dim': part['p_dim'],
        'Q_dim': part['q_dim'],
        'elapsed': float(elapsed),
    }


def run_growing_cas(
    atom: str = 'N 0 0 0; N 0 0 1.098',
    basis: str = 'cc-pVDZ',
    n_active: int = 10,
    n_active_elec: Tuple[int, int] = (5, 5),
    n_core: int = 2,
    n_occ_A: int = 6,
    n_orb_B0: int = 1,
    n_orb_Bt: int = 1,
    p_blocks: List[int] = [10],
    svd_eps: float = 1e-3,
    k_max: int = 1,
    eps_conv: float = 1e-8,
    max_rounds: Optional[int] = None,
    n_workers: int = 1,
    scheme: str = 'A',
    output_dir: Optional[str] = None,
    verbose: bool = True,
) -> Dict:
    """Growing active-space DCI: sequentially enlarge CAS.

    Each round runs a full dmSVD + Neumann Heff on a progressively
    larger active space, converging to the full CAS result.
    """
    from pyscf import gto, scf

    t_total = time.perf_counter()

    if verbose:
        print("=" * 70)
        print("Growing Active-Space DCI")
        print("=" * 70)
        print(f"  System:     {atom.strip()}, {basis}")
        print(f"  Full CAS:   ({n_active},{sum(n_active_elec)}), n_core={n_core}")
        print(f"  Bootstrap:  A₀={n_occ_A} orbs, B₀={n_orb_B0} orbs "
              f"({n_occ_A + n_orb_B0} total)")
        print(f"  Extension:  +{n_orb_Bt} orbital(s) per round")
        print(f"  P-blocks:   {p_blocks}, SVD ε={svd_eps}, Neumann k={k_max}")
        print(f"  Scheme:     {scheme}, n_workers={n_workers}")

    # ── Molecule setup ──
    mol = gto.M(atom=atom, basis=basis, verbose=0, spin=0)
    mf = scf.RHF(mol).run(verbose=0)

    if verbose:
        print(f"\n  RHF E: {mf.e_tot:.12f} Ha, n_mo={mol.nao_nr()}")

    # ── Compute orbital energies for sorting ──
    # Active MOs (after frozen core) sorted by energy
    n_total_mo = mol.nao_nr()
    mo_energy_active = mf.mo_energy[n_core:n_core + n_active]
    orb_order = np.argsort(mo_energy_active)  # lowest energy first

    # ── Determine round sizes ──
    base_size = n_occ_A + n_orb_B0  # initial active orbitals
    remaining = n_active - base_size

    if max_rounds is None:
        max_rounds = remaining // n_orb_Bt + (1 if remaining % n_orb_Bt else 0)

    round_sizes = [base_size]
    for r in range(max_rounds):
        next_size = base_size + (r + 1) * n_orb_Bt
        if next_size <= n_active:
            round_sizes.append(next_size)
    if round_sizes[-1] < n_active:
        round_sizes.append(n_active)  # always include full CAS

    if verbose:
        print(f"\n  Round schedule ({len(round_sizes)} rounds): "
              f"{round_sizes}")
        print(f"  Orbital order (by energy): {list(orb_order)}")

    # ── Run rounds ──
    all_results = []
    E_neumann_history = []

    for r, n_orbs in enumerate(round_sizes):
        if verbose:
            print(f"\n{'='*70}")
            print(f"ROUND {r}: CASCI({n_orbs},{sum(n_active_elec)})")
            print(f"{'='*70}")

        # Select the n_orbs lowest-energy active orbitals
        active_orb_indices = orb_order[:n_orbs]

        # n_occ_A for this round: use the same value, but capped at n_orbs
        n_occ_A_round = min(n_occ_A, n_orbs - 1)

        result = run_single_round(
            mf, mol, n_core,
            active_orb_indices=active_orb_indices,
            n_active_elec=n_active_elec,
            n_occ_A=n_occ_A_round,
            p_blocks=p_blocks,
            svd_eps=svd_eps,
            k_max=k_max,
            n_workers=n_workers,
            scheme=scheme,
            verbose=verbose,
        )

        all_results.append(result)
        E_neumann_history.append(result['E_neumann'])

        # Convergence check
        if r > 0 and len(E_neumann_history) >= 2:
            dE = abs(E_neumann_history[-1] - E_neumann_history[-2])
            if verbose:
                print(f"\n  ΔE vs previous round: {dE:.3e} Ha")
            if dE < eps_conv:
                if verbose:
                    print(f"  ✓ CONVERGED: |ΔE| = {dE:.3e} < {eps_conv:.1e}")
                break

    # ── Summary ──
    elapsed_total = time.perf_counter() - t_total
    E_fci_full = all_results[-1]['E_fci']  # last round = full CAS reference

    if verbose:
        print(f"\n{'='*70}")
        print(f"CONVERGENCE SUMMARY")
        print(f"{'='*70}")
        header = f"  {'Round':>5} {'n_act':>6} {'D_schmidt':>10} " \
                 f"{'E_neumann':>18} {'dE (mH)':>12} {'time (s)':>10}"
        print(header)
        print(f"  {'-'*len(header)}")
        for r, res in enumerate(all_results):
            dE = (res['E_neumann'] - E_fci_full) * 1000
            print(f"  {r:>5} {res['n_act']:>6} {res['D_schmidt']:>10} "
                  f"{res['E_neumann']:>18.12f} {dE:>+11.3f} "
                  f"{res['elapsed']:>9.1f}")
        print(f"\n  Total wall time: {elapsed_total:.1f}s")

    # ── Build output ──
    output = {
        'config': {
            'atom': atom, 'basis': basis,
            'n_active': n_active, 'n_active_elec': list(n_active_elec),
            'n_core': n_core, 'n_occ_A': n_occ_A,
            'n_orb_B0': n_orb_B0, 'n_orb_Bt': n_orb_Bt,
            'p_blocks': p_blocks, 'svd_eps': svd_eps,
            'k_max': k_max, 'eps_conv': eps_conv,
            'scheme': scheme, 'n_workers': n_workers,
        },
        'E_fci_full': float(E_fci_full),
        'n_rounds': len(all_results),
        'converged': len(E_neumann_history) >= 2 and
                     abs(E_neumann_history[-1] - E_neumann_history[-2]) < eps_conv,
        'E_neumann_history': [float(e) for e in E_neumann_history],
        'timing_total': float(elapsed_total),
        'rounds': all_results,
    }

    # ── Save JSON ──
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fname = os.path.join(output_dir, 'grow_cas_results.json')
        with open(fname, 'w') as f:
            json.dump(output, f, indent=2)
        if verbose:
            print(f"\n  Results saved to {fname}")

    return output


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def _parse_args():
    p = argparse.ArgumentParser(
        description='Growing Active-Space DCI via dmSVD + Neumann Heff')

    p.add_argument('--atom', default='N 0 0 0; N 0 0 1.098')
    p.add_argument('--basis', default='cc-pVDZ')
    p.add_argument('--n-active', type=int, default=10)
    p.add_argument('--n-alpha', type=int, default=5)
    p.add_argument('--n-beta', type=int, default=5)
    p.add_argument('--n-core', type=int, default=2)
    p.add_argument('--n-occ-A', type=int, default=6)
    p.add_argument('--n-orb-B0', type=int, default=1)
    p.add_argument('--n-orb-Bt', type=int, default=1)
    p.add_argument('--p-blocks', default='10')
    p.add_argument('--svd-eps', type=float, default=1e-3)
    p.add_argument('--k-max', type=int, default=1)
    p.add_argument('--eps-conv', type=float, default=1e-8)
    p.add_argument('--max-rounds', type=int, default=None)
    p.add_argument('--n-workers', type=int, default=1)
    p.add_argument('--scheme', default='A', choices=['A', 'B'])
    p.add_argument('--output-dir', default=None)
    p.add_argument('--quiet', action='store_true')

    return p.parse_args()


def main():
    args = _parse_args()
    p_blocks = [int(x.strip()) for x in args.p_blocks.split(',')]

    results = run_growing_cas(
        atom=args.atom,
        basis=args.basis,
        n_active=args.n_active,
        n_active_elec=(args.n_alpha, args.n_beta),
        n_core=args.n_core,
        n_occ_A=args.n_occ_A,
        n_orb_B0=args.n_orb_B0,
        n_orb_Bt=args.n_orb_Bt,
        p_blocks=p_blocks,
        svd_eps=args.svd_eps,
        k_max=args.k_max,
        eps_conv=args.eps_conv,
        max_rounds=args.max_rounds,
        n_workers=args.n_workers,
        scheme=args.scheme,
        output_dir=args.output_dir,
        verbose=not args.quiet,
    )

    if 'error' in results:
        print(f"\nERROR: {results['error']}")
        sys.exit(1)

    dE_final = results['rounds'][-1]['dE_neumann_mH']
    status = ("✓ CONVERGED" if results['converged']
              else f"Not converged, final dE={dE_final:.3f} mH")
    print(f"\nFinal status: {status}")
    print(f"  Rounds: {results['n_rounds']}")
    print(f"  Total time: {results['timing_total']:.0f}s")


if __name__ == "__main__":
    main()
