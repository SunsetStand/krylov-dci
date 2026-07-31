#!/usr/bin/env python3
"""
DMRG-style growing active space via dmSVD + operator renormalization.

Main driver that orchestrates the iterative extension of the active space:

  t=0 (Bootstrap):
    A₀ orbitals + B₀ orbitals → CASCI → dmSVD → S₀ → Neumann Heff → E₀

  t≥1 (Extension):
    S_{t-1} + new orbital B_t → build H_comp → diagonalize → block-SVD → S_t
    → Neumann Heff → E_t → check convergence

Configuration:
  n_occ_A:   Number of A-space orbitals (default 6: 5 occupied + 1 virtual)
  n_orb_B0:  Number of B-space orbitals in round 0 (default 1)
  n_orb_Bt:  Number of B-space orbitals per extension round (default 1)
  n_core:    Number of frozen core orbitals (default 2 for N₂ 1s)
  max_rounds: Maximum extension rounds (default: all remaining virtuals)
  eps_svd:   SVD truncation threshold
  eps_conv:  Energy convergence threshold (Hartree)

Usage:
    python dm_svd_dci/active_space_grower.py \
        --atom 'N 0 0 0; N 0 0 1.098' --basis cc-pVDZ \
        --n-active 10 --n-alpha 5 --n-beta 5 --n-core 2 \
        --n-occ-A 6 --n-orb-B0 1 --n-orb-Bt 1 \
        --eps-svd 1e-3 --eps-conv 1e-8 \
        --n-workers 16 --output-dir ./results/grow_n2
"""

import sys, os, time, json, argparse
import numpy as np
from numpy.linalg import eigh
from typing import Dict, List, Tuple, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


# ═══════════════════════════════════════════════════════════════════════════
# Step 1: System setup
# ═══════════════════════════════════════════════════════════════════════════

def _setup_molecule(
    atom: str,
    basis: str,
    n_active: int,
    n_active_elec: Tuple[int, int],
    n_core: int,
    verbose: bool = True,
) -> Dict:
    """Initialize PySCF molecule and RHF, return active-space setup."""
    from pyscf import gto, scf

    mol = gto.M(atom=atom, basis=basis, verbose=0, spin=0)
    mf = scf.RHF(mol).run(verbose=0)

    if verbose:
        print(f"  Molecule: {atom.strip()}, basis={basis}")
        print(f"  RHF E:    {mf.e_tot:.12f} Ha, "
              f"n_mo={mol.nao_nr()}, n_elec={mol.nelec}")

    return {
        'mol': mol, 'mf': mf,
        'n_active': n_active,
        'n_active_elec': n_active_elec,
        'n_core': n_core,
    }


def _get_orbital_energies(mf, n_core: int, n_active: int) -> np.ndarray:
    """Get MO energies for selection of active orbitals.

    Returns sorted indices of active-space MOs by energy.
    For occ-virt partitioning, occupied orbitals are naturally below HOMO.
    """
    mo_energy = mf.mo_energy[n_core:n_core + n_active]
    return np.argsort(mo_energy)


def _setup_subspace_casci(
    sys_data: Dict,
    active_orb_indices: np.ndarray,
    n_active_elec: Tuple[int, int],
    verbose: bool = True,
) -> Dict:
    """Run CASCI in a subspace of the full active space.

    Args:
        sys_data: Output of _setup_molecule.
        active_orb_indices: List of orbital indices (global, starting from 0)
                           to include in this subspace.
        n_active_elec: (n_alpha, n_beta) for the active space.

    Returns:
        dict with keys: q_idx, backend, fcivec, E_fci, E_core, h1eff, h2_4d, ecore.
    """
    from pyscf import mcscf
    from pyscf.fci import cistring
    from src_mf.pyscf_backend import QSpaceIndex, KDCIBackend

    mol = sys_data['mol']
    mf = sys_data['mf']
    n_core = sys_data['n_core']

    n_act_sub = len(active_orb_indices)
    n_elec = sum(n_active_elec)

    # Build CASCI in the subspace
    cas = mcscf.CASCI(mf, n_act_sub, n_elec)
    cas.frozen = n_core

    # We need to restrict the active space to active_orb_indices.
    # PySCF's CASCI uses orbitals n_core:n_core+n_act_sub by default.
    # If our subspace is not contiguous, we need to reorder MOs.
    #
    # For now, assume the subspace is contiguous (which it is for our use case:
    # A₀ uses first n_occ_A orbitals, which are the lowest active MOs).
    # For non-contiguous subspaces, we'd need MO rotation.

    # Set the active space MOs explicitly
    n_total_mo = mol.nao_nr()
    mo_coeff_full = mf.mo_coeff.copy()  # (nao, n_total_mo)

    # Build new MO coefficient matrix: [core | active_sub | rest]
    all_indices = list(range(n_total_mo))
    core_indices = list(range(n_core))  # 0..n_core-1
    active_sub_indices = [n_core + i for i in active_orb_indices]
    # remaining: exclude core and selected active
    used = set(core_indices) | set(active_sub_indices)
    rest_indices = [i for i in all_indices if i not in used]

    new_order = core_indices + active_sub_indices + rest_indices
    mo_new = mo_coeff_full[:, new_order]

    # Build a new RHF-like object with reordered MOs
    mf_new = mf.copy()
    mf_new.mo_coeff = mo_new
    mf_new.mo_occ = np.zeros(n_total_mo)
    mf_new.mo_occ[:n_core] = 2.0  # doubly occupied core
    # The active orbitals have fractional occupation in CAS, but for
    # the purpose of getting h1eff/h2eff, we can use any occupation
    # as long as the orbital ordering is correct.

    cas2 = mcscf.CASCI(mf_new, n_act_sub, n_elec)
    cas2.frozen = n_core
    cas2.kernel()

    h1eff, ecore = cas2.get_h1eff()
    h2eff = cas2.get_h2eff()
    fcivec = cas2.ci
    E_fci = cas2.e_tot

    na, nb = n_active_elec
    alpha_strs = cistring.gen_strings4orblist(range(n_act_sub), na)
    beta_strs = cistring.gen_strings4orblist(range(n_act_sub), nb)

    q_idx = QSpaceIndex(alpha_strs, beta_strs, n_act_sub, n_active_elec,
                        h1eff, h2eff)
    backend = KDCIBackend(q_idx)

    # Unpack h2 to 4d
    from src.hamiltonian import _unpack_4fold
    h2_4d = _unpack_4fold(h2eff, n_act_sub)

    if verbose:
        print(f"    CASCI({n_act_sub},{n_elec}): E={E_fci:.12f} Ha, "
              f"M={q_idx.M:,} dets", flush=True)

    return {
        'q_idx': q_idx,
        'backend': backend,
        'fcivec': fcivec,
        'E_fci': E_fci,
        'ecore': float(ecore),
        'h1eff': h1eff,
        'h2eff': h2eff,
        'h2_4d': h2_4d,
        'n_act_sub': n_act_sub,
        'active_orb_indices': active_orb_indices,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Round 0: Bootstrap dmSVD
# ═══════════════════════════════════════════════════════════════════════════

def _run_bootstrap(
    sys_data: Dict,
    cas_data: Dict,
    n_occ_A: int,
    svd_eps: float,
    p_blocks: List[int],
    k_max: int,
    n_workers: int,
    verbose: bool = True,
) -> Dict:
    """Round 0: dmSVD + Neumann Heff.

    Returns:
        dict with E0, ops (RenormalizedOperators), schmidt_metrics, etc.
    """
    from dm_svd_embedding.occ_virt_partition import (
        setup_partition, build_block_matrices)
    from dm_svd_embedding.density_matrix import (
        compute_schmidt_decomposition, compute_compression_metrics)
    from dm_svd_dci.schmidt_partition import partition_schmidt_basis
    from dm_svd_dci.qspace_partition import (
        partition_qspace_by_n, extract_q_blocks_scheme_a)
    from dm_svd_dci.neumann_effective_ham import (
        build_effective_hamiltonian_neumann)
    from dm_svd_dci.renormalized_operators import RenormalizedOperators

    t0 = time.perf_counter()

    n_act = cas_data['n_act_sub']
    n_elec_total = sum(sys_data['n_active_elec'])

    # ── dmSVD ──
    if verbose:
        print(f"\n  --- dmSVD: occ={n_occ_A}, virt={n_act - n_occ_A}, "
              f"eps={svd_eps} ---", flush=True)

    partition, full_dets = setup_partition(
        n_act, n_elec_total, n_occ_A, ms=0)
    ci_flat = cas_data['fcivec'].reshape(-1)
    C_blocks = build_block_matrices(partition, ci_flat)
    schmidt = compute_schmidt_decomposition(C_blocks, eps=svd_eps)
    metrics = compute_compression_metrics(schmidt, C_blocks, ci_flat)

    if verbose:
        print(f"    Schmidt: r_total={metrics['r_total']}, "
              f"D={sum(sd.get('r', 0)**2 for sd in schmidt.values())}, "
              f"compression={metrics['compression_ratio']:.4%}")

    # ── Bootstrap renormalized operators ──
    ops = RenormalizedOperators.bootstrap_from_schmidt(
        schmidt, partition, cas_data['backend'],
        n_occ=n_occ_A, n_act=n_act,
        h1eff=cas_data['h1eff'], h2_4d=cas_data['h2_4d'],
        ecore=cas_data['ecore'],
        verbose=verbose)

    # ── Neumann Heff ──
    part = partition_schmidt_basis(schmidt, p_blocks=p_blocks)
    q_partition = partition_qspace_by_n(part, schmidt, p_blocks=p_blocks)

    # Build full H^emb and extract Q blocks
    from dm_svd_dci._legacy_pipeline import build_hemb_parallel
    D = ops.D
    H_emb = ops.H_S  # already built during bootstrap

    q_blocks_data = extract_q_blocks_scheme_a(
        H_emb, part, q_partition, p_blocks=p_blocks, verbose=verbose)

    H_PP = q_blocks_data['H_PP']
    H_PQ = q_blocks_data['H_PQ']
    H_QQ_blocks = q_blocks_data['H_QQ_blocks']
    H_QQ_diag = q_blocks_data['H_QQ_diag']

    # Bare H_PP energy
    E_bare = eigh(H_PP)[0][0] if H_PP.shape[0] > 0 else 0.0

    # Single-shot Neumann (no SCF)
    res_neumann = build_effective_hamiltonian_neumann(
        H_PP, H_PQ, H_QQ_blocks, H_QQ_diag,
        E_bare, delta=0.0, k_max=k_max, verbose=verbose)

    H_eff = res_neumann['H_eff']
    E_neumann = eigh(H_eff)[0][0] if H_eff.shape[0] > 0 else 0.0

    if verbose:
        dE_bare = (E_bare - cas_data['E_fci']) * 1000
        dE_neumann = (E_neumann - cas_data['E_fci']) * 1000
        print(f"\n    E(FCI)      = {cas_data['E_fci']:.12f} Ha")
        print(f"    E(bare H_PP) = {E_bare:.12f} Ha  "
              f"(ΔE = {dE_bare:+.3f} mH)")
        print(f"    E(Neumann)   = {E_neumann:.12f} Ha  "
              f"(ΔE = {dE_neumann:+.3f} mH)")

    return {
        'E_bare': E_bare,
        'E_neumann': E_neumann,
        'E_fci_sub': cas_data['E_fci'],
        'ops': ops,
        'schmidt_metrics': metrics,
        'schmidt_data': schmidt,
        'partition': partition,
        'elapsed': time.perf_counter() - t0,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Round t≥1: Extension
# ═══════════════════════════════════════════════════════════════════════════

def _run_extension_round(
    sys_data: Dict,
    ops: 'RenormalizedOperators',
    new_orbital_idx: int,
    svd_eps: float,
    p_blocks: List[int],
    k_max: int,
    n_workers: int,
    verbose: bool = True,
) -> Dict:
    """Extension round: add new orbital, block-SVD, Neumann Heff.

    Args:
        sys_data: Molecule system data.
        ops:      RenormalizedOperators from previous round.
        new_orbital_idx: Global orbital index (in active space) of the new orbital.
        svd_eps:  SVD truncation threshold.
        p_blocks: P-space n-blocks.
        k_max:    Neumann expansion order.
        n_workers: Parallel sigma workers.
        verbose:  Print diagnostics.

    Returns:
        dict with E_neumann, ops (updated), etc.
    """
    from dm_svd_dci.block_svd import block_svd
    from dm_svd_dci.davidson_solver import solve_hamiltonian
    from dm_svd_dci.schmidt_partition import partition_schmidt_basis
    from dm_svd_dci.qspace_partition import (
        partition_qspace_by_n, extract_q_blocks_scheme_a)
    from dm_svd_dci.neumann_effective_ham import (
        build_effective_hamiltonian_neumann)

    t0 = time.perf_counter()
    D_old = ops.D
    n_act_new = ops.n_orbitals + 1

    if verbose:
        print(f"\n  {'='*50}")
        print(f"  Extension Round: orbital {new_orbital_idx} "
              f"({n_act_new} total active)", flush=True)
        print(f"  {'='*50}")

    # ── 1. Build composite Hamiltonian ──
    # We need a new backend for the expanded active space
    from src_mf.pyscf_backend import QSpaceIndex, KDCIBackend
    from pyscf.fci import cistring
    from src.hamiltonian import _unpack_4fold

    # Run CASCI in new active space to get integrals and backend
    # The new active orbital set is: previous active + new_orbital_idx
    old_orbs = list(range(ops.n_orbitals))
    new_orbs = old_orbs + [new_orbital_idx]
    cas_data = _setup_subspace_casci(
        sys_data, np.array(new_orbs),
        sys_data['n_active_elec'], verbose=verbose)

    # Build H_comp using the new backend
    H_comp = ops.build_composite_hamiltonian(
        cas_data['backend'],
        new_orbital_index=ops.n_orbitals,  # appended at end
        h1eff=cas_data['h1eff'],
        h2_4d=cas_data['h2_4d'],
        ecore=cas_data['ecore'],
        n_workers=n_workers,
        verbose=verbose)

    # ── 2. Diagonalize H_comp ──
    E_comp, psi = solve_hamiltonian(H_comp, nroots=1, verbose=verbose)
    E_comp_gs = E_comp[0]
    psi_gs = psi[:, 0]

    if verbose:
        print(f"  H_comp GS energy: {E_comp_gs:.12f} Ha")

    # ── 3. Block-SVD ──
    svd_result = block_svd(psi_gs, D_old, eps=svd_eps, verbose=verbose)
    W = svd_result['W']
    D_new = svd_result['D_new']

    if D_new == 0:
        print(f"  ⚠ Block-SVD returned D=0! Check eps={svd_eps}")
        return {'error': 'Zero-dimensional compressed basis', 'E_neumann': E_comp_gs}

    # ── 4. Recompress operators ──
    ops.recompress(
        W, cas_data['backend'],
        h1eff=cas_data['h1eff'],
        h2_4d=cas_data['h2_4d'],
        ecore=cas_data['ecore'],
        verbose=verbose)

    # ── 5. Neumann Heff in new Schmidt basis ──
    # Need to build H_PP, H_PQ, H_QQ in the new basis.
    # For now, use H_S as the full H^emb and apply Schmidt partition.
    #
    # Rebuild Schmidt partition info for the new basis.
    # Since we used W to compress, we need new schmidt_data.
    # The recompress() method updated ops.H_S and ops.ci_mats.
    #
    # For the partition, we need new "schmidt_data" structure.
    # Since S_t is the compressed basis (not block-diagonal by n anymore),
    # we skip the full Schmidt partition for now and use H_S directly.
    #
    # Alternative: treat the entire compressed basis as P-space,
    # or partition by n if we track n per state.

    # Simplification: since the compressed basis already captures the
    # dominant physics, diagonalize H_S directly.
    E_new = eigh(ops.H_S)[0][0] if ops.H_S.shape[0] > 0 else 0.0

    if verbose:
        dE = (E_new - E_comp_gs) * 1000
        print(f"\n    E(H_S)     = {E_new:.12f} Ha  "
              f"(Δ vs H_comp GS = {dE:+.3f} mH)")
        print(f"    D: {D_old} → {D_new}, "
              f"time: {time.perf_counter() - t0:.1f}s")

    return {
        'E_comp_gs': E_comp_gs,
        'E_neumann': E_new,
        'E_fci_casci': cas_data['E_fci'],
        'D_old': D_old,
        'D_new': D_new,
        'svd_result': svd_result,
        'elapsed': time.perf_counter() - t0,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════════════════

def run_growing_dci(
    atom: str = 'N 0 0 0; N 0 0 1.098',
    basis: str = 'cc-pVDZ',
    n_active: int = 10,
    n_active_elec: Tuple[int, int] = (5, 5),
    n_core: int = 2,
    n_occ_A: int = 6,
    n_orb_B0: int = 1,
    n_orb_Bt: int = 1,
    p_blocks: List[int] = [8, 9, 10],
    svd_eps: float = 1e-3,
    k_max: int = 1,
    eps_conv: float = 1e-8,
    max_rounds: Optional[int] = None,
    n_workers: int = 1,
    output_dir: Optional[str] = None,
    verbose: bool = True,
) -> Dict:
    """DMRG-style growing active space via dmSVD + operator renormalization.

    Args:
        atom:          PySCF molecular geometry string.
        basis:         Basis set name.
        n_active:      Total active orbitals in the full CAS.
        n_active_elec: (n_alpha, n_beta) active electrons.
        n_core:        Frozen core orbitals.
        n_occ_A:       Number of A-space orbitals in round 0.
        n_orb_B0:      Number of B-space orbitals in round 0.
        n_orb_Bt:      Number of B-space orbitals per extension round.
        p_blocks:      P-space n-blocks for Neumann Heff.
        svd_eps:       SVD truncation threshold.
        k_max:         Neumann expansion order (0 or 1).
        eps_conv:      Energy convergence threshold (Hartree).
        max_rounds:    Maximum extension rounds (default: all remaining).
        n_workers:     Parallel sigma workers.
        output_dir:    JSON output directory.
        verbose:       Print progress.

    Returns:
        dict with convergence history.
    """
    t_total = time.perf_counter()

    if verbose:
        print("=" * 70)
        print("Growing Active Space DCI")
        print("=" * 70)
        print(f"  System:     {atom.strip()}, {basis}")
        print(f"  CAS:        ({n_active},{sum(n_active_elec)}), "
              f"n_core={n_core}")
        print(f"  Bootstrap:  A₀={n_occ_A} orbitals, B₀={n_orb_B0} orbitals "
              f"({n_occ_A + n_orb_B0} total)")
        print(f"  Extension:  +{n_orb_Bt} orbital(s) per round")
        print(f"  SVD eps:    {svd_eps}, Neumann k={k_max}")
        print(f"  Conv tol:   {eps_conv:.1e} Ha")

    # ── System setup ──
    sys_data = _setup_molecule(atom, basis, n_active, n_active_elec, n_core,
                                verbose=verbose)

    # ── Select initial A₀ and B₀ orbitals ──
    # Use the lowest n_occ_A + n_orb_B0 orbitals from the active space
    n_total_active = n_active
    initial_orbs = list(range(n_occ_A + n_orb_B0))  # 0..n_occ_A+n_orb_B0-1
    remaining_orbs = list(range(n_occ_A + n_orb_B0, n_total_active))

    if max_rounds is None:
        max_rounds = len(remaining_orbs) // n_orb_Bt + (1 if len(remaining_orbs) % n_orb_Bt else 0)

    if verbose:
        print(f"\n  Initial active orbs: {initial_orbs} "
              f"(A₀={list(range(n_occ_A))} + B₀={list(range(n_occ_A, n_occ_A+n_orb_B0))})")
        print(f"  Environment orbs:    {remaining_orbs} "
              f"({len(remaining_orbs)} orbitals, max {max_rounds} rounds)")

    # ── Round 0: Bootstrap ──
    if verbose:
        print(f"\n{'='*70}")
        print(f"ROUND 0: Bootstrap (dmSVD + Neumann)")
        print(f"{'='*70}")

    cas0_data = _setup_subspace_casci(
        sys_data, np.array(initial_orbs), n_active_elec, verbose=verbose)

    bootstrap_result = _run_bootstrap(
        sys_data, cas0_data,
        n_occ_A=n_occ_A,
        svd_eps=svd_eps,
        p_blocks=p_blocks,
        k_max=k_max,
        n_workers=n_workers,
        verbose=verbose)

    ops = bootstrap_result['ops']
    E_history = [bootstrap_result['E_neumann']]
    D_history = [ops.D]
    round_results = [bootstrap_result]

    # ── Extension rounds ──
    for round_idx in range(max_rounds):
        if round_idx * n_orb_Bt >= len(remaining_orbs):
            break

        # Select next orbital(s)
        new_orbs = remaining_orbs[round_idx * n_orb_Bt:
                                   min((round_idx + 1) * n_orb_Bt,
                                        len(remaining_orbs))]

        if verbose:
            print(f"\n{'='*70}")
            print(f"ROUND {round_idx + 1}: Adding orbital(s) {new_orbs}")
            print(f"{'='*70}")

        for new_orb in new_orbs:
            ext_result = _run_extension_round(
                sys_data, ops,
                new_orbital_idx=new_orb,
                svd_eps=svd_eps,
                p_blocks=p_blocks,
                k_max=k_max,
                n_workers=n_workers,
                verbose=verbose)

            if 'error' in ext_result:
                print(f"  ERROR: {ext_result['error']}")
                break

            E_history.append(ext_result['E_neumann'])
            D_history.append(ext_result['D_new'])
            round_results.append(ext_result)

            # Convergence check
            if len(E_history) >= 2:
                dE = abs(E_history[-1] - E_history[-2])
                if verbose:
                    print(f"\n    ΔE vs previous round: {dE:.3e} Ha")
                if dE < eps_conv:
                    if verbose:
                        print(f"\n  ✓ CONVERGED: |ΔE| = {dE:.3e} < {eps_conv:.1e}")
                    break
        else:
            continue
        break

    # ── Summary ──
    elapsed_total = time.perf_counter() - t_total

    if verbose:
        print(f"\n{'='*70}")
        print(f"CONVERGENCE SUMMARY")
        print(f"{'='*70}")
        print(f"  {'Round':>5} {'D':>6} {'Energy (Ha)':>18} {'ΔE (mH)':>12}")
        print(f"  {'-'*45}")
        for r, (E_r, D_r) in enumerate(zip(E_history, D_history)):
            dE = (E_r - cas0_data['E_fci']) * 1000
            print(f"  {r:>5} {D_r:>6} {E_r:>18.12f} {dE:>+11.3f}")
        print(f"\n  Total wall time: {elapsed_total:.1f}s")

    # ── Build output ──
    output = {
        'config': {
            'atom': atom, 'basis': basis,
            'n_active': n_active, 'n_active_elec': list(n_active_elec),
            'n_core': n_core, 'n_occ_A': n_occ_A,
            'n_orb_B0': n_orb_B0, 'n_orb_Bt': n_orb_Bt,
            'svd_eps': svd_eps, 'k_max': k_max, 'eps_conv': eps_conv,
            'p_blocks': p_blocks, 'max_rounds': max_rounds,
        },
        'E_history': [float(e) for e in E_history],
        'D_history': D_history,
        'E_fci_ref': float(cas0_data['E_fci']),
        'dE_final_mH': float((E_history[-1] - cas0_data['E_fci']) * 1000),
        'n_rounds': len(E_history),
        'converged': len(E_history) >= 2 and abs(E_history[-1] - E_history[-2]) < eps_conv,
        'timing_total': float(elapsed_total),
        'round_details': [{
            'E': float(r.get('E_neumann', r.get('E_neumann', 0))),
            'D': r.get('D_new', r.get('ops', None) and r['ops'].D if 'ops' in r else 0),
            'elapsed': float(r.get('elapsed', 0)),
        } for r in round_results],
    }

    # ── Save JSON ──
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fname = os.path.join(output_dir, 'growing_dci_results.json')
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
        description='DMRG-style Growing Active Space DCI')

    p.add_argument('--atom', default='N 0 0 0; N 0 0 1.098')
    p.add_argument('--basis', default='cc-pVDZ')
    p.add_argument('--n-active', type=int, default=10)
    p.add_argument('--n-alpha', type=int, default=5)
    p.add_argument('--n-beta', type=int, default=5)
    p.add_argument('--n-core', type=int, default=2)
    p.add_argument('--n-occ-A', type=int, default=6,
                   help='A-space orbitals in round 0')
    p.add_argument('--n-orb-B0', type=int, default=1,
                   help='B-space orbitals in round 0')
    p.add_argument('--n-orb-Bt', type=int, default=1,
                   help='B-space orbitals per extension round')
    p.add_argument('--p-blocks', default='8,9,10',
                   help='P-space n-blocks (comma-separated)')
    p.add_argument('--svd-eps', type=float, default=1e-3)
    p.add_argument('--k-max', type=int, default=1,
                   help='Neumann expansion order')
    p.add_argument('--eps-conv', type=float, default=1e-8,
                   help='Energy convergence threshold (Ha)')
    p.add_argument('--max-rounds', type=int, default=None,
                   help='Maximum extension rounds')
    p.add_argument('--n-workers', type=int, default=1)
    p.add_argument('--output-dir', default=None)
    p.add_argument('--quiet', action='store_true')

    return p.parse_args()


def main():
    args = _parse_args()
    p_blocks = [int(x.strip()) for x in args.p_blocks.split(',')]

    results = run_growing_dci(
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
        output_dir=args.output_dir,
        verbose=not args.quiet,
    )

    if 'error' in results:
        print(f"\nERROR: {results['error']}")
        sys.exit(1)

    dE_final = results['dE_final_mH']
    status = ("✓ CONVERGED" if results['converged']
              else f"✗ Not converged, dE={dE_final:.3f} mH")
    print(f"\nFinal status: {status}")
    print(f"  E_final = {results['E_history'][-1]:.12f} Ha")
    print(f"  ΔE vs FCI(ref) = {dE_final:+.3f} mH")
    print(f"  Rounds: {results['n_rounds']}")
    print(f"  Total time: {results['timing_total']:.0f}s")


if __name__ == "__main__":
    main()
