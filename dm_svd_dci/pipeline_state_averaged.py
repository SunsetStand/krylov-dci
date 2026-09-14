#!/usr/bin/env python3
"""Production pipeline for state-averaged self-consistent dmSVD downfolding."""

import json
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from dm_svd_dci.initializers import (
    build_initial_states,
    evaluate_reference_energies,
    seed_block_support,
)
from dm_svd_dci.pipeline_v2 import setup_system
from dm_svd_dci.qspace_partition import (
    extract_q_blocks_scheme_a,
    extract_q_blocks_scheme_b,
    partition_qspace_by_n,
)
from dm_svd_dci.schmidt_partition import partition_schmidt_basis
from dm_svd_dci.state_averaged_solver import solve_state_averaged_schmidt
from dm_svd_embedding.density_matrix import (
    compute_compression_metrics,
    normalize_state_weights,
)
from dm_svd_embedding.occ_virt_partition import (
    build_block_matrices,
    setup_partition,
)


def _multi_root_casci(sys_data: Dict, n_states: int) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Compute the common-orbital CASCI roots used to initialize the SA loop."""
    from pyscf import mcscf

    cas = mcscf.CASCI(
        sys_data['mf'], sys_data['n_active'], sum(sys_data['n_active_elec']))
    cas.frozen = sys_data['n_core']
    cas.fcisolver.nroots = n_states
    cas.kernel()

    energies = np.atleast_1d(np.asarray(cas.e_tot, dtype=float)).reshape(-1)
    if n_states == 1:
        roots = [np.asarray(cas.ci).reshape(-1)]
    else:
        roots = [np.asarray(cas.ci[root]).reshape(-1) for root in range(n_states)]
    if len(energies) < n_states or len(roots) < n_states:
        raise RuntimeError(
            f"CASCI returned {len(roots)} vectors and {len(energies)} energies "
            f"for {n_states} requested states")
    return energies[:n_states], roots[:n_states]


def run_state_averaged_dci(
    atom: str = 'N 0 0 0; N 0 0 1.098',
    basis: str = 'cc-pVDZ',
    n_active: int = 10,
    n_active_elec: Tuple[int, int] = (5, 5),
    n_core: int = 2,
    n_occ: int = 5,
    ms: int = 0,
    svd_eps: float = 1e-3,
    sa_states: int = 3,
    state_weights: Optional[np.ndarray] = None,
    p_blocks: Optional[List[int]] = None,
    outer_mixing: float = 1.0,
    outer_density_tol: float = 1e-7,
    outer_energy_tol: float = 1e-8,
    outer_max_iter: int = 20,
    wave_damping: float = 0.5,
    wave_residual_tol: float = 1e-9,
    wave_energy_tol: float = 1e-10,
    wave_max_iter: int = 100,
    min_denominator: float = 1e-6,
    n_workers: int = 1,
    scheme: str = 'A',
    seed: str = 'exact',
    seed_subspace_size: int = 64,
    seed_perturbation_scale: float = 0.1,
    seed_random_seed: int = 0,
    seed_complete_blocks: bool = True,
    seed_lanczos_steps: int = 3,
    rank_mode: str = 'rectangular',
    apply_dressing: bool = True,
    omega_mode: str = 'shared',
    compute_reference: bool = True,
    embedded_spectrum: bool = False,
    output_dir: Optional[str] = None,
    verbose: bool = True,
) -> Dict:
    """Run residual-dressed, wave-operator self-consistent SA-dmSVD.

    The state weights enter both reduced densities and the common
    wave-operator residual.  ``scheme`` changes only how the existing embedded
    Hamiltonian blocks are constructed; neither H_AB nor RDM contractions are
    modified by this pipeline.
    """
    if sa_states <= 0:
        raise ValueError("sa_states must be positive")
    if p_blocks is None:
        p_blocks = [8, 9, 10]
    weights = normalize_state_weights(sa_states, state_weights)
    total_start = time.perf_counter()

    if verbose:
        print("=" * 72)
        print("STATE-AVERAGED SELF-CONSISTENT dmSVD DOWNFOLDING")
        print("=" * 72)
        print(f"  states={sa_states}, weights={np.array2string(weights, precision=5)}")
        print(f"  scheme={scheme}, P blocks={p_blocks}")

    # solve_exact=False: the exact CASCI kernel in setup_system is dead weight
    # for this path, because the integrals are taken before it would run.
    sys_data = setup_system(
        atom=atom, basis=basis,
        n_active=n_active, n_active_elec=n_active_elec,
        n_core=n_core, nroots=1, verbose=verbose, solve_exact=False)

    # Initializer boundary.  Only seed='exact' reads exact CI, and it is a
    # control rather than a production path.
    partition, _ = setup_partition(
        n_active, sum(n_active_elec), n_occ, ms=ms)
    ci_roots, seed_provenance = build_initial_states(
        sys_data, sa_states, seed=seed,
        subspace_size=seed_subspace_size,
        perturbation_scale=seed_perturbation_scale,
        random_seed=seed_random_seed,
        partition=partition, complete_blocks=seed_complete_blocks,
        lanczos_steps=seed_lanczos_steps, verbose=verbose)

    # Evaluator boundary.  reference_energies is used for error reporting only
    # and is never consumed by the solver, the root selector or the Schmidt
    # builder.
    reference_energies = (
        evaluate_reference_energies(sys_data, sa_states)
        if compute_reference else None)

    initial_state_blocks = [
        build_block_matrices(partition, root) for root in ci_roots]

    # A seed with no weight in an electron-number block causes that block to be
    # deleted by the Schmidt decomposition, and the outer loop cannot recover
    # it.  Record it rather than silently converging to a trapped fixed point.
    seed_provenance['block_support'] = seed_block_support(initial_state_blocks)
    if verbose and seed_provenance['block_support']['has_empty_block']:
        print(f"  WARNING: seed has zero weight in blocks "
              f"{seed_provenance['block_support']['empty_blocks']}; "
              f"those blocks will be deleted and cannot be recovered",
              flush=True)

    build_history = []

    def build_problem(schmidt_data: Dict[int, Dict]) -> Dict:
        build_start = time.perf_counter()
        part_info = partition_schmidt_basis(schmidt_data, p_blocks=p_blocks)
        q_partition = partition_qspace_by_n(
            part_info, schmidt_data, p_blocks=p_blocks)

        if scheme == 'A':
            from dm_svd_dci._legacy_pipeline import build_hemb_parallel

            H_emb, _, hemb_norms = build_hemb_parallel(
                schmidt_data, partition,
                sys_data['q_idx'], sys_data['backend'],
                h1_full=sys_data['h1eff'], h2_full=sys_data['h2_4d'],
                n_occ=n_occ, n_act=n_active,
                n_workers=n_workers, verbose=verbose)
            if part_info['total_dim'] > 0:
                H_emb += sys_data['ecore'] * np.eye(part_info['total_dim'])
            q_data = extract_q_blocks_scheme_a(
                H_emb, part_info, q_partition,
                p_blocks=p_blocks, verbose=verbose)
        elif scheme in ('B', 'B_streaming'):
            q_data = extract_q_blocks_scheme_b(
                schmidt_data, partition, part_info, q_partition,
                p_blocks, sys_data['backend'], n_occ, n_active,
                n_workers=n_workers, ecore=sys_data['ecore'],
                verbose=verbose)
            hemb_norms = {
                'norm_HA': 0.0,
                'norm_HB': 0.0,
                'norm_HAB': float(np.linalg.norm(q_data['H_PP'])),
                'norm_total': float(np.linalg.norm(q_data['H_PP'])),
                'asymmetry': 0.0,
            }
        else:
            raise ValueError(f"unknown Hamiltonian construction scheme: {scheme}")

        build_record = {
            'seconds': time.perf_counter() - build_start,
            'D_total': part_info['total_dim'],
            'P_dim': part_info['p_dim'],
            'Q_dim': part_info['q_dim'],
            'Q_active_dim': int(sum(
                matrix.shape[1] for matrix in q_data['H_PQ'].values())),
        }
        build_history.append(build_record)
        return {
            'H_PP': q_data['H_PP'],
            'H_PQ': q_data['H_PQ'],
            'H_QQ_blocks': q_data['H_QQ_blocks'],
            'D_by_n': q_data['H_QQ_diag'],
            'part_info': part_info,
            'q_partition': q_partition,
            'q_data': q_data,
            'hemb_norms': hemb_norms,
            'build_record': build_record,
        }

    result = solve_state_averaged_schmidt(
        initial_state_blocks,
        build_problem=build_problem,
        svd_eps=svd_eps,
        state_weights=weights,
        outer_mixing=outer_mixing,
        density_tol=outer_density_tol,
        energy_tol=outer_energy_tol,
        max_outer_iter=outer_max_iter,
        wave_options={
            'damping': wave_damping,
            'residual_tol': wave_residual_tol,
            'energy_tol': wave_energy_tol,
            'max_iter': wave_max_iter,
            'min_denominator': min_denominator,
            'apply_dressing': apply_dressing,
            'omega_mode': omega_mode,
        },
        rank_mode=rank_mode,
        verbose=verbose)

    final_problem = result['problem']
    final_part = final_problem['part_info']

    # Exact diagonalization of the final embedded Hamiltonian.  This separates
    # the two error sources that must not be allowed to cancel: the Schmidt
    # truncation error, which is E_embedded - E_reference, and the
    # wave-operator error, which is E_downfolded - E_embedded.  Scoring only
    # against the reference cannot tell them apart.
    embedded_energies = None
    if embedded_spectrum:
        from dm_svd_dci.wave_operator import assemble_qspace_hamiltonian
        assembled = assemble_qspace_hamiltonian(
            final_problem['H_PQ'], final_problem['H_QQ_blocks'],
            final_problem['D_by_n'])
        h_pp = np.asarray(final_problem['H_PP'])
        h_pq = np.asarray(assembled['H_PQ'])
        h_qq = np.asarray(assembled['H_QQ'])
        if h_qq.shape[0] == 0:
            full = h_pp
        else:
            full = np.block([[h_pp, h_pq], [h_pq.T.conj(), h_qq]])
        full = 0.5 * (full + full.T.conj())
        embedded_energies = np.linalg.eigvalsh(full)[:sa_states]

        # Spectral radius of B A on the embedded Q space.  The residual
        # dressing is a damped preconditioned iteration with matrix
        # (1-w) I + w (A B), so convergence requires w < 2 / (rho + 1).
        # Recording it turns a damping choice into a checkable condition.
        if h_qq.shape[0] > 0:
            diagonal = np.diag(h_qq)
            reference = float(np.min(np.linalg.eigvalsh(h_pp)))
            gap = reference - diagonal
            gap[np.abs(gap) < 1e-12] = 1e-12
            resolvent = 1.0 / gap
            off_diagonal = h_qq - np.diag(diagonal)
            spectral_radius = float(np.max(np.abs(np.linalg.eigvals(
                off_diagonal * resolvent[np.newaxis, :]))))
        else:
            spectral_radius = 0.0
    metrics = compute_compression_metrics(
        result['schmidt_data'], result['state_blocks'][0])
    energies = np.asarray(result['energies'])
    # result['energies'] is in overlap-matched order, while the reference and
    # the embedded spectrum are in ascending energy order.  Subtracting them
    # directly compares different labelings and manufactures an error whenever
    # the overlap match is not the energy order.  Both orderings are therefore
    # reported: 'errors_mH' keeps the matched labeling, and the '_sorted'
    # variants compare the spectra as sets, which is what the target
    # definition of "the lowest n levels" actually asks.
    energies_sorted = np.sort(energies)
    errors_mh = (None if reference_energies is None
                 else (energies - reference_energies) * 1000.0)
    errors_sorted_mh = (
        None if reference_energies is None
        else (energies_sorted - np.sort(reference_energies)) * 1000.0)
    wall_time = time.perf_counter() - total_start

    output = {
        'method': 'state-averaged residual-dressed self-consistent dmSVD',
        'energies': energies,
        'reference_energies': reference_energies,
        'errors_mH': errors_mh,
        'errors_sorted_mH': errors_sorted_mh,
        'energies_sorted': energies_sorted,
        'seed_provenance': seed_provenance,
        'embedded_exact_energies': embedded_energies,
        'spectral_radius_BA': (
            None if not embedded_spectrum else spectral_radius),
        'damping_bound': (
            None if not embedded_spectrum else 2.0 / (spectral_radius + 1.0)),
        'damping_bound_satisfied': (
            None if not embedded_spectrum
            else bool(wave_damping < 2.0 / (spectral_radius + 1.0))),
        'schmidt_truncation_errors_mH': (
            None if (embedded_energies is None or reference_energies is None)
            else (np.sort(embedded_energies) - np.sort(reference_energies))
            * 1000.0),
        'wave_operator_errors_mH': (
            None if embedded_energies is None
            else (energies_sorted - np.sort(embedded_energies)) * 1000.0),
        'state_weights': weights,
        'converged': result['converged'],
        'n_outer_iter': result['n_outer_iter'],
        'outer_history': result['history'],
        'final_wave_converged': result['wave_result']['converged'],
        'final_wave_iterations': result['wave_result']['n_iter'],
        'final_wave_residual_rms': result['wave_result']['residuals']['weighted_rms'],
        'final_wave_residual_root_norms':
            result['wave_result']['residuals']['root_norms'],
        'final_wave_residual_max':
            result['wave_result']['residuals']['max_norm'],
        'final_wave_history': result['wave_result']['history'],
        'apply_dressing': result['wave_result'].get('apply_dressing', True),
        'omega_mode': result['wave_result'].get('omega_mode', 'shared'),
        'schmidt_metrics': {
            'r_total': metrics['r_total'],
            'r_A_total': metrics['r_A_total'],
            'r_B_total': metrics['r_B_total'],
            'product_dim': metrics['product_dim'],
            'dim_fci': metrics['dim_fci'],
            'compression_ratio': metrics['compression_ratio'],
            'product_compression_ratio': metrics['product_compression_ratio'],
            'discarded_weight': metrics['discarded_weight'],
            'ranks': {
                int(label): {
                    'r_A': int(data.get('r_A', data['r'])),
                    'r_B': int(data.get('r_B', data['r'])),
                }
                for label, data in result['schmidt_data'].items()},
        },
        'partition_info': {
            'D_total': final_part['total_dim'],
            'P_dim': final_part['p_dim'],
            'Q_dim': final_part['q_dim'],
            'p_blocks': list(p_blocks),
        },
        'build_history': build_history,
        'hemb_norms': final_problem['hemb_norms'],
        'wall_time_seconds': wall_time,
        'parameters': {
            'atom': atom,
            'basis': basis,
            'n_active': n_active,
            'n_active_elec': list(n_active_elec),
            'n_core': n_core,
            'n_occ': n_occ,
            'svd_eps': svd_eps,
            'sa_states': sa_states,
            'scheme': scheme,
            'outer_mixing': outer_mixing,
            'wave_damping': wave_damping,
            'seed': seed,
            'seed_complete_blocks': seed_complete_blocks,
            'seed_lanczos_steps': seed_lanczos_steps,
            'rank_mode': rank_mode,
            'apply_dressing': apply_dressing,
            'omega_mode': omega_mode,
            'compute_reference': compute_reference,
        },
    }

    if verbose:
        print("\n" + "=" * 72)
        print("STATE-AVERAGED SELF-CONSISTENT SUMMARY")
        print("=" * 72)
        if reference_energies is None:
            for root, energy in enumerate(energies):
                print(f"  S{root}: E={energy:.12f}  (no reference computed)")
        else:
            for root, (energy, reference, error) in enumerate(zip(
                    energies, reference_energies, errors_mh)):
                print(
                    f"  S{root}: E={energy:.12f}  CASCI={reference:.12f}  "
                    f"dE={error:+.3f} mH")
        print(
            f"  outer converged={result['converged']} "
            f"in {result['n_outer_iter']} iterations")
        print(
            f"  final wave residual="
            f"{result['wave_result']['residuals']['weighted_rms']:.3e}")
        print(
            f"  Schmidt r_total={metrics['r_total']}, "
            f"D={final_part['total_dim']}, |P|={final_part['p_dim']}, "
            f"|Q|={final_part['q_dim']}")
        print(f"  wall time={wall_time:.1f}s", flush=True)

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(
            output_dir, 'state_averaged_sc_dmsvd_results.json')
        with open(output_path, 'w', encoding='utf-8') as handle:
            json.dump(_make_serializable(output), handle, indent=2)
        if verbose:
            print(f"  Results saved to {output_path}")

    return output


def _make_serializable(value):
    if isinstance(value, dict):
        return {str(key): _make_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_make_serializable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    return value
