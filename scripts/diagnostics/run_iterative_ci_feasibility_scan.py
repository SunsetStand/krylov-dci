#!/usr/bin/env python3
"""Pre-registered scan for the iterative-CI feasibility protocol.

Implements docs/theory/iterative_ci_feasibility_protocol.md.  Every threshold
and every pass condition is taken from that document; nothing is decided here.

The scan is organized by hypothesis rather than as a full product of the control
axes, because most cells of that product carry no information.  Each group
varies exactly the axis its hypothesis is about and holds the rest fixed.

Every cell records the error decomposition that the protocol requires in order
to keep truncation error and solver error from cancelling:

    total error       = E_downfolded - E_reference
    Schmidt error     = E_embedded   - E_reference
    wave-operator err = E_downfolded - E_embedded

where ``E_embedded`` is the exact diagonalization of the final embedded
Hamiltonian.  Scoring only against the reference cannot separate these.
"""

import argparse
import json
import os
import resource
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from dm_svd_dci.pipeline_state_averaged import (  # noqa: E402
    run_state_averaged_dci,
)

PROTOCOL = 'docs/theory/iterative_ci_feasibility_protocol.md'

# Thresholds fixed by the protocol.  Do not tune these here.
TOL_E_MH = 0.05
CHEMICAL_ACCURACY_MH = 1.6
MATCHED_DIMENSION_TOLERANCE = 0.02

SYSTEMS = {
    'h2o': dict(
        atom='O 0 0 0; H 0 0.757 0.586; H 0 -0.757 0.586', basis='sto-3g',
        n_active=5, n_active_elec=(3, 3), n_core=2, n_occ=3,
        sa_states=3, p_blocks=[4, 5, 6]),
    'h2': dict(
        atom='H 0 0 0; H 0 0 0.74', basis='sto-3g',
        n_active=2, n_active_elec=(1, 1), n_core=0, n_occ=1,
        sa_states=2, p_blocks=[1, 2]),
}

CONTROLS = dict(
    outer_mixing=0.7, outer_density_tol=1e-6, outer_energy_tol=1e-7,
    outer_max_iter=12, wave_damping=0.7, wave_residual_tol=1e-9,
    wave_energy_tol=1e-10, wave_max_iter=200, min_denominator=1e-6)

THRESHOLDS = (1e-2, 1e-3, 1e-4)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--system', choices=sorted(SYSTEMS), default='h2o')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--thresholds', type=float, nargs='+',
                        default=list(THRESHOLDS))
    parser.add_argument('--groups', nargs='+',
                        default=['h1_h2', 'h3', 'h4', 'h6', 'h7'])
    return parser.parse_args()


def peak_rss_mib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def run_cell(system, label, **overrides):
    """One scan point, with the full protocol measurement set."""
    start = time.perf_counter()
    settings = dict(SYSTEMS[system])
    settings.update(CONTROLS)
    settings.update(overrides)
    settings.setdefault('seed', 'lanczos')
    settings.update(embedded_spectrum=True, verbose=False)

    try:
        result = run_state_averaged_dci(**settings)
    except Exception as error:                            # noqa: BLE001
        return {'label': label, 'classification': 'ERROR',
                'error': f'{type(error).__name__}: {error}',
                'settings': {k: v for k, v in overrides.items()}}

    weights = np.asarray(result['state_weights'])
    total = np.abs(np.asarray(result['errors_sorted_mH']))
    schmidt = np.abs(np.asarray(result['schmidt_truncation_errors_mH']))
    wave = np.abs(np.asarray(result['wave_operator_errors_mH']))
    history = result['outer_history']
    last = history[-1]
    projector = last.get('schmidt_projector_distance')
    permutations = [list(item['root_permutation']) for item in history]
    identity = list(range(settings['sa_states']))

    cell = {
        'label': label,
        'settings': {k: v for k, v in overrides.items()},
        'energies': result['energies'],
        'reference_energies': result['reference_energies'],
        'embedded_exact_energies': result['embedded_exact_energies'],
        'weighted_total_error_mH': float(np.dot(weights, total)),
        'max_total_error_mH': float(np.max(total)),
        'weighted_schmidt_error_mH': float(np.dot(weights, schmidt)),
        'weighted_wave_operator_error_mH': float(np.dot(weights, wave)),
        'embedded_dimension': result['partition_info']['D_total'],
        'p_dim': result['partition_info']['P_dim'],
        'q_dim': result['partition_info']['Q_dim'],
        'schmidt_ranks': result['schmidt_metrics']['ranks'],
        'discarded_weight': result['schmidt_metrics']['discarded_weight'],
        'outer_converged': bool(result['converged']),
        'n_outer_iter': result['n_outer_iter'],
        'inner_converged': bool(result['final_wave_converged']),
        'n_inner_iter': result['final_wave_iterations'],
        'residual_weighted_rms': result['final_wave_residual_rms'],
        'residual_root_norms': result['final_wave_residual_root_norms'],
        'schmidt_projector_distance': (
            None if projector is None else projector['total']),
        'root_permutations': permutations,
        'root_reorder': any(p != identity for p in permutations),
        'spectral_radius_BA': result['spectral_radius_BA'],
        'damping_bound': result['damping_bound'],
        'damping_bound_satisfied': result['damping_bound_satisfied'],
        'seed_provenance': result['seed_provenance'],
        'wall_time_seconds': float(time.perf_counter() - start),
        'peak_rss_mib': peak_rss_mib(),
    }
    cell['classification'] = classify(cell)
    return cell


def classify(cell):
    """Protocol labels.  A fallback may only emit INCONCLUSIVE."""
    if not cell.get('damping_bound_satisfied', True):
        return 'DAMPING_BOUND_VIOLATED'
    if cell['settings'].get('outer_max_iter') == 1:
        return 'FROZEN_BASELINE'
    if not cell['inner_converged']:
        return 'INNER_NONCONVERGED'
    if not cell['outer_converged']:
        return 'OUTER_NONCONVERGED'
    if cell['root_reorder']:
        return 'ROOT_REORDER'
    return 'PASS'


def matched_dimension_threshold(system, target_dimension, **overrides):
    """Find the threshold at which a variant matches a target embedded dimension.

    Both H3 and H6 must be scored at equal cost.  Comparing a frozen basis
    against a self-consistent one, or a symmetric rank allocation against a
    rectangular one, at a common threshold is invalid, because the two arms do
    not produce the same embedded dimension at that threshold.  One arm would
    then be credited for accuracy it bought with extra dimensions.
    """
    candidates = np.logspace(-1, -6, 26)
    best = None
    for threshold in candidates:
        cell = run_cell(system, f'match_eps_{threshold:.2e}',
                        svd_eps=float(threshold), **overrides)
        if cell['classification'] == 'ERROR':
            continue
        gap = abs(cell['embedded_dimension'] - target_dimension)
        relative = gap / max(target_dimension, 1)
        if best is None or relative < best[0]:
            best = (relative, float(threshold), cell)
        if relative <= MATCHED_DIMENSION_TOLERANCE:
            break
    return best


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    started = time.perf_counter()
    report = {
        'protocol': PROTOCOL,
        'system': args.system,
        'system_definition': SYSTEMS[args.system],
        'fixed_controls': CONTROLS,
        'thresholds': list(args.thresholds),
        'protocol_thresholds': {
            'tol_E_mH': TOL_E_MH,
            'chemical_accuracy_mH': CHEMICAL_ACCURACY_MH,
            'matched_dimension_tolerance': MATCHED_DIMENSION_TOLERANCE,
        },
        'groups': {},
    }

    def write():
        path = os.path.join(args.output_dir,
                            'iterative_ci_feasibility_scan.json')
        temporary = path + '.tmp'
        with open(temporary, 'w', encoding='utf-8') as handle:
            json.dump(_serializable(report), handle, indent=2)
        os.replace(temporary, path)
        return path

    def show(cell):
        print(f"    {cell['label']:<34} D={cell.get('embedded_dimension','-'):>5} "
              f"tot={cell.get('weighted_total_error_mH', float('nan')):>10.4f} "
              f"schmidt={cell.get('weighted_schmidt_error_mH', float('nan')):>10.4f} "
              f"wave={cell.get('weighted_wave_operator_error_mH', float('nan')):>9.4f} "
              f"{cell['classification']}", flush=True)

    # ---- H1 and H2: reachability and stability over seeds -----------------
    if 'h1_h2' in args.groups:
        print('\n[H1/H2] seed reachability and stability', flush=True)
        cells = []
        seeds = [('lanczos', {'seed': 'lanczos', 'seed_lanczos_steps': k})
                 for k in (2, 3, 4, 6)]
        seeds += [('control', {'seed': name})
                  for name in ('exact', 'hf', 'cis', 'selci')]
        for threshold in args.thresholds:
            for family, settings in seeds:
                name = settings.get('seed')
                steps = settings.get('seed_lanczos_steps')
                label = (f"{name}"
                         + (f"_k{steps}" if steps else "")
                         + f"_eps{threshold:.0e}")
                cell = run_cell(args.system, label, svd_eps=float(threshold),
                                **settings)
                cell['family'] = family
                cells.append(cell)
                show(cell)
        report['groups']['h1_h2'] = {'cells': cells}
        write()

    # ---- H3: frozen versus self-consistent, at matched dimension ----------
    if 'h3' in args.groups:
        print('\n[H3] frozen versus self-consistent at matched dimension',
              flush=True)
        cells = []
        for threshold in args.thresholds:
            sc = run_cell(args.system, f'self_consistent_eps{threshold:.0e}',
                          svd_eps=float(threshold))
            show(sc)
            match = matched_dimension_threshold(
                args.system, sc['embedded_dimension'], outer_max_iter=1)
            if match is None:
                cells.append({'self_consistent': sc, 'frozen': None})
                continue
            relative, threshold_frozen, frozen = match
            frozen['label'] = (f'frozen_matched_to_eps{threshold:.0e}'
                               f'_at_eps{threshold_frozen:.2e}')
            frozen['matched_dimension_relative_gap'] = relative
            frozen['matched_within_tolerance'] = bool(
                relative <= MATCHED_DIMENSION_TOLERANCE)
            show(frozen)
            difference = (frozen['weighted_total_error_mH']
                          - sc['weighted_total_error_mH'])
            cells.append({
                'self_consistent': sc, 'frozen': frozen,
                'frozen_minus_self_consistent_mH': float(difference),
                'self_consistency_helps': bool(difference > TOL_E_MH),
                'indistinguishable': bool(abs(difference) <= TOL_E_MH),
                'self_consistency_hurts': bool(difference < -TOL_E_MH),
            })
        report['groups']['h3'] = {'cells': cells}
        write()

    # ---- H4: residual dressing on versus off ------------------------------
    if 'h4' in args.groups:
        print('\n[H4] residual dressing on versus off', flush=True)
        cells = []
        for threshold in args.thresholds:
            for label, extra in (('dressed', {}),
                                 ('undressed', {'apply_dressing': False})):
                cell = run_cell(args.system, f'{label}_eps{threshold:.0e}',
                                svd_eps=float(threshold), **extra)
                cells.append(cell)
                show(cell)
        report['groups']['h4'] = {'cells': cells}
        write()

    # ---- H6: rank mode at matched embedded dimension ----------------------
    if 'h6' in args.groups:
        print('\n[H6] rank allocation at matched embedded dimension', flush=True)
        cells = []
        for threshold in args.thresholds:
            rect = run_cell(args.system, f'rectangular_eps{threshold:.0e}',
                            svd_eps=float(threshold), rank_mode='rectangular')
            show(rect)
            match = matched_dimension_threshold(
                args.system, rect['embedded_dimension'])
            if match is None:
                rect['matched'] = None
                cells.append(rect)
                continue
            relative, threshold_sym, sym = match
            sym['label'] = (f'symmetric_matched_to_eps{threshold:.0e}'
                            f'_at_eps{threshold_sym:.2e}')
            sym['matched_dimension_relative_gap'] = relative
            sym['matched_within_tolerance'] = bool(
                relative <= MATCHED_DIMENSION_TOLERANCE)
            show(sym)
            difference = (sym['weighted_total_error_mH']
                          - rect['weighted_total_error_mH'])
            cells.append({'rectangular': rect, 'symmetric': sym,
                          'symmetric_minus_rectangular_mH': float(difference),
                          'rectangular_wins': bool(difference > TOL_E_MH),
                          'indistinguishable': bool(abs(difference) <= TOL_E_MH)})
        report['groups']['h6'] = {'cells': cells}
        write()

    # ---- H7: shared versus per-state wave operator ------------------------
    if 'h7' in args.groups:
        print('\n[H7] shared versus per-state wave operator', flush=True)
        cells = []
        for threshold in args.thresholds:
            for mode in ('shared', 'per_state'):
                cell = run_cell(args.system, f'{mode}_eps{threshold:.0e}',
                                svd_eps=float(threshold), omega_mode=mode)
                cells.append(cell)
                show(cell)
        report['groups']['h7'] = {'cells': cells}
        write()

    report['wall_time_seconds'] = float(time.perf_counter() - started)
    report['peak_rss_mib'] = peak_rss_mib()
    path = write()
    print(f"\nreport: {path}", flush=True)
    print(f"wall={report['wall_time_seconds']:.1f}s "
          f"peakRSS={report['peak_rss_mib']:.0f} MiB", flush=True)


def _serializable(value):
    if isinstance(value, dict):
        return {str(k): _serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serializable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


if __name__ == '__main__':
    main()
