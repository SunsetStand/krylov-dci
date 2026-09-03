#!/usr/bin/env python3
"""H2O truncated state-averaged dmSVD validation scan.

This diagnostic intentionally treats the solver as immutable.  It executes the
nine points defined in docs/theory/state_averaged_validation_protocol.md and
writes an incremental JSON report so completed points survive a later failure.
"""

import argparse
import csv
import json
import os
import sys
import time
import traceback

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from dm_svd_dci.pipeline_state_averaged import run_state_averaged_dci


GEOMETRY = 'O 0 0 0; H 0 0.757 0.586; H 0 -0.757 0.586'
THRESHOLDS = (1e-2, 1e-3, 1e-4)
EQUAL_WEIGHTS = np.array([1.0 / 3.0] * 3)
BIASED_WEIGHTS = np.array([0.60, 0.25, 0.15])


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run the documented H2O state-averaged truncation scan')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--n-workers', type=int, default=1)
    parser.add_argument('--scheme', choices=['A', 'B', 'B_streaming'], default='A')
    parser.add_argument('--verbose-runs', action='store_true')
    return parser.parse_args()


def serializable(value):
    if isinstance(value, dict):
        return {str(key): serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def rank_pattern(history_item):
    ranks = history_item.get('schmidt_ranks', {})
    pattern = []
    for label in sorted(ranks, key=lambda value: int(value)):
        data = ranks[label]
        pattern.append((int(label), int(data['r_A']), int(data['r_B'])))
    return tuple(pattern)


def has_rank_oscillation(history):
    if len(history) < 4:
        return False
    patterns = [rank_pattern(item) for item in history[-4:]]
    return patterns[0] == patterns[2] and patterns[1] == patterns[3] \
        and patterns[0] != patterns[1]


def summarize_result(label, family, eps, weights, outer_mode, result, seconds):
    energies = np.asarray(result['energies'], dtype=float)
    references = np.asarray(result['reference_energies'], dtype=float)
    errors = np.asarray(result['errors_mH'], dtype=float)
    normalized_weights = np.asarray(result['state_weights'], dtype=float)
    history = result['outer_history']

    finite = (
        np.all(np.isfinite(energies))
        and np.all(np.isfinite(references))
        and np.all(np.isfinite(errors)))
    all_inner_converged = all(
        bool(item['wave_converged']) for item in history)
    root_reorder = any(
        list(np.asarray(item['root_permutation'], dtype=int))
        != list(range(len(energies)))
        for item in history)
    rank_oscillation = has_rank_oscillation(history)

    if not finite:
        classification = 'ERROR'
    elif not all_inner_converged:
        classification = 'INNER_NONCONVERGED'
    elif outer_mode == 'frozen':
        classification = 'FROZEN_BASELINE'
    elif not result['converged']:
        classification = 'OUTER_NONCONVERGED'
    elif rank_oscillation:
        classification = 'RANK_OSCILLATION'
    else:
        classification = 'PASS'

    return {
        'label': label,
        'family': family,
        'svd_eps': eps,
        'weights': normalized_weights,
        'outer_mode': outer_mode,
        'classification': classification,
        'flags': {
            'root_reorder': root_reorder,
            'rank_oscillation': rank_oscillation,
        },
        'energies': energies,
        'reference_energies': references,
        'errors_mH': errors,
        'weighted_abs_error_mH': float(np.dot(
            normalized_weights, np.abs(errors))),
        'max_abs_error_mH': float(np.max(np.abs(errors))),
        'outer_converged': bool(result['converged']),
        'outer_iterations': int(result['n_outer_iter']),
        'all_inner_converged': all_inner_converged,
        'final_wave_converged': bool(result['final_wave_converged']),
        'final_wave_iterations': int(result['final_wave_iterations']),
        'final_wave_residual_rms': float(result['final_wave_residual_rms']),
        'schmidt_metrics': result['schmidt_metrics'],
        'partition_info': result['partition_info'],
        'outer_history': history,
        'build_history': result['build_history'],
        'wall_time_seconds': float(seconds),
        'pipeline_wall_time_seconds': float(result['wall_time_seconds']),
        'delta_vs_equal_frozen_mH': None,
    }


def write_reports(output_dir, report):
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, 'h2o_state_averaged_scan.json')
    temporary_path = json_path + '.tmp'
    with open(temporary_path, 'w', encoding='utf-8') as handle:
        json.dump(serializable(report), handle, indent=2)
    os.replace(temporary_path, json_path)

    csv_path = os.path.join(output_dir, 'h2o_state_averaged_scan.csv')
    fields = [
        'label', 'family', 'svd_eps', 'outer_mode', 'classification',
        'weights', 'errors_mH', 'weighted_abs_error_mH',
        'max_abs_error_mH', 'outer_converged', 'outer_iterations',
        'all_inner_converged', 'final_wave_iterations',
        'final_wave_residual_rms', 'D_total', 'P_dim', 'Q_dim',
        'root_reorder', 'rank_oscillation', 'delta_vs_equal_frozen_mH',
        'wall_time_seconds',
    ]
    with open(csv_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in report['records']:
            if record.get('classification') == 'ERROR':
                writer.writerow({
                    'label': record['label'],
                    'family': record['family'],
                    'svd_eps': record['svd_eps'],
                    'outer_mode': record['outer_mode'],
                    'classification': 'ERROR',
                })
                continue
            writer.writerow({
                'label': record['label'],
                'family': record['family'],
                'svd_eps': record['svd_eps'],
                'outer_mode': record['outer_mode'],
                'classification': record['classification'],
                'weights': json.dumps(serializable(record['weights'])),
                'errors_mH': json.dumps(serializable(record['errors_mH'])),
                'weighted_abs_error_mH': record['weighted_abs_error_mH'],
                'max_abs_error_mH': record['max_abs_error_mH'],
                'outer_converged': record['outer_converged'],
                'outer_iterations': record['outer_iterations'],
                'all_inner_converged': record['all_inner_converged'],
                'final_wave_iterations': record['final_wave_iterations'],
                'final_wave_residual_rms': record['final_wave_residual_rms'],
                'D_total': record['partition_info']['D_total'],
                'P_dim': record['partition_info']['P_dim'],
                'Q_dim': record['partition_info']['Q_dim'],
                'root_reorder': record['flags']['root_reorder'],
                'rank_oscillation': record['flags']['rank_oscillation'],
                'delta_vs_equal_frozen_mH': json.dumps(serializable(
                    record['delta_vs_equal_frozen_mH'])),
                'wall_time_seconds': record['wall_time_seconds'],
            })


def run_point(label, family, eps, weights, outer_mode, args):
    point_dir = os.path.join(args.output_dir, label)
    max_outer_iter = 1 if outer_mode == 'frozen' else 12
    start = time.perf_counter()
    result = run_state_averaged_dci(
        atom=GEOMETRY,
        basis='sto-3g',
        n_active=5,
        n_active_elec=(3, 3),
        n_core=2,
        n_occ=3,
        ms=0,
        svd_eps=eps,
        sa_states=3,
        state_weights=weights,
        p_blocks=[4, 5, 6],
        outer_mixing=0.7,
        outer_density_tol=1e-6,
        outer_energy_tol=1e-7,
        outer_max_iter=max_outer_iter,
        wave_damping=0.7,
        wave_residual_tol=1e-9,
        wave_energy_tol=1e-10,
        wave_max_iter=200,
        min_denominator=1e-6,
        n_workers=args.n_workers,
        scheme=args.scheme,
        output_dir=point_dir,
        verbose=args.verbose_runs)
    return summarize_result(
        label, family, eps, weights, outer_mode,
        result, time.perf_counter() - start)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    configurations = []
    for eps in THRESHOLDS:
        suffix = f'{eps:.0e}'.replace('-', 'm')
        configurations.append((
            f'equal_frozen_eps_{suffix}', 'equal_frozen',
            eps, EQUAL_WEIGHTS, 'frozen'))
    for eps in THRESHOLDS:
        suffix = f'{eps:.0e}'.replace('-', 'm')
        configurations.append((
            f'equal_sc_eps_{suffix}', 'equal_sc',
            eps, EQUAL_WEIGHTS, 'self_consistent'))
    for eps in THRESHOLDS:
        suffix = f'{eps:.0e}'.replace('-', 'm')
        configurations.append((
            f'biased_sc_eps_{suffix}', 'biased_sc',
            eps, BIASED_WEIGHTS, 'self_consistent'))

    report = {
        'protocol': 'docs/theory/state_averaged_validation_protocol.md',
        'system': {
            'atom': GEOMETRY,
            'basis': 'sto-3g',
            'active_space': [6, 5],
            'n_core': 2,
            'n_occ': 3,
            'sa_states': 3,
            'p_blocks': [4, 5, 6],
            'scheme': args.scheme,
        },
        'records': [],
        'started_at_epoch_seconds': time.time(),
        'completed': False,
    }
    frozen_by_eps = {}

    for index, (label, family, eps, weights, outer_mode) in enumerate(
            configurations, start=1):
        print(
            f'[{index}/{len(configurations)}] {label}: '
            f'weights={weights.tolist()}, mode={outer_mode}',
            flush=True)
        try:
            record = run_point(
                label, family, eps, weights, outer_mode, args)
            if family == 'equal_frozen':
                frozen_by_eps[eps] = np.asarray(record['energies'])
            elif eps in frozen_by_eps:
                record['delta_vs_equal_frozen_mH'] = (
                    (np.asarray(record['energies']) - frozen_by_eps[eps]) * 1000.0)
            print(
                f"  {record['classification']}: "
                f"max_error={record['max_abs_error_mH']:.6f} mH, "
                f"outer={record['outer_iterations']}, "
                f"R={record['final_wave_residual_rms']:.3e}, "
                f"D={record['partition_info']['D_total']}",
                flush=True)
        except Exception as error:
            traceback.print_exc()
            record = {
                'label': label,
                'family': family,
                'svd_eps': eps,
                'weights': weights,
                'outer_mode': outer_mode,
                'classification': 'ERROR',
                'error_type': type(error).__name__,
                'error_message': str(error),
            }
            print(f'  ERROR: {type(error).__name__}: {error}', flush=True)

        report['records'].append(record)
        write_reports(args.output_dir, report)

    report['completed'] = True
    report['finished_at_epoch_seconds'] = time.time()
    classifications = {}
    for record in report['records']:
        key = record['classification']
        classifications[key] = classifications.get(key, 0) + 1
    report['classification_counts'] = classifications
    write_reports(args.output_dir, report)

    print('Scan classification counts:', classifications, flush=True)
    print('JSON:', os.path.join(
        args.output_dir, 'h2o_state_averaged_scan.json'), flush=True)
    print('CSV:', os.path.join(
        args.output_dir, 'h2o_state_averaged_scan.csv'), flush=True)
    return 2 if classifications.get('ERROR', 0) else 0


if __name__ == '__main__':
    raise SystemExit(main())
