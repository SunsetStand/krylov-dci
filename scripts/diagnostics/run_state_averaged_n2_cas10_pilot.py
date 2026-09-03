#!/usr/bin/env python3
"""N2/cc-pVDZ CAS(10,10) three-state dmSVD pilot.

The solver is intentionally immutable in this diagnostic.  The frozen
baseline is written before the self-consistent point starts, so a later
failure still leaves a complete same-threshold control.
"""

import argparse
import csv
import json
import os
import resource
import sys
import time
import traceback

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from dm_svd_dci.pipeline_state_averaged import run_state_averaged_dci


GEOMETRY = 'N 0 0 0; N 0 0 1.098'
SVD_EPS = 1e-3
STATE_WEIGHTS = np.array([1.0 / 3.0] * 3)
PROTOCOL = 'docs/theory/state_averaged_n2_cas10_pilot.md'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run the documented N2 CAS(10,10) three-state pilot')
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
    return (
        patterns[0] == patterns[2]
        and patterns[1] == patterns[3]
        and patterns[0] != patterns[1]
    )


def summarize_result(label, outer_mode, result, seconds):
    energies = np.asarray(result['energies'], dtype=float)
    references = np.asarray(result['reference_energies'], dtype=float)
    errors = np.asarray(result['errors_mH'], dtype=float)
    weights = np.asarray(result['state_weights'], dtype=float)
    history = result['outer_history']

    finite = (
        np.all(np.isfinite(energies))
        and np.all(np.isfinite(references))
        and np.all(np.isfinite(errors))
    )
    enough_p_states = int(result['partition_info']['P_dim']) >= len(energies)
    all_inner_converged = all(bool(item['wave_converged']) for item in history)
    root_reorder = any(
        list(np.asarray(item['root_permutation'], dtype=int))
        != list(range(len(energies)))
        for item in history
    )
    rank_oscillation = has_rank_oscillation(history)

    if not finite or not enough_p_states:
        classification = 'ERROR'
    elif not all_inner_converged:
        classification = 'INNER_NONCONVERGED'
    elif outer_mode == 'frozen':
        classification = 'FROZEN_BASELINE'
    elif rank_oscillation:
        classification = 'RANK_OSCILLATION'
    elif not result['converged']:
        classification = 'OUTER_NONCONVERGED'
    else:
        classification = 'PASS'

    return {
        'label': label,
        'family': 'equal_frozen' if outer_mode == 'frozen' else 'equal_sc',
        'svd_eps': SVD_EPS,
        'weights': weights,
        'outer_mode': outer_mode,
        'classification': classification,
        'flags': {
            'root_reorder': root_reorder,
            'rank_oscillation': rank_oscillation,
        },
        'energies': energies,
        'reference_energies': references,
        'errors_mH': errors,
        'weighted_abs_error_mH': float(np.dot(weights, np.abs(errors))),
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
        'process_peak_rss_kib': int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        'delta_vs_frozen_mH': None,
    }


def write_reports(output_dir, report):
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, 'n2_cas10_three_state_pilot.json')
    temporary_path = json_path + '.tmp'
    with open(temporary_path, 'w', encoding='utf-8') as handle:
        json.dump(serializable(report), handle, indent=2)
    os.replace(temporary_path, json_path)

    csv_path = os.path.join(output_dir, 'n2_cas10_three_state_pilot.csv')
    fields = [
        'label', 'family', 'svd_eps', 'outer_mode', 'classification',
        'weights', 'energies', 'reference_energies', 'errors_mH',
        'weighted_abs_error_mH', 'max_abs_error_mH', 'outer_converged',
        'outer_iterations', 'all_inner_converged', 'final_wave_iterations',
        'final_wave_residual_rms', 'D_total', 'P_dim', 'Q_dim',
        'root_reorder', 'rank_oscillation', 'delta_vs_frozen_mH',
        'wall_time_seconds', 'process_peak_rss_kib',
    ]
    with open(csv_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in report['records']:
            if record.get('classification') == 'ERROR' and 'partition_info' not in record:
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
                'energies': json.dumps(serializable(record['energies'])),
                'reference_energies': json.dumps(serializable(
                    record['reference_energies'])),
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
                'delta_vs_frozen_mH': json.dumps(serializable(
                    record['delta_vs_frozen_mH'])),
                'wall_time_seconds': record['wall_time_seconds'],
                'process_peak_rss_kib': record['process_peak_rss_kib'],
            })


def run_point(label, outer_mode, args):
    point_dir = os.path.join(args.output_dir, label)
    max_outer_iter = 1 if outer_mode == 'frozen' else 12
    start = time.perf_counter()
    result = run_state_averaged_dci(
        atom=GEOMETRY,
        basis='cc-pVDZ',
        n_active=10,
        n_active_elec=(5, 5),
        n_core=2,
        n_occ=5,
        ms=0,
        svd_eps=SVD_EPS,
        sa_states=3,
        state_weights=STATE_WEIGHTS,
        p_blocks=[8, 9, 10],
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
        verbose=args.verbose_runs,
    )
    return summarize_result(
        label, outer_mode, result, time.perf_counter() - start)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    configurations = [
        ('equal_frozen_eps_1e-3', 'frozen'),
        ('equal_sc_eps_1e-3', 'self_consistent'),
    ]
    report = {
        'protocol': PROTOCOL,
        'system': {
            'atom': GEOMETRY,
            'basis': 'cc-pVDZ',
            'active_space': [10, 10],
            'active_spin_sector': [5, 5],
            'n_core': 2,
            'n_occ': 5,
            'sa_states': 3,
            'state_weights': STATE_WEIGHTS,
            'p_blocks': [8, 9, 10],
            'svd_eps': SVD_EPS,
            'scheme': args.scheme,
        },
        'records': [],
        'started_at_epoch_seconds': time.time(),
        'completed': False,
    }
    frozen_energies = None

    for index, (label, outer_mode) in enumerate(configurations, start=1):
        print(
            f'[{index}/{len(configurations)}] {label}: '
            f'weights={STATE_WEIGHTS.tolist()}, mode={outer_mode}',
            flush=True,
        )
        try:
            record = run_point(label, outer_mode, args)
            if outer_mode == 'frozen':
                frozen_energies = np.asarray(record['energies'])
            elif frozen_energies is not None:
                record['delta_vs_frozen_mH'] = (
                    (np.asarray(record['energies']) - frozen_energies) * 1000.0
                )
            print(
                f"  {record['classification']}: "
                f"max_error={record['max_abs_error_mH']:.6f} mH, "
                f"outer={record['outer_iterations']}, "
                f"R={record['final_wave_residual_rms']:.3e}, "
                f"D={record['partition_info']['D_total']}, "
                f"peak_rss={record['process_peak_rss_kib'] / 1024.0:.1f} MiB",
                flush=True,
            )
        except Exception as error:
            traceback.print_exc()
            record = {
                'label': label,
                'family': (
                    'equal_frozen' if outer_mode == 'frozen' else 'equal_sc'),
                'svd_eps': SVD_EPS,
                'weights': STATE_WEIGHTS,
                'outer_mode': outer_mode,
                'classification': 'ERROR',
                'error_type': type(error).__name__,
                'error_message': str(error),
                'process_peak_rss_kib': int(
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
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

    print('Pilot classification counts:', classifications, flush=True)
    print('JSON:', os.path.join(
        args.output_dir, 'n2_cas10_three_state_pilot.json'), flush=True)
    print('CSV:', os.path.join(
        args.output_dir, 'n2_cas10_three_state_pilot.csv'), flush=True)
    return 2 if classifications.get('ERROR', 0) else 0


if __name__ == '__main__':
    raise SystemExit(main())
