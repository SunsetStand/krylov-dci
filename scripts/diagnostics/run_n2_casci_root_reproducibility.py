#!/usr/bin/env python3
"""Reproduce and characterize N2 CASCI reference-root selection."""

import argparse
import json
import os
import sys
import time
import traceback
from itertools import permutations

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from dm_svd_dci.pipeline_v2 import setup_system


GEOMETRY = 'N 0 0 0; N 0 0 1.098'
BASIS = 'cc-pVDZ'
N_ACTIVE = 10
N_ACTIVE_ELEC = (5, 5)
N_CORE = 2
PRODUCTION_ROOTS = 3
INVENTORY_ROOTS = 8
PROTOCOL = 'docs/theory/n2_casci_root_reproducibility_protocol.md'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Diagnose N2 CASCI root reproducibility')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--fresh-replicas', type=int, default=3)
    parser.add_argument('--shared-mf-replicas', type=int, default=3)
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


def write_report(output_dir, report):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'n2_casci_root_reproducibility.json')
    temporary_path = path + '.tmp'
    with open(temporary_path, 'w', encoding='utf-8') as handle:
        json.dump(serializable(report), handle, indent=2)
    os.replace(temporary_path, path)


def convergence_value(value):
    if value is None:
        return None
    array = np.asarray(value)
    if array.ndim == 0:
        return bool(array)
    return [bool(item) for item in array.reshape(-1)]


def run_multiroot(label, mf, nroots):
    from pyscf import mcscf
    from pyscf.fci import spin_op

    start = time.perf_counter()
    cas = mcscf.CASCI(mf, N_ACTIVE, sum(N_ACTIVE_ELEC))
    cas.frozen = N_CORE
    cas.fcisolver.nroots = nroots
    cas.kernel()

    energies = np.atleast_1d(np.asarray(cas.e_tot, dtype=float)).reshape(-1)
    active_energies = np.atleast_1d(
        np.asarray(cas.e_cas, dtype=float)).reshape(-1)
    if nroots == 1:
        roots = [np.asarray(cas.ci)]
    else:
        roots = [np.asarray(root) for root in cas.ci]
    if len(roots) < nroots or len(energies) < nroots:
        raise RuntimeError(
            f'{label} returned {len(roots)} vectors and {len(energies)} '
            f'energies for {nroots} roots')

    root_records = []
    normalized_roots = []
    for index, root in enumerate(roots[:nroots]):
        root = np.asarray(root)
        norm = float(np.linalg.norm(root))
        normalized = root.reshape(-1) / norm
        spin_squared, multiplicity = spin_op.spin_square(
            root, N_ACTIVE, N_ACTIVE_ELEC)
        normalized_roots.append(normalized)
        root_records.append({
            'index': index,
            'total_energy': float(energies[index]),
            'active_energy': float(active_energies[index]),
            'excitation_mH': float((energies[index] - energies[0]) * 1000.0),
            'spin_squared': float(spin_squared),
            'multiplicity': float(multiplicity),
            'ci_norm': norm,
        })

    record = {
        'label': label,
        'requested_roots': nroots,
        'returned_roots': len(normalized_roots),
        'casci_converged': convergence_value(getattr(cas, 'converged', None)),
        'fci_converged': convergence_value(
            getattr(cas.fcisolver, 'converged', None)),
        'fci_conv_tol': float(cas.fcisolver.conv_tol),
        'fci_max_cycle': int(cas.fcisolver.max_cycle),
        'roots': root_records,
        'wall_time_seconds': float(time.perf_counter() - start),
    }
    return record, normalized_roots


def compare_root_sets(left_label, left_record, left_roots,
                      right_label, right_record, right_roots):
    overlap = np.abs(np.asarray([
        [np.vdot(left, right) for right in right_roots]
        for left in left_roots
    ]))
    n_left = len(left_roots)
    if len(right_roots) < n_left:
        raise ValueError('right root set is smaller than left root set')
    permutation = max(
        permutations(range(len(right_roots)), n_left),
        key=lambda perm: sum(overlap[index, perm[index]] for index in range(n_left)),
    )
    left_energies = np.asarray([
        root['total_energy'] for root in left_record['roots']])
    right_energies = np.asarray([
        root['total_energy'] for root in right_record['roots']])
    matched_overlaps = [
        float(overlap[index, permutation[index]]) for index in range(n_left)]
    matched_differences = [
        float(right_energies[permutation[index]] - left_energies[index])
        for index in range(n_left)
    ]
    return {
        'left': left_label,
        'right': right_label,
        'absolute_overlap': overlap,
        'best_permutation': list(permutation),
        'matched_overlaps': matched_overlaps,
        'matched_energy_differences_Ha': matched_differences,
    }


def compare_orbitals(reference_mf, candidate_mf):
    overlap_ao = reference_mf.get_ovlp()
    stop = N_CORE + N_ACTIVE
    overlap = (
        reference_mf.mo_coeff[:, :stop].T
        @ overlap_ao
        @ candidate_mf.mo_coeff[:, :stop]
    )
    absolute = np.abs(overlap)
    diagonal = np.diag(absolute)
    off_diagonal = absolute - np.diag(diagonal)
    return {
        'core_active_absolute_overlap': absolute,
        'minimum_diagonal_overlap': float(np.min(diagonal)),
        'maximum_off_diagonal_overlap': float(np.max(off_diagonal)),
    }


def all_flags_true(value):
    if value is None:
        return False
    if isinstance(value, list):
        return all(value)
    return bool(value)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    report = {
        'protocol': PROTOCOL,
        'system': {
            'atom': GEOMETRY,
            'basis': BASIS,
            'active_space': [10, 10],
            'active_spin_sector': list(N_ACTIVE_ELEC),
            'n_core': N_CORE,
            'production_roots': PRODUCTION_ROOTS,
            'inventory_roots': INVENTORY_ROOTS,
        },
        'fresh_records': [],
        'shared_mf_records': [],
        'inventory_record': None,
        'fresh_orbital_comparisons': [],
        'root_comparisons': [],
        'errors': [],
        'started_at_epoch_seconds': time.time(),
        'completed': False,
    }

    fresh_runtime = []
    for index in range(args.fresh_replicas):
        label = f'fresh_{index}'
        print(f'Running {label}', flush=True)
        try:
            system = setup_system(
                atom=GEOMETRY,
                basis=BASIS,
                n_active=N_ACTIVE,
                n_active_elec=N_ACTIVE_ELEC,
                n_core=N_CORE,
                nroots=1,
                verbose=False,
            )
            record, roots = run_multiroot(
                label, system['mf'], PRODUCTION_ROOTS)
            report['fresh_records'].append(record)
            fresh_runtime.append((record, roots, system['mf']))
            print(
                f"  gaps={[root['excitation_mH'] for root in record['roots']]}",
                flush=True,
            )
        except Exception as error:
            traceback.print_exc()
            report['errors'].append({
                'label': label,
                'type': type(error).__name__,
                'message': str(error),
            })
        write_report(args.output_dir, report)

    if fresh_runtime:
        reference_mf = fresh_runtime[0][2]
        for record, roots, mf in fresh_runtime[1:]:
            report['fresh_orbital_comparisons'].append({
                'left': fresh_runtime[0][0]['label'],
                'right': record['label'],
                **compare_orbitals(reference_mf, mf),
            })
        for index in range(len(fresh_runtime)):
            for right_index in range(index + 1, len(fresh_runtime)):
                left = fresh_runtime[index]
                right = fresh_runtime[right_index]
                report['root_comparisons'].append(compare_root_sets(
                    left[0]['label'], left[0], left[1],
                    right[0]['label'], right[0], right[1],
                ))
        write_report(args.output_dir, report)

    print('Building fixed RHF reference', flush=True)
    shared_system = setup_system(
        atom=GEOMETRY,
        basis=BASIS,
        n_active=N_ACTIVE,
        n_active_elec=N_ACTIVE_ELEC,
        n_core=N_CORE,
        nroots=1,
        verbose=False,
    )
    shared_runtime = []
    for index in range(args.shared_mf_replicas):
        label = f'shared_mf_{index}'
        print(f'Running {label}', flush=True)
        try:
            record, roots = run_multiroot(
                label, shared_system['mf'], PRODUCTION_ROOTS)
            report['shared_mf_records'].append(record)
            shared_runtime.append((record, roots))
            print(
                f"  gaps={[root['excitation_mH'] for root in record['roots']]}",
                flush=True,
            )
        except Exception as error:
            traceback.print_exc()
            report['errors'].append({
                'label': label,
                'type': type(error).__name__,
                'message': str(error),
            })
        write_report(args.output_dir, report)

    for index in range(len(shared_runtime)):
        for right_index in range(index + 1, len(shared_runtime)):
            left = shared_runtime[index]
            right = shared_runtime[right_index]
            report['root_comparisons'].append(compare_root_sets(
                left[0]['label'], left[0], left[1],
                right[0]['label'], right[0], right[1],
            ))

    print('Running eight-root inventory', flush=True)
    try:
        inventory_record, inventory_roots = run_multiroot(
            'shared_mf_inventory_8', shared_system['mf'], INVENTORY_ROOTS)
        report['inventory_record'] = inventory_record
        for record, roots in shared_runtime:
            report['root_comparisons'].append(compare_root_sets(
                record['label'], record, roots,
                inventory_record['label'], inventory_record, inventory_roots,
            ))
        print(
            '  inventory gaps=',
            [root['excitation_mH'] for root in inventory_record['roots']],
            flush=True,
        )
    except Exception as error:
        traceback.print_exc()
        report['errors'].append({
            'label': 'shared_mf_inventory_8',
            'type': type(error).__name__,
            'message': str(error),
        })

    all_records = report['fresh_records'] + report['shared_mf_records']
    if report['inventory_record'] is not None:
        all_records.append(report['inventory_record'])
    unconverged = any(
        not all_flags_true(record['fci_converged'])
        or not all_flags_true(record['casci_converged'])
        for record in all_records
    )
    poor_match = any(
        min(comparison['matched_overlaps']) < 0.999999
        for comparison in report['root_comparisons']
        if comparison['right'] != 'shared_mf_inventory_8'
    )
    inventory_subset_drift = any(
        comparison['best_permutation'] != list(range(PRODUCTION_ROOTS))
        for comparison in report['root_comparisons']
        if comparison['right'] == 'shared_mf_inventory_8'
    )
    orbital_drift = any(
        comparison['minimum_diagonal_overlap'] < 0.999999
        or comparison['maximum_off_diagonal_overlap'] > 1e-6
        for comparison in report['fresh_orbital_comparisons']
    )

    classifications = []
    if report['errors']:
        classifications.append('ERROR')
    if unconverged:
        classifications.append('UNCONVERGED_REFERENCE')
    if orbital_drift:
        classifications.append('ORBITAL_DRIFT')
    if poor_match or inventory_subset_drift:
        classifications.append('ROOT_SUBSET_DRIFT')
    if not classifications:
        classifications.append('REPRODUCIBLE')

    report['classifications'] = classifications
    report['completed'] = True
    report['finished_at_epoch_seconds'] = time.time()
    write_report(args.output_dir, report)
    print('Classifications:', classifications, flush=True)
    print('Report:', os.path.join(
        args.output_dir, 'n2_casci_root_reproducibility.json'), flush=True)
    return 2 if report['errors'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
