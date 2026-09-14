#!/usr/bin/env python3
"""Measure whether state-averaged Schmidt ranks are genuinely rectangular.

The state-averaged dmSVD retains the left and right ranks of each
electron-number block independently, so a block has dimension
``r_A(n) * r_B(n)`` rather than ``r(n)^2``.  This script establishes, as data,
three things that claim depends on:

1.  A *pure* bipartite state has identical rho_A and rho_B spectra, so
    ``r_A == r_B`` exactly.  Any asymmetry must therefore come from the state
    average, not from ``dim_A != dim_B``.  This is the control.
2.  Whether the asymmetry survives when *both* sides are genuinely truncated.
    A block where ``r_B == dim_B`` is not evidence of rectangularity: that side
    simply was not truncated.
3.  The embedded dimension of the rectangular allocation against the two
    symmetric allocations it sits between, which is what an equal-cost
    comparison must be matched on.

Reference-layer diagnostic.  It does not call the wave-operator solver.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from dm_svd_dci.reference_bundle import (  # noqa: E402
    bundle_ci_vectors,
    load_reference_bundle,
)
from dm_svd_embedding.density_matrix import (  # noqa: E402
    compute_schmidt_decomposition,
)
from dm_svd_embedding.occ_virt_partition import (  # noqa: E402
    build_block_matrices,
    setup_partition,
)

DEFAULT_THRESHOLDS = (3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--n-occ', type=int, default=5,
                        help='size of Schmidt space A, in spatial orbitals')
    parser.add_argument('--thresholds', type=float, nargs='+',
                        default=list(DEFAULT_THRESHOLDS))
    return parser.parse_args()


def block_summary(schmidt):
    """Per-block ranks, dimensions, saturation and asymmetry."""
    rows = []
    for n in sorted(schmidt):
        data = schmidt[n]
        r_a, r_b = int(data['r_A']), int(data['r_B'])
        dim_a, dim_b = int(data['dim_A']), int(data['dim_B'])
        if r_a == 0 and r_b == 0:
            continue
        genuine = bool(r_a < dim_a and r_b < dim_b)
        rows.append({
            'n': int(n),
            'r_A': r_a, 'r_B': r_b,
            'dim_A': dim_a, 'dim_B': dim_b,
            'saturated_A': bool(r_a >= dim_a),
            'saturated_B': bool(r_b >= dim_b),
            'genuinely_truncated_both_sides': genuine,
            'asymmetric': bool(r_a != r_b),
            'genuine_and_asymmetric': bool(genuine and r_a != r_b),
        })
    return rows


def dimensions(rows):
    rect = sum(row['r_A'] * row['r_B'] for row in rows)
    sym_min = sum(min(row['r_A'], row['r_B']) ** 2 for row in rows)
    sym_max = sum(max(row['r_A'], row['r_B']) ** 2 for row in rows)
    return {'D_rectangular': rect, 'D_symmetric_min': sym_min,
            'D_symmetric_max': sym_max}


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    start = time.perf_counter()

    bundle = load_reference_bundle(args.bundle_dir)
    metadata = bundle['metadata']
    vectors = bundle_ci_vectors(bundle)
    n_active = metadata['system']['n_active']
    n_elec = sum(metadata['system']['n_active_elec'])

    partition, _ = setup_partition(n_active, n_elec, args.n_occ, ms=0)
    blocks = [build_block_matrices(partition, vector) for vector in vectors]

    norms = [float(sum(np.sum(block ** 2) for block in state.values()))
             for state in blocks]
    if max(abs(value - 1.0) for value in norms) > 1e-8:
        raise RuntimeError(
            f'determinant ordering mismatch: block norms {norms}')

    report = {
        'bundle_checksum': metadata['checksum'],
        'system': metadata['system'],
        'n_occ': args.n_occ,
        'n_states': len(vectors),
        'block_norms': norms,
        'thresholds': list(args.thresholds),
    }

    # Control: each root alone is a pure bipartite state.
    control = []
    for index, state in enumerate(blocks):
        schmidt = compute_schmidt_decomposition(
            state, eps=args.thresholds[-1], state_average=[state])
        rows = block_summary(schmidt)
        control.append({
            'root': index,
            'n_blocks': len(rows),
            'n_asymmetric': sum(row['asymmetric'] for row in rows),
            'max_abs_rank_difference': max(
                (abs(row['r_A'] - row['r_B']) for row in rows), default=0),
        })
    report['pure_state_control'] = control
    pure_asymmetric = sum(item['n_asymmetric'] for item in control)

    scan = {}
    for threshold in args.thresholds:
        schmidt = compute_schmidt_decomposition(
            blocks[0], eps=threshold, state_average=blocks)
        rows = block_summary(schmidt)
        scan[f'{threshold:.0e}'] = {
            'blocks': rows,
            'n_blocks': len(rows),
            'n_genuine_asymmetric': sum(
                row['genuine_and_asymmetric'] for row in rows),
            'n_genuine_symmetric': sum(
                row['genuinely_truncated_both_sides']
                and not row['asymmetric'] for row in rows),
            'n_saturated': sum(
                not row['genuinely_truncated_both_sides'] for row in rows),
            'dimensions': dimensions(rows),
        }
    report['state_averaged_scan'] = scan
    report['pure_state_asymmetric_blocks_total'] = pure_asymmetric
    report['conclusion'] = (
        'asymmetry_caused_by_state_averaging'
        if pure_asymmetric == 0 else 'asymmetry_present_even_for_pure_states')
    report['wall_time_seconds'] = float(time.perf_counter() - start)

    path = os.path.join(args.output_dir, 'rectangular_schmidt_rank_scan.json')
    temporary = path + '.tmp'
    with open(temporary, 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2)
    os.replace(temporary, path)

    print(f"pure-state control: {pure_asymmetric} asymmetric blocks "
          f"across {len(control)} roots (expected 0)", flush=True)
    print(f"{'eps':>8} {'genuine_asym':>13} {'genuine_sym':>12} "
          f"{'saturated':>10} {'D_rect':>8} {'D_sym_min':>10} "
          f"{'D_sym_max':>10}", flush=True)
    for key, entry in scan.items():
        dims = entry['dimensions']
        print(f"{key:>8} {entry['n_genuine_asymmetric']:>13} "
              f"{entry['n_genuine_symmetric']:>12} {entry['n_saturated']:>10} "
              f"{dims['D_rectangular']:>8} {dims['D_symmetric_min']:>10} "
              f"{dims['D_symmetric_max']:>10}", flush=True)
    print(f"conclusion: {report['conclusion']}", flush=True)
    print(f"report: {path}", flush=True)


if __name__ == '__main__':
    main()
