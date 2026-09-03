#!/usr/bin/env python3
"""CLI for state-averaged residual-dressed self-consistent dmSVD."""

import argparse
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)


def _parse_csv_ints(text):
    return [int(value.strip()) for value in text.split(',') if value.strip()]


def _parse_csv_floats(text):
    if text is None:
        return None
    return np.array([
        float(value.strip()) for value in text.split(',') if value.strip()],
        dtype=float)


def parse_args():
    parser = argparse.ArgumentParser(
        description='State-averaged residual-dressed self-consistent dmSVD')
    parser.add_argument('--atom', default='N 0 0 0; N 0 0 1.098')
    parser.add_argument('--basis', default='cc-pVDZ')
    parser.add_argument('--n-active', type=int, default=10)
    parser.add_argument('--n-alpha', type=int, default=5)
    parser.add_argument('--n-beta', type=int, default=5)
    parser.add_argument('--n-core', type=int, default=2)
    parser.add_argument('--n-occ', type=int, default=5)
    parser.add_argument('--ms', type=int, default=0)
    parser.add_argument('--svd-eps', type=float, default=1e-3)
    parser.add_argument('--sa-states', type=int, default=3)
    parser.add_argument(
        '--state-weights', default=None,
        help='comma-separated weights; default is equal weighting')
    parser.add_argument('--p-blocks', default='8,9,10')
    parser.add_argument('--scheme', choices=['A', 'B', 'B_streaming'], default='A')
    parser.add_argument('--outer-mixing', type=float, default=1.0)
    parser.add_argument('--outer-density-tol', type=float, default=1e-7)
    parser.add_argument('--outer-energy-tol', type=float, default=1e-8)
    parser.add_argument('--outer-max-iter', type=int, default=20)
    parser.add_argument('--wave-damping', type=float, default=0.5)
    parser.add_argument('--wave-residual-tol', type=float, default=1e-9)
    parser.add_argument('--wave-energy-tol', type=float, default=1e-10)
    parser.add_argument('--wave-max-iter', type=int, default=100)
    parser.add_argument('--min-denominator', type=float, default=1e-6)
    parser.add_argument('--n-workers', type=int, default=1)
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--quiet', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    weights = _parse_csv_floats(args.state_weights)
    if weights is not None and len(weights) != args.sa_states:
        raise SystemExit(
            f"--state-weights has {len(weights)} entries; "
            f"--sa-states is {args.sa_states}")

    from dm_svd_dci.pipeline_state_averaged import run_state_averaged_dci

    result = run_state_averaged_dci(
        atom=args.atom,
        basis=args.basis,
        n_active=args.n_active,
        n_active_elec=(args.n_alpha, args.n_beta),
        n_core=args.n_core,
        n_occ=args.n_occ,
        ms=args.ms,
        svd_eps=args.svd_eps,
        sa_states=args.sa_states,
        state_weights=weights,
        p_blocks=_parse_csv_ints(args.p_blocks),
        outer_mixing=args.outer_mixing,
        outer_density_tol=args.outer_density_tol,
        outer_energy_tol=args.outer_energy_tol,
        outer_max_iter=args.outer_max_iter,
        wave_damping=args.wave_damping,
        wave_residual_tol=args.wave_residual_tol,
        wave_energy_tol=args.wave_energy_tol,
        wave_max_iter=args.wave_max_iter,
        min_denominator=args.min_denominator,
        n_workers=args.n_workers,
        scheme=args.scheme,
        output_dir=args.output_dir,
        verbose=not args.quiet)

    if not result['converged']:
        print("WARNING: outer state-averaged loop did not converge")
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
