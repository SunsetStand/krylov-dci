#!/usr/bin/env python
"""N2 four-state scan over svd_eps and Lanczos steps for the irrep-complete seed.

Measures whether ``seed='lanczos_symm'`` reaches chemical accuracy on all four
of the three lowest N2 levels, including the exactly degenerate 3Pi_g pair that
the default ``lanczos`` seed cannot reach at all
(``docs/development/seed_irrep_coverage_finding.md``).

Writes a small JSON summary only; no checkpoints, no vectors. Safe to commit.

Example:
    python scripts/diagnostics/run_n2_symm_seed_scan.py \
        --points 10@1e-3 14@1e-3 10@5e-4 --output-dir results/symm_seed_scan
"""
import argparse
import json
import os
import resource
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from dm_svd_dci.pipeline_state_averaged import run_state_averaged_dci  # noqa: E402

SYSTEM = dict(atom='N 0 0 0; N 0 0 1.098', basis='cc-pVDZ', n_active=9,
              n_active_elec=(5, 5), n_core=2, n_occ=5, p_blocks=[8, 9, 10],
              sa_states=4)
# Pre-registered numerical controls, docs/theory/state_averaged_validation_protocol.md
CONTROLS = dict(outer_mixing=0.7, outer_density_tol=1e-6, outer_energy_tol=1e-7,
                outer_max_iter=12, wave_damping=0.7, wave_residual_tol=1e-9,
                wave_energy_tol=1e-10, wave_max_iter=200, min_denominator=1e-6)
# Committed bundle, checksum ad5b19d6674d54411f704bd3ed4b65aab9741d5373037868fc8fb76e85133d9d
BUNDLE = [-109.0401239716, -108.7408683094, -108.7218810454, -108.7218810454]
LABELS = ['S0 Ag', 'S1 B1u', 'S2 B2g', 'S3 B3g']
CHEMICAL_ACCURACY_MH = 1.6


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--points', nargs='+', required=True,
                   help="STEPS@EPS pairs, e.g. 10@1e-3 14@5e-4")
    p.add_argument('--seed', default='lanczos_symm')
    p.add_argument('--symmetry', default='D2h',
                   help="Abelian point group; required by lanczos_symm")
    p.add_argument('--enrichment', type=float, default=0.0)
    p.add_argument('--output-dir', required=True)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, 'n2_symm_seed_scan.json')
    rows = []

    for spec in args.points:
        steps_text, eps_text = spec.split('@')
        steps, eps = int(steps_text), float(eps_text)
        started = time.perf_counter()
        settings = dict(SYSTEM)
        settings.update(CONTROLS)
        settings.update(svd_eps=eps, seed=args.seed, seed_lanczos_steps=steps,
                        symmetry=args.symmetry,
                        enrichment_strength=args.enrichment,
                        embedded_spectrum=True, verbose=False)
        try:
            result = run_state_averaged_dci(**settings)
        except Exception as error:                              # noqa: BLE001
            print(f"steps={steps} eps={eps:.0e}  ERROR "
                  f"{type(error).__name__}: {error}", flush=True)
            rows.append(dict(steps=steps, svd_eps=eps,
                             error=f"{type(error).__name__}: {error}"))
            json.dump(rows, open(out, 'w'), indent=1)
            continue

        energies = np.sort(np.asarray(result['energies']).ravel())
        per_state = [abs(float(e) - b) * 1000.0
                     for e, b in zip(energies, BUNDLE)]
        weights = np.asarray(result['state_weights'])
        ranks = result['schmidt_metrics']['ranks']
        block9 = ranks.get(9) or ranks.get('9') or {'r_A': 0, 'r_B': 0}
        row = dict(
            steps=steps, svd_eps=eps, seed=args.seed, symmetry=args.symmetry,
            D=result['partition_info']['D_total'],
            P_dim=result['partition_info']['P_dim'],
            Q_dim=result['partition_info']['Q_dim'],
            energies=[float(e) for e in energies],
            per_state_error_mH=per_state,
            weighted_error_mH=float(np.dot(weights, per_state)),
            max_error_mH=float(np.max(per_state)),
            degenerate_split_mH=float((energies[3] - energies[2]) * 1000.0),
            wave_operator_error_mH=float(np.dot(
                weights, np.abs(np.asarray(result['wave_operator_errors_mH'])))),
            discarded_weight=result['schmidt_metrics']['discarded_weight'],
            block9_rank=[int(block9['r_A']), int(block9['r_B'])],
            converged=bool(result['converged']),
            n_outer_iter=result['n_outer_iter'],
            all_states_chemically_accurate=bool(
                max(per_state) < CHEMICAL_ACCURACY_MH),
            wall_seconds=time.perf_counter() - started,
            peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        )
        rows.append(row)
        json.dump(rows, open(out, 'w'), indent=1)

        detail = "  ".join(f"{LABELS[i]}={v:.3f}"
                           for i, v in enumerate(per_state))
        print(f"steps={steps:<3} eps={eps:.0e} D={row['D']:<6d} "
              f"weighted={row['weighted_error_mH']:8.3f} mH | {detail} | "
              f"split={row['degenerate_split_mH']:.3f} "
              f"blk9={block9['r_A']}x{block9['r_B']} "
              f"conv={row['converged']} "
              f"{row['wall_seconds']:.0f}s {row['peak_rss_mib']:.0f}MiB",
              flush=True)

    print(f"\nwrote {out}", flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
