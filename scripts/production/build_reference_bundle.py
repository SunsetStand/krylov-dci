#!/usr/bin/env python3
"""Build and store a deterministic multi-root CASCI reference bundle.

Frozen and self-consistent downfolding runs must consume exactly the same
reference roots.  See docs/theory/n2_root_targeting_mechanism_protocol.md.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from dm_svd_dci.reference_bundle import (  # noqa: E402
    build_reference_bundle,
    load_reference_bundle,
    save_reference_bundle,
)

SYSTEMS = {
    'n2_cas9': {
        'atom': 'N 0 0 0; N 0 0 1.098',
        'basis': 'cc-pVDZ',
        'n_active': 9,
        'n_active_elec': (5, 5),
        'target_levels': 3,
        'point_group': 'D2h',
    },
    'n2_cas10_legacy': {
        'atom': 'N 0 0 0; N 0 0 1.098',
        'basis': 'cc-pVDZ',
        'n_active': 10,
        'n_active_elec': (5, 5),
        'target_levels': 3,
        'point_group': 'D2h',
    },
    'h2o_sto3g': {
        'atom': 'O 0 0 0; H 0 0.757 0.586; H 0 -0.757 0.586',
        'basis': 'sto-3g',
        'n_active': 5,
        'n_active_elec': (3, 3),
        'target_levels': 3,
        'point_group': 'C2v',
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--system', choices=sorted(SYSTEMS), required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--target-levels', type=int, default=None)
    parser.add_argument('--roots-per-irrep', type=int, default=3)
    parser.add_argument('--max-margin', type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    definition = dict(SYSTEMS[args.system])
    if args.target_levels is not None:
        definition['target_levels'] = args.target_levels

    print(f'Building reference bundle for {args.system}', flush=True)
    bundle = build_reference_bundle(
        roots_per_irrep=args.roots_per_irrep,
        max_margin=args.max_margin,
        **definition)
    paths = save_reference_bundle(bundle, args.output_dir)

    reloaded = load_reference_bundle(args.output_dir, verify=True)
    metadata = reloaded['metadata']
    print('\nReference bundle', flush=True)
    print(f"  checksum   = {metadata['checksum']}", flush=True)
    print(f"  levels     = {metadata['target']['level_grouping']}", flush=True)
    print(f"  margin     = {metadata['provenance']['overshoot_margin']}",
          flush=True)
    print(f"  agreement  = "
          f"{metadata['provenance']['cross_check_max_disagreement_Ha']:.3e} Ha",
          flush=True)
    print(f"{'idx':>4} {'E(Ha)':>16} {'dE(mH)':>10} {'S^2':>7} {'irrep':>7} "
          f"{'resid':>10}", flush=True)
    for record in metadata['target']['root_character']:
        print(f"{record['index']:>4} {record['total_energy']:>16.10f} "
              f"{record['excitation_mH']:>10.4f} "
              f"{record['spin_squared']:>7.3f} {record['irrep']:>7} "
              f"{record['residual_norm']:>10.2e}", flush=True)
    for name, path in paths.items():
        print(f"  {name}: {path}", flush=True)


if __name__ == '__main__':
    main()
