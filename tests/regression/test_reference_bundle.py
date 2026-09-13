#!/usr/bin/env python3
"""Regression checks for the deterministic multi-root reference bundle.

Protects the properties that make a bundle usable as a shared reference for
frozen and self-consistent downfolding:

  * the target energies are the true lowest levels, to 1e-8 Ha
  * a degenerate level is retained whole and its members stay degenerate
  * the bundle is bit-reproducible, so its checksum is stable
  * a tampered bundle is rejected on load
  * the naive nroots = n_target solve, which this machinery exists to replace,
    is demonstrably wrong -- a negative control

Run directly:  python tests/regression/test_reference_bundle.py
"""

import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from dm_svd_dci.reference_bundle import (  # noqa: E402
    ReferenceBundleError,
    _Hamiltonian,
    build_reference_bundle,
    group_into_levels,
    load_reference_bundle,
    save_reference_bundle,
)

N2 = {
    'atom': 'N 0 0 0; N 0 0 1.098',
    'basis': 'cc-pVDZ',
    'n_active': 9,
    'n_active_elec': (5, 5),
    'target_levels': 3,
    'point_group': 'D2h',
}

# Locked reference values for N2/cc-pVDZ CAS(10e,9o), (5a,5b), frozen core 2.
EXPECTED_ENERGIES = [
    -109.0401239716,
    -108.7408683094,
    -108.7218810454,
    -108.7218810454,
]
EXPECTED_IRREPS = ['Ag', 'B1u', 'B2g', 'B3g']
EXPECTED_SPIN_SQUARED = [0.0, 2.0, 2.0, 2.0]
EXPECTED_LEVELS = [[0], [1], [2, 3]]

ENERGY_TOL = 1e-8
DEGENERACY_TOL = 1e-9
RESIDUAL_TOL = 1e-4


def _check(condition, message):
    if not condition:
        raise AssertionError(message)
    print(f'  ok: {message}', flush=True)


def test_group_into_levels():
    print('group_into_levels', flush=True)
    levels = group_into_levels([-1.0, -0.5, -0.5, -0.2], tolerance=1e-9)
    _check(levels == [[0], [1, 2], [3]], f'degenerate pair grouped: {levels}')
    levels = group_into_levels([-1.0, -1.0 + 1e-6], tolerance=1e-9)
    _check(levels == [[0], [1]], 'near-degenerate pair not merged at 1e-9')


def test_bundle_contents(bundle):
    print('bundle contents', flush=True)
    metadata = bundle['metadata']
    target = metadata['target']

    _check(target['n_target_roots'] == 4, 'four roots cover three levels')
    _check(target['level_grouping'] == EXPECTED_LEVELS,
           f"level grouping {target['level_grouping']}")

    errors = [abs(a - b) for a, b in zip(target['energies'],
                                         EXPECTED_ENERGIES)]
    _check(max(errors) < ENERGY_TOL,
           f'energies match reference to {max(errors):.2e} Ha')

    split = abs(target['energies'][3] - target['energies'][2])
    _check(split < DEGENERACY_TOL,
           f'degenerate pair split {split:.2e} Ha below {DEGENERACY_TOL:.0e}')

    irreps = [record['irrep'] for record in target['root_character']]
    _check(irreps == EXPECTED_IRREPS, f'irreps {irreps}')

    for record, expected in zip(target['root_character'],
                                EXPECTED_SPIN_SQUARED):
        _check(abs(record['spin_squared'] - expected) < 1e-6,
               f"root {record['index']} S^2 = {record['spin_squared']:.4f}")
        _check(record['residual_norm'] < RESIDUAL_TOL,
               f"root {record['index']} residual "
               f"{record['residual_norm']:.2e}")

    disagreement = metadata['provenance']['cross_check_max_disagreement_Ha']
    _check(disagreement < ENERGY_TOL,
           f'symmetry and overshoot procedures agree to {disagreement:.2e} Ha')

    vectors = bundle['arrays']['ci_vectors']
    _check(vectors.shape[0] == 4, f'four CI vectors stored: {vectors.shape}')
    norms = np.linalg.norm(vectors, axis=1)
    _check(np.allclose(norms, 1.0, atol=1e-10),
           f'CI vectors normalized, max deviation {np.max(abs(norms-1)):.2e}')


def test_checksum_reproducible(bundle):
    print('checksum reproducibility', flush=True)
    rebuilt = build_reference_bundle(verbose=False, **N2)
    _check(rebuilt['metadata']['checksum'] == bundle['metadata']['checksum'],
           'independent rebuild reproduces the checksum')


def test_roundtrip_and_tamper_rejection(bundle):
    print('save, load and tamper rejection', flush=True)
    directory = tempfile.mkdtemp(prefix='krylov_bundle_')
    try:
        save_reference_bundle(bundle, directory)
        reloaded = load_reference_bundle(directory, verify=True)
        _check(reloaded['metadata']['checksum'] == bundle['metadata']['checksum'],
               'round trip preserves the checksum')

        arrays = dict(reloaded['arrays'])
        arrays['energies'] = arrays['energies'] + 1e-6
        np.savez_compressed(
            os.path.join(directory, 'reference_bundle.npz'), **arrays)
        try:
            load_reference_bundle(directory, verify=True)
        except ReferenceBundleError:
            _check(True, 'tampered bundle rejected on load')
        else:
            raise AssertionError('tampered bundle was accepted')
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_naive_solve_is_wrong(bundle):
    """Negative control: the procedure this module replaces must fail."""
    print('negative control -- naive nroots solve', flush=True)
    hamiltonian = _Hamiltonian(N2['atom'], N2['basis'], N2['n_active'],
                               N2['n_active_elec'], N2['point_group'])
    n_target = bundle['metadata']['target']['n_target_roots']
    energies, _ = hamiltonian.solve(n_target)
    deviation = float(np.max(np.abs(
        np.asarray(energies[:n_target])
        - np.asarray(bundle['metadata']['target']['energies']))))
    _check(deviation > 1e-3,
           f'naive nroots={n_target} solve is wrong by {deviation*1000:.2f} mH, '
           'so the bundle machinery is load bearing')


def main():
    print('Reference bundle regression checks', flush=True)
    test_group_into_levels()
    print('building bundle...', flush=True)
    bundle = build_reference_bundle(verbose=False, **N2)
    test_bundle_contents(bundle)
    test_roundtrip_and_tamper_rejection(bundle)
    test_naive_solve_is_wrong(bundle)
    test_checksum_reproducible(bundle)
    print('\nAll reference bundle checks passed.', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
