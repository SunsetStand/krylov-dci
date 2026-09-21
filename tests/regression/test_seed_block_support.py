#!/usr/bin/env python3
"""Seed block support and the self-trapping fixed point of the outer map.

Pins two behaviours that must not drift:

1.  With block completion enabled, every seed family reaches the same fixed
    point at a threshold where nothing is truncated.  This is the H1
    reachability check on the smallest system.
2.  With block completion disabled, a seed that has no weight in an
    electron-number block still fails, by the documented mechanism.  The trap
    is a real property of the map, not a bug that was removed, so the failure
    is pinned deliberately rather than deleted.

See docs/development/seed_block_support_finding.md.

Run directly:  python tests/regression/test_seed_block_support.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from dm_svd_dci.initializers import (SEED_FAMILIES,  # noqa: E402
                                     SEEDS_REQUIRING_SYMMETRY)

# lanczos_symm needs setup_system(symmetry=...) for orbsym, which this
# harness does not build; tests/regression/test_seed_irrep_coverage.py
# covers it.
PLAIN_SEEDS = tuple(s for s in SEED_FAMILIES
                    if s not in SEEDS_REQUIRING_SYMMETRY)
from dm_svd_dci.pipeline_state_averaged import (  # noqa: E402
    run_state_averaged_dci,
)

H2 = dict(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', n_active=2,
          n_active_elec=(1, 1), n_core=0, n_occ=1, sa_states=2,
          p_blocks=[1, 2], svd_eps=1e-10, outer_max_iter=12, verbose=False)

# At svd_eps=1e-10 nothing is truncated, so a correct run is exact.
EXACT_TOL_MH = 1e-9
# Seeds that carry no weight in the n_A=0 block, which holds the double
# excitation, when block completion is switched off.
TRAPPED_SEEDS = ('cis', 'perturbed')
TRAP_ERROR_MH = 20.5


def _check(condition, message):
    if not condition:
        raise AssertionError(message)
    print(f'  ok: {message}', flush=True)


def _run(seed, complete):
    return run_state_averaged_dci(seed=seed, seed_complete_blocks=complete, **H2)


def test_completion_makes_every_seed_reach_the_same_fixed_point():
    print('H1 on H2: all seeds reach the same fixed point', flush=True)
    for seed in PLAIN_SEEDS:
        result = _run(seed, True)
        error = float(np.max(np.abs(result['errors_mH'])))
        support = result['seed_provenance']['block_support']
        _check(not support['has_empty_block'],
               f'seed {seed!r} has no empty block after completion')
        _check(error < EXACT_TOL_MH,
               f'seed {seed!r} is exact to {error:.3e} mH')


def test_trap_is_still_reproducible_without_completion():
    print('the trap remains reproducible when completion is disabled', flush=True)
    for seed in TRAPPED_SEEDS:
        result = _run(seed, False)
        error = float(np.max(np.abs(result['errors_mH'])))
        support = result['seed_provenance']['block_support']
        _check(support['empty_blocks'] == [0],
               f'seed {seed!r} leaves block 0 empty without completion')
        _check(abs(error - TRAP_ERROR_MH) < 0.1,
               f'seed {seed!r} is trapped at {error:.3f} mH')
        _check(bool(result['converged']),
               f'seed {seed!r} reports convergence while being wrong, '
               'which is what makes the trap dangerous')


def test_empty_block_predicts_failure():
    print('empty-block diagnostic predicts failure exactly', flush=True)
    for seed in PLAIN_SEEDS:
        for complete in (False, True):
            if seed == 'exact' and not complete:
                continue
            result = _run(seed, complete)
            empty = result['seed_provenance']['block_support']['has_empty_block']
            failed = float(np.max(np.abs(result['errors_mH']))) > EXACT_TOL_MH
            _check(empty == failed,
                   f'seed {seed!r} complete={complete}: '
                   f'empty_block={empty} matches failure={failed}')


def main():
    print('Seed block support regression checks', flush=True)
    test_completion_makes_every_seed_reach_the_same_fixed_point()
    test_trap_is_still_reproducible_without_completion()
    test_empty_block_predicts_failure()
    print('\nSeed block support: PASS', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
