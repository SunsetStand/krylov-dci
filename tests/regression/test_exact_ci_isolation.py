#!/usr/bin/env python3
"""H5: the production path must never read exact CASCI or FCI coefficients.

The test arms tripwires on the exact CI kernels, then runs the full
state-averaged pipeline from a non-exact seed.  If any part of the initializer,
root selector, Schmidt builder or solver reaches for exact CI, the tripwire
fires and the test fails.

A second check runs the same pipeline with seed='exact' and asserts the
tripwire DOES fire, so that a silently broken tripwire cannot make the first
check pass vacuously.

See docs/theory/iterative_ci_feasibility_protocol.md, hypothesis H5.

Run directly:  python tests/regression/test_exact_ci_isolation.py
"""

import contextlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from dm_svd_dci.initializers import (  # noqa: E402
    NON_EXACT_SEEDS,
    build_initial_states,
)
from dm_svd_dci.pipeline_state_averaged import (  # noqa: E402
    run_state_averaged_dci,
)
from dm_svd_dci.pipeline_v2 import setup_system  # noqa: E402

H2 = {
    'atom': 'H 0 0 0; H 0 0 0.74',
    'basis': 'sto-3g',
    'n_active': 2,
    'n_active_elec': (1, 1),
    'n_core': 0,
    'n_occ': 1,
    'sa_states': 2,
    'p_blocks': [1, 2],
}


class ExactCIAccessed(AssertionError):
    """Raised by a tripwire when an exact CI kernel is called."""


@contextlib.contextmanager
def exact_ci_tripwires():
    """Make every exact CI kernel raise while inside this block."""
    from pyscf import mcscf
    from pyscf.fci import direct_spin0, direct_spin1, direct_spin1_symm

    targets = [
        (mcscf.casci.CASCI, 'kernel'),
        (direct_spin1.FCISolver, 'kernel'),
        (direct_spin0.FCISolver, 'kernel'),
        (direct_spin1_symm.FCISolver, 'kernel'),
    ]
    saved = []
    for owner, name in targets:
        if hasattr(owner, name):
            saved.append((owner, name, getattr(owner, name)))

    def tripwire(self, *args, **kwargs):
        raise ExactCIAccessed(
            f'exact CI kernel {type(self).__name__}.kernel was called')

    try:
        for owner, name, _ in saved:
            setattr(owner, name, tripwire)
        yield
    finally:
        for owner, name, original in saved:
            setattr(owner, name, original)


def _check(condition, message):
    if not condition:
        raise AssertionError(message)
    print(f'  ok: {message}', flush=True)


def test_tripwire_actually_fires():
    """Guard against a vacuous pass: the tripwire must detect exact CI."""
    print('tripwire self-check', flush=True)
    fired = False
    try:
        with exact_ci_tripwires():
            setup_system(
                atom=H2['atom'], basis=H2['basis'],
                n_active=H2['n_active'], n_active_elec=H2['n_active_elec'],
                n_core=H2['n_core'], verbose=False, solve_exact=True)
    except ExactCIAccessed:
        fired = True
    _check(fired, 'tripwire fires when an exact CI kernel is called')


def test_setup_system_skips_exact_solve():
    print('setup_system with solve_exact=False', flush=True)
    with exact_ci_tripwires():
        data = setup_system(
            atom=H2['atom'], basis=H2['basis'],
            n_active=H2['n_active'], n_active_elec=H2['n_active_elec'],
            n_core=H2['n_core'], verbose=False, solve_exact=False)
    _check(data['E_fci'] is None, 'E_fci is None when the solve is skipped')
    _check(data['ci_flat'] is None, 'ci_flat is None when the solve is skipped')
    _check(data['h1eff'] is not None and data['h2eff'] is not None,
           'active-space integrals are still produced')


def test_every_non_exact_seed_is_isolated():
    print('non-exact seeds touch no exact CI kernel', flush=True)
    with exact_ci_tripwires():
        data = setup_system(
            atom=H2['atom'], basis=H2['basis'],
            n_active=H2['n_active'], n_active_elec=H2['n_active_elec'],
            n_core=H2['n_core'], verbose=False, solve_exact=False)
        for seed in NON_EXACT_SEEDS:
            vectors, provenance = build_initial_states(
                data, H2['sa_states'], seed=seed, verbose=False)
            _check(len(vectors) == H2['sa_states'],
                   f"seed {seed!r} produced {H2['sa_states']} states")
            _check(provenance['reads_exact_ci'] is False,
                   f'seed {seed!r} is declared exact-CI free')


def test_full_pipeline_from_non_exact_seed():
    print('full state-averaged pipeline from a non-exact seed', flush=True)
    with exact_ci_tripwires():
        result = run_state_averaged_dci(
            atom=H2['atom'], basis=H2['basis'],
            n_active=H2['n_active'], n_active_elec=H2['n_active_elec'],
            n_core=H2['n_core'], n_occ=H2['n_occ'],
            sa_states=H2['sa_states'], p_blocks=H2['p_blocks'],
            svd_eps=1e-10, seed='cis', compute_reference=False,
            outer_max_iter=4, verbose=False)
    _check(result['seed_provenance']['reads_exact_ci'] is False,
           'pipeline recorded an exact-CI-free seed')
    _check(result['reference_energies'] is None,
           'no reference energies were computed')
    _check(len(result['energies']) == H2['sa_states'],
           f"pipeline returned {H2['sa_states']} energies")


def main():
    print('Exact-CI isolation checks (H5)', flush=True)
    test_tripwire_actually_fires()
    test_setup_system_skips_exact_solve()
    test_every_non_exact_seed_is_isolated()
    test_full_pipeline_from_non_exact_seed()
    print('\nH5 isolation: PASS', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
