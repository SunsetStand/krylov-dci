"""A seed must carry weight in every irrep whose states are targeted.

A Krylov expansion cannot create weight in an irrep its starting block does not
touch, and the outer map cannot increase a retained rank, so a target state in
an unseeded irrep is unreachable however many outer iterations are run. On N2
CAS(10e,9o) the default ``lanczos`` seed puts 6.8e-26 on the degenerate 3Pi_g
pair, which is the same exclusion Gate B measured at 6.76e-26 for a naive
Davidson guess, and the pair's energies then come out about 29 mH high while the
ground state reaches 1.3 mH.

This pins the defect and the fix together, so it cannot pass vacuously.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from dm_svd_dci.initializers import (InitializerError, build_initial_states,
                                     evaluate_reference_energies,
                                     seed_irrep_weights, _ActiveSpace)
from dm_svd_dci.pipeline_v2 import setup_system
from dm_svd_embedding.occ_virt_partition import setup_partition

SYSTEM = dict(atom='N 0 0 0; N 0 0 1.098', basis='cc-pVDZ', n_active=9,
              n_active_elec=(5, 5), n_core=2, nroots=1, solve_exact=False,
              verbose=False)
BUNDLE = np.array([-109.0401239716, -108.7408683094,
                   -108.7218810454, -108.7218810454])
# D2h irrep ids in PySCF's convention
AG, B1G, B2G, B3G, AU, B1U, B2U, B3U = range(8)
FAILURES = []


def _check(condition, message):
    print(f"  {'ok' if condition else 'FAIL'}: {message}")
    if not condition:
        FAILURES.append(message)


def main():
    print("Seed irrep coverage regression")

    # --- symmetry is a prerequisite, and its absence must be explicit ---
    plain = setup_system(**SYSTEM)
    _check(plain['orbsym'] is None,
           "without symmetry, setup_system reports no orbsym")
    partition, _ = setup_partition(9, 10, 5, ms=0)
    try:
        build_initial_states(plain, 4, seed='lanczos_symm',
                             partition=partition, verbose=False)
        _check(False, "lanczos_symm without symmetry must raise")
    except InitializerError:
        _check(True, "lanczos_symm without symmetry raises InitializerError")

    # --- with symmetry the active orbitals are pure irreps ---
    sym = setup_system(symmetry='D2h', **SYSTEM)
    orbsym = sym['orbsym']
    _check(orbsym is not None and len(orbsym) == 9,
           f"with symmetry, orbsym has 9 entries: {orbsym}")
    _check(sorted(orbsym.tolist()) == sorted([AG, AG, AG, B1U, B1U,
                                              B2U, B3U, B2G, B3G]),
           "the active space holds complete degenerate pi pairs, "
           "B2u/B3u and B2g/B3g")
    _check(abs(sym['mf'].e_tot - plain['mf'].e_tot) < 1e-10,
           f"the RHF energy is unchanged, "
           f"{abs(sym['mf'].e_tot - plain['mf'].e_tot):.2e} Ha")

    # --- the reference must survive symmetry being switched on ---
    energies = evaluate_reference_energies(sym, 4)
    deviation = float(np.max(np.abs(energies - BUNDLE))) * 1000.0
    _check(deviation < 1e-3,
           f"the reference still matches the bundle, {deviation:.2e} mH, so the "
           f"solver was not silently restricted to one wfnsym")

    space = _ActiveSpace(sym)

    # --- the defect: the default seed misses the pi_g irreps ---
    plain_seed, _ = build_initial_states(sym, 4, seed='lanczos',
                                         partition=partition, verbose=False)
    plain_w = seed_irrep_weights(plain_seed, space, orbsym)
    _check(plain_w[B2G] < 1e-12 and plain_w[B3G] < 1e-12,
           f"the default lanczos seed still misses B2g and B3g, at "
           f"{plain_w[B2G]:.1e} and {plain_w[B3G]:.1e}, so the fix is needed")

    # --- the fix: an irrep-complete starting block reaches them ---
    symm_seed, provenance = build_initial_states(
        sym, 4, seed='lanczos_symm', partition=partition,
        lanczos_steps=6, verbose=False)
    _check(provenance.get('irrep_complete_start') is True,
           "lanczos_symm records an irrep-complete starting block")
    symm_w = seed_irrep_weights(symm_seed, space, orbsym)
    _check(symm_w[B2G] > 1e-3 and symm_w[B3G] > 1e-3,
           f"lanczos_symm carries B2g and B3g weight, "
           f"{symm_w[B2G]:.3f} and {symm_w[B3G]:.3f}")
    _check(abs(symm_w[B2G] - symm_w[B3G]) < 1e-6,
           "the degenerate pair is seeded symmetrically, as GOK requires")
    _check(provenance.get('selects_determinants') is False,
           "lanczos_symm still selects no determinants")

    print()
    if FAILURES:
        print(f"Seed irrep coverage: FAIL ({len(FAILURES)})")
        return 1
    print("Seed irrep coverage: PASS")
    return 0


if __name__ == '__main__':
    sys.exit(main())
