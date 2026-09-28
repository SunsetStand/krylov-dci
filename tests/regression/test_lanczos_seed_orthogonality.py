"""The Lanczos seed must stay variational at any number of steps.

With single-pass modified Gram-Schmidt the block Krylov basis lost
orthogonality geometrically on N2 CAS(10e,9o): max|K^T K - I| was 1.9e-5 at 10
steps, 3.8e-3 at 12 and 0.63 at 14. The Rayleigh-Ritz is a standard
eigenproblem, so it stopped being variational: at 14 steps its lowest root lay
17.5 Ha below the exact ground state, the seed returned states 2.2 to 2.8 Ha too
high, and the outer loop diverged to 321 mH on the ground state (Slurm job 20910,
task 2). Two-pass MGS plus a verified-orthonormality guard fixes it.

This test asserts the fix at 14 and 20 steps and pins the 10-step seed to its
pre-fix values, so previously recorded 10-step results stay valid.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from dm_svd_dci.initializers import _ActiveSpace, build_initial_states  # noqa: E402
from dm_svd_dci.pipeline_v2 import setup_system                         # noqa: E402
from dm_svd_embedding.occ_virt_partition import setup_partition          # noqa: E402

BUNDLE = np.array([-109.0401239716, -108.7408683094,
                   -108.7218810454, -108.7218810454])
# Rayleigh quotient minus bundle, mH, measured with the pre-fix code at 10 steps.
TEN_STEP_PREFIX = np.array([0.000, 21.434, 4.457, 5.135])
FAILURES = []


def _check(condition, message):
    print(f"  {'ok' if condition else 'FAIL'}: {message}")
    if not condition:
        FAILURES.append(message)


def _rayleigh_minus_bundle(sd, space, partition, steps):
    states, _ = build_initial_states(sd, 4, seed='lanczos_symm',
                                     partition=partition,
                                     lanczos_steps=steps, verbose=False)
    vectors = [np.asarray(v).ravel() for v in states]
    quotients = sorted(float(v @ space.sigma(v) / (v @ v)) + sd['ecore']
                       for v in vectors)
    gram = np.array([[a @ b for b in vectors] for a in vectors])
    return (np.array(quotients) - BUNDLE) * 1000.0, float(
        np.max(np.abs(gram - np.eye(4))))


def main():
    print("Lanczos seed orthogonality regression")
    sd = setup_system(atom='N 0 0 0; N 0 0 1.098', basis='cc-pVDZ',
                      n_active=9, n_active_elec=(5, 5), n_core=2, nroots=1,
                      solve_exact=False, verbose=False, symmetry='D2h')
    space = _ActiveSpace(sd)
    partition, _ = setup_partition(9, 10, 5, ms=0)

    ten, _ = _rayleigh_minus_bundle(sd, space, partition, 10)
    _check(float(np.max(np.abs(ten - TEN_STEP_PREFIX))) < 1e-3,
           f"10-step seed unchanged by the fix: {np.array2string(ten, precision=3)}")

    for steps in (14, 20):
        excess, gram = _rayleigh_minus_bundle(sd, space, partition, steps)
        _check(bool(np.all(excess > -1e-6)),
               f"{steps} steps: no seed state below the exact energy "
               f"(variational), {np.array2string(excess, precision=3)} mH")
        _check(float(excess[0]) < 1e-2,
               f"{steps} steps: ground-state seed within 0.01 mH, "
               f"{excess[0]:.4f} mH (was 2197 mH before the fix at 14)")
        _check(gram < 1e-10,
               f"{steps} steps: returned states orthonormal, {gram:.1e}")

    print()
    if FAILURES:
        print(f"Lanczos seed orthogonality: FAIL ({len(FAILURES)})")
        return 1
    print("Lanczos seed orthogonality: PASS")
    return 0


if __name__ == '__main__':
    sys.exit(main())
