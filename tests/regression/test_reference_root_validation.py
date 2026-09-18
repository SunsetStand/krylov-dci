"""The reference evaluator must not return an unvalidated root subset.

A plain ``nroots = n_states`` CASCI solve on N2 CAS(10e,9o) returns four roots
that all report ``converged = True`` while missing one member of the exactly
degenerate 3Pi_g level, putting the fourth reference energy 27.27 mH too high.
Because ``evaluate_reference_energies`` is what every reported error is measured
against, that silently corrupts the scoring of every multi-state result.

This test pins the defect and the fix together: it asserts that the naive solve
still exhibits the defect, so the test cannot pass vacuously if PySCF changes,
and that the evaluator agrees with the committed reference bundle.
"""
import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from dm_svd_dci.initializers import evaluate_reference_energies

# Committed bundle, checksum ad5b19d6674d54411f704bd3ed4b65aab9741d5373037868fc8fb76e85133d9d
BUNDLE = np.array([-109.0401239716, -108.7408683094,
                   -108.7218810454, -108.7218810454])
FAILURES = []


def _check(condition, message):
    print(f"  {'ok' if condition else 'FAIL'}: {message}")
    if not condition:
        FAILURES.append(message)


def main():
    from pyscf import gto, scf, mcscf

    print("Reference root validation regression")
    mol = gto.M(atom='N 0 0 0; N 0 0 1.098', basis='cc-pVDZ', verbose=0)
    mf = scf.RHF(mol).run()
    sys_data = dict(mf=mf, n_active=9, n_active_elec=(5, 5), n_core=2)

    # 1. The naive solve must still be wrong, or this test proves nothing.
    cas = mcscf.CASCI(mf, 9, 10)
    cas.frozen = 2
    cas.fcisolver.nroots = 4
    cas.kernel()
    naive = np.sort(np.atleast_1d(np.asarray(cas.e_tot, dtype=float)).ravel())
    naive_split = abs(naive[3] - naive[2]) * 1000.0
    _check(naive_split > 1.0,
           f"naive nroots=4 still splits the degenerate level, by "
           f"{naive_split:.4f} mH, so the fix is still needed")

    # 2. The evaluator must reproduce the bundle.
    energies = evaluate_reference_energies(sys_data, 4)
    _check(energies.shape == (4,), f"returns four energies, got {energies.shape}")
    deviation = float(np.max(np.abs(energies - BUNDLE))) * 1000.0
    _check(deviation < 1e-3,
           f"agrees with the committed bundle to {deviation:.2e} mH")

    # 3. The degenerate level must come back degenerate.
    split = abs(energies[3] - energies[2]) * 1000.0
    _check(split < 1e-6,
           f"the 3Pi_g level is returned degenerate, splitting {split:.2e} mH")

    # 4. Asking for fewer states must still be right.
    two = evaluate_reference_energies(sys_data, 2)
    _check(float(np.max(np.abs(two - BUNDLE[:2]))) * 1000.0 < 1e-3,
           "n_states=2 also agrees with the bundle")

    print()
    if FAILURES:
        print(f"Reference root validation: FAIL ({len(FAILURES)})")
        return 1
    print("Reference root validation: PASS")
    return 0


if __name__ == '__main__':
    sys.exit(main())
