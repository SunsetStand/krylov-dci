"""Stable regression for the H_AB RDM/Jordan-Wigner implementation.

This check intentionally calls the maintained small H2O/STO-3G comparison in
``dm_svd_embedding``. The exploratory scripts under
``scripts/diagnostics/hab_rdm`` remain useful for investigation, but are not
treated as tests because most report intermediate matrices without assertions.
"""

from __future__ import annotations

import numpy as np

from dm_svd_embedding.embedded_hamiltonian import test_hemb_rdm_vs_sigma


EXPECTED_LOWEST_EIGENVALUES = np.array(
    [-12.816468, -12.313281, -12.234426, -12.197997, -12.120189]
)


def main() -> None:
    """Compare RDM and sigma-vector H_emb constructions and fixed baseline."""
    h_reference, h_rdm, max_matrix_diff, eigenvalue_diffs = (
        test_hemb_rdm_vs_sigma()
    )

    assert max_matrix_diff < 1.0e-10, (
        f"H_emb RDM/sigma mismatch: {max_matrix_diff:.6e}"
    )
    assert float(np.max(np.abs(eigenvalue_diffs))) < 1.0e-10, (
        "H_emb eigenvalues changed between the RDM and sigma constructions"
    )
    np.testing.assert_allclose(h_rdm, h_rdm.T.conj(), atol=1.0e-12, rtol=0.0)
    np.testing.assert_allclose(h_reference, h_reference.T.conj(), atol=1.0e-12, rtol=0.0)

    lowest = np.linalg.eigvalsh(h_rdm)[: EXPECTED_LOWEST_EIGENVALUES.size]
    np.testing.assert_allclose(
        lowest,
        EXPECTED_LOWEST_EIGENVALUES,
        atol=1.0e-6,
        rtol=0.0,
        err_msg="H2O/STO-3G H_emb baseline changed",
    )

    print(f"H_AB_RDM_MAX_DIFF={max_matrix_diff:.6e}", flush=True)
    print(
        "H_EMB_LOWEST=" + ",".join(f"{value:.9f}" for value in lowest),
        flush=True,
    )
    print("HAB_RDM_REGRESSION_PASS", flush=True)


if __name__ == "__main__":
    main()
