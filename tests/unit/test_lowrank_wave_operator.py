#!/usr/bin/env python3
"""Matrix-free low-rank wave operator must match the dense solver exactly.

The dense solver forms the q by q embedded Q-space Hamiltonian.  The
matrix-free one never does: H_QQ enters only through an apply callback, and
Omega is kept as W Omega_tilde with W orthonormal, which is justified by Omega
being numerically rank n_states
(docs/theory/wave_operator_low_rank_structure.md).

These checks pin that the two agree to machine precision, that the number of
H_QQ applications stays bounded as the Q space grows, and that the retained
rank respects max_rank.

Run directly:  python tests/unit/test_lowrank_wave_operator.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from dm_svd_dci.wave_operator import (  # noqa: E402
    solve_state_averaged_wave_operator,
    solve_state_averaged_wave_operator_lowrank,
)

TOL = 1e-10


def _check(condition, message):
    if not condition:
        raise AssertionError(message)
    print(f'  ok: {message}', flush=True)


def _problem(p_dim, q_dim, seed):
    rng = np.random.default_rng(seed)
    h_pp = rng.standard_normal((p_dim, p_dim))
    h_pp = 0.5 * (h_pp + h_pp.T)
    h_pq = 0.15 * rng.standard_normal((p_dim, q_dim))
    off = 0.05 * rng.standard_normal((q_dim, q_dim))
    h_qq = 0.5 * (off + off.T) + np.diag(np.arange(q_dim) + 5.0)
    return h_pp, h_pq, h_qq


def _solve_both(p_dim, q_dim, n_states, seed, max_rank=None):
    h_pp, h_pq, h_qq = _problem(p_dim, q_dim, seed)
    common = dict(n_states=n_states, max_iter=200, damping=0.5,
                  residual_tol=1e-9, energy_tol=1e-10, verbose=False)
    dense = solve_state_averaged_wave_operator(
        H_PP=h_pp, H_PQ={0: h_pq}, H_QQ_blocks={(0, 0): h_qq},
        D_by_n={0: np.diag(h_qq)}, **common)
    counter = {'n': 0}

    def apply(vectors):
        counter['n'] += vectors.shape[1]
        return h_qq @ vectors

    free = solve_state_averaged_wave_operator_lowrank(
        H_PP=h_pp, H_PQ=h_pq, hqq_apply=apply, diagonal=np.diag(h_qq),
        max_rank=max_rank, **common)
    return dense, free, counter['n']


def test_matches_dense_solver():
    print('matrix-free reproduces the dense solver', flush=True)
    for p_dim, q_dim, n_states, seed in ((12, 40, 3, 1), (20, 120, 4, 2),
                                         (30, 300, 5, 3)):
        dense, free, _ = _solve_both(p_dim, q_dim, n_states, seed)
        error = float(np.max(np.abs(
            np.sort(dense['energies']) - np.sort(free['energies']))))
        _check(error < TOL,
               f'P={p_dim} Q={q_dim} s={n_states}: energies agree to {error:.1e}')
        _check(bool(free['converged']),
               f'P={p_dim} Q={q_dim}: matrix-free run converged')
        _check(free['n_iter'] <= dense['n_iter'] + 2,
               f'P={p_dim} Q={q_dim}: {free["n_iter"]} iterations against '
               f'{dense["n_iter"]} for the dense solver')


def test_applications_do_not_scale_with_q():
    print('H_QQ applications stay bounded as Q grows', flush=True)
    counts = []
    for q_dim in (300, 800, 2000):
        _, free, applications = _solve_both(40, q_dim, 4, 5)
        counts.append((q_dim, applications))
        _check(applications < q_dim,
               f'Q={q_dim}: {applications} applications, fewer than Q')
    growth = counts[-1][1] / max(counts[0][1], 1)
    span = counts[-1][0] / counts[0][0]
    _check(growth < 0.25 * span,
           f'applications grew {growth:.2f}x while Q grew {span:.1f}x, '
           'so the cost is not proportional to the Q dimension')


def test_max_rank_is_respected():
    print('max_rank caps the retained rank', flush=True)
    for cap in (8, 12):
        _, free, _ = _solve_both(30, 300, 4, 7, max_rank=cap)
        _check(free['rank'] <= cap,
               f'max_rank={cap}: retained rank {free["rank"]}')


def test_truncation_is_a_saving_not_a_crutch():
    """Truncation must buy cost, and must not be needed for correctness.

    An earlier version of the solver appeared to break down without truncation.
    That was a loss of orthonormality in the factorization rather than a
    property of the method, so this check pins both halves: the untruncated run
    is correct, and the truncated one reaches the same answer more cheaply.
    """
    print('truncation buys cost, and is not needed for correctness', flush=True)
    h_pp, h_pq, h_qq = _problem(50, 2000, 13)
    common = dict(n_states=6, max_iter=300, damping=0.5,
                  residual_tol=1e-9, energy_tol=1e-10, verbose=False)
    dense = solve_state_averaged_wave_operator(
        H_PP=h_pp, H_PQ={0: h_pq}, H_QQ_blocks={(0, 0): h_qq},
        D_by_n={0: np.diag(h_qq)}, **common)
    kept = solve_state_averaged_wave_operator_lowrank(
        H_PP=h_pp, H_PQ=h_pq, hqq_apply=lambda v: h_qq @ v,
        diagonal=np.diag(h_qq), rank_tol=1e-13, **common)
    none = solve_state_averaged_wave_operator_lowrank(
        H_PP=h_pp, H_PQ=h_pq, hqq_apply=lambda v: h_qq @ v,
        diagonal=np.diag(h_qq), rank_tol=0.0, **common)

    def error(result):
        return float(np.max(np.abs(
            np.sort(result['energies']) - np.sort(dense['energies']))))

    _check(error(kept) < TOL,
           f'with truncation the energies agree to {error(kept):.1e}')
    _check(error(none) < TOL,
           f'without truncation the energies also agree, to {error(none):.1e}, '
           'so truncation is not propping up correctness')
    _check(kept['hqq_applications'] < none['hqq_applications'],
           f"truncation is cheaper than not truncating, "
           f"{kept['hqq_applications']} applications against "
           f"{none['hqq_applications']}")
    _check(kept['rank'] < none['rank'],
           f"and it retains a smaller rank, {kept['rank']} against "
           f"{none['rank']}")
    # The saving that matters is against the dense path, which needs the whole
    # q by q matrix, i.e. q columns.  Truncation against no truncation is a
    # second-order effect by comparison.
    _check(kept['hqq_applications'] < 0.25 * 2000,
           f"and both are far below the dense cost of 2000 columns: "
           f"{kept['hqq_applications']} applications, "
           f"{100 * kept['hqq_applications'] / 2000:.1f} percent")


def test_factored_form_reconstructs_omega():
    print('the factored form reproduces Omega', flush=True)
    _, free, _ = _solve_both(20, 120, 4, 9)
    rebuilt = free['omega_basis'] @ free['omega_coefficients']
    error = float(np.max(np.abs(rebuilt - free['omega'])))
    _check(error < 1e-12, f'W Omega_tilde equals Omega to {error:.1e}')
    basis = free['omega_basis']
    overlap = basis.T @ basis
    deviation = float(np.max(np.abs(overlap - np.eye(overlap.shape[0]))))
    _check(deviation < 1e-10, f'the basis is orthonormal to {deviation:.1e}')


def main():
    print('Low-rank matrix-free wave operator checks', flush=True)
    test_matches_dense_solver()
    test_applications_do_not_scale_with_q()
    test_max_rank_is_respected()
    test_truncation_is_a_saving_not_a_crutch()
    test_factored_form_reconstructs_omega()
    print('\nLow-rank wave operator: PASS', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
