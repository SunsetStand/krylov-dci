#!/usr/bin/env python3
"""Unit checks for the instrumentation the feasibility protocol requires.

Covers the four quantities the protocol needs that the solver previously
computed and discarded, or never computed at all:

  * the Schmidt projector distance between consecutive outer iterations, which
    did not exist;
  * the rank_mode control that forces a symmetric allocation, for H6;
  * the apply_dressing control that disables the residual dressing, for H4.

Run directly:  python tests/unit/test_feasibility_instrumentation.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from dm_svd_dci.state_averaged_solver import (  # noqa: E402
    schmidt_projector_distance,
)
from dm_svd_dci.wave_operator import (  # noqa: E402
    solve_state_averaged_wave_operator,
)
from dm_svd_embedding.density_matrix import (  # noqa: E402
    compute_schmidt_decomposition,
)


def _check(condition, message):
    if not condition:
        raise AssertionError(message)
    print(f'  ok: {message}', flush=True)


def _schmidt(rank_a, rank_b, dim_a=4, dim_b=3, offset=0):
    """A minimal schmidt_data-like block with controllable ranks."""
    rng = np.random.default_rng(17 + offset)
    u = np.linalg.qr(rng.standard_normal((dim_a, dim_a)))[0][:, :rank_a]
    v = np.linalg.qr(rng.standard_normal((dim_b, dim_b)))[0][:, :rank_b]
    return {0: {'U': u, 'V': v, 'dim_A': dim_a, 'dim_B': dim_b,
                'r_A': rank_a, 'r_B': rank_b}}


def test_projector_distance_basics():
    print('Schmidt projector distance', flush=True)
    current = _schmidt(2, 2)
    _check(schmidt_projector_distance(None, current) is None,
           'returns None on the first outer iteration')

    same = schmidt_projector_distance(current, current)
    _check(same['total'] < 1e-12,
           f"identical bases give distance {same['total']:.2e}")

    deleted = _schmidt(0, 0)
    dropped = schmidt_projector_distance(current, deleted)
    _check(dropped['total'] > 1.0,
           f"deleting a block gives a finite jump of {dropped['total']:.3f}, "
           'so irreversible rank loss is visible')

    rotated = _schmidt(2, 2, offset=5)
    moved = schmidt_projector_distance(current, rotated)
    _check(moved['total'] > 1e-6,
           f"a rotated subspace gives distance {moved['total']:.3f}")
    _check(set(moved['per_block']) == {0}, 'per-block detail is reported')


def test_projector_distance_is_rank_invariant_in_shape():
    print('projector distance survives a rank change', flush=True)
    before = _schmidt(3, 2)
    after = _schmidt(1, 2, offset=5)
    result = schmidt_projector_distance(before, after)
    _check(np.isfinite(result['total']),
           'a change from rank 3 to rank 1 gives a finite distance')
    _check(result['total_A'] > 0.0, 'the A-side change is detected')


def test_rank_mode_control():
    print('rank_mode control for H6', flush=True)
    rng = np.random.default_rng(3)
    states = [{0: rng.standard_normal((6, 4))} for _ in range(3)]
    for state in states:
        state[0] /= np.linalg.norm(state[0])

    rect = compute_schmidt_decomposition(
        states[0], eps=1e-6, state_average=states)
    sym = compute_schmidt_decomposition(
        states[0], eps=1e-6, state_average=states, rank_mode='symmetric')

    _check(sym[0]['r_A'] == sym[0]['r_B'],
           f"symmetric mode gives r_A = r_B = {sym[0]['r_A']}")
    _check(sym[0]['r_A'] == min(rect[0]['r_A'], rect[0]['r_B']),
           'symmetric rank is the minimum of the rectangular pair')
    _check(sym[0]['U'].shape[1] == sym[0]['r_A']
           and sym[0]['V'].shape[1] == sym[0]['r_B'],
           'symmetric bases are truncated consistently with their ranks')
    _check(rect[0]['rank_mode'] == 'rectangular'
           and sym[0]['rank_mode'] == 'symmetric',
           'rank_mode is recorded in the result')

    try:
        compute_schmidt_decomposition(states[0], eps=1e-6,
                                      state_average=states, rank_mode='bogus')
    except ValueError:
        _check(True, 'an unknown rank_mode is rejected')
    else:
        raise AssertionError('an unknown rank_mode was accepted')


def test_apply_dressing_control():
    print('apply_dressing control for H4', flush=True)
    rng = np.random.default_rng(11)
    p_dim, q_dim = 4, 5
    h_pp = rng.standard_normal((p_dim, p_dim))
    h_pp = 0.5 * (h_pp + h_pp.T)
    h_pq = 0.2 * rng.standard_normal((p_dim, q_dim))
    h_qq = rng.standard_normal((q_dim, q_dim))
    h_qq = 0.5 * (h_qq + h_qq.T) + np.diag(np.arange(q_dim) + 3.0)

    kwargs = dict(H_PP=h_pp, H_PQ={0: h_pq}, H_QQ_blocks={(0, 0): h_qq},
                  D_by_n={0: np.diag(h_qq)}, n_states=2, max_iter=30,
                  verbose=False)
    dressed = solve_state_averaged_wave_operator(**kwargs)
    bare = solve_state_averaged_wave_operator(apply_dressing=False, **kwargs)

    _check(bare['apply_dressing'] is False,
           'the undressed run records apply_dressing=False')
    _check(float(np.linalg.norm(bare['omega'])) < 1e-14,
           f"Omega stays zero without dressing, norm "
           f"{float(np.linalg.norm(bare['omega'])):.2e}")
    bare_energies = np.sort(np.asarray(bare['energies']))
    reference = np.sort(np.linalg.eigvalsh(h_pp))[:len(bare_energies)]
    _check(np.max(np.abs(bare_energies - reference)) < 1e-10,
           'without dressing the Ritz problem reduces to diagonalizing H_PP')
    _check(float(np.linalg.norm(dressed['omega'])) > 1e-8,
           'the dressed run does build a non-zero Omega')


def main():
    print('Feasibility instrumentation unit checks', flush=True)
    test_projector_distance_basics()
    test_projector_distance_is_rank_invariant_in_shape()
    test_rank_mode_control()
    test_apply_dressing_control()
    print('\nInstrumentation: PASS', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
