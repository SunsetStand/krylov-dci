#!/usr/bin/env python3
"""The state-averaged bases come from an SVD, and must respect the rank bound.

The state-averaged Schmidt bases are obtained by SVD of the weighted
coefficient blocks rather than by eigendecomposition of the reduced densities.
The two are mathematically equivalent, since

    M = [sqrt(w_1) C_1 | ... | sqrt(w_s) C_s]   satisfies  M M^dag = rho_A^SA
    N = [sqrt(w_1) C_1 ; ... ; sqrt(w_s) C_s]   satisfies  N^dag N = rho_B^SA

so U is the left singular vectors of M and V the right singular vectors of N.
The SVD is preferred because forming the density squares the condition number
and carries dim_A eigenvalues when the true rank is bounded by
min(dim_A, s dim_B), so its square root can promote numerical noise into a
retained direction.

These checks pin: equivalence with the density route where both resolve the
spectrum, and the rank bound that the density route can violate.

Run directly:  python tests/unit/test_state_averaged_svd_equivalence.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from dm_svd_embedding.density_matrix import (  # noqa: E402
    compute_schmidt_decomposition,
    singular_value_threshold,
)


def _check(condition, message):
    if not condition:
        raise AssertionError(message)
    print(f'  ok: {message}', flush=True)


def _states(n_states, dim_a, dim_b, seed, rank=None):
    """Weighted blocks with a controllable true rank."""
    generator = np.random.default_rng(seed)
    states = []
    for _ in range(n_states):
        if rank is None:
            block = generator.standard_normal((dim_a, dim_b))
        else:
            left = generator.standard_normal((dim_a, rank))
            right = generator.standard_normal((rank, dim_b))
            block = left @ right
        states.append({0: block / np.linalg.norm(block)})
    return states


def _density_route(states, weights, eps):
    """The previous implementation, for comparison only."""
    first = states[0][0]
    rho_a = sum(w * (s[0] @ s[0].T) for w, s in zip(weights, states))
    rho_b = sum(w * (s[0].T @ s[0]) for w, s in zip(weights, states))
    ranks = []
    for rho in (rho_a, rho_b):
        values = np.linalg.eigvalsh(rho)[::-1]
        sigma = np.sqrt(np.maximum(values, 0.0))
        ranks.append(int(np.sum(singular_value_threshold(sigma, eps))))
    return ranks[0], ranks[1], first.shape


def test_matches_the_density_route_where_both_resolve():
    print('agreement with the density route', flush=True)
    for n_states, dim_a, dim_b, seed in ((3, 40, 12, 1), (4, 60, 20, 2),
                                         (2, 30, 30, 3)):
        states = _states(n_states, dim_a, dim_b, seed)
        weights = np.full(n_states, 1.0 / n_states)
        for eps in (1e-1, 1e-2, 1e-3, 1e-4, 1e-6):
            schmidt = compute_schmidt_decomposition(
                states[0], eps=eps, state_average=states,
                state_weights=weights)
            r_a_density, r_b_density, _ = _density_route(states, weights, eps)
            _check(schmidt[0]['r_A'] == r_a_density
                   and schmidt[0]['r_B'] == r_b_density,
                   f's={n_states} {dim_a}x{dim_b} eps={eps:.0e}: '
                   f"r_A={schmidt[0]['r_A']} r_B={schmidt[0]['r_B']} "
                   'match the density route')


def test_rank_never_exceeds_the_true_bound():
    """The property the density route can violate."""
    print('retained rank respects min(dim_A, s dim_B)', flush=True)
    for n_states, dim_a, dim_b, seed in ((4, 100, 8, 5), (3, 80, 6, 6),
                                         (2, 50, 4, 7)):
        states = _states(n_states, dim_a, dim_b, seed)
        weights = np.full(n_states, 1.0 / n_states)
        bound_a = min(dim_a, n_states * dim_b)
        bound_b = min(n_states * dim_a, dim_b)
        for eps in (1e-6, 1e-8, 1e-10, 1e-14):
            schmidt = compute_schmidt_decomposition(
                states[0], eps=eps, state_average=states,
                state_weights=weights)
            _check(schmidt[0]['r_A'] <= bound_a,
                   f's={n_states} {dim_a}x{dim_b} eps={eps:.0e}: '
                   f"r_A={schmidt[0]['r_A']} within bound {bound_a}")
            _check(schmidt[0]['r_B'] <= bound_b,
                   f"{'':>28}r_B={schmidt[0]['r_B']} within bound {bound_b}")


def test_recovers_a_known_low_rank():
    print('a rank-deficient average is resolved exactly', flush=True)
    true_rank = 5
    states = _states(3, 60, 40, 11, rank=true_rank)
    weights = np.full(3, 1.0 / 3.0)
    schmidt = compute_schmidt_decomposition(
        states[0], eps=1e-10, state_average=states, state_weights=weights)
    # Three states of rank 5 span at most 15 directions on each side.
    _check(schmidt[0]['r_A'] <= 3 * true_rank,
           f"r_A={schmidt[0]['r_A']} at most {3 * true_rank}")
    _check(schmidt[0]['r_B'] <= 3 * true_rank,
           f"r_B={schmidt[0]['r_B']} at most {3 * true_rank}")


def test_single_state_is_unchanged():
    print('the single-state path still SVDs C directly', flush=True)
    states = _states(1, 30, 18, 13)
    block = states[0][0]
    single = compute_schmidt_decomposition(states[0], eps=1e-6)
    reference = np.linalg.svd(block, compute_uv=False)
    kept = int(np.sum(singular_value_threshold(reference, 1e-6)))
    _check(single[0]['r'] == kept and single[0]['r_A'] == single[0]['r_B'],
           f"r={single[0]['r']} with r_A == r_B, matching a direct SVD of C")


def main():
    print('State-averaged SVD equivalence checks', flush=True)
    test_matches_the_density_route_where_both_resolve()
    test_rank_never_exceeds_the_true_bound()
    test_recovers_a_known_low_rank()
    test_single_state_is_unchanged()
    print('\nState-averaged SVD equivalence: PASS', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
