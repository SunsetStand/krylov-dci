#!/usr/bin/env python3
"""Pure-matrix tests for state-averaged wave-operator downfolding."""

import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from dm_svd_dci.state_averaged_solver import (
    assemble_embedded_state_coefficients,
    orthonormalize_state_blocks,
    schmidt_product_coefficients_to_blocks,
    solve_state_averaged_schmidt,
)
from dm_svd_dci.schmidt_partition import partition_schmidt_basis
from dm_svd_dci.wave_operator import (
    assemble_qspace_hamiltonian,
    solve_state_averaged_wave_operator,
)
from dm_svd_embedding.density_matrix import (
    compute_schmidt_decomposition,
    normalize_state_weights,
)


def test_weighted_state_average():
    state_0 = {1: np.array([[1.0, 0.0], [0.0, 0.0]])}
    state_1 = {1: np.array([[0.0, 0.0], [0.0, 1.0]])}

    equal = compute_schmidt_decomposition(
        state_0, eps=0.5, state_average=[state_0, state_1])
    weighted = compute_schmidt_decomposition(
        state_0, eps=0.5, state_average=[state_0, state_1],
        state_weights=np.array([0.8, 0.2]))

    assert equal[1]['r'] == 2
    assert weighted[1]['r'] == 1
    assert np.allclose(
        weighted[1]['rho_A_state_averaged'], np.diag([0.8, 0.2]))
    assert np.allclose(np.abs(weighted[1]['U'][:, 0]), [1.0, 0.0])
    assert np.allclose(normalize_state_weights(2, [2.0, 1.0]), [2 / 3, 1 / 3])

    for bad_weights in ([1.0], [-1.0, 2.0], [0.0, 0.0]):
        try:
            normalize_state_weights(2, bad_weights)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid weights accepted: {bad_weights}")


def test_block_assembly():
    H_PQ = {
        2: np.array([[1.0], [2.0]]),
        4: np.array([[3.0, 4.0], [5.0, 6.0]]),
    }
    H_QQ_blocks = {
        (2, 2): np.array([[7.0]]),
        (2, 4): np.array([[0.1, 0.2]]),
        (4, 4): np.array([[8.0, 0.3], [0.3, 9.0]]),
    }
    assembled = assemble_qspace_hamiltonian(
        H_PQ, H_QQ_blocks,
        {2: np.array([7.0]), 4: np.array([8.0, 9.0])})
    assert assembled['q_labels'] == [2, 4]
    assert np.allclose(
        assembled['H_PQ'], np.array([[1.0, 3.0, 4.0], [2.0, 5.0, 6.0]]))
    assert np.allclose(
        assembled['H_QQ'],
        np.array([[7.0, 0.1, 0.2], [0.1, 8.0, 0.3], [0.2, 0.3, 9.0]]))


def test_wave_operator_recovers_dense_roots():
    rng = np.random.default_rng(7)
    raw = rng.normal(size=(5, 5))
    full_hamiltonian = 0.5 * (raw + raw.T)
    full_hamiltonian[:2, :2] -= 4.0 * np.eye(2)
    full_hamiltonian[2:, 2:] += 4.0 * np.eye(3)

    result = solve_state_averaged_wave_operator(
        full_hamiltonian[:2, :2],
        {3: full_hamiltonian[:2, 2:]},
        {(3, 3): full_hamiltonian[2:, 2:]},
        {3: np.diag(full_hamiltonian[2:, 2:])},
        n_states=2,
        state_weights=np.array([0.75, 0.25]),
        damping=0.7,
        residual_tol=1e-10,
        energy_tol=1e-11,
        max_iter=200,
        verbose=False)

    exact = np.linalg.eigvalsh(full_hamiltonian)[:2]
    assert result['converged']
    assert np.allclose(result['energies'], exact, atol=2e-10)
    assert result['residuals']['weighted_rms'] < 1.1e-10
    assert np.allclose(
        result['full_coefficients'].T @ result['full_coefficients'],
        np.eye(2), atol=1e-12)


def test_schmidt_product_reconstruction_and_index_map():
    schmidt = {
        0: {
            'r': 1,
            'U': np.array([[1.0]]),
            'V': np.array([[1.0]]),
        },
        1: {
            'r': 2,
            'U': np.eye(2),
            'V': np.eye(2),
        },
    }
    coefficients = np.array([
        [0.1, 0.2],
        [1.0, 0.0],
        [2.0, 0.0],
        [3.0, 0.0],
        [4.0, 1.0],
    ])
    blocks = schmidt_product_coefficients_to_blocks(coefficients, schmidt)
    assert np.allclose(blocks[0][0], [[0.1]])
    assert np.allclose(blocks[0][1], [[1.0, 2.0], [3.0, 4.0]])
    assert np.allclose(blocks[1][1], [[0.0, 0.0], [0.0, 1.0]])

    wave_result = {
        'model_coefficients': np.array([[1.0, 2.0], [3.0, 4.0]]),
        'q_coefficients': np.array([[5.0, 6.0]]),
        'q_slices': {0: slice(0, 1)},
    }
    part_info = {
        'total_dim': 3,
        'p_indices': np.array([1, 2]),
        'q_indices': np.array([0]),
    }
    q_partition = {'q_blocks': {0: {'indices': np.array([0])}}}
    embedded = assemble_embedded_state_coefficients(
        wave_result, part_info, q_partition)
    assert np.allclose(embedded, [[5.0, 6.0], [1.0, 2.0], [3.0, 4.0]])


def test_rectangular_state_averaged_schmidt_loop():
    # Physical ordering: one scalar n=0 block followed by a 2x2 n=1 block.
    physical_h = np.diag([2.0, -2.0, -1.0, 0.0, 1.0])
    physical_h[0, 1] = physical_h[1, 0] = 0.3
    physical_h[0, 2] = physical_h[2, 0] = 0.2
    physical_h[1, 2] = physical_h[2, 1] = 0.1

    state_0 = {
        0: np.array([[0.1]]),
        1: np.array([[np.sqrt(0.99), 0.0], [0.0, 0.0]]),
    }
    state_1 = {
        0: np.array([[0.12]]),
        1: np.array([[0.0, 0.0], [0.0, np.sqrt(1.0 - 0.12 ** 2)]]),
    }
    initial = orthonormalize_state_blocks([state_0, state_1])

    def build_problem(schmidt):
        labels = sorted(schmidt)
        basis_columns = []
        for label in labels:
            data = schmidt[label]
            r_A = data.get('r_A', data['r'])
            r_B = data.get('r_B', data['r'])
            for alpha in range(r_A):
                for beta in range(r_B):
                    physical_vector = []
                    for other in labels:
                        if other == label:
                            product = np.outer(
                                data['U'][:, alpha], data['V'][:, beta])
                            physical_vector.extend(product.reshape(-1))
                        else:
                            other_data = schmidt[other]
                            physical_vector.extend(np.zeros(
                                other_data['dim_A'] * other_data['dim_B']))
                    basis_columns.append(physical_vector)

        basis = np.asarray(basis_columns).T
        embedded_h = basis.T @ physical_h @ basis
        part = partition_schmidt_basis(schmidt, p_blocks=[1])
        p_idx, q_idx = part['p_indices'], part['q_indices']
        H_QQ = embedded_h[np.ix_(q_idx, q_idx)]
        return {
            'H_PP': embedded_h[np.ix_(p_idx, p_idx)],
            'H_PQ': {0: embedded_h[np.ix_(p_idx, q_idx)]},
            'H_QQ_blocks': {(0, 0): H_QQ},
            'D_by_n': {0: np.diag(H_QQ)},
            'part_info': part,
            'q_partition': {
                'q_blocks': {0: {'indices': np.arange(len(q_idx))}}},
        }

    result = solve_state_averaged_schmidt(
        initial, build_problem,
        svd_eps=1e-10,
        state_weights=np.array([0.7, 0.3]),
        density_tol=1e-9,
        energy_tol=1e-10,
        max_outer_iter=8,
        wave_options={
            'damping': 0.7,
            'residual_tol': 1e-11,
            'energy_tol': 1e-12,
            'max_iter': 200,
        },
        verbose=False)

    assert result['converged']
    assert result['n_outer_iter'] == 2
    assert np.allclose(
        result['energies'], np.linalg.eigvalsh(physical_h)[:2], atol=2e-10)
    final_rank = result['history'][-1]['schmidt_ranks'][1]
    assert final_rank == {'r_A': 1, 'r_B': 2}


def main():
    test_weighted_state_average()
    test_block_assembly()
    test_wave_operator_recovers_dense_roots()
    test_schmidt_product_reconstruction_and_index_map()
    test_rectangular_state_averaged_schmidt_loop()
    print("state-averaged wave-operator unit tests: PASS")


if __name__ == '__main__':
    main()
