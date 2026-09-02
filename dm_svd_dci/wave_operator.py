#!/usr/bin/env python3
"""State-averaged residual-dressed wave-operator solver.

The solver represents a common graph subspace

    X = [I_P, Omega]^T

for all target states.  Rayleigh--Ritz diagonalization in that graph gives
orthonormal dressed states, while the Q-space Schrödinger residual is used to
update ``Omega`` with a diagonal resolvent preconditioner.  Only the selected
state span updates the shared operator, making this a low-rank block method.

This module is deliberately independent of PySCF and the H_AB/RDM builders.
It consumes the existing P/Q Hamiltonian block contract and is therefore
testable with small dense matrices.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.linalg import eigh

from dm_svd_embedding.density_matrix import normalize_state_weights


def assemble_qspace_hamiltonian(
    H_PQ: Dict[int, np.ndarray],
    H_QQ_blocks: Dict[Tuple[int, int], np.ndarray],
    D_by_n: Dict[int, np.ndarray],
) -> Dict:
    """Assemble the active Q blocks in the same order used by ``H_PQ``.

    Missing off-diagonal blocks are interpreted as selection-rule zeros.
    Missing diagonal matrices are reconstructed from ``D_by_n``.  If both
    orientations of an off-diagonal block are supplied, they are averaged to
    remove projection noise without double counting.
    """
    q_labels = sorted(H_PQ.keys())
    q_slices = {}
    offset = 0

    dtypes = [np.asarray(mat).dtype for mat in H_PQ.values()]
    dtypes += [np.asarray(mat).dtype for mat in H_QQ_blocks.values()]
    dtype = np.result_type(*(dtypes or [float]))

    for label in q_labels:
        hpq = np.asarray(H_PQ[label])
        if hpq.ndim != 2:
            raise ValueError(f"H_PQ[{label}] must be a matrix")
        dim = hpq.shape[1]
        if label not in D_by_n:
            raise ValueError(f"missing Q-space diagonal for block {label}")
        diag = np.asarray(D_by_n[label])
        if diag.shape != (dim,):
            raise ValueError(
                f"D_by_n[{label}] has shape {diag.shape}, expected {(dim,)}")
        q_slices[label] = slice(offset, offset + dim)
        offset += dim

    if q_labels:
        p_dims = {np.asarray(H_PQ[label]).shape[0] for label in q_labels}
        if len(p_dims) != 1:
            raise ValueError("all H_PQ blocks must have the same P dimension")
        H_PQ_full = np.concatenate(
            [np.asarray(H_PQ[label], dtype=dtype) for label in q_labels], axis=1)
        diagonal = np.concatenate(
            [np.asarray(D_by_n[label], dtype=dtype) for label in q_labels])
    else:
        H_PQ_full = np.zeros((0, 0), dtype=dtype)
        diagonal = np.zeros(0, dtype=dtype)

    H_QQ = np.zeros((offset, offset), dtype=dtype)
    for i, left in enumerate(q_labels):
        sl_left = q_slices[left]
        dim_left = sl_left.stop - sl_left.start

        diagonal_block = H_QQ_blocks.get((left, left))
        if diagonal_block is None:
            diagonal_block = np.diag(D_by_n[left])
        diagonal_block = np.asarray(diagonal_block, dtype=dtype)
        if diagonal_block.shape != (dim_left, dim_left):
            raise ValueError(
                f"H_QQ_blocks[{left}, {left}] has shape "
                f"{diagonal_block.shape}, expected {(dim_left, dim_left)}")
        H_QQ[sl_left, sl_left] = 0.5 * (
            diagonal_block + diagonal_block.T.conj())

        for right in q_labels[i + 1:]:
            sl_right = q_slices[right]
            expected = (dim_left, sl_right.stop - sl_right.start)
            forward = H_QQ_blocks.get((left, right))
            reverse = H_QQ_blocks.get((right, left))
            if forward is None and reverse is None:
                continue
            if forward is None:
                block = np.asarray(reverse, dtype=dtype).T.conj()
            elif reverse is None:
                block = np.asarray(forward, dtype=dtype)
            else:
                block = 0.5 * (
                    np.asarray(forward, dtype=dtype)
                    + np.asarray(reverse, dtype=dtype).T.conj())
            if block.shape != expected:
                raise ValueError(
                    f"H_QQ block ({left}, {right}) has shape {block.shape}, "
                    f"expected {expected}")
            H_QQ[sl_left, sl_right] = block
            H_QQ[sl_right, sl_left] = block.T.conj()

    return {
        'H_PQ': H_PQ_full,
        'H_QQ': H_QQ,
        'diagonal': diagonal,
        'q_labels': q_labels,
        'q_slices': q_slices,
        'q_dim': offset,
    }


def graph_ritz(
    H_PP: np.ndarray,
    H_PQ: np.ndarray,
    H_QQ: np.ndarray,
    omega: np.ndarray,
    n_states: int,
    metric_floor: float = 1e-12,
) -> Dict:
    """Rayleigh--Ritz solve in the graph of a shared wave operator.

    The generalized problem ``X^H H X c = E X^H X c`` is transformed with
    the symmetric inverse square root of the graph metric.  Returned full
    states are orthonormal up to numerical precision.
    """
    H_PP = np.asarray(H_PP)
    H_PQ = np.asarray(H_PQ)
    H_QQ = np.asarray(H_QQ)
    omega = np.asarray(omega)
    p_dim = H_PP.shape[0]

    if H_PP.shape != (p_dim, p_dim):
        raise ValueError("H_PP must be square")
    if H_PQ.shape != (p_dim, H_QQ.shape[0]):
        raise ValueError("H_PQ and H_QQ dimensions are inconsistent")
    if H_QQ.shape[0] != H_QQ.shape[1]:
        raise ValueError("H_QQ must be square")
    if omega.shape != (H_QQ.shape[0], p_dim):
        raise ValueError("omega must have shape (q_dim, p_dim)")
    if n_states <= 0 or n_states > p_dim:
        raise ValueError("n_states must satisfy 1 <= n_states <= p_dim")

    metric = np.eye(p_dim, dtype=omega.dtype) + omega.T.conj() @ omega
    projected = (
        H_PP
        + H_PQ @ omega
        + omega.T.conj() @ H_PQ.T.conj()
        + omega.T.conj() @ H_QQ @ omega
    )
    projected = 0.5 * (projected + projected.T.conj())

    metric_evals, metric_vecs = eigh(metric)
    if metric_evals[0] <= metric_floor:
        raise np.linalg.LinAlgError(
            f"wave-operator graph metric is singular: min={metric_evals[0]:.3e}")
    metric_inv_sqrt = (
        metric_vecs * (1.0 / np.sqrt(metric_evals))[np.newaxis, :]
    ) @ metric_vecs.T.conj()

    orthogonal_h = metric_inv_sqrt @ projected @ metric_inv_sqrt
    orthogonal_h = 0.5 * (orthogonal_h + orthogonal_h.T.conj())
    energies, orthogonal_vectors = eigh(orthogonal_h)
    energies = energies[:n_states]
    model_coefficients = metric_inv_sqrt @ orthogonal_vectors[:, :n_states]
    q_coefficients = omega @ model_coefficients
    full_coefficients = np.vstack([model_coefficients, q_coefficients])

    return {
        'energies': energies,
        'model_coefficients': model_coefficients,
        'q_coefficients': q_coefficients,
        'full_coefficients': full_coefficients,
        'metric': metric,
        'projected_hamiltonian': projected,
    }


def compute_state_residuals(
    H_PP: np.ndarray,
    H_PQ: np.ndarray,
    H_QQ: np.ndarray,
    energies: np.ndarray,
    model_coefficients: np.ndarray,
    q_coefficients: np.ndarray,
    state_weights: Optional[np.ndarray] = None,
) -> Dict:
    """Compute P/Q Schrödinger residuals and state-averaged norms."""
    n_states = len(energies)
    weights = normalize_state_weights(n_states, state_weights)
    residual_p = (
        H_PP @ model_coefficients + H_PQ @ q_coefficients
        - model_coefficients * energies[np.newaxis, :]
    )
    residual_q = (
        H_PQ.T.conj() @ model_coefficients + H_QQ @ q_coefficients
        - q_coefficients * energies[np.newaxis, :]
    )
    root_norms = np.sqrt(
        np.sum(np.abs(residual_p) ** 2, axis=0)
        + np.sum(np.abs(residual_q) ** 2, axis=0))
    weighted_rms = float(np.sqrt(np.dot(weights, root_norms ** 2)))
    return {
        'residual_p': residual_p,
        'residual_q': residual_q,
        'root_norms': root_norms,
        'weighted_rms': weighted_rms,
        'max_norm': float(np.max(root_norms)),
    }


def _protect_denominators(
    denominators: np.ndarray,
    min_denominator: float,
) -> np.ndarray:
    if min_denominator <= 0.0:
        raise ValueError("min_denominator must be positive")
    protected = np.array(denominators, copy=True)
    small = np.abs(protected) < min_denominator
    signs = np.where(np.real(protected) < 0.0, -1.0, 1.0)
    protected[small] = signs[small] * min_denominator
    return protected


def solve_state_averaged_wave_operator(
    H_PP: np.ndarray,
    H_PQ: Dict[int, np.ndarray],
    H_QQ_blocks: Dict[Tuple[int, int], np.ndarray],
    D_by_n: Dict[int, np.ndarray],
    n_states: int,
    state_weights: Optional[np.ndarray] = None,
    omega_init: Optional[np.ndarray] = None,
    damping: float = 0.5,
    residual_tol: float = 1e-9,
    energy_tol: float = 1e-10,
    max_iter: int = 100,
    min_denominator: float = 1e-6,
    pinv_rcond: float = 1e-12,
    verbose: bool = True,
) -> Dict:
    """Solve several roots with one residual-dressed wave operator.

    At iteration ``t`` the graph of ``Omega_t`` is diagonalized variationally.
    For each target root, the Q residual is preconditioned by
    ``(E_k - diag(H_QQ))^-1``.  The state corrections are fitted back to one
    low-rank operator update.  ``state_weights`` control both the aggregate
    convergence norm and the relative update size of each root.
    """
    H_PP = np.asarray(H_PP)
    p_dim = H_PP.shape[0]
    if H_PP.shape != (p_dim, p_dim):
        raise ValueError("H_PP must be square")
    if n_states <= 0 or n_states > p_dim:
        raise ValueError("n_states must satisfy 1 <= n_states <= p_dim")
    if not 0.0 < damping <= 1.0:
        raise ValueError("damping must satisfy 0 < damping <= 1")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")

    weights = normalize_state_weights(n_states, state_weights)
    assembled = assemble_qspace_hamiltonian(H_PQ, H_QQ_blocks, D_by_n)
    H_PQ_full = assembled['H_PQ']
    H_QQ = assembled['H_QQ']
    diagonal = assembled['diagonal']
    q_dim = assembled['q_dim']

    if q_dim == 0:
        energies, vectors = eigh(0.5 * (H_PP + H_PP.T.conj()))
        energies = energies[:n_states]
        vectors = vectors[:, :n_states]
        return {
            'energies': energies,
            'model_coefficients': vectors,
            'q_coefficients': np.zeros((0, n_states)),
            'full_coefficients': vectors,
            'omega': np.zeros((0, p_dim)),
            'converged': True,
            'n_iter': 0,
            'history': [],
            'residuals': {
                'residual_p': np.zeros_like(vectors),
                'residual_q': np.zeros((0, n_states)),
                'root_norms': np.zeros(n_states),
                'weighted_rms': 0.0,
                'max_norm': 0.0,
            },
            'state_weights': weights,
            **assembled,
        }

    if H_PQ_full.shape[0] != p_dim:
        raise ValueError(
            f"H_PQ has P dimension {H_PQ_full.shape[0]}, expected {p_dim}")

    dtype = np.result_type(H_PP.dtype, H_PQ_full.dtype, H_QQ.dtype)
    if omega_init is None:
        omega = np.zeros((q_dim, p_dim), dtype=dtype)
    else:
        omega = np.asarray(omega_init, dtype=dtype).copy()
        if omega.shape != (q_dim, p_dim):
            raise ValueError(
                f"omega_init has shape {omega.shape}, expected {(q_dim, p_dim)}")

    history: List[Dict] = []
    previous_energies = None
    converged = False
    update_scales = np.sqrt(weights / np.max(weights))

    if verbose:
        print("\n  State-averaged residual-dressed wave-operator solver")
        print(f"    roots={n_states}, |P|={p_dim}, |Q|={q_dim}")
        print(f"    weights={np.array2string(weights, precision=4)}")
        print(f"    damping={damping:.3f}, residual_tol={residual_tol:.1e}")

    for iteration in range(max_iter):
        ritz = graph_ritz(H_PP, H_PQ_full, H_QQ, omega, n_states)
        residuals = compute_state_residuals(
            H_PP, H_PQ_full, H_QQ,
            ritz['energies'], ritz['model_coefficients'],
            ritz['q_coefficients'], weights)

        if previous_energies is None:
            energy_change = np.inf
        else:
            energy_change = float(np.max(np.abs(
                ritz['energies'] - previous_energies)))

        history.append({
            'iteration': iteration,
            'energies': ritz['energies'].copy(),
            'energy_change': energy_change,
            'residual_rms': residuals['weighted_rms'],
            'residual_max': residuals['max_norm'],
            'omega_norm': float(np.linalg.norm(omega)),
        })

        if verbose:
            dE_text = "---" if not np.isfinite(energy_change) else f"{energy_change:.3e}"
            print(
                f"    iter {iteration:3d}: E0={ritz['energies'][0]:.12f} "
                f"max|dE|={dE_text}  "
                f"R_SA={residuals['weighted_rms']:.3e}",
                flush=True)

        if (residuals['weighted_rms'] < residual_tol
                and energy_change < energy_tol):
            converged = True
            break

        denominators = (
            ritz['energies'][np.newaxis, :] - diagonal[:, np.newaxis])
        denominators = _protect_denominators(denominators, min_denominator)
        state_corrections = residuals['residual_q'] / denominators
        state_corrections *= update_scales[np.newaxis, :]

        # Minimum-norm shared operator whose action on the target model span
        # reproduces the preconditioned state corrections.
        delta_omega = state_corrections @ np.linalg.pinv(
            ritz['model_coefficients'], rcond=pinv_rcond)
        omega += damping * delta_omega
        previous_energies = ritz['energies'].copy()

    # Re-evaluate after the last update (or reuse the converged iterate).
    final_ritz = graph_ritz(H_PP, H_PQ_full, H_QQ, omega, n_states)
    final_residuals = compute_state_residuals(
        H_PP, H_PQ_full, H_QQ,
        final_ritz['energies'], final_ritz['model_coefficients'],
        final_ritz['q_coefficients'], weights)
    if not converged and previous_energies is not None:
        final_energy_change = float(np.max(np.abs(
            final_ritz['energies'] - previous_energies)))
        converged = (
            final_residuals['weighted_rms'] < residual_tol
            and final_energy_change < energy_tol)

    return {
        **final_ritz,
        'omega': omega,
        'converged': converged,
        'n_iter': len(history),
        'history': history,
        'residuals': final_residuals,
        'state_weights': weights,
        **assembled,
    }
