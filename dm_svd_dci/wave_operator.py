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


def graph_ritz_per_state(
    H_PP: np.ndarray,
    H_PQ: np.ndarray,
    H_QQ: np.ndarray,
    omegas: np.ndarray,
    n_states: int,
    previous_coefficients: Optional[np.ndarray] = None,
    metric_floor: float = 1e-12,
) -> Dict:
    """Rayleigh--Ritz solve with one wave operator per state.

    Each state k is solved in its own graph subspace ``X_k = [I_P ; Omega_k]``,
    so the downfolding is centred on that state rather than shared across all
    of them.  This is the controlled counterpart of :func:`graph_ritz`, built
    because the project's own record shows that a single resolvent centre
    shared across states breaks excited states, while per-state centring fixes
    them.

    Two properties of the shared construction are deliberately given up here,
    and callers must not assume them:

    * the returned states live in **different** subspaces, so they are not
      mutually orthonormal in the full P+Q metric;
    * root k is no longer simply the k-th eigenvalue of one operator, so it is
      tracked by overlap against the previous iterate rather than by index.

    Returns the same keys as :func:`graph_ritz`, plus ``root_indices`` giving
    which root of each state's own spectrum was selected.
    """
    H_PP = np.asarray(H_PP)
    H_PQ = np.asarray(H_PQ)
    H_QQ = np.asarray(H_QQ)
    omegas = np.asarray(omegas)
    p_dim = H_PP.shape[0]
    q_dim = H_QQ.shape[0]

    if omegas.shape != (n_states, q_dim, p_dim):
        raise ValueError(
            f"omegas must have shape {(n_states, q_dim, p_dim)}, "
            f"got {omegas.shape}")
    if n_states <= 0 or n_states > p_dim:
        raise ValueError("n_states must satisfy 1 <= n_states <= p_dim")

    energies = np.empty(n_states, dtype=float)
    model_coefficients = np.empty((p_dim, n_states), dtype=omegas.dtype)
    q_coefficients = np.empty((q_dim, n_states), dtype=omegas.dtype)
    root_indices = np.empty(n_states, dtype=int)

    for state in range(n_states):
        omega = omegas[state]
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
                f"per-state graph metric for root {state} is singular: "
                f"min={metric_evals[0]:.3e}")
        metric_inv_sqrt = (
            metric_vecs * (1.0 / np.sqrt(metric_evals))[np.newaxis, :]
        ) @ metric_vecs.T.conj()

        orthogonal_h = metric_inv_sqrt @ projected @ metric_inv_sqrt
        orthogonal_h = 0.5 * (orthogonal_h + orthogonal_h.T.conj())
        local_energies, local_vectors = eigh(orthogonal_h)
        candidates = metric_inv_sqrt @ local_vectors

        if previous_coefficients is None:
            chosen = state
        else:
            # Index-based selection is a documented trap for excited states, so
            # the root is followed by overlap once a previous iterate exists.
            reference = previous_coefficients[:, state]
            overlaps = np.abs(candidates.T.conj() @ reference)
            chosen = int(np.argmax(overlaps))

        energies[state] = float(local_energies[chosen])
        model_coefficients[:, state] = candidates[:, chosen]
        q_coefficients[:, state] = omega @ candidates[:, chosen]
        root_indices[state] = chosen

    return {
        'energies': energies,
        'model_coefficients': model_coefficients,
        'q_coefficients': q_coefficients,
        'full_coefficients': np.vstack([model_coefficients, q_coefficients]),
        'root_indices': root_indices,
    }


def make_qspace_apply(
    H_PQ: Dict[int, np.ndarray],
    H_QQ_blocks: Dict[Tuple[int, int], np.ndarray],
    D_by_n: Dict[int, np.ndarray],
) -> Dict:
    """Build an ``H_QQ`` apply callback without assembling the q by q matrix.

    :func:`assemble_qspace_hamiltonian` allocates a dense ``q_dim x q_dim``
    array.  This returns the same operator as a callable that contracts the
    per-electron-number blocks directly, so the quadratic array is never
    created.  Selection rules leave only the ``|dn| <= 2`` blocks non-zero,
    which is what makes the block form cheaper than the assembled one.

    Returns ``H_PQ`` concatenated in block order, the Q-space diagonal, the
    total Q dimension, and ``apply``.
    """
    q_labels = sorted(H_PQ.keys())
    slices = {}
    offset = 0
    for label in q_labels:
        dim = np.asarray(H_PQ[label]).shape[1]
        slices[label] = slice(offset, offset + dim)
        offset += dim

    if q_labels:
        h_pq_full = np.concatenate(
            [np.asarray(H_PQ[label]) for label in q_labels], axis=1)
        diagonal = np.concatenate(
            [np.asarray(D_by_n[label]) for label in q_labels])
    else:
        h_pq_full = np.zeros((0, 0))
        diagonal = np.zeros(0)

    # Resolve each block once, symmetrizing where both orientations exist, so
    # the apply itself does no bookkeeping.
    resolved = []
    for left in q_labels:
        for right in q_labels:
            block = H_QQ_blocks.get((left, right))
            transposed = H_QQ_blocks.get((right, left))
            if block is None and transposed is None:
                if left == right:
                    resolved.append((slices[left], slices[left],
                                     np.diag(np.asarray(D_by_n[left]))))
                continue
            if block is None:
                block = np.asarray(transposed).T.conj()
            elif transposed is not None and left != right:
                block = 0.5 * (np.asarray(block)
                               + np.asarray(transposed).T.conj())
            resolved.append((slices[left], slices[right], np.asarray(block)))

    def apply(vectors: np.ndarray) -> np.ndarray:
        vectors = np.asarray(vectors)
        result = np.zeros((offset, vectors.shape[1]), dtype=vectors.dtype)
        for row, column, block in resolved:
            result[row] += block @ vectors[column]
        return result

    return {'H_PQ': h_pq_full, 'diagonal': diagonal, 'q_dim': offset,
            'apply': apply, 'n_blocks': len(resolved),
            'q_slices': slices, 'q_labels': q_labels}


def solve_state_averaged_wave_operator_lowrank(
    H_PP: np.ndarray,
    H_PQ: np.ndarray,
    hqq_apply,
    diagonal: np.ndarray,
    n_states: int,
    state_weights: Optional[np.ndarray] = None,
    damping: float = 0.5,
    residual_tol: float = 1e-9,
    energy_tol: float = 1e-10,
    max_iter: int = 100,
    min_denominator: float = 1e-6,
    pinv_rcond: float = 1e-12,
    max_rank: Optional[int] = None,
    rank_tol: float = 1e-13,
    q_slices: Optional[Dict] = None,
    q_labels: Optional[List] = None,
    verbose: bool = True,
) -> Dict:
    """Matrix-free wave-operator solve, with Omega kept in factored form.

    Identical mathematics to :func:`solve_state_averaged_wave_operator`, but
    ``H_QQ`` is never formed.  It enters only through ``hqq_apply(V)``, which
    must return ``H_QQ @ V`` for ``V`` of shape ``(q_dim, m)``.

    The enabling fact is that ``Omega`` is numerically rank ``n_states``
    (``docs/theory/wave_operator_low_rank_structure.md``).  Keeping it as
    ``Omega = W Omega_tilde`` with ``W`` orthonormal of width ``k`` means the
    generalized Ritz problem needs only ``H_PQ W`` of size ``p x k`` and
    ``W^dag H_QQ W`` of size ``k x k``.  ``H_QQ`` is applied to the ``k``
    columns of ``W`` and nowhere else, so the cost per iteration is ``k``
    applications rather than an ``O(q_dim^2)`` build.

    The retained rank is controlled by ``rank_tol``, which drops factorization
    directions whose singular value is negligible relative to the largest.
    ``max_rank`` is an optional hard ceiling; leaving it unset is recommended,
    because a count cap floors the residual and makes the solver report
    non-convergence while its energies are already exact to machine precision.

    Truncation is a cost saving, not a stability requirement.  An earlier
    version of this solver appeared to need it, but that was a loss of
    orthonormality in the factorization, not a property of the method: the
    appended directions are now projected out of the basis twice, and the basis
    is re-orthonormalized after each rotation with the same transformation
    carried through the stored application and the coefficients.  With that in
    place the untruncated run is also correct, and truncation simply costs
    fewer applications of ``H_QQ`` for the same answer.
    """
    H_PP = np.asarray(H_PP)
    H_PQ = np.asarray(H_PQ)
    diagonal = np.asarray(diagonal).reshape(-1)
    p_dim = H_PP.shape[0]
    q_dim = diagonal.shape[0]
    if H_PQ.shape != (p_dim, q_dim):
        raise ValueError(
            f"H_PQ has shape {H_PQ.shape}, expected {(p_dim, q_dim)}")
    if not 0 < damping <= 1.0:
        raise ValueError("damping must satisfy 0 < damping <= 1")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")
    weights = normalize_state_weights(n_states, state_weights)
    # No hard cap by default: the rank is controlled by rank_tol, which keeps
    # only directions carrying real weight.  A count cap would make the solver
    # report non-convergence while already being numerically exact in the
    # energies, because the residual floors out before the energies move.
    update_scales = np.sqrt(weights / np.max(weights))

    # Omega = basis @ coefficients, with basis orthonormal (q_dim x k).
    basis = np.zeros((q_dim, 0))
    coefficients = np.zeros((0, p_dim))
    hqq_basis = np.zeros((q_dim, 0))          # H_QQ applied to basis columns
    applications = 0

    history: List[Dict] = []
    previous_energies = None
    converged = False

    if verbose:
        print("\n  Matrix-free low-rank wave-operator solver")
        print(f"    roots={n_states}, |P|={p_dim}, |Q|={q_dim}, "
              f"max_rank={max_rank}")

    for iteration in range(max_iter):
        k = basis.shape[1]
        hpq_basis = H_PQ @ basis                       # p x k
        hqq_small = basis.T.conj() @ hqq_basis         # k x k
        hqq_small = 0.5 * (hqq_small + hqq_small.T.conj())

        metric = np.eye(p_dim) + coefficients.T.conj() @ coefficients
        projected = H_PP.copy()
        if k:
            cross = hpq_basis @ coefficients
            projected = (projected + cross + cross.T.conj()
                         + coefficients.T.conj() @ hqq_small @ coefficients)
        projected = 0.5 * (projected + projected.T.conj())

        metric_evals, metric_vecs = eigh(metric)
        metric_inv_sqrt = (
            metric_vecs * (1.0 / np.sqrt(np.maximum(metric_evals, 1e-14)))
        ) @ metric_vecs.T.conj()
        orthogonal_h = metric_inv_sqrt @ projected @ metric_inv_sqrt
        orthogonal_h = 0.5 * (orthogonal_h + orthogonal_h.T.conj())
        energies, vectors = eigh(orthogonal_h)
        energies = energies[:n_states]
        model_coefficients = metric_inv_sqrt @ vectors[:, :n_states]

        small_q = coefficients @ model_coefficients if k else np.zeros((0, n_states))
        q_coefficients = basis @ small_q if k else np.zeros((q_dim, n_states))

        residual_p = (H_PP @ model_coefficients + H_PQ @ q_coefficients
                      - model_coefficients * energies[np.newaxis, :])
        # H_QQ q = H_QQ basis (coefficients c), reusing the stored application.
        hqq_q = hqq_basis @ small_q if k else np.zeros((q_dim, n_states))
        residual_q = (H_PQ.T.conj() @ model_coefficients + hqq_q
                      - q_coefficients * energies[np.newaxis, :])
        root_norms = np.sqrt(np.sum(np.abs(residual_p) ** 2, axis=0)
                             + np.sum(np.abs(residual_q) ** 2, axis=0))
        weighted_rms = float(np.sqrt(np.dot(weights, root_norms ** 2)))

        energy_change = (np.inf if previous_energies is None
                         else float(np.max(np.abs(energies - previous_energies))))
        history.append({
            'iteration': iteration, 'energies': energies.copy(),
            'energy_change': energy_change, 'residual_rms': weighted_rms,
            'residual_max': float(np.max(root_norms)),
            'rank': int(k), 'hqq_applications': int(applications),
        })
        if verbose:
            text = "---" if not np.isfinite(energy_change) else f"{energy_change:.3e}"
            print(f"    iter {iteration:3d}: E0={energies[0]:.12f} "
                  f"max|dE|={text}  R_SA={weighted_rms:.3e}  rank={k}",
                  flush=True)

        if weighted_rms < residual_tol and energy_change < energy_tol:
            converged = True
            break

        denominators = _protect_denominators(
            energies[np.newaxis, :] - diagonal[:, np.newaxis], min_denominator)
        corrections = (residual_q / denominators) * update_scales[np.newaxis, :]
        delta_coefficients = np.linalg.pinv(
            model_coefficients, rcond=pinv_rcond)          # s x p

        # Append the new directions, re-orthonormalize, then truncate by SVD.
        # Project the new directions out of the current basis twice.  One
        # pass of Gram-Schmidt loses orthogonality; two is the standard remedy.
        new = np.array(corrections, copy=True)
        for _ in range(2):
            if basis.shape[1]:
                new = new - basis @ (basis.T.conj() @ new)
        norms = np.linalg.norm(new, axis=0)
        keep = norms > rank_tol * max(1.0, float(np.max(norms)))
        if np.any(keep):
            added, _ = np.linalg.qr(new[:, keep])
            basis = np.hstack([basis, added]) if k else added
            hqq_basis = (np.hstack([hqq_basis, hqq_apply(added)])
                         if k else hqq_apply(added))
            applications += added.shape[1]
            coefficients = np.vstack(
                [coefficients, np.zeros((added.shape[1], p_dim))])
        coefficients = coefficients + damping * (
            basis.T.conj() @ corrections) @ delta_coefficients

        # Truncate the factorization by singular value, not by a count.  The
        # rotation has orthonormal columns, so the rotated basis stays
        # orthonormal and H_QQ(basis @ rotation) = (H_QQ basis) @ rotation,
        # which means no extra applications are needed.
        if basis.shape[1]:
            left, sigma, right = np.linalg.svd(coefficients,
                                               full_matrices=False)
            largest = float(sigma[0]) if sigma.size else 0.0
            kept = int(np.sum(sigma > rank_tol * max(largest, 1e-300)))
            kept = max(kept, n_states)
            if max_rank is not None:
                kept = min(kept, max_rank)
            if kept < basis.shape[1]:
                rotation = left[:, :kept]
                basis = basis @ rotation
                hqq_basis = hqq_basis @ rotation
                coefficients = (sigma[:kept, np.newaxis] * right[:kept, :])
                # Rotations accumulate a small loss of orthonormality over many
                # iterations.  Restore it and carry the same transformation
                # through the stored application and the coefficients, so that
                # Omega and H_QQ Omega stay exactly consistent.
                basis, upper = np.linalg.qr(basis)
                inverse = np.linalg.inv(upper)
                hqq_basis = hqq_basis @ inverse
                coefficients = upper @ coefficients
        previous_energies = energies.copy()

    omega = basis @ coefficients if basis.shape[1] else np.zeros((q_dim, p_dim))
    return {
        'energies': energies, 'model_coefficients': model_coefficients,
        'q_coefficients': q_coefficients,
        'full_coefficients': np.vstack([model_coefficients, q_coefficients]),
        'omega': omega, 'omega_basis': basis, 'omega_coefficients': coefficients,
        'converged': converged, 'n_iter': len(history), 'history': history,
        'residuals': {'residual_p': residual_p, 'residual_q': residual_q,
                      'root_norms': root_norms, 'weighted_rms': weighted_rms,
                      'max_norm': float(np.max(root_norms))},
        'state_weights': weights, 'rank': int(basis.shape[1]),
        'hqq_applications': int(applications),
        'q_slices': q_slices, 'q_labels': q_labels, 'q_dim': q_dim,
        'H_PQ': H_PQ, 'diagonal': diagonal,
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
    apply_dressing: bool = True,
    omega_mode: str = 'shared',
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
    if omega_mode not in ('shared', 'per_state'):
        raise ValueError(f"unknown omega_mode {omega_mode!r}")
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
            'omega_mode': omega_mode,
            **assembled,
        }

    if H_PQ_full.shape[0] != p_dim:
        raise ValueError(
            f"H_PQ has P dimension {H_PQ_full.shape[0]}, expected {p_dim}")

    dtype = np.result_type(H_PP.dtype, H_PQ_full.dtype, H_QQ.dtype)
    shared = omega_mode == 'shared'
    expected = ((q_dim, p_dim) if shared else (n_states, q_dim, p_dim))
    if omega_init is None:
        omega = np.zeros(expected, dtype=dtype)
    else:
        omega = np.asarray(omega_init, dtype=dtype).copy()
        if omega.shape != expected:
            raise ValueError(
                f"omega_init has shape {omega.shape}, expected {expected}")

    history: List[Dict] = []
    previous_energies = None
    converged = False
    update_scales = np.sqrt(weights / np.max(weights))

    if verbose:
        print("\n  State-averaged residual-dressed wave-operator solver")
        print(f"    roots={n_states}, |P|={p_dim}, |Q|={q_dim}")
        print(f"    weights={np.array2string(weights, precision=4)}")
        print(f"    damping={damping:.3f}, residual_tol={residual_tol:.1e}")

    previous_coefficients = None
    for iteration in range(max_iter):
        if shared:
            ritz = graph_ritz(H_PP, H_PQ_full, H_QQ, omega, n_states)
        else:
            ritz = graph_ritz_per_state(
                H_PP, H_PQ_full, H_QQ, omega, n_states,
                previous_coefficients=previous_coefficients)
        previous_coefficients = ritz['model_coefficients'].copy()
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

        if not apply_dressing:
            # H4 control: Omega is never updated, so the generalized Ritz
            # problem reduces to diagonalizing H_PP in the model space.
            previous_energies = ritz['energies'].copy()
            break

        denominators = (
            ritz['energies'][np.newaxis, :] - diagonal[:, np.newaxis])
        denominators = _protect_denominators(denominators, min_denominator)
        state_corrections = residuals['residual_q'] / denominators

        if shared:
            # The weights balance states competing for one operator.
            state_corrections = state_corrections * update_scales[np.newaxis, :]
            # Minimum-norm shared operator whose action on the target model
            # span reproduces the preconditioned state corrections.
            delta_omega = state_corrections @ np.linalg.pinv(
                ritz['model_coefficients'], rcond=pinv_rcond)
            omega += damping * delta_omega
        else:
            # With one operator per state there is no competition, so the
            # weights do not apply and the pseudoinverse over states collapses
            # to a rank-one minimum-norm update per state.
            for state in range(n_states):
                coefficients = ritz['model_coefficients'][:, state]
                norm_squared = float(
                    np.real(coefficients.conj() @ coefficients))
                if norm_squared <= 0.0:
                    continue
                omega[state] += damping * np.outer(
                    state_corrections[:, state],
                    coefficients.conj()) / norm_squared
        previous_energies = ritz['energies'].copy()

    # Re-evaluate after the last update (or reuse the converged iterate).
    if shared:
        final_ritz = graph_ritz(H_PP, H_PQ_full, H_QQ, omega, n_states)
    else:
        final_ritz = graph_ritz_per_state(
            H_PP, H_PQ_full, H_QQ, omega, n_states,
            previous_coefficients=previous_coefficients)
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
        'apply_dressing': bool(apply_dressing),
        'omega_mode': omega_mode,
        **assembled,
    }
