#!/usr/bin/env python3
"""Outer self-consistency for state-averaged dmSVD downfolding.

The outer loop is expressed against a Hamiltonian-builder callback so that the
Schmidt/wave-operator mathematics remains separate from PySCF and from the
existing H_A/H_B/H_AB implementations.  Each iteration performs

1. weighted state-averaged Schmidt compression,
2. construction of the P/Q Hamiltonian in that common basis,
3. a shared residual-dressed wave-operator solve,
4. reconstruction of all dressed physical CI coefficient blocks, and
5. density/basis feedback into the next Schmidt decomposition.
"""

from itertools import permutations
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from dm_svd_embedding.density_matrix import (
    compute_schmidt_decomposition,
    normalize_state_weights,
)
from dm_svd_dci.wave_operator import (
    make_qspace_apply,
    solve_state_averaged_wave_operator,
    solve_state_averaged_wave_operator_lowrank,
)


StateBlocks = Dict[int, np.ndarray]


def state_averaged_reduced_densities(
    state_blocks: List[StateBlocks],
    state_weights: Optional[np.ndarray] = None,
) -> Dict[int, Dict[str, np.ndarray]]:
    """Build weighted left/right reduced densities for every number block."""
    weights = normalize_state_weights(len(state_blocks), state_weights)
    labels = sorted(set().union(*(blocks.keys() for blocks in state_blocks)))
    result = {}

    for label in labels:
        template = next(
            (blocks[label] for blocks in state_blocks if label in blocks), None)
        if template is None or np.asarray(template).ndim != 2:
            raise ValueError(f"invalid coefficient block for n={label}")
        dim_a, dim_b = np.asarray(template).shape
        dtype = np.result_type(*[
            np.asarray(blocks[label]).dtype
            for blocks in state_blocks if label in blocks
        ])
        rho_a = np.zeros((dim_a, dim_a), dtype=dtype)
        rho_b = np.zeros((dim_b, dim_b), dtype=dtype)

        for weight, blocks in zip(weights, state_blocks):
            coefficient = blocks.get(label)
            if coefficient is None:
                continue
            coefficient = np.asarray(coefficient)
            if coefficient.shape != (dim_a, dim_b):
                raise ValueError(
                    f"state block n={label} has inconsistent shape "
                    f"{coefficient.shape}, expected {(dim_a, dim_b)}")
            rho_a += weight * (coefficient @ coefficient.T.conj())
            rho_b += weight * (coefficient.T.conj() @ coefficient)

        result[label] = {'rho_A': rho_a, 'rho_B': rho_b}
    return result


def state_averaged_density_distance(
    reference: List[StateBlocks],
    candidate: List[StateBlocks],
    state_weights: Optional[np.ndarray] = None,
) -> float:
    """Frobenius distance between two weighted state-averaged densities."""
    if len(reference) != len(candidate):
        raise ValueError("reference and candidate must contain the same states")
    rho_ref = state_averaged_reduced_densities(reference, state_weights)
    rho_new = state_averaged_reduced_densities(candidate, state_weights)
    labels = sorted(set(rho_ref) | set(rho_new))
    distance_sq = 0.0
    for label in labels:
        if label not in rho_ref or label not in rho_new:
            raise ValueError("reference and candidate use different number blocks")
        distance_sq += np.linalg.norm(
            rho_ref[label]['rho_A'] - rho_new[label]['rho_A']) ** 2
        distance_sq += np.linalg.norm(
            rho_ref[label]['rho_B'] - rho_new[label]['rho_B']) ** 2
    return float(np.sqrt(distance_sq))


def state_overlap(left: StateBlocks, right: StateBlocks) -> complex:
    """Physical CI overlap evaluated directly from number-block matrices."""
    labels = set(left) | set(right)
    overlap = 0.0
    for label in labels:
        if label not in left or label not in right:
            continue
        if left[label].shape != right[label].shape:
            raise ValueError(f"inconsistent state block shape for n={label}")
        overlap += np.vdot(left[label], right[label])
    return overlap


def _linear_combination(
    state_blocks: List[StateBlocks],
    coefficients: np.ndarray,
) -> StateBlocks:
    labels = sorted(set().union(*(blocks.keys() for blocks in state_blocks)))
    result = {}
    for label in labels:
        template = next(blocks[label] for blocks in state_blocks if label in blocks)
        dtype = np.result_type(
            coefficients.dtype,
            *[blocks[label].dtype for blocks in state_blocks if label in blocks])
        combined = np.zeros(template.shape, dtype=dtype)
        for coefficient, blocks in zip(coefficients, state_blocks):
            if label in blocks:
                combined += coefficient * blocks[label]
        result[label] = combined
    return result


def orthonormalize_state_blocks(
    state_blocks: List[StateBlocks],
    metric_floor: float = 1e-12,
) -> List[StateBlocks]:
    """Symmetrically orthonormalize physical states stored as block matrices."""
    n_states = len(state_blocks)
    if n_states == 0:
        raise ValueError("at least one state is required")
    gram = np.empty((n_states, n_states), dtype=complex)
    for i in range(n_states):
        for j in range(n_states):
            gram[i, j] = state_overlap(state_blocks[i], state_blocks[j])
    gram = 0.5 * (gram + gram.T.conj())
    eigvals, eigvecs = np.linalg.eigh(gram)
    if eigvals[0] <= metric_floor:
        raise np.linalg.LinAlgError(
            f"state set is linearly dependent: min metric={eigvals[0]:.3e}")
    inv_sqrt = (
        eigvecs * (1.0 / np.sqrt(eigvals))[np.newaxis, :]
    ) @ eigvecs.T.conj()
    result = [
        _linear_combination(state_blocks, inv_sqrt[:, root])
        for root in range(n_states)
    ]

    # Preserve real storage when all numerical imaginary parts vanish.
    for blocks in result:
        for label, coefficient in blocks.items():
            blocks[label] = np.real_if_close(coefficient)
    return result


def align_state_blocks(
    reference: List[StateBlocks],
    candidate: List[StateBlocks],
) -> Tuple[List[StateBlocks], np.ndarray, np.ndarray]:
    """Match roots by physical overlap and fix their arbitrary phases."""
    n_states = len(reference)
    if len(candidate) != n_states:
        raise ValueError("reference and candidate must contain the same states")
    overlaps = np.empty((n_states, n_states), dtype=complex)
    for i in range(n_states):
        for j in range(n_states):
            overlaps[i, j] = state_overlap(reference[i], candidate[j])

    if n_states <= 8:
        permutation = max(
            permutations(range(n_states)),
            key=lambda perm: sum(abs(overlaps[i, perm[i]]) for i in range(n_states)))
    else:
        available = set(range(n_states))
        chosen = []
        for i in range(n_states):
            root = max(available, key=lambda j: abs(overlaps[i, j]))
            chosen.append(root)
            available.remove(root)
        permutation = tuple(chosen)

    phases = np.ones(n_states, dtype=complex)
    aligned = []
    for i, root in enumerate(permutation):
        overlap = overlaps[i, root]
        if abs(overlap) > 0.0:
            phases[i] = overlap.conjugate() / abs(overlap)
        aligned.append(_linear_combination([candidate[root]], np.array([phases[i]])))
    return (aligned, np.asarray(permutation, dtype=int), phases,
            np.asarray(overlaps))


def mix_state_blocks(
    reference: List[StateBlocks],
    candidate: List[StateBlocks],
    mixing: float,
) -> List[StateBlocks]:
    """Linearly damp and re-orthonormalize a tracked set of states."""
    if not 0.0 < mixing <= 1.0:
        raise ValueError("mixing must satisfy 0 < mixing <= 1")
    mixed = []
    for old, new in zip(reference, candidate):
        mixed.append(_linear_combination(
            [old, new], np.array([1.0 - mixing, mixing])))
    return orthonormalize_state_blocks(mixed)


def schmidt_product_coefficients_to_blocks(
    coefficients: np.ndarray,
    schmidt_data: Dict[int, Dict],
) -> List[StateBlocks]:
    """Expand Schmidt-product coefficients into physical CI block matrices.

    ``coefficients`` follows the full embedded basis order used by
    :func:`partition_schmidt_basis`: increasing number block, then
    ``alpha * r_B + beta`` within a block.
    """
    coefficients = np.asarray(coefficients)
    if coefficients.ndim == 1:
        coefficients = coefficients[:, np.newaxis]
    if coefficients.ndim != 2:
        raise ValueError("coefficients must be a vector or matrix")

    expected_dim = sum(
        data.get('r_A', data['r']) * data.get('r_B', data['r'])
        for data in schmidt_data.values())
    if coefficients.shape[0] != expected_dim:
        raise ValueError(
            f"coefficient dimension {coefficients.shape[0]} != {expected_dim}")

    state_blocks: List[StateBlocks] = [
        {} for _ in range(coefficients.shape[1])]
    offset = 0
    for label in sorted(schmidt_data):
        data = schmidt_data[label]
        r_A = data.get('r_A', data['r'])
        r_B = data.get('r_B', data['r'])
        size = r_A * r_B
        for root in range(coefficients.shape[1]):
            product_coefficients = coefficients[
                offset:offset + size, root].reshape(r_A, r_B)
            state_blocks[root][label] = (
                data['U'] @ product_coefficients @ data['V'].T.conj())
        offset += size
    return state_blocks


def assemble_embedded_state_coefficients(
    wave_result: Dict,
    part_info: Dict,
    q_partition: Dict,
) -> np.ndarray:
    """Map P-first/stacked-Q wave amplitudes to the full embedded ordering."""
    p_coefficients = np.asarray(wave_result['model_coefficients'])
    q_coefficients = np.asarray(wave_result['q_coefficients'])
    n_states = p_coefficients.shape[1]
    full = np.zeros(
        (part_info['total_dim'], n_states),
        dtype=np.result_type(p_coefficients.dtype, q_coefficients.dtype))
    full[part_info['p_indices'], :] = p_coefficients

    for label, q_slice in wave_result['q_slices'].items():
        if label not in q_partition['q_blocks']:
            raise ValueError(f"missing Q partition metadata for block {label}")
        q_local = q_partition['q_blocks'][label]['indices']
        embedded_indices = part_info['q_indices'][q_local]
        if len(embedded_indices) != q_slice.stop - q_slice.start:
            raise ValueError(f"Q block {label} dimension mismatch")
        full[embedded_indices, :] = q_coefficients[q_slice, :]
    return full


def schmidt_projector_distance(
    previous: Optional[Dict[int, Dict]],
    current: Dict[int, Dict],
) -> Optional[Dict]:
    """Frobenius distance between consecutive Schmidt projectors, per block.

    The retained left subspace of block ``n`` is represented by the projector
    ``P_A(n) = U(n) U(n)^dagger``, which has a fixed ``dim_A x dim_A`` shape
    regardless of the retained rank, so it stays comparable across outer
    iterations even when the rank changes.  A block that is absent or has rank
    zero contributes the zero projector, which is what makes an irreversible
    block deletion visible as a finite jump rather than as a missing key.

    Returns ``None`` on the first outer iteration, when there is nothing to
    compare against.
    """
    if previous is None:
        return None

    per_block: Dict[int, Dict[str, float]] = {}
    total_a_sq = 0.0
    total_b_sq = 0.0
    for label in sorted(set(previous) | set(current)):
        old = previous.get(label)
        new = current.get(label)
        reference = new if new is not None else old
        dim_a = int(reference['dim_A'])
        dim_b = int(reference['dim_B'])

        def projector(data: Optional[Dict], key: str, dim: int) -> np.ndarray:
            if data is None:
                return np.zeros((dim, dim))
            basis = np.asarray(data[key])
            if basis.size == 0 or basis.shape[1] == 0:
                return np.zeros((dim, dim))
            return basis @ basis.conj().T

        delta_a = projector(new, 'U', dim_a) - projector(old, 'U', dim_a)
        delta_b = projector(new, 'V', dim_b) - projector(old, 'V', dim_b)
        distance_a = float(np.linalg.norm(delta_a))
        distance_b = float(np.linalg.norm(delta_b))
        total_a_sq += distance_a ** 2
        total_b_sq += distance_b ** 2
        per_block[int(label)] = {'A': distance_a, 'B': distance_b}

    total_a = float(np.sqrt(total_a_sq))
    total_b = float(np.sqrt(total_b_sq))
    return {
        'per_block': per_block,
        'total_A': total_a,
        'total_B': total_b,
        'total': float(np.sqrt(total_a_sq + total_b_sq)),
    }


def solve_state_averaged_schmidt(
    initial_state_blocks: List[StateBlocks],
    build_problem: Callable[[Dict[int, Dict]], Dict],
    svd_eps: float = 1e-3,
    state_weights: Optional[np.ndarray] = None,
    outer_mixing: float = 1.0,
    density_tol: float = 1e-7,
    energy_tol: float = 1e-8,
    max_outer_iter: int = 20,
    wave_options: Optional[Dict] = None,
    rank_mode: str = 'rectangular',
    enrichment: Optional[Callable] = None,
    enrichment_strength: float = 0.0,
    enrichment_decay: float = 1.0,
    verbose: bool = True,
) -> Dict:
    """Close the state-averaged Schmidt/wave-operator feedback loop.

    ``build_problem(schmidt_data)`` must return ``H_PP``, ``H_PQ``,
    ``H_QQ_blocks``, ``D_by_n``, ``part_info``, and ``q_partition``.  Extra
    entries are preserved as diagnostics in the returned ``problem`` object.
    """
    if max_outer_iter <= 0:
        raise ValueError("max_outer_iter must be positive")
    n_states = len(initial_state_blocks)
    weights = normalize_state_weights(n_states, state_weights)
    current_states = orthonormalize_state_blocks(initial_state_blocks)
    wave_options = dict(wave_options or {})
    # The outer driver owns these arguments.
    wave_options.pop('n_states', None)
    wave_options.pop('state_weights', None)
    omega_solver = wave_options.pop('omega_solver', 'dense')
    if omega_solver not in ('dense', 'lowrank'):
        raise ValueError(f"unknown omega_solver {omega_solver!r}")

    previous_energies = None
    previous_schmidt: Optional[Dict[int, Dict]] = None
    enrichment_blocks: Optional[List[StateBlocks]] = None
    enrichment_norms: Optional[List[float]] = None
    history: List[Dict] = []
    converged = False
    final_problem = None
    final_schmidt = None
    final_wave = None
    final_states = current_states
    final_permutation = np.arange(n_states)

    if verbose:
        print("\nState-averaged self-consistent Schmidt downfolding")
        print(f"  states={n_states}, weights={np.array2string(weights, precision=4)}")
        print(f"  svd_eps={svd_eps:.1e}, outer_mixing={outer_mixing:.3f}")

    for outer_iteration in range(max_outer_iter):
        # Residual-driven Krylov enrichment.  The reconstructed coefficients
        # C_new = U T V^dagger lie inside the span of the basis that produced
        # them, so without enrichment the retained rank is monotonically
        # non-increasing and the iteration can only lose directions.  The
        # full-space residual R_k = H C_k - E_k C_k leaves that span by
        # construction, and it vanishes at the exact solution, so the exact
        # answer remains a fixed point.  Weights carry ||R_k||^2 rather than a
        # normalized direction, so the enrichment fades as the residual does.
        # The residual does NOT vanish: a state in a truncated space can never
        # be an exact eigenvector of the full Hamiltonian, so ||R_k|| plateaus
        # and an undecayed enrichment is a permanent rotating perturbation that
        # prevents the basis from ever settling.  The strength is therefore
        # annealed, which is what DMRG does with its noise term.
        current_enrichment = enrichment_strength * (
            enrichment_decay ** outer_iteration)
        averaged_states = list(current_states)
        averaged_weights = list(weights)
        if enrichment_blocks is not None and current_enrichment > 1e-12:
            for index, extra in enumerate(enrichment_blocks):
                averaged_states.append(extra)
                averaged_weights.append(
                    current_enrichment * float(weights[index]))
        averaged_weights = np.asarray(averaged_weights, dtype=float)
        if averaged_weights.sum() > 0.0:
            averaged_weights = averaged_weights / averaged_weights.sum()

        schmidt = compute_schmidt_decomposition(
            averaged_states[0], eps=svd_eps,
            state_average=averaged_states, state_weights=averaged_weights,
            rank_mode=rank_mode)
        projector_distance = schmidt_projector_distance(
            previous_schmidt, schmidt)
        problem = build_problem(schmidt)
        required = {
            'H_PP', 'H_PQ', 'H_QQ_blocks', 'D_by_n',
            'part_info', 'q_partition',
        }
        missing = required - set(problem)
        if missing:
            raise KeyError(f"build_problem result is missing {sorted(missing)}")
        if problem['H_PP'].shape[0] < n_states:
            raise ValueError(
                "state-averaged Schmidt truncation left fewer P-space "
                f"vectors ({problem['H_PP'].shape[0]}) than roots ({n_states})")

        local_wave_options = dict(wave_options)
        local_wave_options.setdefault('verbose', verbose)
        if omega_solver == 'dense':
            wave = solve_state_averaged_wave_operator(
                problem['H_PP'], problem['H_PQ'],
                problem['H_QQ_blocks'], problem['D_by_n'],
                n_states=n_states, state_weights=weights,
                **local_wave_options)
        else:
            # Matrix-free: the q by q Q-space Hamiltonian is never assembled.
            # These options belong only to the dense path.
            for unsupported in ('omega_mode', 'apply_dressing', 'omega_init'):
                local_wave_options.pop(unsupported, None)
            qspace = make_qspace_apply(
                problem['H_PQ'], problem['H_QQ_blocks'], problem['D_by_n'])
            wave = solve_state_averaged_wave_operator_lowrank(
                problem['H_PP'], qspace['H_PQ'], qspace['apply'],
                qspace['diagonal'], n_states=n_states, state_weights=weights,
                q_slices=qspace['q_slices'], q_labels=qspace['q_labels'],
                **local_wave_options)

        embedded_coefficients = assemble_embedded_state_coefficients(
            wave, problem['part_info'], problem['q_partition'])
        dressed_states = schmidt_product_coefficients_to_blocks(
            embedded_coefficients, schmidt)
        dressed_states = orthonormalize_state_blocks(dressed_states)
        aligned_states, permutation, phases, overlap_matrix = (
            align_state_blocks(current_states, dressed_states))
        ordered_energies = wave['energies'][permutation]

        density_change = state_averaged_density_distance(
            current_states, aligned_states, weights)
        if previous_energies is None:
            energy_change = np.inf
        else:
            energy_change = float(np.max(np.abs(
                ordered_energies - previous_energies)))

        history.append({
            'outer_iteration': outer_iteration,
            'energies': ordered_energies.copy(),
            'energy_change': energy_change,
            'density_change': density_change,
            'wave_converged': wave['converged'],
            'wave_iterations': wave['n_iter'],
            'wave_residual_rms': wave['residuals']['weighted_rms'],
            'wave_residual_root_norms': np.asarray(
                wave['residuals']['root_norms']).copy(),
            'wave_residual_max': wave['residuals']['max_norm'],
            'wave_history': wave['history'],
            'root_permutation': permutation.copy(),
            'root_overlap_matrix': np.abs(overlap_matrix).copy(),
            'schmidt_projector_distance': projector_distance,
            'rank_mode': rank_mode,
            'enrichment_strength': float(current_enrichment),
            'enrichment_residual_norms': (
                None if enrichment_norms is None else list(enrichment_norms)),
            'schmidt_ranks': {
                int(label): {
                    'r_A': int(data.get('r_A', data['r'])),
                    'r_B': int(data.get('r_B', data['r'])),
                }
                for label, data in schmidt.items()},
        })

        if verbose:
            dE_text = "---" if not np.isfinite(energy_change) else f"{energy_change:.3e}"
            print(
                f"  outer {outer_iteration:2d}: E0={ordered_energies[0]:.12f} "
                f"max|dE|={dE_text}  d_rho={density_change:.3e} "
                f"R={wave['residuals']['weighted_rms']:.3e}",
                flush=True)

        final_problem = problem
        final_schmidt = schmidt
        final_wave = wave
        final_states = aligned_states
        final_permutation = permutation

        if (wave['converged']
                and density_change < density_tol
                and energy_change < energy_tol):
            converged = True
            break

        current_states = mix_state_blocks(
            current_states, aligned_states, outer_mixing)
        previous_energies = ordered_energies.copy()
        previous_schmidt = schmidt
        next_enrichment = enrichment_strength * (
            enrichment_decay ** (outer_iteration + 1))
        if enrichment is not None and next_enrichment > 1e-12:
            enrichment_blocks, enrichment_norms = enrichment(
                current_states, ordered_energies)
        else:
            enrichment_blocks, enrichment_norms = None, None

    return {
        'energies': final_wave['energies'][final_permutation],
        'state_blocks': final_states,
        'schmidt_data': final_schmidt,
        'problem': final_problem,
        'wave_result': final_wave,
        'state_weights': weights,
        'converged': converged,
        'n_outer_iter': len(history),
        'history': history,
        'root_permutation': final_permutation,
    }
