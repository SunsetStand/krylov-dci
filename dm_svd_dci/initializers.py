"""Non-exact initial CI coefficients for the state-averaged outer loop.

The production self-consistent path must not read exact CASCI or FCI
coefficients.  Exact CI is an offline validation oracle only: it may enter an
evaluator that reports errors, never the initializer, the root selector, the
Schmidt builder or the solver.  See
``docs/theory/iterative_ci_feasibility_protocol.md``, hypothesis H5.

Every seed here builds a small subspace of determinants, diagonalizes the
Hamiltonian inside it, and embeds the resulting vectors back into the full CAS
determinant ordering.  None of them calls an exact CI kernel.

``exact`` is deliberately included and deliberately labelled: it is the
upper-bound control the protocol requires, and it is the one seed that is not a
production path.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

SEED_FAMILIES = ('exact', 'hf', 'cis', 'trunc', 'selci', 'perturbed')
NON_EXACT_SEEDS = tuple(name for name in SEED_FAMILIES if name != 'exact')


class InitializerError(RuntimeError):
    """Raised when a seed cannot produce the requested number of states."""


def _excitation_rank_map(n_active: int,
                         n_active_elec: Tuple[int, int]) -> np.ndarray:
    """Excitation rank of every determinant relative to the HF determinant."""
    from pyscf.fci import cistring

    alpha = cistring.gen_strings4orblist(range(n_active), n_active_elec[0])
    beta = cistring.gen_strings4orblist(range(n_active), n_active_elec[1])
    hf_alpha = (1 << n_active_elec[0]) - 1
    hf_beta = (1 << n_active_elec[1]) - 1
    ranks = np.empty((len(alpha), len(beta)), dtype=np.int16)
    for i, a_str in enumerate(alpha):
        a_holes = int(hf_alpha & ~int(a_str)).bit_count()
        for j, b_str in enumerate(beta):
            ranks[i, j] = a_holes + int(hf_beta & ~int(b_str)).bit_count()
    return ranks.reshape(-1)


class _ActiveSpace:
    """Sigma products and diagonal for one fixed active-space Hamiltonian."""

    def __init__(self, sys_data: Dict):
        from pyscf.fci import cistring, direct_spin1

        self.n_active = sys_data['n_active']
        self.n_elec = tuple(sys_data['n_active_elec'])
        self.solver = direct_spin1.FCI(sys_data['mol'])
        self.h2e = self.solver.absorb_h1e(
            sys_data['h1eff'], sys_data['h2eff'],
            self.n_active, self.n_elec, 0.5)
        n_alpha = cistring.num_strings(self.n_active, self.n_elec[0])
        n_beta = cistring.num_strings(self.n_active, self.n_elec[1])
        self.shape = (n_alpha, n_beta)
        self.dimension = n_alpha * n_beta
        self.hdiag = self.solver.make_hdiag(
            sys_data['h1eff'], sys_data['h2eff'],
            self.n_active, self.n_elec).reshape(-1)

    def sigma(self, vector: np.ndarray) -> np.ndarray:
        return self.solver.contract_2e(
            self.h2e, np.asarray(vector, dtype=float).reshape(self.shape),
            self.n_active, self.n_elec).reshape(-1)


def _diagonalize_in_subspace(space: _ActiveSpace, addresses: Sequence[int],
                             n_states: int) -> List[np.ndarray]:
    """Diagonalize H restricted to ``addresses`` and embed the lowest roots."""
    addresses = np.asarray(sorted(set(int(a) for a in addresses)), dtype=int)
    if len(addresses) < n_states:
        raise InitializerError(
            f'subspace of {len(addresses)} determinants cannot supply '
            f'{n_states} states')
    matrix = np.empty((len(addresses), len(addresses)))
    unit = np.zeros(space.dimension)
    for column, address in enumerate(addresses):
        unit[:] = 0.0
        unit[address] = 1.0
        matrix[:, column] = space.sigma(unit)[addresses]
    matrix = 0.5 * (matrix + matrix.T)
    _, coefficients = np.linalg.eigh(matrix)

    vectors = []
    for root in range(n_states):
        vector = np.zeros(space.dimension)
        vector[addresses] = coefficients[:, root]
        norm = np.linalg.norm(vector)
        if norm == 0.0:
            raise InitializerError(f'seed root {root} has zero norm')
        vectors.append(vector / norm)
    return vectors


def _block_completion_addresses(partition: Dict, space: _ActiveSpace,
                                addresses: Sequence[int]) -> List[int]:
    """Determinants needed so that every electron-number block is represented.

    A seed with no determinant in a block produces a zero state-averaged
    density there, the block is truncated to rank zero, and the reconstructed
    coefficients inherit the same empty support.  Deletion is irreversible, so
    the outer loop is then trapped at a fixed point that omits the block.
    Adding the lowest-diagonal determinant of each unrepresented block makes
    the seed admissible without changing the solver.
    """
    covered = {int(a) for a in addresses}
    extra: List[int] = []
    for _, block in sorted(partition.items()):
        members = [int(det) for (_, _, det) in block['coeff_map']]
        if not members or any(member in covered for member in members):
            continue
        extra.append(min(members, key=lambda member: space.hdiag[member]))
    return extra


def _lowest_diagonal_addresses(space: _ActiveSpace, count: int) -> np.ndarray:
    return np.argsort(space.hdiag)[:max(count, 1)]


def _selected_ci_addresses(space: _ActiveSpace, count: int) -> np.ndarray:
    """Cheap selection by coupling to the lowest-diagonal determinant."""
    seed_address = int(np.argmin(space.hdiag))
    unit = np.zeros(space.dimension)
    unit[seed_address] = 1.0
    coupling = np.abs(space.sigma(unit))
    gap = np.abs(space.hdiag - space.hdiag[seed_address])
    score = coupling ** 2 / np.maximum(gap, 1e-8)
    score[seed_address] = np.inf
    return np.argsort(-score)[:max(count, 1)]


def build_initial_states(
        sys_data: Dict,
        n_states: int,
        seed: str = 'exact',
        subspace_size: int = 64,
        perturbation_scale: float = 0.1,
        random_seed: int = 0,
        partition: Optional[Dict] = None,
        complete_blocks: bool = True,
        verbose: bool = True,
) -> Tuple[List[np.ndarray], Dict]:
    """Build ``n_states`` flat CI vectors in the CAS determinant ordering.

    Returns the vectors and a provenance record.  With ``seed='exact'`` this
    reads exact CASCI coefficients and is a control, not a production path;
    every other value touches no exact CI kernel.
    """
    if seed not in SEED_FAMILIES:
        raise InitializerError(
            f'unknown seed {seed!r}; expected one of {SEED_FAMILIES}')

    provenance = {
        'seed': seed,
        'n_states': int(n_states),
        'reads_exact_ci': seed == 'exact',
        'subspace_size': int(subspace_size),
    }

    if seed == 'exact':
        from pyscf import mcscf
        cas = mcscf.CASCI(sys_data['mf'], sys_data['n_active'],
                          sum(sys_data['n_active_elec']))
        cas.frozen = sys_data['n_core']
        cas.fcisolver.nroots = n_states
        cas.kernel()
        if n_states == 1:
            vectors = [np.asarray(cas.ci).reshape(-1)]
        else:
            vectors = [np.asarray(cas.ci[k]).reshape(-1)
                       for k in range(n_states)]
        provenance['note'] = 'upper-bound control; not a production path'
        provenance['block_completion_applied'] = False
        if verbose:
            print('  seed: exact CASCI (control, reads exact CI)', flush=True)
        return [v / np.linalg.norm(v) for v in vectors], provenance

    space = _ActiveSpace(sys_data)
    size = max(subspace_size, 4 * n_states)

    if seed == 'hf':
        addresses = _lowest_diagonal_addresses(space, size)
    elif seed == 'cis':
        ranks = _excitation_rank_map(space.n_active, space.n_elec)
        addresses = np.flatnonzero(ranks <= 1)
    elif seed == 'trunc':
        ranks = _excitation_rank_map(space.n_active, space.n_elec)
        addresses = np.flatnonzero(ranks <= 2)
    elif seed == 'selci':
        addresses = _selected_ci_addresses(space, size)
    elif seed == 'perturbed':
        ranks = _excitation_rank_map(space.n_active, space.n_elec)
        addresses = np.flatnonzero(ranks <= 1)
    else:                                                    # pragma: no cover
        raise InitializerError(f'unhandled seed {seed!r}')

    completion: List[int] = []
    if complete_blocks and partition is not None:
        completion = _block_completion_addresses(partition, space, addresses)
        if completion:
            addresses = np.concatenate(
                [np.asarray(addresses, dtype=int),
                 np.asarray(completion, dtype=int)])
    provenance['block_completion_addresses'] = [int(a) for a in completion]
    provenance['block_completion_applied'] = bool(completion)
    provenance['subspace_dimension'] = int(len(set(int(a) for a in addresses)))
    vectors = _diagonalize_in_subspace(space, addresses, n_states)

    if seed == 'perturbed':
        generator = np.random.default_rng(random_seed)
        support = np.asarray(sorted(set(int(a) for a in addresses)), dtype=int)
        perturbed = []
        for vector in vectors:
            noise = np.zeros_like(vector)
            # Perturb only inside the seed support, so particle number, spin
            # projection and spatial symmetry of the subspace are preserved.
            noise[support] = generator.standard_normal(len(support))
            noise *= perturbation_scale * np.linalg.norm(vector) / np.linalg.norm(noise)
            candidate = vector + noise
            perturbed.append(candidate / np.linalg.norm(candidate))
        vectors = _symmetric_orthonormalize(perturbed)
        provenance['perturbation_scale'] = float(perturbation_scale)
        provenance['random_seed'] = int(random_seed)

    if verbose:
        note = (f', +{len(completion)} block-completing determinants'
                if completion else '')
        print(f'  seed: {seed}, subspace '
              f"{provenance['subspace_dimension']} determinants{note}, "
              f'{n_states} states (no exact CI read)', flush=True)
    return vectors, provenance


def _symmetric_orthonormalize(vectors: List[np.ndarray]) -> List[np.ndarray]:
    matrix = np.column_stack(vectors)
    overlap = matrix.T @ matrix
    eigenvalues, eigenvectors = np.linalg.eigh(overlap)
    eigenvalues = np.maximum(eigenvalues, 1e-14)
    inverse_root = eigenvectors @ np.diag(
        eigenvalues ** -0.5) @ eigenvectors.T
    orthonormal = matrix @ inverse_root
    return [orthonormal[:, k] for k in range(orthonormal.shape[1])]


def seed_block_support(state_blocks: List[Dict[int, np.ndarray]],
                       tolerance: float = 1e-12) -> Dict:
    """Per-electron-number-block weight carried by a set of seed states.

    A block with zero seed weight is deleted by the Schmidt decomposition, and
    the outer loop cannot recover it: the reconstructed coefficients inherit
    the same empty support, so the iteration is trapped at a fixed point that
    omits that block.  Recording this makes the failure visible instead of
    silent.  See docs/development/seed_block_support_finding.md.
    """
    labels = sorted({label for state in state_blocks for label in state})
    weights = {}
    for label in labels:
        total = 0.0
        for state in state_blocks:
            block = state.get(label)
            if block is not None:
                total += float(np.sum(np.asarray(block) ** 2))
        weights[int(label)] = total
    empty = [label for label, weight in weights.items() if weight <= tolerance]
    return {
        'block_weights': weights,
        'empty_blocks': empty,
        'has_empty_block': bool(empty),
        'tolerance': float(tolerance),
    }


def evaluate_reference_energies(sys_data: Dict, n_states: int) -> Optional[np.ndarray]:
    """Exact CASCI energies, for error reporting only.

    This is the evaluator boundary.  Nothing in the production path may consume
    its output, and it may be skipped entirely.
    """
    from pyscf import mcscf
    cas = mcscf.CASCI(sys_data['mf'], sys_data['n_active'],
                      sum(sys_data['n_active_elec']))
    cas.frozen = sys_data['n_core']
    cas.fcisolver.nroots = n_states
    cas.kernel()
    return np.atleast_1d(np.asarray(cas.e_tot, dtype=float)).reshape(-1)[:n_states]
