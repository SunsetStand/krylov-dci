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

SEED_FAMILIES = ('exact', 'lanczos', 'hf', 'cis', 'trunc', 'selci',
                 'perturbed')
PREFERRED_SEED = 'lanczos'
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


def _lanczos_seed(space: _ActiveSpace, n_states: int, steps: int,
                  partition: Optional[Dict] = None) -> List[np.ndarray]:
    """Early-stopped block Lanczos in the FULL CAS space.

    This is the preferred non-exact seed, and it selects no determinants at
    all.  Every determinant participates; their amplitudes are decided by the
    Hamiltonian rather than by an importance score.

    The determinant-selection seeds below (``cis``, ``trunc``, ``selci``) are
    inherited from the Krylov-dCI line, where the object being chosen was a
    **P space of determinants**.  Here the seed is used to build a **Schmidt
    basis** from state-averaged reduced densities, which depends on the
    entanglement structure of the seed rather than on which determinants it
    contains.  A wavefunction can carry the right determinants and the wrong
    entanglement structure, which is exactly what a CIS seed does, so those
    families are retained only as controls.

    The starting block is the lowest-diagonal determinant of every
    electron-number block, plus the globally lowest determinants needed to
    reach ``n_states``.  That choice is structural, not an importance ranking,
    and it makes block completion unnecessary by construction: every block is
    represented in the starting block, so the Krylov space carries weight
    there from the first step.

    Cost is ``steps`` sigma applications per starting vector, matrix-free, with
    no convergence requirement.
    """
    start: List[int] = []
    if partition is not None:
        for _, block in sorted(partition.items()):
            members = [int(det) for (_, _, det) in block['coeff_map']]
            if members:
                start.append(min(members, key=lambda m: space.hdiag[m]))
    for address in np.argsort(space.hdiag)[:max(n_states, 1)]:
        start.append(int(address))
    start = sorted(set(start))

    basis: List[np.ndarray] = []
    for address in start:
        vector = np.zeros(space.dimension)
        vector[address] = 1.0
        for existing in basis:
            vector -= float(np.dot(existing, vector)) * existing
        norm = float(np.linalg.norm(vector))
        if norm > 1e-12:
            basis.append(vector / norm)

    frontier = list(basis)
    for _ in range(max(steps, 0)):
        new_frontier = []
        for vector in frontier:
            candidate = space.sigma(vector)
            for existing in basis:
                candidate -= float(np.dot(existing, candidate)) * existing
            norm = float(np.linalg.norm(candidate))
            if norm > 1e-10:
                candidate /= norm
                basis.append(candidate)
                new_frontier.append(candidate)
        if not new_frontier:
            break
        frontier = new_frontier

    krylov = np.column_stack(basis)
    projected = krylov.T @ np.column_stack(
        [space.sigma(krylov[:, i]) for i in range(krylov.shape[1])])
    projected = 0.5 * (projected + projected.T)
    _, coefficients = np.linalg.eigh(projected)
    if coefficients.shape[1] < n_states:
        raise InitializerError(
            f'Lanczos space of {coefficients.shape[1]} vectors cannot supply '
            f'{n_states} states')

    vectors = []
    for root in range(n_states):
        vector = krylov @ coefficients[:, root]
        vectors.append(vector / np.linalg.norm(vector))
    return vectors


def _block_members(partition: Dict) -> Dict[int, List[int]]:
    return {int(label): [int(det) for (_, _, det) in block['coeff_map']]
            for label, block in partition.items()}


def _complete_block_support(space: _ActiveSpace, partition: Dict,
                            addresses: Sequence[int], n_states: int,
                            max_passes: int = 4,
                            weight_floor: float = 1e-10
                            ) -> Tuple[np.ndarray, List[int]]:
    """Grow the seed subspace until every block carries real weight.

    A seed with no weight in an electron-number block produces a zero
    state-averaged density there, the block is truncated to rank zero, and the
    reconstructed coefficients inherit the same empty support.  Deletion is
    irreversible, so the outer loop is trapped at a fixed point that omits the
    block.

    Selecting the block's lowest-diagonal determinant is not sufficient: that
    determinant is often symmetry-decoupled from the rest of the seed
    subspace, so it becomes its own eigenvector and the low-lying roots keep
    zero amplitude on it.  Determinants are therefore chosen by their coupling
    to the current seed, which is the selected-CI criterion restricted to one
    block, and the process is repeated until the block weight is actually
    non-zero or the pass budget is spent.
    """
    members = _block_members(partition)
    addresses = np.asarray(sorted({int(a) for a in addresses}), dtype=int)
    added: List[int] = []

    for _ in range(max_passes):
        vectors = _diagonalize_in_subspace(space, addresses, n_states)
        combined = np.zeros(space.dimension)
        for vector in vectors:
            combined += np.abs(vector)
        coupling = np.abs(space.sigma(combined))

        deficient = []
        for label, block_addresses in sorted(members.items()):
            if not block_addresses:
                continue
            weight = sum(float(np.sum(vector[block_addresses] ** 2))
                         for vector in vectors)
            if weight <= weight_floor:
                deficient.append((label, block_addresses))
        if not deficient:
            break

        covered = set(int(a) for a in addresses)
        new = []
        for _, block_addresses in deficient:
            candidates = [a for a in block_addresses if a not in covered]
            if not candidates:
                continue
            # Largest coupling to the current seed, falling back to the lowest
            # diagonal element when every coupling vanishes by symmetry.
            best = max(candidates, key=lambda a: (coupling[a], -space.hdiag[a]))
            new.append(int(best))
        if not new:
            break
        added.extend(new)
        addresses = np.asarray(
            sorted(set(int(a) for a in addresses) | set(new)), dtype=int)

    return addresses, added


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
        lanczos_steps: int = 3,
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

    if seed == 'lanczos':
        vectors = _lanczos_seed(space, n_states, lanczos_steps, partition)
        provenance.update({
            'lanczos_steps': int(lanczos_steps),
            'selects_determinants': False,
            'block_completion_applied': False,
            'subspace_dimension': int(space.dimension),
        })
        if verbose:
            print(f'  seed: lanczos, {lanczos_steps} steps in the full CAS '
                  f'space, no determinant selection', flush=True)
        return vectors, provenance

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
        addresses, completion = _complete_block_support(
            space, partition, addresses, n_states)
    provenance['selects_determinants'] = True
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
