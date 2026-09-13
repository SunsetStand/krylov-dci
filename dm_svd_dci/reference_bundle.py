"""Deterministic multi-root CASCI reference bundles.

The frozen and self-consistent downfolding paths must consume exactly the same
reference roots, otherwise their accuracy comparison is uncontrolled.  A naive
``nroots = n_target`` CASCI call does not provide that: it can return roots tens
of millihartree above the true spectrum while reporting convergence for every
root, because the default Davidson guess can have identically zero projection on
a target root's spatial irreducible representation, and a projection that starts
at zero stays zero under Krylov expansion.

This module builds a bundle by two independent procedures and requires them to
agree:

``symmetry``
    Solve the lowest roots separately in each irreducible representation of an
    Abelian point group, merge and re-sort by energy.  Coverage of every irrep
    is guaranteed by construction, and the members of a degenerate level each
    come from their own irrep, which makes them canonical rather than an
    arbitrary rotation.

``overshoot``
    Solve for ``n_target + m`` roots and keep the lowest ``n_target``,
    increasing ``m`` until the kept energies stop changing.

The residual norm is recorded but is deliberately NOT used as a correctness
test.  A wrong-root solve returns genuine eigenpairs, so its residuals are
indistinguishable from a correct solve's.  Only margin stability and
cross-procedure agreement discriminate.

See ``docs/theory/n2_root_targeting_mechanism_protocol.md``.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

BUNDLE_FORMAT_VERSION = 1

DEFAULT_SOLVE_CONV_TOL = 1e-12
DEFAULT_SOLVE_MAX_CYCLE = 400
DEFAULT_SOLVE_MAX_SPACE = 30
DEFAULT_DEGENERATE_TOL = 1e-9
DEFAULT_ENERGY_AGREEMENT_TOL = 1e-8
DEFAULT_NONCONVERGENCE_RESIDUAL_TOL = 1e-4
DEFAULT_MAX_MARGIN = 8


class ReferenceBundleError(RuntimeError):
    """Raised when a bundle cannot be built or fails verification."""


def _normalize(vector: np.ndarray) -> np.ndarray:
    flat = np.asarray(vector, dtype=float).reshape(-1)
    norm = np.linalg.norm(flat)
    if norm == 0.0:
        raise ReferenceBundleError('zero-norm CI vector')
    return flat / norm


def group_into_levels(energies: Sequence[float],
                      tolerance: float = DEFAULT_DEGENERATE_TOL
                      ) -> List[List[int]]:
    """Group ascending energies into degenerate levels."""
    levels: List[List[int]] = []
    for index, energy in enumerate(energies):
        if levels and abs(energy - energies[levels[-1][0]]) <= tolerance:
            levels[-1].append(index)
        else:
            levels.append([index])
    return levels


def _git_commit() -> Optional[str]:
    try:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return subprocess.check_output(
            ['git', '-C', root, 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:                                    # pragma: no cover
        return None


def _environment() -> Dict[str, Optional[str]]:
    import pyscf
    import scipy
    return {
        'pyscf': pyscf.__version__,
        'numpy': np.__version__,
        'scipy': scipy.__version__,
        'python': platform.python_version(),
        'commit': _git_commit(),
    }


class _Hamiltonian:
    """One immutable active-space Hamiltonian shared by every solve."""

    def __init__(self, atom: str, basis: str, n_active: int,
                 n_active_elec: Tuple[int, int], point_group: str):
        from pyscf import gto, mcscf, scf, symm
        from pyscf.fci import cistring, direct_spin1

        self.atom = atom
        self.basis = basis
        self.n_active = n_active
        self.n_active_elec = tuple(n_active_elec)
        self.point_group = point_group

        self.mol = gto.M(atom=atom, basis=basis, spin=0,
                         symmetry=point_group, verbose=0)
        self.mf = scf.RHF(self.mol).run(verbose=0)
        if not self.mf.converged:
            raise ReferenceBundleError('RHF did not converge')

        cas = mcscf.CASCI(self.mf, n_active, sum(self.n_active_elec))
        self.h1eff, self.ecore = cas.get_h1eff()
        self.h2eff = cas.get_h2eff()
        self.n_core = (self.mol.nelectron - sum(self.n_active_elec)) // 2
        self.mo_active = np.asarray(
            self.mf.mo_coeff[:, self.n_core:self.n_core + n_active])
        self.orbsym = np.asarray(symm.label_orb_symm(
            self.mol, self.mol.irrep_id, self.mol.symm_orb, self.mo_active))
        self.orbsym_names = [str(name) for name in symm.label_orb_symm(
            self.mol, self.mol.irrep_name, self.mol.symm_orb, self.mo_active)]

        alpha = cistring.gen_strings4orblist(
            range(n_active), self.n_active_elec[0])
        beta = cistring.gen_strings4orblist(
            range(n_active), self.n_active_elec[1])
        self.shape = (len(alpha), len(beta))
        self.dimension = self.shape[0] * self.shape[1]

        self._sigma_solver = direct_spin1.FCI(self.mol)
        self._h2e = self._sigma_solver.absorb_h1e(
            self.h1eff, self.h2eff, n_active, self.n_active_elec, 0.5)

    def sigma(self, vector: np.ndarray) -> np.ndarray:
        reshaped = np.asarray(vector, dtype=float).reshape(self.shape)
        return self._sigma_solver.contract_2e(
            self._h2e, reshaped, self.n_active,
            self.n_active_elec).reshape(-1)

    def residual_norm(self, vector: np.ndarray, total_energy: float) -> float:
        unit = _normalize(vector)
        return float(np.linalg.norm(
            self.sigma(unit) - (total_energy - self.ecore) * unit))

    def spin_square(self, vector: np.ndarray) -> Tuple[float, float]:
        from pyscf.fci import spin_op
        value, multiplicity = spin_op.spin_square(
            _normalize(vector).reshape(self.shape), self.n_active,
            self.n_active_elec)
        return float(value), float(multiplicity)

    def wavefunction_irrep(self, vector: np.ndarray) -> str:
        from pyscf import symm
        from pyscf.fci import addons
        try:
            identifier = addons.guess_wfnsym(
                _normalize(vector).reshape(self.shape), self.n_active,
                self.n_active_elec, self.orbsym)
            return str(symm.irrep_id2name(self.mol.groupname, identifier))
        except Exception as error:                       # pragma: no cover
            return f'UNDETERMINED:{type(error).__name__}'

    def solve(self, nroots: int, wfnsym: Optional[int] = None,
              conv_tol: float = DEFAULT_SOLVE_CONV_TOL
              ) -> Tuple[np.ndarray, List[np.ndarray]]:
        from pyscf.fci import direct_spin1, direct_spin1_symm

        if wfnsym is None:
            solver = direct_spin1.FCI(self.mol)
            kwargs: Dict = {}
        else:
            solver = direct_spin1_symm.FCI(self.mol)
            solver.orbsym = self.orbsym
            solver.wfnsym = wfnsym
            kwargs = {'orbsym': self.orbsym, 'wfnsym': wfnsym}
        solver.nroots = nroots
        solver.conv_tol = conv_tol
        solver.max_cycle = DEFAULT_SOLVE_MAX_CYCLE
        solver.max_space = DEFAULT_SOLVE_MAX_SPACE

        energies, vectors = solver.kernel(
            self.h1eff, self.h2eff, self.n_active, self.n_active_elec,
            nroots=nroots, ecore=self.ecore, verbose=0, **kwargs)
        energies = np.atleast_1d(np.asarray(energies, dtype=float)).reshape(-1)
        roots = ([_normalize(vectors)] if nroots == 1
                 else [_normalize(vector) for vector in vectors[:nroots]])
        order = np.argsort(energies[:len(roots)])
        return energies[order], [roots[index] for index in order]


def _solve_by_symmetry(hamiltonian: _Hamiltonian, n_target: int,
                       roots_per_irrep: int
                       ) -> Tuple[List[float], List[np.ndarray], List[str]]:
    from pyscf import symm

    merged: List[Tuple[float, str, np.ndarray]] = []
    for irrep_id in sorted({int(item) for item in hamiltonian.mol.irrep_id}):
        name = str(symm.irrep_id2name(hamiltonian.mol.groupname, irrep_id))
        try:
            energies, roots = hamiltonian.solve(
                roots_per_irrep, wfnsym=irrep_id)
        except Exception:
            continue
        for energy, root in zip(energies, roots):
            merged.append((float(energy), name, root))
    merged.sort(key=lambda item: item[0])
    if len(merged) < n_target:
        raise ReferenceBundleError(
            'symmetry-resolved solve produced fewer roots than the target')
    kept = merged[:n_target]
    return ([item[0] for item in kept], [item[2] for item in kept],
            [item[1] for item in kept])


def _solve_by_overshoot(hamiltonian: _Hamiltonian, n_target: int,
                        max_margin: int, energy_tol: float,
                        degenerate_tol: float
                        ) -> Tuple[List[float], List[np.ndarray], int, List[Dict]]:
    history: List[Dict] = []
    previous: Optional[np.ndarray] = None
    previous_roots: Optional[List[np.ndarray]] = None
    for margin in range(max_margin + 1):
        nroots = n_target + margin
        if nroots > hamiltonian.dimension:
            break
        energies, roots = hamiltonian.solve(nroots)
        kept = energies[:n_target]
        levels = group_into_levels(energies, degenerate_tol)
        straddles = any(
            min(level) < n_target <= max(level) for level in levels)
        shift = (float(np.max(np.abs(kept - previous)))
                 if previous is not None else None)
        history.append({
            'margin': margin,
            'nroots': nroots,
            'energies': [float(value) for value in kept],
            'shift_from_previous_margin_Ha': shift,
            'degenerate_level_straddles_boundary': bool(straddles),
        })
        if (previous is not None and shift is not None
                and shift < energy_tol and not straddles):
            return ([float(value) for value in kept], roots[:n_target],
                    margin - 1, history)
        previous = kept
        previous_roots = roots[:n_target]
    if previous_roots is None:
        raise ReferenceBundleError('overshoot procedure produced no roots')
    raise ReferenceBundleError(
        f'overshoot did not stabilize within margin {max_margin}')


def build_reference_bundle(
        atom: str,
        basis: str,
        n_active: int,
        n_active_elec: Tuple[int, int],
        target_levels: int,
        point_group: str = 'D2h',
        roots_per_irrep: int = 3,
        max_margin: int = DEFAULT_MAX_MARGIN,
        inventory_roots: int = 10,
        energy_tol: float = DEFAULT_ENERGY_AGREEMENT_TOL,
        degenerate_tol: float = DEFAULT_DEGENERATE_TOL,
        residual_tol: float = DEFAULT_NONCONVERGENCE_RESIDUAL_TOL,
        verbose: bool = True,
) -> Dict:
    """Build a verified multi-root reference bundle.

    The target is the lowest ``target_levels`` energy LEVELS, so a degenerate
    level is retained as a complete block and never divided.
    """
    start = time.perf_counter()
    hamiltonian = _Hamiltonian(atom, basis, n_active, n_active_elec,
                               point_group)
    if verbose:
        print(f'  active space CAS({sum(n_active_elec)}e,{n_active}o), '
              f'dim {hamiltonian.dimension}', flush=True)

    inventory_energies, _ = hamiltonian.solve(
        min(inventory_roots, hamiltonian.dimension))
    levels = group_into_levels(inventory_energies, degenerate_tol)
    if len(levels) < target_levels:
        raise ReferenceBundleError('inventory contains fewer levels than the target')
    target_indices = [index for level in levels[:target_levels]
                      for index in level]
    n_target = len(target_indices)
    if verbose:
        print(f'  target = {n_target} roots over {target_levels} levels: '
              f'{levels[:target_levels]}', flush=True)

    sym_energies, sym_roots, sym_irreps = _solve_by_symmetry(
        hamiltonian, n_target, roots_per_irrep)
    over_energies, _, margin, history = _solve_by_overshoot(
        hamiltonian, n_target, max_margin, energy_tol, degenerate_tol)

    disagreement = float(np.max(np.abs(
        np.asarray(sym_energies) - np.asarray(over_energies))))
    if disagreement >= energy_tol:
        raise ReferenceBundleError(
            'symmetry-resolved and overshoot procedures disagree by '
            f'{disagreement:.3e} Ha, above {energy_tol:.1e}')
    if verbose:
        print(f'  procedures agree to {disagreement:.3e} Ha '
              f'(overshoot margin {margin})', flush=True)

    characters = []
    for index, (energy, root) in enumerate(zip(sym_energies, sym_roots)):
        spin_squared, multiplicity = hamiltonian.spin_square(root)
        residual = hamiltonian.residual_norm(root, energy)
        if residual >= residual_tol:
            raise ReferenceBundleError(
                f'root {index} residual {residual:.3e} indicates non-convergence')
        characters.append({
            'index': index,
            'total_energy': float(energy),
            'excitation_mH': float((energy - sym_energies[0]) * 1000.0),
            'residual_norm': residual,
            'spin_squared': spin_squared,
            'multiplicity': multiplicity,
            'irrep': sym_irreps[index],
            'wavefunction_irrep': hamiltonian.wavefunction_irrep(root),
        })

    target_levels_local = group_into_levels(sym_energies, degenerate_tol)
    vectors = np.asarray(sym_roots, dtype=float)

    bundle = {
        'format_version': BUNDLE_FORMAT_VERSION,
        'environment': _environment(),
        'system': {
            'atom': atom,
            'basis': basis,
            'point_group': point_group,
            'point_group_detected': str(hamiltonian.mol.groupname),
            'n_active': n_active,
            'n_active_elec': list(n_active_elec),
            'n_core': hamiltonian.n_core,
            'determinant_dimension': hamiltonian.dimension,
            'determinant_shape': list(hamiltonian.shape),
            'active_orbital_irreps': hamiltonian.orbsym_names,
            'rhf_energy': float(hamiltonian.mf.e_tot),
            'ecore': float(hamiltonian.ecore),
        },
        'target': {
            'target_levels': target_levels,
            'n_target_roots': n_target,
            'level_grouping': target_levels_local,
            'energies': [float(value) for value in sym_energies],
            'root_character': characters,
        },
        'provenance': {
            'primary_procedure': 'symmetry',
            'cross_check_procedure': 'overshoot',
            'cross_check_energies': over_energies,
            'cross_check_max_disagreement_Ha': disagreement,
            'overshoot_margin': margin,
            'overshoot_history': history,
            'inventory_energies': [float(v) for v in inventory_energies],
            'inventory_levels': levels,
        },
        'thresholds': {
            'energy_agreement_tol': energy_tol,
            'degenerate_tol': degenerate_tol,
            'nonconvergence_residual_tol': residual_tol,
            'solve_conv_tol': DEFAULT_SOLVE_CONV_TOL,
        },
        'wall_time_seconds': float(time.perf_counter() - start),
    }
    arrays = {
        'ci_vectors': vectors,
        'energies': np.asarray(sym_energies, dtype=float),
        'mo_active': hamiltonian.mo_active,
        'h1eff': np.asarray(hamiltonian.h1eff, dtype=float),
        'h2eff': np.asarray(hamiltonian.h2eff, dtype=float),
        'orbsym': np.asarray(hamiltonian.orbsym),
    }
    bundle['checksum'] = compute_checksum(bundle, arrays)
    return {'metadata': bundle, 'arrays': arrays}


def compute_checksum(metadata: Dict, arrays: Dict[str, np.ndarray]) -> str:
    """Deterministic hash over the scientific content of a bundle."""
    digest = hashlib.sha256()
    scientific = {key: value for key, value in metadata.items()
                  if key not in ('checksum', 'wall_time_seconds',
                                 'environment')}
    digest.update(json.dumps(scientific, sort_keys=True,
                             default=str).encode('utf-8'))
    for key in sorted(arrays):
        digest.update(key.encode('utf-8'))
        digest.update(np.ascontiguousarray(
            arrays[key], dtype=np.float64).tobytes())
    return digest.hexdigest()


def save_reference_bundle(bundle: Dict, output_dir: str) -> Dict[str, str]:
    """Write the bundle arrays and a small JSON summary."""
    os.makedirs(output_dir, exist_ok=True)
    array_path = os.path.join(output_dir, 'reference_bundle.npz')
    summary_path = os.path.join(output_dir, 'reference_bundle_summary.json')
    np.savez_compressed(array_path, **bundle['arrays'])
    temporary = summary_path + '.tmp'
    with open(temporary, 'w', encoding='utf-8') as handle:
        json.dump(bundle['metadata'], handle, indent=2, default=str)
    os.replace(temporary, summary_path)
    return {'arrays': array_path, 'summary': summary_path}


def load_reference_bundle(directory: str, verify: bool = True) -> Dict:
    """Load a bundle and verify its checksum."""
    array_path = os.path.join(directory, 'reference_bundle.npz')
    summary_path = os.path.join(directory, 'reference_bundle_summary.json')
    if not os.path.exists(array_path) or not os.path.exists(summary_path):
        raise ReferenceBundleError(f'no reference bundle in {directory}')
    with open(summary_path, encoding='utf-8') as handle:
        metadata = json.load(handle)
    with np.load(array_path) as data:
        arrays = {key: np.asarray(data[key]) for key in data.files}
    if verify:
        expected = metadata.get('checksum')
        actual = compute_checksum(metadata, arrays)
        if expected != actual:
            raise ReferenceBundleError(
                f'bundle checksum mismatch: stored {expected}, computed {actual}')
    return {'metadata': metadata, 'arrays': arrays}


def bundle_ci_vectors(bundle: Dict) -> List[np.ndarray]:
    """Target CI vectors as flat arrays, in ascending energy order."""
    return [np.asarray(row, dtype=float).reshape(-1)
            for row in bundle['arrays']['ci_vectors']]
