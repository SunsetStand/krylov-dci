#!/usr/bin/env python3
"""Determine the mechanism that excludes low N2 Ms=0 roots from a CASCI solve.

Implements docs/theory/n2_root_targeting_mechanism_protocol.md.

This script performs reference-level FCI calculations only.  It does not
construct a Schmidt basis and it does not call the downfolded wave-operator
solver.  Every acceptance test is computed independently of the eigensolver's
own convergence flag, because that flag has been observed to report success
while returning roots tens of millihartree above the true spectrum.
"""

import argparse
import json
import os
import platform
import time
from itertools import permutations

import numpy as np

GEOMETRY = 'N 0 0 0; N 0 0 1.098'
BASIS = 'cc-pVDZ'
POINT_GROUP = 'D2h'
N_CORE = 2
N_ACTIVE_ELEC = (5, 5)
TARGET_LEVELS = 3
INVENTORY_ROOTS = 10
PROTOCOL = 'docs/theory/n2_root_targeting_mechanism_protocol.md'

ACTIVE_SPACES = {
    'legacy': 10,
    'corrected': 9,
}

# The residual norm certifies that a returned pair is an eigenpair.  It does
# NOT certify that it is one of the LOWEST eigenpairs: a solve that homes on
# roots tens of millihartree too high still returns residuals in the 1e-6
# range, indistinguishable from a correct solve.  The residual test is
# therefore a non-convergence detector only, and the operative correctness
# test is margin stability.  Measured separation: genuinely unconverged
# vectors give ~1e-3, converged ones ~1e-7 to 1e-5.
NONCONVERGENCE_RESIDUAL_TOL = 1e-4
DEGENERATE_TOL = 1e-9
ENERGY_AGREEMENT_TOL = 1e-8
SOLVE_CONV_TOL = 1e-12
SOLVE_MAX_CYCLE = 400
SOLVE_MAX_SPACE = 30


def parse_args():
    parser = argparse.ArgumentParser(
        description='Diagnose the N2 CASCI root-targeting mechanism')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--active-space', choices=sorted(ACTIVE_SPACES),
                        default='corrected')
    parser.add_argument('--max-margin', type=int, default=6,
                        help='largest overshoot margin m to scan')
    parser.add_argument('--replicas', type=int, default=3,
                        help='independent rebuilds used for the stability test')
    return parser.parse_args()


def serializable(value):
    if isinstance(value, dict):
        return {str(key): serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def write_report(output_dir, report):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'n2_root_targeting_mechanism.json')
    temporary = path + '.tmp'
    with open(temporary, 'w', encoding='utf-8') as handle:
        json.dump(serializable(report), handle, indent=2)
    os.replace(temporary, path)
    return path


def flags_to_list(value):
    if value is None:
        return None
    array = np.atleast_1d(np.asarray(value))
    return [bool(item) for item in array.reshape(-1)]


def normalize(vector):
    flat = np.asarray(vector, dtype=float).reshape(-1)
    norm = np.linalg.norm(flat)
    if norm == 0.0:
        raise ValueError('zero-norm CI vector')
    return flat / norm


def excitation_rank(alpha_string, beta_string, hf_alpha, hf_beta):
    alpha_holes = int(hf_alpha & ~int(alpha_string)).bit_count()
    beta_holes = int(hf_beta & ~int(beta_string)).bit_count()
    return alpha_holes + beta_holes


def make_rank_map(alpha_strings, beta_strings, hf_alpha, hf_beta):
    ranks = np.empty((len(alpha_strings), len(beta_strings)), dtype=np.int8)
    for alpha_index, alpha_string in enumerate(alpha_strings):
        for beta_index, beta_string in enumerate(beta_strings):
            ranks[alpha_index, beta_index] = excitation_rank(
                int(alpha_string), int(beta_string), hf_alpha, hf_beta)
    return ranks.reshape(-1)


class ReferenceSystem:
    """One immutable Hamiltonian shared by every calculation in the report."""

    def __init__(self, n_active):
        from pyscf import gto, mcscf, scf, symm

        self.n_active = n_active
        self.n_elec = N_ACTIVE_ELEC
        self.mol = gto.M(atom=GEOMETRY, basis=BASIS, spin=0,
                         symmetry=POINT_GROUP, verbose=0)
        self.mf = scf.RHF(self.mol).run(verbose=0)
        cas = mcscf.CASCI(self.mf, n_active, sum(N_ACTIVE_ELEC))
        self.h1eff, self.ecore = cas.get_h1eff()
        self.h2eff = cas.get_h2eff()
        self.n_core = (self.mol.nelectron - sum(N_ACTIVE_ELEC)) // 2
        active = self.mf.mo_coeff[:, self.n_core:self.n_core + n_active]
        self.orbsym = np.asarray(symm.label_orb_symm(
            self.mol, self.mol.irrep_id, self.mol.symm_orb, active))
        self.orbsym_names = [str(name) for name in symm.label_orb_symm(
            self.mol, self.mol.irrep_name, self.mol.symm_orb, active)]

        from pyscf.fci import cistring
        self.alpha_strings = cistring.gen_strings4orblist(
            range(n_active), N_ACTIVE_ELEC[0])
        self.beta_strings = cistring.gen_strings4orblist(
            range(n_active), N_ACTIVE_ELEC[1])
        self.shape = (len(self.alpha_strings), len(self.beta_strings))
        self.dimension = self.shape[0] * self.shape[1]
        hf_alpha = (1 << N_ACTIVE_ELEC[0]) - 1
        hf_beta = (1 << N_ACTIVE_ELEC[1]) - 1
        self.rank_map = make_rank_map(
            self.alpha_strings, self.beta_strings, hf_alpha, hf_beta)

        from pyscf.fci import direct_spin1
        self._sigma_solver = direct_spin1.FCI(self.mol)
        self.h2e = self._sigma_solver.absorb_h1e(
            self.h1eff, self.h2eff, n_active, N_ACTIVE_ELEC, 0.5)

    def sigma(self, vector):
        reshaped = np.asarray(vector, dtype=float).reshape(self.shape)
        return self._sigma_solver.contract_2e(
            self.h2e, reshaped, self.n_active, N_ACTIVE_ELEC).reshape(-1)

    def residual_norm(self, vector, total_energy):
        unit = normalize(vector)
        active_energy = total_energy - self.ecore
        return float(np.linalg.norm(self.sigma(unit) - active_energy * unit))

    def energy_variance(self, vector):
        unit = normalize(vector)
        hv = self.sigma(unit)
        mean = float(np.dot(unit, hv))
        return float(np.dot(hv, hv) - mean * mean)

    def spin_square(self, vector):
        from pyscf.fci import spin_op
        unit = normalize(vector).reshape(self.shape)
        value, multiplicity = spin_op.spin_square(
            unit, self.n_active, N_ACTIVE_ELEC)
        return float(value), float(multiplicity)

    def wavefunction_irrep(self, vector):
        from pyscf import symm
        from pyscf.fci import addons
        unit = normalize(vector).reshape(self.shape)
        try:
            identifier = addons.guess_wfnsym(
                unit, self.n_active, N_ACTIVE_ELEC, self.orbsym)
            return str(symm.irrep_id2name(self.mol.groupname, identifier))
        except Exception as error:                      # pragma: no cover
            return f'UNDETERMINED:{type(error).__name__}'

    def rank_weights(self, vector):
        unit = normalize(vector)
        maximum = int(np.max(self.rank_map))
        weights = np.bincount(self.rank_map, weights=unit ** 2,
                              minlength=maximum + 1)
        return [float(item) for item in weights]

    def characterize(self, vector, total_energy):
        spin_squared, multiplicity = self.spin_square(vector)
        return {
            'total_energy': float(total_energy),
            'residual_norm': self.residual_norm(vector, total_energy),
            'energy_variance': self.energy_variance(vector),
            'spin_squared': spin_squared,
            'multiplicity': multiplicity,
            'wavefunction_irrep': self.wavefunction_irrep(vector),
            'rank_weights': self.rank_weights(vector),
        }


def group_into_levels(energies, tolerance=DEGENERATE_TOL):
    """Group ascending energies into degenerate levels."""
    levels = []
    for index, energy in enumerate(energies):
        if levels and abs(energy - energies[levels[-1][0]]) <= tolerance:
            levels[-1].append(index)
        else:
            levels.append([index])
    return levels


def solve(system, label, nroots, ci0=None, wfnsym=None, fix_spin=None):
    """One FCI solve on the shared Hamiltonian, with independent acceptance."""
    from pyscf.fci import addons, direct_spin1, direct_spin1_symm

    start = time.perf_counter()
    if wfnsym is None:
        solver = direct_spin1.FCI(system.mol)
        kwargs = {}
    else:
        solver = direct_spin1_symm.FCI(system.mol)
        solver.orbsym = system.orbsym
        solver.wfnsym = wfnsym
        kwargs = {'orbsym': system.orbsym, 'wfnsym': wfnsym}
    if fix_spin is not None:
        addons.fix_spin_(solver, shift=0.5, ss=fix_spin)
    solver.nroots = nroots
    solver.conv_tol = SOLVE_CONV_TOL
    solver.max_cycle = SOLVE_MAX_CYCLE
    solver.max_space = SOLVE_MAX_SPACE

    energies, vectors = solver.kernel(
        system.h1eff, system.h2eff, system.n_active, N_ACTIVE_ELEC,
        ci0=ci0, nroots=nroots, ecore=system.ecore, verbose=0, **kwargs)
    energies = np.atleast_1d(np.asarray(energies, dtype=float)).reshape(-1)
    if nroots == 1:
        roots = [normalize(vectors)]
    else:
        roots = [normalize(vector) for vector in vectors[:nroots]]

    order = np.argsort(energies[:len(roots)])
    energies = energies[order]
    roots = [roots[index] for index in order]

    characters = [system.characterize(root, energy)
                  for root, energy in zip(roots, energies)]
    return {
        'label': label,
        'nroots_requested': int(nroots),
        'energies': [float(item) for item in energies],
        'solver_converged': flags_to_list(solver.converged),
        'conv_tol': float(solver.conv_tol),
        'max_space': int(solver.max_space),
        'max_cycle': int(solver.max_cycle),
        'pspace_size': int(solver.pspace_size),
        'max_residual_norm': max(item['residual_norm'] for item in characters),
        'residual_test_passed': all(
            item['residual_norm'] < NONCONVERGENCE_RESIDUAL_TOL
            for item in characters),
        'root_character': characters,
        'wall_time_seconds': float(time.perf_counter() - start),
    }, roots


def subspace_principal_angles(block_a, block_b):
    """Cosines of principal angles between two sets of vectors."""
    qa, _ = np.linalg.qr(np.column_stack(block_a))
    qb, _ = np.linalg.qr(np.column_stack(block_b))
    singular = np.linalg.svd(qa.T.conj() @ qb, compute_uv=False)
    return [float(min(1.0, max(0.0, value))) for value in singular]


def compare_to_inventory(label, energies, roots, inventory_energies,
                         inventory_roots, inventory_levels):
    """Match by energy first, then verify degenerate blocks as subspaces."""
    overlap = np.abs(np.asarray([
        [float(np.dot(root, reference)) for reference in inventory_roots]
        for root in roots]))
    count = len(roots)
    assignment = max(
        permutations(range(len(inventory_roots)), count),
        key=lambda candidate: sum(
            overlap[index, candidate[index]] for index in range(count)))

    block_angles = {}
    for level_index, members in enumerate(inventory_levels):
        if len(members) < 2 or max(members) >= count:
            continue
        block_angles[str(level_index)] = subspace_principal_angles(
            [roots[index] for index in members],
            [inventory_roots[index] for index in members])

    energy_differences = [
        float(energies[index] - inventory_energies[index])
        for index in range(min(count, len(inventory_energies)))]
    return {
        'label': label,
        'inventory_assignment': list(assignment),
        'matched_overlaps': [float(overlap[index, assignment[index]])
                             for index in range(count)],
        'energy_differences_Ha': energy_differences,
        'max_abs_energy_difference_Ha': float(
            max(abs(value) for value in energy_differences))
        if energy_differences else 0.0,
        'degenerate_block_principal_angles': block_angles,
    }


def guess_projections(system, guesses, inventory_roots):
    normalized = [normalize(guess) for guess in guesses]
    basis, _ = np.linalg.qr(np.column_stack(normalized))
    return {
        'n_vectors': len(normalized),
        'inventory_root_span_weights': [
            float(np.linalg.norm(basis.T.conj() @ root) ** 2)
            for root in inventory_roots],
        'per_vector_overlaps': [
            [float(abs(np.dot(guess, root))) for root in inventory_roots]
            for guess in normalized],
    }, normalized


def build_cis_guesses(system, count):
    """Diagonalize H in the RHF-plus-all-singles space."""
    addresses = np.flatnonzero(system.rank_map <= 1)
    matrix = np.empty((len(addresses), len(addresses)))
    for column, address in enumerate(addresses):
        unit = np.zeros(system.dimension)
        unit[address] = 1.0
        matrix[:, column] = system.sigma(unit)[addresses]
    matrix = 0.5 * (matrix + matrix.T)
    values, vectors = np.linalg.eigh(matrix)

    guesses = []
    records = []
    for root in range(min(count, len(values))):
        vector = np.zeros(system.dimension)
        vector[addresses] = vectors[:, root]
        vector = normalize(vector)
        guesses.append(vector)
        spin_squared, multiplicity = system.spin_square(vector)
        records.append({
            'index': int(root),
            'total_energy': float(values[root] + system.ecore),
            'spin_squared': spin_squared,
            'multiplicity': multiplicity,
            'wavefunction_irrep': system.wavefunction_irrep(vector),
        })
    return {'dimension': len(addresses), 'roots': records}, guesses


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    n_active = ACTIVE_SPACES[args.active_space]

    import pyscf
    import scipy

    report = {
        'protocol': PROTOCOL,
        'status': 'RUNNING',
        'classification': None,
        'environment': {
            'pyscf': pyscf.__version__,
            'numpy': np.__version__,
            'scipy': scipy.__version__,
            'python': platform.python_version(),
            'platform': platform.platform(),
        },
        'system': {
            'geometry': GEOMETRY,
            'basis': BASIS,
            'point_group': POINT_GROUP,
            'active_space': args.active_space,
            'n_active': n_active,
            'n_active_elec': list(N_ACTIVE_ELEC),
            'n_core': N_CORE,
            'target_levels': TARGET_LEVELS,
        },
        'thresholds': {
            'nonconvergence_residual_tol': NONCONVERGENCE_RESIDUAL_TOL,
            'solve_conv_tol': SOLVE_CONV_TOL,
            'degenerate_tol': DEGENERATE_TOL,
            'energy_agreement_tol': ENERGY_AGREEMENT_TOL,
        },
    }
    write_report(args.output_dir, report)

    total_start = time.perf_counter()
    print(f'Building CAS({sum(N_ACTIVE_ELEC)}e,{n_active}o) reference...',
          flush=True)
    system = ReferenceSystem(n_active)
    report['system'].update({
        'rhf_energy': float(system.mf.e_tot),
        'rhf_converged': bool(system.mf.converged),
        'ecore': float(system.ecore),
        'determinant_dimension': int(system.dimension),
        'active_orbital_irreps': system.orbsym_names,
        'point_group_detected': str(system.mol.groupname),
    })
    write_report(args.output_dir, report)

    print('Running inventory solve...', flush=True)
    inventory, inventory_roots = solve(system, 'inventory', INVENTORY_ROOTS)
    inventory_energies = inventory['energies']
    levels = group_into_levels(inventory_energies)
    target_roots = [index for level in levels[:TARGET_LEVELS]
                    for index in level]
    n_target = len(target_roots)
    inventory['levels'] = levels
    report['inventory'] = inventory
    report['target'] = {
        'levels': levels[:TARGET_LEVELS],
        'root_indices': target_roots,
        'n_target_roots': n_target,
        'energies': [inventory_energies[index] for index in target_roots],
    }
    write_report(args.output_dir, report)
    print(f'  target = {n_target} roots covering {TARGET_LEVELS} levels: '
          f'{levels[:TARGET_LEVELS]}', flush=True)

    families = {}

    print(f'Running default solve at nroots={n_target}...', flush=True)
    default_run, default_roots = solve(system, 'default', n_target)
    families['default'] = {
        'run': default_run,
        'comparison': compare_to_inventory(
            'default', default_run['energies'], default_roots,
            inventory_energies, inventory_roots, levels),
    }
    write_report(args.output_dir, report)

    print('Scanning overshoot margins...', flush=True)
    overshoot = {}
    for margin in range(args.max_margin + 1):
        nroots = n_target + margin
        if nroots > INVENTORY_ROOTS:
            break
        run, roots = solve(system, f'overshoot_m{margin}', nroots)
        comparison = compare_to_inventory(
            f'overshoot_m{margin}', run['energies'][:n_target],
            roots[:n_target], inventory_energies, inventory_roots, levels)
        boundary_splits = any(
            min(level) < n_target <= max(level) for level in levels)
        overshoot[str(margin)] = {
            'nroots': nroots,
            'max_residual_norm': run['max_residual_norm'],
            'residual_test_passed': run['residual_test_passed'],
            'solver_converged': run['solver_converged'],
            'max_abs_energy_difference_Ha':
                comparison['max_abs_energy_difference_Ha'],
            'lowest_target_energies': run['energies'][:n_target],
            'recovers_target': bool(
                comparison['max_abs_energy_difference_Ha']
                < ENERGY_AGREEMENT_TOL and run['residual_test_passed']),
            'degenerate_level_split_at_boundary': bool(boundary_splits),
            'wall_time_seconds': run['wall_time_seconds'],
        }
        print(f"  m={margin} nroots={nroots} "
              f"dE_max={comparison['max_abs_energy_difference_Ha']:.3e} "
              f"resid={run['max_residual_norm']:.2e} "
              f"recovers={overshoot[str(margin)]['recovers_target']}",
              flush=True)
    margins = sorted(overshoot, key=int)
    for earlier, later in zip(margins, margins[1:]):
        shift = max(
            abs(a - b) for a, b in zip(
                overshoot[earlier]['lowest_target_energies'],
                overshoot[later]['lowest_target_energies']))
        overshoot[earlier]['energy_shift_to_next_margin_Ha'] = float(shift)
        overshoot[earlier]['stable_against_next_margin'] = bool(
            shift < ENERGY_AGREEMENT_TOL)
    if margins:
        overshoot[margins[-1]]['energy_shift_to_next_margin_Ha'] = None
        overshoot[margins[-1]]['stable_against_next_margin'] = None
    families['overshoot'] = overshoot

    print('Inspecting default initial guesses...', flush=True)
    from pyscf.fci import direct_spin1
    guess_solver = direct_spin1.FCI(system.mol)
    hdiag = guess_solver.make_hdiag(
        system.h1eff, system.h2eff, system.n_active, N_ACTIVE_ELEC).reshape(-1)
    guess_records = {}
    normalized_guesses = {}
    for count in sorted({n_target, INVENTORY_ROOTS}):
        raw = guess_solver.get_init_guess(
            system.n_active, N_ACTIVE_ELEC, count, hdiag)
        record, normalized = guess_projections(system, raw, inventory_roots)
        guess_records[str(count)] = record
        normalized_guesses[count] = normalized
    report['default_initial_guesses'] = guess_records

    print('Running symmetry-resolved solves...', flush=True)
    symmetry_resolved = {}
    merged = []
    for irrep_id in sorted({int(item) for item in system.mol.irrep_id}):
        from pyscf import symm
        name = str(symm.irrep_id2name(system.mol.groupname, irrep_id))
        try:
            run, roots = solve(system, f'sym_{name}', 3, wfnsym=irrep_id)
        except Exception as error:
            symmetry_resolved[name] = {'skipped': f'{type(error).__name__}: {error}'}
            continue
        symmetry_resolved[name] = {
            'energies': run['energies'],
            'max_residual_norm': run['max_residual_norm'],
            'residual_test_passed': run['residual_test_passed'],
            'root_character': run['root_character'],
        }
        for energy, root in zip(run['energies'], roots):
            merged.append((energy, name, root))
    merged.sort(key=lambda item: item[0])
    merged_energies = [item[0] for item in merged[:n_target]]
    merged_roots = [item[2] for item in merged[:n_target]]
    symmetry_comparison = compare_to_inventory(
        'symmetry_resolved', merged_energies, merged_roots,
        inventory_energies, inventory_roots, levels) if merged_roots else None
    families['symmetry_resolved'] = {
        'per_irrep': symmetry_resolved,
        'merged_energies': merged_energies,
        'merged_irreps': [item[1] for item in merged[:n_target]],
        'comparison': symmetry_comparison,
        'recovers_target': bool(
            symmetry_comparison is not None
            and symmetry_comparison['max_abs_energy_difference_Ha']
            < ENERGY_AGREEMENT_TOL),
    }

    print('Running spin-resolved solves...', flush=True)
    spin_resolved = {}
    spin_merged = []
    for name, ss in (('singlet', 0), ('triplet', 2)):
        try:
            run, roots = solve(system, f'spin_{name}', 4, fix_spin=ss)
        except Exception as error:
            spin_resolved[name] = {'skipped': f'{type(error).__name__}: {error}'}
            continue
        keep = [index for index, character in enumerate(run['root_character'])
                if abs(character['spin_squared'] - ss) < 1e-3]
        spin_resolved[name] = {
            'energies': run['energies'],
            'retained_indices': keep,
            'max_residual_norm': run['max_residual_norm'],
            'root_character': run['root_character'],
        }
        for index in keep:
            spin_merged.append((run['energies'][index], name, roots[index]))
    spin_merged.sort(key=lambda item: item[0])
    spin_energies = [item[0] for item in spin_merged[:n_target]]
    spin_roots = [item[2] for item in spin_merged[:n_target]]
    spin_comparison = compare_to_inventory(
        'spin_resolved', spin_energies, spin_roots,
        inventory_energies, inventory_roots, levels) if spin_roots else None
    families['spin_resolved'] = {
        'per_spin': spin_resolved,
        'merged_energies': spin_energies,
        'comparison': spin_comparison,
        'recovers_target': bool(
            spin_comparison is not None
            and len(spin_roots) == n_target
            and spin_comparison['max_abs_energy_difference_Ha']
            < ENERGY_AGREEMENT_TOL),
    }

    print('Running CIS-seeded solve...', flush=True)
    cis_record, cis_guesses = build_cis_guesses(system, INVENTORY_ROOTS)
    cis_projection, _ = guess_projections(system, cis_guesses, inventory_roots)
    cis_run, cis_roots = solve(
        system, 'cis_seeded_matched', n_target,
        ci0=[guess.copy() for guess in cis_guesses[:max(n_target, 6)]])
    cis_comparison = compare_to_inventory(
        'cis_seeded_matched', cis_run['energies'], cis_roots,
        inventory_energies, inventory_roots, levels)
    families['cis_seeded_matched'] = {
        'cis_subspace': cis_record,
        'cis_span_projection': cis_projection,
        'run': cis_run,
        'comparison': cis_comparison,
        'recovers_target': bool(
            cis_comparison['max_abs_energy_difference_Ha']
            < ENERGY_AGREEMENT_TOL and cis_run['residual_test_passed']),
    }

    print('Running oracle-seeded reachability control...', flush=True)
    oracle_run, oracle_roots = solve(
        system, 'oracle_seeded', n_target,
        ci0=[inventory_roots[index].copy() for index in target_roots])
    oracle_comparison = compare_to_inventory(
        'oracle_seeded', oracle_run['energies'], oracle_roots,
        inventory_energies, inventory_roots, levels)
    families['oracle_seeded'] = {
        'run': oracle_run,
        'comparison': oracle_comparison,
        'recovers_target': bool(
            oracle_comparison['max_abs_energy_difference_Ha']
            < ENERGY_AGREEMENT_TOL),
    }

    report['families'] = families
    write_report(args.output_dir, report)

    smallest_margin = None
    for margin in sorted(overshoot, key=int):
        entry = overshoot[margin]
        if entry['recovers_target'] and entry.get(
                'stable_against_next_margin') is not False:
            smallest_margin = int(margin)
            break

    print('Running stability replicas...', flush=True)
    replicas = []
    if smallest_margin is not None:
        for replica in range(args.replicas):
            fresh = ReferenceSystem(n_active)
            run, _ = solve(fresh, f'replica_{replica}',
                           n_target + smallest_margin)
            replicas.append({
                'replica': replica,
                'energies': run['energies'][:n_target],
                'max_residual_norm': run['max_residual_norm'],
                'deviation_Ha': [
                    float(run['energies'][index]
                          - report['target']['energies'][index])
                    for index in range(n_target)],
            })
    stable = bool(replicas) and all(
        max(abs(value) for value in item['deviation_Ha'])
        < ENERGY_AGREEMENT_TOL for item in replicas)
    report['stability'] = {
        'margin_used': smallest_margin,
        'replicas': replicas,
        'stable': stable,
    }

    default_recovers = bool(
        families['default']['comparison']['max_abs_energy_difference_Ha']
        < ENERGY_AGREEMENT_TOL and default_run['residual_test_passed'])
    missing = [index for index in target_roots
               if guess_records[str(n_target)]
               ['inventory_root_span_weights'][index] < 1e-12]

    if not inventory['residual_test_passed']:
        classification = 'UNCONVERGED_REFERENCE'
    elif default_recovers:
        classification = 'NO_ROOT_SELECTION_FAILURE'
    elif families['symmetry_resolved']['recovers_target']:
        classification = 'SYMMETRY_RESOLVED_TARGETING_CONFIRMED'
    elif families['spin_resolved']['recovers_target']:
        classification = 'SPIN_RESOLVED_TARGETING_CONFIRMED'
    elif families['cis_seeded_matched']['recovers_target']:
        classification = 'BRILLOUIN_SINGLE_SEEDING_CONFIRMED'
    elif stable:
        classification = 'GENERIC_INITIAL_SUBSPACE_CONFIRMED'
    elif families['oracle_seeded']['recovers_target']:
        classification = 'ORACLE_SEED_ONLY'
    else:
        classification = 'ROOT_TARGETING_MECHANISM_UNRESOLVED'

    report['analysis'] = {
        'default_recovers_target': default_recovers,
        'target_roots_with_zero_default_projection': missing,
        'hypothesis_B_symmetry_exclusion': bool(missing),
        'hypothesis_D_false_convergence': bool(
            not default_recovers
            and default_run['solver_converged'] is not None
            and all(default_run['solver_converged'])),
        'hypothesis_E_root_homing': bool(
            not default_recovers and smallest_margin is not None
            and smallest_margin > 0),
        'smallest_recovering_margin': smallest_margin,
    }
    report['classification'] = classification
    report['status'] = 'COMPLETE'
    report['wall_time_seconds'] = float(time.perf_counter() - total_start)
    path = write_report(args.output_dir, report)

    print('\nDiagnostic summary', flush=True)
    print(f"  active space        = CAS({sum(N_ACTIVE_ELEC)}e,{n_active}o), "
          f"dim {system.dimension}", flush=True)
    print(f"  target roots        = {target_roots}", flush=True)
    print(f"  default recovers    = {default_recovers}", flush=True)
    print(f"  zero-projection     = {missing}", flush=True)
    print(f"  smallest margin     = {smallest_margin}", flush=True)
    print(f"  replicas stable     = {stable}", flush=True)
    print(f"  classification      = {classification}", flush=True)
    print(f"  report              = {path}", flush=True)


if __name__ == '__main__':
    main()
