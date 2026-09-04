#!/usr/bin/env python3
"""Diagnose why default three-root CASCI skips low N2 Ms=0 roots.

This script performs reference-level FCI calculations only.  It does not
construct a Schmidt basis or call the downfolded wave-operator solver.
"""

import argparse
import json
import os
import time
from itertools import permutations

import numpy as np


GEOMETRY = 'N 0 0 0; N 0 0 1.098'
BASIS = 'cc-pVDZ'
N_ACTIVE = 10
N_ACTIVE_ELEC = (5, 5)
N_CORE = 2
TARGET_ROOTS = 3
INVENTORY_ROOTS = 8
PROTOCOL = 'docs/theory/n2_lowest_ms0_root_selection_protocol.md'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Diagnose N2 lowest-three Ms=0 CASCI root selection')
    parser.add_argument('--output-dir', required=True)
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
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_report(output_dir, report):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'n2_lowest_ms0_root_selection.json')
    temporary = path + '.tmp'
    with open(temporary, 'w', encoding='utf-8') as handle:
        json.dump(serializable(report), handle, indent=2)
    os.replace(temporary, path)


def convergence_value(value):
    if value is None:
        return None
    array = np.asarray(value)
    if array.ndim == 0:
        return bool(array)
    return [bool(item) for item in array.reshape(-1)]


def all_converged(value):
    if isinstance(value, list):
        return all(value)
    return bool(value)


def occupied_orbitals(bit_string):
    return [orbital for orbital in range(N_ACTIVE)
            if int(bit_string) & (1 << orbital)]


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


def normalize(vector):
    flat = np.asarray(vector, dtype=float).reshape(-1)
    norm = np.linalg.norm(flat)
    if norm == 0.0:
        raise ValueError('zero-norm CI vector')
    return flat / norm


def determinant_record(address, coefficient, alpha_strings, beta_strings,
                       rank_map):
    n_beta = len(beta_strings)
    alpha_index, beta_index = divmod(int(address), n_beta)
    alpha_string = int(alpha_strings[alpha_index])
    beta_string = int(beta_strings[beta_index])
    return {
        'flat_address': int(address),
        'alpha_address': int(alpha_index),
        'beta_address': int(beta_index),
        'alpha_occupied': occupied_orbitals(alpha_string),
        'beta_occupied': occupied_orbitals(beta_string),
        'excitation_rank_from_hf': int(rank_map[address]),
        'coefficient': float(coefficient),
        'weight': float(abs(coefficient) ** 2),
    }


def characterize_vector(vector, rank_map, alpha_strings, beta_strings,
                        top_count=8):
    normalized = normalize(vector)
    maximum_rank = int(np.max(rank_map))
    weights = np.bincount(
        rank_map, weights=np.abs(normalized) ** 2,
        minlength=maximum_rank + 1)
    leading = np.argsort(-np.abs(normalized))[:top_count]
    return {
        'norm': float(np.linalg.norm(vector)),
        'rank_weights': {str(rank): float(weights[rank])
                         for rank in range(len(weights))},
        'weight_rank_0': float(weights[0]),
        'weight_rank_1': float(weights[1]),
        'weight_rank_2': float(weights[2]),
        'weight_rank_ge_3': float(np.sum(weights[3:])),
        'leading_determinants': [
            determinant_record(
                address, normalized[address], alpha_strings, beta_strings,
                rank_map)
            for address in leading
        ],
    }


def run_fci(label, h1eff, h2eff, ecore, mol, ci0=None, nroots=3):
    from pyscf.fci import direct_spin1

    solver = direct_spin1.FCI(mol)
    solver.nroots = nroots
    start = time.perf_counter()
    energies, vectors = solver.kernel(
        h1eff, h2eff, N_ACTIVE, N_ACTIVE_ELEC,
        ci0=ci0, nroots=nroots, ecore=ecore, verbose=0)
    energies = np.atleast_1d(np.asarray(energies, dtype=float)).reshape(-1)
    if nroots == 1:
        roots = [normalize(vectors)]
    else:
        roots = [normalize(vector) for vector in vectors[:nroots]]
    if len(energies) < nroots or len(roots) < nroots:
        raise RuntimeError(
            f'{label} returned {len(roots)} vectors and {len(energies)} '
            f'energies for {nroots} requested roots')
    return {
        'label': label,
        'energies': energies[:nroots],
        'excitation_mH': (energies[:nroots] - energies[0]) * 1000.0,
        'converged': convergence_value(solver.converged),
        'conv_tol': float(solver.conv_tol),
        'max_cycle': int(solver.max_cycle),
        'max_space': int(solver.max_space),
        'pspace_size': int(solver.pspace_size),
        'davidson_only': bool(solver.davidson_only),
        'wall_time_seconds': float(time.perf_counter() - start),
    }, roots, solver


def match_to_inventory(label, record, roots, inventory_record,
                       inventory_roots):
    overlap = np.abs(np.asarray([
        [np.vdot(root, inventory) for inventory in inventory_roots]
        for root in roots
    ]))
    permutation = max(
        permutations(range(len(inventory_roots)), len(roots)),
        key=lambda candidate: sum(
            overlap[index, candidate[index]] for index in range(len(roots))))
    matched_overlaps = [
        float(overlap[index, permutation[index]])
        for index in range(len(roots))]
    inventory_energies = np.asarray(inventory_record['energies'])
    energy_differences = [
        float(record['energies'][index]
              - inventory_energies[permutation[index]])
        for index in range(len(roots))]
    return {
        'label': label,
        'absolute_overlap': overlap,
        'inventory_mapping': list(permutation),
        'matched_overlaps': matched_overlaps,
        'matched_energy_differences_Ha': energy_differences,
    }


def initial_guess_records(guesses, hdiag, inventory_roots, rank_map,
                          alpha_strings, beta_strings):
    normalized_guesses = [normalize(guess) for guess in guesses]
    records = []
    for index, guess in enumerate(normalized_guesses):
        dominant = int(np.argmax(np.abs(guess)))
        records.append({
            'index': index,
            'dominant_determinant': determinant_record(
                dominant, guess[dominant], alpha_strings, beta_strings,
                rank_map),
            'dominant_hdiag': float(hdiag[dominant]),
            'nonzero_count_above_1e-12': int(
                np.count_nonzero(np.abs(guess) > 1e-12)),
            'overlap_with_inventory': [
                float(abs(np.vdot(guess, root)))
                for root in inventory_roots
            ],
        })

    matrix = np.column_stack(normalized_guesses)
    q_basis, _ = np.linalg.qr(matrix)
    span_weights = [
        float(np.linalg.norm(q_basis.T.conj() @ root) ** 2)
        for root in inventory_roots
    ]
    return records, normalized_guesses, span_weights


def build_cis_guesses(operator_solver, h1eff, h2eff, ecore, rank_map,
                      alpha_strings, beta_strings, inventory_roots):
    from pyscf.fci import spin_op

    cis_addresses = np.flatnonzero(rank_map <= 1)
    n_alpha = len(alpha_strings)
    n_beta = len(beta_strings)
    h2e = operator_solver.absorb_h1e(
        h1eff, h2eff, N_ACTIVE, N_ACTIVE_ELEC, 0.5)
    h_cis = np.empty((len(cis_addresses), len(cis_addresses)))
    for column, address in enumerate(cis_addresses):
        unit = np.zeros((n_alpha, n_beta))
        unit.reshape(-1)[address] = 1.0
        sigma = operator_solver.contract_2e(
            h2e, unit, N_ACTIVE, N_ACTIVE_ELEC).reshape(-1)
        h_cis[:, column] = sigma[cis_addresses]
    h_cis = 0.5 * (h_cis + h_cis.T)
    energies, coefficients = np.linalg.eigh(h_cis)

    guesses = []
    records = []
    for root in range(min(8, len(energies))):
        vector = np.zeros(n_alpha * n_beta)
        vector[cis_addresses] = coefficients[:, root]
        vector = normalize(vector)
        guesses.append(vector)
        spin_squared, multiplicity = spin_op.spin_square(
            vector.reshape(n_alpha, n_beta), N_ACTIVE, N_ACTIVE_ELEC)
        records.append({
            'index': root,
            'total_energy': float(energies[root] + ecore),
            'excitation_mH': float((energies[root] - energies[0]) * 1000.0),
            'spin_squared': float(spin_squared),
            'multiplicity': float(multiplicity),
            'overlap_with_inventory': [
                float(abs(np.vdot(vector, target)))
                for target in inventory_roots
            ],
        })
    return {
        'dimension': int(len(cis_addresses)),
        'n_hf_determinants': int(np.count_nonzero(rank_map == 0)),
        'n_single_determinants': int(np.count_nonzero(rank_map == 1)),
        'roots': records,
    }, guesses


def main():
    from pyscf import gto, mcscf, scf
    from pyscf.fci import cistring, direct_spin1, spin_op

    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    report = {
        'protocol': PROTOCOL,
        'status': 'RUNNING',
        'classification': None,
        'system': {
            'geometry': GEOMETRY,
            'basis': BASIS,
            'n_active': N_ACTIVE,
            'n_active_elec': list(N_ACTIVE_ELEC),
            'n_core': N_CORE,
            'ms': 0,
            'target': 'lowest three energies in the fixed Ms=0 sector',
        },
    }
    write_report(args.output_dir, report)

    total_start = time.perf_counter()
    mol = gto.M(atom=GEOMETRY, basis=BASIS, spin=0, verbose=0)
    mf = scf.RHF(mol).run(verbose=0)
    cas = mcscf.CASCI(mf, N_ACTIVE, sum(N_ACTIVE_ELEC))
    cas.frozen = N_CORE
    h1eff, ecore = cas.get_h1eff()
    h2eff = cas.get_h2eff()
    report['system']['rhf_converged'] = bool(mf.converged)
    report['system']['rhf_energy'] = float(mf.e_tot)
    report['system']['ecore'] = float(ecore)

    alpha_strings = cistring.gen_strings4orblist(
        range(N_ACTIVE), N_ACTIVE_ELEC[0])
    beta_strings = cistring.gen_strings4orblist(
        range(N_ACTIVE), N_ACTIVE_ELEC[1])
    hf_alpha = (1 << N_ACTIVE_ELEC[0]) - 1
    hf_beta = (1 << N_ACTIVE_ELEC[1]) - 1
    rank_map = make_rank_map(
        alpha_strings, beta_strings, hf_alpha, hf_beta)
    determinant_count = len(alpha_strings) * len(beta_strings)
    report['determinant_space'] = {
        'full_dimension': int(determinant_count),
        'counts_by_excitation_rank': {
            str(rank): int(np.count_nonzero(rank_map == rank))
            for rank in range(int(np.max(rank_map)) + 1)
        },
        'contains_all_singles': int(np.count_nonzero(rank_map == 1)) == 50,
        'contains_all_doubles': int(np.count_nonzero(rank_map == 2)) == 825,
    }

    print('Running eight-root full-CI inventory...', flush=True)
    inventory_record, inventory_roots, inventory_solver = run_fci(
        'inventory_8', h1eff, h2eff, ecore, mol,
        nroots=INVENTORY_ROOTS)
    inventory_character = []
    for root_index, root in enumerate(inventory_roots):
        spin_squared, multiplicity = spin_op.spin_square(
            root.reshape(len(alpha_strings), len(beta_strings)),
            N_ACTIVE, N_ACTIVE_ELEC)
        character = characterize_vector(
            root, rank_map, alpha_strings, beta_strings)
        character.update({
            'index': root_index,
            'total_energy': float(inventory_record['energies'][root_index]),
            'excitation_mH': float(
                inventory_record['excitation_mH'][root_index]),
            'spin_squared': float(spin_squared),
            'multiplicity': float(multiplicity),
        })
        inventory_character.append(character)
    inventory_record['root_character'] = inventory_character
    report['inventory'] = inventory_record
    write_report(args.output_dir, report)

    print('Inspecting PySCF default initial guesses...', flush=True)
    hdiag = inventory_solver.make_hdiag(
        h1eff, h2eff, N_ACTIVE, N_ACTIVE_ELEC).reshape(-1)
    guesses_3 = inventory_solver.get_init_guess(
        N_ACTIVE, N_ACTIVE_ELEC, TARGET_ROOTS, hdiag)
    guesses_8 = inventory_solver.get_init_guess(
        N_ACTIVE, N_ACTIVE_ELEC, INVENTORY_ROOTS, hdiag)
    guess_records_3, normalized_guesses_3, span_weights_3 = (
        initial_guess_records(
            guesses_3, hdiag, inventory_roots, rank_map,
            alpha_strings, beta_strings))
    guess_records_8, normalized_guesses_8, span_weights_8 = (
        initial_guess_records(
            guesses_8, hdiag, inventory_roots, rank_map,
            alpha_strings, beta_strings))
    report['default_initial_guesses'] = {
        'nroots_3': {
            'vectors': guess_records_3,
            'inventory_root_span_weights': span_weights_3,
        },
        'nroots_8': {
            'vectors': guess_records_8,
            'inventory_root_span_weights': span_weights_8,
        },
    }

    print('Measuring RHF-to-excitation-rank Hamiltonian couplings...', flush=True)
    n_alpha = len(alpha_strings)
    n_beta = len(beta_strings)
    alpha_lookup = {int(string): index
                    for index, string in enumerate(alpha_strings)}
    beta_lookup = {int(string): index
                   for index, string in enumerate(beta_strings)}
    hf_address = alpha_lookup[hf_alpha] * n_beta + beta_lookup[hf_beta]
    hf_vector = np.zeros((n_alpha, n_beta))
    hf_vector.reshape(-1)[hf_address] = 1.0
    h2e = inventory_solver.absorb_h1e(
        h1eff, h2eff, N_ACTIVE, N_ACTIVE_ELEC, 0.5)
    h_hf = inventory_solver.contract_2e(
        h2e, hf_vector, N_ACTIVE, N_ACTIVE_ELEC).reshape(-1)
    coupling_by_rank = {}
    for rank in range(int(np.max(rank_map)) + 1):
        values = h_hf[rank_map == rank]
        coupling_by_rank[str(rank)] = {
            'norm': float(np.linalg.norm(values)),
            'max_abs': float(np.max(np.abs(values))) if len(values) else 0.0,
            'nonzero_above_1e-12': int(
                np.count_nonzero(np.abs(values) > 1e-12)),
        }
    report['hf_hamiltonian_coupling_by_excitation_rank'] = coupling_by_rank
    report['brillouin_single_coupling_zero'] = bool(
        coupling_by_rank['1']['max_abs'] < 1e-10)
    write_report(args.output_dir, report)

    print('Running default three-root calculation...', flush=True)
    default_record, default_roots, _ = run_fci(
        'default_3', h1eff, h2eff, ecore, mol, nroots=TARGET_ROOTS)
    default_match = match_to_inventory(
        'default_3', default_record, default_roots,
        inventory_record, inventory_roots)
    report['default_three_root'] = {
        'run': default_record,
        'match': default_match,
    }

    print('Running inventory-seeded three-root reachability control...', flush=True)
    inventory_seeded_record, inventory_seeded_roots, _ = run_fci(
        'inventory_seeded_3', h1eff, h2eff, ecore, mol,
        ci0=[root.copy() for root in inventory_roots[:TARGET_ROOTS]],
        nroots=TARGET_ROOTS)
    inventory_seeded_match = match_to_inventory(
        'inventory_seeded_3', inventory_seeded_record,
        inventory_seeded_roots, inventory_record, inventory_roots)
    report['inventory_seeded_three_root'] = {
        'run': inventory_seeded_record,
        'match': inventory_seeded_match,
    }

    print('Running expanded-default-seed three-root control...', flush=True)
    expanded_record, expanded_roots, _ = run_fci(
        'expanded_default_seeded_3', h1eff, h2eff, ecore, mol,
        ci0=[guess.copy() for guess in normalized_guesses_8],
        nroots=TARGET_ROOTS)
    expanded_match = match_to_inventory(
        'expanded_default_seeded_3', expanded_record, expanded_roots,
        inventory_record, inventory_roots)
    report['expanded_default_seeded_three_root'] = {
        'run': expanded_record,
        'match': expanded_match,
    }

    print('Building RHF+all-singles CIS initial subspace...', flush=True)
    cis_record, cis_guesses = build_cis_guesses(
        inventory_solver, h1eff, h2eff, ecore, rank_map,
        alpha_strings, beta_strings, inventory_roots)
    report['cis_subspace'] = cis_record

    print('Running CIS-seeded three-root calculation...', flush=True)
    cis_seeded_record, cis_seeded_roots, _ = run_fci(
        'cis_seeded_3', h1eff, h2eff, ecore, mol,
        ci0=[guess.copy() for guess in cis_guesses[:TARGET_ROOTS]],
        nroots=TARGET_ROOTS)
    cis_seeded_match = match_to_inventory(
        'cis_seeded_3', cis_seeded_record, cis_seeded_roots,
        inventory_record, inventory_roots)
    report['cis_seeded_three_root'] = {
        'run': cis_seeded_record,
        'match': cis_seeded_match,
    }

    target_mapping = list(range(TARGET_ROOTS))
    default_mapping = default_match['inventory_mapping']
    intervention_mappings = {
        'inventory_seeded': inventory_seeded_match['inventory_mapping'],
        'expanded_default_seeded': expanded_match['inventory_mapping'],
        'cis_seeded': cis_seeded_match['inventory_mapping'],
    }
    skipped_lower_roots = [
        root for root in range(max(default_mapping) + 1)
        if root not in default_mapping]
    missing_target_roots = [
        root for root in target_mapping if root not in default_mapping]
    default_target_coverage = [span_weights_3[root]
                               for root in missing_target_roots]
    missing_single_weights = [
        inventory_character[root]['weight_rank_1']
        for root in missing_target_roots]
    all_runs_converged = all([
        all_converged(inventory_record['converged']),
        all_converged(default_record['converged']),
        all_converged(inventory_seeded_record['converged']),
        all_converged(expanded_record['converged']),
        all_converged(cis_seeded_record['converged']),
    ])

    if not all_runs_converged:
        classification = 'UNCONVERGED_REFERENCE'
    elif default_mapping == target_mapping:
        classification = 'NO_ROOT_SELECTION_FAILURE'
    elif inventory_seeded_match['inventory_mapping'] != target_mapping:
        classification = 'UNRESOLVED_EIGENSOLVER_FAILURE'
    elif (
        report['brillouin_single_coupling_zero']
        and missing_single_weights
        and min(missing_single_weights) > 0.10
        and default_target_coverage
        and max(default_target_coverage) < 1e-10
        and cis_seeded_match['inventory_mapping'] == target_mapping
    ):
        classification = 'BRILLOUIN_SINGLE_SEEDING_CONFIRMED'
    else:
        classification = 'INITIAL_SUBSPACE_COVERAGE_CONFIRMED'

    report['analysis'] = {
        'target_inventory_mapping': target_mapping,
        'default_inventory_mapping': default_mapping,
        'skipped_lower_inventory_roots': skipped_lower_roots,
        'missing_target_inventory_roots': missing_target_roots,
        'missing_target_default_span_weights': default_target_coverage,
        'missing_target_single_excitation_weights': missing_single_weights,
        'intervention_inventory_mappings': intervention_mappings,
        'all_runs_converged': all_runs_converged,
        'explicit_single_or_double_truncation': False,
    }
    report['classification'] = classification
    report['status'] = 'COMPLETE'
    report['wall_time_seconds'] = float(time.perf_counter() - total_start)
    write_report(args.output_dir, report)

    print('\nDiagnostic summary', flush=True)
    print(f"  full determinant dimension={determinant_count}", flush=True)
    print(
        f"  HF->singles max={coupling_by_rank['1']['max_abs']:.3e}, "
        f"norm={coupling_by_rank['1']['norm']:.3e}", flush=True)
    print(f"  default mapping={default_mapping}", flush=True)
    print(
        f"  intervention mappings={intervention_mappings}", flush=True)
    print(f"  classification={classification}", flush=True)
    print(
        '  report=' + os.path.join(
            args.output_dir, 'n2_lowest_ms0_root_selection.json'), flush=True)


if __name__ == '__main__':
    main()
