#!/usr/bin/env python3
"""Compare Neumann truncation against Krylov-Galerkin resummation.

Both routes approximate the same object, the Q-space resolvent

    (E I - H_QQ)^-1 = A sum_k (B A)^k ,
    A = (E I - D_QQ)^-1 ,  B = H_QQ - D_QQ .

Neumann truncates that series at order k.  The Krylov route builds
``K_m = span{A H_QP, (AB) A H_QP, ..., (AB)^m A H_QP}`` and then inverts
*exactly* inside it.  They are not independent alternatives: the Krylov basis
spans exactly the terms the Neumann series truncates, and the Galerkin
projection is the optimal polynomial in that subspace, whereas the truncated
series is one fixed polynomial.  Krylov is therefore never worse at equal
subspace, and the interesting question is cost, not correctness.

The decisive quantity is the spectral radius ``rho(BA)``.  The Neumann series
converges only for ``rho(BA) < 1``.  The Galerkin projection has no such
requirement.  This script measures both, and estimates ``rho(BA)`` for a bond
scan so the regime of validity is known rather than assumed.

Reference-layer diagnostic.  It does not call the wave-operator solver.
"""

import argparse
import json
import os
import time

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--p-sizes', type=int, nargs='+', default=[10, 20, 40])
    parser.add_argument('--max-neumann-order', type=int, default=4)
    parser.add_argument('--max-krylov-layer', type=int, default=2)
    parser.add_argument('--scan-scales', type=float, nargs='+',
                        default=[0.8, 1.0, 1.5, 2.0, 2.5, 3.0])
    return parser.parse_args()


def dense_cas_hamiltonian(mf, mol, n_cas, n_elec):
    """Full dense active-space Hamiltonian, for small systems only."""
    from pyscf import mcscf
    from pyscf.fci import cistring, direct_spin1

    cas = mcscf.CASCI(mf, n_cas, n_elec)
    h1, ecore = cas.get_h1eff()
    h2 = cas.get_h2eff()
    pair = (n_elec // 2, n_elec // 2)
    solver = direct_spin1.FCI(mol)
    h2e = solver.absorb_h1e(h1, h2, n_cas, pair, 0.5)
    n_str = cistring.num_strings(n_cas, pair[0])
    dimension = n_str * n_str
    matrix = np.zeros((dimension, dimension))
    unit = np.zeros(dimension)
    for index in range(dimension):
        unit[:] = 0.0
        unit[index] = 1.0
        matrix[:, index] = solver.contract_2e(
            h2e, unit.reshape(n_str, n_str), n_cas, pair).ravel()
    return 0.5 * (matrix + matrix.T), float(ecore), dimension


def compare_at_partition(hamiltonian, ecore, p_indices, max_order, max_layer):
    """Neumann and Krylov errors against the exact resolvent, by matvec count."""
    dimension = hamiltonian.shape[0]
    q_indices = np.setdiff1d(np.arange(dimension), p_indices)
    h_pp = hamiltonian[np.ix_(p_indices, p_indices)]
    h_pq = hamiltonian[np.ix_(p_indices, q_indices)]
    h_qq = hamiltonian[np.ix_(q_indices, q_indices)]
    h_qp = h_pq.T
    n_p = len(p_indices)

    reference_energy = float(np.linalg.eigvalsh(h_pp)[0])
    diagonal = np.diag(h_qq)
    off_diagonal = h_qq - np.diag(diagonal)
    resolvent = 1.0 / (reference_energy - diagonal)
    spectral_radius = float(np.max(np.abs(
        np.linalg.eigvals(off_diagonal * resolvent[np.newaxis, :]))))

    exact = np.linalg.solve(
        reference_energy * np.eye(len(q_indices)) - h_qq, h_qp)
    exact_energy = float(
        np.linalg.eigvalsh(h_pp + h_pq @ exact)[0]) + ecore

    weighted = resolvent[:, np.newaxis] * h_qp
    records = []

    term = weighted.copy()
    accumulated = weighted.copy()
    records.append({
        'method': 'neumann', 'order': 0, 'matvecs': 0,
        'energy': float(np.linalg.eigvalsh(h_pp + h_pq @ accumulated)[0]) + ecore})
    for order in range(1, max_order + 1):
        term = resolvent[:, np.newaxis] * (off_diagonal @ term)
        accumulated = accumulated + term
        records.append({
            'method': 'neumann', 'order': order, 'matvecs': order * n_p,
            'energy': float(
                np.linalg.eigvalsh(h_pp + h_pq @ accumulated)[0]) + ecore})

    layers = [weighted]
    for layer in range(max_layer + 1):
        if layer > 0:
            layers.append(resolvent[:, np.newaxis] * (off_diagonal @ layers[-1]))
        basis = np.linalg.qr(np.hstack(layers))[0]
        width = basis.shape[1]
        h_kk = basis.T @ h_qq @ basis
        h_pk = h_pq @ basis
        downfolded = h_pp + h_pk @ np.linalg.solve(
            reference_energy * np.eye(width) - h_kk, h_pk.T)
        records.append({
            'method': 'krylov', 'order': layer,
            'matvecs': layer * n_p + (layer + 1) * width,
            'basis_width': width,
            'energy': float(np.linalg.eigvalsh(downfolded)[0]) + ecore})

    for record in records:
        record['error_vs_exact_resolvent_mH'] = (
            record['energy'] - exact_energy) * 1000.0
    return {
        'n_P': int(n_p), 'n_Q': int(len(q_indices)),
        'spectral_radius_BA': spectral_radius,
        'neumann_converges': bool(spectral_radius < 1.0),
        'exact_resolvent_energy': exact_energy,
        'records': records,
    }


def spectral_radius_scan(scales, n_cas=9, n_elec=10, basis='cc-pVDZ',
                         p_size=200, iterations=60):
    """Matrix-free power iteration for rho(BA) along an N2 bond scan."""
    from pyscf import gto, mcscf, scf
    from pyscf.fci import cistring, direct_spin1

    equilibrium = 1.098
    pair = (n_elec // 2, n_elec // 2)
    results = []
    for scale in scales:
        distance = equilibrium * scale
        mol = gto.M(atom=f'N 0 0 0; N 0 0 {distance:.6f}', basis=basis,
                    spin=0, symmetry='D2h', verbose=0)
        mf = scf.RHF(mol).run(verbose=0)
        cas = mcscf.CASCI(mf, n_cas, n_elec)
        h1, ecore = cas.get_h1eff()
        h2 = cas.get_h2eff()
        solver = direct_spin1.FCI(mol)
        h2e = solver.absorb_h1e(h1, h2, n_cas, pair, 0.5)
        n_str = cistring.num_strings(n_cas, pair[0])
        dimension = n_str * n_str
        hdiag = solver.make_hdiag(h1, h2, n_cas, pair).reshape(-1)
        mask = np.ones(dimension, dtype=bool)
        mask[np.argsort(hdiag)[:p_size]] = False
        energies, _ = solver.kernel(h1, h2, n_cas, pair, nroots=1,
                                    ecore=0.0, verbose=0)
        reference_energy = float(np.atleast_1d(energies)[0])
        diagonal = hdiag[mask]
        weight = 1.0 / (reference_energy - diagonal)

        def apply(vector):
            full = np.zeros(dimension)
            full[mask] = weight * vector
            image = solver.contract_2e(
                h2e, full.reshape(n_str, n_str), n_cas, pair).ravel()
            return image[mask] - diagonal * (weight * vector)

        generator = np.random.default_rng(0)
        vector = generator.standard_normal(int(mask.sum()))
        vector /= np.linalg.norm(vector)
        radius = 0.0
        for _ in range(iterations):
            image = apply(vector)
            norm = float(np.linalg.norm(image))
            if norm == 0.0:
                break
            radius = norm
            vector = image / norm
        results.append({
            'scale': float(scale), 'distance_angstrom': float(distance),
            'reference_energy': reference_energy + float(ecore),
            'spectral_radius_BA': radius,
            'neumann_converges': bool(radius < 1.0),
        })
    return results


def main():
    from pyscf import gto, scf

    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    start = time.perf_counter()
    report = {'partitions': {}, 'n2_spectral_radius_scan': None}

    angle = 104.5 * np.pi / 180.0
    for label, bond in (('equilibrium', 0.9578), ('stretched_2x', 1.9156)):
        y = bond * np.sin(angle / 2)
        z = bond * np.cos(angle / 2)
        mol = gto.M(atom=f'O 0 0 0; H 0 {y:.6f} {z:.6f}; H 0 {-y:.6f} {z:.6f}',
                    basis='sto-3g', spin=0, verbose=0)
        mf = scf.RHF(mol).run(verbose=0)
        hamiltonian, ecore, dimension = dense_cas_hamiltonian(mf, mol, 5, 6)
        exact = float(np.linalg.eigvalsh(hamiltonian)[0]) + ecore
        order = np.argsort(np.diag(hamiltonian))
        entry = {'exact_casci_energy': exact, 'dimension': dimension,
                 'partitions': {}}
        for size in args.p_sizes:
            result = compare_at_partition(
                hamiltonian, ecore, np.sort(order[:size]),
                args.max_neumann_order, args.max_krylov_layer)
            for record in result['records']:
                record['error_vs_casci_mH'] = (
                    record['energy'] - exact) * 1000.0
            entry['partitions'][str(size)] = result
        report['partitions'][label] = entry

        print(f"\n### H2O/STO-3G CAS(6e,5o) {label}", flush=True)
        for size, result in entry['partitions'].items():
            print(f"  |P|={size} rho(BA)={result['spectral_radius_BA']:.3f} "
                  f"({'converges' if result['neumann_converges'] else 'DIVERGES'})",
                  flush=True)
            for record in result['records']:
                print(f"    {record['method']:>8} {record['order']} "
                      f"matvecs={record['matvecs']:>4} "
                      f"err_vs_resolvent={record['error_vs_exact_resolvent_mH']:>12.4f} mH",
                      flush=True)

    print("\n### N2/cc-pVDZ CAS(10e,9o) spectral-radius scan", flush=True)
    scan = spectral_radius_scan(args.scan_scales)
    report['n2_spectral_radius_scan'] = scan
    for item in scan:
        print(f"  R/Re={item['scale']:.1f} rho(BA)={item['spectral_radius_BA']:.3f} "
              f"{'converges' if item['neumann_converges'] else 'DIVERGES'}",
              flush=True)

    report['wall_time_seconds'] = float(time.perf_counter() - start)
    path = os.path.join(args.output_dir, 'neumann_vs_krylov_comparison.json')
    temporary = path + '.tmp'
    with open(temporary, 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2)
    os.replace(temporary, path)
    print(f"\nreport: {path}", flush=True)


if __name__ == '__main__':
    main()
