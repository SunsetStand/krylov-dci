#!/usr/bin/env python
"""N2 ground-state dissociation curve; see docs/theory/n2_dissociation_protocol.md.

Writes a small JSON summary only. Example:
    python scripts/diagnostics/run_n2_dissociation.py --output-dir results/n2_dissociation
"""
import argparse
import json
import os
import resource
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from dm_svd_dci.pipeline_state_averaged import run_state_averaged_dci  # noqa: E402
from dm_svd_embedding.occ_virt_partition import setup_partition        # noqa: E402

R_GRID = (0.9, 1.0, 1.098, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.7, 3.0)
N_ACT, N_ELEC, N_CORE, N_OCC = 8, (5, 5), 2, 5
P_FIXED = [8, 9, 10]
P_ALL = [4, 5, 6, 7, 8, 9, 10]
CONTROLS = dict(outer_mixing=0.7, outer_density_tol=1e-6, outer_energy_tol=1e-7,
                outer_max_iter=20, wave_damping=0.7, wave_residual_tol=1e-9,
                wave_energy_tol=1e-10, wave_max_iter=200, min_denominator=1e-6)


def flat_from_blocks(blocks, partition, dimension):
    """Reassemble a CAS CI vector from its electron-number blocks."""
    flat = np.zeros(dimension)
    for n, block in partition.items():
        if n not in blocks:
            continue
        matrix = np.asarray(blocks[n])
        if np.iscomplexobj(matrix):
            # The solver stores real states in complex arrays; the imaginary
            # part has been measured to be exactly zero. Refuse anything else.
            assert float(np.max(np.abs(matrix.imag), initial=0.0)) < 1e-12
            matrix = matrix.real
        for a, b, det in block['coeff_map']:
            flat[int(det)] = matrix[int(a), int(b)]
    return flat / np.linalg.norm(flat)


def reference(atom):
    """Exact CASCI ground state, evaluator side only: energy, vector, <S^2>."""
    from pyscf import gto, scf, mcscf
    from pyscf.fci import direct_spin1, spin_op
    mol = gto.M(atom=atom, basis='cc-pVDZ', symmetry='D2h', verbose=0)
    mf = scf.RHF(mol)
    mf.max_cycle = 200
    mf.run()
    cas = mcscf.CASCI(mf, N_ACT, sum(N_ELEC))
    cas.frozen = N_CORE
    cas.fcisolver = direct_spin1.FCI(mol)
    cas.fcisolver.nroots = 4
    cas.kernel()
    order = np.argsort(np.asarray(cas.e_tot))
    vec = np.asarray(cas.ci[order[0]])
    s2 = spin_op.spin_square(vec, N_ACT, N_ELEC)[0]
    gaps = [(float(cas.e_tot[i]) - float(cas.e_tot[order[0]])) * 1000.0
            for i in order[1:]]
    return float(cas.e_tot[order[0]]), vec.ravel(), float(s2), gaps, vec.shape


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--eps', type=float, nargs='+', default=[1e-3, 3e-4])
    parser.add_argument('--r', type=float, nargs='+', default=list(R_GRID))
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, 'n2_dissociation.json')
    from pyscf.fci import spin_op

    partition, dets = setup_partition(N_ACT, sum(N_ELEC), N_OCC, ms=0)
    rows = []
    for R in args.r:
        atom = f'N 0 0 0; N 0 0 {R}'
        e_ref, v_ref, s2_ref, gaps, shape = reference(atom)
        weight_by_block = {}
        for n, block in partition.items():
            idx = [int(det) for (_, _, det) in block['coeff_map']]
            weight_by_block[int(n)] = float(np.sum(v_ref[idx] ** 2))
        outside_p = 1.0 - sum(weight_by_block.get(n, 0.0) for n in P_FIXED)
        for eps in args.eps:
            for arm, p_blocks in (('fixed_P', P_FIXED), ('P_all', P_ALL)):
                started = time.perf_counter()
                settings = dict(atom=atom, basis='cc-pVDZ', n_active=N_ACT,
                                n_active_elec=N_ELEC, n_core=N_CORE, n_occ=N_OCC,
                                p_blocks=p_blocks, sa_states=1, svd_eps=eps,
                                seed='lanczos_symm', seed_lanczos_steps=10,
                                symmetry='D2h', enrichment_strength=0.0,
                                embedded_spectrum=True, verbose=False)
                settings.update(CONTROLS)
                row = dict(R=R, svd_eps=eps, arm=arm, p_blocks=p_blocks,
                           reference_energy=e_ref, reference_s2=s2_ref,
                           higher_root_gaps_mH=gaps,
                           exact_weight_outside_fixed_P=outside_p,
                           exact_weight_by_block=weight_by_block)
                try:
                    r = run_state_averaged_dci(**settings)
                    energy = float(np.asarray(r['energies']).ravel()[0])
                    state = flat_from_blocks(r['final_state_blocks'][0],
                                             partition, len(dets))
                    s2 = spin_op.spin_square(state.reshape(shape), N_ACT, N_ELEC)[0]
                    row.update(
                        energy=energy, error_mH=(energy - e_ref) * 1000.0,
                        D=r['partition_info']['D_total'],
                        P_dim=r['partition_info']['P_dim'],
                        Q_dim=r['partition_info']['Q_dim'],
                        converged=bool(r['converged']), n_outer=r['n_outer_iter'],
                        s2=float(s2),
                        overlap_with_exact=float(abs(state @ v_ref)),
                        wave_error_mH=float(np.abs(np.asarray(
                            r['wave_operator_errors_mH'])).ravel()[0]),
                        wall_seconds=time.perf_counter() - started,
                        peak_rss_mib=resource.getrusage(
                            resource.RUSAGE_SELF).ru_maxrss / 1024.0)
                except Exception as error:                     # noqa: BLE001
                    row.update(error=f'{type(error).__name__}: {error}')
                rows.append(row)
                json.dump(rows, open(out, 'w'), indent=1)
                if 'error' in row:
                    print(f"R={R:<5} eps={eps:.0e} {arm:<7} ERROR {row['error']}",
                          flush=True)
                else:
                    print(f"R={R:<5} eps={eps:.0e} {arm:<7} err={row['error_mH']:9.4f} mH "
                          f"D={row['D']:<5} |P|={row['P_dim']:<4} |Q|={row['Q_dim']:<4} "
                          f"S2={row['s2']:.4f} ovl={row['overlap_with_exact']:.6f} "
                          f"conv={row['converged']} it={row['n_outer']} "
                          f"outsideP={outside_p:.4f} {row['wall_seconds']:.0f}s",
                          flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == '__main__':
    main()
