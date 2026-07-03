#!/usr/bin/env python3
"""
Phase 18: m-convergence test with correct A/B decomposition for Krylov-dCI.

N2/cc-pVDZ, CAS(10,10), P=400, HF-perturbation P-space.
Compares new results (delta != 0, B = H_O' - Delta*I) against old Stage C
results (delta = 0).

Delta = E(DMRG-CI) - E0(P)  — exact shift from reference.
"""

import sys
sys.path.insert(0, '/data/home/wangcx/krylov-dci')

import numpy as np
from numpy.linalg import eigh
import time
import json
import os

from src_mf.pyscf_backend import QSpaceIndex, KDCIBackend
from src.effective_h import build_effective_H, diagonalize_effective_H
from pyscf import gto, scf, ao2mo
from pyscf.fci import cistring, direct_spin1, selected_ci


def build_cas1010_n2():
    """Build N2/cc-pVDZ CAS(10,10) system."""
    mol = gto.M(atom='N 0 0 0; N 0 0 1.1', basis='cc-pVDZ', verbose=0)
    mf = scf.RHF(mol).run(verbose=0)

    n_core = 2  # freeze 2 core orbitals
    n_act = 10
    norb = mf.mo_coeff.shape[1]
    n_core_orbs = list(range(n_core))
    n_act_orbs = list(range(n_core, n_core + n_act))
    n_virt_orbs = list(range(n_core + n_act, norb))

    # Transform integrals to MO basis
    h1e_ao = mf.get_hcore()
    h1e_mo = mf.mo_coeff.T @ h1e_ao @ mf.mo_coeff
    eri_ao = mol.intor('int2e')
    eri_mo = ao2mo.full(eri_ao, mf.mo_coeff, compact=False)
    eri_mo = eri_mo.reshape(norb, norb, norb, norb)

    # Restrict to active space
    h1e_act = h1e_mo[np.ix_(n_act_orbs, n_act_orbs)]
    eri_act = eri_mo[np.ix_(n_act_orbs, n_act_orbs, n_act_orbs, n_act_orbs)]

    nelec = (mol.nelec[0] - n_core, mol.nelec[1] - n_core)

    # Generate all CAS strings
    alpha_strs = cistring.gen_strings4orblist(range(n_act), nelec[0])
    beta_strs = cistring.gen_strings4orblist(range(n_act), nelec[1])

    q_idx = QSpaceIndex(alpha_strs, beta_strs, n_act, nelec,
                        h1e_act, eri_act)
    return mol, mf, q_idx, n_act, nelec, h1e_act, eri_act


def select_p_space_hf_pt(q_idx, N_target):
    """Select P-space using HF perturbation theory.

    Returns list of (alpha_str, beta_str) tuples, sorted by PT weight.
    """
    from src.hamiltonian import Hamiltonian
    
    # Get HF determinant
    from src.determinants import hf_determinant
    hf_a, hf_b = hf_determinant(*q_idx.nelec)
    
    # Build Hamiltonian for matrix elements
    h2_4d = ao2mo.restore('s1', q_idx.eri, q_idx.norb).reshape(
        q_idx.norb, q_idx.norb, q_idx.norb, q_idx.norb)
    ham = Hamiltonian(h1=q_idx.h1e, h2=h2_4d, E_nuc=0.0, E_HF=0.0)
    
    # Compute HF diagonal
    from pyscf.fci import cistring
    E_HF = ham.matrix_element((hf_a, hf_b), (hf_a, hf_b))
    
    # Generate SD excitations and compute PT2 weights
    from src.determinants import bit_positions
    alpha_occ = bit_positions(hf_a)
    beta_occ = bit_positions(hf_b)
    all_orbs = list(range(q_idx.norb))
    alpha_virt = [p for p in all_orbs if p not in alpha_occ]
    beta_virt = [p for p in all_orbs if p not in beta_occ]
    
    scores = []
    
    # Singles: alpha
    for i in alpha_occ:
        for a in alpha_virt:
            det = (hf_a ^ (1 << i) | (1 << a), hf_b)
            hij = ham.matrix_element(det, (hf_a, hf_b))
            hdd = ham.matrix_element(det, det)
            denom = E_HF - hdd
            if abs(denom) > 1e-12:
                scores.append((det, -(hij * hij) / denom))
    
    # Singles: beta
    for i in beta_occ:
        for a in beta_virt:
            det = (hf_a, hf_b ^ (1 << i) | (1 << a))
            hij = ham.matrix_element(det, (hf_a, hf_b))
            hdd = ham.matrix_element(det, det)
            denom = E_HF - hdd
            if abs(denom) > 1e-12:
                scores.append((det, -(hij * hij) / denom))
    
    # Doubles: alpha-alpha
    import itertools
    for (i1, i2) in itertools.combinations(alpha_occ, 2):
        for (a1, a2) in itertools.combinations(alpha_virt, 2):
            new_a = hf_a ^ (1 << i1) ^ (1 << i2) | (1 << a1) | (1 << a2)
            det = (new_a, hf_b)
            hij = ham.matrix_element(det, (hf_a, hf_b))
            hdd = ham.matrix_element(det, det)
            denom = E_HF - hdd
            if abs(denom) > 1e-12:
                scores.append((det, -(hij * hij) / denom))
    
    # Doubles: beta-beta
    for (i1, i2) in itertools.combinations(beta_occ, 2):
        for (a1, a2) in itertools.combinations(beta_virt, 2):
            new_b = hf_b ^ (1 << i1) ^ (1 << i2) | (1 << a1) | (1 << a2)
            det = (hf_a, new_b)
            hij = ham.matrix_element(det, (hf_a, hf_b))
            hdd = ham.matrix_element(det, det)
            denom = E_HF - hdd
            if abs(denom) > 1e-12:
                scores.append((det, -(hij * hij) / denom))
    
    # Doubles: alpha-beta
    for i in alpha_occ:
        for j in beta_occ:
            for a in alpha_virt:
                for b in beta_virt:
                    new_a = hf_a ^ (1 << i) | (1 << a)
                    new_b = hf_b ^ (1 << j) | (1 << b)
                    det = (new_a, new_b)
                    hij = ham.matrix_element(det, (hf_a, hf_b))
                    hdd = ham.matrix_element(det, det)
                    denom = E_HF - hdd
                    if abs(denom) > 1e-12:
                        scores.append((det, -(hij * hij) / denom))
    
    scores.sort(key=lambda x: x[1], reverse=True)
    
    # Always include HF det
    selected = [(hf_a, hf_b)]
    for det, score in scores:
        if det not in selected:
            selected.append(det)
        if len(selected) >= N_target:
            break
    
    return selected[:N_target]


def run_kdci_layers(q_idx, p_dets, E0_P, delta, m_max, e_dmrg, nroots=6):
    """Run Krylov-dCI with multi-layer propagation."""
    backend = KDCIBackend(q_idx)
    
    # Build H_PP
    from src.hamiltonian import Hamiltonian
    h2_4d = ao2mo.restore('s1', q_idx.eri, q_idx.norb).reshape(
        q_idx.norb, q_idx.norb, q_idx.norb, q_idx.norb)
    ham = Hamiltonian(h1=q_idx.h1e, h2=h2_4d, E_nuc=0.0, E_HF=0.0)
    
    N = len(p_dets)
    H_PP = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            H_PP[i, j] = ham.matrix_element(p_dets[i], p_dets[j])
    H_PP = 0.5 * (H_PP + H_PP.T)
    
    # Build H_QP
    t0 = time.perf_counter()
    H_QP = backend.build_hqp(p_dets, verbose=False)
    t_hqp = time.perf_counter() - t0
    
    results = []
    
    for m in range(m_max + 1):
        t_layer = time.perf_counter()
        
        # Build Krylov basis up to layer m
        basis, d_total, d_layers = backend.build_krylov_layers(
            H_QP, E0_P, delta=delta, m_max=m, verbose=False)
        
        # Build projected blocks
        H_QQ_t, H_PQ_t = backend.build_projected_blocks(basis, p_dets, verbose=False)
        
        # Effective Hamiltonian
        H_eff = build_effective_H(H_PP, H_PQ_t, H_QQ_t, E0_P, delta=delta)
        ev, _ = diagonalize_effective_H(H_eff, n_states=nroots)
        
        t_elapsed = time.perf_counter() - t_layer
        
        dE0_mH = (ev[0] - e_dmrg[0]) * 1000
        ex_dE_mH = [(ev[i] - e_dmrg[i]) * 1000 for i in range(1, min(nroots, len(ev)))]
        
        results.append({
            'm': m,
            'd_basis': d_total,
            'd_layers': d_layers,
            'dE0_mH': dE0_mH,
            'ev_total': [float(e) for e in ev[:nroots]],
            'ex_dE_mH': ex_dE_mH,
            'wall_s': t_elapsed,
        })
        
        print(f"  m={m}: d={d_total} layers={d_layers}, "
              f"dE0={dE0_mH:+.3f} mH, wall={t_elapsed:.0f}s", flush=True)
    
    return results


def main():
    P_TARGET = 400
    M_MAX = 3
    NROOTS = 6
    
    print("=" * 60)
    print("Phase 18: m-convergence with correct A/B decomposition")
    print(f"N2/cc-pVDZ CAS(10,10) P={P_TARGET} m=0..{M_MAX}")
    print(f"Date: {time.strftime('%c')}")
    print("=" * 60)
    
    # Build system
    print("\nBuilding N2/cc-pVDZ CAS(10,10)...", flush=True)
    mol, mf, q_idx, n_act, nelec, h1e_act, eri_act = build_cas1010_n2()
    M = q_idx.M
    print(f"  Active: {nelec[0]}a + {nelec[1]}b electrons in {n_act} orbitals")
    print(f"  M = {M:,} determinants")
    
    # DMRG-CI reference
    print("\nComputing DMRG-CI reference...", flush=True)
    from pyscf.fci import direct_spin1
    e_fci, ci_fci = direct_spin1.FCI().kernel(
        h1e_act, eri_act, n_act, nelec, nroots=NROOTS, verbose=0)
    e_dmrg = [float(e) for e in np.atleast_1d(e_fci)[:NROOTS]]
    print(f"  DMRG-CI E0 = {e_dmrg[0]:.8f} Ha")
    
    # P-space selection
    print(f"\nSelecting P={P_TARGET} via HF perturbation...", flush=True)
    p_dets = select_p_space_hf_pt(q_idx, P_TARGET)
    N = len(p_dets)
    print(f"  N = {N} determinants")
    
    # Build H_PP and get E0_P
    from src.hamiltonian import Hamiltonian
    h2_4d = ao2mo.restore('s1', q_idx.eri, q_idx.norb).reshape(
        q_idx.norb, q_idx.norb, q_idx.norb, q_idx.norb)
    ham = Hamiltonian(h1=q_idx.h1e, h2=h2_4d, E_nuc=0.0, E_HF=0.0)
    H_PP = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            H_PP[i, j] = ham.matrix_element(p_dets[i], p_dets[j])
    H_PP = 0.5 * (H_PP + H_PP.T)
    E0_P = float(eigh(H_PP)[0][0])
    
    # Exact delta from FCI reference
    delta = e_dmrg[0] - E0_P
    dE0_P_mH = (E0_P - e_dmrg[0]) * 1000
    
    print(f"  E0(P)   = {E0_P:.8f} Ha")
    print(f"  Delta   = {delta:.8f} Ha ({delta*1000:.3f} mH)")
    print(f"  P-only dE0 = {dE0_P_mH:+.3f} mH")
    
    # Run Krylov-dCI
    print(f"\nRunning Krylov-dCI layers m=0..{M_MAX}...", flush=True)
    t_start = time.perf_counter()
    
    results = run_kdci_layers(q_idx, p_dets, E0_P, delta, M_MAX, e_dmrg, nroots=NROOTS)
    
    t_total = time.perf_counter() - t_start
    
    # Summary table
    print("\n" + "=" * 60)
    print("Results: New (delta != 0, B = H_O' - Delta*I)")
    print("-" * 60)
    print(f"{'m':>3}  {'d_basis':>8}  {'d_layers':>20}  {'dE0/mH':>10}  {'wall/s':>8}")
    print("-" * 60)
    for r in results:
        print(f"{r['m']:>3}  {r['d_basis']:>8}  {str(r['d_layers']):>20}  "
              f"{r['dE0_mH']:>+10.3f}  {r['wall_s']:>8.1f}")
    
    # Comparison with old Stage C
    print("\n" + "=" * 60)
    print("Comparison: New vs Old (Stage C, delta=0)")
    print("-" * 60)
    old_results = [
        {'m': 0, 'dE0_mH': -0.689},
        {'m': 1, 'dE0_mH': +5.309},
        {'m': 2, 'dE0_mH': +3.451},
        {'m': 3, 'dE0_mH': +3.357},
    ]
    print(f"{'m':>3}  {'New dE0/mH':>12}  {'Old dE0/mH':>12}  {'Diff/mH':>10}")
    print("-" * 60)
    for rn, ro in zip(results, old_results):
        diff = rn['dE0_mH'] - ro['dE0_mH']
        print(f"{rn['m']:>3}  {rn['dE0_mH']:>+12.3f}  {ro['dE0_mH']:>+12.3f}  "
              f"{diff:>+10.3f}")
    
    # Save results
    output = {
        'P': P_TARGET,
        'N': N,
        'M': M,
        'delta': float(delta),
        'dE0_P_ref_mH': float(dE0_P_mH),
        'e_dmrg_total': e_dmrg,
        'results': results,
        'old_stageC_results': [
            {'m': 0, 'dE0_mH': -0.689},
            {'m': 1, 'dE0_mH': +5.309},
            {'m': 2, 'dE0_mH': +3.451},
            {'m': 3, 'dE0_mH': +3.357},
        ],
        'total_wall_s': t_total,
    }

    outdir = '/data/home/wangcx/krylov-dci/checkpoints_phase18'
    os.makedirs(outdir, exist_ok=True)
    outpath = f'{outdir}/P0400_delta_exact.json'
    with open(outpath, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {outpath}")
    print(f"Total wall: {t_total:.0f}s")


if __name__ == '__main__':
    main()
