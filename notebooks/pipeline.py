"""
Full Krylov-dCI pipeline: end-to-end computation for a benchmark system.

Usage:
    python pipeline.py --system H2O --basis sto-3g --m-max 3

This script performs the three verification experiments from the 2026-06-28
discussion:
  1. Non-self-consistent (fixed Δ = E_FCI - E0): verify Krylov convergence.
  2. Self-consistent Δ iteration: verify SCF convergence.
  3. SVD compression scan: verify accuracy vs compression trade-off.
"""

import sys
import os
import argparse
import time
import numpy as np
from numpy.linalg import eigh, norm
from typing import Tuple, List, Dict

# Add src to path
# Add both project root and src/ to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from pyscf import gto, scf, fci


# ============================================================================
# System definitions
# ============================================================================

SYSTEMS = {
    'H2': {
        'atom': 'H 0 0 0; H 0 0 0.74',
        'n_orb': 2,
        'n_elec': 2,
        'charge': 0,
        'spin': 0,
        'n_cas_orb': 2,
        'n_cas_elec': 2,
    },
    'H2O': {
        'atom': 'O 0 0 0; H 1.0 0 0; H -0.2774 0.9605 0',
        'n_orb': 7,
        'n_elec': 10,
        'charge': 0,
        'spin': 0,
        'n_cas_orb': 4,
        'n_cas_elec': 4,
    },
    'N2': {
        'atom': 'N 0 0 0; N 0 0 1.098',
        'n_orb': 28,
        'n_elec': 14,
        'charge': 0,
        'spin': 0,
        'n_cas_orb': 6,    # Full valence CAS(6,6)
        'n_cas_elec': 6,
    },
    'C2': {
        'atom': 'C 0 0 0; C 0 0 1.243',
        'n_orb': 28,
        'n_elec': 12,
        'charge': 0,
        'spin': 0,
        'n_cas_orb': 6,
        'n_cas_elec': 6,
    },
}


def setup_system(system_name: str, basis: str = 'sto-3g'):
    """Set up PySCF molecule and mean-field for a benchmark system."""
    spec = SYSTEMS[system_name]

    mol = gto.M(
        atom=spec['atom'],
        basis=basis,
        charge=spec['charge'],
        spin=spec['spin'],
        verbose=3,
    )
    mf = scf.RHF(mol)
    mf.kernel()

    return mol, mf, spec


# ============================================================================
# FCI reference
# ============================================================================

def compute_fci_reference(mol, mf, n_orb: int = None, n_elec: int = None):
    """Compute exact FCI ground state energy.

    Args:
        mol, mf: PySCF objects.
        n_orb:   Number of spatial orbitals (default: all active).
        n_elec:  Number of electrons (default: from mol.nelec).

    Returns:
        (E_FCI, ci_vector) — FCI ground state energy and wavefunction.
    """
    if n_orb is None:
        n_orb = mol.nao
    if n_elec is None:
        n_elec = sum(mol.nelec)

    nelec_a = mol.nelec[0]
    nelec_b = mol.nelec[1]

    h1e = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    h2e = mol.intor('int2e', aosym='s8')
    if n_orb < mol.nao:
        # Truncate to active space
        h1e = h1e[:n_orb, :n_orb]
        from pyscf.ao2mo import _ao2mo
        h2e = _ao2mo.nr_e2(h2e, mf.mo_coeff[:, :n_orb], 's8', 's8')

    from pyscf.fci import direct_nosym
    solver = direct_nosym.FCI()
    solver.verbose = 3
    E_fci, ci = solver.kernel(h1e, h2e, n_orb,
                              (nelec_a, nelec_b), ecore=mf.energy_nuc())

    return E_fci, ci


# ============================================================================
# Krylov-dCI pipeline
# ============================================================================

def run_krylov_dci(system_name: str,
                   basis: str = 'sto-3g',
                   m_max: int = 3,
                   delta_mode: str = 'fixed',
                   svd_threshold: float = 0.0,
                   p_strategy: str = 'cas',
                   p_size: int = None,
                   verbose: bool = True) -> Dict:
    """End-to-end Krylov-dCI computation.

    Args:
        system_name:   'H2', 'H2O', 'N2', or 'C2'.
        basis:         Basis set name.
        m_max:         Maximum Krylov order.
        delta_mode:    'fixed' (use FCI Δ) or 'scf' (self-consistent).
        svd_threshold: SVD truncation threshold (0 = no truncation).
        p_strategy:    P-space selection: 'cas' or 'energy_window'.
        p_size:        Target P-space size (for energy_window strategy).
        verbose:       Print detailed progress.

    Returns:
        Dict with results for each m = 0, 1, ..., m_max.
    """
    from src.determinants import generate_determinants_ms
    from src.hamiltonian import from_pyscf
    from src.partitioning import (partition_cas, partition_energy_window,
                              compute_reference_energy, extract_subspace)
    from src.krylov import (compute_A, compute_H_off_diag, build_H_QP,
                        generate_layer_0, propagate_layer,
                        modified_gram_schmidt, build_krylov_subspace)
    from src.svd_compression import compress_layer
    from src.effective_h import (build_effective_H, compute_with_fixed_delta,
                            self_consistent_iteration,
                            build_H_Qtilde_Qtilde, build_H_PQtilde)

    t_start = time.time()

    # --- Setup ---
    mol, mf, spec = setup_system(system_name, basis)
    n_orb_total = spec['n_orb']
    n_elec_total = spec['n_elec']

    ham = from_pyscf(mol, mf)
    dets_all = generate_determinants_ms(n_orb_total, n_elec_total, ms=0)
    n_fci = len(dets_all)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Krylov-dCI: {system_name}/{basis}")
        print(f"  FCI dimension: {n_fci}")
        print(f"  Active: {n_elec_total}e, {n_orb_total}o")
        print(f"  m_max = {m_max}, Δ mode = {delta_mode}, "
              f"SVD θ = {svd_threshold}")
        print(f"{'='*60}")

    # --- FCI reference ---
    if verbose:
        print("\n[0/3] Computing FCI reference...")
    E_FCI, ci_fci = compute_fci_reference(mol, mf, n_orb_total, n_elec_total)
    if verbose:
        print(f"  E_FCI = {E_FCI:.12f} Ha")

    # --- P/Q partition ---
    if verbose:
        print(f"\n[1/3] Partitioning P/Q (strategy={p_strategy})...")

    if p_strategy == 'cas':
        n_cas_orb = spec['n_cas_orb']
        n_cas_elec = spec['n_cas_elec']
        p_idx, q_idx = partition_cas(
            n_orb_total, n_elec_total, n_cas_orb, n_cas_elec
        )
    elif p_strategy == 'energy_window':
        p_idx, q_idx = partition_energy_window(
            ham, dets_all, window_width=0.5, max_p_size=p_size
        )
    else:
        raise ValueError(f"Unknown P-space strategy: {p_strategy}")

    N = len(p_idx)
    M = len(q_idx)
    p_dets = [dets_all[i] for i in p_idx]
    q_dets = [dets_all[i] for i in q_idx]

    if verbose:
        print(f"  P-space: {N} determinants")
        print(f"  Q-space: {M} determinants")
        print(f"  Compression target: {N}/{n_fci} = "
              f"{100*N/n_fci:.1f}% of FCI")

    # --- H_PP, reference energy ---
    if verbose:
        print("\n[2/3] Building H_PP and computing reference energy...")

    E0 = compute_reference_energy(ham, dets_all, p_idx)
    if verbose:
        print(f"  E^(0) = {E0:.12f} Ha")

    # Build H_PP
    H_PP = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            H_PP[i, j] = ham.matrix_element(p_dets[i], p_dets[j])
    H_PP = 0.5 * (H_PP + H_PP.T)

    # --- Early exit if Q is empty ---
    if M == 0:
        if verbose:
            print(f"\n  Q-space empty — P covers entire {n_fci}-dim FCI space.")
            print(f"  Effective H = H_PP")
        E_result = eigh(H_PP)[0][0]
        return {
            'm': 0,
            'E': E_result,
            'delta_E_mH': (E_result - E_FCI) * 1000,
            'E_FCI': E_FCI,
            'E0': E0,
            'delta_exact': delta_exact,
            'delta_used': delta_exact,
            'n_vecs_total': 0,
            'n_fci': n_fci,
            'N_p': N,
            'M_q': M,
            'compression_ratio': 0.0,
            'layer_sizes': [0],
            'sigma_per_layer': [],
            'converged': True,
            'n_iter': 1,
            't_wall': time.time() - t_start,
        }

    # --- Q-space Hamiltonian structure ---
    if verbose:
        print("\n[3/3] Building Q-space Hamiltonian and Krylov subspace...")

    diag_H_QQ = np.array(
        [ham.diagonal_element(a, b) for a, b in q_dets]
    )
    A_diag = compute_A(E0, diag_H_QQ)
    H_off = compute_H_off_diag(ham, q_dets)
    H_QP_mat = build_H_QP(ham, p_dets, q_dets)

    # --- Krylov subspace construction (layer by layer, with SVD) ---
    # This is done incrementally to support SVD compression at each layer.

    # Compute exact Delta (for fixed mode)
    delta_exact = E_FCI - E0
    if delta_mode == 'fixed':
        delta_use = delta_exact
        if verbose:
            print(f"  Using FCI Delta = {delta_exact:.10f} Ha")
    else:
        delta_use = 0.0

    # Accumulate basis incrementally
    all_basis = np.zeros((M, 0))  # (M, d) orthonormal compressed basis
    layer_sizes = []
    sigma_per_layer = []

    # Layer 0
    layer0_raw = generate_layer_0(H_QP_mat, A_diag)

    if svd_threshold > 0:
        layer0_compressed, sigma0, r0 = compress_layer(
            layer0_raw, A_diag, threshold=svd_threshold, verbose=verbose
        )
        sigma_per_layer.append(sigma0)
        layer0_orth, _ = modified_gram_schmidt(layer0_compressed, all_basis)
    else:
        layer0_orth, _ = modified_gram_schmidt(layer0_raw, all_basis)
        r0 = layer0_orth.shape[1]
        sigma_per_layer.append(np.array([]))

    all_basis = layer0_orth
    layer_sizes.append(r0)

    if verbose:
        svd_tag = f" (SVD θ={svd_threshold})" if svd_threshold > 0 else ""
        print(f"  Layer 0: {N} → {r0} vectors{svd_tag} "
              f"(basis size: {all_basis.shape[1]})")

    # Higher layers
    prev_layer_raw = layer0_raw
    m_actual = 0  # track actual Krylov order (layer index)
    for j in range(1, m_max + 1):
        m_actual = j
        new_raw = propagate_layer(prev_layer_raw, H_off, A_diag, delta_use)

        if svd_threshold > 0:
            new_compressed, sigma_j, rj = compress_layer(
                new_raw, A_diag, threshold=svd_threshold, verbose=verbose
            )
            sigma_per_layer.append(sigma_j)
            new_orth, _ = modified_gram_schmidt(new_compressed, all_basis)
        else:
            new_orth, _ = modified_gram_schmidt(new_raw, all_basis)
            rj = new_orth.shape[1]
            sigma_per_layer.append(np.array([]))

        if rj == 0:
            if verbose:
                print(f"  Layer {j}: Krylov subspace exhausted "
                      f"(all new vectors linearly dependent)")
            break

        all_basis = np.hstack([all_basis, new_orth])
        layer_sizes.append(rj)
        prev_layer_raw = new_raw

        if verbose:
            svd_tag = f" (SVD θ={svd_threshold})" if svd_threshold > 0 else ""
            print(f"  Layer {j}: → {rj} vectors{svd_tag} "
                  f"(basis size: {all_basis.shape[1]})")

    total_basis_dim = all_basis.shape[1]
    if verbose:
        print(f"\n  Total compressed Q-basis: {total_basis_dim} vectors "
              f"(from {M} raw Q determinants, "
              f"ratio = {total_basis_dim/M:.3f})")

    # --- Build effective Hamiltonian ---
    if verbose:
        print(f"\n[Effective Hamiltonian] Building H_P^eff...")

    if total_basis_dim == 0:
        # No Q-space: effective H = H_PP
        E_result = eigh(H_PP)[0][0]
        return {
            'm': m_max,
            'E': E_result,
            'delta_E_mH': (E_result - E_FCI) * 1000,
            'E_FCI': E_FCI,
            'E0': E0,
            'delta_used': delta_use,
            'n_vecs_total': total_basis_dim,
            'n_fci': n_fci,
            'compression_ratio': 0.0,
            'layer_sizes': layer_sizes,
            't_wall': time.time() - t_start,
        }

    # Build H blocks in compressed basis
    H_PQtilde = build_H_PQtilde(ham, all_basis, p_dets, q_dets)
    H_Qtilde_Qtilde = build_H_Qtilde_Qtilde(ham, all_basis, q_dets)

    if delta_mode == 'fixed':
        # Single-shot with exact Δ
        E_krylov, evec = compute_with_fixed_delta(
            H_PP, H_PQtilde, H_Qtilde_Qtilde, E0, delta_exact
        )
        delta_used = delta_exact
        converged = True
        n_iter = 1
    else:
        # Self-consistent iteration
        scf_result = self_consistent_iteration(
            H_PP, H_PQtilde, H_Qtilde_Qtilde, E0,
            delta_init=0.0, verbose=verbose
        )
        E_krylov = scf_result['E_final']
        delta_used = scf_result['delta_final']
        converged = scf_result['converged']
        n_iter = scf_result['n_iter']

    delta_EmH = (E_krylov - E_FCI) * 1000.0  # mH

    if verbose:
        print(f"\n  {'='*50}")
        print(f"  FINAL RESULT (m={m_actual}, comp={all_basis.shape[1]}):")
        print(f"    E_krylov = {E_krylov:.12f} Ha")
        print(f"    E_FCI    = {E_FCI:.12f} Ha")
        print(f"    ΔE       = {delta_EmH:+.3f} mHartree")
        print(f"    Δ used   = {delta_used:.10f} Ha")
        print(f"    SCF conv = {converged} ({n_iter} iters)")
        print(f"    Basis    = {total_basis_dim} vectors "
              f"(layers: {layer_sizes})")
        print(f"    Time     = {time.time()-t_start:.1f}s")
        print(f"  {'='*50}")

    return {
        'm': m_actual,
        'E': E_krylov,
        'delta_E_mH': delta_EmH,
        'E_FCI': E_FCI,
        'E0': E0,
        'delta_exact': delta_exact,
        'delta_used': delta_used,
        'n_vecs_total': total_basis_dim,
        'n_fci': n_fci,
        'N_p': N,
        'M_q': M,
        'compression_ratio': total_basis_dim / n_fci,
        'layer_sizes': layer_sizes,
        'sigma_per_layer': sigma_per_layer,
        'converged': converged,
        'n_iter': n_iter,
        't_wall': time.time() - t_start,
    }


# ============================================================================
# Verification experiments
# ============================================================================

def experiment_1_fixed_delta_convergence(system_name: str = 'H2O',
                                         basis: str = 'sto-3g',
                                         m_max: int = 4,
                                         p_strategy: str = 'cas'):
    """Experiment 1: Verify Krylov convergence with fixed (exact) Δ.

    This removes self-consistency as a confounding variable. We use
    Δ = E_FCI - E0 and check how H_P^eff(m) converges with m.
    """
    print("=" * 70)
    print("EXPERIMENT 1: Fixed-Δ Krylov Convergence")
    print("=" * 70)

    results = []
    for m in range(m_max + 1):
        res = run_krylov_dci(
            system_name, basis, m_max=m,
            delta_mode='fixed', svd_threshold=0.0,
            p_strategy=p_strategy
        )
        results.append(res)

    # Summary table
    print(f"\n{'m':>3s}  {'E (Ha)':>16s}  {'ΔE (mH)':>10s}  "
          f"{'n_vecs':>7s}  {'ratio':>8s}  {'t (s)':>7s}")
    print("-" * 65)
    prev_E = None
    for res in results:
        dE = (f"{(res['E'] - prev_E)*1000:+.3f}" if prev_E is not None
              else "--")
        print(f"{res['m']:3d}  {res['E']:16.12f}  "
              f"{res['delta_E_mH']:+10.3f}  "
              f"{res['n_vecs_total']:7d}  "
              f"{res['compression_ratio']:8.4f}  "
              f"{res['t_wall']:7.1f}  "
              f"dE={dE} mH")
        prev_E = res['E']

    return results


def experiment_2_scf_vs_fixed(system_name: str = 'H2O',
                               basis: str = 'sto-3g'):
    """Experiment 2: Compare self-consistent vs fixed-Δ.

    For a single (best) m, compare:
      - Fixed Δ (from FCI): removes SCF convergence as variable
      - Self-consistent Δ: iterative update
    """
    print("=" * 70)
    print("EXPERIMENT 2: Self-Consistent vs Fixed Δ")
    print("=" * 70)

    res_fixed = run_krylov_dci(
        system_name, basis, m_max=2,
        delta_mode='fixed', svd_threshold=0.0
    )

    res_scf = run_krylov_dci(
        system_name, basis, m_max=2,
        delta_mode='scf', svd_threshold=0.0
    )

    print(f"\n  Fixed Δ: E = {res_fixed['E']:.12f}, "
          f"ΔE = {res_fixed['delta_E_mH']:+.3f} mH")
    print(f"  SCF  Δ: E = {res_scf['E']:.12f}, "
          f"ΔE = {res_scf['delta_E_mH']:+.3f} mH, "
          f"iters = {res_scf['n_iter']}, "
          f"conv = {res_scf['converged']}")
    print(f"  Difference: ΔE_scf - ΔE_fixed = "
          f"{res_scf['delta_E_mH'] - res_fixed['delta_E_mH']:+.6f} mH")

    return res_fixed, res_scf


def experiment_3_svd_compression(system_name: str = 'H2O',
                                  basis: str = 'sto-3g',
                                  m: int = 2):
    """Experiment 3: SVD compression accuracy vs cost trade-off.

    Scan SVD threshold θ and measure:
      - Accuracy: ΔE vs FCI (mHartree)
      - Compression: total basis dimension
      - Key question: Can SVD maintain DMET-like accuracy with
        drastically reduced computational cost?
    """
    print("=" * 70)
    print("EXPERIMENT 3: SVD Compression Accuracy vs Cost")
    print("=" * 70)

    # Fixed Δ mode for clean comparison
    thetas = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    results = []

    # Get baseline (no SVD)
    baseline = run_krylov_dci(
        system_name, basis, m_max=m,
        delta_mode='fixed', svd_threshold=0.0
    )
    E_baseline = baseline['E']
    n_baseline = baseline['n_vecs_total']
    n_fci = baseline['n_fci']

    print(f"\n  Baseline (θ=0, no compression):")
    print(f"    E = {E_baseline:.12f}, n_vecs = {n_baseline}, "
          f"compression = {n_baseline/n_fci:.4f}")

    print(f"\n  {'θ':>8s}  {'E (Ha)':>16s}  {'ΔE (mH)':>10s}  "
          f"{'n_vecs':>7s}  {'ratio':>8s}  {'ΔE vs θ=0':>12s}")
    print("-" * 75)

    for theta in thetas:
        if theta == 0.0:
            results.append({
                'theta': 0.0,
                'E': E_baseline,
                'delta_E_mH': baseline['delta_E_mH'],
                'n_vecs_total': n_baseline,
                'compression_ratio': n_baseline / n_fci,
            })
            continue

        res = run_krylov_dci(
            system_name, basis, m_max=m,
            delta_mode='fixed', svd_threshold=theta
        )
        delta_from_baseline = (res['E'] - E_baseline) * 1000  # mH
        results.append({
            'theta': theta,
            'E': res['E'],
            'delta_E_mH': res['delta_E_mH'],
            'n_vecs_total': res['n_vecs_total'],
            'compression_ratio': res['compression_ratio'],
        })

        print(f"  {theta:8.1e}  {res['E']:16.12f}  "
              f"{res['delta_E_mH']:+10.3f}  "
              f"{res['n_vecs_total']:7d}  "
              f"{res['compression_ratio']:8.4f}  "
              f"{delta_from_baseline:+12.6f} mH")

    # Key finding
    print(f"\n  ═══════════════════════════════════════════════")
    print(f"  KEY METRIC: How small can the subspace be?")
    for r in results:
        delta_mH = r['delta_E_mH']
        ok = "✅" if abs(delta_mH) < 1.6 else "❌"
        print(f"  θ={r['theta']:.0e}: {r['n_vecs_total']} vectors, "
              f"ΔE={delta_mH:+.3f} mH {ok}")

    return results


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Krylov-dCI verification pipeline'
    )
    parser.add_argument('--system', default='H2O',
                        choices=['H2', 'H2O', 'N2', 'C2'])
    parser.add_argument('--basis', default='sto-3g')
    parser.add_argument('--m-max', type=int, default=3,
                        help='Maximum Krylov order')
    parser.add_argument('--experiment', choices=['1', '2', '3', 'all'],
                        default='all',
                        help='Which experiment to run')
    parser.add_argument('--delta-mode', choices=['fixed', 'scf'],
                        default='fixed')
    parser.add_argument('--svd-threshold', type=float, default=0.0,
                        help='SVD truncation threshold')
    parser.add_argument('--p-strategy', default='cas',
                        choices=['cas', 'energy_window'])
    args = parser.parse_args()

    if args.experiment == 'all' or args.experiment == '1':
        experiment_1_fixed_delta_convergence(
            args.system, args.basis, args.m_max, args.p_strategy
        )

    if args.experiment == 'all' or args.experiment == '2':
        experiment_2_scf_vs_fixed(args.system, args.basis)

    if args.experiment == 'all' or args.experiment == '3':
        experiment_3_svd_compression(args.system, args.basis, m=args.m_max)


if __name__ == '__main__':
    main()
