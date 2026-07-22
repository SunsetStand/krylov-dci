#!/usr/bin/env python3
"""
Command-line interface for dmSVD + Krylov-dCI combined method.

Usage:
    python scripts/run_dm_svd_dci.py \
        --atom 'N 0 0 0; N 0 0 1.098' \
        --basis cc-pVDZ \
        --n-active 10 --n-alpha 5 --n-beta 5 --n-core 2 --n-occ 5 \
        --svd-eps 1e-3 \
        --mode gs \
        --p-blocks 8,9,10 \
        --m-max 1 \
        --n-workers 4 \
        --output-dir ./results/n2_cas10

Modes:
    gs  — Ground-state only (single-state dmSVD)
    sa  — State-averaged dmSVD (specify --sa-states N)

Parallel:
    Use --n-workers N to enable ThreadPoolExecutor parallelism for
    sigma-vector computation. Set OMP_NUM_THREADS=1 in the environment
    before running (contract_2e is C-level and BLAS parallelism should
    be limited to 1 thread when using Python-level threading).
"""

import sys, os, argparse
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def parse_args():
    p = argparse.ArgumentParser(
        description='dmSVD + Krylov-dCI: Schmidt basis combined with Löwdin downfolding',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ground-state only, serial
  python run_dm_svd_dci.py --mode gs --n-workers 1

  # Ground-state, 16-thread parallel
  OMP_NUM_THREADS=1 python run_dm_svd_dci.py --mode gs --n-workers 16

  # State-averaged (5 states), parallel
  OMP_NUM_THREADS=1 python run_dm_svd_dci.py --mode sa --sa-states 5 --n-workers 16

  # Custom P-blocks
  python run_dm_svd_dci.py --p-blocks 7,8,9,10
        """)

    # ── System parameters ──
    p.add_argument('--atom', type=str, default='N 0 0 0; N 0 0 1.098',
                   help='Molecular geometry (PySCF format)')
    p.add_argument('--basis', type=str, default='cc-pVDZ',
                   help='Basis set name')
    p.add_argument('--n-active', type=int, default=10,
                   help='Number of active spatial orbitals')
    p.add_argument('--n-alpha', type=int, default=5,
                   help='Number of alpha electrons in active space')
    p.add_argument('--n-beta', type=int, default=5,
                   help='Number of beta electrons in active space')
    p.add_argument('--n-core', type=int, default=2,
                   help='Number of frozen core orbitals')
    p.add_argument('--n-occ', type=int, default=5,
                   help='Number of occupied (A-space) orbitals for occ-virt partition')
    p.add_argument('--ms', type=int, default=0,
                   help='2*Sz quantum number')

    # ── dmSVD parameters ──
    p.add_argument('--svd-eps', type=float, default=1e-3,
                   help='SVD truncation threshold for Schmidt decomposition')

    # ── Mode ──
    p.add_argument('--mode', type=str, default='gs', choices=['gs', 'sa'],
                   help='SVD mode: gs (ground-state only) or sa (state-averaged)')
    p.add_argument('--sa-states', type=int, default=5,
                   help='Number of states for state-averaged SVD (only if --mode sa)')

    # ── P/Q partition ──
    p.add_argument('--p-blocks', type=str, default='8,9,10',
                   help='Comma-separated n values for P-space (default: 8,9,10)')

    # ── Krylov-dCI ──
    p.add_argument('--m-max', type=int, default=1,
                   help='Maximum Krylov propagation order (0 or 1)')
    p.add_argument('--delta', type=float, default=0.0,
                   help='Energy shift in Bloch resolvent (Ha)')
    p.add_argument('--lindep-threshold', type=float, default=1e-10,
                   help='Linear dependence threshold for MGS')

    # ── Parallel ──
    p.add_argument('--n-workers', type=int, default=1,
                   help='Number of parallel threads for sigma-vector computation')

    # ── Output ──
    p.add_argument('--output-dir', type=str, default=None,
                   help='Directory to save results JSON')
    p.add_argument('--quiet', action='store_true',
                   help='Suppress verbose output')

    return p.parse_args()


def main():
    args = parse_args()

    # Parse p_blocks
    p_blocks = [int(x.strip()) for x in args.p_blocks.split(',')]

    # Determine sa_states
    sa_states = 1
    if args.mode == 'sa':
        sa_states = args.sa_states

    verbose = not args.quiet

    # Import and run
    from dm_svd_dci.pipeline import run_dm_svd_dci

    print("=" * 70)
    print("dmSVD + Krylov-dCI Pipeline")
    print("=" * 70)
    print(f"  System:    {args.atom.strip()}, {args.basis}")
    print(f"  CAS:       ({args.n_active},{args.n_alpha + args.n_beta})")
    print(f"  n_core:    {args.n_core}, n_occ (A-space): {args.n_occ}")
    print(f"  SVD mode:  {args.mode} (sa_states={sa_states})")
    print(f"  SVD eps:   {args.svd_eps}")
    print(f"  P blocks:  {p_blocks}")
    print(f"  m_max:     {args.m_max}")
    print(f"  Workers:   {args.n_workers}")
    print(f"  Output:    {args.output_dir or '(none)'}")
    print()

    results = run_dm_svd_dci(
        atom=args.atom,
        basis=args.basis,
        n_active=args.n_active,
        n_active_elec=(args.n_alpha, args.n_beta),
        n_core=args.n_core,
        n_occ=args.n_occ,
        ms=args.ms,
        svd_eps=args.svd_eps,
        sa_states=sa_states,
        p_blocks=p_blocks,
        m_max=args.m_max,
        delta=args.delta,
        lindep_threshold=args.lindep_threshold,
        n_workers=args.n_workers,
        output_dir=args.output_dir,
        verbose=verbose,
    )

    # Final status
    if 'error' in results:
        print(f"\nERROR: {results['error']}")
        sys.exit(1)

    dE_m0 = results.get('dE_m0_mH', float('nan'))
    dE_m1 = results.get('dE_m1_mH', None)

    print(f"\n{'=' * 70}")
    print("Final Results")
    print(f"{'=' * 70}")
    print(f"  E(FCI)     = {results['E_fci']:.12f} Ha")
    print(f"  E(m=0)     = {results['E_eff_m0']:.12f} Ha  (ΔE = {dE_m0:+.3f} mH)")
    if dE_m1 is not None:
        print(f"  E(m=1)     = {results['E_eff_m1']:.12f} Ha  "
              f"(ΔE = {dE_m1:+.3f} mH)")
    print(f"  Schmidt:   r_total={results['schmidt_metrics']['r_total']}, "
          f"D={results['partition_info']['D_total']}")
    print(f"  P|Q:       |P|={results['partition_info']['P_dim']}, "
          f"|Q|={results['partition_info']['Q_dim']}")
    print(f"  Krylov:    r₀={results['krylov_dims']['r0']}" +
          (f", r₁={results['krylov_dims'].get('r1', 'N/A')}"
           if args.m_max >= 1 else ""))

    # Chemical accuracy check
    threshold_mH = 1.0  # 1 mH = chemical accuracy
    if abs(dE_m0) < threshold_mH:
        status = "✓ WITHIN chemical accuracy (< 1 mH)"
    elif abs(dE_m0) < 1.6:
        status = "✓ Within 1 kcal/mol (~1.6 mH)"
    else:
        status = f"✗ {abs(dE_m0):.1f} mH above chemical accuracy"
    print(f"\n  Status: {status}")


if __name__ == '__main__':
    main()