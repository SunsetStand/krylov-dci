# N2/cc-pVDZ CAS(10,10) Three-State Pilot Protocol

## Purpose

This pilot is the first transfer of the state-averaged residual-dressed dmSVD
iteration from the small H2O validation system to the 63,504-determinant
N2/cc-pVDZ CAS(10,10) space.  It tests whether the validated fixed controls
remain numerically and operationally viable at the larger determinant count.

The H2O truncation scan is the prerequisite for this run.  It established that
the equal-weight self-consistent calculation at `svd_eps = 1e-3` is a stable
fixed point with all three energy errors below 0.004 mH.  This pilot therefore
uses that threshold and does not repeat a threshold or weight scan.

## Immutable scope

The pilot is observational.  It must not modify:

- the shared wave-operator or outer Schmidt solver;
- damping, denominator floors, convergence tolerances, or iteration limits;
- H_A, H_B, H_AB, transition-RDM, or Jordan--Wigner conventions;
- P-block selection in response to an incomplete calculation.

The frozen and self-consistent calculations are run from committed code.  A
numerical or resource failure is recorded before any corresponding change is
proposed.

## Molecular and state definition

| Parameter | Value |
|---|---|
| Geometry | `N 0 0 0; N 0 0 1.098` Angstrom |
| Basis | `cc-pVDZ` |
| Reference | RHF followed by common-orbital multi-root CASCI |
| Active space | CAS `(10e, 10o)` |
| Active spin sector | `(n_alpha, n_beta) = (5, 5)`, `M_S = 0` |
| Frozen core | 2 spatial orbitals |
| dmSVD A space | first 5 active spatial orbitals |
| Number of retained roots | 3 |
| State weights | `(1/3, 1/3, 1/3)` |
| P blocks | `n_A = 8, 9, 10` |
| Schmidt threshold | `1e-3` |
| Hamiltonian route | Scheme A, sigma-vector projection |

The retained states are the three lowest roots returned by the common-orbital
CASCI calculation in the stated `M_S = 0` sector.  The pilot does not impose an
additional spin or spatial-symmetry filter.  Root continuity in the outer loop
is determined by physical-state overlaps, not by energy ordering alone.

## Calculations

The job contains exactly two serial calculations:

1. `equal_frozen_eps_1e-3`: construct the equal-weight CASCI Schmidt basis once
   and converge the shared wave operator in that basis.  One outer evaluation
   is intentional.
2. `equal_sc_eps_1e-3`: repeat from the same molecular definition and converge
   the residual-dressed state-averaged Schmidt feedback.

The frozen calculation is the same-threshold control for quantifying the
effect of self-consistency.  It is run first so that a later failure leaves a
usable baseline.

## Fixed numerical controls

| Control | Value |
|---|---:|
| Outer state mixing | `0.7` |
| Outer density tolerance | `1e-6` |
| Outer energy tolerance | `1e-7 Ha` |
| Maximum outer iterations | `12` |
| Wave-operator damping | `0.7` |
| Weighted residual tolerance | `1e-9` |
| Wave energy tolerance | `1e-10 Ha` |
| Maximum inner iterations | `200` |
| Minimum denominator magnitude | `1e-6 Ha` |

These are the H2O validation controls without per-system tuning.

## Required measurements

For both calculations, preserve incrementally:

- all CASCI reference and downfolded energies and per-root errors in mH;
- weighted and maximum absolute energy errors;
- inner convergence, iteration count, and final weighted residual;
- the full outer energy, density, root-permutation, and Schmidt-rank history;
- rectangular `r_A` and `r_B` ranks, total product dimension, and P/Q sizes;
- discarded density weight and all embedded-Hamiltonian build times;
- pipeline and end-to-end wall time;
- exception type and message if a calculation fails.

For the self-consistent calculation, also record each root's energy change
relative to the frozen result.  Slurm elapsed time and maximum resident memory
are collected separately from the scheduler and `/usr/bin/time -v`.

## Classification

- `FROZEN_BASELINE`: finite frozen result with every inner solve converged.
- `PASS`: finite self-consistent result satisfying both outer tolerances with
  every inner solve converged.
- `INNER_NONCONVERGED`: any shared wave-operator solve reaches 200 iterations.
- `OUTER_NONCONVERGED`: the self-consistent run reaches 12 outer iterations.
- `RANK_OSCILLATION`: the last four rectangular-rank patterns alternate.
- `ROOT_REORDER`: any non-identity root permutation; this is an additional
  diagnostic flag rather than an automatic failure.
- `RESOURCE_LIMIT`: the scheduler terminates the job for memory or wall time.
- `ERROR`: an exception, missing root, non-finite result, or P dimension below
  the three-state target.

Chemical accuracy is not a pass condition.  The pilot asks first whether the
larger calculation is stable and affordable under the validated mathematics.

## Resource envelope and decision gates

The Slurm pilot requests one AMD node allocation with 16 CPU cores, 96 GB of
memory, and 24 hours.  Numerical libraries remain single-threaded while the
existing Hamiltonian builder receives the 16 workers.

After the job:

1. Proceed to a wider N2 scan only if the self-consistent calculation passes
   and its resource profile leaves a practical safety margin.
2. Treat a stable but less accurate self-consistent result as method evidence,
   not as permission to retune the run.  Inspect root-by-root changes and
   Schmidt-rank redistribution first.
3. If the wave residual stalls, inspect residual histories and denominators
   before considering a targeted stabilization change.
4. If roots reorder, inspect overlap matrices and reference gaps before
   changing root tracking.
5. If memory or wall time is limiting, preserve the solver mathematics and
   evaluate block or matrix-free `H_QQ` application as a separate engineering
   change.

No solver change is authorized by this protocol alone; the archived pilot
data must expose a concrete, reproducible failure mode.
