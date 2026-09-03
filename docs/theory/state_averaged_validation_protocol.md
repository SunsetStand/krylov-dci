# State-Averaged Self-Consistent dmSVD Validation Protocol

## Purpose

This protocol determines whether the state-averaged residual-dressed dmSVD
iteration remains numerically stable when the Schmidt basis is genuinely
truncated.  The existing H2/STO-3G test establishes algebraic and interface
correctness at essentially full rank; it does not test compression-induced
basis changes.

The immediate questions are:

1. Does the inner shared wave-operator iteration converge for every retained
   state and truncation threshold?
2. Does feedback from dressed states to the state-averaged density converge,
   or does the discontinuous Schmidt cutoff cause rank oscillation?
3. Are roots retained continuously across outer iterations?
4. Does self-consistency improve or degrade the frozen-Schmidt energy error?
5. How do rectangular ranks `r_A(n) != r_B(n)` change with the threshold and
   state weights?

## Immutable scope for this experiment

The validation run must not modify:

- the wave-operator solver;
- the outer state-averaged Schmidt solver;
- H_A, H_B, or H_AB construction;
- transition-RDM or Jordan--Wigner conventions;
- denominator floors, damping rules, or convergence definitions in response
  to a partially completed scan.

Any numerical problem is first recorded as data.  A solver change is proposed
only after the complete scan identifies a reproducible failure mode.

The two pre-existing legacy failures in sparse-sigma API compatibility and the
frozen-core CAS Hamiltonian are outside this protocol.

## Reference smoke test

The H2/STO-3G two-state test remains the exact wiring check:

| Parameter | Value |
|---|---:|
| CAS | `(2e, 2o)` |
| State weights | `(0.5, 0.5)` |
| P blocks | `n_A = 1, 2` |
| Schmidt threshold | `1e-10` |
| Required maximum CASCI error | `1e-6 mH` |

This test must pass before and after the molecular truncation scan, but its
result is not evidence that the truncated outer iteration is stable.

## H2O truncated state-averaged scan

### Molecular and active-space definition

| Parameter | Value |
|---|---|
| Geometry | `O 0 0 0; H 0 0.757 0.586; H 0 -0.757 0.586` |
| Basis | `STO-3G` |
| Reference | RHF followed by common-orbital multi-root CASCI |
| Active space | CAS `(6e, 5o)` |
| Active spin sector | `(n_alpha, n_beta) = (3, 3)` |
| Frozen core | 2 spatial orbitals |
| dmSVD A space | first 3 active spatial orbitals |
| Number of states | 3 |
| P blocks | `n_A = 4, 5, 6` |
| Hamiltonian route | Scheme A, sigma-vector projection |

The P-space selection is fixed for the entire scan.  A run that loses a target
because its P projection becomes singular is reported; P is not enlarged
mid-scan.

### Scan matrix

The scan contains nine independent calculations:

| Family | Weights | Schmidt thresholds | Outer treatment |
|---|---|---|---|
| `equal_frozen` | `(1/3, 1/3, 1/3)` | `1e-2, 1e-3, 1e-4` | one outer evaluation |
| `equal_sc` | `(1/3, 1/3, 1/3)` | `1e-2, 1e-3, 1e-4` | self-consistent |
| `biased_sc` | `(0.60, 0.25, 0.15)` | `1e-2, 1e-3, 1e-4` | self-consistent |

`frozen` means that the initial CASCI state-averaged Schmidt basis is built
once and the shared wave operator is converged in that basis.  It is
implemented as one outer iteration; outer non-convergence is expected and is
not counted as a failure for this family.

### Fixed numerical controls

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

These values are held fixed across all scan points.  The scan does not tune a
parameter separately for a difficult threshold.

## Required measurements

For every calculation, record:

- all three CASCI reference and downfolded energies;
- per-root errors in mH;
- weighted and maximum absolute energy error;
- inner convergence, iteration count, and final weighted residual;
- outer convergence and iteration count;
- the full outer history of energy and density changes;
- root permutations across outer iterations;
- per-block `r_A` and `r_B`, total product dimension, and P/Q dimensions;
- discarded density weight;
- Hamiltonian-build and total wall times;
- the exception and completed scan points if a calculation fails.

For each self-consistent point, additionally report the per-root energy change
relative to the equal-weight frozen calculation at the same threshold.

## Classification rules

The scan labels outcomes rather than silently adjusting the solver:

- `PASS`: all values are finite, the inner iteration converged, and a
  self-consistent run met both outer tolerances.
- `FROZEN_BASELINE`: the one-step frozen calculation completed with a
  converged inner iteration.
- `INNER_NONCONVERGED`: the shared wave operator reached its iteration limit.
- `OUTER_NONCONVERGED`: a self-consistent run reached 12 outer iterations.
- `RANK_OSCILLATION`: the final four outer iterations alternate between two
  rectangular-rank patterns.
- `ROOT_REORDER`: any recorded root permutation is non-identity.  This is a
  diagnostic flag, not automatically a failure.
- `ERROR`: an exception, missing root, non-finite result, or P dimension below
  the number of target states.

Chemical accuracy is not a hard pass condition in this stage.  Energy errors
are used to compare frozen and self-consistent behavior, while residual and
density convergence determine numerical stability.

## Decision gates after the scan

1. If every self-consistent point passes, proceed to the N2/cc-pVDZ CAS(10,10)
   three-state pilot without changing the solver.
2. If rank oscillation is isolated, document the singular values crossing the
   cutoff before considering rank hysteresis or density damping changes.
3. If the inner residual stalls near small denominators, record the minimum
   denominator and affected roots before evaluating a level shift.
4. If roots reorder, inspect physical-state overlaps and near-degeneracies
   before changing root tracking.
5. If memory or runtime, rather than convergence, is limiting, preserve the
   mathematics and move next to block/matrix-free `H_QQ` application.

The complete scan report is the evidence required to authorize any solver
modification.
