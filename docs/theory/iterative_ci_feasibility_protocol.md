# Iterative-CI Feasibility Pre-Registration Protocol

## Purpose

The core scientific question is whether the outer map

```text
C^(k) -> state-averaged reduced densities -> blockwise dmSVD Schmidt bases
      -> P/Q partition and embedded Hamiltonian
      -> shared residual-dressed wave operator Omega
      -> generalized Ritz states -> reconstructed C^(k+1)
      -> root matching, mixing, convergence test
```

converges to accurate, reproducible multi-state solutions **without** an exact
CASCI or FCI seed. That a program runs is not evidence. This protocol fixes the
hypotheses, the controls, the measurements and the numerical thresholds **before
any result is seen**, so that a threshold cannot be moved afterwards.

It also carries two obligations that the literature review turned into
requirements rather than preferences. The novelty case rests almost entirely on
two claims, so the experiments that test them are primary hypotheses here, not
ancillary ablations:

- whether the Schmidt-to-`Omega`-to-Schmidt feedback buys anything a one-shot
  basis does not (H3);
- whether rectangular independent left and right Schmidt ranks beat a symmetric
  allocation **at matched embedded dimension** (H6).

## Immutable scope for this stage

This protocol may add instrumentation, seeding options and ablation switches. It
must not change the Hamiltonian, `H_AB`, transition-RDM or Jordan--Wigner algebra,
and it must not retune denominator floors, damping rules or convergence
definitions in response to a partially completed scan.

Any numerical problem is first recorded as data. A solver change is proposed only
after the complete scan identifies a reproducible failure mode.

The reference roots are read from the immutable bundle built under
`docs/theory/n2_root_targeting_mechanism_protocol.md`. Frozen and self-consistent
runs consume exactly the same bundle.

## Hypotheses

Each is stated so that one measurement can falsify it.

```text
H1  Reachability
    Without an exact CI seed, the outer map reaches the same fixed point as
    an exact-seeded run.
    Falsified if any non-exact seed converges to a state subspace whose
    principal angles against the exact-seeded fixed point exceed the
    subspace tolerance.

H2  Stability
    The fixed point is insensitive to the choice of seed and to small
    symmetry-preserving perturbations.
    Falsified if two seeds from the admissible family converge to fixed
    points that differ by more than the agreement tolerance, or if a
    perturbation below the perturbation scale changes the converged energies
    by more than the agreement tolerance.

H3  Value  [PRIMARY -- decides whether there is a method]
    Self-consistent Schmidt updating gives a reproducible accuracy or
    compression benefit over a frozen Schmidt basis, and the benefit is not
    error cancellation.
    Falsified if the converged self-consistent energies lie within the
    frozen-basis error bar at matched embedded dimension, or if an apparent
    energy gain is not accompanied by a corresponding improvement in the
    RDM distance, the Schmidt projector distance and the root-subspace
    overlap.

H4  Necessity of residual dressing
    Residual dressing carries information rather than being numerical
    decoration.
    Falsified if a run with dressing disabled matches a dressed run within
    the agreement tolerance on every recorded metric.

H5  Hygiene
    The production path never reads exact FCI or CASCI coefficients.
    Falsified by any call-path from initializer, root selector, Schmidt
    builder or solver to an exact CI source.

H6  Rectangular ranks  [PRIMARY -- novelty candidate]
    At matched total embedded dimension, retaining r_A(n) and r_B(n)
    independently is more accurate than the best symmetric allocation.
    Falsified if a symmetric allocation at the same embedded dimension
    matches or beats the rectangular allocation on weighted absolute energy
    error.

H7  Shared versus per-state wave operator
    One shared Omega is sufficient; a per-state Omega does not materially
    improve the target states.
    Falsified if per-state Omega improves the weighted absolute energy error
    by more than the agreement tolerance at matched embedded dimension.
```

H3 and H6 are marked primary because the novelty assessment in
`docs/literature/iterative_ci_schmidt_downfolding_review.md` found the rest of the
construction to be precedented. If H3 is falsified the self-consistent map is
decoration; if H6 is falsified the one unlocated ingredient is lost.

## Gating diagnostic, to be run before any solver run

The residual dressing is a damped preconditioned iteration with matrix
`(1 - w) I + w (A B)`, so convergence requires

```text
w < 2 / (rho(BA) + 1).
```

`rho(BA)` is measured by matrix-free power iteration at about 60 matvecs, as in
`scripts/diagnostics/run_neumann_vs_krylov_comparison.py`. **Every run records
`rho(BA)` and the damping bound it implies.** A run whose damping violates the
bound is recorded as `DAMPING_BOUND_VIOLATED` and is not interpreted as a
convergence failure of the method.

## System definition

| Parameter | H2 smoke | H2O main |
|---|---|---|
| Geometry | `H 0 0 0; H 0 0 0.74` | `O 0 0 0; H 0 0.757 0.586; H 0 -0.757 0.586` |
| Basis | `STO-3G` | `STO-3G` |
| Reference | canonical RHF | canonical RHF |
| Active space | CAS(2e,2o) | CAS(6e,5o) |
| Spin sector | `(1,1)` | `(3,3)`, `M_S = 0` |
| Frozen core | `0` | `2` |
| Determinants | `4` | `100` |
| Schmidt space A | first 1 active orbital | first 3 active spatial orbitals |
| Target | 2 lowest levels | 3 lowest levels |
| P blocks | `n_A = 1, 2` | `n_A = 4, 5, 6` |
| Hamiltonian route | Scheme A | Scheme A |

H2 alone is not sufficient: its symmetry hides integral-index and sign errors.
H2O is the discriminating system and carries the scan. N2 is deferred to Gate E
and is not part of this protocol.

Target states are the lowest energy **levels**, so a degenerate level is retained
whole. Within a degenerate block the state weights are **equal**, which the GOK
ensemble condition requires; unequal weighting inside a degenerate block breaks
point-group invariance and is recorded as a specification error, not a result.

## Initialization family

No initializer may read exact CI coefficients.

| ID | Seed |
|---|---|
| `exact` | exact CASCI roots; **upper-bound control only**, never a production path |
| `hf` | Hartree-Fock determinant, plus the lowest-diagonal determinants needed to reach the target count |
| `cis` | RHF plus all single excitations, diagonalized in that subspace |
| `selci` | a cheap selected-CI wavefunction at a loose threshold |
| `trunc` | truncated CI at fixed excitation rank |
| `rand_0,1,2` | symmetry- and spin-preserving random perturbations of the `cis` seed at perturbation scale `0.1` |

`cis` is expected to be the strongest non-exact seed: the project's own record
shows a CIS-informed P space took S1 from `+636 mH` to `+0.8 mH`. That is a prior
expectation, stated here so that confirming it is not mistaken for a discovery.

## Control groups

| Axis | Settings |
|---|---|
| Schmidt basis | `frozen` (one outer iteration), `self_consistent` |
| Residual dressing | `on`, `off` |
| Wave operator | `shared`, `per_state` |
| Rank allocation | `rectangular`, `symmetric_matched_dimension` |
| Seed | the six families above |

The rank-allocation control is matched on **embedded dimension**, not on
threshold. The rectangular run fixes `svd_eps`; the symmetric run loosens its
threshold until its total embedded dimension is within `2 percent` of the
rectangular one. Comparing rectangular against `min(r_A, r_B)` at the same
threshold is not a valid test, because the rectangular space is also larger.

## Fixed numerical controls

Inherited unchanged from `docs/theory/state_averaged_validation_protocol.md` so
results stay comparable, with one addition.

| Control | Value |
|---|---:|
| Outer state mixing | `0.7` |
| Outer density tolerance | `1e-6` |
| Outer energy tolerance | `1e-7 Ha` |
| Maximum outer iterations | `12` |
| Wave-operator damping | `0.7`, subject to the `rho(BA)` bound |
| Weighted residual tolerance | `1e-9` |
| Wave energy tolerance | `1e-10 Ha` |
| Maximum inner iterations | `200` |
| Minimum denominator magnitude | `1e-6 Ha` |
| Schmidt thresholds scanned | `1e-2, 1e-3, 1e-4` |
| Numerical library threads | `1` |

## Pre-registered thresholds

Fixed now, before any result.

| Quantity | Symbol | Value |
|---|---|---|
| Agreement tolerance, weighted absolute energy | `tol_E` | `0.05 mH` |
| Chemical accuracy reference | | `1.6 mH` |
| Subspace tolerance, largest principal angle | `tol_theta` | `0.02 rad` |
| RDM distance tolerance | `tol_rho` | `1e-4` |
| Schmidt projector distance tolerance | `tol_P` | `1e-3` |
| Seed perturbation scale | | `0.1` |
| Matched-dimension tolerance | | `2 percent` |
| Rank oscillation | | same two rank patterns alternating over the final 4 outer iterations |

A difference below `tol_E` is "not distinguishable", not "equal". A hypothesis is
marked `INCONCLUSIVE` rather than confirmed when the measured difference is below
tolerance but the run did not converge.

## Required measurements

Per outer iteration, for every run:

- all target energies, and error against the reference bundle, per root and
  weighted
- **per-root** residual norms, not only the weighted RMS
- the full inner iteration history, including `Omega` norm
- root-subspace overlap by **principal angles**, with degenerate levels matched as
  blocks; a per-vector permutation is meaningless inside a degenerate block
- state-averaged RDM distance
- **Schmidt projector distance** between consecutive outer iterations; this does
  not currently exist and must be implemented
- per-block `r_A(n)`, `r_B(n)`, total embedded dimension, P and Q dimensions
- discarded singular-value weight
- root permutation, and `<S^2>` and spatial irrep of every tracked root
- `rho(BA)` and the implied damping bound
- wall time and peak RSS

Provenance recorded with every run: commit hash, reference bundle checksum, seed
family, random seed, and exact PySCF, NumPy and SciPy versions.

The JSON report is written after every completed scan point so partial results
survive a later failure.

## Comparison rule

Downfolded energies are compared against **the exact resolvent evaluated at the
same `E_0`**, not only against exact CASCI. Measured on H2O, a Neumann `k=1`
downfolding appears *better* than the exact resolvent when scored against CASCI,
`+0.35 mH` against `-0.69 mH`, because truncation error cancels the error from
evaluating at `E_0` rather than at the self-consistent energy. Scoring only
against CASCI will mistake that cancellation for accuracy.

## Coincidence test

A result that survives only at one parameter setting is not a result. This
project has been burned by exactly that: a Phase 18 result of `|dE| <= 76 mH` was
later shown to be coincidence, and the recorded tell was that "rerunning with any
parameter change destroys the good numbers".

Every confirmed hypothesis must therefore be re-checked under a deliberate
parameter perturbation: mixing `0.7 -> 0.5`, damping within the `rho(BA)` bound,
and the Schmidt threshold moved by one step. A conclusion that does not survive
all three is recorded as `COINCIDENCE_SUSPECTED`.

## Classification

| Label | Meaning |
|---|---|
| `PASS` | the hypothesis is confirmed within its threshold and survives the coincidence test |
| `FALSIFIED` | the stated falsification condition is met |
| `INCONCLUSIVE` | difference below tolerance but convergence not achieved |
| `FROZEN_BASELINE` | a frozen-basis reference point, single outer iteration, not a failure |
| `INNER_NONCONVERGED` | inner residual tolerance not reached within the iteration cap |
| `OUTER_NONCONVERGED` | outer tolerances not reached within the iteration cap |
| `DAMPING_BOUND_VIOLATED` | damping exceeds `2/(rho(BA)+1)`; not a method failure |
| `RANK_OSCILLATION` | rank pattern alternates over the final four outer iterations |
| `ROOT_REORDER` | a non-identity root permutation; diagnostic flag, not automatically a failure |
| `DEGENERATE_LEVEL_SPLIT` | a degenerate level is divided, or weighted unequally |
| `COINCIDENCE_SUSPECTED` | the result does not survive the parameter perturbation |
| `ERROR` | exception, missing root, non-finite value, or P dimension below the target count |

A fallback branch may emit only `INCONCLUSIVE`. No label may be assigned by a
branch that has not tested its own positive condition.

## Decision gates

1. **H5 first.** Until the call-path isolation test passes, no other result is
   admissible, because an exact-CI leak would invalidate everything downstream.
2. **H3 decides continuation.** If self-consistency does not beat a frozen basis
   at matched embedded dimension, and the difference is not explained by the RDM
   and projector metrics, the feasibility case fails and the project should stop
   and reconsider rather than proceed to N2.
3. **H6 decides one novelty claim.** If a symmetric allocation matches the
   rectangular one at matched dimension, the rectangular-rank claim is dropped
   from the paper. That is a reporting decision, not a reason to change the code.
4. **H1 and H2 gate Gate E.** The N2 pilot is not run until at least three
   independent non-exact seeds reach the same fixed point on H2O.
5. **H4 and H7 are informative, not gating.** A negative result on either is a
   publishable simplification, not a failure.
6. At least one negative control must fail. If every ablation passes, the
   experiment lacks discriminating power and the design, not the method, is at
   fault.

This protocol authorizes no change to the Hamiltonian construction, and no
retuning of stability parameters in response to partial results.
