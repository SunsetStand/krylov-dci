# N2 Root-Targeting Mechanism Diagnostic Protocol

## Purpose

The lowest-three-`M_S=0` root-selection diagnostic
(`docs/theory/n2_lowest_ms0_root_selection_protocol.md`) established that the default
three-root CASCI call does not return the three lowest roots of the fixed `M_S = 0`
sector, and the CASCI root-reproducibility diagnostic
(`docs/theory/n2_casci_root_reproducibility_protocol.md`) established that this
invalidates any frozen versus self-consistent accuracy comparison built on it. Neither
protocol identified the mechanism, and neither authorized a production correction.

This protocol determines the mechanism and specifies a deterministic, non-oracle
procedure for obtaining a fixed reference root set. It answers five questions:

1. Which mechanism excludes the missing roots from the returned set?
2. Is the exclusion a property of the initial subspace, of the number of tracked
   roots, of the spin sector, of the spatial symmetry sector, or of the eigensolver's
   convergence test?
3. Can a procedure that never reads exact reference roots recover the target set?
4. Is the recovered set reproducible across independent rebuilds and across
   numerical-library versions?
5. Is the target set itself well defined, given the degeneracy structure of the
   spectrum?

## Immutable scope for this experiment

This is a reference-layer diagnostic. It must not construct a Schmidt basis, call the
downfolded wave-operator solver, or modify production solver behavior. It must not
change dmSVD thresholds, wave-operator damping, denominator floors, or any inner or
outer convergence definition.

Any numerical problem is first recorded as data. A solver change is proposed only
after the complete diagnostic identifies a reproducible failure mode.

## Prior result that motivated this protocol

Job `16990` on the group cluster reported classification
`INITIAL_SUBSPACE_COVERAGE_CONFIRMED`. That label is the final `else` branch of the
classifier in `scripts/diagnostics/run_n2_lowest_ms0_root_selection.py`: it is emitted
only after every positive test has already failed, so it asserts a mechanism on no
evidence. It must not be treated as a confirmed result. The classification logic is
corrected as part of this protocol.

## Hypotheses

Each hypothesis is stated so that a single measurement can falsify it.

```text
A  Brillouin-only
   The missing roots are excluded because the RHF determinant has vanishing
   Hamiltonian coupling to single excitations, so a single-excitation-informed
   initial subspace recovers them.
   Falsified if a CIS-informed seed that spans the missing roots still fails,
   or if the missing roots are recovered without any singles-informed seed.

B  Spatial-symmetry-sector coverage
   The default initial subspace has identically zero projection on the spatial
   irreducible representation of the missing roots, and a projection that starts
   at zero remains zero under Krylov expansion.
   Falsified if the measured projection of the default guess span on a missing
   root is not numerically zero.

C  Spin-sector or root-targeting
   A spin-incomplete direct_spin1 Davidson solve that treats singlets and
   triplets together mis-targets roots.
   Falsified if spin-resolved solves reproduce the same defective set.

D  Numerical convergence
   The solver's convergence flag does not certify that the returned roots are the
   globally lowest n eigenvalues.
   Falsified if every run that reports convergence also returns the lowest n
   energies.

E  Root-count and root homing
   With n tracked roots the solver discards the Ritz vector carrying a target
   root's character before it converges, even when the initial subspace does span
   that root.
   Falsified if requesting more roots than needed never changes which roots are
   obtained, given a fixed initial subspace.
```

Hypotheses B, D and E are not mutually exclusive and may act together.

## Fixed system

| Parameter | Value |
|---|---|
| Geometry | `N 0 0 0; N 0 0 1.098` (Angstrom) |
| Basis | `cc-pVDZ` |
| Reference | canonical RHF, `symmetry=True` |
| Point group | `Dooh`, with `D2h` used where PySCF requires an Abelian group |
| Frozen core | `2` |
| Spin sector | `(n_alpha, n_beta) = (5, 5)`, `M_S = 0` |
| Legacy active space | CAS(10e,10o), `63504` determinants |
| Corrected active space | CAS(10e,9o), `15876` determinants |
| Inventory root request | `8` or more |

## The active-space defect

The legacy CAS(10e,10o) active space is not symmetry complete. The active molecular
orbital irreducible representations are

```text
A1g  A1u  A1g  E1uy  E1ux  E1gx  E1gy  A1u  A1g  E1uy
```

`E1uy` appears twice and `E1ux` once. The last active orbital is one member of a
degenerate `E1u` pair whose partner, degenerate with it at `0.87230 Ha`, is excluded.
The active space therefore breaks the `x`/`y` symmetry, and the two components of the
lowest `3Pi_g` state are split by `2.99 mH`. That splitting is an artifact of the
active space, not physics.

Consequence for the target definition: with the defect present, "the three lowest
roots" silently keeps one member of a near-degenerate pair and discards the other.
With the defect corrected, the pair becomes exactly degenerate and the choice of which
member to keep is numerically arbitrary. The target set must therefore be defined by
energy levels, not by root count.

| Candidate active space | Determinants | `3Pi_g` splitting | Status |
|---|---|---|---|
| CAS(10e,10o) | `63504` | `2.99 mH` | Defective, replaced |
| CAS(10e,9o) | `15876` | `0.000 mH` | Adopted |
| CAS(10e,11o) | `213444` | `0.000 mH` | Rejected, `3.4x` cost |
| CAS(10e,8o) | `3136` | `0.000 mH` | Rejected, drops correlation |

## Target definition

The target is the three lowest energy **levels** of the fixed `M_S = 0` sector, which
in CAS(10e,9o) is **four** determinant-space roots:

| Level | Roots | Degeneracy | Spatial irrep | `S^2` |
|---|---|---|---|---|
| 0 | `0` | 1 | `A1g` | `0.000` |
| 1 | `1` | 1 | `A1u` | `2.000` |
| 2 | `2, 3` | 2 | `E1gy`, `E1gx` | `2.000` |

A degenerate level is retained as a complete block. Splitting a degenerate block is
recorded as a failure, because the individual eigenvectors within it are an arbitrary
rotation and are not reproducible.

## Calculations

| ID | Description |
|---|---|
| `repro_legacy` | Rerun the existing diagnostic unchanged on CAS(10e,10o); compare every recorded quantity against job `16990` |
| `inventory` | Common-Hamiltonian solve for at least eight roots; per-root energy, `S^2`, multiplicity, spatial irrep, excitation-rank weights and residual norm |
| `default_n` | Solve requesting exactly the number of target roots |
| `overshoot_n_plus_m` | Solve requesting `n_target + m` roots for increasing `m`, retaining the lowest `n_target` by energy |
| `symmetry_resolved` | Solve the lowest roots separately in each `D2h` irreducible representation, merge, and re-sort by energy |
| `spin_resolved` | Solve with the total spin constrained to singlet and to the `M_S = 0` component of triplet, merge, and re-sort |
| `symmetry_and_spin_resolved` | Combined constraint |
| `cis_seeded_matched` | Seed from the RHF-plus-all-singles subspace using enough CIS roots to span the target irreps, not merely the lowest three |
| `oracle_seeded` | Seed from the exact inventory roots; a reachability control only, never a production procedure |

## Fixed numerical controls

| Control | Value |
|---|---|
| Maximum Davidson subspace | `30`, recorded |
| Maximum Davidson cycles | `400`, recorded |
| FCI convergence tolerance for bundle-quality solves | `1e-12` |
| Non-convergence residual threshold | `1e-4` |
| Degenerate-level tolerance | `1e-9 Ha` |
| Energy agreement threshold | `1e-8 Ha` |
| Numerical library threads | `1` for OMP, MKL, OpenBLAS and NumExpr |

## Required measurements

For every returned root of every calculation:

- total energy, and excitation energy relative to the lowest root
- explicit residual norm `||Hc - Ec||` computed independently of the solver's own
  convergence flag, and the energy variance `<c|H^2|c> - <c|H|c>^2`, interpreted
  according to the limitation recorded below
- `S^2` and multiplicity
- spatial irreducible representation of the wavefunction
- CI weight decomposed by excitation rank from the RHF determinant, as a full vector
  over ranks, not only the leading determinants

For every initial guess family:

- the projection of each inventory root onto the span of the guess vectors
- the dominant determinant of each guess vector, its occupation and its diagonal
  Hamiltonian element

For every pair of calculations compared:

- absolute CI overlap matrix, and subspace principal angles between degenerate blocks
- the overlap-maximizing assignment, with degenerate blocks matched as blocks
- matched energy differences

Environment provenance recorded with every report: commit hash, and the exact
versions of PySCF, NumPy and SciPy. The root-selection outcome has been observed to
differ between library versions, so version capture is mandatory, not optional.

The JSON report is written after every completed calculation so partial results
survive a later failure.

## What the residual norm can and cannot certify

A small residual certifies that a returned pair is an eigenpair. It does **not**
certify that it is one of the lowest eigenpairs. A solve that homes on roots tens of
millihartree above the target returns residuals indistinguishable from a correct
solve, because those roots are genuine eigenvectors that simply are not the lowest
ones. Measured on CAS(10e,9o) at `conv_tol = 1e-12`:

| Solve | Energy error of the four returned roots | Residual norms |
|---|---|---|
| `nroots = 4` | `0.0`, `+18.99`, `+27.27`, `+50.38 mH` | `5.9e-07` and below |
| `nroots = 6`, lowest four | all `0.0` | `5.8e-07` and below |

The wrong-root solve produced residuals as small as the correct one. The residual
test is therefore retained only as a **non-convergence detector**, with a loose
threshold of `1e-4`: genuinely unconverged vectors measure around `1e-3`, converged
ones between `1e-7` and `1e-5`. The operative correctness test is margin stability,
defined below. A solver convergence flag is likewise insufficient: every wrong-root
solve recorded here reported `converged = True` for every root.

## Classification

Only a non-oracle, executable intervention that stably recovers the target set may be
recorded as confirmed.

| Label | Meaning |
|---|---|
| `BRILLOUIN_SINGLE_SEEDING_CONFIRMED` | A CIS or singles-informed seed, carrying no exact root information, stably recovers the target set |
| `SYMMETRY_RESOLVED_TARGETING_CONFIRMED` | Solving per spatial irreducible representation and merging stably recovers the target set |
| `SPIN_RESOLVED_TARGETING_CONFIRMED` | Solving per total spin and merging stably recovers the target set |
| `GENERIC_INITIAL_SUBSPACE_CONFIRMED` | Some procedure using no exact root information, such as overshoot-and-verify, stably recovers the target set |
| `ORACLE_SEED_ONLY` | Only seeding with exact reference roots recovers the target set |
| `ROOT_TARGETING_MECHANISM_UNRESOLVED` | No non-oracle procedure has been shown to stably recover the target set |
| `UNCONVERGED_REFERENCE` | Any calculation failed its convergence or residual test |
| `DEGENERATE_LEVEL_SPLIT` | A returned set divides a degenerate level |

A fallback branch may emit only `ROOT_TARGETING_MECHANISM_UNRESOLVED`. No label may be
assigned by an `else` clause that has not tested its own positive condition.

"Stably" means the target set is recovered on at least three independent rebuilds of
the molecule, reference and Hamiltonian, with agreement to `1e-8 Ha`.

## Decision gates

1. If the measured projection of the default guess span on a missing root is
   numerically zero, hypothesis B is confirmed for that guess family, and any
   remedy must change the initial subspace or the number of tracked roots rather
   than the convergence tolerance.
2. If a run reports convergence while returning energies above the inventory values,
   hypothesis D is confirmed, and the solver's convergence flag is thereafter
   insufficient evidence. Residual norms become a required acceptance test.
3. If an enlarged initial subspace spans a target root but the run still misses it,
   hypothesis E is confirmed, and the number of requested roots, not the guess, is
   the operative control.
4. If a non-oracle procedure is confirmed, it is promoted into the reference-bundle
   builder and locked by a regression test. Frozen and self-consistent downfolding
   must then consume exactly the same stored bundle.
5. This protocol authorizes no change to the dmSVD or wave-operator layers.

## Reference-bundle requirement

The deterministic procedure is not a fixed root count. The required margin is
system-dependent, so the builder must escalate and self-validate:

```text
given n_target and margin m:
  solve for n_target + m roots
  accept only if
    (a) every returned residual norm < 1e-4
        (non-convergence detector only; it cannot detect wrong-root
         selection, so it is necessary but far from sufficient)
    (b) no degenerate level straddles the n_target boundary,
        tested with the degenerate-level tolerance
    (c) the lowest n_target energies are unchanged, to 1e-8 Ha,
        when m is increased by one
  otherwise increase m and repeat
```

Criterion (c) is the operative test. It is the only one of the three that
discriminates a correct root set from a converged but wrong one.

Independently, the symmetry-resolved procedure is preferred where an Abelian point
group is available, because it guarantees coverage of every irreducible
representation by construction rather than relying on a guess happening to span
them, and because it keeps the members of a degenerate level together: each
component is the lowest root of its own irreducible representation. The bundle
builder runs both and requires them to agree to the energy agreement threshold.

The bundle stores geometry, basis, frozen core, active-space definition, molecular
orbital coefficients, active-space integrals, the target CI vectors, energies, `S^2`,
spatial irreps, the level grouping that identifies degenerate blocks, residual norms,
the commit hash, and the exact library versions. It is checksummed and immutable.
