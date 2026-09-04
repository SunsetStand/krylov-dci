# N2 CASCI Root-Reproducibility Diagnostic Protocol

## Motivation

The first N2/cc-pVDZ CAS(10,10) three-state pilot completed, but its frozen
and self-consistent calculations did not receive the same CASCI excited-state
reference set.  The ground-state references agreed to about `8e-11 Ha`, while
the two excited-state gaps changed from `(299.258, 345.162) mH` to
`(315.148, 318.133) mH`.  The frozen/self-consistent accuracy comparison is
therefore not controlled.

This diagnostic isolates reference-root generation before any dmSVD or
wave-operator change is considered.

## Immutable scope

The job performs RHF and CASCI only.  It does not build a Schmidt basis,
partition P/Q space, construct an embedded Hamiltonian, or invoke either the
inner or outer state-averaged solver.  Existing solver tolerances and method
code remain unchanged.

## System definition

| Parameter | Value |
|---|---|
| Geometry | `N 0 0 0; N 0 0 1.098` Angstrom |
| Basis | `cc-pVDZ` |
| Reference orbitals | RHF |
| Active space | CAS `(10e, 10o)` |
| Frozen core | 2 spatial orbitals |
| Active spin sector | `(n_alpha, n_beta) = (5, 5)` |
| Production root request | 3 |
| Inventory root request | 8 |

No spin or spatial-symmetry constraint is added: this deliberately reproduces
the current production reference path.

## Replicas

Two ensembles are evaluated serially in one Slurm process:

1. `fresh_0..2`: three repetitions of the exact production initialization
   sequence, each with a new molecule, RHF object, single-root setup CASCI,
   and subsequent three-root CASCI.
2. `shared_mf_0..2`: three independent three-root CASCI objects using one
   fixed converged RHF object and orbital set.

Finally, an eight-root CASCI inventory is computed on the fixed RHF object.
It determines whether the three-root replicas select different members of a
larger low-energy manifold.

## Required measurements

For every CASCI calculation, record:

- total and active-space energies;
- CASCI and FCI-solver convergence flags;
- `<S^2>` and multiplicity for each returned root;
- CI-vector norms and the requested/returned root count;
- wall time.

For every replica pair, record the absolute CI-overlap matrix, the permutation
that maximizes total overlap, matched overlaps, and matched energy differences.
For fresh-RHF replicas, also record the occupied/active orbital overlap matrix
relative to `fresh_0`, including the largest off-diagonal overlap.  Raw CI
overlaps are interpreted only when the molecular-orbital bases agree.

The JSON report is written after every completed replica so partial results
survive a later failure.

## Decision rules

- `REPRODUCIBLE`: matched overlaps exceed `0.999999`, matched energy
  differences are below `1e-8 Ha`, and spin labels agree.
- `ROOT_SUBSET_DRIFT`: a three-root replica maps with high overlap to
  different members of the eight-root inventory.
- `UNCONVERGED_REFERENCE`: CASCI or its FCI solver reports non-convergence.
- `ORBITAL_DRIFT`: fresh RHF active orbitals differ beyond signs and numerical
  noise, making raw cross-replica CI overlaps ambiguous.
- `ERROR`: a calculation raises an exception or returns too few roots.

If the fixed-orbital replicas drift while all calculations claim convergence,
the immediate correction belongs to reference-root control: build one CASCI
reference ensemble and reuse it for both frozen and self-consistent runs.  A
wave-operator stabilization change is not authorized by that outcome.

If reference roots are reproducible, the frozen residual stall is then
isolated as a separate wave-operator experiment using that fixed ensemble.
