# N2 Lowest-Three-Ms=0 Root-Selection Diagnostic Protocol

## Objective

Determine why the default three-root N2/cc-pVDZ CAS(10,10) CASCI call does
not return the lowest three eigenstates in the fixed `(N_alpha,N_beta)=(5,5)`
sector.  In particular, test the proposed causal chain

1. the low triplet roots are dominated by single excitations relative to the
   canonical RHF determinant;
2. Brillouin's theorem makes the RHF-to-single matrix elements vanish;
3. the default three-vector Davidson initial subspace therefore has no
   component in one or more target-state symmetry sectors; and
4. adding a single-excitation-informed initial subspace recovers the true
   three lowest `M_S=0` roots.

This is a reference-layer diagnostic.  It must not construct a Schmidt basis,
run the wave-operator solver, or modify production solver behavior.

## Important distinction from the archived selected-CI issue

The archived HFPT2 selected-CI workflow explicitly ranked determinants by
`|<D|H|HF>|^2/(E_HF-H_DD)`.  It therefore assigned exactly zero score to
canonical-HF single excitations and omitted them unless they were forced into
the seed.

The current state-averaged pipeline instead calls PySCF full CASCI/FCI over all
`C(10,5)^2 = 63,504` determinants.  No single or double determinant is removed
from the Hamiltonian.  A similar failure can nevertheless occur in the
iterative diagonalizer if its initial vectors have zero projection onto an
invariant symmetry sector.  The diagnostic must distinguish explicit
determinant truncation from insufficient initial-subspace coverage.

## Fixed system

| Item | Value |
|---|---|
| Geometry | N--N = 1.098 Angstrom |
| Basis | cc-pVDZ |
| Reference orbitals | canonical RHF |
| Frozen core | 2 orbitals |
| Active space | CAS(10,10) |
| Spin sector | `(N_alpha,N_beta)=(5,5)`, hence `M_S=0` |
| Target ensemble | three lowest energies in this fixed spin-projection sector |
| Inventory | lowest eight roots from one common Hamiltonian |

All comparisons must reuse the same RHF orbitals and the same active-space
integrals.

## Measurements

### 1. Exact content of the default initial guesses

Use the installed PySCF solver's own `get_init_guess` method with `nroots=3`
and `nroots=8`.  For each initial vector, record

- the dominant determinant and its excitation rank relative to RHF;
- its diagonal Hamiltonian element;
- its projection onto every converged eight-root inventory state.

Also record the total projection of every inventory root onto the span of the
default three-vector and eight-vector initial spaces.  For an exact eigenstate
`|Psi_k>`, a zero initial projection remains zero under every Krylov power,
because `<Psi_k|H^m|g> = E_k^m <Psi_k|g>`.

### 2. Brillouin coupling test

Apply the full active-space Hamiltonian to the RHF determinant and decompose
the resulting vector by excitation rank.  Report the norm and maximum absolute
element in the single-excitation sector.  Canonical RHF should give numerical
zero, while the double-excitation sector remains coupled.

### 3. Excitation character of the inventory roots

For each of the eight roots, report the CI norm carried by excitation ranks
0, 1, 2, and greater than or equal to 3 relative to RHF, together with
`<S^2>`, multiplicity, and the leading determinants.  This tests whether the
missing low roots are in fact single-excitation dominated.

### 4. Causal initial-guess interventions

Run three additional three-root calculations on the same Hamiltonian:

1. `inventory_seeded`: initialize from inventory roots 0, 1, and 2.  This is a
   reachability control, not a production prescription.
2. `expanded_default_seeded`: request three eigenvalues but initialize from the
   solver's eight-root default guess set.  This isolates initial-subspace size
   from determinant character.
3. `cis_seeded`: diagonalize the Hamiltonian in the RHF-plus-all-singles space
   and use its three lowest vectors as the full-CI Davidson guesses.

Match every returned root to the eight-root inventory by maximum absolute CI
overlap.  The target mapping is `[0,1,2]` with monotonically increasing
energies.

## Interpretation rules

- `EXPLICIT_TRUNCATION` is ruled out if the solver applies the full-CI sigma
  operator to 63,504-dimensional vectors and the single-excitation components
  are present in the inventory vectors.
- `INITIAL_SUBSPACE_COVERAGE` is established if the default call maps to a
  non-lowest subset, one or more target roots have negligible projection onto
  the default three-vector initial span, and either controlled seed recovers
  `[0,1,2]`.
- `BRILLOUIN_SINGLE_SEEDING` is established as the specific mechanism only if
  RHF-to-single coupling is numerically zero, the missing target roots have
  substantial single-excitation weight, the default three guesses contain no
  useful projection onto those roots, and the CIS-informed guesses recover
  `[0,1,2]`.
- If the CIS-informed calculation still misses `[0,1,2]`, investigate explicit
  point-group/spin-adapted root targeting before changing the dmSVD solver.

## Acceptance and next action

The diagnostic passes when all calculations converge, the overlap mappings
are unambiguous, and the evidence selects one of the mechanisms above.

If initial-subspace coverage is confirmed, the production correction must be
made at the immutable CASCI reference-bundle boundary.  Frozen and
self-consistent downfolding must consume exactly the same stored three roots.
No wave-operator or dmSVD stabilization change is justified by this test.
