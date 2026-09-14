# Iterative-CI feasibility: research state

Updated at commit `08c2144` on branch `research/iterative-ci-feasibility`,
based on `origin/feat/residual-dressed-sc-dmsvd` at `e218187`.

## The question

Can the outer map

```text
C^(k) -> state-averaged densities -> blockwise dmSVD/Schmidt bases
      -> P/Q partition and embedded Hamiltonian
      -> shared residual-dressed wave operator Omega
      -> generalized Ritz states -> reconstructed C^(k+1)
      -> root matching, mixing, convergence test
```

reach accurate, reproducible multi-state solutions without exact CASCI or FCI
seeding? Program execution is not evidence. Convergence domain, root stability,
error, initial-guess dependence, ablations and failure modes must all be shown.

## Gate status

| Gate | Scope | Status |
|---|---|---|
| 0 | Checkout, branch, environment | Complete |
| A | Literature and novelty | Complete, first pass. See `docs/literature/iterative_ci_schmidt_downfolding_review.md` |
| B | Deterministic lowest-three-level root bundle | Complete |
| C | Decouple production path from the exact CI seed | H5 complete; H3 and H6 outstanding |
| D | Ablation and initial-guess sensitivity | Not started |
| E | N2 state-averaged pilot | Blocked on B through D |

## Data-backed conclusions

Every item here was measured locally and the artifact is named.

1. **The legacy classification is unusable.** `INITIAL_SUBSPACE_COVERAGE_CONFIRMED`
   is the classifier's final `else` branch and tests no condition of its own.
   Artifact: `gateB/repro_16990/`. Detail: `docs/development/job16990_local_reproduction.md`.
2. **The default three-root result is environment dependent.** Cluster `[0,1,4]`
   with all roots converged; local `[0,1,3]` with the third root unconverged at
   residual `1.9e-03`. Enlarging the Davidson subspace locally reproduces the
   cluster answer, converged and `+30.014 mH` high.
3. **Symmetry exclusion is real and exactly zero.** The default guess has
   projection of order `1e-21` on the four `E1g` roots in CAS(10e,10o), and
   `6.76e-26` on target root 1 (`B1u`) in CAS(10e,9o). A projection that starts
   at zero stays zero under Krylov expansion.
4. **A solver convergence flag is not evidence.** Every wrong-root solve recorded
   reported `converged = True` for every root.
5. **A residual norm cannot detect wrong-root selection.** At `conv_tol = 1e-12`
   the `nroots = 4` solve, wrong by up to `50.38 mH`, returned residuals at or
   below `5.9e-07`, indistinguishable from the correct `nroots = 6` solve. Wrong
   roots are genuine eigenpairs. The residual is retained only as a
   non-convergence detector at `1e-4`.
6. **The legacy active space was defective.** CAS(10e,10o) contains `E1uy` twice
   and `E1ux` once, cutting a degenerate pair whose members are degenerate at
   `0.87230 Ha`. This split the two `3Pi_g` components by `2.99 mH`, an
   artifact. CAS(10e,9o) restores exact degeneracy and is adopted.
7. **Two non-oracle procedures recover the target exactly.** Per-irrep `D2h`
   solves merged and re-sorted, and overshoot at margin 2. They agree to
   `1.8e-13 Ha`. Spin-resolved solving alone does not: its triplet block misses
   the `B1u` root and returns only one member of the degenerate pair. CIS
   seeding as previously specified does not.
8. **The bundle is reproducible.** Independent rebuilds give an identical
   checksum and energies stable to `1.4e-13 Ha` across three replicas.
9. **Decoupling from the exact CI seed is cheap.** Exact CI enters the
   state-averaged production path at exactly two sites,
   `dm_svd_dci/pipeline_v2.py:64` (dead for this path) and
   `dm_svd_dci/pipeline_state_averaged.py:102`. Everything below
   `solve_state_averaged_schmidt` is already exact-CI-free.

## The locked reference bundle

N2, `N 0 0 0; N 0 0 1.098`, cc-pVDZ, RHF, frozen core 2, CAS(10e,9o),
`(5a,5b)`, `D2h`, `15876` determinants. Target is the three lowest energy
levels, which is four roots.

| Root | Energy / Ha | dE / mH | S^2 | Irrep |
|---|---|---|---|---|
| 0 | `-109.0401239716` | `0.0000` | `0` | `Ag` |
| 1 | `-108.7408683094` | `299.2557` | `2` | `B1u` |
| 2 | `-108.7218810454` | `318.2429` | `2` | `B2g` |
| 3 | `-108.7218810454` | `318.2429` | `2` | `B3g` |

Checksum `ad5b19d6674d54411f704bd3ed4b65aab9741d5373037868fc8fb76e85133d9d`.
Built with PySCF 2.14.0, NumPy 2.5.3, SciPy 1.18.1.
Artifact: `~/work/krylov-dci-run-artifacts/reference/n2_cas9/`.
Committed summary: `results/reference/n2_cas9_reference_bundle.json`.

## Prior art inside this repository

The excited-state energy deviation has been met and solved before in this project.
Four results from `hku_report/` and `reports/` bear directly on Gates C and D.

1. **CIS seeding is the established fix for excited-state P-space quality.** The
   `+636 mH` S1 plateau was caused by HFPT2 scoring assigning zero weight to single
   excitations, since Brillouin makes `<HF|H|singles>` vanish, leaving the P space
   blind to the single-excitation character of triplets. Seeding P with all
   single-excitation determinants plus overlap and `<S^2>` tracking gave, at
   `P = 2000`, `m = 1`: S1 `+636 -> +0.8 mH`, S2 `~+640 -> +0.8 mH`, S3
   `~+50 -> +1.0 mH`. Source: `hku_report/Six_Week_Comprehensive_Report.md`.
   Consequence: the CIS initializer in Gate C is not a guess, it is the documented
   winner, and should be the first non-exact seed tried.

2. **Sharing a resolvent or basis across states is known to break excited states.**
   "The earlier 'excited states degrade with m' was an artifact of shared Krylov
   bases." Per-state `E_0^(k)` Löwdin centering gives chemical accuracy for all
   states. Source: `hku_report/Six_Week_Comprehensive_Report.md` key lesson 3, and
   `hku_report/Phase17-18_Progress_Report.md`: "The Krylov basis MUST be centered at
   the target state's energy."

   **This is in tension with the current design.** `state_averaged_sc_dmsvd.md`
   specifies ONE shared wave operator across all states. The residual dressing does
   keep per-state denominators, `delta q_k = R_Q,k / (E_k - diag(H_QQ))`, but the
   pseudoinverse fit `delta Omega = [delta q_k] pinv([c_k])` then collapses every
   per-state correction into a single operator, which is exactly where the per-state
   information is lost. Per-state Omega is therefore not merely a nice-to-have
   ablation; prior evidence suggests it may be structurally required. It is approved
   for implementation in Gate D and should be treated as a leading hypothesis rather
   than a control.

3. **Never select roots by index; track by overlap and `<S^2>`.** A Phase 18 result
   of `|dE| <= 76 mH` for all roots was later shown to be coincidence: the script
   used `ev[0]`, the lowest eigenvalue, for every root. Proper overlap tracking,
   `m*_k = argmax_m |<c_m^eff | c_k^(P)>|`, revealed true errors above `600 mH`.
   Source: `hku_report/Phase19-20_IterativeP_ExcitedStates.md` section 4.

   The diagnostic offered there is directly reusable: "rerunning with any parameter
   change destroys the good numbers -- a hallmark of coincidence, not convergence."
   Gate D must therefore include a parameter-perturbation robustness test as an
   explicit coincidence detector, not only a seed-perturbation test.

4. **State averaging of the Schmidt basis is established as necessary.** A
   ground-state-only Schmidt basis overestimates every triplet by `+323` to
   `+372 mH` and collapses states 2 to 4 to near-degeneracy, because the GS
   `rho_A^(n)` encodes closed-shell correlation dominated by doubles with
   Brillouin-suppressed singles, while open-shell triplets have a different
   entanglement structure. The state-averaged `rho_A^SA` and `rho_B^SA` keep all
   errors below `1 mH`. Source: `reports/Phase1_DensityMatrix_SVD_Embedding.md`
   section 3.6.1.

**Validated benchmark to reproduce.** N2 CAS(10,10), P blocks `n = [8,9,10]`,
`m = 1`, state-averaged mode, job 15372: S0 `+0.395`, S1 `-0.816`, S2 `-0.476`,
S3 `-0.829`, S4 `-0.224 mH`, all overlaps above `0.997`. Expanding P to
`n = [7,8,9,10]` changed the ground state by `0.004 mH`, so the P-block choice
should not be widened. Note this benchmark predates the active-space correction and
was computed in the defective CAS(10e,10o); the P-block choice must be re-derived
for CAS(10e,9o), where the electron-number blocks differ.

## Open questions

1. **Novelty is weak and the framing must change.** Four independent surveys
   agree the composite claim is likely precedented. State-averaged DMRG owns the
   core loop including the exact averaging formula; TPSCI owns the selected-CI
   version; the Hermitian generalized Ritz form is des Cloizeaux; a shared wave
   operator is the standard Bloch formalism; and Lee and Suzuki 1980 already build
   a shared omega from target states by an inverse over states. The surviving
   candidates are the specific combination, rectangular independent left/right
   ranks per block, and the omega update rule. The last is **not safe to claim
   until Killingbeck and Jolicard, J. Phys. A 2003, 36 (20), has been read** --
   the one source none of the agents could retrieve.
2. **Does the target set survive the active-space change scientifically?**
   CAS(10e,9o) shifts absolute energies from every prior N2 result in the
   repository. Nothing downstream has been re-run against it yet.
3. **Is the state-averaged pilot envelope still 96 GB?** CAS(10e,9o) is four
   times smaller than the space that figure came from, but the target is now
   four states rather than three. Unmeasured.
4. **Does the outer map converge from a non-exact seed at all?** This is the
   actual research question and nothing has been measured yet.
5. **What replaces the broken smoke test?** `tests/integration/smoke_sacis.py`
   is named in `SKILL.md` as the minimum pre-commit check but cannot pass
   anywhere: it targets a driver now under `scripts/archived/` which hard-codes
   its own `PROJECT_ROOT`.

## Speculation, not yet supported by data

- That the self-consistent Schmidt update improves on a frozen basis for a
  reason other than error cancellation. The H2O scan showed stability, not
  benefit.
- That a shared Omega is preferable to a per-state Omega. Prior work in this
  repository points the other way: shared Krylov bases were found to break excited
  states, and per-state centering fixed them. Approved for implementation in Gate D
  and now treated as a leading hypothesis rather than a control.
- That residual dressing is necessary rather than decorative. No bypass exists
  yet, so H4 has never been tested.

## Gate C progress

**H5 is done and gated correctly.** `dm_svd_dci/initializers.py` provides five
non-exact seeds; `setup_system` gained `solve_exact`; exact energies reach the
pipeline only through `evaluate_reference_energies`, which can be switched off.
`tests/regression/test_exact_ci_isolation.py` arms tripwires on the CASCI and FCI
kernels, runs the whole pipeline from a CIS seed, and separately asserts the
tripwire fires on an exact solve so it cannot pass vacuously. Default
`seed='exact'` still reproduces CASCI to `6.7e-13 mH`.

**First substantive feasibility result, and it is a partial falsification of H1.**
At `svd_eps = 1e-10` on H2, where nothing should be truncated and every seed
should give the exact answer, the `cis` and `perturbed` seeds are wrong by
`20.5 mH` while `exact`, `hf`, `trunc` and `selci` are exact to
`6.7e-13 mH`.

A seed with zero weight in an electron-number block makes that block's
state-averaged density vanish, so the block is deleted, and the reconstructed
coefficients inherit the same empty support. The fixed point is self-trapping:
the loop reports convergence at a density change of `1.2e-16` while being wrong.
The CIS seed lacks the `n_A = 0` block because that block is a double excitation.

This matters because the CIS seed is the one the project's own record identifies
as the fix for excited states, so the recommended non-exact seed is the one that
triggers the failure.

**Resolved at iteration zero, by seed completion.** The planned remedy, taking
the block structure from the union of all seed states, turned out not to apply:
the block keys are already complete and identical for every seed, and the failure
is a zero weight rather than a missing key. Instead `build_initial_states` now
adds the lowest-diagonal determinant of any unrepresented block before
diagonalizing. All six seed families then reach the same fixed point to
`6.7e-13 mH`, which is H1 satisfied on H2. The solver was not changed.

**Not resolved in general.** Completion guarantees non-zero support at iteration
zero only. A block whose weight falls below `svd_eps` at a later outer iteration
is still deleted permanently, which at the production threshold of `1e-3` is
realistic. The outer map can lose rank irreversibly and reports convergence when
it does. Detail and three candidate remedies, including DMRG-style density-matrix
perturbation: `docs/development/seed_block_support_finding.md`.

**Downfolding route decided.** Krylov-Galerkin is retained over Neumann
truncation, and the Krylov basis compression SVD is dropped in favour of a plain
QR. The dmSVD on CI coefficients in the A|B bipartition is untouched and remains
the core of the method. Evidence: `docs/theory/neumann_versus_krylov_downfolding.md`.

**Instrumentation complete.** The four quantities the protocol needs are now
recorded per outer iteration: per-root residual norms, the full inner history
including the Omega norm, the root-overlap matrix, and the Schmidt projector
distance, which did not previously exist. Two ablation controls are in place,
`rank_mode='symmetric'` for H6 and `apply_dressing=False` for H4, the latter
verified to reduce the generalized Ritz problem to diagonalizing `H_PP`.

**All five control axes now exist.** Seed family, frozen versus self-consistent,
`apply_dressing`, `omega_mode` and `rank_mode`. The per-state wave operator
solves each state in its own graph subspace, which gives up mutual
orthonormality and index-based root ordering, both documented; root k is
followed by overlap instead. Its dressing collapses to a rank-one minimum-norm
update per state, removing the multi-state pseudoinverse that the literature
identified as the Lee and Suzuki construction. Both modes converge to the exact
eigenvalues of the full P+Q matrix to about `5e-15`.

**A second seeding defect, found and fixed.** The H2 completion did not carry
over: on H2O the CIS seed still left two blocks empty, because adding a
determinant to the subspace is not the same as giving the seed weight in the
block. The lowest-diagonal determinant of a block is often symmetry-decoupled,
becomes its own eigenvector, and the low-lying roots keep zero amplitude on it.
Completion now selects by coupling to the current seed and iterates until the
weight is non-zero. Every seed family is empty-block free on both systems.

**The scan design was nearly invalidated, and the fix is measured.** Before this,
the embedded dimension on H2O was `6` at every threshold from `1e-2` to `1e-4`,
so a threshold scan would have discriminated nothing. With completion repaired it
responds properly for `exact` (`6, 80, 124, 148`), `hf`, `trunc` and `selci`.

**A distinction the project had conflated.** `cis` remains nearly flat
(`5, 7, 8, 8`) even with no empty blocks. Completion made it admissible, not
representative. The record that a CIS seed fixed excited states concerns
selecting a **P space of determinants**; here the seed builds a **Schmidt basis**
from state-averaged densities, which depends on entanglement structure rather
than determinant content. A wavefunction can carry the right determinants and the
wrong entanglement structure. `cis` and `perturbed` will therefore show weak
threshold dependence for a structural reason, which must be reported as a
property of those seeds and not mistaken for insensitivity of the method.

**Two early readings, not yet conclusions.** On H2O at `svd_eps = 1e-3` with the
protocol's partition, `rectangular` and `symmetric` give identical energies, so
the rank asymmetry does not reach the P space at that threshold; and dressing on
versus off differ by only `0.67 mH`. Both are single points from a smoke test,
taken before the scan and before the matched-dimension rule is applied, so
neither is evidence for or against H4 or H6 yet. They do suggest the scan must
cover the threshold range where the N2 measurement showed the asymmetry is
genuine, roughly `3e-2` to `3e-4`.

**Rectangular ranks measured.** The asymmetry is real, caused by state averaging
rather than by unequal block dimensions, and universal within its operating
window. Detail: `docs/development/rectangular_schmidt_rank_measurement.md`.

## Corrections to earlier conclusions

1. **Overshoot alone does not fix a symmetry miss.** Because H is exactly block
   diagonal by irrep and the Davidson preconditioner is diagonal in the same
   labelling, the search space stays inside the guess irreps to machine precision.
   The solver was correctly solving a block-restricted problem. Overshoot worked
   only because requesting more roots made the guess generator reach further down
   the diagonal and happen to pick up the missing irrep. The per-irrep solve is the
   guarantee; overshoot is a reliable fix only for a cluster miss. The bundle
   already uses symmetry as primary and overshoot as cross-check, which is correct,
   but the ordering is now justified rather than incidental.
2. **The shared-omega tension needs a sharper statement.** Prior work in this
   repository found that shared *Krylov bases centered at one energy* break excited
   states. That is a statement about a truncated resolvent basis, not about the
   wave operator formalism: a single shared omega is the standard Bloch construction
   and produces all model-space roots by design. The open question is narrower --
   whether the pseudoinverse fit that collapses per-state corrections into one
   operator loses the per-state resolvent centering that was shown to matter. That
   is what the Gate D per-state variant must test.
3. **Degenerate members must carry equal weight.** The GOK ensemble condition
   requires all members of a degenerate subspace to enter the ensemble with equal
   weight; unequal or partial weighting breaks point-group invariance and produces
   symmetry-broken orbitals and densities. The state weights passed to the
   state-averaged solver must therefore be equal within the degenerate block. This
   is a hard constraint on Gate C and D, not a preference.

## Acceptance tests still to add to the bundle

- **Gap gate**: require a clear separation between root n and root n+1.
- **Symmetry census**: irrep weight per accepted root; an irrep with zero weight
  across all returned roots means the guess never spanned it.
- **Randomized-restart reproducibility** from a guess dense in every irrep.
- **Sylvester inertia count**, an LDL^T factorization of H - sigma I counting
  exactly how many eigenvalues lie below sigma. This is the only deterministic
  guarantee. Per-irrep it is cheap, since each block is far smaller than the full
  space, and it should be run once to certify the pipeline.

## New open question from Gate C

How should irreversible rank loss be handled at a finite threshold? Seed
completion fixes iteration zero but not later iterations. The candidates are a
rank floor, DMRG-style density-matrix perturbation, and residual-informed block
re-entry, the last being the closest fit since the Q-space residual already says
where the wavefunction wants weight. This is a solver change and needs the
threshold scan behind it.

## Next single priority

Gate C step 3: build the scan driver that runs the pre-registered cross of seed
family, Schmidt basis treatment, dressing, wave operator and rank allocation,
scoring each cell against the exact resolvent at the same `E_0` rather than
against CASCI alone, and applying the matched-dimension rule for H6 by adjusting
the symmetric run's threshold until its embedded dimension is within 2 percent
of the rectangular one. H5 is already gated, so results become admissible.

Two obligations from Gate A now belong in that protocol as primary hypotheses
rather than ancillary controls, because the novelty case depends on them:

- **H3 is now load bearing.** Freezing the Schmidt basis after the first iteration
  and re-running is the test of whether the self-consistent map is decoration. If
  the converged answer sits inside the error bar of the one-shot answer, there is
  no method. This was independently identified by the novelty audit as the single
  experiment that decides the paper.
- **Competitors, not oracles.** Accuracy must eventually be reported against
  state-averaged DMRG at matched retained dimension and against SA-TPSCI at matched
  per-block basis size. FCI is the oracle; those are the competitors.
