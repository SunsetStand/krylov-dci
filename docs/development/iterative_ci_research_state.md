# Iterative-CI feasibility: research state

Updated at commit `6393f4c` on branch `research/iterative-ci-feasibility`,
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
| A | Literature and novelty | Not started; agents blocked by a session rate limit |
| B | Deterministic lowest-three-level root bundle | Complete |
| C | Decouple production path from the exact CI seed | Not started |
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

## Open questions

1. **Novelty is unassessed.** Gate A produced no report. One partial lead needs
   chasing urgently: an existing method appears to be named "downfolded CI
   (dCI)", which is a naming and possibly a priority collision.
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
- That a shared Omega is preferable to a per-state Omega. Approved for
  implementation as a controlled ablation in Gate D; currently untested.
- That residual dressing is necessary rather than decorative. No bypass exists
  yet, so H4 has never been tested.

## Next single priority

Gate C step 1: write `docs/theory/iterative_ci_feasibility_protocol.md`,
pre-registering H1 through H5, the initialization family, the control groups,
the metric set and the pass, fail and inconclusive thresholds, before any solver
code is touched. Thresholds must be fixed in writing before results are seen.
