# Where the method stands, and what publication still needs

Assessment at commit `5c3fe82` on `research/iterative-ci-feasibility`. Every
number quoted here is measured and its artifact is named in
`docs/development/iterative_ci_research_state.md`,
`gate_c_feasibility_results.md` or `gate_c_n2_results.md`. Nothing is projected.

The method under assessment is the live state-averaged path only:

```text
C^(k) -> state-averaged densities -> blockwise dmSVD over the A|B orbital
      bipartition -> P/Q partition of the Schmidt product basis -> H_emb
      -> shared residual-dressed wave operator Omega -> generalized Ritz
      -> C^(k+1) -> root matching, mixing, convergence
```

The legacy Krylov/Scheme B line is deprecated and is excluded.

## One-paragraph summary

The method is internally sound and unusually well characterized: it runs from
seeds that read no exact CI, it reaches the exact-seeded fixed point, its error
budget is cleanly separated, and the one structural defect that made it provably
unable to work has been found, proved, and repaired. **It reaches chemical
accuracy on a single state**, measured at `1.209 mH` for `D = 810` and
`0.283 mH` for `D = 2466`, self-consistently and from a non-exact seed. What it
does not yet have is that accuracy for several states sharing one basis, where
the error at comparable dimension is about twelve times larger, nor any
comparison against a competitor. The multi-state gap, not accuracy as such, is
what stands between the current state and a publishable performance claim, and
the record already contains a candidate fix for it.

## The assets, stated precisely

These are worth listing because they are what any paper would be built on.

1. **Exact-CI independence is real and gated.** Five non-exact seed families
   exist; the Lanczos seed reaches the exact-seeded fixed point at the same
   embedded dimension (H2O, `svd_eps=1e-4`: `0.0012 mH` against `exact`'s
   `0.0000`), converging monotonically in one parameter
   (`0.616, 0.024, 0.0012, 0.0004, 0.0000 mH` at 1, 2, 3, 4, 6 steps). A
   tripwire test fails if the production path touches a CASCI or FCI kernel, and
   separately asserts the tripwire fires on an exact solve so it cannot pass
   vacuously.

2. **The error budget is separable, and one half is exactly zero.** Splitting
   total error into a Schmidt part (`E_embedded - E_reference`) and a
   wave-operator part (`E_downfolded - E_embedded`), the second is `0.0000 mH`
   in every cell of the H2O scan. At convergence the residual-dressed Omega
   reproduces the exact diagonalization of `H_emb`. **Omega is a solver, not an
   approximation.** Any claim for it must therefore be a cost claim, and all of
   the method's accuracy is a statement about dmSVD truncation.

3. **A structural theorem, and its repair.** `outer_map_rank_contraction.md`
   proves `C_new(n) = U(n) T(n) V(n)^dag` lies in the span of the basis that
   produced it, so retained rank is monotonically non-increasing block by block
   by construction. Measured: no rank increase anywhere on either system; on N2
   the basis collapsed `720 -> 216` while degrading the energy by `2.44 mH`, and
   the loop reported convergence while doing it. Residual-driven enrichment in
   the determinant basis breaks the contraction from `lambda = 0.2` upward.

4. **The enrichment had to be annealed, for a reason that is a result in its own
   right.** Constant strength never converged: the projector distance stayed
   between `5.7` and `6.2` at every iteration. `||R_k||` does **not** vanish,
   because a state in a truncated space can never be an exact eigenvector of
   the full Hamiltonian, so it plateaued near `0.13` and acted as a permanent
   rotating perturbation. The map was in a limit cycle. With
   `lambda_t = lambda_0 * decay^t` the loop converges in 13 outer iterations and
   the projector distance falls to `1.1e-11` at `decay = 0.3`.

5. **A deterministic, checksummed multi-state reference.** N2 CAS(10e,9o), four
   roots covering the three lowest levels, reproducible to `1.4e-13 Ha` across
   three replicas, with the active-space defect that had split the `3Pi_g`
   components by `2.99 mH` corrected. Two independent non-oracle procedures
   recover it and agree to `1.8e-13 Ha`.

6. **Residual dressing is necessary, not decorative.** Undressed is 9.8 to 9700
   times worse on H2O.

## The shortcomings, ranked by how much they threaten publication

### S1. Accuracy: one state is there, and the residue is the degenerate pair

N2 CAS(10e,9o), Lanczos seed, self-consistent, `enrichment_strength = 0`, scored
against the committed bundle. Full curves and caveats:
`docs/development/n2_threshold_curves.md`.

Ground state alone, converged in every row:

| `svd_eps` | `D` | error |
|---|---|---|
| 1e-2 | 342 | `3.009 mH` |
| 5e-3 | 810 | **`1.209 mH`** |
| 3e-3 | 942 | **`0.771 mH`** |
| 1e-3 | 2466 | **`0.283 mH`** |

Four states covering the three lowest levels, sharing one basis:

| `svd_eps` | `D` | weighted | S0 `Ag` | S1 `B1u` | S2 `B2g` | S3 `B3g` | conv |
|---|---|---|---|---|---|---|---|
| 1e-2 | 216 | `25.022` | `7.119` | `15.385` | `38.591` | `38.991` | no |
| 5e-3 | 978 | `19.529` | `2.639` | `9.720` | `32.234` | `33.522` | no |
| 3e-3 | 2517 | `16.292` | **`1.321`** | `4.454` | `29.321` | `30.070` | yes |
| 2e-3 | 2841 | `12.996` | **`0.850`** | `4.276` | `22.533` | `24.323` | yes |
| 1e-3 | 4987 | `7.321` | **`0.342`** | **`1.249`** | `13.411` | `14.280` | yes |

Four statements, all measured:

1. **Chemical accuracy is reached**, for the ground state from `eps = 5e-3` alone
   and from `3e-3` inside the four-state calculation, and for the `B1u` triplet at
   `1e-3`. From a seed that reads no exact CI, self-consistently, in the corrected
   active space.
2. **The residual error is concentrated in the exactly degenerate `3Pi_g` pair**,
   `13.4` and `14.3 mH` at `eps = 1e-3` against `0.34` and `1.25` for the other
   two. S2 and S3 track each other to within `0.4` to `1.8 mH` and converge about
   three times more slowly in `eps`, so they behave as one object that the shared
   basis represents badly.
3. **The curve has not plateaued.** `16.292 -> 12.996 -> 7.321 mH` from `3e-3` to
   `1e-3`, the last step nearly halving. The measurement stopped at this
   machine's memory wall, `3920 MiB` at `D = 4987`, not at a scientific limit.
4. **Sharing the basis costs the ground state almost nothing**: `0.283 mH` alone
   at `D = 2466` against `0.342 mH` inside the four-state run. The cost falls
   entirely on the states the shared basis represents worst.

**H7 does not fix it, which was the prediction and it was wrong.**
`omega_mode='per_state'` at `eps = 3e-3` gives `9.405` against the shared
`9.473 mH` on the old scoring, an advantage of `0.068 mH` barely above `tol_E`,
it did not converge, it took 4762 s against 534 s, and at `2e-3` and `1e-3` it
raises `LinAlgError: state set is linearly dependent` with a per-state graph
metric whose minimum eigenvalue is `-5.6e-17`. So it is also numerically fragile
and is not the equivalent of job 15372's per-state Löwdin centering.

**The degeneracy splitting is in the Schmidt basis, not in root matching.** At
`eps = 3e-3` and `1e-3` the method's own pair splitting and the splitting of the
exact diagonalization of the same `H_emb` agree to every printed digit, `0.7497`
and `0.8685 mH`. The wave-operator error stays at `1e-10 mH` throughout, so the
downfolding remains exact and the whole error is dmSVD truncation.

**Consequence, and it merges two lines of work.** Chemical accuracy on the
`3Pi_g` pair needs a larger `D` than this machine reaches, and what blocks a
larger `D` is the `4.2 x M x D` determinant-space expansion in the
embedded-Hamiltonian build. So S9, the scalability item, is not a separate
reach-extension exercise: it is the route to the remaining accuracy. The rank-`r`
Q-space compression analysed in
`docs/theory/hpq_svd_qspace_compression_analysis.md` and the streamed build are
therefore on the critical path for the accuracy claim, not after it.

### S1b. Why the earlier assessment got this backwards

Recorded because the failure modes are reusable, and there were three.

**The threshold axis had only been sampled at its loose end.** Every N2 number
then available came from the H3 enrichment scan and the H6 rank comparison, whose
artifacts give `eps` of `0.02`, `0.011`, `0.01`, `0.005` and `0.0147`. Those experiments
needed a small `D` to be affordable, so they sampled only the loose end of the
threshold axis, and the production threshold of `1e-3` had never been run on N2
at all. Reading a shallow segment at the loose end as the method's accuracy
ceiling was an extrapolation from data that did not support it. The high-accuracy
numbers that were available, `0.0034` and `0.0012 mH`, were H2O at `1e-3` and
`1e-4`, so system and threshold had both changed at once and neither was held
fixed. The recorded `±1 mH` five-state result was also simply overlooked.

**The reference itself was wrong, which produced a false plateau.**
`evaluate_reference_energies` took an unvalidated `nroots = n_states` solve,
which on this system returns the fourth reference `27.2736 mH` too high. Scored
against it the four-state progression read `9.473 -> 7.653 -> 6.999 mH` and
looked like diminishing returns; scored correctly it is
`16.292 -> 12.996 -> 7.321` and is still falling steeply. A wrong reference is
indistinguishable from a wrong method, which is why the fix in `7b6c72d` raises
rather than returning an unvalidated answer.

**Per-state numbers from an unconverged run were read as a diagnosis.** The
`per_state` run at `eps = 3e-3` reported `S2 = 29.479` and `S3 = 2.212 mH`,
which was read as the degenerate pair being split by 27 mH. That run did not
converge and was scored against the defective reference; the converged shared run
gives `13.411` and `14.280`, a splitting of `0.87 mH`. Per-state detail from a
run whose `converged` flag is false is not evidence, which is the same lesson
Gate B recorded about convergence flags in the opposite direction.

### S2. There is no comparison against any competitor, only against the oracle

Everything is measured against exact CASCI. Nothing has been measured against
state-averaged DMRG at matched retained dimension, SA-TPSCI at matched per-block
basis size, SHCI or CIPSI at matched parameter count, or dressed selected CI on
the same P space. The last of these is the sharpest: if the residual-dressed
generalized Ritz reproduces dressed-SCI energies to within noise, there is no
method, and that experiment has never been run.

Given S1, a cost argument cannot be substituted either, because none exists (S9).
The method currently claims neither better accuracy nor lower cost than anything.

### S3. Novelty is thin, and the nearest precedent was not considered until Gate A

Four independent literature surveys agreed: **composite claim likely
precedented**. The near-identical prior art is tensor product selected CI, which
the repository had no record of considering:

- Abraham & Mayhall, *J. Chem. Theory Comput.* **2020**, 16, 6098, DOI
  `10.1021/acs.jctc.0c00141`;
- Braunscheidel, Abraham & Mayhall, *J. Phys. Chem. A* **2023**, 127, 8179, DOI
  `10.1021/acs.jpca.3c03161` -- "we compute a single global basis in a
  state-averaged way ... average the cluster-RDMs from each TPSCI eigenvector",
  with the loop written as update the basis by sparse SVD, test convergence,
  return;
- Braunscheidel et al., *Faraday Discuss.* **2024**, DOI `10.1039/D4FD00049H`,
  which builds Bloch effective Hamiltonians on that basis.

That is the claimed self-consistent map minus only the dressed Omega. The
remaining differences are real -- a global bipartition with blockwise Schmidt
bases rather than localized clusters in a tensor-product basis -- but they have
to be argued with numbers.

Of the three ingredients no survey located:

- the specific combination of a self-rebuilt Schmidt basis with P/Q
  wave-operator downfolding and a non-orthogonal generalized Ritz in one loop;
- **rectangular independent left and right ranks per electron-number block**,
  which is currently supported at **one of two** tested dimensions
  (`+0.2217 mH` at `D ~ 1539` against `+0.0377 mH` at `D ~ 720`, the latter
  below tolerance). One win and one tie is not a claim;
- the Omega update rule, **which is not safe to claim at all** until
  Killingbeck & Jolicard, *J. Phys. A* **2003**, 36 (20) has been read. It is
  the one source none of the four surveys could retrieve.

The Hermitian graph Ritz `X^dag H X c = E X^dag X c` is des Cloizeaux
(Okamoto, Fujii & Suzuki 2005 write the family literally); a single shared Omega
is the Bloch definition and the Jeziorski-Monkhorst state-universal ansatz;
`Omega` built from target states by an inverse over states is Lee & Suzuki 1980.
These must be cited, not claimed.

### S4. Every load-bearing result rests on one molecule

H2O cannot test H6 (its blocks carry no rank asymmetry) and cannot test H3 (its
basis was not collapsing). N2 is the only system that expresses either. So the
H3 reversal, the enrichment calibration and the entire H6 case each rest on a
single molecule at a single geometry in a single basis. There is no
bond-dissociation curve, no basis-set variation, no second system at N2's scale,
and no system outside the two.

### S5. The repair introduced two uncalibrated parameters that move both accuracy and cost

`lambda_0` and `decay` were chosen from small scans on one system at one
threshold. They change the answer and the dimension together:
`lambda = 0.5` gives `15.132 mH` at `D = 396`, `lambda = 1.0` gives
`14.125 mH` at `D = 490` with a strongly oscillating trajectory. Whether the
constant-strength regime has a fixed point at all, or only a limit cycle, is
unknown. Until that is settled a referee can ask whether the converged answer is
a property of the method or of the annealing schedule, and the honest answer
today is that it is not known.

### S6. Irreversible rank loss is patched only at iteration zero

Seed completion guarantees non-zero block support at iteration zero. A block
whose weight falls below `svd_eps` at a later outer iteration is still deleted
permanently, and the loop reports convergence when it does. At the production
threshold of `1e-3` that is realistic, not hypothetical. Three candidate
remedies are recorded -- a rank floor, DMRG-style density-matrix perturbation,
and residual-informed block re-entry -- and none is implemented.

### S7. The bipartition is the unexamined choice, and it is what controls S1

`A` is five occupied orbitals and `B` is the rest, fixed throughout. There is no
study of the A|B size, of orbital localization, of natural versus canonical
orbitals, or of optimizing the cut. Since asset 2 makes all error Schmidt
truncation error, and Schmidt spectra depend strongly on where the cut falls and
on how the orbitals are chosen, **this is simultaneously the largest untested
design assumption and the most promising route to fixing S1.** DMRG's
effectiveness comes largely from optimizing exactly this. It is the one
shortcoming on this list that might be an opportunity.

### S8. Gate D never ran

The pre-registered ablation cross -- seed family x frozen/self-consistent x
dressing on/off x shared/per-state Omega x thresholds -- is not started. H4 is
confirmed on H2O only; H7 (shared versus per-state Omega) is indistinguishable
on H2O only, which is the system that discriminates least. The project's own
record recommends a parameter-perturbation robustness test as an explicit
coincidence detector, after a Phase 18 result of `|dE| <= 76 mH` turned out to
be an artifact of using `ev[0]` for every root; that detector is not
implemented.

### S9. No cost argument exists, and the known bottleneck is unfixed

Peak memory on N2 is a constant `4.2 x M x D` across three decades of `D`; the
`q x q` Hamiltonian is `0.1` to `2.3 percent` of peak. The bottleneck is the
determinant-space expansion in the embedded-Hamiltonian build. The matrix-free
low-rank Omega solver, though correct and verified to `7e-12 mH` end to end with
identical iteration counts, does not touch it -- an earlier claim that it did
was wrong and has been withdrawn. CAS(14,10) needs that build streamed.

### S10. Physical and formal checks that have never been run

- **Size consistency is not tested, mentioned or documented anywhere in the
  repository.** For a truncated CI-based method this is a standard referee
  question, and the Schmidt truncation is on a bipartition, so the answer is not
  obvious. Two non-interacting fragments at large separation is the test.
- **Spin purity of the reconstructed states is never evaluated on the live
  path.** The Schmidt blocks are indexed by electron number, not by spin, and
  no `<S^2>` is computed for the generalized Ritz states. For a multi-state
  method targeting triplets this needs to be shown, not assumed.
- **The discarded weight exists but has never been calibrated against the energy
  error.** An earlier version of this document said there was no analogue of
  DMRG's discarded weight. That was wrong: `density_matrix.py:288` computes
  `discarded_weight = sum of sigma^2 below the threshold` and both pipelines
  record it. What is missing is any measured relation between it and the energy
  error, which is what would turn it into a usable a posteriori estimator. The
  curves in S1 now provide the data to fit one: for the ground state the recorded
  pairs run from `1.2e-4` at `3.009 mH` to the `1e-3` threshold's value.
- **`D` is a product-grid count and can exceed the CI dimension, which matters
  because every matched-cost comparison uses `D` as its axis.** Measured on H2O:
  the CAS has 100 determinants, `sum_n dim_A(n) dim_B(n) = 196`, and the reported
  `D_total` at `eps = 1e-4` is 148. `C^(n)` is structurally sparse in `Ms`, only
  100 of the 196 grid positions correspond to real `Ms = 0` determinants, and the
  Schmidt product basis is formed from all `r_A x r_B` combinations with no
  compatibility filter (`schmidt_partition.py:93-95`). For N2 CAS(10e,9o) the
  ceiling is 43,186 against 15,876 determinants. Whether the surplus directions
  are null, redundant or physically meaningful has not been established, and
  neither has the effect on the H3 and H6 matched-dimension comparisons, which
  all match on `D`.
- **The GOK equal-weight condition is a default, not a constraint.** Equal
  weights are the default in `density_matrix.py`, which satisfies it, but
  nothing rejects a user-supplied weight vector that splits a degenerate block,
  and the N2 target contains an exactly two-fold degenerate level.
- **Root matching for the degenerate block.** Individual eigenvectors of an
  exactly degenerate pair are an arbitrary rotation, so matching must be by
  subspace principal angles rather than per-vector overlap. This is documented
  as required; that it is what the code does has not been asserted in a test.

### S11. Infrastructure

`tests/integration/smoke_sacis.py`, named in `SKILL.md` as the minimum
pre-commit check, cannot pass anywhere: it fails with
`PermissionError: /data`, a hard-coded cluster path. Root-selection outcomes were
shown to be PySCF-version dependent, so versions are a scientific control and
not a detail. The deprecated legacy path was found in `5c3fe82` to have nine
stale imports and an unconditional crash, which is evidence that anything
outside the live path should be assumed rotten until run.

## Three framings, and a recommendation

**Framing A, a performance paper**: "a new self-consistent multi-state
downfolding method". This is now the more likely option, and an earlier version
of this document wrongly rated it as barely reachable. The ground state already
reaches `0.283 mH` at `D = 2466` self-consistently from a non-exact seed, so the
machinery delivers chemical accuracy; what is missing is the same for several
states at once, where a known candidate fix exists and has never been tested on a
system that can express it. The remaining requirement is then the S2 comparisons,
which must at least be drawn.

**Framing B, an analysis paper** -- *recommended*: "self-consistent basis
rebuilding in truncated CI: why the naive outer map cannot work, and what
repairs it". The load-bearing content already exists and is measured:

- the rank contraction theorem, which is a **general** negative result about any
  method that rebuilds its basis from its own truncated wavefunction, and
  therefore speaks to a family that includes the TPSCI self-consistency loop and
  DMRG without a noise term;
- the demonstration that the naive map provably cannot increase rank, measured
  collapsing `720 -> 216` while reporting convergence;
- the proof that enrichment must be annealed, with the reason -- `||R_k||` cannot
  vanish in a truncated space, so constant strength is a permanent rotating
  perturbation and the map enters a limit cycle. This reframes DMRG's noise term
  as structurally necessary rather than a numerical convenience, which is a
  transferable and non-obvious statement;
- the clean error-budget separation showing the dressed wave operator is a
  solver contributing exactly zero error, which tells any such method where its
  accuracy actually comes from.

This framing uses what is already established, needs far less new computation,
positions the work as complementary to TPSCI rather than competing with it, and
does not require the novelty claims that Gate A found unsafe. Its risk is that
it is a methods-analysis contribution rather than a new-method contribution, so
journal fit matters.

**Framing C, the rectangular-rank paper**: too thin. One win, one tie, one
system.

**Recommendation: run H7 on N2 at `eps = 1e-3` first, then decide.** It is one
cheap experiment, it is the difference between the two framings, and the record
predicts it is worth roughly an order of magnitude on the excited states. If the
four-state error drops toward the `±1 mH` the project has already recorded with
per-state centering, Framing A is the paper and the analysis content becomes a
strong section inside it. If it does not, Framing B is the honest paper and the
multi-state cost becomes the recorded negative result. Either way the Framing B
content is already measured and is not lost, so the decision costs nothing to
defer by one experiment.

Note that S7, the bipartition study, was previously named as the deciding
experiment. It is not: with one state already at `0.283 mH`, the bipartition is
evidently good enough, and the open question moved to the multi-state basis.

## What to add before submission

Ordered. Item 0 decides the framing; items 1-6 are required under either.

0. **H7 on N2 at `eps = 1e-3`, per-state against shared Omega, four states.**
   The highest-value experiment available and cheap: the single-state run at that
   threshold took 309 s and 1.7 GiB. It decides which paper this is.
1. **Read Killingbeck & Jolicard 2003.** Cheap, blocking, and it gates a claim.
2. **Size consistency.** Two non-interacting fragments at large separation, error
   against the sum of the monomers, at several `svd_eps`. Cheap and currently a
   hole a referee will find immediately.
3. **Spin purity.** Evaluate `<S^2>` for every generalized Ritz state on the live
   path and report it with the energies. Also assert principal-angle matching for
   the degenerate block, and reject weight vectors that split one.
4. **Fix the named smoke test**, and put the live path under a check that runs.
5. **Settle S5.** Calibrate `lambda_0` and `decay` on more than one system and
   threshold, and determine whether constant strength has a fixed point or only a
   limit cycle. Report the sensitivity of the converged answer to both.
6. **The coincidence detector.** Parameter-perturbation robustness, per the
   project's own Phase 18 lesson.
7. **S7, the bipartition study.** A|B size, canonical versus natural versus
   localized orbitals, several cuts, on N2 and one more system. Demoted from
   deciding experiment to ordinary sensitivity study now that one state reaches
   `0.283 mH`, but still needed, because nothing establishes that the present cut
   is a good choice rather than merely an adequate one, and because the
   multi-state basis cost may depend on it.
8. **Competitor comparisons (S2).** Dressed selected CI on the same P space
   first, because it is the one that can falsify the method outright. Then
   SA-DMRG at matched retained dimension. C2 is the natural shared benchmark and
   is Li & Yang's system.
9. **A second and third system, and a dissociation curve (S4).** Required before
   anything is reported as general.
10. **Gate D in full (S8).**
11. **Stream the embedded-Hamiltonian build (S9)**, which is what CAS(14,10)
    needs and the only route to a cost claim.
12. **Rank-loss remedy beyond iteration zero (S6).** Residual-informed block
    re-entry is the closest fit, since the Q-space residual already indicates
    where the wavefunction wants weight.
13. **Re-verify every `[unverified]` reference** in the Gate A review against a
    publisher record.

## What would falsify the method

Recording these in advance, so that a negative result is a finding rather than a
disappointment.

- Dressed selected CI on the same P space reproduces the energies to within
  noise. Then the dressed Omega is a re-parameterization and there is no method.
- Per-state Omega does not materially close the four-state gap, and no cut from
  the bipartition study improves the multi-state error-versus-`D` curve either.
  Then the multi-state cost is structural for this construction, and no
  multi-state performance claim is available at any dimension it can reach.
- The converged answer depends materially on `lambda_0` or `decay`. Then the
  fixed point is a property of the schedule, not of the method.
- SA-DMRG at matched retained dimension is better on every system tested. Then
  the honest contribution is the analysis, not the method.
