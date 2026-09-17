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

The method is now internally sound and unusually well characterized: it runs
from seeds that read no exact CI, it reaches the exact-seeded fixed point, its
error budget is cleanly separated, and the one structural defect that made it
provably unable to work has been found, proved, and repaired. What it does not
have is any demonstrated advantage over any competitor, and its accuracy at
every dimension tested is roughly an order of magnitude short of chemical
accuracy. That gap, not any missing feature, is what stands between the current
state and a publishable performance claim.

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

### S1. Accuracy is the binding constraint, and the curve is shallow

All recorded four-state weighted absolute errors on N2 CAS(10e,9o), against the
exact bundle, with their provenance:

| `D` | error / mH | configuration |
|---|---|---|
| 178 | `17.5816` | annealed `decay=0.5`, converged |
| 177 | `20.3947` | frozen, matched to the above |
| 203 | `17.2539` | annealed `decay=0.3`, converged |
| 396 | `15.1315` | constant `lambda=0.5` |
| 398 | `17.8437` | frozen, matched to the above |
| 715 | `15.4898` | rectangular ranks |
| 720 | `15.5275` | symmetric ranks |
| 1559 | `12.1716` | rectangular ranks |
| 1539 | `12.3933` | symmetric ranks |

These are separate runs at different thresholds and settings, not one curve, and
they must not be read as one. The observation that survives that caveat is
blunt: **across `D` from about 180 to 1560, that is from roughly 1 to 10 percent
of the 15876-determinant space, the error moves only from about 17.6 to 12.2 mH,
and no recorded configuration comes within 7 times of chemical accuracy
(1.6 mH).**

Because of asset 2, every one of those millihartrees is dmSVD truncation error.
So the method's accuracy is entirely a statement about how fast the Schmidt
spectrum of this A|B bipartition decays, and on N2 with a five-orbital A space
it decays slowly.

The self-consistency advantage is real and well outside tolerance
(`+2.71 mH` at matched `D ~ 397`, 54 times `tol_E`; `+2.53` to `+2.81 mH` for
the annealed runs) but it is small next to the 12-18 mH absolute error. A
referee will read that as a correction to a poor starting approximation rather
than as evidence of a competitive method.

**This is the single fact most likely to stop a performance paper.**

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
- **No error estimator links `svd_eps` to the energy error.** DMRG has discarded
  weight; there is no analogue here, a priori or a posteriori.
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
downfolding method". Requires closing S1 by a large factor and winning or at
least drawing the S2 comparisons. This is the highest-value and by far the
highest-risk option, and on current evidence the accuracy gap is too large to
close by tuning; it would need S7 to pay off.

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

**Recommendation: build toward B now, and treat S7 as the experiment that
decides whether A is also available.** They share most of the required work, so
committing to B costs little optionality.

## What to add before submission

Ordered. Items 1-6 are required under either framing.

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
   localized orbitals, several cuts, on N2 and one more system. This is the
   experiment that decides whether Framing A is reachable: if a better cut moves
   the 12 mH at `D ~ 1560` down by a large factor, the performance paper is live;
   if it does not, S1 is structural and Framing B is the honest paper.
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
- The bipartition study finds no cut that materially improves the error-versus-`D`
  curve. Then S1 is structural for this class of bipartition, and no performance
  claim is available at any dimension reachable by this construction.
- The converged answer depends materially on `lambda_0` or `decay`. Then the
  fixed point is a property of the schedule, not of the method.
- SA-DMRG at matched retained dimension is better on every system tested. Then
  the honest contribution is the analysis, not the method.
