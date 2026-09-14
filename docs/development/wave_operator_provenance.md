# Provenance of the residual-dressed wave operator

This note answers a specific question: where did the shared residual-dressed wave
operator `Omega` come from, and on what evidence was it adopted? It was raised
because `Omega` is not part of the original project design. The answer is
established from the repository history, not from recollection.

## Summary

`Omega` entered the project on **2026-09-02** in two commits made one minute
apart, documentation first:

| Commit | Date | Subject |
|---|---|---|
| `d412b0b` | 2026-09-02 22:46:21 | `docs(method): define residual-dressed wave-operator iteration` |
| `d322381` | 2026-09-02 22:46:39 | `feat(method): implement state-averaged self-consistent dmSVD` |

`d322381` added `dm_svd_dci/wave_operator.py` (392 lines),
`dm_svd_dci/state_averaged_solver.py` (406 lines) and
`dm_svd_dci/pipeline_state_averaged.py` (290 lines) in one step. Neither commit
message records a motivation, a comparison against the existing downfolding
routes, or an experiment that prompted the change. **No design rationale is
recorded anywhere in the repository.**

## What the project already had

Before `Omega`, the authoritative formulation in `docs/theory/formalisms.md`
specified a Krylov-compressed resolvent:

```text
H^eff = H_PP + H_PK (E_0 I - H_KK)^-1 H_KP
```

with the resolvent centred on a **per-state** reference energy `E_0^(k)` taken
from the k-th eigenvalue of `H_PP`. Two implementations of that family exist in
the tree, `dm_svd_dci/_legacy_effective_ham.py` and
`dm_svd_dci/neumann_effective_ham.py`. `Omega` is a **third**, parallel
downfolding route, not a replacement of either: both older modules remain.

## The validated result predates Omega and did not use it

The best multi-state result the project has recorded is job `15372`, reported in
`hku_report/Six_Week_Comprehensive_Report.md`, dated **2026-08-05**: N2
CAS(10,10), P blocks `n = [8,9,10]`, `m = 1`, state-averaged mode, with all five
states within `±1 mH` of FCI and overlaps above `0.997`. Its own cost table names
the downfolding step "Krylov-dCI", at 632 s of a 5061 s run.

That is the Krylov/Löwdin route with per-state `E_0^(k)` centring, a month before
`Omega` existed. The corresponding key lesson in the same report reads: "Per-state
E0 Löwdin works for excited states. Using each state's own H_PP eigenvalue as the
Bloch resolvent center gives chemical accuracy for all states."

**`results/` contains no output from the state-averaged `Omega` pipeline.** Every
committed result directory comes from the earlier routes. The `Omega` formulation
has therefore never reproduced the project's best validated number, and has no
committed numerical result of its own.

## How Omega is documented

`d412b0b` added 133 lines to `docs/theory/state_averaged_sc_dmsvd.md`, which
specifies the construction fully and correctly. But the only change it made to the
authoritative `docs/theory/formalisms.md` was a seven-line cross-reference placed
**inside section 10, "References"**:

> The state-averaged residual-dressed extension is specified in
> `docs/theory/state_averaged_sc_dmsvd.md`. Its key distinction from the
> state-specific formulation above is that all selected roots share both the
> weighted Schmidt basis and a single graph wave operator. State-averaged left and
> right density ranks are retained independently, so each number block has
> dimension `r_A(n) * r_B(n)` rather than an assumed `r(n)^2`.

So the file designated as "the single source of truth for the mathematical
formulation" still describes the Krylov resolvent as the method, and mentions the
wave operator only in its bibliography section. A reader following the stated
authority order would not learn that the production state-averaged path uses a
different downfolding operator.

Note also that **rectangular independent left and right ranks entered in the same
commit as `Omega`**. The two are historically entangled but mathematically
separable: rectangular ranks belong to the Schmidt/basis construction, `Omega` to
the solver.

## Reconstructed rationale

No rationale is recorded, so the following is inferred from the mathematics and is
labelled as inference, not history.

The Krylov route has three properties that are awkward for a multi-state method:

1. It is **energy dependent**. The resolvent `(E_0 I - H_KK)^-1` must be centred
   somewhere, and the project established empirically that a single centre shared
   across states breaks excited states, forcing a separate basis per state.
2. It is **not variational**. Nothing guarantees the downfolded energies bound the
   true ones from above.
3. Per-state bases mean per-state Hamiltonian builds, and the H_emb build is 87 to
   93 percent of the run time.

The graph construction addresses all three at once. With `X = [I_P ; Omega]`, the
problem `X^dag H X c = E X^dag X c` is **energy independent** (no reference energy
appears), **Hermitian**, and **variational** by Hylleraas-Undheim-MacDonald, since
it is Rayleigh-Ritz in the non-orthogonal basis `{|i> + Omega|i>}`. One `Omega`
serves every root, so one Hamiltonian build serves every root. The per-state
information that the Krylov route carried in `E_0^(k)` is retained only in the
dressing denominators, `delta q_k = R_Q,k / (E_k - diag(H_QQ))`.

That is a coherent design motive. It is also, as the literature survey found,
exactly the classical construction: `X^dag H X c = E X^dag X c` is the des
Cloizeaux Hermitian effective Hamiltonian in non-orthogonal form, and Lee and
Suzuki build the shared `omega` from target states by an inverse over states.

## Consequences

1. **The adoption of `Omega` was never tested against the route it parallels.**
   The comparison that should have been made, `Omega` versus per-state Krylov on
   N2 CAS(10,10) against the job 15372 numbers, does not exist.
2. **`formalisms.md` is out of date** with respect to the production path and
   should be corrected, since project rules designate it authoritative.
3. **The per-state information is now concentrated in one place**: the dressing
   denominators. The pseudoinverse fit that follows,
   `delta Omega = [delta q_k] pinv([c_k])`, collapses the per-state corrections
   into a single operator. Whether that step discards the per-state resolvent
   centring the project proved necessary is an open, testable question, and is the
   reason a per-state `Omega` variant is being built as a controlled comparison.

## Recommended actions

- Reproduce job 15372 with the `Omega` route and compare against the recorded
  per-state Krylov numbers, on the same reference bundle. Until that exists,
  `Omega` is an unvalidated reformulation.
- Correct `docs/theory/formalisms.md` so the authoritative formulation describes
  the downfolding operator the production path actually uses.
- Keep the rectangular-rank question separate from the `Omega` question in all
  experiments, since they entered together but are independent.
