# N2 error versus threshold: the measurement that had never been made

Every N2 accuracy number in this repository before 2026-09-17 came from the H3
enrichment scan or the H6 rank comparison, whose artifacts give `svd_eps` of
`0.02`, `0.0147`, `0.011`, `0.01` and `0.005`. Those experiments needed a small
`D` to be affordable, so the threshold axis had only been sampled at its loose
end, and the protocol's own production threshold of `1e-3` had never been run on
N2 at all. This document records the missing curve.

## Setup

N2, `N 0 0 0; N 0 0 1.098`, cc-pVDZ, RHF, frozen core 2, CAS(10e,9o), `(5,5)`,
`15876` determinants, product-grid ceiling `43186`. Schmidt space A = 5
occupied orbitals, P blocks `n_A = 8,9,10`. **Lanczos seed**, so no exact CI is
read; self-consistent outer loop with the pre-registered controls
(`outer_mixing=0.7`, `density_tol=1e-6`, `energy_tol=1e-7`, `max_iter=12`,
`damping=0.7`, `residual_tol=1e-9`, `min_denominator=1e-6`);
`enrichment_strength=0`; shared wave operator. Errors are absolute, against the
checksummed reference bundle, in mH. Local WSL, one thread.

## Ground state alone

| `svd_eps` | `D` | error | converged | wall | peak RSS |
|---|---|---|---|---|---|
| 1e-2 | 342 | `3.009` | yes | 24 s | 287 MiB |
| 5e-3 | 810 | **`1.209`** | yes | 89 s | 525 MiB |
| 3e-3 | 942 | **`0.771`** | yes | 108 s | 595 MiB |
| 1e-3 | 2466 | **`0.283`** | yes | 309 s | 1659 MiB |

**Chemical accuracy, `1.6 mH`, is crossed at `eps = 5e-3` and `D = 810`**, and
the error falls to `0.283 mH` by `eps = 1e-3` at `D = 2466`, which is 5.7 percent
of the product-grid ceiling. The seed reads no exact CI and the outer loop
converges in every row.

## Four states covering the three lowest levels, sharing one basis

| `svd_eps` | `D` | `|P|` | `|Q|` | weighted error | worst state | converged | wall | peak RSS |
|---|---|---|---|---|---|---|---|---|
| 1e-2 | 216 | 199 | 17 | `18.203` | `38.591` | no | 29 s | 454 MiB |
| 5e-3 | 978 | 635 | 343 | `12.710` | `32.234` | no | 156 s | 844 MiB |
| 3e-3 | 2517 | 905 | 1612 | `9.473` | `29.321` | yes | 534 s | 1714 MiB |
| 2e-3 | 2841 | 945 | 1896 | `7.653` | `22.533` | yes | 1069 s | 2249 MiB |
| 1e-3 | 4987 | 1117 | 3870 | `6.999` | `13.411` | yes | 1664 s | 4082 MiB |

## What the two curves say together

**The threshold is no longer the limiting variable for the multi-state case.**
From `3e-3` to `1e-3` the weighted error falls only `9.473 -> 7.653 -> 6.999 mH`
while `D` nearly doubles, `2517 -> 4987`. The returns are clearly diminishing.

**At `eps = 1e-3` the two differ by a factor of 25**, `0.283` against
`6.999 mH`, and the four-state run holds the larger basis, `4987` against `2466`.

So the accuracy of the construction is not in question. What costs is requiring
one basis to serve four states at once, two of which are the exactly degenerate
`3Pi_g` components. The record contains the converse measurement and the two
agree rather than conflict: a ground-state-only basis overestimates every triplet
by `+323` to `+372 mH`, so a shared basis is necessary; these curves show it is
also expensive.

The wave-operator error is `1.4e-10` to `2.8e-10 mH` in every row, so as before
the entire error is dmSVD truncation of the model space and the downfolding is a
solver.

## The candidate fix is in the record, not in a new construction

| Job | Space | `eps` | `D` | grid ceiling | Error | Downfolding |
|---|---|---|---|---|---|---|
| 15371, gs | CAS(10,10) | 1e-3 | 4,668 | 184,756 | `+0.144 mH` | per-state Löwdin `m=1` |
| 15372, sa 5 states | CAS(10,10) | 1e-3 | 15,198 | 184,756 | all within `±1 mH` | per-state Löwdin `m=1` |
| this document, sa 4 states | CAS(10e,9o) | 1e-3 | 4,987 | 43,186 | `6.999 mH` | shared residual-dressed Omega |

Chemical accuracy on five states has already been obtained in this project, with
**per-state** Löwdin centering rather than one shared wave operator, and
`wave_operator.py`'s own docstring at line 201 states the mechanism: shared
across states breaks excited states, per-state centering fixes it. Hypothesis H7
had been tested only on H2O, which returned `INDISTINGUISHABLE` on the system
least able to express the effect.

H7 on N2 at these thresholds is therefore the deciding experiment, and it needs
no new code: `omega_mode='per_state'` already exists.

## Caveats

- `enrichment_strength = 0`, so the outer map is the rank contraction proved in
  `outer_map_rank_contraction.md`. The `1e-2` and `5e-3` four-state rows did not
  converge, and their `D` is post-contraction. The tighter rows converged, so the
  contraction is less damaging at a tight threshold, which is itself new.
- One geometry, one basis set, one molecule. These curves establish the shape of
  the threshold axis for N2 and nothing more general.
- `D` is a product-grid count and exceeds the CI dimension in general, so the
  fractions quoted against the ceiling `43186` are fractions of the grid, not of
  the determinant space.

Artifacts: `~/work/krylov-dci-run-artifacts/gateC/n2_threshold_curve.json`,
`n2_per_state_gs.json`.
