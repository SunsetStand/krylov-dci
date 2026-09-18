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

**Scored against the committed reference bundle.** An earlier version of this
document reported this table against `evaluate_reference_energies`, whose
unvalidated `nroots = 4` solve placed the fourth reference `27.2736 mH` too
high by missing a member of the degenerate `3Pi_g` level. That defect is fixed
in `7b6c72d`; these are the corrected numbers, and the per-state resolution is
new.

| `svd_eps` | `D` | weighted | S0 `Ag` | S1 `B1u` | S2 `B2g` | S3 `B3g` | converged | wall | peak RSS |
|---|---|---|---|---|---|---|---|---|---|
| 1e-2 | 216 | `25.022` | `7.119` | `15.385` | `38.591` | `38.991` | no | 30 s | 454 MiB |
| 5e-3 | 978 | `19.529` | `2.639` | `9.720` | `32.234` | `33.522` | no | 152 s | 844 MiB |
| 3e-3 | 2517 | `16.292` | **`1.321`** | `4.454` | `29.321` | `30.070` | yes | 522 s | 1662 MiB |
| 2e-3 | 2841 | `12.996` | **`0.850`** | `4.276` | `22.533` | `24.323` | yes | 1010 s | 2168 MiB |
| 1e-3 | 4987 | `7.321` | **`0.342`** | **`1.249`** | `13.411` | `14.280` | yes | 1609 s | 3920 MiB |

## What the two curves say together

**The multi-state curve has not plateaued.** The corrected weighted error falls
`16.292 -> 12.996 -> 7.321 mH` from `eps = 3e-3` to `1e-3`, and the last step
nearly halves it. An earlier version of this document concluded the opposite,
that returns were diminishing and the threshold had stopped being the limiting
variable. That conclusion was an artifact of the defective reference, which
compressed the apparent progression to `9.473 -> 7.653 -> 6.999`.

**The error is concentrated in the degenerate pair, and the states separate
cleanly by symmetry.** At `eps = 1e-3` the `Ag` ground state is at `0.342 mH`
and the `B1u` triplet at `1.249 mH`, both inside chemical accuracy, while the two
`3Pi_g` components sit at `13.411` and `14.280 mH`. S2 and S3 track each other
throughout, splitting by only `0.4` to `1.8 mH`, so they behave as one object.
They converge roughly three times more slowly in `eps` than S0 and S1.

**The degeneracy splitting originates in the Schmidt basis, not in root matching
or reconstruction.** Measured at `eps = 3e-3` and `1e-3`, the splitting of the
method's own pair and the splitting of the exact diagonalization of the same
`H_emb` are identical to every printed digit, `0.7497` and `0.8685 mH`
respectively. The wave-operator error stays at `1e-10 mH` in every row, so the
downfolding is exact and the entire error is dmSVD truncation, as before.

**The measurement is at the local memory wall, not at a scientific limit.** The
`eps = 1e-3` four-state point needs `3920 MiB` at `D = 4987`; one further
tightening would exceed this machine. So the honest statement is that the four
state curve was still falling steeply where the measurement had to stop.

**Consequence: the accuracy goal and the scalability work are the same problem.**
Chemical accuracy on the `3Pi_g` pair evidently needs a larger `D` than this
machine reaches, and what blocks a larger `D` is the `4.2 x M x D` determinant
space expansion in the embedded-Hamiltonian build. Streaming that build, or the
rank-`r` Q-space compression analysed in
`docs/theory/hpq_svd_qspace_compression_analysis.md`, is therefore not merely a
reach-extension exercise: it is the route to the remaining accuracy.

Single state against four states, for reference: at `eps = 1e-3` the ground state
alone gives `0.283 mH` at `D = 2466`, and inside the four-state calculation the
same state gives `0.342 mH` at `D = 4987`. Sharing the basis costs the ground
state very little. What it costs is concentrated on the states whose entanglement
structure the shared basis represents worst.

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
