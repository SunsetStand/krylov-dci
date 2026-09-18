# The excited-state error is a seed problem, and no existing seed solves it

N2 CAS(10e,9o)/cc-pVDZ, four states covering the three lowest levels, scored
against the committed bundle. This localizes the `13` to `30 mH` error on the
`3Pi_g` pair, which `docs/development/n2_threshold_curves.md` reports, to a
specific and measurable cause, and shows that the obvious remedy does not work.

## 1. The error is entirely in the embedded space

For every configuration tested, the method's energies equal the exact
diagonalization of its own `H_emb` to the printed precision:

| configuration | `D` | S0 | S1 | S2 | S3 | method minus embedded-exact |
|---|---|---|---|---|---|---|
| exact seed, frozen | 4306 | `1.054` | `1.015` | `1.596` | `10.723` | `0.0000 mH`, all four |
| lanczos, converged | 2517 | `1.321` | `4.454` | `29.321` | `30.070` | `0.0000 mH`, all four |

The downfolding is exact and root matching is not implicated. The whole error is
in the construction of the embedded space, as the wave-operator error of
`1e-10 mH` has said throughout.

## 2. The basis is capable; the loop does not reach it

With a basis built from the exact states and no iteration, three of the four
states reach chemical accuracy at `eps = 3e-3`: `1.054`, `1.015`, `1.596 mH`.
The self-consistent loop from a Lanczos seed reaches `29.321` and `30.070 mH` on
the same two `3Pi_g` states at the same threshold, while keeping S0 at
`1.321 mH`. So the threshold is sufficient and the fixed point is not.

Enrichment does not close it. At `eps = 3e-3`, `lambda = 0.5` moves the weighted
error from `16.291` to `14.982 mH` and repairs S1 from `4.454` to `1.675`, which
is the rank contraction behaving as documented, but `3Pi_g` only moves from
`29.321 / 30.070` to `28.339 / 28.346`. **This is a different failure from the
rank contraction.**

## 3. The mechanism: the seed does not span the irrep

Projection of each reference state onto the span of the seed states, measured as
`||Q^dag psi_ref||^2` with `Q` an orthonormal basis of the seed span:

| seed | S0 `Ag` | S1 `B1u` | S2 `B2g` | S3 `B3g` | `3Pi_g` total |
|---|---|---|---|---|---|
| `lanczos` (the default) | `0.9994` | `0.1269` | **`6.84e-26`** | **`2.54e-25`** | `3.2e-25` |
| `hf` | `0.9774` | `0.9675` | `5.88e-25` | `1.21e-25` | `7.1e-25` |
| `cis` | `0.9393` | `0.9690` | `6.58e-25` | `2.01e-25` | `8.6e-25` |
| `selci` | `0.9975` | `8.72e-33` | `3.47e-31` | `4.89e-29` | `4.9e-29` |
| `perturbed` | `0.9310` | `0.9601` | `3.88e-04` | `8.32e-04` | `1.2e-03` |
| **`trunc`** | `0.9989` | `0.9587` | **`0.9699`** | **`0.9699`** | **`1.940`** |
| `exact` (control) | `1.0000` | `1.0000` | `0.9735` | `0.0265` | `1.0000` |

The `exact` row shows why the span and not the individual overlap is the right
quantity: an exactly degenerate pair is defined only up to rotation, so its
per-vector overlaps split arbitrarily while the total is `1.0000`.

**The default seed's projection on the `3Pi_g` pair is `6.84e-26`.** Gate B
measured `6.76e-26` for the naive Davidson guess on the same system and
established that a projection which starts at zero stays at zero under Krylov
expansion. It is the same exclusion. Gate B fixed it for the *reference* solve,
by resolving per irrep in `D2h`; nobody checked the *seed*.

The physical reason is direct: the `3Pi_g` states carry substantial
double-excitation character relative to the closed-shell reference, and every
seed except `trunc` and `exact` is built from the Hartree-Fock determinant plus
single excitations (`perturbed` and `cis` use excitation rank `<= 1`), or from a
Krylov expansion started at the lowest-diagonal determinant of each block, which
is symmetry-trapped.

The consequence propagates through the construction:

1. the state-averaged density carries no `3Pi_g` structure;
2. block `n_A = 9`, where the `3Pi_g` pair holds `92.7 percent` of its weight
   and S1 holds `96.5 percent`, retains rank `6 x 6` instead of its **full**
   `10 x 8` (`dim_A(9) = C(10,9) = 10`, `dim_B(1) = 8`), which the exact seed
   does reach;
3. the rank contraction theorem then guarantees the loop can never recover it.

## 4. Irrep coverage is necessary but not sufficient

`trunc` is the one non-exact seed that spans the pair, at `0.9699` on each
member. It does not fix the problem, it makes it worse:

| seed | `D` | weighted | S0 | S1 | S2 | S3 | block 9 | outer |
|---|---|---|---|---|---|---|---|---|
| `lanczos` | 2517 | `16.291` | `1.321` | `4.454` | `29.321` | `30.070` | `6x6` | 12 |
| `trunc` | **1059** | `27.232` | `2.690` | `33.268` | `36.485` | `36.486` | `8x6` | 7 |
| `perturbed` | **67** | `60.713` | `86.037` | `37.467` | `57.855` | `61.493` | `8x8` | 10 |

`trunc` starts with `0.97` of the pair and still converges to a basis less than
half the size, with a worse error on every state. Its `3Pi_g` content does not
survive the loop because the whole basis collapses.

So a seed needs **two independent properties**, and the seven existing families
provide at most one each:

- **irrep coverage**: non-zero projection on every target state. Only `trunc`
  and `exact` have it here.
- **representative entanglement structure**: the state-averaged density must
  have the rank structure the converged states need. Only `lanczos` and `exact`
  have it, which is what the recorded H2O scan already showed, where at
  `eps = 1e-4` `lanczos` gives `0.0012 mH` against `selci` `10.25`, `hf` `10.99`
  and `cis` `36.91`.

**No non-exact seed has both.** That is the finding, and it explains why the
ground state reaches `0.283 mH` while the `3Pi_g` pair does not: S0 needs only
the second property, and the `3Pi_g` pair needs both.

## 5. Scope correction to H1

H1, "a non-exact seed reaches the same fixed point as an exact seed", is recorded
as **CONFIRMED** on the strength of the H2O scan and the N2 ground state. That
verdict is sound for what it tested and does not hold in general. The correct
statement is narrower:

> A non-exact seed reaches the exact-seeded fixed point **only within the
> irreducible representations its own span covers.** A target state in an irrep
> the seed does not span is unreachable, because the state-averaged density never
> acquires that structure and the outer map cannot increase a retained rank.

H1 should be restated in symmetry-resolved form and re-tested on N2, which is the
system that expresses the effect. H2O could not have detected it.

## 6. Remedy, not yet implemented

Apply the Gate B recipe to the seed rather than only to the reference: build the
Lanczos seed **per irrep** and merge, or start the block Lanczos from a set of
determinants that spans every target irrep. That gives irrep coverage from
`trunc` and entanglement structure from `lanczos`, which are the two properties
no single existing seed has. `wfnsym`-resolved solves in `D2h` are already used
by `reference_bundle.py`, so the machinery exists.

Two supporting changes belong with it:

1. **A seed coverage assertion.** For every target state, assert the projection
   onto the seed span exceeds a threshold before the outer loop starts. That
   turns a silent `1e-26` into an immediate failure and would have caught this.
   It is the seed-side analogue of the reference validation added in `7b6c72d`.
2. **Report the block ranks against their ceilings.** Block `n_A = 9` sitting at
   `6 x 6` of a possible `10 x 8` is the visible symptom, and nothing currently
   reports the ratio.

Artifacts: `~/work/krylov-dci-run-artifacts/gateC/pig_diagnosis.json`,
`contraction_vs_basis.json`, `seed_fix_test.json`.
