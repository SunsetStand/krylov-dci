# Gate C on N2: the rank contraction, its repair, and H6

N2/cc-pVDZ CAS(10e,9o), four states covering the three lowest levels, `(5,5)`,
frozen core 2, Schmidt space A = 5 orbitals, P blocks `n_A = 8,9,10`, Lanczos
seed. Thresholds and tolerances are the pre-registered ones: agreement
tolerance `tol_E = 0.05 mH`, matched-dimension tolerance `2 percent`.

H2O could not test either of these questions: it has no rank asymmetry, and its
basis was not collapsing. N2 has both.

## H3 reversed: self-consistency helps, once the contraction is repaired

The earlier falsification of H3 on H2O, and the much worse behaviour on N2, were
measuring a broken map. `docs/theory/outer_map_rank_contraction.md` shows the
outer map cannot increase a retained rank, so every basis it reaches is a
subspace of the frozen one, and on N2 it lost 70 percent of the dimension while
degrading the energy by `2.44 mH`.

Residual-driven Krylov enrichment adds the full-space residual
`R_k = H C_k - E_k C_k` to the state-averaged density with weight
`lambda * w_k`. It is computed in the determinant basis, because the embedded
residual already lies inside the retained span, and it is not normalized, so it
fades as `||R_k||^2` and leaves the exact solution a fixed point.

Effect at `eps = 1e-2`, against a frozen baseline of `15.490 mH` at `D = 720`:

| `lambda` | final `D` | error | rank increases | dimension trajectory |
|---|---|---|---|---|
| 0.00 | 216 | 18.203 | **0** | `720, 296, 229, 216, ...` |
| 0.05 | 203 | 17.387 | 0 | `720, 282, 220, 203, ...` |
| 0.20 | 239 | 16.848 | 5 | `720, 358, 272, 295, ...` |
| 0.50 | 396 | 15.132 | 8 | `720, 414, 372, 382, ...` |
| 1.00 | 490 | 14.125 | 13 | `720, 528, 395, 591, ...` |

The contraction breaks from `lambda = 0.2` upward. At `lambda = 1.0` the
dimension trajectory oscillates strongly, so `0.5` is the usable setting.

**At matched embedded dimension**, which is the only fair comparison:

| | `D` | error |
|---|---|---|
| self-consistent, `lambda = 0.5` | 396 | **`15.1315 mH`** |
| frozen | 398 | `17.8437 mH` |

Dimension gap `0.5 percent`, inside the `2 percent` tolerance. Self-consistency
wins by **`2.71 mH`**, which is 54 times the agreement tolerance.

**H3 is confirmed on N2, conditional on the enrichment.** Without it, H3 is
falsified on both systems. The honest statement is therefore not "self-consistency
works" but "self-consistency works only once the map is made rank-increasing;
as originally formulated it cannot work, for a reason that is provable rather
than empirical".

### Reaching a converged fixed point

The constant-strength runs above never converged, and the diagnosis is specific:
the projector distance stayed at `5.7` to `6.2` at **every** iteration, so the
Schmidt basis was rotating completely each time and never settling, while the
density change plateaued near `4e-3` against a `1e-6` tolerance.

The cause was a wrong assumption in the design. The enrichment was expected to
fade because `||R_k||` would vanish, but **it does not**: a state in a truncated
space can never be an exact eigenvector of the full Hamiltonian, so `||R_k||`
plateaued at about `0.13` and the enrichment acted as a permanent rotating
perturbation. The map was in a limit cycle.

The strength is therefore **annealed**, `lambda_t = lambda_0 * decay^t`, which is
what DMRG does with its noise term. This converges:

| `decay` | final `D` | error | converged | outer iterations | final `d_rho` | final projector distance |
|---|---|---|---|---|---|---|
| 1.0 | 412 | `15.2087` | **no** | 16 | `3.6e-03` | `5.70` |
| 0.7 | 155 | `17.7453` | no | 16 | `5.7e-06` | `1.3e-03` |
| 0.5 | 178 | `17.5816` | **yes**, 13 | 13 | `3.5e-07` | `7.9e-05` |
| 0.3 | 203 | `17.2539` | **yes**, 13 | 13 | `3.2e-07` | `1.1e-11` |

At `decay = 0.3` the projector distance falls to `1.1e-11`, so the basis is
completely settled.

Annealing to zero lets the contraction resume, which prunes the basis from about
`400` down to `180` to `200` and gives back part of the absolute accuracy. **But
at matched dimension the converged runs still win decisively:**

| `decay` | self-consistent, converged | frozen at matched `D` | gap | advantage |
|---|---|---|---|---|
| 0.5 | `D=178`, `17.5816 mH` | `D=177`, `20.3947 mH` | `0.6 percent` | **`+2.8131 mH`** |
| 0.3 | `D=203`, `17.2539 mH` | `D=201`, `19.7860 mH` | `1.0 percent` | **`+2.5322 mH`** |

Both dimension matches are inside the `2 percent` tolerance and both advantages
are about 50 times the agreement tolerance. **H3 is confirmed with a genuinely
converged fixed point**, not only as a best-within-budget number.

The trade-off is explicit and should be reported as such: constant enrichment
gives a better absolute error at a larger basis but no fixed point, while
annealed enrichment gives a converged fixed point at a smaller basis and a
larger margin over frozen at equal cost.

## H6: partially supported, and the effect tracks the asymmetry

Rectangular ranks are real on N2 and absent on H2O. At `eps = 1e-2` the retained
ranks are

```text
n=6: 1x1   n=7: 16x15   n=8: 22x21   n=9: 4x4   n=10: 1x1
```

so two blocks are asymmetric, by one unit each.

Compared against a symmetric allocation at directly matched dimension, with no
interpolation, on a frozen basis:

| matched `D` | gap | rectangular | symmetric | advantage | verdict |
|---|---|---|---|---|---|
| 715 vs 720 | `0.7 percent` | `15.4898` | `15.5275` | `+0.0377 mH` | **indistinguishable** |
| 1559 vs 1539 | `1.3 percent` | `12.1716` | `12.3933` | `+0.2217 mH` | **rectangular wins** |

The advantage is larger where the asymmetry is larger: the `D = 1539` point
carries three asymmetric blocks against two at `D = 720`. That is the trend the
claim predicts, but two points do not establish it.

**A methodological warning worth keeping.** An earlier version of this
comparison interpolated the symmetric error-versus-dimension curve instead of
matching directly, and reported `+0.1434 mH` at `D = 720` rather than
`+0.0377 mH`. The curve is convex, so linear interpolation overestimates the
symmetric error and biases the comparison toward rectangular. Direct matching
turned one of the two verdicts from a win into a tie. **Interpolated
matched-cost comparisons must not be used for this claim.**

## Status of H6 as a novelty candidate

It is the only surviving candidate from the literature survey, and it is now
**partially supported**: real, measurable, in the predicted direction, and
correlated with the amount of asymmetry, but at one of two tested dimensions it
falls below the agreement tolerance. Establishing it needs a denser sweep of
matched dimensions, and ideally a system with stronger asymmetry than two or
three blocks differing by one unit.

## What is settled and what is not

Settled:

- the outer map is a rank contraction, by construction, verified with no rank
  increase anywhere on either system;
- residual-driven enrichment breaks it, and at matched dimension turns
  self-consistency from harmful into a `2.71 mH` benefit;
- the wave-operator error is `1e-10` or smaller on both systems, so accuracy is
  set by the Schmidt truncation and the wave operator is a solver.

Not settled:

- the enrichment strength and decay, chosen as `0.5` and `0.3` from small scans
  on one system at one threshold;
- whether the constant-strength regime has a fixed point at all, or only a limit
  cycle;
- H6, which is supported at one matched dimension and not at the other.
