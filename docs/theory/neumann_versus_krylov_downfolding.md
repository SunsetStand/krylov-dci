# Neumann truncation versus Krylov-Galerkin resummation

This note answers a design question: can the Neumann first-order term replace the
Krylov subspace basis, and which is more efficient? The Neumann implementation
already exists in `dm_svd_dci/neumann_effective_ham.py`, whose own docstring says
it "replaces the Krylov-subspace + Löwdin-resolvent approach", so the substitution
has effectively been made once already without a recorded comparison.

Reproduce with:

```bash
python scripts/diagnostics/run_neumann_vs_krylov_comparison.py --output-dir <artifacts>
```

Cost: `4.1 s`, `109 MiB`.

## They are not independent alternatives

Both approximate the same object:

```text
(E I - H_QQ)^-1 = A sum_k (B A)^k ,   A = (E I - D_QQ)^-1 ,   B = H_QQ - D_QQ
```

Neumann truncates that series at order `k`. The Krylov route builds

```text
K_m = span{ A H_QP, (AB) A H_QP, ..., (AB)^m A H_QP }
```

and then inverts **exactly** inside `K_m`. But `K_m` is precisely the span of the
terms the Neumann series truncates. So the two use the same information; the
difference is what they do with it. The truncated series is one fixed polynomial
in `(AB)`, whereas the Galerkin projection is the optimal polynomial of the same
degree over that subspace. This is the standard relation between a Richardson or
Jacobi iteration and a Krylov projection method.

Consequently the Krylov route is never worse at equal subspace, and the real
question is cost, not correctness.

## Equal-cost measurement, weakly correlated regime

H2O/STO-3G CAS(6e,5o) at equilibrium, error against the exact resolvent at the
same `E_0`, counting applications of `B` as the cost unit:

| `\|P\|` | Neumann `k=1` | Krylov `m=0` |
|---|---|---|
| 10 | `+1.0390 mH` at 10 matvecs | `+0.5536 mH` at 10 matvecs |
| 20 | `+0.8820 mH` at 20 matvecs | `+0.4757 mH` at 20 matvecs |
| 40 | `+0.9872 mH` at 40 matvecs | `+0.4102 mH` at 40 matvecs |

At identical matvec count the Galerkin step is about **twice as accurate**. That
is the expected ordering and it is modest. On its own it would not justify the
extra machinery, which is the legitimate part of the case for Neumann.

## The decisive variable is the spectral radius

The Neumann series converges only when `rho(BA) < 1`. The Galerkin projection has
no such requirement: it converges monotonically in subspace dimension regardless.

For the stretched water case the difference is not a matter of degree:

| Method | matvecs | error vs exact resolvent |
|---|---|---|
| Neumann `k=1` | 40 | `+207.42 mH` |
| Neumann `k=2` | 80 | `-223.69 mH` |
| Neumann `k=3` | 120 | `+175.34 mH` |
| Neumann `k=4` | 160 | `-914.58 mH` |
| Krylov `m=0` | 40 | `+48.35 mH` |
| Krylov `m=1` | 160 | `-0.0000 mH` |

At `rho(BA) = 1.579` the Neumann sequence oscillates with growing amplitude, which
is divergence, not slow convergence: **increasing the order makes it worse**.
Krylov `m=1` reaches the exact resolvent to machine precision at the same matvec
count at which Neumann `k=4` is off by nearly one hartree.

## Where N2 actually sits

Matrix-free power iteration for `rho(BA)`, N2/cc-pVDZ CAS(10e,9o), `|P| = 200`:

| `R/Re` | `R` (Angstrom) | `rho(BA)` | Neumann |
|---|---|---|---|
| 0.8 | `0.8784` | `0.736` | converges |
| 1.0 | `1.0980` | `0.884` | converges, slowly |
| 1.5 | `1.6470` | `1.945` | **diverges** |
| 2.0 | `2.1960` | `3.442` | **diverges** |
| 2.5 | `2.7450` | `3.820` | **diverges** |
| 3.0 | `3.2940` | `3.868` | **diverges** |

`SKILL.md` specifies potential energy curves at exactly `R = 0.8, 1.0, 1.5, 2.0,
2.5, 3.0 Re`. **Four of those six required benchmark points have a divergent
Neumann series.** Even at equilibrium, `rho = 0.884` means the truncation error
falls only as `0.884^k`, so roughly 37 orders would be needed for two digits.

This is not a defect of the implementation. It is the convergence radius of the
expansion, and no amount of code quality changes it.

## A consequence for the residual-dressed wave operator

The `Omega` inner iteration uses the same diagonal preconditioner. Writing
`q = Omega c`, the Q-space equation is `(E - H_QQ) q = H_QP c`, and the update
`delta q = R / (E - diag H_QQ)` with damping `w` gives the iteration matrix

```text
M = (1 - w) I + w (A B)
```

so an eigenvalue `lambda` of `AB` maps to `1 - w + w*lambda`. Convergence requires
`|1 - w + w*lambda| < 1` for every eigenvalue. Since

```text
|1 - w + w*lambda| >= w|lambda| - (1 - w)   for 0 < w <= 1
```

convergence **requires** `w < 2 / (|lambda| + 1)`. With the measured spectral
radii this gives a hard upper bound on the damping:

| `R/Re` | `rho(BA)` | required `w <` | current default `w = 0.7` |
|---|---|---|---|
| 1.0 | `0.884` | `1.062` | fine |
| 1.5 | `1.945` | `0.679` | **already violated** |
| 2.0 | `3.442` | `0.450` | violated |
| 3.0 | `3.868` | `0.411` | violated |

The current wave-operator damping of `0.7` is above the necessary bound from
`R = 1.5 Re` outward. No choice of damping in `(0, 1]` rescues `R >= 2.0 Re`
within a useful iteration count. This is a prediction, not yet a measurement of
the solver, but it is a rigorous bound rather than an intuition, and it offers a
candidate explanation for the frozen residual stall recorded in
`docs/theory/n2_casci_root_reproducibility_protocol.md`.

## Answer and recommendation

**In the weakly correlated regime the two are comparable**, with Krylov about
twice as accurate per matvec. If the target were equilibrium geometries only,
Neumann `k=1` would be a reasonable simplification, and the instinct behind the
question is sound: the project's own `lessons_learned.md` records that the SVD
compression of the Krylov basis achieves *zero* compression at `P <= 4000`, so
that step currently buys nothing while costing memory and stability.

**In the strongly correlated regime Neumann is not less efficient, it is
unusable**, because the series it sums does not converge. Since strong
correlation is the entire purpose of a multireference downfolding method, Neumann
cannot be the primary route.

The useful middle path, which the measurement above already uses, is to keep the
Galerkin step but drop the SVD compression: form `K = A H_QP`, orthonormalize by
QR, and invert exactly inside it. That is Krylov `m = 0`. It costs the same
matvecs as Neumann `k = 1`, is twice as accurate at equilibrium, remains
well behaved when the series diverges, and avoids the MGS-plus-SVD machinery that
the project has documented as its main source of numerical fragility.

Three concrete actions follow.

1. **Make `rho(BA)` a required, recorded diagnostic.** It costs about 60 matvecs
   by power iteration and it decides whether any Neumann-type expansion, including
   the `Omega` dressing, can converge at all.
2. **Do not tune the wave-operator damping empirically before checking the bound
   `w < 2/(rho + 1)`.** A damping that violates it cannot be rescued by more
   iterations.
3. **Beware the error-cancellation trap.** Measured against exact CASCI rather
   than against the exact resolvent, Neumann `k=1` looks *better* than the exact
   resolvent itself, `+0.35 mH` against `-0.69 mH` at `|P| = 10`. That is
   truncation error cancelling the error from evaluating the resolvent at `E_0`
   instead of the self-consistent energy. Any comparison must be made against the
   exact resolvent at the same `E_0`, or this cancellation will be mistaken for
   accuracy.
