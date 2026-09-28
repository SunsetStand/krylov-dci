# The Lanczos seed lost orthogonality, and stopped being variational

Found when the cluster run at `steps = 14`, `eps = 1e-3` (Slurm job 20910,
task 2) returned a ground state `321 mH` out and did not converge, while
`steps = 10` at the same threshold reached `0.906 mH` weighted. Fixed in
`c4c953d`.

## Mechanism

`_lanczos_seed` grew a block Krylov basis with **single-pass** modified
Gram-Schmidt and then solved `K^T H K c = E c` as a **standard** eigenproblem,
which is variational only if `K^T K = I`. N2 CAS(10e,9o), irrep-complete
starting block of 17 vectors:

| steps | basis | `max abs(K^T K - I)` | lowest Ritz minus exact | same basis, orthonormalized first |
|---|---|---|---|---|
| 6 | 119 | `5.5e-10` | `+0.045 mH` | `+0.045 mH` |
| 8 | 153 | `9.8e-08` | `+0.003 mH` | `+0.003 mH` |
| 10 | 187 | `1.9e-05` | `+0.000 mH` | `+0.000 mH` |
| 12 | 221 | `3.8e-03` | **`-0.268 mH`** | `+0.000 mH` |
| 14 | 255 | `6.3e-01` | **`-17530.9 mH`** | `+0.000 mH` |
| 16 | 289 | `9.3e-01` | **`-28083.4 mH`** | `+0.000 mH` |

Orthogonality degrades geometrically, about two orders of magnitude every two
steps. From 12 steps the lowest root is **below the exact ground state**, which
no variational calculation can produce, so it is spurious. The rightmost column
shows the Krylov *space* was always adequate; only the orthonormality
assumption failed.

The candidates were not near-noise vectors, as first suspected: the smallest
retained norm ratio after projection stays at `5.9e-3` and the basis keeps full
numerical rank. This is the classical loss of orthogonality of Lanczos once
Ritz values converge, accumulating through the chain.

At 14 steps the seed's returned states had true Rayleigh quotients `2197`,
`2299`, `2532` and `2808 mH` above the exact energies. The state-averaged density
built from them carried no useful structure, and the outer loop could not
recover, which is the observed `321 mH` ground state.

## Fix

Orthogonalize twice at every step, which holds orthogonality at machine
precision, and verify orthonormality before the Rayleigh-Ritz, falling back to an
SVD re-orthonormalization that preserves the span if the Gram error exceeds
`1e-10`. After the fix, returned seed states lie above the exact energies by:

| steps | S0 | S1 | S2 | S3 | S1 coverage | pair coverage |
|---|---|---|---|---|---|---|
| 10 | `0.000` | `21.434` | `4.457` | `5.135` | `0.709` | `1.955` |
| 12 | `0.000` | `9.285` | `0.592` | `0.670` | `0.869` | `1.998` |
| 14 | `0.000` | `5.532` | `0.085` | `0.096` | `0.899` | `2.000` |
| 20 | `0.000` | `1.447` | `0.000` | `0.000` | `0.988` | `2.000` |

None lies below. **The 10-step row is unchanged from before the fix to the
printed precision**, so every recorded 10-step result, including all of job
20910 except task 2, remains valid. Those results were obtained two steps from
the failure.

Regression test: `tests/regression/test_lanczos_seed_orthogonality.py`.

## What this exposed

The 20-step row shows the seed on its own is nearly exact: better than the full
method's final answer at `eps = 5e-4`. That raises the question of what the seed
costs relative to the exact solve it is meant to avoid, which is answered in
`docs/development/seed_cost_versus_exact_solve.md`.
