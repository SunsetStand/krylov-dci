# Streaming H_emb route: the H_QQ discrepancy, found and fixed

The streaming route is wired into the state-averaged pipeline as
`scheme='streaming'`. It uses `dm_svd_dci/streaming_ops.py`'s `StreamBuilder`,
which batches the determinant-space expansion and exposes `H_QQ` only as a
matvec, feeding the matrix-free low-rank wave-operator solver.

The two Hamiltonian construction routes disagreed. The cause was a defect in
the batched `H_QQ` apply; it is now fixed and the routes agree.

## The memory result, which is real

N2 CAS(10e,9o), `M = 15876` determinants:

| `eps` | `D` | scheme | wall | peak | `peak / (M x D)` |
|---|---|---|---|---|---|
| `2e-2` | 97 | A | `3.6 s` | `99.3 MiB` | `8.45` |
| `2e-2` | 97 | streaming | `3.2 s` | **`36.0 MiB`** | **`3.06`** |
| `1e-2` | 720 | A | `10.0 s` | `363.3 MiB` | `4.17` |
| `1e-2` | 720 | streaming | `108.4 s` | **`136.6 MiB`** | **`1.57`** |

Peak memory falls by about `2.7x`, and the constant multiplying `M x D` falls
from `4.17` to `1.57`. That is the saving the CAS(14,10) OOM needed.

The cost is time: `108 s` against `10 s` at `D = 720`, roughly eleven times
slower, because the expansion is batched rather than done once. That trade is
the point of streaming, but the factor is large enough to need attention before
the route is used in production.

## The blocker, localized

Comparing the two routes on identical Schmidt data at `eps = 1e-2`,
`D = 720`, `P = 479`, `Q = 241`:

| Object | `max |A - streaming|` | relative |
|---|---|---|
| `H_PP` | `0.000e+00` | exact |
| `H_PQ` | `1.162e-16` | exact |
| `H_QQ` diagonal | `1.421e-14` | exact |
| **`H_QQ @ v`** | **`1.015e+00`** | **`3.19e-04`** |

Everything agrees except the **off-diagonal action of `H_QQ`**. The resulting
ground-state energies differ by `1.11e-04 Ha`, that is `0.111 mH`, which is far
above numerical noise and above chemical accuracy relevance for this work.

At the looser threshold `eps = 2e-2`, `D = 97`, the two routes agree to
`2.4e-13 Ha`, so the discrepancy grows with the size of the Q space rather than
being a constant offset such as a misplaced `ecore`.

## The defect

`StreamBuilder.apply_hqq_batch` computes

```text
result[i, k] = sum_j <CI_i| H |CI_j> B[j, k]
```

where the sum over `j` must run over the **whole** Q space. The batched loop
took `B_sub = B[q_start:q_end, :]`, restricting that sum to the current batch,
so every cross-batch coupling was dropped and only the **block-diagonal** part
of `H_QQ` survived, with blocks equal to the batches.

That accounts for the whole observation. The diagonal always falls inside its
own batch, so it stayed correct while the off-diagonal action did not, which is
exactly the localization above. The error vanished whenever the batch reached
`|Q|`, which is why `Q = 4, 17, 24` were exact and `Q = 43, 241` were not: the
default batch size is 32.

Confirmed by sweeping the batch size on H2O with `Q = 43`:

| `q_batch_size` | relative error, before the fix |
|---|---|
| 8 | `5.08e-04` |
| 16 | `5.15e-04` |
| 32 | `2.47e-04` |
| 43 | `3.34e-17` |
| 64 | `3.34e-17` |

## The fix

Contract first, then apply the Hamiltonian:

```text
psi_k = sum_j B[j, k] CI_j          one pass over Q, batched
H psi_k                             r sigma evaluations, not one per Q index
result[i, k] = <CI_i| H psi_k>      one pass over Q, batched
```

Batching is kept for the two expansion passes, which is where the memory is,
but the contraction is over all of Q by construction. The corrected form is also
much cheaper: `r` sigma evaluations per matvec instead of `|Q|`.

After the fix the relative error is `3.3e-17` on H2O and `4.0e-17` on N2, at
every batch size from 4 upward, and end to end the two routes agree:

| `eps` | `D` | scheme A | streaming | difference |
|---|---|---|---|---|
| `2e-2` | 97 | `-109.0254229124` | `-109.0254229124` | `1.3e-13 Ha` |
| `1e-2` | 720 | `-109.0354440546` | `-109.0354440546` | `2.4e-13 Ha` |
| `5e-3` | 1539 | `-109.0377261767` | `-109.0377261767` | `2.8e-13 Ha` |

`tests/regression/test_streaming_hqq_batching.py` pins both halves: agreement
with the assembled `H_QQ`, and identity across batch sizes. It uses `Q = 43`
deliberately, because a system with `Q` below the batch size cannot detect the
defect at all, which is why earlier smoke checks missed it.

## Scope of the defect in prior work

`apply_hqq_batch` was reachable before this change only from
`dm_svd_dci/_legacy_pipeline.py`, through `scheme='B'` and `'B_streaming'`,
where it supplies `H_QQ_matvec` for the Krylov propagation. **Any result
produced through that path with `|Q|` larger than the batch size used a
block-diagonal approximation to `H_QQ` without reporting it.** Results from
`scheme='A'`, which assembles `H_emb` densely, are unaffected. Which recorded
results this touches has not been audited.

## Memory, after the fix

| `eps` | `D` | scheme A | streaming | factor | `peak / (M x D)` |
|---|---|---|---|---|---|
| `2e-2` | 97 | `99.3 MiB` | `35.9 MiB` | `2.8x` | `8.45 -> 3.06` |
| `1e-2` | 720 | `363.3 MiB` | `136.4 MiB` | `2.7x` | `4.17 -> 1.56` |
| `5e-3` | 1539 | `775.4 MiB` | `193.2 MiB` | **`4.0x`** | `4.16 -> 1.04` |

The advantage grows with `D`, and at the largest point the peak is essentially
one copy of the `M x D` object rather than four.

The cost is time: `81.6` against `10.1 s` at `D = 720`, and `309.6` against
`26.3 s` at `1539`. Each matvec now re-expands the Q basis twice, and the solver
performs tens of matvecs. `StreamBuilder.prewarm_q_cache` caches those
expansions and is the dial between the two, at the price of holding an `M x Q`
array, which is the object streaming exists to avoid. That trade has not been
measured.

## What this does and does not tell us

It does not tell us which route is wrong. Scheme A extracts `H_QQ` from the
assembled `H_emb`, which has been through extensive `H_AB` and Jordan-Wigner
sign correction (`hku_report/Hemb_RDM_Construction_Summary.md`).
`StreamBuilder.apply_hqq` is far less exercised: before this change it was
reachable only from `dm_svd_dci/_legacy_pipeline.py`, and the state-averaged
path never used it. That makes the streaming side the more likely suspect, but
it is a suspicion, not a finding, and the opposite must be checked too.

This is the class of defect the project has hit repeatedly, and
`.clinerules` is explicit about it: check the code first, theorize second, and
most "method failures" were code bugs. It should be resolved as a focused
investigation with its own diagnostic, not patched inside a feature change.

## Suggested next step

Compare `StreamBuilder.apply_hqq` against the assembled `H_QQ` on a system small
enough for both to be exact, such as H2O/STO-3G, and bisect by Q-block. The
localization above already rules out `H_PP`, `H_PQ` and the diagonal, so the
defect is confined to the off-diagonal contraction, which is a small target.

The route is now consistent with scheme A to `3e-13 Ha`. What remains before it
is production-ready is the time cost above, and an audit of which recorded
results came through the affected legacy path.
