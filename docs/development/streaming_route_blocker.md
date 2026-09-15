# Streaming H_emb route: memory win confirmed, blocked by an H_QQ discrepancy

The streaming route is wired into the state-averaged pipeline as
`scheme='streaming'`. It uses `dm_svd_dci/streaming_ops.py`'s `StreamBuilder`,
which batches the determinant-space expansion and exposes `H_QQ` only as a
matvec, feeding the matrix-free low-rank wave-operator solver.

**It is not usable yet.** The two Hamiltonian construction routes disagree.

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

Until that is closed, `scheme='streaming'` should be treated as experimental and
must not be used for any reported number.
