# Rectangular state-averaged Schmidt ranks: first measurement

The literature survey identified rectangular independent left and right Schmidt
ranks per electron-number block as the one ingredient of this method for which no
precedent was located. Two independent searches reached that conclusion. Before
any claim is built on it, the feature has to be shown to exist, to be caused by
what the method says causes it, and to be measurable at the thresholds the method
actually runs at. This note reports that measurement.

Reproduce with:

```bash
python scripts/diagnostics/run_rectangular_schmidt_rank_scan.py \
    --bundle-dir <reference bundle> --output-dir <artifacts> --n-occ 5
```

System: N2, cc-pVDZ, CAS(10e,9o), `(5a,5b)`, frozen core 2, the four target roots
of the locked reference bundle `ad5b19d6...`, Schmidt space A being the first five
active spatial orbitals. Cost: `0.49 s`, `107 MiB`.

## Result 1: the asymmetry is caused by state averaging

A pure bipartite state has identical `rho_A` and `rho_B` non-zero spectra, so its
left and right ranks must agree. Treating each of the four roots individually as a
one-state average:

| Roots tested | Blocks with `r_A != r_B` | Largest rank difference |
|---|---|---|
| 4 | **0 of 9, every root** | `0` |

The `n = 6` spectra agree to `3.2e-17`, machine precision. Taking the four-state
average instead, at the same threshold, five of nine blocks become asymmetric and
the `n = 6` spectra differ by `1.5e-2`.

This rules out the obvious alternative explanation. The blocks have very unequal
dimensions, `dim_A` up to `250` against `dim_B` of `56`, so one might suspect the
asymmetry is a dimensional artifact. It is not: at identical dimensions and
identical threshold, a pure state gives `r_A == r_B` exactly and a state average
does not. The cause is that a weighted average of pure states is a mixed state,
for which the Schmidt equality does not hold.

## Result 2: where the effect is genuine, and where it is saturation

A block with `r_B == dim_B` is not evidence of rectangularity; that side simply was
not truncated. Counting only blocks where **both** sides are genuinely truncated:

| Threshold | Genuine and asymmetric | Genuine and symmetric | Saturated | `D_rect` | `D_sym(min)` | `D_sym(max)` |
|---|---|---|---|---|---|---|
| `3e-2` | 4 | 0 | 1 | `139` | `69` | `303` |
| `1e-2` | 4 | 0 | 1 | `574` | `426` | `839` |
| `3e-3` | 2 | 0 | 3 | `3917` | `2530` | `6165` |
| `1e-3` | 3 | 0 | 3 | `7883` | `4597` | `13610` |
| `3e-4` | 3 | 0 | 4 | `16973` | `9513` | `30526` |
| `1e-4` | 0 | 1 | 6 | `25633` | `12345` | `54114` |
| `3e-5` | 0 | 0 | 8 | `31365` | `12869` | `78370` |

Two things follow.

**The effect has an operating window**, roughly `3e-2` to `3e-4`. Below it every
block saturates and the apparent asymmetry is only one side being untruncated. The
project's standing default of `1e-3`, fixed in
`docs/theory/state_averaged_validation_protocol.md`, sits inside that window,
which is fortunate but was not by design and should now be stated as deliberate.

**Where the effect is genuine it is never symmetric.** The "genuine and symmetric"
column is zero at every threshold but one. Whenever both sides are actually
truncated, the retained ranks differ. That is a stronger statement than
"rectangularity sometimes occurs", and it is the form the claim should take.

## Result 3: the equal-cost comparison is well posed

`D_rect` lies strictly between the two symmetric allocations at every threshold,
so rectangular retention is a genuine intermediate allocation rather than simply a
larger space. At `1e-3` the rectangular embedded dimension is `7883`, against
`4597` for `min(r_A, r_B)` and `13610` for `max(r_A, r_B)`.

This settles how the claim must be tested. It is not enough to show that the
rectangular basis is more accurate than the `min` allocation, since it is also
larger. The required experiment is a **matched-dimension** comparison: hold the
embedded dimension fixed near `7883` and compare the rectangular allocation at
`eps = 1e-3` against a symmetric allocation whose threshold is loosened until it
reaches the same total dimension. If the rectangular allocation does not win at
matched cost, the claim should be dropped.

## What this does and does not establish

Established: the feature exists, it is caused by state averaging and not by
dimensional imbalance, it has a definite operating window that contains the
project's default threshold, and within that window it is universal rather than
occasional.

Not established: that it buys accuracy. No energy has been computed here. The
matched-dimension experiment above is the next step and belongs in the
feasibility protocol as a primary hypothesis, not an ablation.

One caveat for positioning. Symmetry-adapted DMRG keeps independent numbers of
renormalized states per quantum-number sector, and active-space decomposition
DMRG renormalizes asymmetrically across an inter-fragment cut. A referee may
argue those already retain independent left and right ranks implicitly. The
distinction this project can defend is that those choices are made per sector for
bookkeeping, whereas here the asymmetry is a *derived consequence* of averaging
over a state ensemble, is quantified against a pure-state control, and is used
deliberately. That argument needs the matched-cost result to stand on.
