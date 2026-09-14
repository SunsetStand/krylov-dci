# Gate C feasibility scan: results and verdicts

Scan of `docs/theory/iterative_ci_feasibility_protocol.md` on H2O/STO-3G
CAS(6e,5o), three states, `(3,3)`, frozen core 2, Schmidt space A = 3 orbitals,
P blocks `n_A = 4,5,6`. Thresholds and pass conditions were fixed before any
result was seen.

Reproduce with:

```bash
python scripts/diagnostics/run_iterative_ci_feasibility_scan.py \
    --system h2o --output-dir <artifacts>
```

Cost for the whole scan: about `25 s`, `120 MiB`.

## The finding that reframes the rest

**The wave-operator error is exactly zero in every cell of the scan.** Splitting
the total error into its two sources,

```text
Schmidt error     = E_embedded   - E_reference
wave-operator err = E_downfolded - E_embedded
```

where `E_embedded` is the exact diagonalization of the final embedded
Hamiltonian, gives `0.0000 mH` for the second term at every threshold, every
seed, both rank modes and both wave-operator modes.

At convergence the residual-dressed wave operator **reproduces the exact
diagonalization of `H_emb`**. It is a solver, not an approximation. It follows
that the accuracy of the whole method is determined entirely by the dmSVD
Schmidt truncation, and that the wave operator's contribution has to be argued
on **cost** -- it avoids forming and diagonalizing `H_emb` -- and not on
accuracy. Every verdict below is a consequence of this.

## H1 and H2, reachability and stability: CONFIRMED

Weighted absolute error against CASCI, in mH:

| Seed | `eps=1e-2` | `eps=1e-3` | `eps=1e-4` | `D` at `1e-4` |
|---|---|---|---|---|
| `lanczos` k=2 | 1.2520 | 0.0273 | 0.0240 | 148 |
| `lanczos` k=3 | 1.2470 | 0.0034 | 0.0012 | 148 |
| `lanczos` k=4 | 1.1010 | 0.0026 | 0.0004 | 148 |
| `lanczos` k=6 | 1.0997 | 0.0022 | **0.0000** | 148 |
| `exact` (control) | 1.0996 | 0.0022 | 0.0000 | 148 |
| `hf` | 11.41 | 10.99 | 10.99 | 122 |
| `selci` | 10.93 | 10.25 | 10.25 | 106 |
| `cis` | 36.94 | 36.91 | 36.91 | 8 |

A seed that reads no exact CI and selects no determinants reaches the
exact-seeded fixed point, monotonically in one parameter, at the same embedded
dimension. By `k=6` it is identical to the exact seed at every threshold.

Every determinant-selection seed fails by one to four orders of magnitude, for
the structural reason recorded in
`docs/development/seed_block_support_finding.md`.

## H3, whether self-consistency buys anything: FALSIFIED

Compared at matched embedded dimension, which is the only fair comparison:

| Threshold | self-consistent | frozen | verdict |
|---|---|---|---|
| `1e-2` | `D=75`, 1.2470 mH | `D=76`, **1.0047 mH** | frozen better by 0.24 mH, above `tol_E` |
| `1e-3` | `D=124`, 0.0034 mH | `D=108`, 0.0148 mH | indistinguishable, but the dimension match failed at 13 percent against a 2 percent tolerance |
| `1e-4` | `D=148`, 0.0012 mH | `D=148`, 0.0012 mH | identical |

Self-consistency never helps. At the loosest threshold it is measurably worse.
The protocol's falsification condition for H3 is met.

This is the experiment the novelty audit independently identified as the one
that decides whether there is a method. On this system the answer is that the
Schmidt-to-`Omega`-to-Schmidt feedback is **decoration**: the Lanczos-seeded
basis is already as good as the converged one.

Two honest caveats. The dimension match failed at `1e-3`, because the embedded
dimension moves in discrete jumps, so that row is weak evidence either way. And
H2O is one system; the conclusion must be retested on N2 before it is reported
as general.

## H4, whether residual dressing carries information: CONFIRMED

| Threshold | dressed | undressed | ratio |
|---|---|---|---|
| `1e-2` | 1.2470 mH | 12.18 mH | 9.8x |
| `1e-3` | 0.0034 mH | 11.62 mH | 3400x |
| `1e-4` | 0.0012 mH | 11.62 mH | 9700x |

Without dressing the generalized Ritz problem reduces to diagonalizing `H_PP`,
so this confirms that the Q space is reached only through the dressing. Given
the zero wave-operator error above, the correct reading is that the dressing is
**how the embedded problem gets solved at all**, not that it improves an
otherwise adequate answer.

## H6, rectangular versus symmetric ranks: NOT TESTABLE ON THIS SYSTEM

At matched embedded dimension the two rank modes are identical at every
threshold, and the matched symmetric threshold comes out equal to the
rectangular one. That is because **H2O has no rank asymmetry**: its blocks are
too small for `r_A(n)` and `r_B(n)` to differ.

H6 therefore cannot be scored here. It must be run on N2 CAS(10e,9o), where the
asymmetry was measured to be genuine and universal within the threshold window
`3e-2` to `3e-4` (`docs/development/rectangular_schmidt_rank_measurement.md`).

## H7, shared versus per-state wave operator: INDISTINGUISHABLE

Identical to every printed digit at every threshold. Given that both modes drive
the wave-operator error to zero, this is the expected result: they are two
solvers for the same embedded problem, and both solve it exactly.

**A shared wave operator is sufficient.** That is a publishable simplification,
and it reverses the earlier expectation that per-state centring would be needed.
The earlier evidence for per-state centring concerned truncated Krylov bases in
the Krylov-dCI line, where the basis itself was state-dependent; here the
embedded space is common to all states by construction.

## Consequences for the paper

1. The wave-operator half contributes **no accuracy**. Any claim for it must be
   a cost claim, benchmarked against forming and diagonalizing `H_emb` directly.
2. The accuracy claim rests entirely on the **dmSVD Schmidt truncation**, which
   is also the half the literature survey found to be closest to prior art in
   DMRG and TPSCI.
3. The self-consistency claim is, on this system, falsified. It must be retested
   on N2 before anything is written about it either way.
4. H6 is the surviving novelty candidate and it has not yet been tested at all,
   because the system chosen for the scan cannot express it.

## Next

Run the same scan on N2 CAS(10e,9o) using the locked reference bundle. That is
the only remaining way to test H6, and it is the retest H3 needs. The N2
state-averaged pilot resource envelope should be re-measured at the same time,
since CAS(10e,9o) is four times smaller than the space the 96 GB figure came
from.
