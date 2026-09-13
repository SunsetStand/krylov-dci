# Local reproduction of cluster job 16990

Job `16990` ran `scripts/diagnostics/run_n2_lowest_ms0_root_selection.py` on the
group cluster and reported classification `INITIAL_SUBSPACE_COVERAGE_CONFIRMED`.
This is the local reproduction of that run, unchanged, at commit `e218187`.

Command:

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
python scripts/diagnostics/run_n2_lowest_ms0_root_selection.py \
    --output-dir ~/work/krylov-dci-run-artifacts/gateB/repro_16990
```

Cost: `16.6 s` wall, `157604 KiB` peak RSS, against `18.59 s` and `151076 KiB`
on the cluster.

## What reproduced and what did not

| Quantity | Job 16990 | Local | Agrees |
|---|---|---|---|
| Full determinant dimension | `63504` | `63504` | yes |
| HF to singles coupling norm | `2.298e-07` | `2.298e-07` | yes |
| HF to singles maximum coupling | `9.309e-08` | `9.522e-08` | same order |
| Inventory-seeded mapping | `[0,1,2]` | `[0,1,2]` | yes |
| Expanded-default-seeded mapping | `[0,1,4]` | `[0,1,4]` | yes |
| CIS-seeded mapping | `[0,1,4]` | `[0,1,4]` | yes |
| **Default three-root mapping** | `[0,1,4]` | `[0,1,3]` | **no** |
| **Classification** | `INITIAL_SUBSPACE_COVERAGE_CONFIRMED` | `UNCONVERGED_REFERENCE` | **no** |

Six of eight quantities reproduce exactly. The two that differ are precisely
the two that depend on the Davidson trajectory of the default three-root solve.

Locally that solve returns `converged = [True, True, False]`: its third root has
residual `1.9e-03` and sits `+2.94 mH` above inventory root 2, with overlap
`0.9921` on inventory root 3. On the cluster the same solve reported every root
converged, which is why the cluster reached the `INITIAL_SUBSPACE_COVERAGE`
branch while the local run short-circuits to `UNCONVERGED_REFERENCE`.

Enlarging the Davidson subspace locally reproduces the cluster's answer: at
`max_space=24, max_cycle=200, conv_tol=1e-10`, and again at `40 / 400 / 1e-12`,
all three roots report converged while the third sits `+30.014 mH` high, which
is exactly inventory root 4, giving mapping `[0,1,4]`.

**The default three-root result is not a stable quantity.** It depends on the
numerical environment. No conclusion may rest on it.

## Why the cluster classification must not be used

`INITIAL_SUBSPACE_COVERAGE_CONFIRMED` is the final `else` branch of the
classifier. It is emitted only after every positive test has already failed, so
it tests no condition of its own and asserts a mechanism on no evidence. Under
the revised taxonomy in
`docs/theory/n2_root_targeting_mechanism_protocol.md` the correct label for the
evidence available at the time was `ROOT_TARGETING_MECHANISM_UNRESOLVED`.

## What the raw JSON does establish

Read from the report rather than the printed summary, two quantities are solid
and environment-independent.

First, the default three-vector guess has **identically zero** projection on
four inventory roots:

| Inventory root | Span weight, 3-vector guess | Span weight, 8-vector guess |
|---|---|---|
| 0 | `9.369e-01` | `9.369e-01` |
| 1 | `6.459e-02` | `4.803e-01` |
| 2 | `1.898e-21` | `4.613e-01` |
| 3 | `6.748e-21` | `4.620e-01` |
| 4 | `4.496e-01` | `4.849e-01` |
| 5 | `3.488e-02` | `4.852e-01` |
| 6 | `2.765e-17` | `4.503e-01` |
| 7 | `1.171e-19` | `4.507e-01` |

Roots 2, 3, 6 and 7 are exactly the four `E1g` states. A projection that starts
at zero stays zero under Krylov expansion, so the default guess cannot reach
them at any tolerance or subspace size. This is the symmetry-coverage
mechanism, measured.

Second, the CIS intervention was mis-specified rather than ineffective. CIS
roots 4 and 5 have overlap `0.960` and `0.961` with inventory roots 2 and 3, so
the singles space does carry the missing character. The script seeds with the
three **lowest** CIS roots, which carry the character of inventory roots 0, 1
and 4. Hypothesis A was therefore never properly tested by this script.

## Consequence

The legacy diagnostic cannot settle the mechanism, and its classification is
unusable. It is superseded by
`scripts/diagnostics/run_n2_root_targeting_mechanism.py`, which computes
residuals independently of the solver's convergence flag, labels every root by
spatial irreducible representation, and runs symmetry-resolved and
spin-resolved procedures.
