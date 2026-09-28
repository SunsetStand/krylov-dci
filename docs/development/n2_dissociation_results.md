# N2 ground-state dissociation: results

Protocol: `docs/theory/n2_dissociation_protocol.md`, committed in `6f99016`
before any point was computed. N2/cc-pVDZ, full-valence CAS(10e,8o), frozen core
2, `D2h`-adapted orbitals, one state, `lanczos_symm` seed with 10 steps, no
enrichment, at most 20 outer iterations, twelve bond lengths from 0.9 to 3.0 A,
thresholds `1e-3` and `3e-4`, two arms. Every reference energy was checked
against the validated evaluator and agrees to `6e-8 mH`. Summary:
`results/n2_dissociation/n2_dissociation.json`; figure:
`docs/manuscript/figures/fig_n2_dissociation.png`.

## Verdicts

| Prediction | Verdict |
|---|---|
| P1: error at fixed threshold grows with `R` | **Not borne out** in the P = all arm, with low power at large `R` |
| P2: fixed-P error exceeds P = all error, growing with the weight outside P | **Confirmed**, strongly |
| P3: `<S^2>` within `0.1` of zero up to 2.4 A | **Falsified** in the fixed-P arm from 2.2 A; confirmed in the P = all arm at every `R` |

## P = all blocks: within 0.053 mH of the exact singlet everywhere

At `eps = 1e-3` the error runs `0.037, 0.019, 0.014, 0.009, 0.008, 0.009, 0.017,
0.041, 0.053, 0.044, 0.031, 0.025 mH` from 0.9 to 3.0 A. All twelve points
converged with `<S^2>` at most `0.003` and overlap with the exact state at least
`0.99977`. **Non-parallelity error `0.045 mH`** at `1e-3` and `0.053 mH` at
`3e-4`, thirty times inside chemical accuracy.

That is why P1 is not borne out, but the test had little power at large `R`: from
1.8 A the embedded dimension sits at 892 at both thresholds, close to the
single-state Schmidt-rank ceiling that a three-orbital B side imposes, so the
threshold truncates almost nothing there. A larger B side is needed to test P1.

## Fixed P = {8, 9, 10}: fails, independently of the threshold

Where the fixed-P arm stays a singlet, its excess over P = all tracks the weight
the exact state has outside P:

| R / A | weight outside P | fixed P | P = all | excess |
|---|---|---|---|---|
| 0.9 | 0.0004 | 0.0374 | 0.0374 | 0.000 |
| 1.098 | 0.0018 | 0.0252 | 0.0141 | 0.011 |
| 1.2 | 0.0037 | 0.0614 | 0.0094 | 0.052 |
| 1.4 | 0.0127 | 0.8019 | 0.0081 | 0.794 |
| 1.6 | 0.0338 | 3.6896 | 0.0087 | 3.681 |
| 1.8 | 0.0822 | 9.4595 | 0.0170 | 9.443 |
| 2.0 | 0.1692 | 15.6399 | 0.0411 | 15.599 |

The excess crosses the `0.05 mH` falsification line at 1.2 A and reaches
`15.6 mH` at 2.0 A. From 1.6 A the fixed-P errors are **identical at `1e-3` and
`3e-4` to every printed digit**, the same threshold independence as the H2 dimer's
`0.507 mH` (`size_consistency_finding.md`). The downfolding error stays below
`2.8e-10 mH` in both arms throughout, so the whole excess is in the
self-consistent basis.

## From 2.2 A the fixed-P arm converges to a quintet

At 2.2, 2.4, 2.7 and 3.0 A the fixed-P arm converges, flagged converged at
`1e-3`, to a state with `<S^2> = 6.00` and overlap about `1e-5` with the exact
singlet. Its "errors" there, `19.06, 9.91, 3.70, 1.35 mH`, are distances to a
different state and are excluded from every statistic, as the protocol requires.

**Mechanism: root flipping in the P space.** Restricting the Hamiltonian to the
316 determinants with `n_A` in `{8, 9, 10}`:

| R / A | lowest root of full H | lowest root of `H_PP` | singlet within `H_PP` |
|---|---|---|---|
| 1.8 | singlet | singlet | lowest |
| 2.0 | singlet | singlet | lowest, quintet 5.08 mH above |
| **2.1** | singlet | **quintet** | 7.11 mH above |
| 2.2 | singlet | quintet | 15.33 mH above |
| 2.4 | singlet | quintet | 24.63 mH above |

Past about 2.05 A the lowest root of the P-space problem is a quintet although the
singlet stays lowest in the full space. The singlet's stabilization depends on
configurations outside P: at 2.2 A it has `0.214` of its weight in `n_A = 10` and
`0.023` in `n_A = 4`, where the quintet has exactly none, and the quintet lives
almost entirely in `n_A = 8` and `6`. The residual-dressed iteration starts from
the P-space problem and converges the wave operator onto its lowest root, which
is a genuine eigenpair of `H_emb`, so the downfolding is still exact to `1e-10`.

This is the classical root-flipping or intruder problem of effective Hamiltonians
with too small a model space, appearing here as a change of spin. The fixed-P
energy is `-108.72629 Ha` from the **first** outer iteration and does not move;
the blocks the singlet needs (`n_A = 5, 9, 4, 10`, deleted at iterations 4 to 13)
are removed afterwards by the rank contraction because the quintet has no weight
there. Block deletion is the consequence, not the cause.

## Corrections

The protocol quoted the nearest higher-spin state as `3.4 mH` above the singlet at
2.6 A and `0.2 mH` at 3.0 A, and the JSON's `higher_root_gaps_mH` field carries
similar numbers. Both came from an unvalidated four-root Davidson solve and are
not reliable: full diagonalization at 2.2 A puts the nearest state, a triplet, at
`4.75 mH`, whereas the four-root solve reported a quintet at `14.7 mH`. The
reference energies themselves are unaffected, as checked above.

## Consequences

1. A P-block set fixed around the Hartree-Fock electron count **cannot** describe
   bond breaking: it degrades smoothly, then switches to a different spin state.
   The P = all arm is the upper bound for any adaptive rule, and it holds
   `0.053 mH` along the whole curve.
2. An adaptive P rule must at least include every block carrying non-negligible
   weight of the current state, and should reject a converged state whose `<S^2>`
   differs from the target. A spin check is a cheap production safeguard that
   would have caught all four quintet points.
3. None of this changes the cost verdict. Exact CASCI in this space takes about
   0.1 s.
