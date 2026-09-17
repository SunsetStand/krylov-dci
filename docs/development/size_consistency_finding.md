# Size consistency is broken by the P-block choice, and the fix is cheap

Size consistency appeared nowhere in this repository: no test, no document, no
mention. It is a standard referee question for any truncated CI method, and here
the truncation acts on an orbital bipartition, so the answer is not obvious. This
records the first measurement.

## Test

Two H2 at `200 A`, sto-3g, against twice one H2, single state, exact-CI-free
Lanczos seed unless stated, `enrichment_strength = 0` unless stated, the
pre-registered outer controls.

- monomer: `n_active=2`, `(1,1)`, `n_occ=1`, `p_blocks=[1,2]`, error `0.000000 mH`
- dimer: `n_active=4`, `(2,2)`, `n_occ=2`, `p_blocks` swept

The control is the CASCI reference itself, which must be size consistent because
a complete active space over non-interacting fragments factorizes. It is:
`E_ref(dimer) - 2 E_ref(monomer) = 2.4e-11 mH`. The setup is therefore sound and
any non-zero result belongs to the method.

## Result: a threshold-independent error of 0.507 mH

```text
   svd_eps   D_mono   D_dim   err_mono/mH   err_dim/mH   size delta/mH
     1e-02        2       5      0.000000     0.507242        0.507242
     1e-03        2       5      0.000000     0.507242        0.507242
     1e-04        2       5      0.000000     0.507242        0.507242
     1e-06        2       5      0.000000     0.507242        0.507242
```

**Identical to six digits across four decades of threshold.** That rules out
truncation immediately: no amount of tightening removes it.

## It is not the seed and not the rank contraction

| variant | `D` | error | Schmidt part | wave part | size delta |
|---|---|---|---|---|---|
| lanczos, no enrichment | 5 | `0.507242` | `0.507242` | `0` | `0.507242` |
| **exact** seed, no enrichment | 6 | `0.507242` | `0.507242` | `4.4e-13` | `0.507242` |
| lanczos, **enrichment 0.5** | 6 | `0.507242` | `0.507242` | `0` | `0.507242` |
| exact seed, **P = all blocks** | 6 | **`0.000000`** | `0.000000` | `4.4e-13` | **`0.000000`** |
| lanczos, **P = all blocks** | 6 | **`0.000000`** | `0.000000` | `4.4e-13` | **`0.000000`** |

An exact seed gives the same error, so it is not seeding. Enrichment gives the
same error, so it is not the rank contraction of
`outer_map_rank_contraction.md`. Putting every block in P removes it exactly.

## The cause is the P-block coverage, and the mechanism is an information bottleneck

Dimer, exact seed, `svd_eps = 1e-6`, sweeping `p_blocks`:

| `p_blocks` | `D` | `|P|` | `|Q|` | error | size delta |
|---|---|---|---|---|---|
| `[4]` | 6 | **1** | 5 | `0.507242` | `0.507242` |
| `[3, 4]` | 6 | **1** | 5 | `0.507242` | `0.507242` |
| `[2, 3, 4]` | 6 | **5** | 1 | **`0.000000`** | **`0.000000`** |
| `[1, 2, 3, 4]` | 6 | 5 | 1 | `0.000000` | `0.000000` |
| `[0, 1, 2, 3, 4]` | 6 | 6 | 0 | `0.000000` | `0.000000` |

`D` is 6 in every row, so the embedded dimension is not what changes. What
changes is `|P|`: at `p_blocks = [3,4]` block 3 carries zero retained rank, so
`|P| = 1`. Adding block 2 takes `|P|` to 5 and the error to exactly zero.

The mechanism is that the outer map reconstructs `C` from the P-space solution
dressed by `Omega`. A one-dimensional P cannot carry the information needed to
rebuild a correlated two-fragment state, so the next Schmidt basis is degraded
and the loop converges to a different six-dimensional space that does not contain
the exact state. Consistently with that, the wave-operator error stays at
`1e-13 mH`: the downfolding is solving its own problem correctly, and the problem
it is given is the wrong one. This is a bottleneck in `|P|`, distinct from the
rank contraction, and enrichment does not touch it because enrichment enlarges
the Schmidt basis rather than the P-block set.

## Why this matters, and why it is not a contradiction

For a single fragment the electron-number distribution over `n_A` is narrow and a
P-block set of two or three blocks around the Hartree-Fock value suffices. For a
product state over `f` non-interacting fragments that distribution is a
convolution of `f` single-fragment distributions, so it **broadens with the
number of fragments**. A P-block set tuned on a monomer therefore fails on a
dimer, and the failure is threshold independent.

This does not contradict the record that widening P from `n = [8,9,10]` to
`[7,8,9,10]` on the N2 monomer changed the ground state by only `0.004 mH`, so
"the P-block choice should not be widened". Both are true: widening is
unnecessary for a single fragment and necessary for several. The correct
statement is that the P-block set must cover the target state's `n_A`
distribution, and that requirement is system dependent rather than fixed.

**Consequence for the production setting.** N2 CAS(10e,9o) uses
`p_blocks = [8,9,10]` out of `n_A` in `0..10`. That is adequate for the monomer,
as measured, and this test says it would not be for a dimer or for a dissociation
curve at large separation. Any future size-extensivity or potential-curve result
must therefore re-derive the P-block set rather than inherit it.

## Recommended follow-up

1. **Replace the fixed `p_blocks` argument with a coverage criterion**: include
   every block whose retained weight in the current state-averaged density
   exceeds a threshold, so the set adapts to the system instead of being
   inherited. This also removes the `|P| = 1` failure mode automatically.
2. **Add a size-consistency regression test** on this H2 dimer, asserting the
   delta is below `1e-3 mH` with an adaptive P-block set, and asserting the
   CASCI control is size consistent so the test cannot pass vacuously.
3. **Re-derive the P blocks before any dissociation curve**, which is on the
   pre-submission list as the system-generality requirement.

Artifacts: `~/work/krylov-dci-run-artifacts/gateC/size_consistency.json`.
