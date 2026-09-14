# Seed block support: a self-trapping fixed point of the outer map

First substantive result from the iterative-CI feasibility work. It is a
falsification mechanism for H1, found on the smallest system in the protocol.

## Observation

H2/STO-3G, CAS(2e,2o), Schmidt space A = 1 orbital, 2 target states,
`svd_eps = 1e-10`. At that threshold nothing should be truncated, so the
embedded space is complete and the downfolded energies should be exact for any
seed. They are not:

| Seed | Empty blocks | `D_total` | Rank of block `n=0` | Max error |
|---|---|---|---|---|
| `exact` | none | 6 | `r_A=1, r_B=1` | `6.7e-13 mH` |
| `hf` | none | 6 | `r_A=1, r_B=1` | `6.7e-13 mH` |
| `trunc` | none | 6 | `r_A=1, r_B=1` | `6.7e-13 mH` |
| `selci` | none | 6 | `r_A=1, r_B=1` | `6.7e-13 mH` |
| **`cis`** | **`[0]`** | **5** | **`r_A=0, r_B=0`** | **`20.5 mH`** |
| **`perturbed`** | **`[0]`** | **5** | **`r_A=0, r_B=0`** | **`20.5 mH`** |

The correlation is exact: every seed with an empty block fails, every seed
without one is exact to machine precision.

## Mechanism

The electron-number block `n_A = 0` holds determinants with both electrons in
space B. For this partition that is the doubly excited determinant. The CIS seed
spans excitation ranks `<= 1` only, so it has **identically zero** amplitude
there.

The Schmidt basis is built from the state-averaged reduced densities of the
current coefficients. A block with zero seed weight gives `rho_A^SA(n) = 0`, so
`r_A(n) = r_B(n) = 0` and the block is dropped. The reconstructed coefficients
`C_k,new(n) = U(n) T_k(n) V(n)^dagger` are then built in a basis that no longer
contains that block, so `C^(k+1)` inherits the same empty support, and so does
every later iteration.

**The fixed point is self-trapping.** The outer loop reports convergence
immediately, with a density change of `1.2e-16` at iteration 0, because nothing
is changing. It converges, quickly and confidently, to the wrong answer.

This is the concrete realization of a risk identified in the code audit: in
`dm_svd_dci/state_averaged_solver.py:314` the first state alone determines which
blocks exist, so a seed with different block support silently changes the shape
of the problem.

## Why it matters beyond H2

The specific block here is an artifact of a two-orbital active space, but the
mechanism is not. Any seed built from a bounded excitation rank will have zero
amplitude in the electron-number blocks that require higher excitations, and
those blocks will be deleted permanently. The CIS seed is precisely the seed the
project's own record identifies as the one that fixed excited states, so this is
not a corner case: **the recommended non-exact seed is the one that triggers the
failure.**

It also sharpens H1. The outer map does not merely have a basin-of-attraction
problem; it has fixed points that are unreachable from certain seeds because the
seed removes the dimensions needed to reach them. Reachability has to be stated
in terms of block support, not just in terms of proximity.

## A correction: the block structure was never the problem

The first remedy considered was to take the block structure from the union of all
seed states rather than from `current_states[0]`. **Measurement shows that does
not apply.** The block keys are already complete and identical for every seed,
because `build_block_matrices` allocates every block in the partition:

```text
partition blocks : {0: (1,1), 1: (2,2), 2: (1,1)}
seed = exact     : keys [0,1,2], per-state weight in n=0  1.27e-02 , 6.04e-32
seed = cis       : keys [0,1,2], per-state weight in n=0  0.00e+00 , 0.00e+00
```

Both seeds carry block `0`. It is present, correctly shaped, and empty. There is
nothing to union. The failure is a **zero weight**, not a missing key, so the fix
has to change either what the seed contains or what the solver does with a
vanishing density.

## What was implemented: seed completion

`build_initial_states` gained `partition` and `complete_blocks`. When a seed's
determinant subspace contains no member of some electron-number block, the
lowest-diagonal determinant of that block is added to the subspace before
diagonalizing. The seed keeps its character, gains admissibility, and the solver
is untouched. This is a seeding option, which the protocol's immutable scope
permits; changing the solver would not have been.

Result on H2/STO-3G at `svd_eps = 1e-10`, where a correct run is exact:

| Seed | `complete_blocks=False` | `complete_blocks=True` |
|---|---|---|
| `exact` | `6.7e-13 mH` | `6.7e-13 mH` |
| `hf` | `6.7e-13 mH` | `6.7e-13 mH` |
| `trunc` | `6.7e-13 mH` | `6.7e-13 mH` |
| `selci` | `6.7e-13 mH` | `6.7e-13 mH` |
| `cis` | **`20.5 mH`** | `6.7e-13 mH` |
| `perturbed` | **`20.5 mH`** | `6.7e-13 mH` |

**All six seed families now reach the same fixed point.** That is H1 satisfied on
H2, with the standing caveat that H2 alone is not a sufficient test.

`tests/regression/test_seed_block_support.py` pins both halves: that completion
makes every seed exact, and that disabling completion still reproduces the trap
at `20.5 mH` while reporting convergence. The failing case is pinned deliberately
rather than deleted, because it is a property of the map.

## What is still unresolved

Seed completion guarantees non-zero support **at iteration zero only**. It does
not make block deletion reversible. A block whose weight falls below `svd_eps`
at any later outer iteration is still truncated to rank zero, and the same
irreversibility applies from that point on. At the production threshold of
`1e-3` this is a realistic possibility, not a corner case.

So the map retains a structural property worth reporting in its own right:
**the outer map can lose rank irreversibly, and it reports convergence when it
does.** The remaining candidate remedies are unchanged in kind:

1. **Rank floor.** Retain a minimum rank per block regardless of weight.
   Fabricates arbitrary directions.
2. **Density-matrix perturbation.** Add a decaying noise term before
   diagonalizing, which is exactly what DMRG does to stop quantum-number sectors
   from being discarded irreversibly (White, *Phys. Rev. B* **72**, 180403(R),
   2005). Literature-backed, but adds a parameter.
3. **Residual-informed block re-entry.** Use the Q-space residual, which already
   says where the wavefunction wants weight, to decide whether a deleted block
   should be re-admitted. The most natural fit for this method, since the
   residual is already computed, and the closest analogue of selected-CI
   re-selection. Also the largest change.

None is implemented. The choice needs the full threshold scan behind it, not this
single full-rank data point, and it is a solver change requiring approval.
