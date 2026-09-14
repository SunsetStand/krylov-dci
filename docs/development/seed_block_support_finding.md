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

## What has been done

Instrumentation only, per the protocol's immutable-scope rule that a numerical
problem is first recorded as data. `dm_svd_dci/initializers.py` gains
`seed_block_support`, and the pipeline records the per-block seed weight, the
list of empty blocks and a warning, in `seed_provenance.block_support` of every
result. No solver behaviour was changed.

## Candidate remedies, none yet implemented

1. **Reject the seed.** Treat an empty block as a specification error and refuse
   to run. Safe and honest, but it disqualifies the CIS seed outright.
2. **Rank floor.** Retain a minimum rank per block regardless of weight, so a
   block can never be deleted. Cheap, but it fabricates support that the seed
   does not have and the added directions are arbitrary.
3. **Union block support.** Take the block structure from the union of all seed
   states, or from the determinant space itself, rather than from
   `current_states[0]`. This preserves the shape of the problem independently of
   the seed, and looks like the principled fix.
4. **Enrich the seed.** Require the seed family to span every block, for instance
   CIS plus the block-completing determinants of lowest diagonal energy. Keeps
   the CIS character while removing the trap.

Remedy 3 addresses the cause and 4 addresses the trigger; they are not exclusive.
A choice between them is a solver change and needs the full scan behind it, not
this single data point.
