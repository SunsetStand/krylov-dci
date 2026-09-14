# The outer map is a rank contraction

This note answers a direct question: why does the self-consistent iteration have
no effect? The answer is that it has a worse-than-no effect, and the reason is
structural rather than numerical.

## The statement

Let the retained Schmidt basis of electron-number block `n` at outer iteration
`t` be `U_t(n)`, of shape `dim_A x r_A(n)`, and `V_t(n)`, of shape
`dim_B x r_B(n)`. The solver reconstructs the coefficients of each state as

```text
C_k,new(n) = U_t(n) T_k(n) V_t(n)^dagger
```

with `T_k(n)` of shape `r_A(n) x r_B(n)`. Therefore

```text
rank C_k,new(n) <= min( r_A(n), r_B(n) )
range C_k,new(n) subset of  range U_t(n)
```

and the next state-averaged density inherits both bounds:

```text
rho_A^SA(n) = sum_k w_k C_k,new(n) C_k,new(n)^dagger
           => range rho_A^SA(n) subset of range U_t(n)
           => r_A(n) at iteration t+1  <=  r_A(n) at iteration t
```

**The retained rank is monotonically non-increasing, block by block, by
construction.** The reconstructed coefficients live inside the span of the basis
that produced them, so the iteration can shed directions and can never acquire
one. Truncation below the singular-value threshold then makes the loss strict.

## The measurement

Verified on both systems, checking every block at every outer iteration for any
rank increase:

| System | `D` over outer iterations | any rank increase? |
|---|---|---|
| H2O/STO-3G, `eps=1e-3` | `140, 124, 124, 124, ...` | **none** |
| N2/cc-pVDZ CAS(10e,9o), `eps=1e-2` | `720, 296, 229, 216, 216, ...` | **none** |

On N2 the basis loses 70 percent of its dimension, and the ground-state energy
degrades monotonically with it, from `-109.03544405` to `-109.03300491`, which
is `2.44 mH` worse. The iteration converges, confidently, to a worse answer in a
smaller space.

## Why this explains the earlier results

- **H3 was falsified because it had to be.** Self-consistency cannot improve on
  the frozen basis, because every basis it can reach is a subspace of the frozen
  one. The only question was how much it would lose, and on H2O the answer was
  "almost nothing", on N2 "most of it".
- **The irreversible block deletion recorded in
  `seed_block_support_finding.md` is the extreme case** of the same theorem,
  where a rank falls to zero and the block disappears.
- **The wave operator is not implicated.** Its error is `1e-10` or smaller on
  both systems, so the loss is entirely in the basis update.

## What the iteration is actually doing

For the record, the current loop is:

```text
C^(k)  -> rho_A^SA, rho_B^SA per block   (state-averaged, weighted)
       -> eigendecompose, truncate by singular value -> U, V
       -> P/Q partition of the Schmidt product basis -> H_emb
       -> shared residual-dressed Omega, inner iteration to residual 1e-9
       -> generalized Ritz X^dag H X c = E X^dag X c
       -> place P and Q amplitudes back into the Schmidt product ordering
       -> C_k,new(n) = U T_k V^dagger        <-- the contraction happens here
       -> orthonormalize, match roots by overlap, mix, repeat
```

Note also that **there is no energy self-consistency in this architecture, and
there is nothing to add.** The Bloch and Neumann formulations carry an
energy-dependent resolvent `(E I - H_QQ)^-1`, which does have to be iterated to
self-consistency in `E`. The graph formulation replaced it: `X^dag H X c =
E X^dag X c` contains no reference energy at all. The energy enters only the
dressing denominators `R_Q,k / (E_k - diag H_QQ)`, which change the convergence
rate of the inner iteration and not its fixed point. That is confirmed by the
measurement above: the converged result reproduces the exact diagonalization of
`H_emb` to `1e-10` regardless of the energy history.

So the self-consistency that the Heff architecture would have offered was
absorbed when the non-Hermitian energy-dependent Bloch operator was replaced by
the Hermitian energy-independent graph operator. What remains is the basis
update, and that update is a contraction.

## What would have to change

The map needs a mechanism that can **add** rank. Three candidates, in
increasing order of how natural they are here:

1. **Rank floor.** Retain a minimum rank per block. Cheap, but the added
   directions are arbitrary.
2. **Density-matrix perturbation.** Add a decaying noise term before
   diagonalizing. This is exactly why DMRG needs it: White, *Phys. Rev. B*
   **72**, 180403(R) (2005) introduced it to stop quantum-number sectors from
   being discarded irreversibly. Literature-backed, adds a parameter.
3. **Krylov enrichment of the dressed states.** Apply the Hamiltonian once to
   each dressed state before rebuilding the densities, so the enriched set
   leaves the span of the current basis by construction. This costs one sigma
   per state, uses machinery the method already has, and is the natural
   counterpart of the Lanczos seed that made H1 succeed. The Q-space residual
   `R_Q,k` is already computed and already says where weight is missing, so it
   can supply the same enrichment without a second sigma.

Option 3 is the one to test first. Until something of this kind is in place,
"self-consistent Schmidt basis" is not a feature of the method; it is a slow
contraction toward a smaller basis, and the frozen basis is strictly better.
