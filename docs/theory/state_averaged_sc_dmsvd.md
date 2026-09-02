# State-Averaged Residual-Dressed Self-Consistent dmSVD

## Scope

This stage implements a common, state-averaged Schmidt basis and one shared
low-rank wave operator for a selected set of roots.  It does not change the
Hamiltonian, H_AB, transition-RDM, or Jordan--Wigner algebra.  Those components
continue to provide the existing P/Q Hamiltonian blocks.

The implementation has two nested fixed-point problems:

1. an inner residual-dressed wave-operator iteration at fixed Schmidt basis;
2. an outer update of the state-averaged reduced densities and Schmidt basis.

## Weighted state average

For normalized target states `k = 0, ..., N_state - 1`, non-negative weights
are normalized so that

```text
sum_k w_k = 1.
```

In every electron-number block `n`, the two reduced densities are

```text
rho_A^SA(n) = sum_k w_k C_k(n) C_k(n)^dagger,
rho_B^SA(n) = sum_k w_k C_k(n)^dagger C_k(n).
```

Equal weights are the default.  The weights are used consistently in the
Schmidt densities, the aggregate residual norm, and the relative root-update
scale.

Unlike a single pure-state Schmidt decomposition, a state average generally
has different left and right ranks.  The retained product block is therefore
rectangular:

```text
dim block(n) = r_A(n) r_B(n),
D_emb        = sum_n r_A(n) r_B(n).
```

Forcing `r_A = r_B = min(r_A, r_B)` discards legitimate state-averaged support
and is not used by the new solver.

## Shared graph wave operator

At fixed Schmidt basis, partition the embedded Hamiltonian as

```text
H = [ H_PP  H_PQ ]
    [ H_QP  H_QQ ].
```

One shared operator `Omega: P -> Q` defines the graph subspace

```text
X = [ I_P ]
    [Omega].
```

The target roots are obtained variationally from the Hermitian generalized
eigenproblem

```text
X^dagger H X c_k = E_k X^dagger X c_k.
```

Thus the dressed state is

```text
Psi_k = [ c_k          ]
        [ Omega c_k    ],
```

and the returned dressed states are mutually orthonormal in the full P+Q
metric.  This avoids diagonalizing a non-Hermitian truncated Bloch Hamiltonian.

## Residual dressing

For each root, the Q-space Schrödinger residual is

```text
R_Q,k = H_QP c_k + H_QQ Omega c_k - E_k Omega c_k.
```

It is preconditioned with the diagonal Q resolvent,

```text
delta q_k = R_Q,k / (E_k - diag(H_QQ)),
```

with a configurable denominator floor.  The block of state corrections is
fitted back to one minimum-norm operator correction,

```text
delta Omega = [delta q_0 ... delta q_s] pinv([c_0 ... c_s]),
Omega <- Omega + damping * delta Omega.
```

The inner loop converges only when both the weighted full residual norm and
the maximum target-energy change meet their thresholds.

## Schmidt self-consistency

After the inner solve, P/Q amplitudes are placed back into the complete
Schmidt-product ordering.  For each number block,

```text
C_k,new(n) = U(n) T_k(n) V(n)^dagger,
```

where `T_k(n)` has shape `r_A(n) x r_B(n)`.  Roots are matched to the preceding
outer iteration by physical CI overlap and their arbitrary phases are fixed.
Optional linear damping is followed by symmetric orthonormalization.

The outer loop stops when all three conditions hold:

1. the inner wave-operator solve converged;
2. the weighted reduced-density distance is below `density_tol`;
3. the maximum tracked-root energy change is below `energy_tol`.

## Entry points

- Python: `dm_svd_dci.pipeline_state_averaged.run_state_averaged_dci`
- CLI: `scripts/production/run_state_averaged_dci.py`
- Pure-matrix regression: `tests/unit/test_state_averaged_wave_operator.py`
- Slurm smoke test: `batch/diagnostics/test_state_averaged_sc_dmsvd.slurm`

The JSON result records every outer iteration, inner convergence status,
weighted residual, rectangular Schmidt ranks, P/Q dimensions, root weights,
and comparison with the initializing multi-root CASCI calculation.
