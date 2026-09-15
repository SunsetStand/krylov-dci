# The wave operator is numerically rank n_states, and what follows

The feasibility scan established that the residual-dressed wave operator
contributes **no accuracy**: it reproduces the exact diagonalization of `H_emb`
to `1e-10` on both test systems, so it is a solver rather than an approximation
(`docs/development/gate_c_feasibility_results.md`). Any claim for it must
therefore be a **cost** claim. This note establishes the structural fact that
such a claim can rest on.

## Measurement: Omega is numerically rank n_states

Singular spectrum of the converged `Omega`, which maps P into Q:

| System | `Omega` shape | singular values for `99.99%` of `\|\|Omega\|\|_F^2` |
|---|---|---|
| H2O/STO-3G, 3 states, `eps=1e-4` | `51 x 97` | **3** of 51 |
| N2/cc-pVDZ CAS(10e,9o), 4 states, `eps=1e-2` | `241 x 479` | **4** of 241 |

The drop after `n_states` is sharp, two to three orders of magnitude. On
synthetic problems of increasing size the pattern is the same:

```text
P=20  Q=120  s=4 :  sv[3]=3.96e-02   sv[4]=5.71e-05    (694x drop)
P=30  Q=300  s=5 :  sv[4]=3.59e-02   sv[5]=1.17e-04    (307x drop)
```

The reason is the construction. Each dressing update is

```text
delta Omega = [delta q_0 ... delta q_s] pinv([c_0 ... c_s])
```

and the row space of `pinv(C)` is the row space of `C`, that is the span of the
`s` model coefficient vectors. Every update therefore has rank at most `s` and
acts in the same `s`-dimensional row space once the `c_k` have settled. The rank
is not *exactly* `s`, because the `c_k` drift between iterations and the
accumulated updates leave a small tail, but the tail is two to three orders of
magnitude down.

## Consequence: the Q-space Hamiltonian never has to be formed

This is an identity, not an approximation. Write `Omega = W Omega_tilde` with `W`
an orthonormal basis of `range(Omega)`, of size `q_dim x k`. Then

```text
X = [I_P ; Omega]
X^dag H X = H_PP + (H_PQ W) Omega_tilde + Omega_tilde^dag (H_PQ W)^dag
                 + Omega_tilde^dag (W^dag H_QQ W) Omega_tilde
X^dag X   = I_P + Omega_tilde^dag Omega_tilde
```

so the entire generalized Ritz problem depends on the Q space only through

```text
H_PQ W          of size  p_dim x k
W^dag H_QQ W    of size  k x k
```

**The `q_dim x q_dim` embedded Hamiltonian is never needed.** Verified against
the full `graph_ritz` at `k = n_states`:

| Problem | Q compression | max energy difference |
|---|---|---|
| `P=20, Q=120, s=4` | `120 -> 4`, `3.3 percent` | `2.5e-12` |
| `P=30, Q=300, s=5` | `300 -> 5`, `1.7 percent` | `4.5e-11` |

*(An earlier version of this check reported errors near `1e-3`. That was a bug
in the probe's generalized-eigenvalue solve, not a property of the method. The
numbers above use the same symmetric metric inverse-square-root as
`graph_ritz`.)*

## Why this matters for this project specifically

The recorded bottleneck is exactly the object that becomes unnecessary.

- `hku_report/Six_Week_Comprehensive_Report.md` reports the embedded-Hamiltonian
  build at **87 to 93 percent** of total runtime, 449 s of 481 s in ground-state
  mode and 4420 s of 5061 s state-averaged.
- The N2 state-averaged pilot launcher requests **96 GB**.
- Growing-CAS at CAS(14,10) was **OOM killed above 503 GB**, and the recorded
  diagnosis was that "the CI expansion step for sigma-vector computation
  requires storing `M x D` matrices", with the conclusion that "matrix-free
  operations are essential for scaling beyond CAS(10,10)".

The low-rank structure says what the matrix-free formulation should be: per
inner iteration, apply `H_QQ` to the `k` current Q-space directions rather than
building `H_QQ`. The cost becomes `k` sigma applications, with `k` of order
`n_states`, instead of an `O(q_dim^2)` build and store.

## Relation to the growing-CAS path

The two directions converge rather than compete. Growing-CAS
(`dm_svd_dci/growing_cas_dmrg.py`) already reached **`0.000 mH` on 7 of 7
configurations at CAS(10,10)**, which is better than anything the
state-averaged line has produced, and it failed at CAS(14,10) for exactly the
memory reason above. It also sidesteps the rank contraction proved in
`docs/theory/outer_map_rank_contraction.md`, because it never rederives a basis
from a truncated wavefunction: each round adds genuinely new orbital degrees of
freedom from the environment.

So the natural combination is growing-CAS for the basis and a matrix-free
low-rank wave operator for the Q space.

## What is established and what is not

Established:

- `Omega` is numerically rank `n_states`, with a two-to-three order drop after
  it, on two real systems and three synthetic ones;
- the graph Ritz problem depends on Q only through `range(Omega)`, which is an
  identity, verified to `1e-11`;
- therefore the `q_dim x q_dim` embedded Hamiltonian is not required by the
  mathematics.

Not established:

- that a matrix-free implementation actually achieves the predicted saving. That
  needs to be built and measured; the argument above is structural, and a
  structural argument is not a benchmark;
- how `k` should be chosen adaptively, and whether the small tail beyond
  `n_states` matters for chemical accuracy on real systems rather than for
  reproducing `graph_ritz`;
- whether the residual-driven enrichment, which needs full-space residuals,
  remains affordable in the matrix-free setting. It costs one sigma per state
  per outer iteration, so it should, but that has not been measured.
