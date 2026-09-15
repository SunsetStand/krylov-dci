# Why Omega compresses and H_QP does not

The project's recorded conclusion was that the coupling matrix is dense and not
low rank, so SVD truncation of the Krylov basis buys nothing:
`docs/development/lessons_learned.md` states that at `P <= 4000` on CAS(10,10)
the SVD keeps `d_basis = P`, that is zero compression, and that "no threshold
tuning changes this". That conclusion is correct. It is also compatible with the
wave operator being sharply low rank, and this note explains why the two coexist.

## What Omega is

Projecting the Schrödinger equation onto the outer space gives, for a target
state `k`,

```text
H_QP c_k + H_QQ (Omega c_k) = E_k (Omega c_k)
=>  Omega c_k = (E_k I - H_QQ)^-1 H_QP c_k
```

So `Omega` does contain the coupling: on the target states it *is* the resolvent
applied to `H_QP`. But that relation constrains `Omega` only on the `s` target
directions. The dressing update

```text
delta Omega = [delta q_0 ... delta q_s] pinv([c_0 ... c_s])
```

is the **minimum-norm** operator reproducing those `s` actions, and a
minimum-norm interpolant of `s` conditions has rank at most `s`.

**`Omega` is low rank because it is only ever asked `s` questions, not because
the Hamiltonian is low rank.**

## The measurement

All four objects come from the same N2 CAS(10e,9o) embedded problem at
`eps = 1e-2`, with `P = 479`, `Q = 241`, `s = 4` target states. The table gives
how many singular values are needed to reach each fraction of the squared
Frobenius norm.

| Object | Shape | 90% | 99% | 99.99% | available |
|---|---|---|---|---|---|
| `H_QP`, the raw coupling | `241 x 479` | 49 | 87 | **121** | 241 |
| `A H_QP`, **what the Krylov route SVDs** | `241 x 479` | 42 | 83 | **119** | 241 |
| `A H_QP C`, restricted to the target states | `241 x 4` | 3 | 4 | **4** | 4 |
| `Omega`, the wave operator | `241 x 479` | 3 | 4 | **4** | 241 |

`A H_QP` needs half of its available singular values to be represented at all,
which is the recorded "no useful compression". `Omega` needs four, and it has
the same spectrum as the coupling projected onto the target states.

## The explanation

The two routes compress different objects.

- The **Krylov route** builds `K = SVD(A H_QP)` and then forms an effective
  Hamiltonian `H_PP + H_PK (E - H_KK)^-1 H_KP` whose eigenvectors are all `p`
  states of the model space. To do that, the compressed basis must represent the
  resolvent's action on **every one of the `p` directions of P**. Since the
  columns of `H_QP` are near-orthogonal, that requires nearly full rank. The
  SVD is not failing; it is being asked to compress something incompressible.

- The **wave operator** only has to be correct on the `s` target states. It
  represents the resolvent's action on an `s`-dimensional subspace of P, so rank
  `s` suffices.

The compression does not come from the Hamiltonian. It comes from the
projection onto the target states, and `p >> s` is what makes it large: here
`479` against `4`.

## What this does and does not claim

It does **not** claim that `Omega` being rank `s` is surprising. Constructed as a
minimum-norm interpolant of `s` conditions, it could hardly be otherwise, and a
referee will say so.

The non-trivial content is the comparison. The Krylov formulation solves for all
`p` model-space states when `s` were wanted, and pays full rank for it; the
wave-operator formulation solves for `s` and pays rank `s`. When only a few
states are needed, and `p` is in the hundreds to thousands while `s` is a
handful, the wave-operator formulation is the one whose central object is
compressible. That is a property of the formulation, not of the system.

Two honest qualifications:

- the Krylov effective Hamiltonian delivers all `p` eigenpairs, so it is doing
  strictly more work and answering a strictly larger question. The comparison is
  fair only when `s` states are what is actually wanted, which is the regime
  this project targets;
- the resulting saving is on the `O(q^2)` Q-space object, which the measurement
  in `docs/theory/wave_operator_low_rank_structure.md` shows is **not** the
  current memory bottleneck. The bottleneck is the `M x D` determinant-space
  expansion in the embedded-Hamiltonian build. This note explains a real
  structural advantage; it does not by itself make anything fit in memory.

## Consequence for the earlier conclusion

The recorded lesson should be read more narrowly than it was written. It is not
that this system lacks low-rank structure in the P-to-Q coupling. It is that
**the object being compressed was the wrong one**: a map required on all of P
cannot be compressed, while the same map restricted to the states actually
sought compresses to rank `s`.
