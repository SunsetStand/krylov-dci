# Is the inner wave-operator loop just a Davidson diagonalization?

Largely yes, and the paper should say so rather than be caught out on it.

## The structural correspondence

| Step | Davidson | Inner wave-operator iteration |
|---|---|---|
| Subspace eigenproblem | `V^dag H V c = E V^dag V c` | `X^dag H X c = E X^dag X c`, with `X = [I_P ; Omega]` |
| Residual | `r = H x - E x` | `R_Q`, the Q component of `H x - E x` |
| Preconditioner | `t = r / (E - diag H)` | `delta q = R_Q / (E - diag H_QQ)` |
| Update | orthogonalize `t`, **append** to `V`, subspace grows | `delta Omega = delta q pinv(c)`, subspace **rotates** at fixed dimension |

Both are Rayleigh-Ritz plus a diagonally preconditioned residual correction. The
one structural difference is that Davidson grows its subspace while the graph
formulation keeps a subspace of fixed dimension `p` that always contains `P`
exactly, and moves it by changing `Omega`.

## The measurement

Block Davidson with a diagonal preconditioner, run on the assembled embedded
matrix, against the low-rank wave-operator solver on the same problem:

| `P` | `Q` | `s` | Davidson iters | Davidson `H v` | Davidson subspace | Omega iters | `H_QQ v` | Omega rank | energy difference |
|---|---|---|---|---|---|---|---|---|---|
| 20 | 120 | 4 | 12 | 96 | 48 | 33 | 127 | 16 | `1.2e-14` |
| 30 | 300 | 5 | 13 | 130 | 65 | 36 | 166 | 21 | `1.5e-14` |
| 40 | 800 | 4 | 17 | 136 | 68 | 34 | 131 | 17 | `2.2e-14` |

They return the same eigenvalues to `1e-14`. Matrix-vector counts are
comparable, slightly favouring Davidson. Davidson needs roughly a third of the
iterations; the graph formulation carries roughly a third of the subspace
dimension.

## The honest conclusion

**The inner loop is a Davidson-class eigensolver and the novelty cannot live
there.** That is consistent with everything else measured: the wave-operator
contribution to the energy error is zero to `1e-10` on both test systems
(`docs/development/gate_c_feasibility_results.md`), so it is a solver, and a
plain Davidson is at least as good a solver.

Two properties of the graph formulation survive the comparison, and both are
about cost rather than accuracy:

- the subspace does not grow, so there is no restart policy and no growing
  storage. Davidson's subspace here reached 48 to 68 vectors in the full `p + q`
  space; the graph form carries a `q x rank` basis with rank 16 to 21;
- the Q space enters only through `range(Omega)`, which is what makes the
  matrix-free form possible
  (`docs/theory/wave_operator_low_rank_structure.md`).

## Where the method is not Davidson

The parts that are not a diagonalization at all:

- the **dmSVD Schmidt construction**, which builds the embedded space from
  state-averaged reduced densities over an `A|B` bipartition, with independent
  left and right ranks per electron-number block. This is where all of the
  accuracy comes from, since the solver contributes none;
- the **outer self-consistency**, which rebuilds that basis from the solution
  and which, once made rank-increasing, is worth `2.5` to `2.8 mH` at matched
  embedded dimension (`docs/development/gate_c_n2_results.md`).

So the defensible description of the method is: **a Davidson-class solver
operating inside a dmSVD-compressed embedded space, with a self-consistent basis
update.** The contribution is the compression and the outer loop. Presenting the
inner iteration as novel would invite exactly the objection this note answers.
