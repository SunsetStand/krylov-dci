# The method costs more than the exact solve it is meant to avoid

Measured on N2 CAS(10e,9o)/cc-pVDZ, `15876` determinants, four states covering
the three lowest levels, after the method reached chemical accuracy on all four
states (`docs/development/n2_threshold_curves.md`). One system; the scaling
argument in the last section is what makes it more than a single data point.

## Accuracy is established, and it is not the question here

Cluster job 20910, `seed = 'lanczos_symm'`, 10 steps, `symmetry = 'D2h'`,
`eps = 5e-4`: every state inside `1.6 mH`, the worst being S1 `B1u` at
`0.997 mH`, weighted `0.524 mH`, converged, degenerate splitting `0.042 mH`,
from a seed that reads no exact CI. That result stands.

## Full-space sigma applications

Both the seed and the exact solver apply `H` to vectors in the same 15876-dim CAS
space, so a sigma application is a fair common unit.

| | full-space sigma calls |
|---|---|
| seed `lanczos_symm`, 3 steps | 119 |
| seed, 6 steps | 221 |
| **seed, 10 steps**, as used for the accuracy result | **357** |
| seed, 14 steps | 493 |
| seed, 20 steps | 697 |
| **exact Davidson, `nroots = 6`**, the Gate B validated margin | **145** |
| exact Davidson, `nroots = 4`, which Gate B showed misses a root | 80 |

Counted by wrapping `_ActiveSpace.sigma` and PySCF's `contract_2e`, the latter
counting each vector of a block, at `conv_tol = 1e-10`.

**The 10-step seed alone applies `H` in the full space 2.5 times as often as
solving the four states exactly**, before the downfolding runs at all. Three steps
is cheaper than Davidson, but at three steps the `B1u` state is crowded out of
the seed by an `Au` state (`ca2cecb`), so it is not a working configuration. The
cheapest seed that covers every target is therefore already about as expensive as
the exact answer.

## Memory and wall time

| | memory | wall, one thread |
|---|---|---|
| exact CASCI, six roots | tens of vectors of length `M`, of order tens of MB | **0.7 s**, local |
| method at `eps = 5e-4`, `D = 9887` | **9.1 GB** peak | **5778 s**, cluster |

The cluster measured 1.36 to 1.4 times slower than the local machine on the
overlap points, so the method is of order `4000 s` locally: **roughly 5000 to
8000 times slower than the exact solve, and four orders of magnitude less
accurate**, `0.524 mH` against `5e-8 mH`.

The memory follows from the embedded-Hamiltonian build, whose peak is a constant
`4.2 x M x D`; at `D = 9887` that is `D / M = 62 percent` of the determinant
count, so the embedded basis is no longer small relative to the space it was
supposed to compress.

## Why this is structural, not an artifact of one small system

Every stage of the method as implemented holds **dense vectors over the full CAS
space**:

- the seed is a Krylov chain of full-space vectors;
- the Schmidt decomposition acts on `C_IJ`, which is a full CAS CI vector
  reshaped by the `A|B` bipartition;
- the embedded Hamiltonian is built by expanding Schmidt product states back into
  full-space vectors, which is the `4.2 x M x D` term.

Davidson needs only tens of such vectors. Any system small enough for this method
to hold its vectors is small enough for Davidson, and Davidson is cheaper there.
So the method, **in its current formulation, cannot reach the regime where the
exact solve is impossible**, which is the only regime in which it could be
useful. Larger active spaces do not help this argument: they raise `M` for both
methods alike, and the method pays `D` times more storage.

This is the answer, measured, to the question raised earlier in the project of
whether the method would turn into Davidson diagonalization. It is worse: the
seed needed to reach every target irrep is itself a Krylov solve in the same full
space, more expensive than Davidson.

## What survives

- **The self-consistent map converges to the correct multi-state fixed point**
  from a non-exact seed, including an exactly degenerate pair. That is a valid
  proof of principle, not a performance result.
- **The analysis content**, independent of cost: the rank contraction theorem and
  its annealed repair; the proof that `||R_k||` cannot vanish in a truncated
  space; the exact error-budget separation showing the dressed wave operator
  contributes zero error; the irrep-coverage condition on seeds, which is general
  for any method that rebuilds a basis from its own wavefunction under a rank
  contraction; and the three silent defects found on the way, the unvalidated
  reference, the symmetry-restricted solver and the non-variational Lanczos seed.

## What would be needed for a cost advantage

A formulation that **never forms dense full-space vectors**. Two directions,
neither tested:

1. **Determinant-space effective Hamiltonian with a compressed Q**, as proposed
   in `docs/theory/hpq_svd_qspace_compression_analysis.md`: a selected P space,
   and Q compressed to `rank(H_PQ) <= |P|` directions obtained from the columns
   of `H_QP`, each of which is a *sparse* vector generated from one P determinant.
   Its cost is set by the sparsity of `H_QP` and the P size, not by `M`. This is
   structurally the regime of Li and Yang's dCI and of selected CI, and its
   viability is an open question, not a result.
2. **Factorized rather than dense Schmidt vectors**, keeping the current
   bipartition but representing the Schmidt states compactly. That converges
   toward DMRG, which the Gate A review identified as the principal prior-art
   collision, so it would need a clearly different angle.

Either is a new method rather than a tuning of the present one.
