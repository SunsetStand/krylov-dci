# Two reduced densities, or one SVD of the CI coefficients?

The current code eigendecomposes `rho_A^SA` and `rho_B^SA` separately. The
original design was a single SVD of the CI coefficient matrix `C_IJ` in the
`|A>|B>` product basis. This settles which is better, in three parts.

## One state: mathematically identical, not a design choice

For a single state, `C = U Sigma V^dag` gives

```text
rho_A = C C^dag = U Sigma^2 U^dag
rho_B = C^dag C = V Sigma^2 V^dag
```

so the eigenvectors of the two densities are exactly the singular vectors of
`C`, and the singular values are the square roots of the eigenvalues. The
single-state branch of `compute_schmidt_decomposition` does in fact call
`svd_truncate_block(C, eps)` directly, and it sets `r_A = r_B = r` by
construction.

## State average: a single SVD of the coefficients does not exist

For several states there is no matrix whose SVD yields both a shared `A` basis
and a shared `B` basis. Writing the weighted states as two different
reshapings of the same data,

```text
M = [ sqrt(w_1) C_1 | ... | sqrt(w_s) C_s ]     (dim_A) x (s dim_B)
N = [ sqrt(w_1) C_1 ; ... ; sqrt(w_s) C_s ]     (s dim_A) x (dim_B)
```

gives `M M^dag = rho_A^SA` and `N^dag N = rho_B^SA`. So the current design
**is already an SVD of the CI coefficients** -- two of them, of two different
arrangements:

- `U` = left singular vectors of `M`;
- `V` = right singular vectors of `N`.

Because `M` and `N` are different matrices, their spectra differ and
`r_A != r_B`. The rectangular rank is therefore **forced by state averaging**,
not chosen. It collapses to the single-state case when `s = 1`, where
`M = N = C`.

## Within that, SVD of M beats eigendecomposition of M M^dag, slightly

Forming `rho = M M^dag` squares the condition number: a singular value `sigma`
becomes `sigma^2`, so the resolvable floor moves from about `1e-16 sigma_max`
to `1e-8 sigma_max`. Worse, `rho` is `dim_A x dim_A` and so carries `dim_A`
eigenvalues, while the true rank is bounded by `min(dim_A, s dim_B)`. Anything
beyond that bound is numerical noise that the square root turns into a
plausible-looking small singular value.

Measured on N2 CAS(10e,9o), four states, comparing retained `r_A` per block:

| `n` | `dim_A` | `dim_B` | true max rank | `1e-2` | `1e-3` | `1e-4` | `1e-6` | `1e-8` |
|---|---|---|---|---|---|---|---|---|
| 3 | 100 | 8 | 32 | 28 | 32 | 32 | 32 | **35** |
| 4 | 200 | 28 | 112 | 92 | 107 | 111 | 112 | **113** |
| 5 | 250 | 56 | 224 | 179 | 214 | 220 | 224 | 224 |
| 7 | 120 | 56 | 120 | 79 | 119 | 120 | 120 | 120 |

via `eig(rho)`. The `svd(M)` route gives **identical ranks at every threshold
from `1e-2` to `1e-6`**, and at `1e-8` it correctly stops at 32 and 112 where
the eigendecomposition reports 35 and 113, exceeding the true maximum rank.
Where both resolve the spectrum they agree to between `1e-16` and `4e-7`
relative.

So the conclusion is narrow and should be stated as such:

- **at every threshold this project actually uses, the two are equivalent** and
  no reported number would change;
- below about `1e-8` the eigendecomposition invents ranks that cannot exist,
  while the SVD cannot;
- the SVD is also cheaper whenever `s dim_B < dim_A`, for instance `100 x 32`
  against `100 x 100` in block 3.

It is a robustness and cost improvement, not an accuracy fix, and it is a
drop-in replacement because it returns the same `U`.

## The design actually being reached for: HOSVD of C[k, I, J]

The question "one decomposition instead of two" does have a real answer, but it
is not a matrix SVD. Treating the multi-state coefficients as a three-index
tensor `C[k, I, J]` and taking a higher-order SVD, or Tucker decomposition,
yields a shared `A` basis, a shared `B` basis **and** a state basis from one
decomposition, with per-mode ranks.

That is what TPSCI does. Abraham and Mayhall define their cluster basis by a
Tucker decomposition of the current sparse CI vector, and Braunscheidel,
Abraham and Mayhall state-average the cluster reduced density matrices into a
single global basis updated by sparse HOSVD
(`docs/literature/iterative_ci_schmidt_downfolding_review.md`). The literature
audit identified it as the closest prior art to this project's central loop.

**No HOSVD or Tucker decomposition exists in this repository.** A search for
`hosvd`, `tucker` and `higher order singular` returns nothing;
`dm_svd_embedding/block_svd_general.py` is a matrix SVD of a reshaped CI vector
for the growing-CAS chain.

So if the intent is genuinely "one decomposition rather than two", the route is
HOSVD, and pursuing it means engaging directly with the method the novelty audit
flagged as the main priority risk.
