# Determinant-space code inventory for the compressed-Q effective Hamiltonian

Gate 0 of the determinant-space line (`research/detspace-heff`, base `6d9cf70`).
It records what exists, whether it runs, and what it means for the proposal in
`docs/theory/hpq_svd_qspace_compression_analysis.md`. It contains no new method.
Every path below was executed on 2026-09-28, locally, PySCF 2.14.0, NumPy 2.5.3,
one thread, on N2 CAS(10e,9o)/cc-pVDZ with D2h orbitals (`M = 15876`), by
`scripts/diagnostics/run_detspace_gate0_checks.py` (commands at the end).

## Bottom line

1. The proposal's primitives exist in `src_mf/sparse_ops.py` and are
   **numerically correct**: sparse `H_QP` columns, `span(W)` by Gram SVD or MGS,
   sparse `H v`, projected `H_KK` and `H_PK`, all within `1e-13` of dense
   references. No test or maintained script had ever called them.
2. They are pure Python, about `3.8 us` per connected determinant. One sparse
   `H v` costs `0.5 to 1.3 s`; one full-space PySCF sigma costs `6.8 ms`.
3. Everything else, the paths behind every reported determinant-space result
   included, applies `H` to dense vectors of length `M`. "Sparse" in
   `KDCISparse` means sparse storage only.
4. On this system the sparsity is used up early. The union of `H_QP` column
   supports covers **14 %** of Q at `p = 40` and **50 %** at `p = 300`.
5. Three silent defects were found (D1 to D3 below). D1 changes how the Phase-1
   determinant-space record should be read.

## Modules

| Module | What it does | Ran? | Maps to the proposal |
|---|---|---|---|
| `src_mf/sparse_ops.py` | Python Slater-Condon via `src.hamiltonian.Hamiltonian`: `generate_connected_determinants`, `build_hqp_sparse`, `sparse_mgs`, `gram_svd`, `sigma_sparse`, `project_hqq`, `project_hpq` | yes, all agree with dense to `<1e-13` | the whole m = 0 proposal, and the `H v` needed for m >= 1 |
| `src_mf/sparse_vector.py` | `SparseQVector`: a `{(alpha, beta): coef}` dict | self-test passes | storage for sparse Q vectors |
| `src_mf/kdci_sparse.py` | `KDCISparse`: streaming MGS of `A_q H_QP`, projection | yes, correct | m = 0 with sparse storage, but **one dense `O(M)` sigma per column and per basis vector** |
| `src_mf/kdci_dense.py`, `qspace.py` | `QSpaceIndex` (all CAS strings, `hdiag` of length M) and `KDCIBackend`: dense `(M, p)` `H_QP`, SVD basis, dense propagation | yes, via the checks | the dense reference; `O(M)` per vector |
| `src_mf/pyscf_backend.py` | verbatim copy of the three files above plus three self-tests | test 1 passes, test 2 fails (D3), test 3 not reached | duplicate; do not extend |
| `src_mf/pspace_ops.py` | vectorized state-averaged CIPSI scoring, `H_PP` by sigma columns | `test_pspace_ops`, `test_hpp_sigma` pass | P selection and `H_PP`; all dense over M |
| `src_mf/bloch_mf.py` | "matrix-free" m = 0 Bloch correction | runs, **returns zero** (D2) | m = 0 diagonal-resolvent correction; unusable as is |
| `src/kdci_pipeline.py` | the Phase-1 determinant pipeline: seed, P expansion, Krylov, per-state `H_eff` | runs, 35 s, 0.56 GB, but see D1 | the old determinant line end to end |
| `src/effective_h.py` | `H_PP + H_PK (E - H_KK)^-1 H_KP` by dense inverse | self-test passes | the `H_eff` assembly |
| `src/krylov.py`, `svd_compression.py`, `partitioning.py`, `determinants.py`, `hamiltonian.py` | dense pure-Python Krylov-dCI for tiny systems | self-tests pass (`hamiltonian` only with `-m`) | historical; the four scripts `tests/integration/test_h2_exact`, `test_h2_convergence`, `test_h2o_correct`, `test_svd_scan` crash on NumPy 2 formatting a size-1 array, in a print before any assertion |
| `src/sparse_sigma.py`, `src/cas_hamiltonian.py` | older sparse `H_O' v`; frozen-core CAS build | self-tests: pass; fail by 28.68 Ha | the two known `test_kdci.py` failures (5 of 7 pass, unchanged) |
| `dm_svd_dci/_legacy_krylov_propagator.py` | `B0 = MGS(A_q H_QP)`, then `A_q (H_QQ - D) B` | toy self-tests pass | m = 0 and its extension, dense `(q, r)` arrays |
| `dm_svd_dci/neumann_qspace.py`, `neumann_effective_ham.py` | Q from S/D excitations into growing-CAS env orbitals, dense `O(q^2)` `H_QQ` blocks, Neumann k = 0, 1 | toy tests pass; `neumann_qspace` has no test and was not run | Neumann truncation, which this line does not use; its single-excitation loops also emit spin-flipped (wrong `M_s`) determinants |

## The three questions

**Are the `H_QP` columns from `build_hqp_sparse` stored sparsely?** Yes. It
returns one `SparseQVector` per P determinant holding `A_q H_QP[:, p]`. P
determinants are excluded. Passing `A_diag_func = 1` gives the bare column. Of
the 560 connected determinants generated per column, only 53 to 89 are nonzero
at `1e-14`; D2h symmetry removes the rest. The cost is 2 ms per column.

**Is `gram_svd` already "compress Q by the SVD of `H_PQ`"?** Yes, when it is
given the bare columns. It diagonalizes the `p x p` Gram matrix `H_PQ H_QP` and
returns the Q-side singular vectors multiplied by their singular values, `W S`,
as sparse vectors. The singular values match a dense SVD to `2e-15`, and the
span matches `span(H_QP)` to `4e-14`. Four caveats:

- The returned vectors are not normalized.
- The cutoff `1e-12` applies to `sigma^2`, which makes it an absolute
  `sigma > 1e-6`, not the relative singular-value cutoff the docstring states.
- Squaring through the Gram matrix loses singular values below about
  `1e-8 sigma_max`.
- Its `weights` argument scales P columns, not Q rows. So the `A_q`
  preconditioning has to come from `build_hqp_sparse`, and `gram_svd` then
  returns the SVD of `A_q H_QP`, which is the legacy m = 0 space.

`rank(H_QP) = p` exactly at `p = 40` and `p = 300`. The smallest singular value
is `0.13` to `0.18`, so the span cannot be compressed any further, which agrees
with the Six-Week report.

**How did the old lines choose P?**

- *Determinant line* (`src/kdci_pipeline.py`, Six-Week Phase 1). It starts from
  HF, all CIS singles, and the top HFPT2 doubles up to `P_init = 200`, tied
  scores included. It then adds 200 determinants per iteration by the
  state-averaged score `w(q) = sum_k |(H c_k)_q|^2 / max(|E_k - H_qq|, 1e-8)`
  over the five lowest `H_PP` roots, whatever their spin or irrep, with `H c_k`
  a dense full-space sigma. The singles are there because HFPT2 gives them zero
  weight (Brillouin); that is what repaired the triplet states.
- *Schmidt line*. P is not a set of determinants. It is the retained Schmidt
  product blocks of the `A|B` bipartition, fixed by `svd_eps`.

## Defects found in this pass

- **D1. The Phase-1 determinant pipeline has no frozen-core potential.**
  `kdci_pipeline.setup_system` takes the active `h1` from the bare `hcore`,
  not from `CASCI.get_h1eff()`. The largest `h1` difference is `3.28 Ha`, and
  the lowest active energy is `-60.384` against `-31.623 Ha`; `h1eff` plus
  `ecore` reproduces the bundle, `-109.0401239716`. The same construction is in
  every Phase-1 driver found by grep (`phaseA_cas10_v10_sacis.py`,
  `v10_tracked.py`, `phase18_mf.py`, `trunc_m1_cis*.py`, `trunc_p2000.py`,
  `sacis_trunc*.py`). Their FCI references use the same `h1`, so their errors
  are internally consistent, but they belong to a different Hamiltonian: N2
  with the core potential switched off. I have not traced which driver produced
  each reported number. Until that is done, the Phase-1 figures, `P ~ 800` for
  chemical accuracy and the CIS seed giving S1 at `+0.8 mH`, should not be
  carried over to CASCI.
- **D2. `bloch_mf.bloch_correction_batched` is wrong twice.** It weights by a
  single factor `sqrt|A_q|` instead of `A_q`, and it zeroes every `A_q < 0`.
  `A_q < 0` holds on all of Q whenever `E0` lies below every `H_qq`, which it
  does for the ground state, so the correction is exactly zero (measured: 0
  against a correct maximum of 0.12 Ha). Nothing reported used it;
  `step2_bloch_benchmark.py` uses the dense path.
- **D3. The `pyscf_backend` self-test is wrong, not the backend.**
  `test_build_hqp_vs_hamiltonian` compares P rows that `build_hqp` zeroes by
  design, which accounts for its `6.08` mismatch. On Q rows the backend and
  Slater-Condon agree to `9e-16`.
- **Conventions that will bite.**
  - The `ham` from `pipeline_v2.setup_system` has `E_nuc = ecore`, so its
    diagonal is `77.42 Ha` away from `q_idx.hdiag`.
  - `sigma_sparse` returns `H v` with P rows included, not `H_QQ v`, and it
    includes the diagonal only when `diag_func` is passed.
  - `kdci_pipeline` weights by `sqrt|A_q|`, not `A_q`, and truncates at
    `1e-3 sigma_max` in both the build and the propagation. Its per-state
    energies are one-shot `H_eff(E_k)` at the `H_PP` root, not
    self-consistent, and therefore not variational. That is consistent with
    the below-exact energies in Six-Week section 1.4.
  - The legacy propagation `A_q (H_QQ - D) B` is a Jacobi-preconditioned
    Krylov sequence, not block Lanczos on `H_QQ`. The `2m+1`-moment property
    cited in the analysis is a property of `span{W, H_QQ W, ...}` and is not
    guaranteed for this sequence.
  - All Gram-Schmidt here is single pass. It measured orthonormal to `5e-15`
    at `r = 300`, but nothing checks it.

## Sparsity and cost, measured

P is the `p` lowest-diagonal determinants.

| `p` | nnz per `H_QP` column | union of supports / Q | MGS basis nnz, mean (max) | `W S` nnz, mean (max) | `H v` support growth |
|---|---|---|---|---|---|
| 40 | 65 to 84, mean 73.5 | 2222 / 15836 = 14 % | 229 (482) | 623 (1078) | x7.3 (240 to 1752) |
| 300 | 53 to 89, mean 65.4 | 7792 / 15576 = 50 % | 639 (1038) | 1796 (5234) | x3.9 (608 to 2386) |

Orthonormalizing destroys the per-column sparsity: the fill-in is about 10x for
MGS and 27x for SVD. At this `M` the pure-Python sparse `H v` loses to one
full-space PySCF sigma once the vector has more than about 3 nonzeros. Exact
Davidson (145 sigmas, 0.7 to 1.0 s) is therefore out of reach of the existing
sparse tooling on this system.

## What the proposal needs and nothing provides

1. A **compiled `H v` for a sparse vector of arbitrary support** that also
   generates the connected space. PySCF `selected_ci` (`contract_2e`,
   `enlarge_space`, `select_strs`) is C-level, but it works on products of α
   and β string sets, whose support can be much larger than the determinant
   support.
2. **Block Lanczos on `H_QQ` from `W`** on sparse vectors, with the P part
   projected out, two-pass orthogonalization, and a `K^T K = I` check. It does
   not exist in any form, dense or sparse.
3. **P selection without dense sigmas.** `score_and_select` needs `H c_k` over
   all M.
4. **A sigma-call counter** for sparse operations, in full-space-equivalent
   units, so that H2 can be measured.

## Open questions for Gate 1 (decisions for the user)

- **Kernel for sparse `H v`.** The options are PySCF `selected_ci` on string
  products, a new compiled determinant-list kernel, or pure Python as it
  stands. The last is correct but cannot pass H2.
- **Basis form.** An orthonormal `W` (fill-in) or the raw sparse columns with
  their Gram metric (a generalized eigenproblem).
- **Energy evaluation.** Diagonalizing in `P + K`, which is variational and
  equals self-consistent `H_eff`, or one-shot `H_eff(E0)`.
- **Compression per state or shared.** `A_q(E_k)` per state, or the bare,
  shared `W`.
- **A system series for H2.** On this `M = 15876`, a P large enough for
  accuracy covers most of Q, so the sub-`M` scaling can only be tested across
  several `M`.

## Reproduce

From the worktree root:

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=~/.venvs/krylov-dci/bin/python
$PY scripts/diagnostics/run_detspace_gate0_checks.py --part paths --n-p 40    # 5.8 s, 0.15 GB
$PY scripts/diagnostics/run_detspace_gate0_checks.py --part paths --n-p 300   # 36 s, 0.49 GB
$PY scripts/diagnostics/run_detspace_gate0_checks.py --part legacy            # 35 s, 0.56 GB
```

The module self-tests were run as `$PY <file>`, or `$PY -m src.hamiltonian`
and `$PY -m src_mf.pyscf_backend`. Logs are in
`~/work/krylov-dci-run-artifacts/detspace/gate0/`.
