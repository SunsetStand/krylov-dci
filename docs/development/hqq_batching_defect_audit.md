# Audit: which recorded results the H_QQ batching defect touched

The defect is described in
`docs/development/streaming_hqq_batching_defect.md`:
`StreamBuilder.apply_hqq_batch` restricted the sum over `j` in
`result[i,k] = sum_j <CI_i|H|CI_j> B[j,k]` to the current Q batch, leaving only
the block-diagonal part of `H_QQ`. It is wrong exactly when
`|Q| > q_batch_size`, whose default is 32.

**Conclusion: no committed result is affected.** The reasoning and the residual
uncertainty are both recorded below.

## Window

| Event | Commit | Date |
|---|---|---|
| Introduced, with `StreamBuilder` | `7b14af7` | 2026-07-27 |
| Fixed | `ea707b0` | 2026-09-16 |

## Reachability

Only one production path reaches the defective function:

```text
_legacy_pipeline.run_dm_svd_dci(scheme='B' | 'B_streaming')
  -> StreamBuilder.build_all()
  -> blocks['H_QQ_matvec'], blocks['H_QQ_batch']
  -> _legacy_effective_ham.build_projected_hqq_matvec
       -> run_effective_ham_at_m / run_effective_ham_per_state
  -> _legacy_krylov_propagator.build_krylov_full_matvec
```

So only the **Krylov-dCI effective-Hamiltonian and Krylov-propagation results
obtained through `_legacy_pipeline` with scheme B** could be wrong.

**Not affected, each verified by reading the call path:**

- every `scheme='A'` path, which assembles `H_emb` densely and slices it;
- `pipeline_v2.py` with scheme B, which calls `extract_q_blocks_scheme_b` in
  `qspace_partition.py` and never touches `streaming_ops`;
- `pipeline_state_averaged.py` with scheme B, same reason. All of the
  state-averaged work in this branch is therefore clean;
- `dm_svd_dci/schmidt_partition.py`, which contains a **second, unrelated**
  function also named `apply_hqq_batch`. It loops over columns calling
  `apply_hqq_on_the_fly`, which expands the vector, applies sigma once and
  projects back onto the whole Q basis. It never batches over Q, so it is
  correct by construction, and it is in fact the reference implementation of
  the form the fix adopted. Note that
  `hku_report/Six_Week_Comprehensive_Report.md` attributes "Scheme B" to
  `schmidt_partition.py`, which is this correct implementation;
- `grow_cas.py`, `pipeline_state_averaged.py` and the new regression test,
  which import only `build_hemb_parallel` from `_legacy_pipeline`.

## Committed results

No committed result records `scheme=B`. Scanning every JSON under `results/`
for a `scheme` field anywhere in the structure returns `A` for
`grow_cas_15519` and `grow_cas_15520` and nothing for the rest.

`batch/production/dm_svd_dci.slurm`, the production launcher, passes no
`--scheme` argument, so it used the default, which is `'A'` in
`scripts/production/run_dm_svd_dci.py`.

The only scheme-B invocation anywhere in the repository is
`batch/diagnostics/consistency_test_A_vs_B.sh`, which runs N2 CAS(10,10),
`svd_eps=1e-3`, `p_blocks=8,9,10`, `--batch-size 32`, writing
`results/consistency_AB_<job>_A` and `..._B`.

**Only the `_A` outputs are in the repository**: `consistency_AB_15397_A`,
`consistency_AB_15421_A`, `consistency_AB_15422_A`. A search of the full git
history for any `consistency_AB_*_B/` path returns nothing, so the B outputs
were never committed, across all three jobs.

## An inference, stated as one

That test's scheme-B run was certainly affected: at N2 CAS(10,10) with
`svd_eps=1e-3` the embedded dimension is of order `4600`, so `|Q|` is far above
the `batch-size 32` the script passes. Its pass criterion is strict, requiring
`E_eff_m0`, `E_eff_m1` and the corresponding `dE` values to agree with scheme A
within `1e-6`, and `m=0` within `1e-4 mH`. With only the block-diagonal part of
`H_QQ` reaching the Krylov propagation, the test would have failed.

Three job IDs ran it, and none of their B outputs was committed while all three
A outputs were. That pattern is consistent with the test failing at the time and
the cause never being found. **It is consistent with, not proof of.** The
repository cannot distinguish a failing run whose output was discarded from a
run that crashed before writing, or from a deliberate decision not to commit
the B side.

If so, the test that should have caught this defect probably did catch it, and
the finding was lost rather than pursued.

## The reports do not quote affected numbers

The scheme-B material in `hku_report/dm_svd_dci_comprehensive_report.md`
sections 3.2 and 3.7, and in `Six_Week_Comprehensive_Report.md`, is
architectural and projected rather than measured: "a matrix-free approach is
required", "the Krylov propagation would then be fully matrix-free", "enables
calculations with `D > 100,000`". No measured scheme-B energy appears in any
report.

## A separate finding that blocks re-verification

`scripts/production/run_dm_svd_dci.py`, the only entry point to the affected
path, is **currently broken**. It does `from dm_svd_dci.pipeline import
run_dm_svd_dci`, but that module was renamed to `_legacy_pipeline` in the
2026-09-02 repository reorganization, so the import raises
`ModuleNotFoundError`. Both of its callers,
`batch/production/dm_svd_dci.slurm` and
`batch/diagnostics/consistency_test_A_vs_B.sh`, are broken with it.

This means the affected path cannot currently be run at all, which bounds the
damage, and also means the consistency test cannot be re-run to confirm the fix
until the import is repaired. That repair is one line and is not made here,
because it is outside the audit.

## Recommended follow-up

1. Repair the import in `scripts/production/run_dm_svd_dci.py` and re-run
   `consistency_test_A_vs_B.sh`. With the fix in place scheme A and scheme B
   should now agree within the test's own `1e-6` criterion. That both closes
   the audit and validates the fix on the path that originally carried it.
2. If scheme B is to be retired rather than repaired, retire it only after that
   run, so the retirement rests on a passing comparison rather than on an
   unexplained absence.
