# CLAUDE.md

Pointer file for coding agents. It does not restate the project rules; it says
where they live and records the few conventions that are easy to get wrong.

## Authority order

1. `SKILL.md` -- full project conventions. Authoritative.
2. `docs/theory/formalisms.md` -- authoritative mathematical formulation. When
   code comments, memory and this file disagree, `formalisms.md` wins.
3. `docs/theory/state_averaged_sc_dmsvd.md` -- the state-averaged wave operator,
   residual dressing and Schmidt self-consistency. `formalisms.md` only
   cross-references it.
4. `CONTRIBUTING.md` -- branching, review and server workflow.
5. `.clinerules` -- diagnosis protocol and known pitfalls.

## Rules that are easy to violate

- **Branch before any backend change.** Method interfaces, numerical algorithms
  and data structures require a branch created before the first edit. Never
  force-push `main` or a feature baseline.
- **Never reimplement `src_mf/` logic inside a script.** Import it. Scripts that
  copy backend code create version drift.
- **Ask first** before changing the mathematical formulation, choosing a
  different algorithm than agreed, or modifying code that already passed tests.
- **Commit before `sbatch`.** Code, then Git, then push, then submit.
- **Never edit code on a remote host** with `sed` or a heredoc. Edit locally and
  copy.
- **Stage explicit paths.** Do not `git add -A` in a worktree holding large
  scientific artifacts.
- English for all comments, docstrings, identifiers, documents and commit
  messages.
- `flush=True` on prints, or `PYTHONUNBUFFERED=1`; stdout is block-buffered when
  redirected into a scheduler log.

## Where things go

| Kind | Location |
|---|---|
| Matrix-free backend | `src_mf/` |
| Schmidt and embedded Hamiltonian | `dm_svd_embedding/` |
| dmSVD-dCI, Neumann, growing-CAS | `dm_svd_dci/` |
| Maintained entry points | `scripts/production/` |
| Exploratory tools | `scripts/diagnostics/` |
| Historical, not a current API | `scripts/archived/`, `batch/archived/` |
| Theory and protocols | `docs/theory/`, protocols named `*_protocol.md` |
| Small reproducible summaries | `results/` |

Checkpoints, `.dat`, memmaps, `.npz`, `.h5` and scheduler logs are machine-local
and must not be committed. They are already in `.gitignore`.

## Execution environment

Local WSL is the default development, test and validation environment. See
`docs/development/wsl_local_environment.md` for the exact setup, and for which
workloads fit locally. Do not connect to the group cluster or submit Slurm jobs
without being asked to. Pin the four numerical-thread variables to `1`.

## Testing

`tests/unit/`, `tests/integration/`, `tests/regression/` are executable entry
points, not an importable package. H2/STO-3G alone is not sufficient: its
symmetry hides integral-index and sign errors. Use H2O/STO-3G or larger whenever
scientific behavior changes.

Known pre-existing local failures, unrelated to new work:

- `tests/regression/test_kdci.py` -- 5 of 7 pass; the sparse-sigma import and
  the frozen-core CAS check are the two legacy failures already declared out of
  scope in `docs/theory/state_averaged_validation_protocol.md`.
- `tests/integration/smoke_sacis.py` -- cannot pass anywhere; it targets a
  driver that the repository reorganization moved into `scripts/archived/`, and
  that driver hard-codes its own `PROJECT_ROOT` under `/data`.

## Numerical lessons that keep recurring

Read `docs/development/lessons_learned.md` before debugging a "method failure".
Most such failures were code bugs. In particular: check the code first, theorize
second; a solver convergence flag is not evidence of correctness; and blind
index-based root selection fails for triplet states.
