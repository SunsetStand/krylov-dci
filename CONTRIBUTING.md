# Contributing to Krylov-dCI

## Start from the intended baseline

Do not assume `main` is the active scientific baseline. Confirm the requested
base commit and working-tree state before creating a branch:

```bash
git status --short --branch
git log -5 --oneline --decorate
git switch -c feat/<description> <approved-base>
```

Use `chore/<description>` for repository maintenance and `docs/<description>`
for documentation-only work. Never rewrite shared branch history or force-push.

## Repository responsibilities

- `src/`: early determinant and Slater-Condon implementation.
- `src_mf/`: matrix-free Krylov/dCI backend.
- `dm_svd_embedding/`: Schmidt decomposition and embedded-Hamiltonian code.
- `dm_svd_dci/`: dmSVD-dCI, Neumann, and growing-CAS code.
- `scripts/production/`: maintained entry points that import library code.
- `scripts/diagnostics/`: exploratory investigation tools.
- `scripts/archived/`: retained historical scripts; do not treat them as current APIs.
- `tests/unit/`, `tests/integration/`, `tests/regression/`: checks grouped by scope.
- `batch/production/`, `batch/diagnostics/`, `batch/archived/`: Slurm launchers.

Do not copy backend implementations into scripts. Add or change reusable logic
in the appropriate package, then call it from an entry point.

## Code conventions

- Python 3.9 or newer.
- English comments, docstrings, and identifiers.
- `snake_case` for functions and variables; `PascalCase` for classes.
- Add type hints where they improve clarity.
- Use unbuffered output or `flush=True` for long Slurm jobs.
- Keep mathematical behavior changes separate from repository maintenance.

## Testing

Run Python only in a local environment or on an allocated compute node. The
maintained direct commands are:

```bash
python tests/unit/test_pspace_ops.py
python tests/integration/smoke_sacis.py
python tests/regression/test_kdci.py
python tests/regression/test_hab_rdm.py
```

Backend changes should cover the relevant unit checks, integration smoke test,
and numerical regression checks. H₂/STO-3G alone is not sufficient because its
symmetry can hide integral-index and sign errors; include H₂O/STO-3G or a more
discriminating system when scientific behavior changes.

## PKU server workflow

The configured SSH alias is `tmc-amd`. It resolves to user `wangcx` at
`10.129.77.222` on port `2933` with the project at
`/data/home/wangcx/krylov-dci`.

```bash
ssh -o BatchMode=yes -o ConnectTimeout=15 tmc-amd \
  "cd /data/home/wangcx/krylov-dci && git status --short --branch"
```

If the alias is unavailable, an explicit diagnostic connection must include the
non-default port:

```bash
ssh -p 2933 wangcx@10.129.77.222
```

The login node is restricted to lightweight shell, Git, and file-management
commands. Do not run Python, PySCF, ORCA, CP2K, GROMACS, or quantum-chemistry
calculations directly there.

Use the existing server environment in Slurm jobs:

```bash
#!/bin/bash
#SBATCH -J kdci_check
#SBATCH -p amd
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH -t 00:30:00
#SBATCH -o /data/home/wangcx/krylov-dci/slurm_outputs/%x_%j.out
#SBATCH -e /data/home/wangcx/krylov-dci/slurm_outputs/%x_%j.err

export MODULEPATH=/data/modulefiles/softwares:/data/modulefiles/libraries
source /etc/profile.d/modules.sh
cd /data/home/wangcx/krylov-dci
export PYTHONPATH=/data/home/wangcx/krylov-dci:${PYTHONPATH:-}
PYTHONUNBUFFERED=1 /data/home/wangcx/LiYF4_Er3+/env/bin/python <entry.py>
```

Submit a maintained launcher with, for example:

```bash
ssh tmc-amd \
  "cd /data/home/wangcx/krylov-dci && sbatch batch/diagnostics/reorg_validation.slurm"
```

## Commit and review

Inspect and stage explicit paths. Do not use `git add -A` in a worktree that may
contain large untracked scientific artifacts:

```bash
git diff --check
git status --short
git add README.md CONTRIBUTING.md batch/ scripts/ tests/
git diff --cached --stat
git diff --cached --check
git commit -m "chore(repo): describe the change"
git push -u origin <branch>
```

Before committing, confirm that scratch data, checkpoints, memory maps, `.dat`
files, scheduler logs, and caches are neither staged nor newly tracked. Keep
documentation, test organization, and scientific algorithm changes in separate
commits when practical.

## Documentation

- `docs/theory/formalisms.md`: authoritative mathematical formulation.
- `docs/development/lessons_learned.md`: known implementation pitfalls.
- `docs/architecture/`: design and performance notes.
- `docs/archive/`: historical phase documentation.
- `SKILL.md`: full project conventions and benchmark protocol.
