# Krylov-dCI

Krylov subspace downfolding for configuration interaction, with matrix-free
Krylov/dCI and density-matrix SVD embedding implementations.

## Repository layout

```text
krylov-dci/
├── src/                         # Early determinant / Slater-Condon backend
├── src_mf/                      # Matrix-free Krylov/dCI backend
├── dm_svd_embedding/            # Schmidt decomposition and H_emb construction
├── dm_svd_dci/                  # dmSVD-dCI, Neumann, and growing-CAS workflows
├── scripts/
│   ├── production/              # Maintained run entry points
│   ├── diagnostics/
│   │   ├── hab_rdm/             # H_AB, RDM, and JW-sign investigation tools
│   │   └── kdci/                # Krylov/dCI diagnostic scripts
│   └── archived/                # Historical phase and migration scripts
├── batch/
│   ├── production/              # Production Slurm launchers
│   ├── diagnostics/             # Regression and diagnostic Slurm launchers
│   └── archived/                # Historical phase launchers
├── tests/
│   ├── unit/                    # Focused module checks
│   ├── integration/             # End-to-end and backend integration checks
│   └── regression/              # Stable numerical regression checks
├── docs/
│   ├── theory/                  # Mathematical formulation and proposals
│   ├── architecture/            # Design and performance notes
│   ├── development/             # Development lessons and implementation notes
│   └── archive/                 # Historical phase documentation
├── reports/                     # Research summaries
├── hku_report/                  # HKU-facing progress reports
├── results/                     # Small, reproducible result summaries only
└── notebooks/                   # Exploratory analysis scripts
```

Large scratch data, checkpoints, memory maps, `.dat` files, and scheduler logs
are machine-local artifacts and are not part of the repository.

## Main entry points

- `scripts/production/run_dm_svd_dci.py`: maintained dmSVD-dCI command-line run.
- `scripts/production/step1_iterative_selection.py`: iterative P-space selection.
- `scripts/production/step2_bloch_benchmark.py`: Bloch benchmark using saved P-space data.
- `dm_svd_dci/grow_cas.py`: growing-CAS workflow.
- `dm_svd_dci/active_space_grower.py`: growing-dCI workflow.
- `dm_svd_embedding/scripts/`: embedding prototypes and their Slurm launchers.

Production Slurm launchers live in `batch/production/`. Diagnostic and
regression launchers live in `batch/diagnostics/`.

## Development environment

Install the Python dependencies in a suitable local or compute-node environment:

```bash
python -m pip install -r requirements.txt
```

The PKU server configuration used by the project is:

| Resource | Value |
|---|---|
| SSH alias | `tmc-amd` |
| Host | `10.129.77.222` |
| SSH port | `2933` |
| User | `wangcx` |
| Project | `/data/home/wangcx/krylov-dci` |
| Scheduler | Slurm, partition `amd` |
| Python | `/data/home/wangcx/LiYF4_Er3+/env/bin/python` |

Use the configured SSH alias:

```bash
ssh -o BatchMode=yes tmc-amd
```

The login node is for lightweight shell, Git, and file-management work only.
Run Python, PySCF, and scientific validation through Slurm. For example:

```bash
ssh tmc-amd "cd /data/home/wangcx/krylov-dci && sbatch batch/diagnostics/reorg_validation.slurm"
```

## Tests

Local or allocated compute-node commands are:

```bash
python tests/unit/test_pspace_ops.py
python tests/integration/smoke_sacis.py
python tests/regression/test_kdci.py
python tests/regression/test_hab_rdm.py
```

On the PKU server, submit the corresponding Slurm launchers instead of running
these commands on the login node:

```bash
sbatch batch/diagnostics/run_regression_tests.sh
sbatch batch/diagnostics/reorg_validation.slurm
```

## Architecture rules

- Core matrix-free algorithms live in `src_mf/`; production scripts import
  them rather than copying backend implementations.
- Schmidt and embedded-Hamiltonian logic lives in `dm_svd_embedding/`.
- dmSVD-dCI, Neumann, and growing-CAS logic lives in `dm_svd_dci/`.
- Exploratory scripts stay under `scripts/diagnostics/`; only checks with stable
  assertions and long-term numerical meaning belong in `tests/regression/`.

The authoritative formulation is
[`docs/theory/formalisms.md`](docs/theory/formalisms.md). Development lessons
are collected in
[`docs/development/lessons_learned.md`](docs/development/lessons_learned.md).
See [`SKILL.md`](SKILL.md) for full project conventions and
[`CONTRIBUTING.md`](CONTRIBUTING.md) for the development workflow.

## References

- Li, J.; Yang, J. *JPCL* **2022**, 13, 10042.
- Löwdin, P.-O. *J. Math. Phys.* **1962**, 3, 969.
- Krylov, A. N. *Izvestiya AN SSSR* **1931**, No. 4, 491-539.
- Saad, Y. *Iterative Methods for Sparse Linear Systems*, 2nd ed., SIAM, 2003.
- Sun, Q.; et al. *WIREs Comput. Mol. Sci.* **2018**, 8, e1340.
