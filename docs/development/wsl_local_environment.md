# WSL local environment and reproducibility

Local WSL is the default environment for development, tests, small PySCF
calculations, root diagnostics and short parameter scans. The group cluster is
treated as an archive of historical results and an optional backend for large
jobs only.

## Measured machine

| Item | Value |
|---|---|
| CPU cores | 32 |
| RAM | 7.6 GiB |
| Swap | 2.0 GiB |
| Root filesystem | ext4, 935 GiB free |
| Kernel | WSL2, Linux 6.6.87 |
| Python | 3.12.3 |
| Git | 2.43.0 |

RAM is the binding constraint. Do not rely on swap to force a job through.

## Checkout and virtual environment

The checkout lives on the Linux filesystem, not under `/mnt/c` or `/mnt/d`,
because the Windows mounts are slow for many small files.

```bash
git clone git@github.com:SunsetStand/krylov-dci.git ~/work/krylov-dci
python3 -m venv ~/.venvs/krylov-dci
~/.venvs/krylov-dci/bin/pip install -r requirements.txt pytest ruff h5py matplotlib
```

Pinned versions in use:

| Package | Version |
|---|---|
| Python | 3.12.3 |
| NumPy | 2.5.3 |
| SciPy | 1.18.1 |
| PySCF | 2.14.0 |

Version capture is mandatory, not cosmetic. The N2 root-selection outcome has
been observed to differ between environments: the same diagnostic that the
cluster classified `INITIAL_SUBSPACE_COVERAGE_CONFIRMED` classifies
`UNCONVERGED_REFERENCE` here, because the third root converges on the cluster
and does not converge locally. Any reference bundle therefore records the exact
library versions that produced it.

## Thread pinning

Always export these before a calculation. Mixed threading changes timings and
can change Davidson behavior.

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
```

## Artifacts

Run artifacts are written outside the repository, mirroring the cluster layout:

```text
~/work/krylov-dci-run-artifacts/
  gateB/        diagnostics
  reference/    reference bundles
```

Only small JSON and CSV summaries are committed, under `results/`.

## What fits locally

| Workload | Determinants | Measured cost | Local? |
|---|---|---|---|
| H2/STO-3G | 6 | seconds | yes |
| H2O/STO-3G CAS(6e,5o) | 441 | seconds | yes |
| N2 CAS(10e,9o) root diagnostic | 15876 | 18 s, 132 MiB | yes |
| N2 CAS(10e,10o) root diagnostic | 63504 | 17 s, 158 MiB | yes |
| N2 CAS(10e,9o) reference bundle | 15876 | 4 s, 121 MiB | yes |
| N2 CAS(10e,11o) inventory | 213444 | 26 s, 287 MiB | yes |
| N2 state-averaged three-state pilot | 63504 | requested 96 GB, 16 cores, 24 h | no |

The Slurm launchers request far more memory than several of these jobs need:
the H2O scan launcher asks for 32 GB for a 441-determinant system. Treat the
launcher request as an upper bound, not an estimate.

The state-averaged N2 pilot is the only workload so far that genuinely cannot
run here. Its envelope should be re-measured against CAS(10e,9o), which is four
times smaller than the space the 96 GB figure was based on.
