#!/bin/bash
#SBATCH --job-name=n2_symm_seed
#SBATCH --output=slurm_outputs/n2_symm_seed_%j.out
#SBATCH --error=slurm_outputs/n2_symm_seed_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=06:00:00

# Single-threaded by design: the numerical controls are pinned so that results
# are bit-comparable with the local WSL measurements.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# VERIFY BOTH PATHS BEFORE SUBMITTING; they are inherited from the older
# launchers in this directory and have not been checked on the current host.
cd /data/home/wangcx/krylov-dci
PY=/data/home/wangcx/LiYF4_Er3+/env/bin/python

mkdir -p slurm_outputs
OUT="./results/n2_symm_seed_${SLURM_JOB_ID}"

echo "============================================"
echo "N2 irrep-complete seed scan"
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Commit:  $(git rev-parse --short HEAD)"
echo "Branch:  $(git rev-parse --abbrev-ref HEAD)"
echo "Date:    $(date)"
echo "============================================"

$PY scripts/diagnostics/run_n2_symm_seed_scan.py \
    --points 10@1e-3 14@1e-3 10@5e-4 \
    --output-dir "$OUT"

echo "Done at $(date) with exit code $?"
