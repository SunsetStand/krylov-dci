#!/bin/bash
#SBATCH --job-name=n2_ms0_roots
#SBATCH --partition=amd
#SBATCH --output=slurm_outputs/n2_ms0_roots_%j.out
#SBATCH --error=slurm_outputs/n2_ms0_roots_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

cd /data/home/wangcx/krylov-dci
PY=/data/home/wangcx/LiYF4_Er3+/env/bin/python
OUTPUT_DIR="${OUTPUT_DIR:-/data/home/wangcx/krylov-dci-run-artifacts/state-averaged/n2_ms0_root_selection_${SLURM_JOB_ID}}"

mkdir -p "${OUTPUT_DIR}"

echo "N2/cc-pVDZ CAS(10,10) lowest-three Ms=0 root-selection diagnostic"
echo "Commit: $(git rev-parse HEAD)"
echo "Output: ${OUTPUT_DIR}"

/usr/bin/time -v "${PY}" \
    scripts/diagnostics/run_n2_lowest_ms0_root_selection.py \
    --output-dir "${OUTPUT_DIR}"

echo "Diagnostic completed: ${OUTPUT_DIR}"
