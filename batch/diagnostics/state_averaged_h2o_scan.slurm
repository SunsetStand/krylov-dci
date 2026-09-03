#!/bin/bash
#SBATCH --job-name=sa_h2o_scan
#SBATCH --output=slurm_outputs/sa_h2o_scan_%j.out
#SBATCH --error=slurm_outputs/sa_h2o_scan_%j.err
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
OUTPUT_DIR="${OUTPUT_DIR:-/data/home/wangcx/krylov-dci-run-artifacts/state-averaged/h2o_scan_${SLURM_JOB_ID}}"

mkdir -p "${OUTPUT_DIR}"

echo "State-averaged H2O truncation scan"
echo "Commit: $(git rev-parse HEAD)"
echo "Output: ${OUTPUT_DIR}"

"${PY}" scripts/diagnostics/run_state_averaged_h2o_scan.py \
    --scheme A \
    --n-workers "${SLURM_CPUS_PER_TASK}" \
    --output-dir "${OUTPUT_DIR}"

echo "Scan completed: ${OUTPUT_DIR}"
