#!/bin/bash
#SBATCH --job-name=sa_n2_cas10
#SBATCH --partition=amd
#SBATCH --output=slurm_outputs/sa_n2_cas10_%j.out
#SBATCH --error=slurm_outputs/sa_n2_cas10_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00

set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

cd /data/home/wangcx/krylov-dci
PY=/data/home/wangcx/LiYF4_Er3+/env/bin/python
OUTPUT_DIR="${OUTPUT_DIR:-/data/home/wangcx/krylov-dci-run-artifacts/state-averaged/n2_cas10_pilot_${SLURM_JOB_ID}}"

mkdir -p "${OUTPUT_DIR}"

echo "State-averaged N2/cc-pVDZ CAS(10,10) three-state pilot"
echo "Commit: $(git rev-parse HEAD)"
echo "Output: ${OUTPUT_DIR}"

/usr/bin/time -v "${PY}" \
    scripts/diagnostics/run_state_averaged_n2_cas10_pilot.py \
    --scheme A \
    --n-workers "${SLURM_CPUS_PER_TASK}" \
    --output-dir "${OUTPUT_DIR}"

echo "Pilot completed: ${OUTPUT_DIR}"
