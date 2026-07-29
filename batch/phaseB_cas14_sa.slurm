#!/bin/bash
#SBATCH --job-name=krylov_B_c14sa
#SBATCH --output=slurm_outputs/phaseB_cas14_sa_%j.out
#SBATCH --error=slurm_outputs/phaseB_cas14_sa_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd /data/home/wangcx/krylov-dci
PY=/data/home/wangcx/LiYF4_Er3+/env/bin/python

echo "============================================"
echo "CAS(14,10) Scheme B — State-Averaged (sa, 5 states)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Date: $(date)"
echo "============================================"

OUT_DIR="./results/phaseB_cas14_sa_${SLURM_JOB_ID}"

$PY scripts/run_dm_svd_dci.py \
    --atom 'N 0 0 0; N 0 0 1.098' \
    --basis cc-pVDZ \
    --n-active 14 --n-alpha 5 --n-beta 5 --n-core 2 --n-occ 5 \
    --svd-eps 1e-3 \
    --mode sa \
    --sa-states 5 \
    --p-blocks 8,9,10 \
    --m-max 1 \
    --scheme B \
    --batch-size 32 \
    --n-workers 16 \
    --output-dir "$OUT_DIR"

exit_code=$?
echo ""
echo "Done at $(date) with exit code ${exit_code}"
exit ${exit_code}