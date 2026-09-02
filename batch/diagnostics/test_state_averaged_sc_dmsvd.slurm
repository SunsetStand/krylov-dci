#!/bin/bash
#SBATCH --job-name=sa_sc_dmsvd_test
#SBATCH --output=slurm_outputs/sa_sc_dmsvd_test_%j.out
#SBATCH --error=slurm_outputs/sa_sc_dmsvd_test_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00

set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

cd /data/home/wangcx/krylov-dci
PY=/data/home/wangcx/LiYF4_Er3+/env/bin/python

echo "Running pure-matrix state-averaged regression"
"${PY}" tests/unit/test_state_averaged_wave_operator.py

echo "Running H2/STO-3G state-averaged self-consistent smoke test"
"${PY}" scripts/production/run_state_averaged_dci.py \
    --atom 'H 0 0 0; H 0 0 0.74' \
    --basis sto-3g \
    --n-active 2 --n-alpha 1 --n-beta 1 --n-core 0 --n-occ 1 \
    --svd-eps 1e-10 \
    --sa-states 2 --state-weights 0.5,0.5 \
    --p-blocks 1 \
    --scheme A \
    --outer-density-tol 1e-8 --outer-energy-tol 1e-9 \
    --outer-max-iter 12 \
    --wave-damping 0.7 --wave-residual-tol 1e-10 \
    --wave-energy-tol 1e-11 --wave-max-iter 200 \
    --n-workers 2 \
    --output-dir "results/sa_sc_dmsvd_test_${SLURM_JOB_ID}"
