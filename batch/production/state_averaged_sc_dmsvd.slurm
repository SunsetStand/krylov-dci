#!/bin/bash
#SBATCH --job-name=sa_sc_dmsvd
#SBATCH --output=slurm_outputs/sa_sc_dmsvd_%j.out
#SBATCH --error=slurm_outputs/sa_sc_dmsvd_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

ATOM="${ATOM:-N 0 0 0; N 0 0 1.098}"
BASIS="${BASIS:-cc-pVDZ}"
N_ACTIVE="${N_ACTIVE:-10}"
N_ALPHA="${N_ALPHA:-5}"
N_BETA="${N_BETA:-5}"
N_CORE="${N_CORE:-2}"
N_OCC="${N_OCC:-5}"
SVD_EPS="${SVD_EPS:-1e-3}"
SA_STATES="${SA_STATES:-3}"
STATE_WEIGHTS="${STATE_WEIGHTS:-}"
P_BLOCKS="${P_BLOCKS:-8,9,10}"
SCHEME="${SCHEME:-A}"
OUTER_MIXING="${OUTER_MIXING:-0.7}"
N_WORKERS="${N_WORKERS:-${SLURM_CPUS_PER_TASK}}"
OUTPUT_DIR="${OUTPUT_DIR:-results/sa_sc_dmsvd_${SLURM_JOB_ID}}"

cd /data/home/wangcx/krylov-dci
PY=/data/home/wangcx/LiYF4_Er3+/env/bin/python

WEIGHT_ARGS=()
if [[ -n "${STATE_WEIGHTS}" ]]; then
    WEIGHT_ARGS=(--state-weights "${STATE_WEIGHTS}")
fi

"${PY}" scripts/production/run_state_averaged_dci.py \
    --atom "${ATOM}" --basis "${BASIS}" \
    --n-active "${N_ACTIVE}" --n-alpha "${N_ALPHA}" --n-beta "${N_BETA}" \
    --n-core "${N_CORE}" --n-occ "${N_OCC}" \
    --svd-eps "${SVD_EPS}" --sa-states "${SA_STATES}" \
    "${WEIGHT_ARGS[@]}" \
    --p-blocks "${P_BLOCKS}" --scheme "${SCHEME}" \
    --outer-mixing "${OUTER_MIXING}" \
    --n-workers "${N_WORKERS}" --output-dir "${OUTPUT_DIR}"
