#!/bin/bash
#SBATCH --job-name=grow_dci
#SBATCH --output=slurm_outputs/grow_dci_%j.out
#SBATCH --error=slurm_outputs/grow_dci_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd /data/home/wangcx/krylov-dci

echo "============================================"
echo "Growing Active Space DCI"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "============================================"
echo ""
echo "Config:"
echo "  A0=6 orbitals (5 occ + 1 virt)"
echo "  B0=1 virt orbital"
echo "  Bt=1 virt orbital per round"
echo "  svd_eps=1e-3, k_max=1"
echo "  n_workers=16"
echo ""

PY=/data/home/wangcx/LiYF4_Er3+/env/bin/python
OUT_DIR="./results/grow_n2_${SLURM_JOB_ID}"

$PY dm_svd_dci/active_space_grower.py \
    --atom 'N 0 0 0; N 0 0 1.098' \
    --basis cc-pVDZ \
    --n-active 10 --n-alpha 5 --n-beta 5 --n-core 2 \
    --n-occ-A 6 --n-orb-B0 1 --n-orb-Bt 1 \
    --p-blocks 8,9,10 \
    --svd-eps 1e-3 \
    --k-max 1 \
    --eps-conv 1e-8 \
    --n-workers 16 \
    --output-dir "$OUT_DIR"

exit_code=$?
echo ""
echo "Done at $(date) with exit code ${exit_code}"
exit ${exit_code}
