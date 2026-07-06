#!/bin/bash
#SBATCH -J step12_pspace
#SBATCH -p amd
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH -o /data/home/wangcx/krylov-dci/logs/step12_pspace_%j.out
#SBATCH -e /data/home/wangcx/krylov-dci/logs/step12_pspace_%j.err

# Two-step P-space convergence experiment
# Step 1: Iterative selection (cheap: ~5-10 min)
# Step 2: Bloch H^eff batch evaluation (more expensive: H_QP builds)

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1

cd /data/home/wangcx/krylov-dci
mkdir -p logs checkpoints_pspace

PYTHON=/data/home/wangcx/LiYF4_Er3+/env/bin/python

echo "=== Step 1: Iterative P-space Selection ==="
echo "Started at $(date)"
$PYTHON scripts/step1_iterative_selection.py
RC1=$?
echo "Step 1 exit code: $RC1 at $(date)"

if [ $RC1 -ne 0 ]; then
    echo "Step 1 FAILED. Aborting."
    exit $RC1
fi

echo ""
echo "=== Step 2: Bloch H^eff Batch Evaluation ==="
echo "Started at $(date)"
$PYTHON scripts/step2_bloch_benchmark.py
RC2=$?
echo "Step 2 exit code: $RC2 at $(date)"

exit $RC2
