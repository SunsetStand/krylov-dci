#!/bin/bash
#SBATCH -J krylov-pspace
#SBATCH -p amd
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=06:00:00
#SBATCH --output=/data/home/wangcx/krylov-dci/logs/krylov_bench_%j.out
#SBATCH --error=/data/home/wangcx/krylov-dci/logs/krylov_bench_%j.err

set -e
cd /data/home/wangcx/krylov-dci
PYTHON=/data/home/wangcx/LiYF4_Er3+/env/bin/python
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1
mkdir -p logs

echo "===== Krylov + Per-State Bloch Benchmark ====="
echo "Started at $(date)"
$PYTHON scripts_new/step3_bloch_krylov.py
echo "Done at $(date)"
