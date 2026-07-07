#!/bin/bash
#SBATCH -J m-conv-p1200
#SBATCH -p amd
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --output=/data/home/wangcx/krylov-dci/logs/m_conv_%j.out
#SBATCH --error=/data/home/wangcx/krylov-dci/logs/m_conv_%j.err

set -e; cd /data/home/wangcx/krylov-dci
PYTHON=/data/home/wangcx/LiYF4_Er3+/env/bin/python
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1
mkdir -p logs
$PYTHON scripts_new/test_m_convergence.py
echo "Done at $(date)"
