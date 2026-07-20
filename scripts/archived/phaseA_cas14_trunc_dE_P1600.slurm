#!/bin/bash
#SBATCH -J trunc-dE-P1600
#SBATCH -p amd
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --output=/data/home/wangcx/krylov-dci/logs/trunc_dE_P1600_%j.out
#SBATCH --error=/data/home/wangcx/krylov-dci/logs/trunc_dE_P1600_%j.err

set -e; cd /data/home/wangcx/krylov-dci
PYTHON=/data/home/wangcx/LiYF4_Er3+/env/bin/python
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1
mkdir -p logs
$PYTHON scripts_new/phaseA_cas14_trunc_dE.py --target-p 1600 --m-max 3 --tag truncP1600
echo "Done at $(date)"
