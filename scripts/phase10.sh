#!/bin/bash
#SBATCH -J kdci_phase10
#SBATCH -p amd
#SBATCH -N 1
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH=/data/home/wangcx/LiYF4_Er3+/env/lib:$LD_LIBRARY_PATH
cd /data/home/wangcx/krylov-dci
/data/home/wangcx/LiYF4_Er3+/env/bin/python scripts/phase10_stageA.py
