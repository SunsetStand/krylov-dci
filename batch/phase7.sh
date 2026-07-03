#!/bin/bash
#SBATCH -J kdci_phase7
#SBATCH -p amd
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
cd /data/home/wangcx/krylov-dci
/data/home/wangcx/LiYF4_Er3+/env/bin/python scripts/phase7_dci_ref.py
