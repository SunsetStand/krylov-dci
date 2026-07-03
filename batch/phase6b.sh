#!/bin/bash
#SBATCH -J kdci_phase6b
#SBATCH -p amd
#SBATCH -N 1
#SBATCH --ntasks-per-node=1

export PYTHONUNBUFFERED=1
cd /data/home/wangcx/krylov-dci
/data/home/wangcx/LiYF4_Er3+/env/bin/python scripts/phase6b_krylov_fix.py
