#!/bin/bash
#SBATCH -J kdci_phase5b
#SBATCH -p amd
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH -o /data/home/wangcx/krylov-dci/logs/phase5b_%j.out
#SBATCH -e /data/home/wangcx/krylov-dci/logs/phase5b_%j.err

export MODULEPATH=/data/modulefiles/softwares:/data/modulefiles/libraries
source /etc/profile.d/modules.sh
export PYTHONUNBUFFERED=1

cd /data/home/wangcx/krylov-dci
PYTHON=/data/home/wangcx/LiYF4_Er3+/env/bin/python
echo 'Phase 5b: P-Space Strategy Exploration'
echo "Date: $(date)"
PYTHONPATH=/data/home/wangcx/krylov-dci $PYTHON -u scripts/phase5b_pstrategies.py
echo "Exit: $? at $(date)"
