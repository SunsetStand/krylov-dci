#!/bin/bash
#SBATCH -J kdci_phase6
#SBATCH -p amd
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH -o /data/home/wangcx/krylov-dci/logs/phase6_%j.out
#SBATCH -e /data/home/wangcx/krylov-dci/logs/phase6_%j.err

export MODULEPATH=/data/modulefiles/softwares:/data/modulefiles/libraries
source /etc/profile.d/modules.sh
export PYTHONUNBUFFERED=1

cd /data/home/wangcx/krylov-dci
PYTHON=/data/home/wangcx/LiYF4_Er3+/env/bin/python
echo 'Phase 6: Krylov Convergence Test'
echo "Date: $(date)"
PYTHONPATH=/data/home/wangcx/krylov-dci $PYTHON -u scripts/phase6_krylov_convergence.py
echo "Exit: $? at $(date)"
