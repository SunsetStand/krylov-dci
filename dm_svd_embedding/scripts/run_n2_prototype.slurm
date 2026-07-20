#!/bin/bash
#SBATCH -J dm_svd_n2
#SBATCH -p amd
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -o /data/home/wangcx/krylov-dci/dm_svd_embedding/logs/n2_prototype_%j.out
#SBATCH -e /data/home/wangcx/krylov-dci/dm_svd_embedding/logs/n2_prototype_%j.err

export MODULEPATH=/data/modulefiles/softwares:/data/modulefiles/libraries
source /etc/profile.d/modules.sh

cd /data/home/wangcx/krylov-dci
PYTHON=/data/home/wangcx/LiYF4_Er3+/env/bin/python
export PYTHONPATH=/data/home/wangcx/krylov-dci:$PYTHONPATH

echo "============================================"
echo "Phase 1: Density Matrix SVD Embedding on N₂"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "Git branch: $(git branch --show-current)"
echo "Git commit: $(git rev-parse --short HEAD)"
echo "============================================"

$PYTHON dm_svd_embedding/scripts/run_n2_prototype.py

EXIT_CODE=$?
echo ""
echo "Job completed with exit code: $EXIT_CODE at $(date)"
exit $EXIT_CODE