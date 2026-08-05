#!/bin/bash
#SBATCH -J dm_svd_exc
#SBATCH -p amd
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -o /data/home/wangcx/krylov-dci/dm_svd_embedding/logs/n2_excited_%j.out
#SBATCH -e /data/home/wangcx/krylov-dci/dm_svd_embedding/logs/n2_excited_%j.err

export MODULEPATH=/data/modulefiles/softwares:/data/modulefiles/libraries
source /etc/profile.d/modules.sh

cd /data/home/wangcx/krylov-dci
PYTHON=/data/home/wangcx/LiYF4_Er3+/env/bin/python
export PYTHONPATH=/data/home/wangcx/krylov-dci:$PYTHONPATH

echo "============================================"
echo "Phase 1 Extension: Excited States with GS Schmidt Basis"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "Git branch: $(git branch --show-current)"
echo "Git commit: $(git rev-parse --short HEAD)"
echo "============================================"
echo ""
echo "Running: run_n2_excited.py --nroots 5 --eps 1e-3"
echo ""

$PYTHON dm_svd_embedding/scripts/run_n2_excited.py --nroots 5 --eps 1e-3

EXIT_CODE=$?
echo ""
echo "Job completed with exit code: $EXIT_CODE at $(date)"
exit $EXIT_CODE