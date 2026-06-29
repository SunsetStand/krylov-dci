#!/bin/bash
#SBATCH -J kdci_phase3_h2o
#SBATCH -p amd
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH -o /data/home/wangcx/krylov-dci/logs/phase3_h2o_%j.out
#SBATCH -e /data/home/wangcx/krylov-dci/logs/phase3_h2o_%j.err

export MODULEPATH=/data/modulefiles/softwares:/data/modulefiles/libraries
source /etc/profile.d/modules.sh

cd /data/home/wangcx/krylov-dci
PYTHON=/data/home/wangcx/LiYF4_Er3+/env/bin/python

echo "============================================"
echo "Phase 3: H2O/STO-3G SVD Compression Analysis"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "============================================"

PYTHONPATH=/data/home/wangcx/krylov-dci $PYTHON scripts/phase3_h2o_svd_analysis.py

EXIT_CODE=$?
echo ""
echo "Job completed with exit code: $EXIT_CODE at $(date)"
exit $EXIT_CODE
