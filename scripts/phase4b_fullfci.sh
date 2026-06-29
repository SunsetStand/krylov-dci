#!/bin/bash
#SBATCH -J kdci_phase4b
#SBATCH -p amd
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH -o /data/home/wangcx/krylov-dci/logs/phase4b_%j.out
#SBATCH -e /data/home/wangcx/krylov-dci/logs/phase4b_%j.err

export MODULEPATH=/data/modulefiles/softwares:/data/modulefiles/libraries
source /etc/profile.d/modules.sh
export PYTHONUNBUFFERED=1

cd /data/home/wangcx/krylov-dci
PYTHON=/data/home/wangcx/LiYF4_Er3+/env/bin/python

echo "============================================"
echo "Phase 4b: N2/cc-pVDZ Full FCI PT2 P-selection"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "============================================"

PYTHONPATH=/data/home/wangcx/krylov-dci $PYTHON -u scripts/phase4b_fullfci_pt2.py

EXIT_CODE=$?
echo ""
echo "Job completed with exit code: $EXIT_CODE at $(date)"
exit $EXIT_CODE
