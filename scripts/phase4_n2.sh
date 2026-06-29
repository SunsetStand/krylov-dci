#!/bin/bash
#SBATCH -J kdci_phase4_n2
#SBATCH -p amd
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH -o /data/home/wangcx/krylov-dci/logs/phase4_n2_%j.out
#SBATCH -e /data/home/wangcx/krylov-dci/logs/phase4_n2_%j.err

export MODULEPATH=/data/modulefiles/softwares:/data/modulefiles/libraries
source /etc/profile.d/modules.sh

cd /data/home/wangcx/krylov-dci
PYTHON=/data/home/wangcx/LiYF4_Er3+/env/bin/python

echo "============================================"
echo "Phase 4: N2/cc-pVDZ P-Space Strategy Test"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "============================================"

PYTHONPATH=/data/home/wangcx/krylov-dci $PYTHON scripts/phase4_n2_pstrategies.py

EXIT_CODE=$?
echo ""
echo "Job completed with exit code: $EXIT_CODE at $(date)"
exit $EXIT_CODE
