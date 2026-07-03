#!/bin/bash
#SBATCH -J kdci_phase5
#SBATCH -p amd
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH -o /data/home/wangcx/krylov-dci/logs/phase5_%j.out
#SBATCH -e /data/home/wangcx/krylov-dci/logs/phase5_%j.err

export MODULEPATH=/data/modulefiles/softwares:/data/modulefiles/libraries
source /etc/profile.d/modules.sh
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

cd /data/home/wangcx/krylov-dci
PYTHON=/data/home/wangcx/LiYF4_Er3+/env/bin/python

echo "============================================"
echo "Phase 5: N2/cc-pVDZ Krylov-dCI — PySCF native tools"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "============================================"

PYTHONPATH=/data/home/wangcx/krylov-dci $PYTHON -u scripts/phase5_pyscf_native.py

EXIT_CODE=$?
echo ""
echo "Job completed with exit code: $EXIT_CODE at $(date)"
exit $EXIT_CODE
