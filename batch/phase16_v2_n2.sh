#!/bin/bash
#SBATCH -J kdci-mf-v2
#SBATCH -p amd
#SBATCH -n 32
#SBATCH --time=02:00:00

export MODULEPATH=/data/modulefiles/softwares:/data/modulefiles/libraries
source /etc/profile.d/modules.sh
export PYTHONPATH=/data/home/wangcx/krylov-dci:$PYTHONPATH
PYEXEC=/data/home/wangcx/LiYF4_Er3+/env/bin/python
cd /data/home/wangcx/krylov-dci

echo "=== Phase 16 v2: Optimized Matrix-Free N2 CAS(10,10) ==="
echo "Date: $(date)  Node: $(hostname)"
echo ""

$PYEXEC scripts/phase16_matrix_free.py --system N2 --P 200 --dense-compare 2>&1

echo ""
echo "Done at $(date)"
