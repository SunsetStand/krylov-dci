#!/bin/bash
#SBATCH -J kdci-dense-N2
#SBATCH -p amd
#SBATCH -n 32
#SBATCH --time=04:00:00

export MODULEPATH=/data/modulefiles/softwares:/data/modulefiles/libraries
source /etc/profile.d/modules.sh

export PYTHONPATH=/data/home/wangcx/krylov-dci:$PYTHONPATH
PYEXEC=/data/home/wangcx/LiYF4_Er3+/env/bin/python

cd /data/home/wangcx/krylov-dci

echo "=== Phase 14: Dense Krylov-dCI (N2, CAS(10,10)) ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo ""

$PYEXEC scripts/phase14_dense_kdci.py --system N2 --P 200 --nroots 6 --benchmark 2>&1

echo ""
echo "Done at $(date)"
