#!/bin/bash
#SBATCH -J kdci-scale
#SBATCH -p amd
#SBATCH -n 32
#SBATCH --time=12:00:00

export MODULEPATH=/data/modulefiles/softwares:/data/modulefiles/libraries
source /etc/profile.d/modules.sh
export PYTHONPATH=/data/home/wangcx/krylov-dci:$PYTHONPATH
PYEXEC=/data/home/wangcx/LiYF4_Er3+/env/bin/python
cd /data/home/wangcx/krylov-dci

echo "=== Phase 17: CAS Scaling Benchmark ==="
echo "Date: $(date)  Node: $(hostname)"
echo "Free memory: $(free -h | grep Mem | awk '{print $7}')"
echo ""

$PYEXEC scripts/phase17_cas_scaling.py --P 200 --nroots 3 --max-cas 14 2>&1

echo ""
echo "Done at $(date)"
