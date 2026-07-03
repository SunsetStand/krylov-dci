#!/bin/bash
#SBATCH -J kdci_bench
#SBATCH -p amd
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH -o /data/home/wangcx/krylov-dci/logs/bench_%j.out
#SBATCH -e /data/home/wangcx/krylov-dci/logs/bench_%j.err

# ============================================================
# Krylov-dCI Benchmark: H2O/STO-3G convergence + SVD scan
# ============================================================

export MODULEPATH=/data/modulefiles/softwares:/data/modulefiles/libraries
source /etc/profile.d/modules.sh

cd /data/home/wangcx/krylov-dci
PYTHON=/data/home/wangcx/LiYF4_Er3+/env/bin/python

echo "============================================"
echo "Krylov-dCI Benchmark: H2O/STO-3G"
echo "Date: $(date)"
echo "============================================"

$PYTHON test_h2o_correct.py

echo ""
echo "Benchmark completed at $(date)"
