#!/bin/bash
#SBATCH -J diag-hab
#SBATCH -p amd
#SBATCH -n 4
#SBATCH -t 00:20:00
#SBATCH -o slurm_outputs/diag_hab_%j.out
#SBATCH -e slurm_outputs/diag_hab_%j.err

export MODULEPATH=/data/modulefiles/softwares:/data/modulefiles/libraries
source /etc/profile.d/modules.sh
cd /data/home/wangcx/krylov-dci
mkdir -p slurm_outputs
export PYTHONPATH=/data/home/wangcx/krylov-dci:$PYTHONPATH
export OMP_NUM_THREADS=2
/data/home/wangcx/LiYF4_Er3+/env/bin/python scripts/diagnostics/hab_rdm/diag_hab_test.py
echo '=== done ==='
