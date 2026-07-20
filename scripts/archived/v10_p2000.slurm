#!/bin/bash
#SBATCH -J v10p2000
#SBATCH -p amd
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH -t 6:00:00
#SBATCH -o /data/home/wangcx/krylov-dci/slurm_outputs/v10_p2000.out
#SBATCH -e /data/home/wangcx/krylov-dci/slurm_outputs/v10_p2000.err
export OMP_NUM_THREADS=12;export MKL_NUM_THREADS=12;export OPENBLAS_NUM_THREADS=12
cd /data/home/wangcx/krylov-dci
/data/home/wangcx/LiYF4_Er3+/env/bin/python scripts_new/phaseA_cas10_v10_sacis.py --P 2000 --m-max 1 --tag p2000trunc
