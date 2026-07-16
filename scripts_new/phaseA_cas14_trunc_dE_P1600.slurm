#!/bin/bash
#SBATCH -J trunc-dE-1600
#SBATCH -p amd
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH -o /data/home/wangcx/krylov-dci/slurm_outputs/trunc_dE_v2_P1600_%j.out
#SBATCH -e /data/home/wangcx/krylov-dci/slurm_outputs/trunc_dE_v2_P1600_%j.err

cd /data/home/wangcx/krylov-dci
export PATH=/data/home/wangcx/LiYF4_Er3+/env/bin:$PATH
export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32
export OPENBLAS_NUM_THREADS=32

python scripts_new/phaseA_cas14_trunc_dE.py --target-p 1600
echo "Done: $(date)"
