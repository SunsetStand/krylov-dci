#!/bin/bash
#SBATCH -J cas14_svd
#SBATCH -p amd
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=20:00:00
#SBATCH -o /data/home/wangcx/krylov-dci/slurm_outputs/phaseA_cas14_svd_m1.out
#SBATCH -e /data/home/wangcx/krylov-dci/slurm_outputs/phaseA_cas14_svd_m1.err

cd /data/home/wangcx/krylov-dci
export PATH=/data/home/wangcx/LiYF4_Er3+/env/bin:$PATH
export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32
export OPENBLAS_NUM_THREADS=32

/data/home/wangcx/LiYF4_Er3+/env/bin/python scripts_new/phaseA_cas14_svd.py \
    --P 400,800,1600,2000 \
    --m-max 1 \
    --tag cas14svd_m1opt

echo "Done: $(date)"
