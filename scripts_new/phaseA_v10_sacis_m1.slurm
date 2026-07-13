#!/bin/bash
#SBATCH -J sa_cis_m1
#SBATCH -p amd
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=4:00:00
#SBATCH -o /data/home/wangcx/krylov-dci/slurm_outputs/phaseA_v10_sacis_m1.out
#SBATCH -e /data/home/wangcx/krylov-dci/slurm_outputs/phaseA_v10_sacis_m1.err

cd /data/home/wangcx/krylov-dci
export PATH=/data/home/wangcx/LiYF4_Er3+/env/bin:$PATH
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16

/data/home/wangcx/LiYF4_Er3+/env/bin/python scripts_new/phaseA_cas10_v10_sacis.py \
    --P 400,800,1200,2000 \
    --m-max 1 \
    --tag sacis_m1

echo "Done: $(date)"
