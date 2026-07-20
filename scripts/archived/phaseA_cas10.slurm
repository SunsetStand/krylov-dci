#!/bin/bash
#SBATCH -J phaseA_c10_v3
#SBATCH -p amd
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/data/home/wangcx/krylov-dci/slurm_outputs/phaseA_c10_v3_%j.out
#SBATCH --error=/data/home/wangcx/krylov-dci/slurm_outputs/phaseA_c10_v3_%j.err

export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16

/data/home/wangcx/LiYF4_Er3+/env/bin/python \
  /data/home/wangcx/krylov-dci/scripts_new/phaseA_cas10_v3.py \
  --P 200,500,1000,2000 \
  --svd-threshold 1e-3 \
  --m-max 1 \
  --tag v3
