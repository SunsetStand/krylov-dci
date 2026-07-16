#!/bin/bash
#SBATCH -J trunc-dE-800
#SBATCH -p amd
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=6:00:00
#SBATCH -o /data/home/wangcx/krylov-dci/slurm_outputs/trunc_dE_P800_%j.out
#SBATCH -e /data/home/wangcx/krylov-dci/slurm_outputs/trunc_dE_P800_%j.err

cd /data/home/wangcx/krylov-dci
mkdir -p slurm_outputs tmp checkpoints_phaseA

export PATH=/data/home/wangcx/LiYF4_Er3+/env/bin:$PATH
export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32
export OPENBLAS_NUM_THREADS=32

python scripts_new/phaseA_cas14_truncation_error.py --target-p 800
echo "Done: $(date)"
