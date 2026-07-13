#!/bin/bash
#SBATCH -J ps_s1_fci
#SBATCH -p amd
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH -o /data/home/wangcx/krylov-dci/slurm_outputs/phaseA_ps_s1_fci.out
#SBATCH -e /data/home/wangcx/krylov-dci/slurm_outputs/phaseA_ps_s1_fci.err

cd /data/home/wangcx/krylov-dci
source /data/home/wangcx/env/pyscf_env/bin/activate
export OMP_NUM_THREADS=32

python scripts_new/phaseA_cas10_perstate_s1.py \
    --P 400,600,800,1000,2000,4000 \
    --eref fci \
    --m-max 1 \
    --tag s1_fci

echo "Done: $(date)"
