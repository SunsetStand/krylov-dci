#!/bin/bash
#SBATCH -J dmrg_c20_v2
#SBATCH -p amd
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH -t 2-00:00:00
#SBATCH -o /data/home/wangcx/krylov-dci/slurm_outputs/dmrg_c20_v2_%j.out
#SBATCH -e /data/home/wangcx/krylov-dci/slurm_outputs/dmrg_c20_v2_%j.err

export OMP_NUM_THREADS=32
cd /data/home/wangcx/krylov-dci
/data/home/wangcx/LiYF4_Er3+/env/bin/python scripts_new/dmrg_standalone.py
