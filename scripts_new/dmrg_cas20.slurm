#!/bin/bash
#SBATCH -J dmrg_c20
#SBATCH -p amd
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/data/home/wangcx/krylov-dci/slurm_outputs/dmrg_c20_%j.out
#SBATCH --error=/data/home/wangcx/krylov-dci/slurm_outputs/dmrg_c20_%j.err

export OMP_NUM_THREADS=4
/data/home/wangcx/LiYF4_Er3+/env/bin/python /data/home/wangcx/krylov-dci/scripts_new/dmrg_cas20.py
