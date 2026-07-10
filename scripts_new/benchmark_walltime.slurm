#!/bin/bash
#SBATCH -J bench_wt
#SBATCH -p amd
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH -t 8:00:00
#SBATCH -o /data/home/wangcx/krylov-dci/slurm_outputs/bench_wt_%j.out
#SBATCH -e /data/home/wangcx/krylov-dci/slurm_outputs/bench_wt_%j.err

export OMP_NUM_THREADS=32
cd /data/home/wangcx/krylov-dci
/data/home/wangcx/LiYF4_Er3+/env/bin/python scripts_new/benchmark_walltime.py
