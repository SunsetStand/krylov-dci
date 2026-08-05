#!/bin/bash
#SBATCH --job-name=phase3_neumann
#SBATCH --output=../slurm_outputs/phase3_neumann_%A_%a.out
#SBATCH --error=../slurm_outputs/phase3_neumann_%A_%a.err
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --partition=amd

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=32
export OPENBLAS_NUM_THREADS=32

PY=/data/home/wangcx/LiYF4_Er3+/env/bin/python
cd /data/home/wangcx/krylov-dci

echo "============================================================"
echo "Phase 3: Growing CAS DMRG + Neumann k=1 — CAS(14,10) Sweep"
echo "Config $SLURM_ARRAY_TASK_ID"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "============================================================"

$PY scripts_new/run_growing_cas_neumann.py \
    --config-id $SLURM_ARRAY_TASK_ID \
    --output-dir ./results/phase3_neumann

echo "Done at $(date)"