#!/bin/bash
#SBATCH --job-name=grow_sweep
#SBATCH --output=slurm_outputs/grow_sweep_%A_%a.out
#SBATCH --error=slurm_outputs/grow_sweep_%A_%a.err
#SBATCH --partition=amd
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=48:00:00
#SBATCH --array=0-6

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=32
export OPENBLAS_NUM_THREADS=32
export PYTHONUNBUFFERED=1

cd /data/home/wangcx/krylov-dci

PY=/data/home/wangcx/LiYF4_Er3+/env/bin/python
OUT_DIR="./results/grow_sweep_${SLURM_ARRAY_JOB_ID}"

mkdir -p "$OUT_DIR"
mkdir -p slurm_outputs

echo "============================================"
echo "Growing CAS DMRG — Parameter Sweep"
echo "============================================"
echo "Job ID:       ${SLURM_JOB_ID}"
echo "Array Task:   ${SLURM_ARRAY_TASK_ID}"
echo "Date:         $(date)"
echo "Node:         $(hostname)"
echo "CPUs:         ${SLURM_CPUS_PER_TASK}"
echo "Memory:       ${SLURM_MEM_PER_NODE}"
echo "============================================"
echo ""
echo "Configuration index: ${SLURM_ARRAY_TASK_ID}"
echo ""

$PY scripts_new/run_growing_cas_sweep.py \
    --config-id ${SLURM_ARRAY_TASK_ID} \
    --output-dir "$OUT_DIR"

exit_code=$?
echo ""
echo "============================================"
echo "Task ${SLURM_ARRAY_TASK_ID} done at $(date)"
echo "Exit code: ${exit_code}"
echo "============================================"
exit ${exit_code}