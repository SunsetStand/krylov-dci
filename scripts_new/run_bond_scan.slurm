#!/bin/bash
#SBATCH -J bondscan
#SBATCH -p amd
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00
#SBATCH -o /data/home/wangcx/krylov-dci/slurm_outputs/bondscan_%j.out
#SBATCH -e /data/home/wangcx/krylov-dci/slurm_outputs/bondscan_%j.err

set -e
cd /data/home/wangcx/krylov-dci

PYTHON=/data/home/wangcx/LiYF4_Er3+/env/bin/python
SCRIPTS=/data/home/wangcx/krylov-dci/scripts_new
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1

mkdir -p slurm_outputs checkpoints_bondscan

echo "N2 Bond Length Scan: P-space + Bloch H^eff"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "Python: $PYTHON"
echo ""

$PYTHON $SCRIPTS/run_bond_scan.py \
    --outdir /data/home/wangcx/krylov-dci/checkpoints_bondscan

echo ""
echo "Done: $(date)"
