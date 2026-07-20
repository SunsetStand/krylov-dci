#!/bin/bash
#SBATCH -J strong-corr
#SBATCH -p amd
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00
#SBATCH --output=/data/home/wangcx/krylov-dci/slurm_outputs/strong_corr_%j.out
#SBATCH --error=/data/home/wangcx/krylov-dci/slurm_outputs/strong_corr_%j.err

set -e; cd /data/home/wangcx/krylov-dci
PYTHON=/data/home/wangcx/LiYF4_Er3+/env/bin/python
SCRIPTS=/data/home/wangcx/krylov-dci/scripts_new
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1
mkdir -p slurm_outputs checkpoints_bondscan_ext

echo "Strong-Correlation N2: Extended P Strategies (A + B)"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "Python: $PYTHON"
echo ""

$PYTHON $SCRIPTS/extend_strong_corr.py --strategy both \
    --outdir /data/home/wangcx/krylov-dci/checkpoints_bondscan_ext

echo ""
echo "Done: $(date)"
