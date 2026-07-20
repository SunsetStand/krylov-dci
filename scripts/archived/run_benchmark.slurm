#!/bin/bash
#SBATCH -J pspace-bench
#SBATCH -p amd
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --output=/data/home/wangcx/krylov-dci/logs/pspace_bench_%j.out
#SBATCH --error=/data/home/wangcx/krylov-dci/logs/pspace_bench_%j.err

set -e
cd /data/home/wangcx/krylov-dci

PYTHON=/data/home/wangcx/LiYF4_Er3+/env/bin/python
SCRIPTS=/data/home/wangcx/krylov-dci/scripts_new
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1

mkdir -p logs

echo "============================================================"
echo "P-space Bloch H^eff Benchmark — Shared + Per-State"
echo "Python: $PYTHON"
echo "Started at $(date)"
echo "============================================================"

# ── Phase 1: Shared P (fixed step1) ──
echo ""
echo "===== PHASE 1: Shared P-space selection ====="
$PYTHON $SCRIPTS/step1_pspace_selection.py --mode shared
echo "Phase 1 step1 done at $(date)"

echo ""
echo "===== PHASE 1b: Shared P Bloch evaluation ====="
$PYTHON $SCRIPTS/step2_bloch_benchmark.py --mode shared
echo "Phase 1 step2 done at $(date)"

# ── Phase 2: Per-state P_k ──
echo ""
echo "===== PHASE 2: Per-state P-space selection ====="
$PYTHON $SCRIPTS/step1_pspace_selection.py --mode per_state
echo "Phase 2 step1 done at $(date)"

echo ""
echo "===== PHASE 2b: Per-state P_k Bloch evaluation ====="
$PYTHON $SCRIPTS/step2_bloch_benchmark.py --mode per_state
echo "Phase 2 step2 done at $(date)"

echo ""
echo "============================================================"
echo "Benchmark complete at $(date)"
echo "============================================================"
