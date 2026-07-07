#!/bin/bash
#SBATCH -J bondscan
#SBATCH -p amd
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00
#SBATCH -o /data/home/wangcx/krylov-dci/slurm_outputs/bondscan_%j.out
#SBATCH -e /data/home/wangcx/krylov-dci/slurm_outputs/bondscan_%j.err

echo "N2 Bond Length Scan: P-space + Bloch H^eff"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo ""

cd /data/home/wangcx/krylov-dci
source ~/env/bin/activate 2>/dev/null || true

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

python3 scripts_new/run_bond_scan.py \
    --outdir /data/home/wangcx/krylov-dci/checkpoints_bondscan

echo ""
echo "Done: $(date)"
