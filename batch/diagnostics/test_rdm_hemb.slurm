#!/bin/bash
#SBATCH -J test_rdm_hemb
#SBATCH -p amd
#SBATCH -n 8
#SBATCH -t 00:30:00
#SBATCH -o /data/home/wangcx/krylov-dci/slurm_outputs/test_rdm_hemb.out
#SBATCH -e /data/home/wangcx/krylov-dci/slurm_outputs/test_rdm_hemb.err

export PYTHONPATH=/data/home/wangcx/krylov-dci:$PYTHONPATH
cd /data/home/wangcx/krylov-dci
/data/home/wangcx/LiYF4_Er3+/env/bin/python dm_svd_embedding/embedded_hamiltonian.py --rdm-test
echo "Exit code: $?"
