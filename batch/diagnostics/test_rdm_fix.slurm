#!/bin/bash
#SBATCH -J test-rdm-fix
#SBATCH -p amd
#SBATCH -n 4
#SBATCH -t 00:30:00
#SBATCH -o slurm_outputs/test_rdm_fix_%j.out
#SBATCH -e slurm_outputs/test_rdm_fix_%j.err

export MODULEPATH=/data/modulefiles/softwares:/data/modulefiles/libraries
source /etc/profile.d/modules.sh

cd /data/home/wangcx/krylov-dci
mkdir -p slurm_outputs

export PYTHONPATH=/data/home/wangcx/krylov-dci:$PYTHONPATH
export OMP_NUM_THREADS=2

/data/home/wangcx/LiYF4_Er3+/env/bin/python -c "
from dm_svd_embedding.embedded_hamiltonian import test_hemb_rdm_vs_sigma
H_ref, H_rdm, diff, devals = test_hemb_rdm_vs_sigma()
print(f'max|H_ref - H_rdm| = {diff:.4e}')
for i in range(min(5, len(devals))):
    print(f'  E[{i}] diff = {devals[i]:.4e}')
"
echo "=== done ==="
