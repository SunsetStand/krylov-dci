#!/bin/bash
#SBATCH -J kdci-reorg-check
#SBATCH -p amd
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH -t 00:20:00
#SBATCH -o /data/home/wangcx/krylov-dci/slurm_outputs/reorg_validation_%j.out
#SBATCH -e /data/home/wangcx/krylov-dci/slurm_outputs/reorg_validation_%j.err

set -euo pipefail

export MODULEPATH=/data/modulefiles/softwares:/data/modulefiles/libraries
source /etc/profile.d/modules.sh

PROJECT_ROOT=/data/home/wangcx/krylov-dci
PYTHON=/data/home/wangcx/LiYF4_Er3+/env/bin/python

cd "${PROJECT_ROOT}"
mkdir -p slurm_outputs
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2

test -f scripts/diagnostics/hab_rdm/diag_hab_test.py
test -f tests/unit/test_pspace_ops.py
test -f tests/regression/test_kdci.py
test -f tests/regression/test_hab_rdm.py

"${PYTHON}" -c "import src, src_mf, dm_svd_embedding, dm_svd_dci; print('IMPORT_CHECK_PASS', flush=True)"
"${PYTHON}" tests/unit/test_pspace_ops.py
"${PYTHON}" tests/regression/test_kdci.py
"${PYTHON}" tests/regression/test_hab_rdm.py
"${PYTHON}" scripts/diagnostics/hab_rdm/diag_hab_test.py

echo "REORG_VALIDATION_PASS"
