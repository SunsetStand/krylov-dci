#!/bin/bash
#SBATCH --job-name=krylov_AB_test
#SBATCH --output=slurm_outputs/consistency_AB_%j.out
#SBATCH --error=slurm_outputs/consistency_AB_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd /data/home/wangcx/krylov-dci
PY=/data/home/wangcx/LiYF4_Er3+/env/bin/python

echo "============================================"
echo "Scheme A vs B Consistency Test (gs, m_max=1)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Date: $(date)"
echo "============================================"

OUT_A="./results/consistency_AB_${SLURM_JOB_ID}_A"
OUT_B="./results/consistency_AB_${SLURM_JOB_ID}_B"

echo ""
echo "=== Test 1: scheme=A (reference) ==="
$PY scripts/run_dm_svd_dci.py \
    --atom 'N 0 0 0; N 0 0 1.098' \
    --basis cc-pVDZ \
    --n-active 10 --n-alpha 5 --n-beta 5 --n-core 2 --n-occ 5 \
    --svd-eps 1e-3 \
    --mode gs \
    --p-blocks 8,9,10 \
    --m-max 1 \
    --scheme A \
    --n-workers 8 \
    --output-dir "$OUT_A"

echo ""
echo "=== Test 2: scheme=B (streaming) ==="
$PY scripts/run_dm_svd_dci.py \
    --atom 'N 0 0 0; N 0 0 1.098' \
    --basis cc-pVDZ \
    --n-active 10 --n-alpha 5 --n-beta 5 --n-core 2 --n-occ 5 \
    --svd-eps 1e-3 \
    --mode gs \
    --p-blocks 8,9,10 \
    --m-max 1 \
    --scheme B \
    --batch-size 32 \
    --n-workers 8 \
    --output-dir "$OUT_B"

echo ""
echo "=== Consistency Check ==="
$PY -c "
import json

with open('${OUT_A}/dm_svd_dci_results.json') as f:
    a = json.load(f)
with open('${OUT_B}/dm_svd_dci_results.json') as f:
    b = json.load(f)

checks = [
    ('E_fci', 'FCI reference'),
    ('E_bare_P', 'Bare H_PP'),
    ('E_eff_m0', 'E(m=0)'),
    ('dE_m0_mH', 'dE(m=0) mH'),
    ('E_eff_m1', 'E(m=1)'),
    ('dE_m1_mH', 'dE(m=1) mH'),
]

all_ok = True
for key, label in checks:
    va = a.get(key, float('nan'))
    vb = b.get(key, float('nan'))
    diff = abs(va - vb) if (isinstance(va, (int,float)) and isinstance(vb, (int,float))) else float('nan')
    print(f'  {label:20s} A={va!s:24s} B={vb!s:24s} diff={diff}')
    if isinstance(diff, (int,float)) and diff > 1e-6:
        all_ok = False

print()
dE_a_m0 = a.get('dE_m0_mH', 0)
dE_b_m0 = b.get('dE_m0_mH', 0)
print(f'  dE(m=0) A = {dE_a_m0:+.6f} mH  vs B = {dE_b_m0:+.6f} mH')
if abs(dE_a_m0 - dE_b_m0) < 1e-4:
    print('  PASS: m=0: Scheme B matches Scheme A within 1e-4 mH')
else:
    print('  FAIL: m=0: Significant discrepancy!')
    all_ok = False

dE_a_m1 = a.get('dE_m1_mH', 0)
dE_b_m1 = b.get('dE_m1_mH', 0)
print(f'  dE(m=1) A = {dE_a_m1:+.6f} mH  vs B = {dE_b_m1:+.6f} mH')
if abs(dE_a_m1 - dE_b_m1) < 1e-4:
    print('  PASS: m=1: Scheme B matches Scheme A within 1e-4 mH')
else:
    print('  FAIL: m=1: Significant discrepancy!')
    all_ok = False

print()
if all_ok:
    print('ALL CHECKS PASSED')
else:
    print('SOME CHECKS FAILED')
    exit(1)
"

exit_code=$?
echo ""
echo "Done at $(date) with exit code ${exit_code}"
exit ${exit_code}