#!/bin/bash
#SBATCH --job-name=dm_svd_dci
#SBATCH --output=slurm_outputs/dm_svd_dci_%j.out
#SBATCH --error=slurm_outputs/dm_svd_dci_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00

# ============================================================================
# dmSVD + Krylov-dCI SLURM Job Script
#
# Multi-core parallel sigma-vector computation via ThreadPoolExecutor.
# contract_2e (C-level libfci) releases the GIL, so Python threads provide
# real parallelism.
#
# To customize parameters, edit the variables below or use:
#   sbatch --export=ALL batch/dm_svd_dci.slurm
#
# Environment variables that can be overridden at submit time:
#   ATOM, BASIS, N_ACTIVE, N_ALPHA, N_BETA, N_CORE, N_OCC
#   SVD_EPS, MODE, SA_STATES, P_BLOCKS, M_MAX, DELTA
#   N_WORKERS, OUTPUT_DIR
# ============================================================================

# ── SLURM sets OMP_NUM_THREADS to cpus-per-task by default ──
# We limit BLAS to 1 thread because Python ThreadPoolExecutor handles parallelism.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# ── System parameters (customizable via --export) ──
ATOM="${ATOM:-N 0 0 0; N 0 0 1.098}"
BASIS="${BASIS:-cc-pVDZ}"
N_ACTIVE="${N_ACTIVE:-10}"
N_ALPHA="${N_ALPHA:-5}"
N_BETA="${N_BETA:-5}"
N_CORE="${N_CORE:-2}"
N_OCC="${N_OCC:-5}"

# ── dmSVD parameters ──
SVD_EPS="${SVD_EPS:-1e-3}"

# ── Mode ──
MODE="${MODE:-gs}"
SA_STATES="${SA_STATES:-1}"

# ── P/Q partition ──
P_BLOCKS="${P_BLOCKS:-8,9,10}"

# ── Krylov-dCI ──
M_MAX="${M_MAX:-1}"
DELTA="${DELTA:-0.0}"

# ── Parallel ──
N_WORKERS="${N_WORKERS:-${SLURM_CPUS_PER_TASK}}"

# ── Output ──
OUTPUT_DIR="${OUTPUT_DIR:-./results/dm_svd_dci_${SLURM_JOB_ID}}"

# ── Project root ──
cd /data/home/wangcx/krylov-dci

echo "============================================"
echo "dmSVD + Krylov-dCI Job"
echo "============================================"
echo "  Job ID:      ${SLURM_JOB_ID}"
echo "  CPUs:        ${SLURM_CPUS_PER_TASK}"
echo "  Workers:     ${N_WORKERS}"
echo "  Node:        ${SLURMD_NODENAME}"
echo "  System:      ${ATOM}, ${BASIS}"
echo "  CAS:         (${N_ACTIVE}, $((N_ALPHA + N_BETA)))"
echo "  Mode:        ${MODE} (sa_states=${SA_STATES})"
echo "  P-blocks:    ${P_BLOCKS}"
echo "  m_max:       ${M_MAX}"
echo "  Output dir:  ${OUTPUT_DIR}"
echo "  OMP threads: ${OMP_NUM_THREADS}"
echo "============================================"
echo ""

# ── Run pipeline ──
/data/home/wangcx/LiYF4_Er3+/env/bin/python scripts/run_dm_svd_dci.py \
    --atom "${ATOM}" \
    --basis "${BASIS}" \
    --n-active "${N_ACTIVE}" \
    --n-alpha "${N_ALPHA}" \
    --n-beta "${N_BETA}" \
    --n-core "${N_CORE}" \
    --n-occ "${N_OCC}" \
    --svd-eps "${SVD_EPS}" \
    --mode "${MODE}" \
    --sa-states "${SA_STATES}" \
    --p-blocks "${P_BLOCKS}" \
    --m-max "${M_MAX}" \
    --delta "${DELTA}" \
    --n-workers "${N_WORKERS}" \
    --output-dir "${OUTPUT_DIR}"

exit_code=$?

echo ""
echo "Job finished with exit code ${exit_code}"
echo "Output saved to ${OUTPUT_DIR}"
exit ${exit_code}