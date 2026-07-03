#!/bin/bash
# Phase 11 Stage B: Submit independent jobs per P value
# Each P value runs in its own SLURM job with checkpointing.
# Rerunning this script will automatically skip already-completed P values.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CKPT_DIR="${PROJECT_DIR}/checkpoints_stageB"
LOG_DIR="${PROJECT_DIR}/logs"
PYTHON="/data/home/wangcx/LiYF4_Er3+/env/bin/python"
SCRIPT="${SCRIPT_DIR}/phase11_stageB.py"

# P values to scan
P_VALUES=(50 100 200 400 600 800 1000)
M_MAX=3
WALLTIME="12:00:00"    # per P

mkdir -p "${LOG_DIR}"
mkdir -p "${CKPT_DIR}"

echo "============================================"
echo "Phase 11 Stage B: P-Convergence Submission"
echo "============================================"
echo "P values: ${P_VALUES[*]}"
echo "m max:    ${M_MAX}"
echo "CKPT dir: ${CKPT_DIR}"
echo "Log dir:  ${LOG_DIR}"
echo "============================================"

JOB_IDS=()

for P in "${P_VALUES[@]}"; do
    JOB_NAME="stgB_P${P}"
    
    # Check if already completed
    if [ -f "${CKPT_DIR}/P$(printf '%04d' ${P})/krylov_m${M_MAX}.npz" ]; then
        echo "P=${P}: already completed (krylov_m${M_MAX}.npz exists), skipping."
        continue
    fi
    
    # Determine CPUs based on P size (larger P needs more memory/CPU)
    if [ ${P} -le 100 ]; then
        NCPU=8
    elif [ ${P} -le 400 ]; then
        NCPU=16
    else
        NCPU=32
    fi
    
    echo "Submitting P=${P} (${NCPU} cores)..."
    
    JOB_ID=$(sbatch --parsable \
        -J "${JOB_NAME}" \
        -p amd \
        -N 1 \
        --ntasks-per-node=${NCPU} \
        --time=${WALLTIME} \
        -o "${LOG_DIR}/%j_stageB_P${P}.out" \
        -e "${LOG_DIR}/%j_stageB_P${P}.err" \
        --wrap "export MODULEPATH=/data/modulefiles/softwares:/data/modulefiles/libraries
                source /etc/profile.d/modules.sh
                ${PYTHON} ${SCRIPT} --P ${P} --m-max ${M_MAX} --ckpt-dir ${CKPT_DIR}"
    )
    
    JOB_IDS+=("${JOB_ID}")
    echo "  → Job ${JOB_ID}"
done

echo "============================================"
echo "Submitted ${#JOB_IDS[@]} jobs: ${JOB_IDS[*]}"
echo ""
echo "Monitor:  squeue -u \$USER | grep stgB"
echo "Check progress: ls ${CKPT_DIR}/P*/krylov_m*.npz"
echo "Gather results:"
echo "  for d in ${CKPT_DIR}/P*/; do"
echo "    P=\$(basename \$d | sed 's/P0*//')"
echo "    if [ -f \$d/summary.json ]; then"
echo "      python3 -c \"import json; d=json.load(open('\$d/summary.json'));"
echo "      print(f'P={d[\\\"P_target\\\"]:4d}  E0={d[\\\"results\\\"][-1][\\\"dE0_mH\\\"]:+.1f} mH')\""
echo "    fi"
echo "  done"
echo "============================================"
