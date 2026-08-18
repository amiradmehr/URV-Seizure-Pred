#!/bin/bash
#SBATCH --job-name=sz2-eval-cpu
#SBATCH --partition=cpu
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --output=logs/eval-cpu-%j.out
#SBATCH --error=logs/eval-cpu-%j.out
#
# CPU fallback for stage 4, for when the gpu partition is saturated.
#
# Written after the GPU eval sat PENDING for three days: every Volta-or-newer
# node in the gpu partition had all of its GPUs allocated, with 778 jobs
# queued, while the cpu partition had 72 pending against 2305 running.
#
# The model is 21,770 parameters, so it does not need a GPU -- it only runs
# faster on one. Measured: 4.0 decisions/s at 32 threads, so the 48,263-decision
# test split takes ~3.4 h of compute versus ~3.5 min on an A100. The 8 h limit
# is 2x the estimate. Prefer 40_evaluate.sbatch whenever a GPU is obtainable.

set -euo pipefail

source /work/pi_phuc_umass_edu/aradmehr/Github/URV-Seizure-Pred/slurm/_env.sh

echo "Running on CPU with ${SLURM_CPUS_PER_TASK} threads (no GPU requested)."

"${PYTHON}" scripts/evaluate_model.py \
    --device cpu \
    --batch-size 8 \
    --num-workers 4 \
    "$@"

echo
echo "=========================================================================="
echo "Finished: $(date)"
echo "=========================================================================="
