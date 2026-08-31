#!/bin/bash
#SBATCH --job-name=sz2-sweep
#SBATCH --partition=gpu,gpu-preempt
#SBATCH --gpus=1
#SBATCH --constraint=sm_70
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=3
#SBATCH --mem=24G
#SBATCH --output=logs/sweep-%A_%a.out
#SBATCH --error=logs/sweep-%A_%a.out
#
# One array task per configuration in scripts/sweep_grid.py.
#
# Each task runs a full 5-fold patient-wise cross-validation and writes its own
# outputs/cv/<tag>/cv_metrics.json, so results accumulate independently and a
# failed or preempted task costs only its own configuration.
#
# Submit with the grid size, which the grid script reports:
#     sbatch --array=0-$(python scripts/sweep_grid.py --count | awk '{print $1-1}') \
#            slurm/70_sweep.sbatch

set -euo pipefail
source /work/pi_phuc_umass_edu/aradmehr/Github/URV-Seizure-Pred/slurm/_env.sh
nvidia-smi || true

FLAGS=$("${PYTHON}" scripts/sweep_grid.py --flags "${SLURM_ARRAY_TASK_ID}")
echo "config ${SLURM_ARRAY_TASK_ID}: ${FLAGS}"

# shellcheck disable=SC2086
"${PYTHON}" scripts/train_cv.py --folds 5 --epochs 25 --num-workers 2 ${FLAGS}

echo "Finished: $(date)"
