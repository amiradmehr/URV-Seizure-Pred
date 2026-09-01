#!/bin/bash
#SBATCH --job-name=sz2-sweep
#SBATCH --partition=gpu,gpu-preempt
#SBATCH --gpus=1
#SBATCH --constraint=sm_75
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=3
#SBATCH --mem=24G
#SBATCH --output=logs/sweep-%A_%a.out
#SBATCH --error=logs/sweep-%A_%a.out
#
# One array task per configuration in scripts/sweep_grid.py.
#
# Each task runs a full 5-fold patient-wise cross-validation and writes its own
# outputs/cv/<tag>/ directory, so results accumulate independently and a failed
# or preempted task can be resubmitted alone.
#
# sm_75, not sm_70. torch 2.11.0+cu128 ships no Volta kernels and its cuDNN
# refuses devices below SM 7.5, so every task that landed on a V100 died with
# either "no kernel image is available" or "cuDNN 91900 is not compatible with
# devices with SM < 7.5" -- 22 of 42 on the first attempt. sm_75 excludes V100
# and still matches more nodes than sm_70 did.
#
# Set the array range from the grid itself:
#     sbatch --array=0-$(($(python scripts/sweep_grid.py --count)-1)) slurm/70_sweep.sbatch

set -euo pipefail
source /work/pi_phuc_umass_edu/aradmehr/Github/URV-Seizure-Pred/slurm/_env.sh

FLAGS=$("${PYTHON}" scripts/sweep_grid.py --flags "${SLURM_ARRAY_TASK_ID}")
echo "config ${SLURM_ARRAY_TASK_ID}: ${FLAGS}"
nvidia-smi || true

# shellcheck disable=SC2086
"${PYTHON}" scripts/train_cv.py --folds 5 --epochs 25 --num-workers 2 ${FLAGS}

echo "Finished: $(date)"
