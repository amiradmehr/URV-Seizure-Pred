#!/bin/bash
#SBATCH --job-name=sz2-eval
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=sm_70
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=logs/eval-%j.out
#SBATCH --error=logs/eval-%j.out
#
# Stage 4: score a split at true prevalence and write the result figures.
#
# Defaults to the validation split. The test split is the untouched held-out
# set, so evaluate it once, after model selection is final:
#     sbatch slurm/40_evaluate.sbatch --split test
#
# Sized for backfill, not for headroom. The validation split (53,407 decisions)
# scored in 3m20s, so a 12-hour reservation on the three scarcest GPU types
# made this job effectively unschedulable -- it sat PENDING for days behind a
# 778-job queue. sm_70 covers every Volta-or-newer card (49 nodes instead of
# 20) and the model is 21,770 parameters, so GPU class is irrelevant here.

set -euo pipefail

source /work/pi_phuc_umass_edu/aradmehr/Github/URV-Seizure-Pred/slurm/_env.sh

nvidia-smi || true

"${PYTHON}" scripts/evaluate_model.py \
    --batch-size 8 \
    --num-workers 4 \
    "$@"

echo
echo "=========================================================================="
echo "Finished: $(date)"
echo "=========================================================================="
