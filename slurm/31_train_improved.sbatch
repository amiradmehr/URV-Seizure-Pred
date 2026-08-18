#!/bin/bash
#SBATCH --job-name=sz2-train2
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=sm_70
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/train2-%j.out
#SBATCH --error=logs/train2-%j.out
#
# Stage 3b: train a variant with the fixes applied, then immediately evaluate it
# on validation at true prevalence. Test is never touched here.
#
# All arguments pass through, so each variant is one submission:
#   sbatch slurm/31_train_improved.sbatch --tag all-fixes \
#       --architecture attention --window-normalize --densify-positive-stride 10
#
# Sized for backfill: sm_70 spans every Volta-or-newer node (49) rather than the
# 20 that a100|l40s|a40 matched, and 12 h is roughly 4x the expected runtime
# rather than the 8x that made the first evaluation job unschedulable.

set -euo pipefail

source /work/pi_phuc_umass_edu/aradmehr/Github/URV-Seizure-Pred/slurm/_env.sh

TAG="run"
PASS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --tag) TAG="$2"; shift 2 ;;
    *) PASS+=("$1"); shift ;;
  esac
done

MODEL_DIR="${REPO_ROOT}/outputs/models/${TAG}"
EVAL_DIR="${REPO_ROOT}/outputs/evaluation_${TAG}"

nvidia-smi || true
echo "Variant tag : ${TAG}"
echo "Extra flags : ${PASS[*]:-none}"

"${PYTHON}" scripts/train_eegnet_baseline.py \
    --epochs 12 \
    --batch-size 8 \
    --num-workers 7 \
    --train-negative-ratio 4 \
    --validation-negative-ratio 20 \
    --encoder-chunk-batch-size 1024 \
    --log-interval 400 \
    --output-dir "${MODEL_DIR}" \
    "${PASS[@]}"

echo
echo "### validation at true prevalence ###"
"${PYTHON}" scripts/evaluate_model.py \
    --checkpoint "${MODEL_DIR}/best_model.pt" \
    --split validation \
    --batch-size 8 \
    --num-workers 7 \
    --output-dir "${EVAL_DIR}"

echo
echo "=========================================================================="
echo "Finished ${TAG}: $(date)"
echo "=========================================================================="
