#!/bin/bash
#SBATCH --job-name=sz2-train2
#SBATCH --partition=gpu,gpu-preempt
#SBATCH --gpus=1
#SBATCH --constraint=sm_70
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=3
#SBATCH --mem=24G
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
# Sized against what is actually free, not against what would be comfortable.
# The gpu partition is routinely CPU-bound rather than GPU-bound: nodes sit in
# "allocated" state with 3-5 idle GPUs, 150-260 GB free memory and *zero* idle
# CPUs, because other jobs took all 24 cores. Measured at submission time,
# --cpus-per-task=8 matched 0 placeable nodes while 3 matched one immediately.
# Hence 3 CPUs / 24 G, and both partitions so gpu-preempt can pick it up first.
# --num-workers is 2 to stay inside the 3 cores; data loading is the bottleneck,
# so expect roughly 2-3x the wall time of an 8-core run.

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
    --num-workers 2 \
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
    --num-workers 2 \
    --output-dir "${EVAL_DIR}"

echo
echo "=========================================================================="
echo "Finished ${TAG}: $(date)"
echo "=========================================================================="
