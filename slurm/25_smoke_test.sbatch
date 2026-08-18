#!/bin/bash
#SBATCH --job-name=sz2-smoke
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=a100|l40s|a40
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/smoke-%j.out
#SBATCH --error=logs/smoke-%j.out
#
# Integration smoke test: one epoch of training on a deliberately tiny negative
# sample, then a full pass through evaluate_model.py on the resulting
# checkpoint.
#
# This exists because evaluate_model.py runs LAST in the pipeline. Without this
# job, a bug in it would only surface after training has already spent hours on
# a GPU. Running the same code path early, on real data, turns that into a
# 15-minute signal.
#
# Scheduled to run CONCURRENTLY with the real training job, not before it, and
# writes to outputs/smoke_test/ rather than the real output directories. It
# therefore cannot delay or corrupt the actual run: if it fails, training is
# unaffected and the failure is recorded in the run report.
#
# All positives are always retained by load_decision_examples, so the smallest
# possible epoch is ~2,484 positives plus 10% negatives.

set -uo pipefail

source /work/pi_phuc_umass_edu/aradmehr/Github/URV-Seizure-Pred/slurm/_env.sh

nvidia-smi || true

SMOKE_DIR="${REPO_ROOT}/outputs/smoke_test"
rm -rf "${SMOKE_DIR}"
mkdir -p "${SMOKE_DIR}"

echo
echo "### 1/2  Training path (1 epoch, minimal negatives) ###"
"${PYTHON}" scripts/train_eegnet_baseline.py \
    --epochs 1 \
    --batch-size 4 \
    --num-workers 8 \
    --train-negative-ratio 0.1 \
    --validation-negative-ratio 0.1 \
    --encoder-chunk-batch-size 1024 \
    --log-interval 100 \
    --output-dir "${SMOKE_DIR}/model"
TRAIN_STATUS=$?
echo "train exit status: ${TRAIN_STATUS}"

echo
echo "### 2/2  Evaluation path (subsampled, exercises every plot) ###"
if [ "${TRAIN_STATUS}" -eq 0 ]; then
    "${PYTHON}" scripts/evaluate_model.py \
        --checkpoint "${SMOKE_DIR}/model/best_model.pt" \
        --split validation \
        --negative-ratio 2 \
        --batch-size 8 \
        --num-workers 8 \
        --output-dir "${SMOKE_DIR}/evaluation"
    EVAL_STATUS=$?
else
    echo "SKIPPED: training path failed, no checkpoint to evaluate."
    EVAL_STATUS=1
fi
echo "evaluate exit status: ${EVAL_STATUS}"

echo
echo "=========================================================================="
echo "SMOKE TEST RESULT"
echo "  training path : $( [ "${TRAIN_STATUS}" -eq 0 ] && echo PASS || echo FAIL )"
echo "  evaluation    : $( [ "${EVAL_STATUS}" -eq 0 ] && echo PASS || echo FAIL )"
echo "  artefacts     : ${SMOKE_DIR}"
find "${SMOKE_DIR}" -type f 2>/dev/null | sed 's|^|    |'
echo "Finished: $(date)"
echo "=========================================================================="

# Surface a non-zero status so the run report shows FAILED for a broken path.
[ "${TRAIN_STATUS}" -eq 0 ] && [ "${EVAL_STATUS}" -eq 0 ]
