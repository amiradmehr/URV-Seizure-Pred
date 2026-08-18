#!/bin/bash
#SBATCH --job-name=sz2-train
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=a100|l40s|a40
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --output=logs/train-%j.out
#SBATCH --error=logs/train-%j.out
#
# Stage 3: train the EEGNet + mean-pooling seizure-risk baseline.
#
# Every decision expands to 540 five-second chunks read lazily out of a
# memory-mapped continuous recording, so this job is as much an I/O pipeline as
# a compute one: one decision is ~8.3 MB of float32. The 12 CPUs exist to keep
# --num-workers loaders saturating the GPU.
#
# Extra flags pass straight through, e.g.:
#     sbatch slurm/30_train_eegnet.sbatch --epochs 30 --learning-rate 5e-4

set -euo pipefail

source /work/pi_phuc_umass_edu/aradmehr/Github/URV-Seizure-Pred/slurm/_env.sh

# Informational only: never let a missing/erroring nvidia-smi kill the job
# under `set -e`. Torch reports the real device on the next line anyway.
nvidia-smi || true

# Model selection uses a fixed, seeded 20:1 negative subsample of validation so
# that per-epoch scoring stays affordable. The ratio is constant across epochs,
# so epoch ranking by average precision remains valid. The final, full-
# prevalence numbers come from scripts/evaluate_model.py afterwards.
"${PYTHON}" scripts/train_eegnet_baseline.py \
    --epochs 20 \
    --batch-size 4 \
    --num-workers 10 \
    --train-negative-ratio 4 \
    --validation-negative-ratio 20 \
    --encoder-chunk-batch-size 1024 \
    --log-interval 200 \
    "$@"

echo
echo "=========================================================================="
echo "Finished: $(date)"
echo "=========================================================================="
