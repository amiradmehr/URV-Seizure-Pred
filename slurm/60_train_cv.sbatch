#!/bin/bash
#SBATCH --job-name=sz2-cv
#SBATCH --partition=gpu,gpu-preempt
#SBATCH --gpus=1
#SBATCH --constraint=sm_70
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=3
#SBATCH --mem=24G
#SBATCH --output=logs/cv-%j.out
#SBATCH --error=logs/cv-%j.out
#
# Patient-wise cross-validation on the precomputed chunk features.
#
# Cheap because a decision is now 58 KB of features rather than 8.3 MB of raw
# EEG, so all five folds fit comfortably inside one short job. Kept to 3 CPUs
# and a short wall time because the gpu partition is routinely CPU-starved:
# nodes sit with idle GPUs and zero free cores.
#
#   sbatch slurm/60_train_cv.sbatch --tag spectral --architecture spectral-attention
#   sbatch slurm/60_train_cv.sbatch --tag linear   --architecture logistic-mean

set -euo pipefail
source /work/pi_phuc_umass_edu/aradmehr/Github/URV-Seizure-Pred/slurm/_env.sh
nvidia-smi || true

"${PYTHON}" scripts/train_cv.py --num-workers 2 "$@"

echo
echo "Finished: $(date)"
