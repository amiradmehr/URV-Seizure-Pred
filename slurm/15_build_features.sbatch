#!/bin/bash
#SBATCH --job-name=sz2-feat
#SBATCH --partition=cpu
#SBATCH --array=0-39
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=logs/feat-%A_%a.out
#SBATCH --error=logs/feat-%A_%a.out
#
# Stage 1b: precompute chunk-level spectral features from the processed shards.
#
# Independent per recording, so it fans out as an array over disjoint stripes.
# Measured ~0.5 s per hour of EEG, 11,367 h total. Sized deliberately small --
# 1 CPU, 8 G, 30 min -- because backfill places small short jobs and a 2-hour
# reservation left 18 of 20 tasks pending behind a 141-job queue. Re-running is
# safe and cheap: existing feature banks are skipped.
# Total output is well under 1 GB against 78 GB of raw shards.

set -euo pipefail

source /work/pi_phuc_umass_edu/aradmehr/Github/URV-Seizure-Pred/slurm/_env.sh

"${PYTHON}" scripts/build_features.py \
    --shard-index "${SLURM_ARRAY_TASK_ID}" \
    --shard-count "${SLURM_ARRAY_TASK_COUNT}"
