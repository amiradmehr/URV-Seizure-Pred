#!/bin/bash
#SBATCH --job-name=sz2-build
#SBATCH --partition=cpu
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/build-%j.out
#SBATCH --error=logs/build-%j.out
#
# Stage 1 of the pipeline: filter every recording at 256 Hz, generate the
# labeled streaming decision points, fit the train-only global z-score, and
# write the standardized shards.
#
# Pass --resume through as the first argument to reuse validated per-recording
# checkpoints in data/interim/unscaled_recordings/ instead of rebuilding them:
#     sbatch slurm/10_build_dataset.sbatch --resume
#
# Memory: the largest recording is ~27 h of 3-channel 256 Hz EEG, which MNE
# holds as float64 and copies several times during filtering. 48G leaves ample
# headroom over the ~3 GB peak.

set -euo pipefail

# sbatch runs a spool copy of this file, so $0 cannot locate the repo.
source /work/pi_phuc_umass_edu/aradmehr/Github/URV-Seizure-Pred/slurm/_env.sh

"${PYTHON}" scripts/build_dataset.py "$@"

echo
echo "=========================================================================="
echo "Finished: $(date)"
echo "Interim size  : $(du -sh data/interim 2>/dev/null | cut -f1)"
echo "Processed size: $(du -sh data/processed 2>/dev/null | cut -f1)"
echo "=========================================================================="
