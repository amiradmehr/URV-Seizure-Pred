#!/bin/bash
#SBATCH --job-name=sz2-validate
#SBATCH --partition=cpu
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/validate-%j.out
#SBATCH --error=logs/validate-%j.out
#
# Stage 2: dataset-integrity checks. This reads every processed shard end to
# end (np.isfinite over ~140 GB), so it is I/O bound rather than compute bound.

set -euo pipefail

source /work/pi_phuc_umass_edu/aradmehr/Github/URV-Seizure-Pred/slurm/_env.sh

"${PYTHON}" scripts/validate_dataset.py

echo
echo "=========================================================================="
echo "Finished: $(date)"
echo "=========================================================================="
