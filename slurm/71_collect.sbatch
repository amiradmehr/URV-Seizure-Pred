#!/bin/bash
#SBATCH --job-name=sz2-collect
#SBATCH --partition=cpu
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/collect-%j.out
#SBATCH --error=logs/collect-%j.out
#
# Assemble every cv_metrics.json into one table and one comparison figure.
# Uses afterany so a partially finished sweep still produces a result table.

set -uo pipefail
source /work/pi_phuc_umass_edu/aradmehr/Github/URV-Seizure-Pred/slurm/_env.sh
"${PYTHON}" scripts/collect_sweep.py
