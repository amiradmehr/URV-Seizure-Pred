#!/bin/bash
#SBATCH --job-name=sz2-download
#SBATCH --partition=cpu
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=logs/download-%j.out
#SBATCH --error=logs/download-%j.out
#
# Download only the SeizeIT2 files this pipeline reads: behind-the-ear EEG
# recordings, their event annotations, and the BIDS root metadata.
#
# The full OpenNeuro dataset is ~117 GiB because it also carries ECG, EMG and
# movement EDFs. This pipeline never opens those, so restricting the sync to
# *_eeg.edf plus sidecars cuts the download to ~44 GiB.

set -euo pipefail

REPO_ROOT="/work/pi_phuc_umass_edu/aradmehr/Github/URV-Seizure-Pred"
RAW_DIR="${REPO_ROOT}/data/raw"

module load awscli-v2/2.15.53

# The OpenNeuro S3 bucket is public; --no-sign-request avoids needing creds.
export AWS_REQUEST_CHECKSUM_CALCULATION=when_required
aws configure set default.s3.max_concurrent_requests 24
aws configure set default.s3.multipart_chunksize 32MB

mkdir -p "${RAW_DIR}"

echo "Destination: ${RAW_DIR}"
echo "Started: $(date)"

aws s3 sync --no-sign-request --only-show-errors \
    s3://openneuro.org/ds005873/ "${RAW_DIR}/" \
    --exclude "*" \
    --include "dataset_description.json" \
    --include "participants.tsv" \
    --include "participants.json" \
    --include "events.json" \
    --include "README" \
    --include "CHANGES" \
    --include "*/eeg/*_eeg.edf" \
    --include "*/eeg/*_eeg.json" \
    --include "*/eeg/*_events.tsv"

echo "Finished: $(date)"

echo
echo "=== Download summary ==="
echo "EEG recordings : $(find "${RAW_DIR}" -name '*_eeg.edf' | wc -l)  (expected 2850)"
echo "Event files    : $(find "${RAW_DIR}" -name '*_events.tsv' | wc -l)  (expected 2850)"
echo "Subjects       : $(find "${RAW_DIR}" -maxdepth 1 -name 'sub-*' -type d | wc -l)  (expected 125)"
echo "Total size     : $(du -sh "${RAW_DIR}" | cut -f1)"

test -f "${RAW_DIR}/dataset_description.json" \
    || { echo "ERROR: BIDS root metadata missing"; exit 1; }
