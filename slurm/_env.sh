#!/bin/bash
# Shared environment for every SLURM stage of the pipeline.
# Source this from an sbatch script: source "$(dirname "$0")/_env.sh"

REPO_ROOT="/work/pi_phuc_umass_edu/aradmehr/Github/URV-Seizure-Pred"
PYTHON="${REPO_ROOT}/.venv/bin/python"

cd "${REPO_ROOT}"

# data/ is a symlink into the scratch workspace; the raw EDFs plus the interim
# and processed arrays total roughly 330 GB, which does not belong on /work.
export SEIZEIT2_WORKSPACE="/scratch4/workspace/aradmehr_umass_edu-seizeit2"

# MNE writes its config and sample-data cache to $HOME by default.
export MNE_DATA="${SEIZEIT2_WORKSPACE}/.mne"
export XDG_CACHE_HOME="${SEIZEIT2_WORKSPACE}/.cache"
mkdir -p "${MNE_DATA}" "${XDG_CACHE_HOME}"

# Keep BLAS from oversubscribing the cores SLURM actually granted.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

export PYTHONUNBUFFERED=1

echo "=========================================================================="
echo "Job          : ${SLURM_JOB_NAME:-interactive} (${SLURM_JOB_ID:-none})"
echo "Node         : $(hostname)"
echo "CPUs         : ${SLURM_CPUS_PER_TASK:-?}   Mem: ${SLURM_MEM_PER_NODE:-?} MB"
echo "Repo         : ${REPO_ROOT}"
echo "Data (real)  : $(readlink -f "${REPO_ROOT}/data")"
echo "Started      : $(date)"
echo "=========================================================================="
