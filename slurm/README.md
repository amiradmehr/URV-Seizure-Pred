# Running the pipeline on Unity

SLURM wrappers around the four pipeline stages, for the UMass Unity cluster.
The scripts in `scripts/` are unchanged and still run standalone; these only
supply resources, environment, and logging.

## One-time setup

```bash
# Python environment (Python 3.12, matched to requirements-lock.txt)
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install --index-url https://download.pytorch.org/whl/cu128 \
    "torch==2.11.0+cu128" "torchaudio==2.11.0+cu128"
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install "numpy==2.4.6" "scipy==1.17.1" \
    "braindecode==1.6.1" "edfio==0.4.14"
.venv/bin/python -m pip install -e .

# Bulk storage: ~330 GB total, far too much for /work.
ws_allocate -F workspace -n seizeit2 -d 30 -r 7
mkdir -p /scratch4/workspace/$USER-seizeit2/data/{raw,interim,processed}
ln -sfn /scratch4/workspace/$USER-seizeit2/data data
```

`torchaudio` **must** carry the `+cu128` local tag. The untagged 2.11.0 wheel is
built against CUDA 13 and fails to load next to a cu128 torch with
`OSError: libcudart.so.13: cannot open shared object file`. braindecode imports
torchaudio at module scope, so this breaks every stage, not just training.

The workspace expires after 30 days. Extend it with
`ws_allocate -x -F workspace -n seizeit2 -d 30`.

## Stages

Run in order; each depends on the previous one's output.

| | Script | Partition | Wall time | Produces |
|---|---|---|---|---|
| 0 | `00_download_data.sbatch` | `cpu` | ~6 min | `data/raw` (44 GiB) |
| 1 | `10_build_dataset.sbatch` | `cpu` | ~3 h | `data/interim`, `data/processed` |
| 2 | `20_validate_dataset.sbatch` | `cpu` | ~1 h | `manifests/validation_summary.csv` |
| 3 | `30_train_eegnet.sbatch` | `gpu` | hours | `outputs/models/eegnet_mean_pool/` |
| 4 | `40_evaluate.sbatch` | `gpu` | ~1 h | `outputs/evaluation/<split>/` |

```bash
sbatch slurm/00_download_data.sbatch
sbatch slurm/10_build_dataset.sbatch          # add --resume to reuse checkpoints
sbatch slurm/20_validate_dataset.sbatch
sbatch slurm/30_train_eegnet.sbatch
sbatch slurm/40_evaluate.sbatch               # --split test when selection is final
```

Extra arguments pass through to the underlying Python script, so
`sbatch slurm/30_train_eegnet.sbatch --epochs 40 --learning-rate 5e-4` works.

Logs land in `logs/<stage>-<jobid>.out`.

## Notes on the download

The full OpenNeuro dataset (`ds005873`) is 117 GiB because it also ships ECG,
EMG, and movement EDFs. This pipeline opens only `*_eeg.edf` plus the events
sidecars, so stage 0 syncs a 44 GiB subset. Integrity is checked by comparing
every downloaded file against the S3 object size.

## Notes on resources

- **Build** is single-threaded per recording and runs at ~10 MB of EDF per
  second, dominated by the zero-phase FIR filter. 2850 recordings take roughly
  1.5 h for pass 1 and another hour for scaler fitting plus shard writing.
- **Training and evaluation are I/O bound, not compute bound.** One decision is
  a 45-minute history: 540 chunks x 3 channels x 1280 samples of float32,
  or 8.3 MB read per example. The GPU jobs request 12 CPUs purely so
  `--num-workers` can keep the device fed.
- Storage lives on `/scratch4` (NAS, ~880 MB/s single stream). Keeping the repo
  and `outputs/` on `/work` and the arrays on scratch avoids the group's `/work`
  quota, which has under 200 GB free.
