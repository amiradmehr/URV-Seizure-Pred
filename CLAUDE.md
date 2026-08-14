# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

Research pipeline for **behind-the-ear (BTE) EEG seizure *prediction*** on the SeizeIT2 dataset. The task is framed as a **streaming risk decision**: at each decision time `t`, the model sees the preceding **45 minutes** of EEG and estimates the probability of a seizure *onset* within the **next 10 minutes**. There is no minimum warning horizon (`prediction_horizon_minutes = 0`), so this is a risk-now task, not a lead-time task.

Everything downstream is driven by a single frozen config object; understand it before changing timing, splits, or channels.

## Commands

Windows + PowerShell, using the checked-in `.venv`. Run all scripts from the repo root.

```bash
.venv\Scripts\python.exe -m pip install -e .
```

```bash
.venv\Scripts\python.exe scripts\build_dataset.py
```

```bash
.venv\Scripts\python.exe scripts\validate_dataset.py
```

```bash
.venv\Scripts\python.exe scripts\train_eegnet_baseline.py --epochs 1
```

- `build_dataset.py --resume` reuses validated per-recording checkpoints in `data/interim/unscaled_recordings/` and only reprocesses missing/corrupt ones. A fresh run (no flag) wipes and rebuilds them.
- The pipeline order is strict: **build → validate → train**. `train_eegnet_baseline.py` reads `data/interim/manifests/processed_shard_manifest.csv` and fails if it is missing.
- `--epochs 1` is the intended smoke test. Real runs default to 10 epochs; on CPU each epoch is slow because every decision expands to 540 chunks.
- There is **no test suite** (`pytest` is installed but no tests exist). "Validation" here means dataset-integrity checks in `validate_dataset.py`, not unit tests.

## Data & output layout (all gitignored)

- `data/raw/` — the downloaded SeizeIT2 **BIDS** dataset (must contain `dataset_description.json`). Recordings are `*_task-szMonitoring_*_eeg.edf` with sibling `*_events.tsv`.
- `data/interim/unscaled_recordings/` — filtered continuous recordings: `<rec>.npy` (2-D, `channels × samples`), `<rec>.csv` (decision metadata), `<rec>_channels.json`, `<rec>_channel_availability.json`. These are the resumable checkpoints.
- `data/interim/manifests/` — `decision_manifest.csv`, `seizure_manifest.csv`, `processed_shard_manifest.csv`, plus summary CSVs.
- `data/interim/scaler_parameters/global_channel_zscore.json` — the one global z-score.
- `data/processed/{train,validation,test}/` — final standardized shards, one set per recording (`_X.npy`, `_y.npy`, `_metadata.csv`, `_channels.json`, `_channel_availability.json`).
- `outputs/models/eegnet_mean_pool/` — `best_model.pt` (selected on best validation **average precision**), `metrics.json`, `learning_curves.png`.

## Architecture

**`src/seizure_prediction/config.py` is the contract.** A single frozen `PreprocessingConfig` dataclass (exported as `CONFIG`) defines all subjects, timing, filtering, channels, and derived directory paths. `CONFIG.validate()` enforces invariants (e.g. `minimum_preseizure_clear_minutes` must cover history + horizon + occurrence; `input_window_seconds` and `input_stride_seconds` must be integer multiples of `chunk_window_seconds`). Every script calls `CONFIG.validate()` first. Changing one timing constant usually requires adjusting others to keep validation passing.

**Storage is continuous, not pre-windowed — this is the key design decision.** Because the decision stride is only 60 s but each history is 45 min, adjacent windows overlap ~98%. Rather than materialize overlapping windows, the pipeline saves each *continuous* filtered recording once, and every decision row stores `history_start_sample` / `decision_end_sample` indices into it. `StreamingDecisionDataset` (in `datasets.py`) memory-maps the recording and slices out the 45-min history lazily in `__getitem__`, reshaping it into `(n_chunks=540, channels=3, chunk_samples=1280)`. If you touch the metadata schema or timing, keep the index columns consistent across `preprocessing.py`, `datasets.py`, and `validate_dataset.py`.

**Canonical 3-channel layout with an availability mask.** Each SeizeIT2 EDF physically contains only **2 of 3** electrode locations. `canonicalize_eeg_channels` maps them into a fixed order `(BTE_LEFT, BTE_RIGHT, CROSS_HEAD)`, zero-fills the absent one, and records a boolean availability mask. Zero-filled channels are **excluded from scaler fitting** and re-zeroed after standardization, so they are never treated as real signal. The mask is fed to the model alongside the EEG.

**No-leakage guarantees (do not weaken these):**
- Whole-**patient** holdout. Subjects 001–100 train, 101–112 validation, 113–125 test (`config.py`). `assign_patient_splits` + `verify_patient_split_isolation` guarantee no patient appears in two splits.
- The global z-score (`fit_global_channel_scaler`) is fit on **training patients only**; `validate_dataset.py` re-checks the saved scaler's provenance.
- The test split is never loaded during training.

**Eligibility & labeling.** A seizure is only a valid positive target if it has `minimum_preseizure_clear_minutes` (60 min) of continuous clean EEG before onset — no non-finite samples, no `bad*` annotations, no overlapping ictal/postictal window, and no non-background/non-seizure events (e.g. impedance checks). See `seizure_has_full_prediction_history` and `create_labeled_prediction_decisions`. Seizure **scope** (`local`/`cross`) is preserved only when the event file has a recognized scope column; otherwise it stays `unknown` and the build prints a warning. **The pipeline never invents scope labels** — do not add heuristic guesses.

**Model (`models.py`).** `EEGNetMeanPoolRiskModel` encodes each 5 s chunk with Braindecode `EEGNet`, mean-pools the 540 chunk embeddings, concatenates the 3-value availability mask, and applies a linear head → one logit. It is deliberately a simple baseline; mean pooling is the intended first thing to replace with recurrent/attention temporal aggregation.

## Gotchas

- Filtering is **zero-phase (noncausal)** FIR — fine for this offline baseline, but a causal variant will be required for any real-time/on-device deployment (noted in `filter_and_prepare`).
- The native 256 Hz rate is preserved deliberately; there is no resampling, and `filter_and_prepare` asserts the rate matches `target_sfreq`.
- Class imbalance is severe. Training subsamples negatives (`--train-negative-ratio`, default 4:1) and uses `BCEWithLogitsLoss` `pos_weight`; model selection is by validation average precision, not accuracy.

## Ad-hoc inspection scripts

`scripts/inspecting/` holds standalone one-off audit/EDA scripts (recording audits, event inventories, pre-seizure contamination checks, signal-quality plots). They are not part of the build/train pipeline and are not imported by it; run them individually when investigating the raw data.
