# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Seizure *prediction* (not detection) research on SeizeIT2 behind-the-ear wearable EEG
(OpenNeuro `ds005873`, BIDS). At each decision time the model sees a preceding **input window**
of EEG and estimates the probability of a seizure onset in the following **seizure occurrence
period**. Decisions are emitted every 60 s; each history is split into non-overlapping 5-second
chunks before temporal aggregation.

Both are swept. Input windows **{30, 15, 10, 5} minutes** cross occurrence periods
**{2, 5, 10} minutes**, giving **12 label definitions**, each tagged `w{window}_h{horizon}`
(`w30_h10`, `w5_h2`, …). Only the decision indices and their labels differ between them, so
the filtered EEG is stored **once** and shared by all twelve — see the on-disk contract.

## Commands

The project uses a local `.venv` (Python 3.14) and is **not** installed as an editable package,
so `seizure_prediction` is only importable when `src/` is on the path.

```bash
# Tests (all 40 pass; PYTHONPATH is required)
PYTHONPATH=src .venv/bin/python -m pytest -q
PYTHONPATH=src .venv/bin/python -m pytest tests/test_normalization.py -q          # one file
PYTHONPATH=src .venv/bin/python -m pytest tests/test_normalization.py -k robust   # one test

# Scripts do NOT need PYTHONPATH - each inserts src/ into sys.path itself
.venv/bin/python scripts/build_dataset.py --help
```

There is no linter, formatter, or CI configured.

### Pipeline order

```bash
python scripts/download_seizeit2.py                       # OpenNeuro -> data/seizeit2/raw
python scripts/build_dataset.py [--resume] \
    [--normalization global|patient|recording] [--statistic meanstd|robust] \
    [--windows 30 15 10 5] [--horizons 2 5 10] [--subjects 001 007]
python scripts/validate_dataset.py --window-minutes 15 --horizon-minutes 5
python scripts/train_eegnet_baseline.py --window-minutes 15 --horizon-minutes 5 --device cuda
python scripts/cache_eegnet_embeddings.py --device cuda   # required before any temporal head
python scripts/train_eegnet_multiscale_temporal.py --device cuda
python scripts/evaluate_eegnet_events.py --device cuda    # alarm-level metrics (validation only)
```

`build_dataset.py` builds **all twelve label definitions by default**; `--windows`/`--horizons`
narrow it. Every other script takes `--window-minutes`/`--horizon-minutes` (added by
`add_label_definition_arguments`) to select one, defaulting to the untagged 30/10 config.

Each EDF is opened and filtered **once**, and all twelve label sets are computed while that
recording is still in memory. Filtering dominates runtime, so twelve combinations cost about
what one does.

`scripts/set_normalization.py --mode <mode> [--statistic <s>] [--dry-run]` switches the
normalization every label definition uses. It fits a new scaler document from the shared
filtered recordings and repoints the manifests — seconds of JSON and CSV, not a rewrite of the
EEG, because standardization is applied at load time. It replaces the former
`restandardize_processed.py`. Derived caches bake in the standardization and go stale;
`--clear-derived-caches` removes them.

Everything under `scripts/inspecting/` is read-only dataset auditing (event inventories,
signal-quality audits, pre-seizure eligibility/contamination checks) and writes CSVs to
`outputs/analysis/`.

## Architecture

### Configuration is centralized and frozen

`src/seizure_prediction/config.py` defines a frozen `PreprocessingConfig` dataclass and the
module-level singleton `CONFIG`. Patient splits, sampling rate (256 Hz, never resampled),
filter band, label horizons, window/stride/chunk geometry, canonical channels, and every
generated-data directory come from it. `CONFIG.validate()` cross-checks these (e.g. history must
divide into whole chunks; the clear-data requirement must cover the input window) and is called
at the top of every script.

One config instance describes one **label definition**. `build_config(window, horizon)` returns a
validated instance via `dataclasses.replace`, tagged `w{window}_h{horizon}`;
`sweep_configurations()` returns all twelve. Scripts get theirs from
`resolve_label_definition(arguments)`. The tag nests `interim_data_dir` and `processed_data_dir`;
`unscaled_recordings_dir` and `scaler_parameters_dir` deliberately do **not** nest — they live
under `interim/_shared/` and are reused by every combination. `clear_generated_directory()`
refuses to delete anything under `_shared`.

Changing task geometry means passing different flags, not editing `CONFIG`.

### Data flow and on-disk contract

```
data/seizeit2/raw/                        BIDS EDFs
data/seizeit2/interim/
  _shared/                                NEVER tagged, NEVER cleared by a build
    unscaled_recordings/                  the ONLY full-size copy of the EEG: one filtered
                                          continuous recording per (subject, session, run):
                                          {recording_id}.npy, _channels.json,
                                          _channel_availability.json, _recording.json
    scaler_parameters/{mode}_{statistic}.json
  <tag>/                                  one per label definition, e.g. w15_h5
    decision_checkpoints/                 per-recording decision tables, resumable
    manifests/                            decision_manifest.csv, seizure_manifest.csv,
                                          processed_shard_manifest.csv
data/seizeit2/processed/<tag>/{train,validation,test}/
                                          labels and metadata ONLY - no EEG
data/seizeit2/embedding_cache/, data/seizeit2/handcrafted_feature_cache/   shared across tags
outputs/models/<run-name>/<tag>/           best_model.pt, metrics.json, *.png
outputs/evaluation/, outputs/analysis/
```

Windows are **never materialized**, and **neither is standardized EEG**. `build_dataset.py`
writes whole filtered continuous recordings in raw filtered units, plus per-decision
`history_start_sample` / `decision_end_sample` indices. `StreamingDecisionDataset`
(`datasets.py`) memory-maps the `.npy`, slices the history on demand, applies the per-channel
affine scaler, and reshapes to `(chunks, 3, 1280)`. Scaling a slice is bit-identical to slicing a
scaled recording, which is what lets one shared store serve every label definition and every
normalization mode. Overlapping decisions therefore cost almost nothing, and the embedding cache
(one embedding per 5-second chunk, per recording — also independent of window and horizon) makes
temporal-head training feasible.

### Disk footprint

The filtered EEG is 11,626 h x 256 Hz x 3 channels x float32 = **128.6 GB**, and it is stored
**once**. All twelve label definitions together add ~3 GB of manifests, labels, and metadata —
a shared-to-per-tag ratio of about 42:1. Generated total is ~132 GB, plus the 44 GB raw BIDS
tree.

This is why the standardized duplicate was removed. Writing standardized shards per tag as well
would have cost 12 x 240 GiB ~ 2.8 TiB; keeping even a single shared standardized copy alongside
the filtered one costs ~257 GB, which does not fit the machine this was built on. Storing only
filtered EEG and scaling at load time is what makes the sweep fit.

**Anything that reads `X_path` must standardize it.** The shards reference unstandardized EEG;
each `processed_shard_manifest.csv` row names the `scaler_document_path` and `scaler_key` that
apply, and `load_scaler_document()` resolves it. Skipping that step silently trains on raw
microvolts.

Every recording carries a 3-value channel availability mask. The canonical channel order is
always `(BTE_LEFT, BTE_RIGHT, CROSS_HEAD)`, but SeizeIT2 records only **two** of the three; the
missing channel is stored as zeros and the mask is fed to the classifier alongside the pooled
EEG. Any code that aggregates across channels must respect the mask.

Manifest paths are stored **relative to the project root**, so a data tree can be copied to
another machine without rewriting every row; pass `project_root` to `resolve_stored_path()` to
anchor them. That function also rewrites Windows drive-letter paths to `/mnt/<drive>/...` when
running under WSL. Use it rather than `Path()` directly.

### Normalization (`normalization.py`)

Three scopes — `global` (train-fitted, the original behavior and the only mode with a
train/test asymmetry), `patient`, `recording` — each with `meanstd` or `robust` (median/IQR)
statistics. `patient`/`recording` are fitted on the data they normalize, including validation
and test; the module docstring explains why this is subject-wise standardization rather than
leakage, and that it is non-causal. Every mode is a per-channel affine map, and because the
stored EEG is in raw filtered units the saved `(center, scale)` applies to it directly — nothing
has to be inverted out of a previous standardization. Scaler fitting reads whole recordings and
so does not depend on window or horizon, which is why one document is shared by all twelve
label definitions and cached at `_shared/scaler_parameters/{mode}_{statistic}.json`.

### Models (`models.py`)

- `BaselineEEGNet` — braindecode `EEGNet` encodes each 5 s chunk; mean pooling over the
  window's chunks deliberately discards order; pooled features + availability mask → one linear
  classifier. This is the reference model everything else is compared against.
- `EEGNetMultiScaleTemporalRiskModel` — frozen baseline encoder/classifier plus a residual head
  over causal nested embedding means, **zero-initialized** so epoch 0 reproduces the baseline
  logit exactly. Its widest scale must equal the input history, so the scale set changes with the
  window: build it with `temporal_windows_for_history()` (30 → `(1,5,15,30)`, 5 → `(1,5)`).
- `EEGNetRecencyWeightedRiskModel` — frozen baseline; the only trainable values are per-minute
  pooling logits, softmax-normalized and KL-constrained toward uniform.

`models_old.py` holds the superseded mean-pool/TCN models, still used by
`train_eegnet_temporal_tcn.py` and `train_eegnet_mean_pool_fair.py`. Do not add to it.

### Evaluation protocol (treat as load-bearing)

- Splits are **patient-level**: subjects 001–100 train, 101–112 validation, 113–125 test.
  `verify_patient_split_isolation()` enforces that no window, preprocessing fit, or model
  selection touches a held-out patient. The **test split is never used** by any script here.
- Training subsamples negatives (`--negative-to-positive-ratio`); validation is always the
  **complete split at natural prevalence**. Model selection is validation average precision.
- Because training prevalence is inflated, probabilities are prior-corrected via
  `sampling_prior_logit_correction` on the model config. Any new head must carry this through,
  or reported probabilities are meaningless.
- `event_evaluation.py` is the clinical view: sweep thresholds, merge alerts into episodes with
  a refractory period, and report seizure sensitivity against false alarms per 24 h and time in
  warning, with `bootstrap_patient_metrics` resampling **patients**, not decisions.

## Conventions

- Fully spelled-out names (`recording_id`, `negative_to_positive_ratio`, `number_of_thresholds`),
  no abbreviations; keyword arguments at call sites; a docstring on every public function.
- Validate inputs up front and raise `ValueError` with a message naming the offending column,
  path, or setting. Manifest readers check for required columns before use.
- Scripts are single-purpose, argparse-driven with long `--flags`, and default their output to a
  named `outputs/models/<run-name>/` directory. A new experiment is a new script plus (if it
  changes the model) a new config dataclass — not a flag bolted onto an existing run.
- `matplotlib.use("Agg")` before importing pyplot; runs emit `.png` summaries next to
  `metrics.json`.
- Docstrings state the *interpretation decided before running* for diagnostics (see
  `train_per_patient_loso.py`). Preserve that framing when editing them.

## Seizure eligibility

A seizure is a valid prediction target only if it has `minimum_preseizure_clear_minutes` of
continuous clear EEG before onset — no non-finite samples, no `bad` annotation, no ictal or
postictal overlap, no non-background event. This is **fixed at 30 minutes across all twelve label
definitions**, not derived from the window or horizon: a constant rule means every combination
scores the same cohort of eligible seizures, so their metrics are directly comparable. Verified
empirically — the eligible-seizure set is identical across all twelve tags.

It is not a claim that 30 minutes is intrinsically required. `validate()` enforces only that it
covers the input window, and each individual decision's own history is checked for clear data
separately in `create_labeled_prediction_decisions`, so a decision reaching back past the
verified region is dropped there regardless.
