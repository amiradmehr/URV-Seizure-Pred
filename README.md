# URV-Seizure-Pred

Seizure *prediction* (not detection) research on SeizeIT2 behind-the-ear wearable EEG
(OpenNeuro `ds005873`, BIDS). At each decision time the model sees a preceding **input window**
of EEG and estimates the probability of a seizure onset in the following **seizure occurrence
period** (the horizon). Input windows **{30, 15, 10, 5} minutes** cross horizons **{2, 5, 10}
minutes**, giving **12 label definitions** (tagged `w{window}_h{horizon}`, e.g. `w30_h10`,
`w5_h2`). See [CLAUDE.md](CLAUDE.md) for the full architecture writeup; this file is the
practical "what do I run, in what order, and when do I need to rerun it" guide.

## Setup

The project uses a local `.venv` (Python 3.14) and is **not** installed as an editable package —
`seizure_prediction` is only importable when `src/` is on the path.

```bash
python3 -m venv .venv
.venv/bin/pip install -e .        # or: .venv/bin/pip install -r requirements.txt if you have one
```

Every script under `scripts/` inserts `src/` onto `sys.path` itself, so scripts run with plain
`.venv/bin/python scripts/whatever.py` — **no `PYTHONPATH` needed**. Tests do not have that
insertion, so they need it explicitly:

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q
```

## The pipeline, in order

Run these in this order the first time. Steps 1–2 are one-time (or rare) setup; step 3 is
optional sanity-checking; steps 4+ you'll repeat per experiment.

```bash
# 1. Download raw EDFs into data/seizeit2/raw (BIDS tree)
.venv/bin/python scripts/download_seizeit2.py

# 2. Filter + label the data for all 12 window/horizon combinations in one pass
.venv/bin/python scripts/build_dataset.py

# 3. Sanity-check one combination's manifests and shards
.venv/bin/python scripts/validate_dataset.py --window-minutes 15 --horizon-minutes 5

# 4. Train the baseline EEGNet on that combination
.venv/bin/python scripts/train_eegnet_baseline.py --window-minutes 15 --horizon-minutes 5 --device cuda

# 5. Cache per-chunk EEGNet embeddings (only needed before training a temporal head)
.venv/bin/python scripts/cache_eegnet_embeddings.py --window-minutes 15 --horizon-minutes 5 --device cuda

# 6. Train a temporal head on top of the frozen baseline
.venv/bin/python scripts/train_eegnet_multiscale_temporal.py --window-minutes 15 --horizon-minutes 5 --device cuda

# 7. Alarm-level clinical metrics (sensitivity vs. false alarms/24h), validation split only
.venv/bin/python scripts/evaluate_eegnet_events.py --window-minutes 15 --horizon-minutes 5 --device cuda
```

`--window-minutes`/`--horizon-minutes` appear on every script from step 3 onward and select which
of the 12 label definitions to use; they default to 30/10 when omitted. Step 2 only needs to run
once — it builds all 12 combinations together — so to try a different combination you repeat
steps 3+ with different flags, not step 2.

### Why one build produces 12 combinations

Each EDF is opened and filtered **once**; the filtered signal itself doesn't depend on window or
horizon, only *which decision timestamps get labeled positive/negative* does. `build_dataset.py`
keeps the filtered recording in memory and labels it for every requested window/horizon
combination before moving on, so building all 12 costs roughly the wall time and disk of building
one. The filtered EEG is written once to `data/seizeit2/interim/_shared/unscaled_recordings/`
and shared by all 12 tags; each tag only adds its own manifests and small metadata shards
(`data/seizeit2/{interim,processed}/w{window}_h{horizon}/`).

## When you need to rerun something

| If you change... | Rerun | Why |
|---|---|---|
| Nothing yet, just want a different `--window-minutes`/`--horizon-minutes` | Nothing — just pass the new flags to steps 3+ | All 12 combinations are already built by step 2 |
| Which subjects/windows/horizons are **built at all** (`--subjects`, `--windows`, `--horizons` on `build_dataset.py`) | `build_dataset.py` (use `--resume` to only fill in what's missing) | Only the requested combinations get manifests |
| Normalization mode or statistic (`--mode`/`--statistic`) | `set_normalization.py`, **not** `build_dataset.py` | Standardization is applied at load time from a small scaler document, not baked into the stored EEG — switching it is a metadata rewrite (seconds), not a full rebuild |
| Normalization, and you've already run `cache_eegnet_embeddings.py` or `cache_handcrafted_features.py` | Also rerun those caching scripts | Caches are computed from *standardized* EEG, so they go stale the moment normalization changes. Pass `--clear-derived-caches` to `set_normalization.py` to delete stale caches instead of leaving them around silently wrong |
| The baseline model's weights (retrained `train_eegnet_baseline.py`) | `cache_eegnet_embeddings.py`, then any temporal head that used the old checkpoint | The embedding cache and every temporal head are frozen on top of one baseline checkpoint; a new baseline invalidates both |
| `minimum_preseizure_clear_minutes`, canonical channels, sampling rate, or anything else in `PreprocessingConfig` | `build_dataset.py` (full rebuild, or `--resume` if only adding subjects) | These affect the filtered signal or which seizures are eligible, which the caches and manifests capture |
| Nothing about the data, just training hyperparameters (learning rate, epochs, ratio, etc.) | Just rerun the training script | Training scripts read already-built manifests; nothing upstream needs to change |

`--resume` on `build_dataset.py` reuses cached filtered recordings (they don't depend on window
or horizon) and, per combination, reuses validated decision checkpoints — it only (re)computes
what's missing or invalid for the geometry requested, rather than trusting a stale cache blindly.

## Command reference

### `download_seizeit2.py`

```bash
.venv/bin/python scripts/download_seizeit2.py
```

No flags. Downloads the SeizeIT2 BIDS dataset (EEG modality only — ECG/EMG/movement are
excluded) into `CONFIG.raw_data_dir` (`data/seizeit2/raw`), unless `dataset_description.json`
already exists there, in which case it just validates the tree and exits.

### `build_dataset.py`

```bash
.venv/bin/python scripts/build_dataset.py [OPTIONS]
```

Builds the filtered, labeled, patient-split dataset. By default this is **every** combination of
window × horizon.

| Flag | Meaning |
|---|---|
| `--resume` | Reuse validated per-combination decision checkpoints and only label recordings still missing one. Filtered recordings are always reused when present — they don't depend on window/horizon. |
| `--normalization {global,patient,recording}` | Scope of the channel z-score fit *at build time* (default `patient`). `patient` gives every patient their own center/scale so between-patient amplitude differences don't dominate the input; `global` reproduces the original single train-fitted transform. See `set_normalization.py` below to change this later without a rebuild. |
| `--statistic {meanstd,robust}` | `meanstd` is the classic mean/std z-score (default). `robust` uses median/IQR, which resists movement and electrode-pop artifacts common in wearable EEG. |
| `--windows MINUTES [MINUTES ...]` | Which input windows to build, in minutes (default: `30 15 10 5`). |
| `--horizons MINUTES [MINUTES ...]` | Which seizure occurrence periods to build, in minutes (default: `2 5 10`). |
| `--subjects ID [ID ...]` | Restrict the build to these subject IDs, without the `sub-` prefix (e.g. `001 007 042`). Omit to use the full configured split. Useful for a fast smoke test before committing to a full ~176 GB build. |

### `validate_dataset.py`

```bash
.venv/bin/python scripts/validate_dataset.py --window-minutes 15 --horizon-minutes 5
```

Checks one label definition's manifests and shards: geometry matches the config, every `X_path`
resolves into the shared filtered store, applying the recorded scaler to a sampled slice produces
standardized EEG, and every split has both classes present (or explains why not, for a
`--subjects`-restricted partial build).

| Flag | Meaning |
|---|---|
| `--window-minutes` | Input window to validate, in minutes (default 30). |
| `--horizon-minutes` | Seizure occurrence period to validate, in minutes (default 10). |

### `set_normalization.py`

```bash
.venv/bin/python scripts/set_normalization.py --mode patient --statistic robust [OPTIONS]
```

Switches which normalization every label definition standardizes with. Fits a new scaler
document from the already-stored filtered recordings and repoints manifests at it — a metadata
rewrite, not a rebuild of the 128 GB signal store. Replaces the old `restandardize_processed.py`.

| Flag | Meaning |
|---|---|
| `--mode {global,patient,recording}` | Target normalization scope. |
| `--statistic {meanstd,robust}` | `meanstd` or `robust` (median/IQR). |
| `--windows`, `--horizons` | Which built label definitions to repoint (default: every one that's been built). |
| `--dry-run` | Fit and report the scalers without writing anything. Use this first to see what would change. |
| `--clear-derived-caches` | Delete the embedding and handcrafted-feature caches, which were computed under the old standardization and are stale the instant it changes. |

### `train_eegnet_baseline.py`

```bash
.venv/bin/python scripts/train_eegnet_baseline.py --window-minutes 15 --horizon-minutes 5 --device cuda
```

Trains the reference EEGNet model: each 5-second chunk of the input window is encoded, chunk
embeddings are mean-pooled, and pooled features plus the channel-availability mask feed one
linear classifier. Training sees every positive decision plus a fresh negative sample each
epoch; validation always uses the complete, natural-prevalence, patient-held-out split; model
selection is validation average precision.

| Flag | Meaning |
|---|---|
| `--epochs`, `--batch-size`, `--learning-rate`, `--weight-decay`, `--dropout` | Standard training hyperparameters. |
| `--embedding-dim` | Size of the per-chunk EEGNet embedding. |
| `--encoder-chunk-batch-size` | How many 5-second chunks are encoded per inner batch (memory/throughput knob, doesn't change results). |
| `--negative-to-positive-ratio` | Training negatives sampled per positive each epoch (default 4:1). Validation is never subsampled. |
| `--sampling-strategy {decision-balanced,patient-event-balanced}` | `decision-balanced` (default) keeps every positive decision; `patient-event-balanced` caps events per patient and rotates one lead-time window per selected seizure, to change exposure without changing the model. |
| `--max-events-per-patient` | Cap on distinct seizures one patient contributes per epoch, under `patient-event-balanced` sampling. |
| `--gradient-accumulation-steps` | Accumulate small GPU batches into a larger effective batch. |
| `--max-grad-norm` | Gradient clipping threshold. |
| `--early-stopping-patience` | Stop after this many epochs without a better validation AP. |
| `--num-workers`, `--seed`, `--device {auto,cpu,cuda}` | Standard runtime knobs. |
| `--output-dir` | Defaults to `outputs/models/eegnet_baseline/<tag>/`. Use a distinct directory per experiment if you want to keep more than one run around. |
| `--window-minutes`, `--horizon-minutes` | Which label definition to train on (default 30/10). |

### `cache_eegnet_embeddings.py`

```bash
.venv/bin/python scripts/cache_eegnet_embeddings.py --window-minutes 15 --horizon-minutes 5 --device cuda
```

Runs the trained baseline's encoder once over every 5-second chunk of every recording and saves
the embeddings to disk (`data/seizeit2/embedding_cache/`), so temporal-head training can look up
and average cached vectors instead of re-encoding overlapping windows from raw EEG every step.
One embedding per chunk is window/horizon-independent, so this cache is shared across all 12
label definitions — build it once per baseline checkpoint. **Required before training any
temporal head.**

| Flag | Meaning |
|---|---|
| `--baseline-checkpoint` | Path to the trained baseline's `best_model.pt` to encode with. |
| `--cache-dir` | Where to write the cache (defaults under `data/seizeit2/embedding_cache/`). |
| `--splits {train,validation,test}` | Which splits to cache (space-separated). |
| `--chunk-batch-size` | How many chunks are encoded per batch. |
| `--storage-dtype {float16,float32}` | Precision the cache is stored at on disk. |
| `--amp` / `--no-amp` | CUDA mixed precision while encoding. Off by default so the cache matches float32 baseline evaluation exactly. |
| `--overwrite` | Recompute and replace an existing cache rather than skipping it. |
| `--device {auto,cpu,cuda}` | Runtime device. |
| `--window-minutes`, `--horizon-minutes` | Which label definition's decisions to cache embeddings for (the cache itself is shared, but decision coverage depends on which manifests exist). |

### `train_eegnet_multiscale_temporal.py`

```bash
.venv/bin/python scripts/train_eegnet_multiscale_temporal.py --window-minutes 15 --horizon-minutes 5 --device cuda
```

Trains a residual head on top of the frozen baseline encoder/classifier, learning causal
contrasts between nested embedding means (e.g. last 1 min vs. last 5 min vs. last 15 min vs. the
whole window). The scale set is derived from the window (`w30` → `1,5,15,30`; `w5` → `1,5`), and
the head is zero-initialized so epoch 0 exactly reproduces the baseline. Requires
`cache_eegnet_embeddings.py` to have been run first.

| Flag | Meaning |
|---|---|
| `--epochs`, `--batch-size`, `--learning-rate`, `--weight-decay` | Standard training hyperparameters. |
| `--temporal-hidden-dim`, `--temporal-dropout` | Size and regularization of the temporal head. |
| `--negative-to-positive-ratio` | Training negatives sampled per positive each epoch. |
| `--max-grad-norm` | Gradient clipping threshold. |
| `--early-stopping-patience` | Stop after this many epochs without a better validation AP. |
| `--num-workers`, `--seed`, `--device` | Standard runtime knobs. |
| `--baseline-checkpoint` | Which trained baseline to freeze and build on top of. |
| `--cache-dir` | Where the embedding cache lives. |
| `--output-dir` | Output directory for this run. |
| `--window-minutes`, `--horizon-minutes` | Which label definition to train on. |

### `train_eegnet_recency_weighted.py`

```bash
.venv/bin/python scripts/train_eegnet_recency_weighted.py --window-minutes 15 --horizon-minutes 5 --device cuda
```

Alternative temporal head: the encoder and classifier stay frozen, and the only trainable values
are one pooling logit per minute of the input window. A softmax keeps weights nonnegative and
normalized; a KL penalty toward uniform keeps the result close to the plain mean-pool baseline
unless the data clearly wants otherwise. Also requires `cache_eegnet_embeddings.py` first.

| Flag | Meaning |
|---|---|
| `--uniform-kl-weight` | Penalty for departing from uniform (mean-pool) weighting. |
| `--max-temporal-strength` | Maximum fraction of weight assigned to the learned (rather than uniform) minute weights. |
| `--softmax-temperature` | Temperature of the softmax over minute logits. |
| `--minimum-ap-improvement` | Minimum validation-AP gain required to replace the saved checkpoint (guards against noise-driven "improvements"). |
| `--negative-to-positive-ratio`, `--max-grad-norm`, `--early-stopping-patience`, `--epochs`, `--batch-size`, `--learning-rate`, `--weight-decay`, `--num-workers`, `--seed`, `--device` | Same meaning as above. |
| `--baseline-checkpoint`, `--cache-dir`, `--output-dir` | Same meaning as above. |
| `--window-minutes`, `--horizon-minutes` | Which label definition to train on. |

### `evaluate_eegnet_events.py`

```bash
.venv/bin/python scripts/evaluate_eegnet_events.py --window-minutes 15 --horizon-minutes 5 --device cuda
```

The clinical view of a trained model. Generates prior-corrected probabilities for every
validation decision, sweeps score thresholds, merges nearby alerts into episodes with a
refractory period, and reports seizure sensitivity against false alarms per 24 hours and time in
warning — with confidence intervals bootstrapped over **patients**, not decisions. **Never**
touches the held-out test split.

| Flag | Meaning |
|---|---|
| `--number-of-thresholds` | Number of score-rank thresholds swept. |
| `--refractory-minutes` | Alerts no farther apart than this merge into one alarm episode. |
| `--operating-false-alarm-budget` | The single operating point (false alarms per 24 interictal hours) to inspect in detail. |
| `--reported-false-alarm-budgets` | Additional false-alarm budgets to report sensitivity at. |
| `--target-sensitivities` | Sensitivities to report the false-alarm cost of. |
| `--bootstrap-samples`, `--seed` | Bootstrap resampling controls. |
| `--batch-size`, `--num-workers`, `--device` | Standard runtime knobs. |
| `--checkpoint` | Which trained model to evaluate (baseline or a temporal head). |
| `--cache-dir` | Embedding cache location, if the checkpoint being evaluated needs one. |
| `--output-dir` | Where to write the evaluation report. |
| `--window-minutes`, `--horizon-minutes` | Which label definition to evaluate. |

## Other scripts

These aren't part of the main pipeline above but are part of the repo:

- `cache_handcrafted_features.py` — caches AVA-style hand-engineered features per whole-minute of
  EEG (shared across label definitions, like the embedding cache). Feeds
  `train_handcrafted_feature_baseline.py`.
- `train_handcrafted_feature_baseline.py` — trains and event-evaluates a feature-only (non-EEGNet)
  risk baseline for comparison.
- `train_eegnet_mean_pool_fair.py`, `train_eegnet_temporal_tcn.py` — older mean-pool/TCN model
  variants (`models_old.py`), kept for comparison against the current models but not extended.
- `train_eegnet_patient_event_balanced.py` — trains the unchanged baseline EEGNet under
  patient/event-balanced exposure instead of the default sampling strategy.
- `train_per_patient_loso.py` — per-patient models with leave-one-recording-out validation, a
  diagnostic rather than the main evaluation path.
- `export_real_overfit_subset.py` — exports a small real (non-synthetic) subset for overfitting
  sanity checks.
- `analyze_patient_relative_psd.py` — compares preictal power spectral density against each
  patient's own interictal baseline.
- `inspecting/` — read-only dataset auditing (event inventories, signal-quality audits,
  pre-seizure eligibility/contamination checks). Writes CSVs to `outputs/analysis/`, never
  touches the generated dataset.

All of the above accept `--window-minutes`/`--horizon-minutes` where the label definition is
relevant, following the same defaults (30/10) as the main pipeline scripts.

## Tests

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q                              # all tests
PYTHONPATH=src .venv/bin/python -m pytest tests/test_normalization.py -q  # one file
PYTHONPATH=src .venv/bin/python -m pytest tests/test_normalization.py -k robust  # one test
```

There is no linter or formatter configured.
