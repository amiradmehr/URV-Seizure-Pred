"""Lazy PyTorch datasets for the processed streaming seizure-risk data."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from seizure_prediction.config import PreprocessingConfig


def densify_positive_decisions(
    examples: pd.DataFrame,
    stride_seconds: float,
    config: PreprocessingConfig,
) -> pd.DataFrame:
    """Add positive decisions between the existing ones, at a finer stride.

    Only ~2,484 training positives exist because positives are generated on the
    same 60-second grid as negatives, giving ``SOP / stride = 10`` decisions per
    seizure. Interpolating that grid multiplies the positive class without
    touching the negatives and without re-reading any annotation.

    This is safe by construction rather than by re-checking. A target seizure is
    only eligible when the full ``minimum_preseizure_clear_minutes`` (60 min)
    before its onset is continuous clean EEG. For any decision time ``t`` lying
    between two existing positives for that seizure, the 45-minute history
    ``[t - H, t]`` is a subset of that already-verified clear interval, and the
    recording-bounds check the original loop applied still holds because ``t``
    never leaves the span of decisions that passed it.

    The added windows overlap their neighbours by more than 98%, so they enlarge
    the gradient signal far more than they enlarge the effective sample size.
    """
    positives = examples[examples["label"] == 1]
    if positives.empty or "target_seizure_id" not in positives.columns:
        return examples

    stride_samples = int(round(stride_seconds * config.target_sfreq))
    history_samples = int(round(config.input_window_seconds * config.target_sfreq))
    if stride_samples <= 0:
        raise ValueError("densify stride must be positive.")

    generated: list[pd.DataFrame] = []
    for _, seizure_rows in positives.groupby("target_seizure_id", sort=False):
        ends = seizure_rows["decision_end_sample"].to_numpy(dtype=np.int64)
        first_end, last_end = int(ends.min()), int(ends.max())
        if last_end - first_end < stride_samples:
            continue

        template = seizure_rows.iloc[0]
        existing = set(ends.tolist())
        new_ends = [
            end
            for end in range(first_end, last_end + 1, stride_samples)
            if end not in existing and end - history_samples >= 0
        ]
        if not new_ends:
            continue

        block = pd.DataFrame([template] * len(new_ends)).reset_index(drop=True)
        block["decision_end_sample"] = new_ends
        block["history_start_sample"] = [e - history_samples for e in new_ends]
        block["decision_time_seconds"] = [e / config.target_sfreq for e in new_ends]
        if "prediction_start_seconds" in block.columns:
            block["prediction_start_seconds"] = block["decision_time_seconds"]
        if "prediction_stop_seconds" in block.columns:
            block["prediction_stop_seconds"] = block["decision_time_seconds"] + 60.0 * (
                config.prediction_horizon_minutes
                + config.seizure_occurrence_period_minutes
            )
        generated.append(block)

    if not generated:
        return examples
    return pd.concat([examples, *generated], ignore_index=True)


def load_decision_examples(
    processed_manifest_path: Path,
    split: str,
    negative_to_positive_ratio: float | None = None,
    seed: int = 0,
    densify_positive_stride_seconds: float | None = None,
    config: PreprocessingConfig | None = None,
) -> pd.DataFrame:
    """Load decision metadata for one split without loading EEG into memory.

    When ``negative_to_positive_ratio`` is supplied, all positive decisions
    are retained and a reproducible random subset of negatives is used. This
    controls the substantial class imbalance during baseline training while
    leaving the underlying processed recordings unchanged.
    """
    manifest = pd.read_csv(processed_manifest_path, dtype={"subject": str})
    split_manifest = manifest[manifest["split"] == split]
    if split_manifest.empty:
        raise ValueError(f"No processed shards were found for split {split!r}.")

    required_manifest_columns = {
        "X_path",
        "metadata_path",
        "channel_availability_path",
    }
    missing_columns = required_manifest_columns - set(split_manifest.columns)
    if missing_columns:
        raise ValueError(
            "Processed manifest is missing columns: "
            f"{sorted(missing_columns)}"
        )

    frames: list[pd.DataFrame] = []
    metadata_dtypes = {
        "recording_id": str,
        "subject": str,
        "session": str,
        "task": str,
        "run": str,
    }
    for manifest_row in split_manifest.itertuples(index=False):
        metadata = pd.read_csv(
            manifest_row.metadata_path,
            dtype=metadata_dtypes,
        )
        metadata["X_path"] = str(manifest_row.X_path)
        metadata["channel_availability_path"] = str(
            manifest_row.channel_availability_path
        )
        frames.append(metadata)

    examples = pd.concat(frames, ignore_index=True)
    required_metadata_columns = {
        "label",
        "history_start_sample",
        "decision_end_sample",
        "X_path",
        "channel_availability_path",
    }
    missing_columns = required_metadata_columns - set(examples.columns)
    if missing_columns:
        raise ValueError(
            "Decision metadata is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if densify_positive_stride_seconds is not None:
        if config is None:
            raise ValueError("densify_positive_stride_seconds requires config.")
        before = int((examples["label"] == 1).sum())
        examples = densify_positive_decisions(
            examples, densify_positive_stride_seconds, config
        )
        after = int((examples["label"] == 1).sum())
        print(
            f"Densified {split!r} positives {before:,} -> {after:,} "
            f"at {densify_positive_stride_seconds:g}s stride",
            flush=True,
        )

    if negative_to_positive_ratio is not None:
        if negative_to_positive_ratio <= 0:
            raise ValueError("negative_to_positive_ratio must be positive.")

        positive_examples = examples[examples["label"] == 1]
        negative_examples = examples[examples["label"] == 0]
        if positive_examples.empty:
            raise ValueError(f"Split {split!r} contains no positive decisions.")

        requested_negative_count = int(
            round(len(positive_examples) * negative_to_positive_ratio)
        )
        sampled_negatives = negative_examples.sample(
            n=min(requested_negative_count, len(negative_examples)),
            random_state=seed,
        )
        examples = pd.concat(
            [positive_examples, sampled_negatives],
            ignore_index=True,
        )

    if not examples["label"].isin([0, 1]).all():
        raise ValueError(f"Split {split!r} has labels outside {{0, 1}}.")

    return examples.sample(frac=1.0, random_state=seed).reset_index(drop=True)


class StreamingDecisionDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Lazily extract one 45-minute decision context from continuous EEG."""

    def __init__(
        self,
        examples: pd.DataFrame,
        config: PreprocessingConfig,
        window_normalize: bool = False,
    ) -> None:
        self.examples = examples.reset_index(drop=True)
        self.config = config
        # Re-standardise each 45-minute history on its own robust statistics.
        # The stored shards carry a single global z-score fitted across all
        # training patients, which leaves per-patient median sigma spanning ~26x
        # -- absolute amplitude (impedance, electrode seating) rather than
        # physiology, and the most likely thing the baseline fitted that could
        # not transfer to unseen patients.
        self.window_normalize = window_normalize
        self._signal_cache: dict[str, np.ndarray] = {}
        self._availability_cache: dict[str, np.ndarray] = {}

        self.chunk_samples = int(
            round(config.chunk_window_seconds * config.target_sfreq)
        )
        self.history_samples = int(
            round(config.input_window_seconds * config.target_sfreq)
        )
        if self.history_samples % self.chunk_samples != 0:
            raise ValueError("The configured history must divide into whole chunks.")

    def __len__(self) -> int:
        return len(self.examples)

    def __getstate__(self) -> dict[str, object]:
        """Do not send open memory maps to DataLoader worker processes."""
        state = self.__dict__.copy()
        state["_signal_cache"] = {}
        state["_availability_cache"] = {}
        state["_median_cache"] = {}
        return state

    def _load_signal(self, signal_path: str) -> np.ndarray:
        if signal_path not in self._signal_cache:
            self._signal_cache[signal_path] = np.load(signal_path, mmap_mode="r")
        return self._signal_cache[signal_path]

    def _load_availability(self, availability_path: str) -> np.ndarray:
        if availability_path not in self._availability_cache:
            with Path(availability_path).open("r", encoding="utf-8") as mask_file:
                availability = np.asarray(json.load(mask_file), dtype=np.float32)
            expected_shape = (len(self.config.canonical_channel_names),)
            if availability.shape != expected_shape or not np.isin(
                availability,
                [0.0, 1.0],
            ).all():
                raise ValueError(
                    f"Invalid channel availability mask: {availability_path}"
                )
            self._availability_cache[availability_path] = availability
        return self._availability_cache[availability_path]

    def _normalize_window(
        self,
        history: np.ndarray,
        availability: np.ndarray,
    ) -> np.ndarray:
        """Median/IQR-standardise each present channel over its own window.

        Median and IQR rather than mean and standard deviation: EEG windows
        carry movement and electrode artefacts whose amplitude dwarfs the
        signal, and those would otherwise set the scale. Absent channels are
        left at exactly zero, matching the stored convention.
        """
        present = availability > 0.5
        if not present.any():
            return history

        quartiles = np.percentile(history[present], [25, 50, 75], axis=1)
        median = quartiles[1][:, None]
        spread = (quartiles[2] - quartiles[0])[:, None]
        spread = np.maximum(spread, self.config.zscore_epsilon)

        normalized = history.copy()
        normalized[present] = (history[present] - median) / spread
        normalized[~present] = 0.0
        return normalized

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.examples.iloc[index]
        signal = self._load_signal(str(row["X_path"]))
        history_start = int(row["history_start_sample"])
        decision_end = int(row["decision_end_sample"])

        if decision_end - history_start != self.history_samples:
            raise ValueError(
                "Decision metadata does not contain the configured 45-minute "
                "history length."
            )
        if history_start < 0 or decision_end > signal.shape[1]:
            raise ValueError("Decision metadata indexes EEG outside the recording.")

        history = np.asarray(
            signal[:, history_start:decision_end],
            dtype=np.float32,
        )
        if self.window_normalize:
            history = self._normalize_window(
                history,
                self._load_availability(str(row["channel_availability_path"])),
            )

        chunks = np.ascontiguousarray(
            history.reshape(
                len(self.config.canonical_channel_names),
                -1,
                self.chunk_samples,
            ).transpose(1, 0, 2)
        )

        return (
            torch.from_numpy(chunks),
            torch.from_numpy(
                self._load_availability(str(row["channel_availability_path"])).copy()
            ),
            torch.tensor(float(row["label"]), dtype=torch.float32),
        )


class ChunkFeatureDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Serve a decision as its precomputed chunk-feature sequence.

    Reads the ``(n_chunks, n_features)`` bank produced by build_features.py and
    slices the 540 chunks covering one 45-minute history. That is 58 KB per
    decision against 8.3 MB of raw EEG, which is what turns training from an I/O
    pipeline into a compute one and makes cross-validation affordable.

    Windows are centred and scaled on their own median/IQR. In log-power space a
    per-patient amplitude gain is a constant offset, so this removes it exactly
    rather than estimating it -- see seizure_prediction.features.
    """

    def __init__(
        self,
        examples: pd.DataFrame,
        config: PreprocessingConfig,
        feature_dir: Path,
        normalize: bool = True,
        history_minutes: float | None = None,
        level_features: bool = False,
    ) -> None:
        self.examples = examples.reset_index(drop=True)
        self.config = config
        self.feature_dir = Path(feature_dir)
        self.normalize = normalize
        # Median centring removes the per-patient gain and the window level in
        # the same operation, and vigilance lives in the level (measured: window
        # normalisation costs 63% of asleep-vs-awake discrimination). Referencing
        # the level to the RECORDING's own median keeps the state while still
        # cancelling the patient gain.
        self.level_features = level_features
        self._median_cache: dict[str, np.ndarray] = {}
        # A shorter history keeps the chunks *closest to the decision*. The
        # stored window always spans the configured 45 minutes; trimming here
        # costs nothing and lets the effective context be swept without
        # rebuilding anything.
        self.history_minutes = history_minutes
        self._cache: dict[str, np.ndarray] = {}
        self._availability_cache: dict[str, np.ndarray] = {}

        self.chunk_samples = int(
            round(config.chunk_window_seconds * config.target_sfreq)
        )
        self.n_chunks = int(
            round(config.input_window_seconds / config.chunk_window_seconds)
        )
        if history_minutes is None:
            self.n_chunks_used = self.n_chunks
        else:
            self.n_chunks_used = int(
                round(history_minutes * 60.0 / config.chunk_window_seconds)
            )
            if not 1 <= self.n_chunks_used <= self.n_chunks:
                raise ValueError(
                    f"history_minutes={history_minutes} maps to "
                    f"{self.n_chunks_used} chunks, outside 1..{self.n_chunks}."
                )

    def __len__(self) -> int:
        return len(self.examples)

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state["_cache"] = {}
        state["_availability_cache"] = {}
        state["_median_cache"] = {}
        return state

    def _bank(self, recording_id: str) -> tuple[np.ndarray, np.ndarray]:
        if recording_id not in self._cache:
            path = self.feature_dir / f"{recording_id}_features.npy"
            availability_path = self.feature_dir / f"{recording_id}_availability.npy"
            self._cache[recording_id] = np.load(path, mmap_mode="r")
            self._availability_cache[recording_id] = np.load(availability_path)
        return self._cache[recording_id], self._availability_cache[recording_id]

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from seizure_prediction.features import normalize_window

        row = self.examples.iloc[index]
        bank, availability_columns = self._bank(str(row["recording_id"]))

        # Anchor on the decision instant, not on history_start_sample. The chunk
        # grid is aligned to the recording start, so the decision's chunk index
        # follows directly from its end sample. Anchoring this way makes the
        # window correct for decision sets built with a history shorter than the
        # configured 45 min; for the standard 45-min decisions it is identical,
        # since history_start_sample == decision_end_sample - 540*chunk_samples.
        stop = int(row["decision_end_sample"]) // self.chunk_samples
        start = stop - self.n_chunks_used
        if start < 0 or stop > bank.shape[0]:
            # A recording whose trailing partial chunk was dropped can fall one
            # short; pad by repeating the last chunk rather than failing.
            window = np.asarray(bank[max(start, 0) : min(stop, bank.shape[0])],
                                dtype=np.float32)
            if window.shape[0] == 0:
                window = np.zeros((self.n_chunks_used, bank.shape[1]), dtype=np.float32)
            elif window.shape[0] < self.n_chunks_used:
                pad = self.n_chunks_used - window.shape[0]
                window = np.concatenate(
                    [np.repeat(window[:1], pad, axis=0), window], axis=0
                )
        else:
            window = np.asarray(bank[start:stop], dtype=np.float32)

        if self.normalize:
            recording_median = None
            if self.level_features:
                recording_id = str(row["recording_id"])
                if recording_id not in self._median_cache:
                    self._median_cache[recording_id] = np.median(
                        np.asarray(bank, dtype=np.float32), axis=0
                    )
                recording_median = self._median_cache[recording_id]
            window = normalize_window(window, availability_columns, recording_median)

        return (
            torch.from_numpy(np.ascontiguousarray(window)),
            torch.from_numpy(availability_columns.astype(np.float32)),
            torch.tensor(float(row["label"]), dtype=torch.float32),
        )
