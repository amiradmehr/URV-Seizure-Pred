"""Lazy PyTorch datasets for the processed streaming seizure-risk data."""

from __future__ import annotations

import json
import os
from pathlib import Path, PureWindowsPath

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

from seizure_prediction.seizeit2.config import PreprocessingConfig


def resolve_stored_path(
    path_value: str | Path,
    project_root: Path | None = None,
) -> Path:
    """Resolve a manifest-stored path on the current machine.

    Manifest paths are recorded relative to the project root that built
    them (see `write_standardized_shards`), so the processed data tree can
    be copied to another machine -- e.g. uploaded to Google Drive and
    unzipped under a Colab clone of this repository -- and still resolve,
    as long as the relative `data/...` layout is preserved. This also
    resolves legacy manifests written with absolute Windows paths when
    training inside WSL.
    """
    path_text = str(path_value)
    native_path = Path(path_text)
    if native_path.is_absolute():
        if native_path.exists() or os.name == "nt":
            return native_path

        windows_path = PureWindowsPath(path_text)
        if windows_path.drive:
            drive_name = windows_path.drive.rstrip(":").lower()
            return Path("/mnt") / drive_name / Path(*windows_path.parts[1:])

        return native_path

    if project_root is not None:
        return project_root / native_path

    return native_path


def load_decision_examples(
    processed_manifest_path: Path,
    split: str,
    negative_to_positive_ratio: float | None = None,
    seed: int = 0,
    project_root: Path | None = None,
) -> pd.DataFrame:
    """Load decision metadata for one split without loading EEG into memory.

    When ``negative_to_positive_ratio`` is supplied, all positive decisions
    are retained and a reproducible random subset of negatives is used. This
    controls the substantial class imbalance during baseline training while
    leaving the underlying processed recordings unchanged.

    ``project_root`` resolves the manifest's relative shard paths (see
    `resolve_stored_path`); pass `config.project_root` so this works
    regardless of the current working directory.
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
            resolve_stored_path(manifest_row.metadata_path, project_root),
            dtype=metadata_dtypes,
        )
        metadata["X_path"] = str(
            resolve_stored_path(manifest_row.X_path, project_root)
        )
        metadata["channel_availability_path"] = str(
            resolve_stored_path(
                manifest_row.channel_availability_path, project_root
            )
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


class BalancedEpochSampler(Sampler[int]):
    """Use all positives and a fresh negative subset in every training epoch.

    Negatives are drawn without replacement from one seeded permutation. This
    guarantees that consecutive epochs do not reuse a negative until the full
    negative pool has been traversed. The combined positive and negative
    indices are shuffled independently for each epoch.
    """

    def __init__(
        self,
        labels: pd.Series | np.ndarray,
        negative_to_positive_ratio: float = 1.0,
        seed: int = 0,
    ) -> None:
        label_values = np.asarray(labels, dtype=np.int64)
        if label_values.ndim != 1 or not np.isin(label_values, [0, 1]).all():
            raise ValueError("labels must be one-dimensional binary values.")
        if negative_to_positive_ratio <= 0:
            raise ValueError("negative_to_positive_ratio must be positive.")

        self.positive_indices = np.flatnonzero(label_values == 1)
        negative_indices = np.flatnonzero(label_values == 0)
        if len(self.positive_indices) == 0 or len(negative_indices) == 0:
            raise ValueError("Balanced sampling requires both classes.")

        self.negative_count = int(
            round(len(self.positive_indices) * negative_to_positive_ratio)
        )
        if self.negative_count > len(negative_indices):
            raise ValueError(
                "Requested more unique negatives per epoch than are available."
            )

        self.seed = seed
        self.epoch = 0
        self.negative_indices = np.random.default_rng(seed).permutation(
            negative_indices
        )

    def __len__(self) -> int:
        return len(self.positive_indices) + self.negative_count

    def set_epoch(self, epoch: int) -> None:
        """Select the deterministic negative slice for a zero-based epoch."""
        if epoch < 0:
            raise ValueError("epoch must be non-negative.")
        self.epoch = epoch

    def negative_indices_for_epoch(self, epoch: int) -> np.ndarray:
        """Return the negative indices used by one epoch."""
        if epoch < 0:
            raise ValueError("epoch must be non-negative.")

        start = (epoch * self.negative_count) % len(self.negative_indices)
        stop = start + self.negative_count
        if stop <= len(self.negative_indices):
            return self.negative_indices[start:stop].copy()

        wrapped_count = stop - len(self.negative_indices)
        return np.concatenate(
            [
                self.negative_indices[start:],
                self.negative_indices[:wrapped_count],
            ]
        )

    def __iter__(self):
        negative_indices = self.negative_indices_for_epoch(self.epoch)
        epoch_indices = np.concatenate(
            [self.positive_indices, negative_indices]
        )
        epoch_rng = np.random.default_rng(self.seed + self.epoch + 1)
        return iter(epoch_rng.permutation(epoch_indices).tolist())


class StreamingDecisionDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Lazily extract one configured-length decision context from continuous EEG."""

    def __init__(
        self,
        examples: pd.DataFrame,
        config: PreprocessingConfig,
    ) -> None:
        self.examples = examples.reset_index(drop=True)
        self.config = config
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
                "Decision metadata does not contain the configured "
                f"{self.config.input_window_seconds / 60.0:g}-minute "
                "history length."
            )
        if history_start < 0 or decision_end > signal.shape[1]:
            raise ValueError("Decision metadata indexes EEG outside the recording.")

        history = np.asarray(
            signal[:, history_start:decision_end],
            dtype=np.float32,
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
