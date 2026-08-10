"""Lazy PyTorch datasets for the processed streaming seizure-risk data."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from seizure_prediction.config import PreprocessingConfig


def load_decision_examples(
    processed_manifest_path: Path,
    split: str,
    negative_to_positive_ratio: float | None = None,
    seed: int = 0,
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
                "Decision metadata does not contain the configured 45-minute "
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
