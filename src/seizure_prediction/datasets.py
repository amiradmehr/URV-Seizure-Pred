"""Lazy PyTorch datasets for the processed streaming seizure-risk data."""

from __future__ import annotations

import json
import os
from pathlib import Path, PureWindowsPath

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

from seizure_prediction.config import PreprocessingConfig


def resolve_stored_path(path_value: str | Path) -> Path:
    """Resolve manifest paths written on Windows when training inside WSL."""
    path_text = str(path_value)
    native_path = Path(path_text)
    if native_path.exists() or os.name == "nt":
        return native_path

    windows_path = PureWindowsPath(path_text)
    if windows_path.drive:
        drive_name = windows_path.drive.rstrip(":").lower()
        return Path("/mnt") / drive_name / Path(*windows_path.parts[1:])

    return native_path


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
            resolve_stored_path(manifest_row.metadata_path),
            dtype=metadata_dtypes,
        )
        metadata["X_path"] = str(resolve_stored_path(manifest_row.X_path))
        metadata["channel_availability_path"] = str(
            resolve_stored_path(manifest_row.channel_availability_path)
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
        self.positive_count = len(self.positive_indices)
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


class PatientEventBalancedEpochSampler(Sampler[int]):
    """Balance patients, seizure events, lead times, and negative recordings.

    At most ``max_events_per_patient`` distinct seizures are selected from a
    patient in one epoch. Selected events rotate across epochs, and exactly
    one of an event's correlated positive decision windows is used. Negative
    quotas are distributed nearly equally across every training patient; each
    patient's negatives are interleaved across recordings before cycling.

    This deliberately changes only training exposure. The underlying dataset,
    model architecture, and complete natural-prevalence validation split stay
    unchanged.
    """

    REQUIRED_COLUMNS = {
        "label",
        "subject",
        "recording_id",
        "target_seizure_id",
    }

    def __init__(
        self,
        examples: pd.DataFrame,
        negative_to_positive_ratio: float = 10.0,
        max_events_per_patient: int = 4,
        seed: int = 0,
    ) -> None:
        missing_columns = self.REQUIRED_COLUMNS - set(examples.columns)
        if missing_columns:
            raise ValueError(
                "Patient/event-balanced sampling requires columns: "
                f"{sorted(missing_columns)}"
            )
        if negative_to_positive_ratio <= 0:
            raise ValueError("negative_to_positive_ratio must be positive.")
        if max_events_per_patient <= 0:
            raise ValueError("max_events_per_patient must be positive.")

        normalized = examples.reset_index(drop=True).copy()
        labels = normalized["label"].to_numpy(dtype=np.int64)
        if labels.ndim != 1 or not np.isin(labels, [0, 1]).all():
            raise ValueError("labels must be one-dimensional binary values.")
        normalized["subject"] = normalized["subject"].astype(str).str.zfill(3)
        normalized["recording_id"] = normalized["recording_id"].astype(str)
        positives = normalized[labels == 1]
        negatives = normalized[labels == 0]
        if positives.empty or negatives.empty:
            raise ValueError("Balanced sampling requires both classes.")
        if positives["target_seizure_id"].isna().any():
            raise ValueError("Every positive decision must name a target seizure.")
        positives = positives.copy()
        positives["target_seizure_id"] = positives[
            "target_seizure_id"
        ].astype(str)
        if (
            positives.groupby("target_seizure_id")["subject"].nunique() > 1
        ).any():
            raise ValueError("A target seizure cannot belong to multiple patients.")

        self.seed = seed
        self.epoch = 0
        self.negative_to_positive_ratio = negative_to_positive_ratio
        self.max_events_per_patient = max_events_per_patient
        rng = np.random.default_rng(seed)

        self._events_by_patient: dict[str, np.ndarray] = {}
        self._positive_indices_by_event: dict[str, np.ndarray] = {}
        for subject, subject_positives in positives.groupby("subject", sort=True):
            event_ids = subject_positives["target_seizure_id"].unique()
            self._events_by_patient[str(subject)] = rng.permutation(event_ids)
            for event_id, event_rows in subject_positives.groupby(
                "target_seizure_id",
                sort=True,
            ):
                self._positive_indices_by_event[str(event_id)] = rng.permutation(
                    event_rows.index.to_numpy(dtype=np.int64)
                )

        self.positive_count = sum(
            min(max_events_per_patient, len(event_ids))
            for event_ids in self._events_by_patient.values()
        )
        self.negative_count = int(
            round(self.positive_count * negative_to_positive_ratio)
        )
        self.unique_seizure_count = len(self._positive_indices_by_event)
        self.positive_patient_count = len(self._events_by_patient)

        self._negative_indices_by_patient: dict[str, np.ndarray] = {}
        for subject, subject_negatives in negatives.groupby("subject", sort=True):
            recording_arrays = [
                rng.permutation(recording_rows.index.to_numpy(dtype=np.int64))
                for _, recording_rows in subject_negatives.groupby(
                    "recording_id",
                    sort=True,
                )
            ]
            recording_order = rng.permutation(len(recording_arrays))
            recording_arrays = [recording_arrays[index] for index in recording_order]
            interleaved: list[int] = []
            maximum_recording_length = max(map(len, recording_arrays))
            for position in range(maximum_recording_length):
                for recording_indices in recording_arrays:
                    if position < len(recording_indices):
                        interleaved.append(int(recording_indices[position]))
            self._negative_indices_by_patient[str(subject)] = np.asarray(
                interleaved,
                dtype=np.int64,
            )

        self.negative_patient_count = len(self._negative_indices_by_patient)
        if self.negative_patient_count == 0:
            raise ValueError("No patients contain negative decisions.")
        maximum_patient_quota = int(
            np.ceil(self.negative_count / self.negative_patient_count)
        )
        undersized_patients = [
            subject
            for subject, indices in self._negative_indices_by_patient.items()
            if len(indices) < maximum_patient_quota
        ]
        if undersized_patients:
            raise ValueError(
                "Some patients do not have enough unique negatives for one "
                f"balanced epoch: {undersized_patients}"
            )

    def __len__(self) -> int:
        return self.positive_count + self.negative_count

    def set_epoch(self, epoch: int) -> None:
        """Select deterministic event, lead-time, and negative rotations."""
        if epoch < 0:
            raise ValueError("epoch must be non-negative.")
        self.epoch = epoch

    @staticmethod
    def _circular_slice(
        values: np.ndarray,
        start: int,
        count: int,
    ) -> np.ndarray:
        """Return ``count`` unique cyclic values from a longer array."""
        if count < 0 or count > len(values):
            raise ValueError("Circular slice count must fit the source array.")
        if count == 0:
            return np.empty(0, dtype=values.dtype)
        positions = (np.arange(count, dtype=np.int64) + start) % len(values)
        return values[positions]

    def selected_events_for_epoch(
        self,
        subject: str,
        epoch: int,
    ) -> np.ndarray:
        """Return the patient's distinct seizure IDs selected for one epoch."""
        if epoch < 0:
            raise ValueError("epoch must be non-negative.")
        subject_key = str(subject).zfill(3)
        if subject_key not in self._events_by_patient:
            raise KeyError(f"Unknown positive patient: {subject_key}")
        event_ids = self._events_by_patient[subject_key]
        count = min(self.max_events_per_patient, len(event_ids))
        start = (epoch * count) % len(event_ids)
        return self._circular_slice(event_ids, start, count)

    def _event_selection_count_before_epoch(
        self,
        subject: str,
        event_id: str,
        epoch: int,
    ) -> int:
        """Count prior selections so lead-time windows rotate without repeats."""
        return sum(
            event_id in self.selected_events_for_epoch(subject, earlier_epoch)
            for earlier_epoch in range(epoch)
        )

    def positive_indices_for_epoch(self, epoch: int) -> np.ndarray:
        """Return one positive decision for every selected seizure event."""
        if epoch < 0:
            raise ValueError("epoch must be non-negative.")
        selected_indices: list[int] = []
        for subject in sorted(self._events_by_patient):
            for event_id_value in self.selected_events_for_epoch(subject, epoch):
                event_id = str(event_id_value)
                event_indices = self._positive_indices_by_event[event_id]
                previous_selections = self._event_selection_count_before_epoch(
                    subject,
                    event_id,
                    epoch,
                )
                selected_indices.append(
                    int(event_indices[previous_selections % len(event_indices)])
                )
        return np.asarray(selected_indices, dtype=np.int64)

    def negative_indices_for_epoch(self, epoch: int) -> np.ndarray:
        """Return negatives distributed evenly across patients and recordings."""
        if epoch < 0:
            raise ValueError("epoch must be non-negative.")
        patients = sorted(self._negative_indices_by_patient)
        base_quota, remainder = divmod(self.negative_count, len(patients))
        remainder_start = epoch % len(patients)
        extra_patients = {
            patients[(remainder_start + offset) % len(patients)]
            for offset in range(remainder)
        }
        selected: list[np.ndarray] = []
        for subject in patients:
            quota = base_quota + int(subject in extra_patients)
            patient_indices = self._negative_indices_by_patient[subject]
            prior_count = 0
            for earlier_epoch in range(epoch):
                earlier_remainder_start = earlier_epoch % len(patients)
                earlier_extra_patients = {
                    patients[(earlier_remainder_start + offset) % len(patients)]
                    for offset in range(remainder)
                }
                prior_count += base_quota + int(
                    subject in earlier_extra_patients
                )
            start = prior_count % len(patient_indices)
            selected.append(self._circular_slice(patient_indices, start, quota))
        return np.concatenate(selected)

    def indices_for_epoch(self, epoch: int) -> np.ndarray:
        """Return the reproducibly shuffled combined epoch indices."""
        positives = self.positive_indices_for_epoch(epoch)
        negatives = self.negative_indices_for_epoch(epoch)
        combined = np.concatenate([positives, negatives])
        epoch_rng = np.random.default_rng(self.seed + epoch + 1)
        return epoch_rng.permutation(combined)

    def __iter__(self):
        return iter(self.indices_for_epoch(self.epoch).tolist())


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


def embedding_cache_path(signal_path: str | Path, cache_root: Path) -> Path:
    """Return the stable cache file for one processed continuous recording."""
    resolved_signal_path = resolve_stored_path(signal_path)
    split_name = resolved_signal_path.parent.name
    return (
        cache_root
        / split_name
        / f"{resolved_signal_path.stem}_eegnet_embeddings.npy"
    )


class CachedEmbeddingDecisionDataset(
    Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
):
    """Read 45-minute contexts from cached continuous EEGNet embeddings.

    Cache files contain one embedding per non-overlapping five-second chunk of
    a processed recording.  Decision histories therefore remain lazy and do
    not duplicate the heavily overlapping 45-minute inputs.
    """

    def __init__(
        self,
        examples: pd.DataFrame,
        config: PreprocessingConfig,
        cache_root: Path,
        embedding_dim: int,
    ) -> None:
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive.")
        self.examples = examples.reset_index(drop=True)
        self.config = config
        self.cache_root = Path(cache_root)
        self.embedding_dim = embedding_dim
        self._embedding_cache: dict[str, np.ndarray] = {}
        self._availability_cache: dict[str, np.ndarray] = {}

        self.chunk_samples = int(
            round(config.chunk_window_seconds * config.target_sfreq)
        )
        self.history_samples = int(
            round(config.input_window_seconds * config.target_sfreq)
        )
        if self.history_samples % self.chunk_samples != 0:
            raise ValueError("The configured history must divide into whole chunks.")
        self.history_chunks = self.history_samples // self.chunk_samples

    def __len__(self) -> int:
        return len(self.examples)

    def __getstate__(self) -> dict[str, object]:
        """Do not send open memory maps to DataLoader worker processes."""
        state = self.__dict__.copy()
        state["_embedding_cache"] = {}
        state["_availability_cache"] = {}
        return state

    def close(self) -> None:
        """Close cached memory maps, which is required before Windows deletion."""
        for embeddings in self._embedding_cache.values():
            memory_map = getattr(embeddings, "_mmap", None)
            if memory_map is not None:
                memory_map.close()
        self._embedding_cache.clear()
        self._availability_cache.clear()

    def __del__(self) -> None:
        """Best-effort cleanup when a loader worker releases the dataset."""
        try:
            self.close()
        except (AttributeError, OSError):
            pass

    def _load_embeddings(self, signal_path: str) -> np.ndarray:
        cache_path = embedding_cache_path(signal_path, self.cache_root)
        cache_key = str(cache_path)
        if cache_key not in self._embedding_cache:
            if not cache_path.exists():
                raise FileNotFoundError(
                    f"Missing EEGNet embedding cache: {cache_path}. "
                    "Run scripts/cache_eegnet_embeddings.py first."
                )
            embeddings = np.load(cache_path, mmap_mode="r")
            if embeddings.ndim != 2 or embeddings.shape[1] != self.embedding_dim:
                raise ValueError(
                    f"Invalid embedding cache shape in {cache_path}: "
                    f"expected (*, {self.embedding_dim}), found {embeddings.shape}."
                )
            self._embedding_cache[cache_key] = embeddings
        return self._embedding_cache[cache_key]

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
        history_start = int(row["history_start_sample"])
        decision_end = int(row["decision_end_sample"])
        if decision_end - history_start != self.history_samples:
            raise ValueError(
                "Decision metadata does not contain the configured 45-minute "
                "history length."
            )
        if (
            history_start < 0
            or history_start % self.chunk_samples != 0
            or decision_end % self.chunk_samples != 0
        ):
            raise ValueError(
                "Decision histories must align to cached five-second chunks."
            )

        embeddings = self._load_embeddings(str(row["X_path"]))
        start_chunk = history_start // self.chunk_samples
        end_chunk = decision_end // self.chunk_samples
        if end_chunk > len(embeddings):
            raise ValueError("Decision metadata indexes outside the embedding cache.")
        history_embeddings = np.asarray(
            embeddings[start_chunk:end_chunk],
            dtype=np.float32,
        )
        if history_embeddings.shape != (
            self.history_chunks,
            self.embedding_dim,
        ):
            raise ValueError(
                "Cached history has an unexpected shape: "
                f"{history_embeddings.shape}."
            )

        return (
            torch.from_numpy(
                np.ascontiguousarray(history_embeddings).copy()
            ),
            torch.from_numpy(
                self._load_availability(
                    str(row["channel_availability_path"])
                ).copy()
            ),
            torch.tensor(float(row["label"]), dtype=torch.float32),
        )
