"""
Reusable preprocessing functions for the SeizeIT2 dataset.

This module handles:

1. Loading BIDS-organized EDF recordings directly.
2. Reading seizure annotations.
3. Applying bandpass and notch filtering.
4. Preserving the native 256-Hz EEG sampling rate.
5. Creating indexed streaming decision points with 45-minute histories.
6. Labeling seizure risk during the following 10 minutes.
7. Preserving local/cross seizure metadata when available.
8. Splitting complete patients into train, validation, and test groups.
9. Fitting one global channel z-score transform on training patients only.
10. Applying those parameters to all splits.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import mne
import numpy as np
import pandas as pd
from seizure_prediction.seizeit2.config import PreprocessingConfig


# ----------------------------------------------------------------------
# Dataset loading
# ----------------------------------------------------------------------


@dataclass
class RawBIDSRecording:
    """One raw EDF recording and the BIDS entities used by this pipeline."""

    edf_path: Path
    description: dict[str, str]
    preload: bool = False

    def load_raw(self) -> mne.io.BaseRaw:
        """Open the EDF only when its turn in preprocessing is reached."""
        return mne.io.read_raw_edf(
            self.edf_path,
            preload=self.preload,
            verbose="ERROR",
        )


def load_bids_recordings(
    config: PreprocessingConfig,
    subjects: Iterable[str] | None = None,
    preload: bool = False,
) -> list[RawBIDSRecording]:
    """
    Load EDF recordings without converting BIDS events to MNE annotations.

    Parameters
    ----------
    config:
        Pipeline configuration.
    subjects:
        Subject IDs without the ``sub-`` prefix. For example:
        ``["001", "002"]``.
    preload:
        Whether MNE should preload each EDF recording.

    Returns
    -------
    list[RawBIDSRecording]
        EDF recordings with BIDS entity dictionaries.
    """
    dataset_root = config.raw_data_dir

    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Dataset directory does not exist: {dataset_root}"
        )

    description_path = dataset_root / "dataset_description.json"

    if not description_path.exists():
        raise FileNotFoundError(
            f"No dataset_description.json found in {dataset_root}. "
            "The path does not appear to be a BIDS root."
        )

    requested_subjects = None if subjects is None else set(subjects)
    recordings: list[RawBIDSRecording] = []

    for edf_path in sorted(dataset_root.rglob("*_eeg.edf")):
        entities = {
            part.split("-", maxsplit=1)[0]: part.split("-", maxsplit=1)[1]
            for part in edf_path.stem.split("_")
            if "-" in part
        }
        subject = entities.get("sub", "")
        task = entities.get("task", "")

        if requested_subjects is not None and subject not in requested_subjects:
            continue

        if task != config.bids_task:
            continue

        recordings.append(
            RawBIDSRecording(
                edf_path=edf_path,
                description={
                    "subject": subject,
                    "session": entities.get("ses", ""),
                    "task": task,
                    "run": entities.get("run", ""),
                },
                preload=preload,
            )
        )

    if not recordings:
        raise RuntimeError(
            "No matching EDF recordings were found. Check the dataset path, "
            "subjects, and BIDS task name."
        )

    return recordings


# ----------------------------------------------------------------------
# BIDS entity handling
# ----------------------------------------------------------------------


def normalize_entity(value: Any) -> str:
    """
    Convert a BIDS entity value to a consistent string.

    This prevents values such as numeric run 1 from becoming ``1.0``.
    """
    if value is None:
        return ""

    if isinstance(value, float) and np.isnan(value):
        return ""

    if isinstance(value, (int, np.integer)):
        return str(int(value))

    if isinstance(value, float) and value.is_integer():
        return str(int(value))

    return str(value)


def extract_recording_entities(
    description: pd.Series | dict[str, Any],
) -> dict[str, str]:
    """
    Extract subject, session, task, and run from a recording description.
    """
    if isinstance(description, pd.Series):
        description_dict = description.to_dict()
    else:
        description_dict = dict(description)

    return {
        "subject": normalize_entity(description_dict.get("subject")),
        "session": normalize_entity(description_dict.get("session")),
        "task": normalize_entity(description_dict.get("task")),
        "run": normalize_entity(description_dict.get("run")),
    }


def numeric_entity_sort_key(value: str) -> tuple[int, str]:
    """
    Create a stable sort key for BIDS session or run values.

    Numeric entities are sorted numerically. Non-numeric entities are
    sorted lexicographically afterward.
    """
    value = normalize_entity(value)

    if value.isdigit():
        return int(value), ""

    match = re.search(r"\d+", value)

    if match:
        return int(match.group()), value

    return 10**9, value


def recording_id_from_entities(entities: dict[str, str]) -> str:
    """Create a filesystem-safe recording identifier."""
    components = [f"sub-{entities['subject']}"]

    if entities["session"]:
        components.append(f"ses-{entities['session']}")

    if entities["task"]:
        components.append(f"task-{entities['task']}")

    if entities["run"]:
        components.append(f"run-{entities['run']}")

    return "_".join(components)


# ----------------------------------------------------------------------
# Event-file loading
# ----------------------------------------------------------------------


def find_events_file(
    dataset_root: Path,
    entities: dict[str, str],
) -> Path:
    """
    Find the events.tsv file matching one EEG recording.
    """
    subject = entities["subject"]
    session = entities["session"]
    task = entities["task"]
    run = entities["run"]

    if not subject:
        raise ValueError("Recording description has no subject value.")

    search_root = dataset_root / f"sub-{subject}"

    if session:
        search_root = search_root / f"ses-{session}"

    search_root = search_root / "eeg"

    if not search_root.exists():
        raise FileNotFoundError(
            f"EEG directory does not exist: {search_root}"
        )

    filename_parts = [f"sub-{subject}"]

    if session:
        filename_parts.append(f"ses-{session}")

    if task:
        filename_parts.append(f"task-{task}")

    if run:
        filename_parts.append(f"run-{run}")

    exact_name = "_".join(filename_parts) + "_events.tsv"
    exact_path = search_root / exact_name

    if exact_path.exists():
        return exact_path

    # Fall back to matching the recording entities in case the dataset has
    # additional BIDS filename entities.
    candidates = list(search_root.glob("*_events.tsv"))

    matching_candidates: list[Path] = []

    for candidate in candidates:
        name = candidate.name

        required_parts = [f"sub-{subject}"]

        if session:
            required_parts.append(f"ses-{session}")

        if task:
            required_parts.append(f"task-{task}")

        if run:
            required_parts.append(f"run-{run}")

        if all(part in name for part in required_parts):
            matching_candidates.append(candidate)

    if len(matching_candidates) == 1:
        return matching_candidates[0]

    if not matching_candidates:
        raise FileNotFoundError(
            "No matching events.tsv file found for "
            f"{recording_id_from_entities(entities)}."
        )

    raise RuntimeError(
        "Multiple matching events.tsv files found for "
        f"{recording_id_from_entities(entities)}: "
        f"{matching_candidates}"
    )


def is_seizure_event(event_type: Any) -> bool:
    """
    Return True when an eventType value represents a seizure.

    SeizeIT2 seizure event names begin with ``sz_``.
    """
    if not isinstance(event_type, str):
        return False

    return event_type.strip().lower().startswith("sz_")


def find_scope_column(events: pd.DataFrame) -> str | None:
    """
    Find a possible local/cross seizure-scope column.

    The pipeline does not invent scope labels. It preserves them only when
    a recognized column exists in the event file.
    """
    normalized_columns = {
        column.lower().replace("_", "").replace("-", ""): column
        for column in events.columns
    }

    candidates = (
        "seizurescope",
        "scope",
        "eventscope",
        "annotationscope",
    )

    for candidate in candidates:
        if candidate in normalized_columns:
            return normalized_columns[candidate]

    return None


def normalize_seizure_scope(value: Any) -> str:
    """
    Normalize known local/cross scope values.

    Unknown values remain ``unknown`` so that unsupported labels are not
    silently invented.
    """
    if not isinstance(value, str):
        return "unknown"

    normalized = value.strip().lower()

    local_values = {
        "local",
        "localized",
        "focal-local",
        "channel-local",
    }

    cross_values = {
        "cross",
        "cross-channel",
        "crosschannel",
        "generalized-across-channels",
    }

    if normalized in local_values:
        return "local"

    if normalized in cross_values:
        return "cross"

    return "unknown"


def read_seizure_events(
    events_path: Path,
    entities: dict[str, str],
) -> pd.DataFrame:
    """
    Read and standardize seizure annotations from one events.tsv file.

    Returns one row per seizure.
    """
    events = pd.read_csv(events_path, sep="\t")

    required_columns = {"onset", "duration", "eventType"}
    missing_columns = required_columns - set(events.columns)

    if missing_columns:
        raise ValueError(
            f"{events_path} is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    seizures = events[
        events["eventType"].map(is_seizure_event)
    ].copy()

    standardized_columns = [
        "seizure_id",
        "subject",
        "session",
        "task",
        "run",
        "onset_seconds",
        "duration_seconds",
        "event_type",
        "seizure_scope",
        "lateralization",
        "localization",
        "vigilance",
        "events_path",
    ]

    if seizures.empty:
        return pd.DataFrame(columns=standardized_columns)

    scope_column = find_scope_column(events)

    rows: list[dict[str, Any]] = []

    recording_id = recording_id_from_entities(entities)

    for event_row_index, event in seizures.iterrows():
        if scope_column is None:
            seizure_scope = "unknown"
        else:
            seizure_scope = normalize_seizure_scope(
                event.get(scope_column)
            )

        seizure_id = (
            f"{recording_id}_seizure-{int(event_row_index):04d}"
        )

        rows.append(
            {
                "seizure_id": seizure_id,
                "subject": entities["subject"],
                "session": entities["session"],
                "task": entities["task"],
                "run": entities["run"],
                "onset_seconds": float(event["onset"]),
                "duration_seconds": float(event["duration"]),
                "event_type": str(event["eventType"]),
                "seizure_scope": seizure_scope,
                "lateralization": event.get(
                    "lateralization",
                    np.nan,
                ),
                "localization": event.get(
                    "localization",
                    np.nan,
                ),
                "vigilance": event.get(
                    "vigilance",
                    np.nan,
                ),
                "events_path": str(events_path),
            }
        )

    return pd.DataFrame(rows, columns=standardized_columns)


def read_recording_events(events_path: Path) -> pd.DataFrame:
    """Read valid event intervals used for signal-history contamination checks."""
    events = pd.read_csv(events_path, sep="\t")
    required_columns = {"onset", "duration", "eventType"}
    missing_columns = required_columns - set(events.columns)

    if missing_columns:
        raise ValueError(
            f"{events_path} is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    result = events.loc[:, ["onset", "duration", "eventType"]].copy()
    result["onset"] = pd.to_numeric(result["onset"], errors="coerce")
    result["duration"] = pd.to_numeric(result["duration"], errors="coerce")
    result["eventType"] = result["eventType"].fillna("").astype(str)

    # A few source files contain impossible negative-duration impedance
    # rows. They cannot describe an interval and are ignored here; seizure
    # annotations are read independently above.
    return result.loc[
        result["onset"].notna()
        & result["duration"].notna()
        & result["duration"].ge(0)
    ].reset_index(drop=True)


# ----------------------------------------------------------------------
# Signal preprocessing
# ----------------------------------------------------------------------


def select_eeg_channels(raw: mne.io.BaseRaw) -> list[str]:
    """Return channel names marked as EEG."""
    channel_types = raw.get_channel_types()

    eeg_channels = [
        channel_name
        for channel_name, channel_type in zip(
            raw.ch_names,
            channel_types,
            strict=True,
        )
        if channel_type == "eeg"
    ]

    if not eeg_channels:
        raise ValueError(
            "The recording contains no channels marked as EEG."
        )

    return eeg_channels


def infer_bte_side(raw: mne.io.BaseRaw) -> str:
    """Return the side of the behind-the-ear electrode in one recording."""
    channel_names = set(raw.ch_names)

    if {"BTEleft SD", "BTEright SD"}.issubset(channel_names):
        return "bilateral"

    if "BTEleft SD" in channel_names:
        return "left"

    if "BTEright SD" in channel_names:
        return "right"

    raise ValueError(
        "Expected either BTEleft SD or BTEright SD in the recording. "
        f"Found channels: {raw.ch_names}"
    )


def channel_availability_mask(
    raw: mne.io.BaseRaw,
    config: PreprocessingConfig,
) -> np.ndarray:
    """Return the canonical-electrode availability mask for one EDF."""
    raw_name_by_canonical_name = {
        "BTE_LEFT": "BTEleft SD",
        "BTE_RIGHT": "BTEright SD",
        "CROSS_HEAD": "CROSStop SD",
    }
    if set(config.canonical_channel_names) != set(raw_name_by_canonical_name):
        raise ValueError(
            "Canonical channel names do not match the supported SeizeIT2 "
            "electrode locations."
        )

    return np.asarray(
        [
            raw_name_by_canonical_name[channel_name] in raw.ch_names
            for channel_name in config.canonical_channel_names
        ],
        dtype=bool,
    )


def canonicalize_eeg_channels(
    raw: mne.io.BaseRaw,
    config: PreprocessingConfig,
) -> mne.io.BaseRaw:
    """
    Put available SeizeIT2 signals into a stable three-electrode model order.

    Each EDF has two physical signals, drawn from left behind-ear, right
    behind-ear, and cross-head locations.  An all-zero placeholder is added
    only for the absent location.  Its availability is saved separately and
    excluded from scaler fitting, so it is never treated as EEG data.
    """
    expected_raw_names = {
        "BTEleft SD",
        "BTEright SD",
        "CROSStop SD",
    }
    channel_names = list(raw.ch_names)
    unexpected_channels = set(channel_names) - expected_raw_names

    if unexpected_channels:
        raise ValueError(
            "Unexpected EEG channel(s) in SeizeIT2 recording: "
            f"{sorted(unexpected_channels)}"
        )

    if len(channel_names) != 2:
        raise ValueError(
            "Expected exactly two available SeizeIT2 EEG channels, "
            f"channel, found: {channel_names}"
        )

    processed = raw.copy()
    raw_to_canonical = {
        "BTEleft SD": "BTE_LEFT",
        "BTEright SD": "BTE_RIGHT",
        "CROSStop SD": "CROSS_HEAD",
    }
    processed.rename_channels(
        {name: raw_to_canonical[name] for name in channel_names}
    )

    for canonical_name in config.canonical_channel_names:
        if canonical_name in processed.ch_names:
            continue

        missing_channel = mne.io.RawArray(
            np.zeros((1, processed.n_times), dtype=np.float64),
            mne.create_info(
                [canonical_name],
                sfreq=processed.info["sfreq"],
                ch_types=["eeg"],
            ),
            verbose=False,
        )
        processed.add_channels([missing_channel], force_update_info=True)

    processed.reorder_channels(list(config.canonical_channel_names))

    return processed


def filter_and_prepare(
    raw: mne.io.BaseRaw,
    config: PreprocessingConfig,
) -> mne.io.BaseRaw:
    """
    Apply the agreed offline preprocessing pipeline at the native rate.

    Order
    -----
    1. Keep all available EEG channels and put them in canonical order.
    2. Bandpass filter from 0.5 to 40 Hz.
    3. Apply a 50-Hz notch filter.
    4. Preserve the native 256-Hz sampling rate.

    Notes
    -----
    Zero-phase filters are appropriate for the offline research baseline,
    but they are noncausal. A causal implementation will be needed for
    eventual real-time microcontroller deployment.
    """
    processed = raw.copy().load_data()

    eeg_channels = select_eeg_channels(processed)

    processed.pick(eeg_channels)
    processed = canonicalize_eeg_channels(processed, config)

    processed.filter(
        l_freq=config.bandpass_low_hz,
        h_freq=config.bandpass_high_hz,
        picks="eeg",
        method="fir",
        phase="zero",
        fir_design="firwin",
        skip_by_annotation=("edge", "bad_acq_skip"),
        verbose=False,
    )

    processed.notch_filter(
        freqs=[config.notch_frequency_hz],
        picks="eeg",
        method="fir",
        phase="zero",
        fir_design="firwin",
        verbose=False,
    )

    if not np.isclose(processed.info["sfreq"], config.target_sfreq):
        raise ValueError(
            "Recording sampling frequency does not match the configured "
            f"native rate: expected {config.target_sfreq} Hz, found "
            f"{processed.info['sfreq']} Hz."
        )

    return processed


# ----------------------------------------------------------------------
# Window labeling
# ----------------------------------------------------------------------


def intervals_overlap(
    first_start: float,
    first_stop: float,
    second_start: float,
    second_stop: float,
) -> bool:
    """Return whether two half-open time intervals overlap."""
    return (
        first_start < second_stop
        and first_stop > second_start
    )


def label_window(
    window_start_seconds: float,
    window_stop_seconds: float,
    seizure_events: pd.DataFrame,
    config: PreprocessingConfig,
) -> tuple[int | None, str, str | None, str]:
    """
    Label one candidate EEG window.

    Returns
    -------
    label:
        ``1`` for preictal, ``0`` for interictal, and ``None`` when the
        window must be excluded.
    label_name:
        Human-readable label.
    target_seizure_id:
        Associated seizure ID for preictal windows.
    seizure_scope:
        ``local``, ``cross``, ``unknown``, or ``none``.

    Positive interval
    -----------------
    For a 60-minute preictal window and 30-minute horizon:

        onset - 90 minutes <= window <= onset - 30 minutes

    Excluded intervals
    ------------------
    Windows are excluded when they overlap:

    - the 30-minute prediction horizon;
    - the seizure itself;
    - the configured postictal period.
    """
    preictal_seconds = (
        config.preictal_window_minutes * 60.0
    )
    horizon_seconds = (
        config.prediction_horizon_minutes * 60.0
    )
    postictal_seconds = (
        config.postictal_exclusion_minutes * 60.0
    )

    for _, seizure in seizure_events.iterrows():
        onset = float(seizure["onset_seconds"])
        duration = float(seizure["duration_seconds"])

        preictal_start = (
            onset
            - horizon_seconds
            - preictal_seconds
        )
        preictal_stop = onset - horizon_seconds

        seizure_stop = onset + duration
        exclusion_stop = seizure_stop + postictal_seconds

        # Require the entire model window to lie inside the positive
        # interval. This avoids ambiguous boundary windows.
        if (
            window_start_seconds >= preictal_start
            and window_stop_seconds <= preictal_stop
        ):
            return (
                1,
                "preictal",
                str(seizure["seizure_id"]),
                str(seizure["seizure_scope"]),
            )

        # Exclude prediction-horizon, ictal, and postictal windows.
        if intervals_overlap(
            window_start_seconds,
            window_stop_seconds,
            preictal_stop,
            exclusion_stop,
        ):
            return (
                None,
                "excluded_near_seizure",
                None,
                "none",
            )

    return 0, "interictal", None, "none"


# ----------------------------------------------------------------------
# Window generation
# ----------------------------------------------------------------------


def create_labeled_windows(
    raw: mne.io.BaseRaw,
    seizure_events: pd.DataFrame,
    entities: dict[str, str],
    bte_side: str,
    config: PreprocessingConfig,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Convert one continuous EEG recording into labeled fixed-size windows.

    Returns
    -------
    X:
        EEG tensor with shape:

            (n_windows, n_channels, n_samples)

    metadata:
        One row per EEG window.

    bte_side:
        Recorded behind-the-ear configuration: ``left``, ``right``, or
        ``bilateral``.
    """
    sampling_frequency = float(raw.info["sfreq"])

    window_samples = int(
        round(
            config.input_window_seconds
            * sampling_frequency
        )
    )

    stride_samples = int(
        round(
            config.input_stride_seconds
            * sampling_frequency
        )
    )

    if window_samples <= 0:
        raise ValueError("Window sample count must be positive.")

    if stride_samples <= 0:
        raise ValueError("Stride sample count must be positive.")

    expected_samples = int(
        round(
            config.input_window_seconds
            * config.target_sfreq
        )
    )

    if window_samples != expected_samples:
        raise ValueError(
            f"Expected {expected_samples} samples per window, "
            f"but the recording produced {window_samples}."
        )

    recording_id = recording_id_from_entities(entities)

    windows: list[np.ndarray] = []
    metadata_rows: list[dict[str, Any]] = []

    start_sample = 0
    candidate_window_index = 0

    while start_sample + window_samples <= raw.n_times:
        stop_sample = start_sample + window_samples

        window_start_seconds = start_sample / sampling_frequency
        window_stop_seconds = stop_sample / sampling_frequency

        (
            label,
            label_name,
            target_seizure_id,
            seizure_scope,
        ) = label_window(
            window_start_seconds=window_start_seconds,
            window_stop_seconds=window_stop_seconds,
            seizure_events=seizure_events,
            config=config,
        )

        if label is not None:
            window_data = raw.get_data(
                start=start_sample,
                stop=stop_sample,
            ).astype(config.signal_dtype, copy=False)

            if window_data.shape[-1] != expected_samples:
                raise RuntimeError(
                    "Generated window has an unexpected sample count: "
                    f"{window_data.shape}"
                )

            if np.isfinite(window_data).all():
                saved_window_index = len(windows)
                windows.append(window_data)

                metadata_rows.append(
                    {
                        "recording_id": recording_id,
                        "window_index_in_shard": saved_window_index,
                        "candidate_window_index": (
                            candidate_window_index
                        ),
                        "subject": entities["subject"],
                        "session": entities["session"],
                        "task": entities["task"],
                        "run": entities["run"],
                        "bte_side": bte_side,
                        "window_start_seconds": (
                            window_start_seconds
                        ),
                        "window_stop_seconds": (
                            window_stop_seconds
                        ),
                        "label": int(label),
                        "label_name": label_name,
                        "target_seizure_id": target_seizure_id,
                        "seizure_scope": seizure_scope,
                    }
                )

        start_sample += stride_samples
        candidate_window_index += 1

    metadata = pd.DataFrame(metadata_rows)

    if not windows:
        empty_shape = (
            0,
            len(raw.ch_names),
            expected_samples,
        )

        return (
            np.empty(
                empty_shape,
                dtype=config.signal_dtype,
            ),
            metadata,
        )

    return np.stack(windows), metadata


def interval_has_bad_annotation(
    raw: mne.io.BaseRaw,
    start_seconds: float,
    stop_seconds: float,
) -> bool:
    """Return whether an interval overlaps a BIDS/MNE bad-data annotation."""
    for onset, duration, description in zip(
        raw.annotations.onset,
        raw.annotations.duration,
        raw.annotations.description,
        strict=True,
    ):
        if not str(description).lower().startswith("bad"):
            continue

        if intervals_overlap(
            start_seconds,
            stop_seconds,
            float(onset),
            float(onset + duration),
        ):
            return True

    return False


def seizure_has_full_prediction_history(
    seizure: pd.Series,
    recording: mne.io.BaseRaw,
    nonfinite_prefix: np.ndarray,
    seizure_events: pd.DataFrame,
    recording_events: pd.DataFrame,
    config: PreprocessingConfig,
) -> bool:
    """Require the configured clear EEG period before a target seizure."""
    sampling_frequency = float(recording.info["sfreq"])
    onset_seconds = float(seizure["onset_seconds"])
    required_history_seconds = (
        config.minimum_preseizure_clear_minutes * 60.0
    )
    start_seconds = onset_seconds - required_history_seconds

    if start_seconds < 0:
        return False

    start_sample = int(round(start_seconds * sampling_frequency))
    stop_sample = int(round(onset_seconds * sampling_frequency))

    if stop_sample > recording.n_times:
        return False

    if nonfinite_prefix[stop_sample] != nonfinite_prefix[start_sample]:
        return False

    if interval_has_bad_annotation(
        recording,
        start_seconds,
        onset_seconds,
    ):
        return False

    if interval_overlaps_nonbackground_event(
        start_seconds,
        onset_seconds,
        recording_events,
    ):
        return False

    return not interval_overlaps_ictal_or_postictal(
        start_seconds,
        onset_seconds,
        seizure_events,
        config,
    )


def interval_overlaps_ictal_or_postictal(
    start_seconds: float,
    stop_seconds: float,
    seizure_events: pd.DataFrame,
    config: PreprocessingConfig,
) -> bool:
    """Return whether an EEG history contains ictal or postictal signal."""
    postictal_seconds = config.postictal_exclusion_minutes * 60.0

    for _, seizure in seizure_events.iterrows():
        seizure_start = float(seizure["onset_seconds"])
        seizure_stop = seizure_start + float(seizure["duration_seconds"])

        if intervals_overlap(
            start_seconds,
            stop_seconds,
            seizure_start,
            seizure_stop + postictal_seconds,
        ):
            return True

    return False


def interval_overlaps_nonbackground_event(
    start_seconds: float,
    stop_seconds: float,
    recording_events: pd.DataFrame,
) -> bool:
    """Exclude impedance and other non-background, non-seizure events."""
    for _, event in recording_events.iterrows():
        event_type = str(event["eventType"]).strip().lower()

        if event_type == "bckg" or event_type.startswith("sz_"):
            continue

        event_start = float(event["onset"])
        event_stop = event_start + float(event["duration"])

        if intervals_overlap(
            start_seconds,
            stop_seconds,
            event_start,
            event_stop,
        ):
            return True

    return False


def label_prediction_decision(
    decision_time_seconds: float,
    seizure_events: pd.DataFrame,
    eligible_seizure_ids: set[str],
    config: PreprocessingConfig,
) -> tuple[int | None, str, str | None, str]:
    """Label one streaming decision for seizure onset in the next 10 minutes."""
    horizon_seconds = config.prediction_horizon_minutes * 60.0
    occurrence_seconds = config.seizure_occurrence_period_minutes * 60.0
    postictal_seconds = config.postictal_exclusion_minutes * 60.0
    target_start = decision_time_seconds + horizon_seconds
    target_stop = target_start + occurrence_seconds

    for _, seizure in seizure_events.iterrows():
        onset = float(seizure["onset_seconds"])
        seizure_stop = onset + float(seizure["duration_seconds"])

        if onset <= decision_time_seconds < seizure_stop + postictal_seconds:
            return None, "excluded_ictal_or_postictal", None, "none"

        if decision_time_seconds < onset < target_start:
            return None, "excluded_inside_prediction_horizon", None, "none"

        if decision_time_seconds < onset and target_start <= onset <= target_stop:
            seizure_id = str(seizure["seizure_id"])

            if seizure_id not in eligible_seizure_ids:
                return None, "excluded_ineligible_target_seizure", None, "none"

            return (
                1,
                "seizure_within_10m",
                seizure_id,
                str(seizure["seizure_scope"]),
            )

    return 0, "no_seizure_within_10m", None, "none"


def create_labeled_prediction_decisions(
    raw: mne.io.BaseRaw,
    seizure_events: pd.DataFrame,
    recording_events: pd.DataFrame,
    entities: dict[str, str],
    bte_side: str,
    config: PreprocessingConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create valid streaming decision points without duplicating EEG history."""
    sampling_frequency = float(raw.info["sfreq"])
    history_samples = int(round(config.input_window_seconds * sampling_frequency))
    stride_samples = int(round(config.input_stride_seconds * sampling_frequency))
    chunk_samples = int(
        round(config.chunk_window_seconds * sampling_frequency)
    )
    lookahead_samples = int(
        round(
            sampling_frequency
            * 60.0
            * (
                config.prediction_horizon_minutes
                + config.seizure_occurrence_period_minutes
            )
        )
    )

    if history_samples <= 0 or stride_samples <= 0 or chunk_samples <= 0:
        raise ValueError("Decision history, stride, and chunk size must be positive.")

    if history_samples % chunk_samples != 0:
        raise ValueError(
            "Decision history must contain an integer number of EEG chunks."
        )

    finite_by_sample = np.isfinite(raw.get_data()).all(axis=0)
    nonfinite_prefix = np.concatenate(
        [np.array([0], dtype=np.int64), np.cumsum(~finite_by_sample)]
    )

    seizure_metadata = seizure_events.copy()

    if seizure_metadata.empty:
        seizure_metadata["eligible_for_prediction"] = pd.Series(dtype=bool)
    else:
        seizure_metadata["eligible_for_prediction"] = seizure_metadata.apply(
            seizure_has_full_prediction_history,
            axis=1,
            recording=raw,
            nonfinite_prefix=nonfinite_prefix,
            seizure_events=seizure_events,
            recording_events=recording_events,
            config=config,
        )

    eligible_seizure_ids = set(
        seizure_metadata.loc[
            seizure_metadata["eligible_for_prediction"], "seizure_id"
        ].astype(str)
    )
    recording_id = recording_id_from_entities(entities)
    rows: list[dict[str, Any]] = []
    decision_end_sample = history_samples
    decision_index = 0

    while decision_end_sample + lookahead_samples <= raw.n_times:
        history_start_sample = decision_end_sample - history_samples
        history_start_seconds = history_start_sample / sampling_frequency
        decision_time_seconds = decision_end_sample / sampling_frequency

        history_is_clear = (
            nonfinite_prefix[decision_end_sample]
            == nonfinite_prefix[history_start_sample]
            and not interval_has_bad_annotation(
                raw,
                history_start_seconds,
                decision_time_seconds,
            )
            and not interval_overlaps_ictal_or_postictal(
                history_start_seconds,
                decision_time_seconds,
                seizure_events,
                config,
            )
            and not interval_overlaps_nonbackground_event(
                history_start_seconds,
                decision_time_seconds,
                recording_events,
            )
        )

        if history_is_clear:
            label, label_name, target_seizure_id, seizure_scope = (
                label_prediction_decision(
                    decision_time_seconds,
                    seizure_events,
                    eligible_seizure_ids,
                    config,
                )
            )

            if label is not None:
                rows.append(
                    {
                        "recording_id": recording_id,
                        "decision_index_in_shard": len(rows),
                        "candidate_decision_index": decision_index,
                        "subject": entities["subject"],
                        "session": entities["session"],
                        "task": entities["task"],
                        "run": entities["run"],
                        "bte_side": bte_side,
                        "history_start_sample": history_start_sample,
                        "decision_end_sample": decision_end_sample,
                        "chunk_samples": chunk_samples,
                        "chunks_per_history": history_samples // chunk_samples,
                        "history_start_seconds": history_start_seconds,
                        "decision_time_seconds": decision_time_seconds,
                        "prediction_start_seconds": (
                            decision_time_seconds
                            + config.prediction_horizon_minutes * 60.0
                        ),
                        "prediction_stop_seconds": (
                            decision_time_seconds
                            + 60.0
                            * (
                                config.prediction_horizon_minutes
                                + config.seizure_occurrence_period_minutes
                            )
                        ),
                        "label": int(label),
                        "label_name": label_name,
                        "target_seizure_id": target_seizure_id,
                        "seizure_scope": seizure_scope,
                    }
                )

        decision_end_sample += stride_samples
        decision_index += 1

    return pd.DataFrame(rows), seizure_metadata


# ----------------------------------------------------------------------
# Continuous recording storage
# ----------------------------------------------------------------------


def save_unscaled_recording(
    signal: np.ndarray,
    decision_metadata: pd.DataFrame,
    channel_names: list[str],
    channel_availability: np.ndarray,
    output_directory: Path,
    recording_id: str,
) -> tuple[Path, Path]:
    """Save one filtered continuous recording and its decision-time metadata."""
    output_directory.mkdir(parents=True, exist_ok=True)

    array_path = output_directory / f"{recording_id}.npy"
    metadata_path = output_directory / f"{recording_id}.csv"
    channels_path = output_directory / f"{recording_id}_channels.json"
    availability_path = output_directory / f"{recording_id}_channel_availability.json"

    if signal.ndim != 2:
        raise ValueError(
            "Continuous recording signal must have shape (channels, samples), "
            f"found {signal.shape}."
        )

    availability = np.asarray(channel_availability, dtype=bool)
    if availability.shape != (signal.shape[0],):
        raise ValueError(
            "Channel availability must contain one value per signal channel; "
            f"found shape {availability.shape} for {signal.shape[0]} channels."
        )

    if not availability.any():
        raise ValueError("A recording must contain at least one available EEG channel.")

    np.save(array_path, signal)

    decision_metadata.to_csv(
        metadata_path,
        index=False,
    )

    with channels_path.open(
        "w",
        encoding="utf-8",
    ) as channel_file:
        json.dump(channel_names, channel_file, indent=2)

    with availability_path.open("w", encoding="utf-8") as availability_file:
        json.dump(availability.astype(int).tolist(), availability_file, indent=2)

    return array_path, metadata_path


def save_filtered_recording_cache(
    raw: mne.io.BaseRaw,
    bte_side: str,
    channel_availability: np.ndarray,
    output_directory: Path,
    recording_id: str,
    output_dtype: str = "float32",
) -> None:
    """Cache one filtered recording so every window/horizon combo can reuse it.

    Filtering (bandpass, notch, channel canonicalization) never depends on
    the window/horizon configuration, so this cache is shared across every
    combination in a sweep instead of being recomputed for each one.

    `raw.get_data()` returns MNE's internal float64 representation
    regardless of the recording's original bit depth; it is downcast to
    `output_dtype` (float32 everywhere else in this pipeline, including the
    final standardized shards) before saving, halving this cache's disk
    footprint with no precision loss beyond what standardization already
    accepts.
    """
    output_directory.mkdir(parents=True, exist_ok=True)

    array_path = output_directory / f"{recording_id}.npy"
    channels_path = output_directory / f"{recording_id}_channels.json"
    availability_path = output_directory / f"{recording_id}_channel_availability.json"
    bte_side_path = output_directory / f"{recording_id}_bte_side.json"
    annotations_path = output_directory / f"{recording_id}_annotations.json"

    np.save(array_path, raw.get_data().astype(output_dtype, copy=False))

    with channels_path.open("w", encoding="utf-8") as channel_file:
        json.dump(list(raw.ch_names), channel_file, indent=2)

    availability = np.asarray(channel_availability, dtype=bool)
    with availability_path.open("w", encoding="utf-8") as availability_file:
        json.dump(availability.astype(int).tolist(), availability_file, indent=2)

    with bte_side_path.open("w", encoding="utf-8") as bte_side_file:
        json.dump(bte_side, bte_side_file)

    with annotations_path.open("w", encoding="utf-8") as annotations_file:
        json.dump(
            {
                "onset": raw.annotations.onset.tolist(),
                "duration": raw.annotations.duration.tolist(),
                "description": list(raw.annotations.description),
            },
            annotations_file,
            indent=2,
        )


def load_filtered_recording_cache(
    recording_id: str,
    directory: Path,
    config: PreprocessingConfig,
) -> tuple[mne.io.BaseRaw, str, np.ndarray] | None:
    """Return a cached filtered recording, or None if none is cached yet.

    Reconstructs an object equivalent to `filter_and_prepare`'s return value
    (same signal, channel order, sampling rate, and annotations) directly
    from the cache, so the caller can skip re-reading and re-filtering the
    raw EDF entirely. A corrupt or incomplete cache entry is discarded so it
    gets rebuilt rather than silently used.
    """
    array_path = directory / f"{recording_id}.npy"
    channels_path = directory / f"{recording_id}_channels.json"
    availability_path = directory / f"{recording_id}_channel_availability.json"
    bte_side_path = directory / f"{recording_id}_bte_side.json"
    annotations_path = directory / f"{recording_id}_annotations.json"
    paths = (array_path, channels_path, availability_path, bte_side_path, annotations_path)

    if not any(path.exists() for path in paths):
        return None

    try:
        if not all(path.exists() for path in paths):
            raise ValueError("one or more required cache files are missing")

        signal = np.load(array_path)

        with channels_path.open("r", encoding="utf-8") as channel_file:
            channel_names = json.load(channel_file)
        if channel_names != list(config.canonical_channel_names):
            raise ValueError(f"unexpected channel layout {channel_names}")

        if signal.ndim != 2 or signal.shape[0] != len(config.canonical_channel_names):
            raise ValueError(f"unexpected signal shape {signal.shape}")
        if signal.shape[1] <= 0:
            raise ValueError("recording contains no samples")

        with availability_path.open("r", encoding="utf-8") as availability_file:
            availability = np.asarray(json.load(availability_file), dtype=bool)
        if availability.shape != (len(config.canonical_channel_names),):
            raise ValueError(f"invalid availability mask shape {availability.shape}")

        with bte_side_path.open("r", encoding="utf-8") as bte_side_file:
            bte_side = json.load(bte_side_file)

        with annotations_path.open("r", encoding="utf-8") as annotations_file:
            annotations_data = json.load(annotations_file)

        raw = mne.io.RawArray(
            signal,
            mne.create_info(
                channel_names,
                sfreq=config.target_sfreq,
                ch_types="eeg",
            ),
            verbose=False,
        )
        raw.set_annotations(
            mne.Annotations(
                onset=annotations_data["onset"],
                duration=annotations_data["duration"],
                description=annotations_data["description"],
            )
        )

        return raw, bte_side, availability
    except (ValueError, KeyError, json.JSONDecodeError, EOFError):
        for path in paths:
            path.unlink(missing_ok=True)
        return None


def load_channel_names_for_shard(
    array_path: Path,
) -> list[str]:
    """Load channel names associated with one shard."""
    channels_path = (
        array_path.parent
        / f"{array_path.stem}_channels.json"
    )

    if not channels_path.exists():
        raise FileNotFoundError(
            f"Missing channel file: {channels_path}"
        )

    with channels_path.open(
        "r",
        encoding="utf-8",
    ) as channel_file:
        channel_names = json.load(channel_file)

    return [str(name) for name in channel_names]


def load_channel_availability_for_shard(array_path: Path) -> np.ndarray:
    """Load the canonical-electrode availability mask for one recording."""
    availability_path = (
        array_path.parent / f"{array_path.stem}_channel_availability.json"
    )
    if not availability_path.exists():
        raise FileNotFoundError(
            f"Missing channel-availability file: {availability_path}"
        )

    with availability_path.open("r", encoding="utf-8") as availability_file:
        availability = np.asarray(json.load(availability_file), dtype=np.int8)

    if not np.isin(availability, [0, 1]).all():
        raise ValueError(
            f"Channel availability must contain only 0/1 values: {availability_path}"
        )

    return availability.astype(bool)


# ----------------------------------------------------------------------
# Chronological splitting
# ----------------------------------------------------------------------


def add_sort_columns(metadata: pd.DataFrame) -> pd.DataFrame:
    """Add stable numeric sort columns for sessions and runs."""
    result = metadata.copy()

    result["_session_numeric"] = result["session"].map(
        lambda value: numeric_entity_sort_key(
            normalize_entity(value)
        )[0]
    )

    result["_session_text"] = result["session"].map(
        lambda value: numeric_entity_sort_key(
            normalize_entity(value)
        )[1]
    )

    result["_run_numeric"] = result["run"].map(
        lambda value: numeric_entity_sort_key(
            normalize_entity(value)
        )[0]
    )

    result["_run_text"] = result["run"].map(
        lambda value: numeric_entity_sort_key(
            normalize_entity(value)
        )[1]
    )

    return result


def assign_chronological_patient_splits(
    metadata: pd.DataFrame,
    config: PreprocessingConfig,
) -> pd.DataFrame:
    """
    Split each patient's windows chronologically without shuffling.

    The earliest 60% of retained windows become training data, the next
    20% become validation data, and the latest 20% become test data.
    """
    required_columns = {
        "subject",
        "session",
        "run",
        "window_start_seconds",
        "recording_id",
        "window_index_in_shard",
    }

    missing_columns = required_columns - set(metadata.columns)

    if missing_columns:
        raise ValueError(
            "Metadata is missing columns required for chronological "
            f"splitting: {sorted(missing_columns)}"
        )

    sortable = add_sort_columns(metadata)

    sortable["split"] = ""

    for subject, subject_indices in sortable.groupby(
        "subject",
        sort=True,
    ).groups.items():
        subject_metadata = sortable.loc[
            subject_indices
        ].sort_values(
            [
                "_session_numeric",
                "_session_text",
                "_run_numeric",
                "_run_text",
                "window_start_seconds",
            ],
            kind="stable",
        )

        ordered_indices = subject_metadata.index.to_numpy()

        number_of_windows = len(ordered_indices)

        if number_of_windows < 3:
            raise ValueError(
                f"Subject {subject} has only {number_of_windows} "
                "retained windows and cannot support three splits."
            )

        train_end = int(
            np.floor(
                number_of_windows
                * config.train_fraction
            )
        )

        validation_end = int(
            np.floor(
                number_of_windows
                * (
                    config.train_fraction
                    + config.validation_fraction
                )
            )
        )

        train_end = max(1, train_end)
        validation_end = max(
            train_end + 1,
            validation_end,
        )

        validation_end = min(
            validation_end,
            number_of_windows - 1,
        )

        sortable.loc[
            ordered_indices[:train_end],
            "split",
        ] = "train"

        sortable.loc[
            ordered_indices[
                train_end:validation_end
            ],
            "split",
        ] = "validation"

        sortable.loc[
            ordered_indices[validation_end:],
            "split",
        ] = "test"

    if sortable["split"].eq("").any():
        raise RuntimeError(
            "Some windows were not assigned to a split."
        )

    return sortable.drop(
        columns=[
            "_session_numeric",
            "_session_text",
            "_run_numeric",
            "_run_text",
        ]
    )


def assign_patient_splits(
    metadata: pd.DataFrame,
    config: PreprocessingConfig,
) -> pd.DataFrame:
    """
    Assign every decision point for a patient to that patient's configured split.

    This prevents the same patient's EEG from appearing in more than one
    split, which is required for an unseen-patient evaluation.
    """
    required_columns = {"subject", "recording_id"}
    missing_columns = required_columns - set(metadata.columns)

    if missing_columns:
        raise ValueError(
            "Metadata is missing columns required for patient-level "
            f"splitting: {sorted(missing_columns)}"
        )

    result = metadata.copy()
    result["subject"] = result["subject"].map(normalize_entity)
    result["split"] = result["subject"].map(config.subject_split_map)

    unassigned_subjects = sorted(
        result.loc[result["split"].isna(), "subject"].unique().tolist()
    )

    if unassigned_subjects:
        raise ValueError(
            "Found windows for subjects absent from the configured patient "
            f"split: {unassigned_subjects}"
        )

    verify_patient_split_isolation(result, config)
    return result


def verify_patient_split_isolation(
    metadata: pd.DataFrame,
    config: PreprocessingConfig,
) -> None:
    """Verify every observed patient belongs to one expected split only."""
    required_columns = {"subject", "split"}
    missing_columns = required_columns - set(metadata.columns)

    if missing_columns:
        raise ValueError(
            "Metadata is missing patient split columns: "
            f"{sorted(missing_columns)}"
        )

    expected_splits = config.subject_split_map

    for subject, subject_metadata in metadata.groupby("subject", sort=True):
        normalized_subject = normalize_entity(subject)
        observed_splits = set(subject_metadata["split"].dropna())

        if len(observed_splits) != 1:
            raise ValueError(
                f"Subject {normalized_subject} appears in multiple splits: "
                f"{sorted(observed_splits)}"
            )

        expected_split = expected_splits.get(normalized_subject)

        if expected_split is None:
            raise ValueError(
                f"Subject {normalized_subject} is not in the configured split."
            )

        if observed_splits != {expected_split}:
            raise ValueError(
                f"Subject {normalized_subject} was assigned to "
                f"{sorted(observed_splits)} instead of {expected_split}."
            )


def verify_no_shuffling(
    metadata: pd.DataFrame,
) -> None:
    """
    Verify that split order never moves backward within a patient.
    """
    split_rank = {
        "train": 0,
        "validation": 1,
        "test": 2,
    }

    sortable = add_sort_columns(metadata)

    for subject, subject_metadata in sortable.groupby(
        "subject",
        sort=True,
    ):
        ordered = subject_metadata.sort_values(
            [
                "_session_numeric",
                "_session_text",
                "_run_numeric",
                "_run_text",
                "window_start_seconds",
            ],
            kind="stable",
        )

        ranks = ordered["split"].map(
            split_rank
        ).to_numpy()

        if np.any(ranks[1:] < ranks[:-1]):
            raise ValueError(
                f"Non-chronological split found for subject {subject}."
            )


# ----------------------------------------------------------------------
# Train-only patient/channel z-score fitting
# ----------------------------------------------------------------------


def fit_patient_channel_scalers(
    metadata: pd.DataFrame,
    unscaled_windows_dir: Path,
    epsilon: float,
) -> dict[str, dict[str, Any]]:
    """
    Fit patient/channel z-score parameters using training windows only.

    The computation is performed incrementally so all EEG windows do not
    need to be loaded simultaneously.
    """
    required_columns = {
        "subject",
        "recording_id",
        "window_index_in_shard",
        "split",
    }

    missing_columns = required_columns - set(metadata.columns)

    if missing_columns:
        raise ValueError(
            "Metadata is missing scaler columns: "
            f"{sorted(missing_columns)}"
        )

    accumulators: dict[str, dict[str, Any]] = {}

    train_metadata = metadata[
        metadata["split"] == "train"
    ]

    if train_metadata.empty:
        raise ValueError("There are no training windows.")

    for recording_id, recording_rows in train_metadata.groupby(
        "recording_id",
        sort=False,
    ):
        array_path = (
            unscaled_windows_dir
            / f"{recording_id}.npy"
        )

        if not array_path.exists():
            raise FileNotFoundError(
                f"Missing unscaled shard: {array_path}"
            )

        X = np.load(
            array_path,
            mmap_mode="r",
        )

        channel_names = load_channel_names_for_shard(
            array_path
        )

        row_indices = recording_rows[
            "window_index_in_shard"
        ].to_numpy(dtype=np.int64)

        selected = np.asarray(
            X[row_indices],
            dtype=np.float64,
        )

        subject = normalize_entity(
            recording_rows["subject"].iloc[0]
        )

        if selected.ndim != 3:
            raise ValueError(
                f"Expected 3-D EEG data for {recording_id}, "
                f"received shape {selected.shape}."
            )

        channel_count = selected.shape[1]

        if len(channel_names) != channel_count:
            raise ValueError(
                f"Channel-name count does not match the data for "
                f"{recording_id}."
            )

        if subject not in accumulators:
            accumulators[subject] = {
                "channel_names": channel_names,
                "sum": np.zeros(
                    channel_count,
                    dtype=np.float64,
                ),
                "sum_squares": np.zeros(
                    channel_count,
                    dtype=np.float64,
                ),
                "count": np.zeros(
                    channel_count,
                    dtype=np.int64,
                ),
            }

        accumulator = accumulators[subject]

        if accumulator["channel_names"] != channel_names:
            raise ValueError(
                f"Channel order changed between recordings for "
                f"subject {subject}.\n"
                f"Expected: {accumulator['channel_names']}\n"
                f"Found: {channel_names}"
            )

        accumulator["sum"] += selected.sum(
            axis=(0, 2),
            dtype=np.float64,
        )

        accumulator["sum_squares"] += np.square(
            selected,
            dtype=np.float64,
        ).sum(
            axis=(0, 2),
            dtype=np.float64,
        )

        samples_per_channel = (
            selected.shape[0]
            * selected.shape[2]
        )

        accumulator["count"] += samples_per_channel

    scalers: dict[str, dict[str, Any]] = {}

    for subject, accumulator in accumulators.items():
        count = accumulator["count"].astype(
            np.float64
        )

        if np.any(count <= 0):
            raise ValueError(
                f"Subject {subject} has an empty scaler channel."
            )

        mean = accumulator["sum"] / count

        variance = (
            accumulator["sum_squares"] / count
            - np.square(mean)
        )

        # Numerical roundoff can produce tiny negative values.
        variance = np.maximum(variance, 0.0)

        standard_deviation = np.sqrt(variance)

        standard_deviation = np.maximum(
            standard_deviation,
            epsilon,
        )

        scalers[subject] = {
            "channel_names": accumulator["channel_names"],
            "mean": mean.tolist(),
            "std": standard_deviation.tolist(),
            "training_samples_per_channel": (
                accumulator["count"].tolist()
            ),
        }

    return scalers


def save_scaler(
    scaler: dict[str, Any],
    output_path: Path,
) -> None:
    """Save global train-fitted scaler parameters as JSON."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as scaler_file:
        json.dump(
            scaler,
            scaler_file,
            indent=2,
        )


def apply_patient_channel_zscore(
    X: np.ndarray,
    subject: str,
    channel_names: list[str],
    scalers: dict[str, dict[str, Any]],
    output_dtype: str,
) -> np.ndarray:
    """
    Apply train-fitted patient/channel z-score parameters.
    """
    subject = normalize_entity(subject)

    if subject not in scalers:
        raise KeyError(
            f"No training scaler exists for subject {subject}."
        )

    scaler = scalers[subject]

    expected_channels = scaler["channel_names"]

    if expected_channels != channel_names:
        raise ValueError(
            f"Channel order does not match the fitted scaler for "
            f"subject {subject}.\n"
            f"Expected: {expected_channels}\n"
            f"Found: {channel_names}"
        )

    means = np.asarray(
        scaler["mean"],
        dtype=np.float32,
    )[None, :, None]

    standard_deviations = np.asarray(
        scaler["std"],
        dtype=np.float32,
    )[None, :, None]

    standardized = (
        np.asarray(X, dtype=np.float32)
        - means
    ) / standard_deviations

    return standardized.astype(
        output_dtype,
        copy=False,
    )


# ----------------------------------------------------------------------
# Global train-only channel z-score fitting
# ----------------------------------------------------------------------


def subject_from_recording_id(recording_id: str) -> str:
    """Extract the subject entity from a `sub-{subject}_...` recording ID."""
    match = re.match(r"^sub-(?P<subject>[^_]+)", recording_id)
    if match is None:
        raise ValueError(f"Could not parse a subject from recording ID: {recording_id!r}")
    return normalize_entity(match.group("subject"))


def fit_global_channel_scaler(
    filtered_recordings_dir: Path,
    epsilon: float,
    config: PreprocessingConfig,
) -> dict[str, Any]:
    """Fit one per-channel z-score transform from training patients only.

    Sourced from every recording in the shared `filtered_recordings_dir`
    cache that belongs to a training subject -- NOT from decision metadata
    -- so the fitted scaler depends only on filtering (bandpass, notch,
    channel canonicalization) and the configured patient split, never on
    window/horizon. A recording contributes here as long as it was
    successfully filtered, regardless of whether any particular
    window/horizon combination happens to retain a usable decision point
    from it. This is what lets every combination in a window/horizon sweep
    fit the exact same scaler and safely share one standardized-recording
    cache (see `write_standardized_shards`) instead of each combination
    needing its own.
    """
    array_paths = sorted(filtered_recordings_dir.glob("*.npy"))

    if not array_paths:
        raise ValueError(
            f"No filtered recordings found in {filtered_recordings_dir}."
        )

    train_subjects = set(config.train_subjects)
    channel_count = len(config.canonical_channel_names)
    signal_sum = np.zeros(channel_count, dtype=np.float64)
    signal_sum_squares = np.zeros(channel_count, dtype=np.float64)
    sample_count = np.zeros(channel_count, dtype=np.int64)
    observed_train_subjects: set[str] = set()

    for array_path in array_paths:
        recording_id = array_path.stem
        subject = subject_from_recording_id(recording_id)

        if subject not in train_subjects:
            continue

        observed_train_subjects.add(subject)

        X = np.load(array_path, mmap_mode="r")
        channel_names = load_channel_names_for_shard(array_path)
        channel_availability = load_channel_availability_for_shard(array_path)

        if tuple(channel_names) != config.canonical_channel_names:
            raise ValueError(
                f"Unexpected canonical channels for {recording_id}. "
                f"Expected {list(config.canonical_channel_names)}, "
                f"found {channel_names}."
            )

        selected = np.asarray(X, dtype=np.float64)

        if selected.ndim != 2 or selected.shape[0] != channel_count:
            raise ValueError(
                f"Expected shape ({channel_count}, samples) for "
                f"{recording_id}, found {selected.shape}."
            )

        if channel_availability.shape != (channel_count,):
            raise ValueError(
                f"Expected {channel_count} channel-availability values for "
                f"{recording_id}, found {channel_availability.shape}."
            )

        present_channels = np.flatnonzero(channel_availability)
        signal_sum[present_channels] += selected[present_channels].sum(
            axis=1,
            dtype=np.float64,
        )
        signal_sum_squares[present_channels] += np.square(
            selected[present_channels],
            dtype=np.float64,
        ).sum(axis=1, dtype=np.float64)
        sample_count[present_channels] += selected.shape[1]

    if not observed_train_subjects:
        raise ValueError(
            "No filtered recordings belong to a configured training subject."
        )

    if np.any(sample_count <= 0):
        raise ValueError("At least one global scaler channel has no samples.")

    mean = signal_sum / sample_count
    variance = signal_sum_squares / sample_count - np.square(mean)
    standard_deviation = np.maximum(np.sqrt(np.maximum(variance, 0.0)), epsilon)

    return {
        "channel_names": list(config.canonical_channel_names),
        "mean": mean.tolist(),
        "std": standard_deviation.tolist(),
        "training_subjects": sorted(observed_train_subjects),
        "training_samples_per_channel": sample_count.tolist(),
        "missing_channels_excluded_from_fit": True,
    }


def apply_global_channel_zscore(
    X: np.ndarray,
    channel_names: list[str],
    channel_availability: np.ndarray,
    scaler: dict[str, Any],
    output_dtype: str,
) -> np.ndarray:
    """Apply the single global, train-fitted z-score transform."""
    expected_channels = scaler["channel_names"]

    if channel_names != expected_channels:
        raise ValueError(
            "Channel order does not match the global fitted scaler. "
            f"Expected: {expected_channels}; found: {channel_names}"
        )

    values = np.asarray(X, dtype=np.float32)
    availability = np.asarray(channel_availability, dtype=bool)

    if availability.shape != (len(channel_names),):
        raise ValueError(
            "Channel availability must contain one value per canonical channel."
        )

    if values.ndim == 2:
        means = np.asarray(scaler["mean"], dtype=np.float32)[:, None]
        standard_deviations = np.asarray(scaler["std"], dtype=np.float32)[:, None]
    elif values.ndim == 3:
        means = np.asarray(scaler["mean"], dtype=np.float32)[None, :, None]
        standard_deviations = np.asarray(scaler["std"], dtype=np.float32)[
            None, :, None
        ]
    else:
        raise ValueError(
            "Global scaling expects 2-D continuous EEG or 3-D windowed EEG, "
            f"found shape {values.shape}."
        )

    standardized = (values - means) / standard_deviations
    if values.ndim == 2:
        standardized[~availability, :] = 0.0
    else:
        standardized[:, ~availability, :] = 0.0
    return standardized.astype(output_dtype, copy=False)


# ----------------------------------------------------------------------
# Final standardized shard creation
# ----------------------------------------------------------------------


def write_standardized_shards(
    metadata: pd.DataFrame,
    unscaled_recordings_dir: Path,
    processed_data_dir: Path,
    scaler: dict[str, Any],
    output_dtype: str,
    project_root: Path,
    standardized_recordings_dir: Path,
) -> pd.DataFrame:
    """
    Standardize every continuous recording and save its decision metadata.

    Each recording belongs to exactly one patient-level split. Each
    recording's unscaled checkpoint under `unscaled_recordings_dir` is
    deleted as soon as its standardized shard has been written, so peak
    disk use does not require both copies of the full dataset at once.
    A `--resume` build interrupted during this pass will need to refilter
    whichever recordings had already been consumed here.

    The manifest records every shard path relative to `project_root`
    (rather than absolute) so the processed data directory tree can be
    copied to a different machine -- e.g. uploaded to Google Drive and
    unzipped under a Colab clone of this repository -- and still resolve,
    as long as scripts are run from the repository root as documented.

    The standardized continuous signal for a recording depends only on the
    filtered EEG and the fitted `scaler` -- never on the window/horizon
    configuration, since `fit_global_channel_scaler` sources training
    recordings from the filtered-recording cache rather than from decision
    metadata -- so it is written once per recording into
    `standardized_recordings_dir` and reused by every combination in a
    sweep (they all fit the identical scaler) instead of duplicating that
    far larger array under each combination's own `processed_data_dir`.
    Only the small per-decision labels/metadata, which do depend on
    window/horizon, are written per combination.
    """
    manifest_rows: list[dict[str, Any]] = []

    for recording_id, recording_metadata in metadata.groupby(
        "recording_id",
        sort=False,
    ):
        array_path = (
            unscaled_recordings_dir
            / f"{recording_id}.npy"
        )

        X = np.load(
            array_path,
            mmap_mode="r",
        )

        channel_names = load_channel_names_for_shard(
            array_path
        )
        channel_availability = load_channel_availability_for_shard(array_path)

        subject = normalize_entity(
            recording_metadata["subject"].iloc[0]
        )

        if recording_metadata["split"].nunique() != 1:
            raise ValueError(
                f"Recording {recording_id} appears in multiple patient splits."
            )

        for split_name in (
            "train",
            "validation",
            "test",
        ):
            split_rows = recording_metadata[
                recording_metadata["split"]
                == split_name
            ].copy()

            if split_rows.empty:
                continue

            standardized_recordings_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            X_path = (
                standardized_recordings_dir
                / f"{recording_id}_X.npy"
            )

            if not X_path.exists():
                X_split_unscaled = np.asarray(
                    X,
                    dtype=np.float32,
                )

                X_split = apply_global_channel_zscore(
                    X=X_split_unscaled,
                    channel_names=channel_names,
                    channel_availability=channel_availability,
                    scaler=scaler,
                    output_dtype=output_dtype,
                )

                np.save(X_path, X_split)
            # else: another combination in this sweep already produced the
            # identical standardized signal for this recording (same scaler
            # fingerprint) -- reuse it instead of recomputing and rewriting
            # a multi-hundred-MB array.

            y_split = split_rows[
                "label"
            ].to_numpy(dtype=np.int64)

            split_rows = split_rows.reset_index(
                drop=True
            )

            split_rows[
                "decision_index_in_processed_shard"
            ] = np.arange(
                len(split_rows),
                dtype=np.int64,
            )

            shard_name = (
                f"{recording_id}_{split_name}"
            )

            split_directory = (
                processed_data_dir
                / split_name
            )

            split_directory.mkdir(
                parents=True,
                exist_ok=True,
            )

            y_path = (
                split_directory
                / f"{shard_name}_y.npy"
            )

            metadata_path = (
                split_directory
                / f"{shard_name}_metadata.csv"
            )

            channels_path = (
                split_directory
                / f"{shard_name}_channels.json"
            )
            availability_path = (
                split_directory
                / f"{shard_name}_channel_availability.json"
            )

            np.save(y_path, y_split)

            split_rows.to_csv(
                metadata_path,
                index=False,
            )

            with channels_path.open(
                "w",
                encoding="utf-8",
            ) as channel_file:
                json.dump(
                    channel_names,
                    channel_file,
                    indent=2,
                )

            with availability_path.open(
                "w",
                encoding="utf-8",
            ) as availability_file:
                json.dump(
                    channel_availability.astype(int).tolist(),
                    availability_file,
                    indent=2,
                )

            manifest_rows.append(
                {
                    "recording_id": recording_id,
                    "subject": subject,
                    "split": split_name,
                    "number_of_decisions": len(y_split),
                    "number_of_positive_decisions": int(
                        np.sum(y_split == 1)
                    ),
                    "number_of_negative_decisions": int(
                        np.sum(y_split == 0)
                    ),
                    "X_path": str(X_path.relative_to(project_root)),
                    "y_path": str(y_path.relative_to(project_root)),
                    "metadata_path": str(
                        metadata_path.relative_to(project_root)
                    ),
                    "channels_path": str(
                        channels_path.relative_to(project_root)
                    ),
                    "channel_availability_path": str(
                        availability_path.relative_to(project_root)
                    ),
                }
            )

        # This recording's full-resolution unscaled copy is never read again
        # in this run. Its disk footprint can rival or exceed the processed
        # shards being written here, so free it immediately rather than
        # waiting for the whole build to finish -- otherwise both copies
        # must coexist for the entire pass, roughly doubling peak disk use.
        del X
        for suffix in (
            ".npy",
            ".csv",
            "_channels.json",
            "_channel_availability.json",
        ):
            (
                unscaled_recordings_dir / f"{recording_id}{suffix}"
            ).unlink(missing_ok=True)

    return pd.DataFrame(manifest_rows)


# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------


def clear_generated_directory(
    directory: Path,
) -> None:
    """
    Delete and recreate a generated-data directory.

    This prevents stale shards from an older preprocessing run from being
    mixed with new output.
    """
    if directory.exists():
        shutil.rmtree(directory)

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


def class_summary(
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Return label counts by split."""
    return (
        metadata.groupby(
            ["split", "label_name"],
            dropna=False,
        )
        .size()
        .rename("count")
        .reset_index()
    )


def patient_class_summary(
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Return positive/negative decision counts for each patient and split."""
    summary = (
        metadata.groupby(["split", "subject", "label_name"], dropna=False)
        .size()
        .unstack("label_name", fill_value=0)
        .reset_index()
    )

    for label_name in ("seizure_within_10m", "no_seizure_within_10m"):
        if label_name not in summary.columns:
            summary[label_name] = 0

    return summary.rename(
        columns={
            "seizure_within_10m": "positive_decisions",
            "no_seizure_within_10m": "negative_decisions",
        }
    ).sort_values(["split", "subject"], kind="stable")


def seizure_scope_summary(
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Return local/cross/unknown counts among positive decisions."""
    positives = metadata[
        metadata["label"] == 1
    ]

    if positives.empty:
        return pd.DataFrame(
            columns=[
                "split",
                "seizure_scope",
                "count",
            ]
        )

    return (
        positives.groupby(
            ["split", "seizure_scope"],
            dropna=False,
        )
        .size()
        .rename("count")
        .reset_index()
    )
