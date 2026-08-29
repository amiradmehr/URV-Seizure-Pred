"""
Reusable preprocessing functions for the SeizeIT2 dataset.

This module handles:

1. Loading BIDS-organized EDF recordings directly.
2. Reading seizure annotations.
3. Applying bandpass and notch filtering.
4. Preserving the native 256-Hz EEG sampling rate.
5. Creating indexed streaming decision points whose history length is the
   configured input window.
6. Labeling seizure risk during the following seizure occurrence period.
7. Preserving local/cross seizure metadata when available.
8. Splitting complete patients into train, validation, and test groups.
9. Applying the channel scalers fitted by seizure_prediction.normalization,
   at global, per-patient, or per-recording scope.

Only steps 5 and 6 depend on the label definition.  Filtering and scaler
fitting are shared by every window/horizon combination, which is why the
filtered EEG is written once and never duplicated per combination.
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
from seizure_prediction.config import PreprocessingConfig
from seizure_prediction.normalization import (
    apply_channel_scaler,
    scaler_key_for,
    select_scaler,
)


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
    """Label one streaming decision for seizure onset in the occurrence period."""
    horizon_seconds = config.prediction_horizon_minutes * 60.0
    occurrence_seconds = config.seizure_occurrence_period_minutes * 60.0
    postictal_seconds = config.postictal_exclusion_minutes * 60.0
    target_start = decision_time_seconds + horizon_seconds
    target_stop = target_start + occurrence_seconds
    occurrence_label = f"{config.seizure_occurrence_period_minutes:g}m"

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
                f"seizure_within_{occurrence_label}",
                seizure_id,
                str(seizure["seizure_scope"]),
            )

    return 0, f"no_seizure_within_{occurrence_label}", None, "none"


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


def filtered_recording_paths(
    output_directory: Path,
    recording_id: str,
) -> tuple[Path, Path, Path, Path]:
    """Return the four files that make up one shared filtered recording."""
    return (
        output_directory / f"{recording_id}.npy",
        output_directory / f"{recording_id}_channels.json",
        output_directory / f"{recording_id}_channel_availability.json",
        output_directory / f"{recording_id}_recording.json",
    )


def save_filtered_recording(
    raw: mne.io.BaseRaw,
    channel_availability: np.ndarray,
    bte_side: str,
    output_directory: Path,
    recording_id: str,
    output_dtype: str,
) -> Path:
    """
    Save one filtered continuous recording to the shared cache.

    Everything needed to relabel this recording under a different input
    window or seizure occurrence period is stored alongside the signal --
    channel order, the availability mask, the sampling rate, the inferred
    behind-the-ear side, and the annotations that mark unusable intervals.
    A later build can therefore produce a new label definition without
    reopening and refiltering the EDF, which is the slow part of the
    pipeline and the reason this cache is never written per combination.
    """
    output_directory.mkdir(parents=True, exist_ok=True)

    (
        array_path,
        channels_path,
        availability_path,
        recording_path,
    ) = filtered_recording_paths(output_directory, recording_id)

    signal = raw.get_data().astype(output_dtype, copy=False)

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

    with channels_path.open("w", encoding="utf-8") as channel_file:
        json.dump(list(raw.ch_names), channel_file, indent=2)

    with availability_path.open("w", encoding="utf-8") as availability_file:
        json.dump(availability.astype(int).tolist(), availability_file, indent=2)

    with recording_path.open("w", encoding="utf-8") as recording_file:
        json.dump(
            {
                "recording_id": recording_id,
                "sampling_frequency_hz": float(raw.info["sfreq"]),
                "number_of_samples": int(signal.shape[1]),
                "bte_side": bte_side,
                "annotations": [
                    {
                        "onset_seconds": float(onset),
                        "duration_seconds": float(duration),
                        "description": str(description),
                    }
                    for onset, duration, description in zip(
                        raw.annotations.onset,
                        raw.annotations.duration,
                        raw.annotations.description,
                    )
                ],
            },
            recording_file,
            indent=2,
        )

    return array_path


def load_filtered_recording(
    output_directory: Path,
    recording_id: str,
    config: PreprocessingConfig,
) -> tuple[mne.io.BaseRaw, str, np.ndarray] | None:
    """
    Rebuild a cached filtered recording, or return ``None`` if unusable.

    The returned object behaves like the output of ``filter_and_prepare`` --
    same channel order, sampling rate, and annotations -- so decision
    labeling cannot tell a cache hit from a fresh filter pass.  Anything
    inconsistent returns ``None`` so the caller refilters from the EDF rather
    than labeling against a partially written file.
    """
    (
        array_path,
        channels_path,
        availability_path,
        recording_path,
    ) = filtered_recording_paths(output_directory, recording_id)

    paths = (array_path, channels_path, availability_path, recording_path)

    if not all(path.exists() for path in paths):
        return None

    try:
        with channels_path.open("r", encoding="utf-8") as channel_file:
            channel_names = json.load(channel_file)

        if channel_names != list(config.canonical_channel_names):
            raise ValueError(f"unexpected channel layout {channel_names}")

        with availability_path.open("r", encoding="utf-8") as availability_file:
            availability = np.asarray(json.load(availability_file), dtype=bool)

        if availability.shape != (len(config.canonical_channel_names),):
            raise ValueError(
                f"invalid availability mask {availability.tolist()}"
            )

        if not availability.any():
            raise ValueError("cached recording has no available EEG channel")

        with recording_path.open("r", encoding="utf-8") as recording_file:
            recording_document = json.load(recording_file)

        sampling_frequency = float(recording_document["sampling_frequency_hz"])

        if not np.isclose(sampling_frequency, config.target_sfreq):
            raise ValueError(
                "cached sampling frequency "
                f"{sampling_frequency} does not match {config.target_sfreq}"
            )

        signal = np.load(array_path, mmap_mode="r")

        if signal.ndim != 2 or signal.shape[0] != len(channel_names):
            raise ValueError(f"unexpected signal shape {signal.shape}")

        if signal.shape[1] != int(recording_document["number_of_samples"]):
            raise ValueError(
                "cached signal length does not match its recording document; "
                "the file is probably truncated"
            )

        raw = mne.io.RawArray(
            np.asarray(signal, dtype=np.float64),
            mne.create_info(
                ch_names=list(channel_names),
                sfreq=sampling_frequency,
                ch_types="eeg",
            ),
            verbose="ERROR",
        )

        annotations = recording_document["annotations"]

        if annotations:
            raw.set_annotations(
                mne.Annotations(
                    onset=[item["onset_seconds"] for item in annotations],
                    duration=[item["duration_seconds"] for item in annotations],
                    description=[item["description"] for item in annotations],
                ),
                verbose="ERROR",
            )

        return raw, str(recording_document["bte_side"]), availability
    except (ValueError, KeyError, TypeError, json.JSONDecodeError, EOFError):
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
# Scaler persistence
# ----------------------------------------------------------------------
#
# Fitting lives in seizure_prediction.normalization, which supports the
# global, patient, and recording scopes over the current continuous
# (channels, samples) layout.  The former patient-scoped helpers here were
# written for the retired 3-D windowed layout and have been removed.


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


# ----------------------------------------------------------------------
# Global train-only channel z-score fitting
# ----------------------------------------------------------------------


def fit_global_channel_scaler(
    metadata: pd.DataFrame,
    unscaled_recordings_dir: Path,
    epsilon: float,
    config: PreprocessingConfig,
) -> dict[str, Any]:
    """Fit one per-channel z-score transform from training patients only."""
    required_columns = {
        "subject",
        "recording_id",
        "split",
    }
    missing_columns = required_columns - set(metadata.columns)

    if missing_columns:
        raise ValueError(
            "Metadata is missing scaler columns: "
            f"{sorted(missing_columns)}"
        )

    train_metadata = metadata[metadata["split"] == "train"]

    if train_metadata.empty:
        raise ValueError("There are no training decision points.")

    observed_train_subjects = set(
        train_metadata["subject"].map(normalize_entity)
    )

    if not observed_train_subjects.issubset(set(config.train_subjects)):
        raise ValueError(
            "Global scaler received non-training subjects: "
            f"{sorted(observed_train_subjects - set(config.train_subjects))}"
        )

    channel_count = len(config.canonical_channel_names)
    signal_sum = np.zeros(channel_count, dtype=np.float64)
    signal_sum_squares = np.zeros(channel_count, dtype=np.float64)
    sample_count = np.zeros(channel_count, dtype=np.int64)

    for recording_id, recording_rows in train_metadata.groupby(
        "recording_id",
        sort=False,
    ):
        array_path = unscaled_recordings_dir / f"{recording_id}.npy"

        if not array_path.exists():
            raise FileNotFoundError(f"Missing unscaled recording: {array_path}")

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


def write_decision_shards(
    metadata: pd.DataFrame,
    unscaled_recordings_dir: Path,
    processed_data_dir: Path,
    scaler_document: dict[str, Any],
    scaler_document_path: Path,
    project_root: Path,
) -> pd.DataFrame:
    """
    Save each recording's decision labels and metadata for one label definition.

    No EEG is written.  Every shard points back at the single shared filtered
    recording in ``unscaled_recordings_dir``, and standardization happens when
    a decision is loaded: it is a per-channel affine map, so scaling a sliced
    history is identical to slicing a scaled recording.  Twelve label
    definitions therefore cost twelve sets of labels rather than twelve copies
    of the EEG.

    The scaler that standardizes a recording is named per shard from
    ``scaler_document`` according to its normalization mode, so the same
    routine serves global, per-patient, and per-recording scaling.  Each row
    also records where that document lives, which makes the manifest
    self-describing: a consumer can standardize the recordings it references
    without being told separately which normalization the build used.

    Each recording belongs to exactly one patient-level split.  Paths are
    stored relative to ``project_root`` so the tree can be moved or copied to
    another machine without rewriting the manifest.
    """
    manifest_rows: list[dict[str, Any]] = []
    normalization_mode = scaler_document["normalization_mode"]

    for recording_id, recording_metadata in metadata.groupby(
        "recording_id",
        sort=False,
    ):
        array_path = (
            unscaled_recordings_dir
            / f"{recording_id}.npy"
        )

        if not array_path.exists():
            raise ValueError(
                f"Recording {recording_id} has decisions but no filtered "
                f"recording at {array_path}."
            )

        channel_names = load_channel_names_for_shard(
            array_path
        )
        channel_availability = load_channel_availability_for_shard(array_path)

        subject = normalize_entity(
            recording_metadata["subject"].iloc[0]
        )

        # Resolve the scaler now so a missing one fails during the build
        # rather than in the middle of a training epoch.
        scaler_key = scaler_key_for(
            normalization_mode,
            subject=subject,
            recording_id=str(recording_id),
        )
        select_scaler(
            scaler_document,
            subject=subject,
            recording_id=str(recording_id),
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
                    "normalization_mode": normalization_mode,
                    "scaler_key": scaler_key,
                    "scaler_document_path": relative_to_project_root(
                        scaler_document_path,
                        project_root,
                    ),
                    "number_of_decisions": len(y_split),
                    "number_of_positive_decisions": int(
                        np.sum(y_split == 1)
                    ),
                    "number_of_negative_decisions": int(
                        np.sum(y_split == 0)
                    ),
                    "X_path": relative_to_project_root(
                        array_path,
                        project_root,
                    ),
                    "y_path": relative_to_project_root(
                        y_path,
                        project_root,
                    ),
                    "metadata_path": relative_to_project_root(
                        metadata_path,
                        project_root,
                    ),
                    "channels_path": relative_to_project_root(
                        channels_path,
                        project_root,
                    ),
                    "channel_availability_path": relative_to_project_root(
                        availability_path,
                        project_root,
                    ),
                }
            )

    return pd.DataFrame(manifest_rows)


# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------


def relative_to_project_root(
    path: Path,
    project_root: Path,
) -> str:
    """
    Return ``path`` written relative to ``project_root`` where possible.

    Manifests record where the shared filtered recordings live.  Storing that
    relative to the project root lets the whole data tree be copied to another
    machine -- a training box, or Colab -- without rewriting every row.  A
    path outside the project root is returned absolute, since there is no
    relative form that would survive the move anyway.
    """
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except ValueError:
        return str(path.resolve())


def clear_generated_directory(
    directory: Path,
) -> None:
    """
    Delete and recreate a generated-data directory.

    This prevents stale shards from an older preprocessing run from being
    mixed with new output.

    Refuses to touch anything under an ``interim/_shared`` tree.  That tree
    holds the single copy of the filtered EEG behind every label definition,
    and rebuilding it costs hours of EDF decoding, so a tagged build clearing
    its own output must never reach it.
    """
    resolved = directory.resolve()

    if "_shared" in resolved.parts:
        raise ValueError(
            "Refusing to clear the shared preprocessing cache at "
            f"{resolved}. Filtered recordings and fitted scalers are reused "
            "by every label definition and are never rebuilt by a tagged run."
        )

    if resolved.exists():
        shutil.rmtree(resolved)

    resolved.mkdir(
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
