"""
Reusable preprocessing functions for the CHB-MIT dataset.

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

from config import PreprocessingConfig


# The eighteen double-banana bipolar derivations stored by the BIDS CHB-MIT
# release, in EDF order.  ``PreprocessingConfig.canonical_channel_names``
# must agree with this list; the montage helpers check that explicitly.
CANONICAL_DOUBLE_BANANA_CHANNELS: tuple[str, ...] = (
    "FP1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "FP1-F7",
    "F7-T7",
    "T7-P7",
    "P7-O1",
    "FZ-CZ",
    "CZ-PZ",
    "FP2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
    "FP2-F8",
    "F8-T8",
    "T8-P8",
    "P8-O2",
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
        ``["01", "02"]``.
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

    The BIDS CHB-MIT release annotates every seizure with the unqualified
    SCORE type ``sz``, because the original database does not record the
    ILAE seizure subtype.  The qualified ``sz_*`` names of the shared event
    dictionary are accepted as well so that a future re-annotation of the
    dataset does not silently drop its seizures.
    """
    if not isinstance(event_type, str):
        return False

    normalized = event_type.strip().lower()

    return normalized == "sz" or normalized.startswith("sz_")


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


def normalize_channel_name(channel_name: str) -> str:
    """
    Map one EDF channel label onto its canonical derivation name.

    EDF headers pad and case labels inconsistently between exports, so the
    label is compared in a whitespace-free upper-case form.
    """
    return re.sub(r"\s+", "", str(channel_name)).upper()


def canonical_name_by_raw_name(
    raw: mne.io.BaseRaw,
    config: PreprocessingConfig,
) -> dict[str, str]:
    """Return the raw-to-canonical channel mapping for one recording."""
    canonical_by_normalized = {
        normalize_channel_name(canonical_name): canonical_name
        for canonical_name in config.canonical_channel_names
    }

    mapping: dict[str, str] = {}

    for channel_name in raw.ch_names:
        canonical_name = canonical_by_normalized.get(
            normalize_channel_name(channel_name)
        )

        if canonical_name is not None:
            mapping[channel_name] = canonical_name

    return mapping


def infer_montage_status(raw: mne.io.BaseRaw) -> str:
    """
    Return the recorded montage configuration of one recording.

    Every EDF in the BIDS CHB-MIT release carries the complete eighteen
    channel double-banana montage, which is reported as ``complete``.  A
    recording holding only part of that montage is reported as ``partial``
    rather than rejected, so the availability mask stays authoritative.
    """
    normalized_names = {
        normalize_channel_name(channel_name)
        for channel_name in raw.ch_names
    }

    canonical_normalized = {
        normalize_channel_name(canonical_name)
        for canonical_name in CANONICAL_DOUBLE_BANANA_CHANNELS
    }

    present = canonical_normalized & normalized_names

    if not present:
        raise ValueError(
            "Expected double-banana bipolar derivations in the recording. "
            f"Found channels: {raw.ch_names}"
        )

    if present == canonical_normalized:
        return "complete"

    return "partial"


def channel_availability_mask(
    raw: mne.io.BaseRaw,
    config: PreprocessingConfig,
) -> np.ndarray:
    """Return the canonical-derivation availability mask for one EDF."""
    if set(config.canonical_channel_names) != set(
        CANONICAL_DOUBLE_BANANA_CHANNELS
    ):
        raise ValueError(
            "Canonical channel names do not match the supported CHB-MIT "
            "double-banana derivations."
        )

    normalized_names = {
        normalize_channel_name(channel_name)
        for channel_name in raw.ch_names
    }

    return np.asarray(
        [
            normalize_channel_name(channel_name) in normalized_names
            for channel_name in config.canonical_channel_names
        ],
        dtype=bool,
    )


def canonicalize_eeg_channels(
    raw: mne.io.BaseRaw,
    config: PreprocessingConfig,
) -> mne.io.BaseRaw:
    """
    Put available CHB-MIT signals into a stable eighteen-derivation order.

    Each EDF in this release holds the full double-banana montage.  An
    all-zero placeholder is added only for a derivation that is absent.
    Its availability is saved separately and excluded from scaler fitting,
    so it is never treated as EEG data.
    """
    raw_to_canonical = canonical_name_by_raw_name(raw, config)

    unexpected_channels = set(raw.ch_names) - set(raw_to_canonical)

    if unexpected_channels:
        raise ValueError(
            "Unexpected EEG channel(s) in CHB-MIT recording: "
            f"{sorted(unexpected_channels)}"
        )

    if len(set(raw_to_canonical.values())) != len(raw_to_canonical):
        raise ValueError(
            "Two EDF channels map onto the same canonical derivation: "
            f"{raw.ch_names}"
        )

    if not raw_to_canonical:
        raise ValueError(
            "The recording contains none of the canonical derivations, "
            f"found: {raw.ch_names}"
        )

    processed = raw.copy()
    processed.rename_channels(raw_to_canonical)

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
    3. Apply a 60-Hz notch filter.
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

        if event_type == "bckg" or is_seizure_event(event_type):
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
    montage_status: str,
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
                        "montage_status": montage_status,
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
    """Load the canonical-derivation availability mask for one recording."""
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
# Patient-level splitting
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
) -> pd.DataFrame:
    """
    Standardize every continuous recording and save its decision metadata.

    Each recording belongs to exactly one patient-level split.
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

            X_path = (
                split_directory
                / f"{shard_name}_X.npy"
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

            np.save(X_path, X_split)
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
                    "X_path": str(X_path),
                    "y_path": str(y_path),
                    "metadata_path": str(
                        metadata_path
                    ),
                    "channels_path": str(
                        channels_path
                    ),
                    "channel_availability_path": str(availability_path),
                }
            )

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
