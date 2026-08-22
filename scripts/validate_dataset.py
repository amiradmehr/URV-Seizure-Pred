"""
Validate the processed SeizeIT2 dataset.

Run from the repository root:

    python scripts/validate_dataset.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIRECTORY = PROJECT_ROOT / "src"

if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))


from seizure_prediction.config import CONFIG  # noqa: E402
from seizure_prediction.normalization import (  # noqa: E402
    load_scaler_document,
    scaler_key_for,
)
from seizure_prediction.preprocessing import (  # noqa: E402
    verify_patient_split_isolation,
)


SEPARATOR = "=" * 90


def print_header(title: str) -> None:
    """Print a consistent console section."""
    print()
    print(SEPARATOR)
    print(title.center(90))
    print(SEPARATOR)


def load_processed_manifest() -> pd.DataFrame:
    """Load the generated processed-shard manifest."""
    manifest_path = (
        CONFIG.manifests_dir
        / "processed_shard_manifest.csv"
    )

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Processed manifest not found: {manifest_path}\n"
            "Run scripts/build_dataset.py first."
        )

    return pd.read_csv(manifest_path)


def validate_single_shard(
    manifest_row: pd.Series,
) -> dict[str, object]:
    """
    Validate one processed shard and return summary statistics.
    """
    X_path = Path(manifest_row["X_path"])
    y_path = Path(manifest_row["y_path"])
    metadata_path = Path(
        manifest_row["metadata_path"]
    )
    channels_path = Path(
        manifest_row["channels_path"]
    )
    availability_path = Path(
        manifest_row["channel_availability_path"]
    )

    for required_path in (
        X_path,
        y_path,
        metadata_path,
        channels_path,
        availability_path,
    ):
        if not required_path.exists():
            raise FileNotFoundError(
                f"Missing processed file: {required_path}"
            )

    X = np.load(
        X_path,
        mmap_mode="r",
    )

    y = np.load(
        y_path,
        mmap_mode="r",
    )

    metadata = pd.read_csv(
        metadata_path,
        dtype={
            "subject": str,
            "session": str,
            "task": str,
            "run": str,
        },
    )

    with channels_path.open(
        "r",
        encoding="utf-8",
    ) as channel_file:
        channel_names = json.load(channel_file)

    with availability_path.open(
        "r",
        encoding="utf-8",
    ) as availability_file:
        channel_availability = np.asarray(
            json.load(availability_file),
            dtype=np.int8,
        )

    if channel_names != list(CONFIG.canonical_channel_names):
        raise ValueError(
            f"Unexpected channel layout in {channels_path}: "
            f"{channel_names}"
        )

    if X.ndim != 2:
        raise ValueError(
            f"{X_path} must be 2-D continuous EEG. Found shape {X.shape}."
        )

    if y.ndim != 1:
        raise ValueError(
            f"{y_path} must be 1-D. Found shape {y.shape}."
        )

    if len(y) != len(metadata):
        raise ValueError(
            f"y/metadata length mismatch for {y_path}: "
            f"{len(y)} versus {len(metadata)}."
        )

    if X.shape[0] != len(channel_names):
        raise ValueError(
            f"Channel count mismatch for {X_path}: "
            f"{X.shape[0]} data channels versus "
            f"{len(channel_names)} channel names."
        )

    if channel_availability.shape != (X.shape[0],) or not np.isin(
        channel_availability,
        [0, 1],
    ).all():
        raise ValueError(
            f"Invalid channel availability mask in {availability_path}: "
            f"{channel_availability.tolist()}"
        )

    missing_channels = channel_availability == 0
    if missing_channels.any() and not np.array_equal(
        np.asarray(X[missing_channels]),
        np.zeros_like(np.asarray(X[missing_channels])),
    ):
        raise ValueError(
            f"Missing channels are not zero-filled in {X_path}."
        )

    expected_history_samples = int(
        round(
            CONFIG.target_sfreq
            * CONFIG.input_window_seconds
        )
    )

    required_decision_columns = {
        "history_start_sample",
        "decision_end_sample",
        "chunk_samples",
        "chunks_per_history",
        "decision_time_seconds",
        "prediction_start_seconds",
        "prediction_stop_seconds",
    }
    missing_decision_columns = required_decision_columns - set(metadata.columns)

    if missing_decision_columns:
        raise ValueError(
            f"Decision metadata is missing columns: "
            f"{sorted(missing_decision_columns)}"
        )

    history_lengths = (
        metadata["decision_end_sample"]
        - metadata["history_start_sample"]
    )

    if not history_lengths.eq(expected_history_samples).all():
        raise ValueError(
            f"Incorrect 45-minute history length in {metadata_path}."
        )

    expected_chunk_samples = int(
        round(CONFIG.chunk_window_seconds * CONFIG.target_sfreq)
    )
    expected_chunks_per_history = (
        expected_history_samples // expected_chunk_samples
    )

    if not metadata["chunk_samples"].eq(expected_chunk_samples).all() or not (
        metadata["chunks_per_history"].eq(expected_chunks_per_history).all()
    ):
        raise ValueError(
            f"Incorrect 5-second chunk metadata in {metadata_path}."
        )

    if (metadata["history_start_sample"] < 0).any() or (
        metadata["decision_end_sample"] > X.shape[1]
    ).any():
        raise ValueError(
            f"Decision history indices exceed signal bounds in {metadata_path}."
        )

    expected_occurrence_seconds = (
        CONFIG.seizure_occurrence_period_minutes * 60.0
    )
    occurrence_lengths = (
        metadata["prediction_stop_seconds"]
        - metadata["prediction_start_seconds"]
    )

    if not np.allclose(occurrence_lengths, expected_occurrence_seconds):
        raise ValueError(
            f"Incorrect prediction occurrence period in {metadata_path}."
        )

    if not np.allclose(
        metadata["prediction_start_seconds"],
        metadata["decision_time_seconds"],
    ):
        raise ValueError(
            f"Prediction interval does not start at the decision time in "
            f"{metadata_path}."
        )

    unique_labels = set(
        np.unique(y).tolist()
    )

    if not unique_labels.issubset({0, 1}):
        raise ValueError(
            f"Invalid labels in {y_path}: {unique_labels}"
        )

    if not np.isfinite(X).all():
        raise ValueError(
            f"NaN or infinite EEG value found in {X_path}."
        )

    if not np.isfinite(y).all():
        raise ValueError(
            f"NaN or infinite label found in {y_path}."
        )

    expected_split = str(
        manifest_row["split"]
    )

    if not metadata["split"].eq(
        expected_split
    ).all():
        raise ValueError(
            f"Incorrect split value in {metadata_path}."
        )

    metadata_labels = metadata[
        "label"
    ].to_numpy(dtype=np.int64)

    if not np.array_equal(
        np.asarray(y),
        metadata_labels,
    ):
        raise ValueError(
            f"Labels do not match metadata in {y_path}."
        )

    positive_metadata = metadata[
        metadata["label"] == 1
    ]

    if not positive_metadata.empty:
        if positive_metadata[
            "target_seizure_id"
        ].isna().any():
            raise ValueError(
                f"A positive decision has no target seizure in "
                f"{metadata_path}."
            )

        allowed_scopes = {
            "local",
            "cross",
            "unknown",
        }

        found_scopes = set(
            positive_metadata[
                "seizure_scope"
            ].dropna()
        )

        if not found_scopes.issubset(
            allowed_scopes
        ):
            raise ValueError(
                f"Unexpected seizure scopes in "
                f"{metadata_path}: {found_scopes}"
            )

    negative_metadata = metadata[
        metadata["label"] == 0
    ]

    if not negative_metadata.empty:
        if not negative_metadata[
            "seizure_scope"
        ].eq("none").all():
            raise ValueError(
                f"Negative decisions have seizure scope values "
                f"in {metadata_path}."
            )

    return {
        "split": expected_split,
        "subject": str(
            manifest_row["subject"]
        ),
        "recording_id": str(
            manifest_row["recording_id"]
        ),
        "number_of_decisions": len(y),
        "number_of_channels": X.shape[0],
        "channel_availability": channel_availability.tolist(),
        "continuous_samples": X.shape[1],
        "positive_decisions": int(
            np.sum(y == 1)
        ),
        "negative_decisions": int(
            np.sum(y == 0)
        ),
        "mean": float(
            np.asarray(X).mean(
                dtype=np.float64
            )
        ),
        "std": float(
            np.asarray(X).std(
                dtype=np.float64
            )
        ),
    }


def load_window_manifest() -> pd.DataFrame:
    """Load complete decision metadata with stable subject IDs."""
    window_manifest_path = (
        CONFIG.manifests_dir
        / "decision_manifest.csv"
    )

    if not window_manifest_path.exists():
        raise FileNotFoundError(
            f"Window manifest not found: "
            f"{window_manifest_path}"
        )

    return pd.read_csv(
        window_manifest_path,
        dtype={
            "subject": str,
            "session": str,
            "task": str,
            "run": str,
        },
    )

def validate_patient_split_isolation() -> list[str]:
    """Verify observed patients stay in their configured split.

    A configured patient may legitimately have no retained decision points
    after the 60-minute clean-history and artifact exclusions. Such a patient
    has no shard and cannot leak across splits, so it is reported rather than
    treated as a split-isolation error.
    """
    metadata = load_window_manifest()
    verify_patient_split_isolation(metadata, CONFIG)

    observed_subjects = set(metadata["subject"].map(str))
    missing_subjects = sorted(
        set(CONFIG.included_subjects) - observed_subjects
    )

    if missing_subjects:
        print(
            "Subjects with no retained decision points: "
            f"{missing_subjects}"
        )

    return missing_subjects


def validate_seizure_split_overlap() -> None:
    """
    Report seizure IDs appearing in multiple splits.

    A patient-level split should keep every target seizure in exactly one
    split. This check makes any violation visible.
    """
    metadata = load_window_manifest()

    positives = metadata[
        metadata["label"] == 1
    ].copy()

    if positives.empty:
        print(
            "No positive decisions exist, so seizure overlap "
            "cannot be checked."
        )
        return

    split_counts = (
        positives.groupby(
            "target_seizure_id"
        )["split"]
        .nunique()
    )

    overlapping_seizures = split_counts[
        split_counts > 1
    ]

    if overlapping_seizures.empty:
        print(
            "No target seizure appears in multiple splits."
        )
        return

    print(
        "\nWARNING: The following target seizures have "
        "positive decisions in multiple splits:"
    )

    print(
        overlapping_seizures.to_string()
    )

    raise ValueError(
        "A target seizure crosses split boundaries. "
        "All windows for a patient must remain in one split."
    )


def validate_target_seizure_eligibility() -> None:
    """Ensure every positive decision targets a 60-minute-clear seizure."""
    seizure_manifest_path = CONFIG.manifests_dir / "seizure_manifest.csv"

    if not seizure_manifest_path.exists():
        raise FileNotFoundError(
            f"Seizure manifest not found: {seizure_manifest_path}"
        )

    seizures = pd.read_csv(seizure_manifest_path, dtype={"subject": str})

    if "eligible_for_prediction" not in seizures.columns:
        raise ValueError(
            "Seizure manifest does not record prediction eligibility."
        )

    eligibility_mask = (
        seizures["eligible_for_prediction"].astype(str).str.lower().eq("true")
    )
    eligible_ids = set(seizures.loc[eligibility_mask, "seizure_id"].astype(str))
    positives = load_window_manifest().query("label == 1")
    target_ids = set(positives["target_seizure_id"].dropna().astype(str))
    ineligible_ids = sorted(target_ids - eligible_ids)

    if ineligible_ids:
        raise ValueError(
            "Positive decisions target seizures without 60 minutes of clear "
            f"pre-seizure EEG: {ineligible_ids}"
        )


def validate_split_class_coverage() -> None:
    """Require each split to include positive and negative decisions."""
    metadata = load_window_manifest()
    expected_labels = {0, 1}

    for split_name, split_metadata in metadata.groupby("split", sort=True):
        labels = set(split_metadata["label"].unique().tolist())

        if labels != expected_labels:
            raise ValueError(
                f"Split {split_name} does not contain both classes. "
                f"Found labels: {sorted(labels)}"
            )


def validate_training_standardization(
    shard_summaries: pd.DataFrame,
) -> None:
    """
    Perform a broad sanity check on training standardization.

    The global channel statistics are fitted only from training patients.
    Individual training shards are not expected to have exact mean zero or
    unit standard deviation.
    """
    training = shard_summaries[
        shard_summaries["split"] == "train"
    ]

    if training.empty:
        raise ValueError(
            "No training shards were found."
        )

    if not np.isfinite(
        training["mean"]
    ).all():
        raise ValueError(
            "A training shard has a non-finite mean."
        )

    if not np.isfinite(
        training["std"]
    ).all():
        raise ValueError(
            "A training shard has a non-finite standard deviation."
        )

    if (
        training["std"] <= 0
    ).any():
        raise ValueError(
            "A training shard has zero or negative standard "
            "deviation."
        )


def validate_global_scaler() -> None:
    """Confirm the saved scalers match the configured normalization scope.

    Only ``global`` mode carries a train/validation/test asymmetry, so only
    that mode is checked for held-out subjects. ``patient`` and ``recording``
    scalers are fitted from the data they normalize by design; they are checked
    instead for complete coverage of the processed dataset.
    """
    document_path = CONFIG.scaler_parameters_dir / "channel_scalers.json"
    legacy_path = CONFIG.scaler_parameters_dir / "global_channel_zscore.json"

    if document_path.exists():
        scaler_path = document_path
    elif legacy_path.exists():
        scaler_path = legacy_path
    else:
        raise FileNotFoundError(
            f"No scaler document found at {document_path} or {legacy_path}"
        )

    document = load_scaler_document(scaler_path)
    mode = document["normalization_mode"]

    if document.get("channel_names") != list(CONFIG.canonical_channel_names):
        raise ValueError(
            "Scaler channel layout does not match the configured canonical "
            "layout."
        )

    print(
        f"Normalization: mode={mode}, "
        f"statistic={document.get('statistic', 'meanstd')}, "
        f"scalers={document.get('scaler_count', len(document['scalers']))}"
    )

    if mode == "global":
        scaler_subjects = set(map(str, document.get("training_subjects", [])))

        if not scaler_subjects:
            raise ValueError("Global scaler records no training subjects.")

        if not scaler_subjects.issubset(set(CONFIG.train_subjects)):
            raise ValueError(
                "Global scaler includes validation/test subjects: "
                f"{sorted(scaler_subjects - set(CONFIG.train_subjects))}"
            )
        return

    # Per-patient and per-recording scaling must cover every processed shard,
    # otherwise some recording would be standardized by another group's stats.
    manifest = load_processed_manifest()
    required_keys = {
        scaler_key_for(
            mode,
            subject=str(row.subject).zfill(3),
            recording_id=str(row.recording_id),
        )
        for row in manifest.itertuples(index=False)
    }
    uncovered = sorted(required_keys - set(document["scalers"]))

    if uncovered:
        raise ValueError(
            f"{len(uncovered)} processed recording(s) have no {mode} scaler, "
            f"first: {uncovered[:5]}"
        )

    if document.get("non_causal_statistics"):
        print(
            "  Note: these statistics summarize whole recordings and are "
            "therefore non-causal. No labels were used and nothing crosses "
            "between patients."
        )


def main() -> None:
    """Run all validation checks."""
    CONFIG.validate()

    print_header("PROCESSED SEIZEIT2 VALIDATION")

    processed_manifest = load_processed_manifest()

    if processed_manifest.empty:
        raise ValueError(
            "The processed shard manifest is empty."
        )

    shard_summaries: list[dict[str, object]] = []

    for shard_number, (_, manifest_row) in enumerate(
        processed_manifest.iterrows(),
        start=1,
    ):
        summary = validate_single_shard(
            manifest_row
        )

        shard_summaries.append(summary)

        print(
            f"[{shard_number:03d}/"
            f"{len(processed_manifest):03d}] "
            f"{summary['recording_id']} "
            f"{summary['split']} "
            f"signal=({summary['number_of_channels']}, "
            f"{summary['continuous_samples']}) "
            f"decisions={summary['number_of_decisions']}"
        )

    summaries = pd.DataFrame(
        shard_summaries
    )

    print_header("PATIENT-LEVEL SPLIT VALIDATION")

    excluded_subjects = validate_patient_split_isolation()
    print(
        "Patient split isolation: PASS"
        + (
            f" ({len(excluded_subjects)} subjects had no usable data)"
            if excluded_subjects
            else ""
        )
    )

    validate_seizure_split_overlap()
    print("Seizure split isolation: PASS")

    validate_target_seizure_eligibility()
    print("Target-seizure 60-minute eligibility: PASS")

    validate_split_class_coverage()
    print("Split class coverage: PASS")

    validate_training_standardization(
        summaries
    )
    print("Standardization sanity checks: PASS")

    validate_global_scaler()
    print("Global train-only scaler provenance: PASS")

    print_header("DATASET SUMMARY")

    split_summary = (
        summaries.groupby("split")
        .agg(
            shards=("recording_id", "count"),
            patients=("subject", "nunique"),
            decisions=("number_of_decisions", "sum"),
            positive_decisions=("positive_decisions", "sum"),
            negative_decisions=(
                "negative_decisions",
                "sum",
            ),
            minimum_shard_mean=("mean", "min"),
            maximum_shard_mean=("mean", "max"),
            minimum_shard_std=("std", "min"),
            maximum_shard_std=("std", "max"),
        )
        .reset_index()
    )

    print(split_summary.to_string(index=False))

    output_path = (
        CONFIG.manifests_dir
        / "validation_summary.csv"
    )

    summaries.to_csv(
        output_path,
        index=False,
    )

    print_header("VALIDATION COMPLETE")

    print("All required checks passed.")
    print(f"Validation summary: {output_path}")


if __name__ == "__main__":
    main()
