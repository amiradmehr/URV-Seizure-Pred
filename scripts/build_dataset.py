"""
Build the filtered, labeled, patient-level split SeizeIT2 dataset.

Run from the repository root:

    python scripts/build_dataset.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# Allow the script to work before or after `pip install -e .`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIRECTORY = PROJECT_ROOT / "src"

if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))


from seizure_prediction.config import CONFIG  # noqa: E402
from seizure_prediction.preprocessing import (  # noqa: E402
    assign_patient_splits,
    channel_availability_mask,
    class_summary,
    clear_generated_directory,
    extract_recording_entities,
    filter_and_prepare,
    find_events_file,
    fit_global_channel_scaler,
    infer_bte_side,
    load_bids_recordings,
    patient_class_summary,
    read_seizure_events,
    read_recording_events,
    recording_id_from_entities,
    save_scaler,
    save_unscaled_recording,
    seizure_scope_summary,
    verify_patient_split_isolation,
    write_standardized_shards,
    create_labeled_prediction_decisions,
)


SEPARATOR = "=" * 90


def print_header(title: str) -> None:
    """Print a consistent console section header."""
    print()
    print(SEPARATOR)
    print(title.center(90))
    print(SEPARATOR)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line options for a fresh build or safe resume."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Reuse validated unscaled recording checkpoints and process only "
            "missing or incomplete recordings."
        ),
    )
    return parser.parse_args()


def checkpoint_paths(recording_id: str) -> tuple[Path, Path, Path, Path]:
    """Return the four required unscaled-recording checkpoint paths."""
    directory = CONFIG.unscaled_recordings_dir
    return (
        directory / f"{recording_id}.npy",
        directory / f"{recording_id}.csv",
        directory / f"{recording_id}_channels.json",
        directory / f"{recording_id}_channel_availability.json",
    )


def discard_checkpoint(recording_id: str) -> None:
    """Remove only the known generated files for one incomplete recording."""
    for checkpoint_path in checkpoint_paths(recording_id):
        checkpoint_path.unlink(missing_ok=True)


def load_resumable_checkpoint(recording_id: str) -> pd.DataFrame | None:
    """Return validated saved decision metadata, or discard a bad checkpoint."""
    array_path, metadata_path, channels_path, availability_path = checkpoint_paths(
        recording_id
    )
    paths = (array_path, metadata_path, channels_path, availability_path)

    if not any(path.exists() for path in paths):
        return None

    try:
        if not all(path.exists() for path in paths):
            raise ValueError("one or more required checkpoint files are missing")

        signal = np.load(array_path, mmap_mode="r")
        if signal.ndim != 2 or signal.shape[0] != len(CONFIG.canonical_channel_names):
            raise ValueError(f"unexpected signal shape {signal.shape}")
        if signal.shape[1] <= 0:
            raise ValueError("recording contains no samples")
        if not np.isfinite(signal[:, [0, -1]]).all():
            raise ValueError("recording endpoint contains a non-finite value")
        del signal

        with channels_path.open("r", encoding="utf-8") as channel_file:
            channel_names = json.load(channel_file)
        if channel_names != list(CONFIG.canonical_channel_names):
            raise ValueError(f"unexpected channel layout {channel_names}")

        with availability_path.open("r", encoding="utf-8") as availability_file:
            availability = np.asarray(json.load(availability_file), dtype=np.int8)
        if availability.shape != (len(CONFIG.canonical_channel_names),) or not np.isin(
            availability,
            [0, 1],
        ).all():
            raise ValueError(f"invalid availability mask {availability.tolist()}")

        metadata = pd.read_csv(
            metadata_path,
            dtype={
                "recording_id": str,
                "subject": str,
                "session": str,
                "task": str,
                "run": str,
            },
        )
        required_columns = {"recording_id", "label", "target_seizure_id"}
        missing_columns = required_columns - set(metadata.columns)
        if metadata.empty or missing_columns:
            raise ValueError(
                "empty decision metadata or missing columns "
                f"{sorted(missing_columns)}"
            )
        if not metadata["recording_id"].eq(recording_id).all():
            raise ValueError("metadata recording ID does not match checkpoint")
        if not metadata["label"].isin([0, 1]).all():
            raise ValueError("metadata contains invalid labels")

        return metadata
    except (ValueError, KeyError, json.JSONDecodeError, EOFError) as error:
        discard_checkpoint(recording_id)
        print(f"    Discarded incomplete checkpoint: {error}")
        return None
    except OSError as error:
        raise RuntimeError(
            "Could not read an existing checkpoint; it was left untouched. "
            f"Resolve the filesystem error and retry: {array_path} ({error})"
        ) from error


def seizure_metadata_from_resumed_checkpoint(
    seizure_events: pd.DataFrame,
    decision_metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Mark target seizures known eligible from an already saved checkpoint."""
    resumed_seizures = seizure_events.copy()
    known_targets = set(
        decision_metadata.loc[
            decision_metadata["label"] == 1,
            "target_seizure_id",
        ]
        .dropna()
        .astype(str)
    )
    # A saved positive decision proves its target passed the clear-history
    # eligibility check. Other seizures are deliberately left unknown rather
    # than incorrectly marking them ineligible without rereading their EEG.
    resumed_seizures["eligible_for_prediction"] = pd.NA
    resumed_seizures.loc[
        resumed_seizures["seizure_id"].astype(str).isin(known_targets),
        "eligible_for_prediction",
    ] = True
    resumed_seizures["eligibility_evaluated"] = False
    return resumed_seizures


def main() -> None:
    """Execute the complete preprocessing workflow."""
    arguments = parse_arguments()
    CONFIG.validate()

    print_header("SEIZEIT2 PREPROCESSING PIPELINE")

    print(f"Project root:       {CONFIG.project_root}")
    print(f"Raw dataset:        {CONFIG.raw_data_dir}")
    print(f"Interim output:     {CONFIG.interim_data_dir}")
    print(f"Processed output:   {CONFIG.processed_data_dir}")
    print(
        "Subjects:           "
        f"{len(CONFIG.included_subjects)} configured "
        "patient-level subjects"
    )

    if arguments.resume:
        CONFIG.unscaled_recordings_dir.mkdir(parents=True, exist_ok=True)
        print("Resume mode: validated existing recordings will be reused.")
    else:
        clear_generated_directory(CONFIG.unscaled_recordings_dir)

    # Final shards and aggregate manifests are rebuilt from the complete
    # checkpoint set after pass 1, whether this is a fresh run or a resume.
    clear_generated_directory(
        CONFIG.processed_data_dir
    )

    clear_generated_directory(CONFIG.manifests_dir)

    CONFIG.scaler_parameters_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    recordings = load_bids_recordings(
        config=CONFIG,
        subjects=CONFIG.included_subjects,
        preload=False,
    )

    print(
        f"\nEDF recordings found: "
        f"{len(recordings)}"
    )

    loaded_subjects = {
        extract_recording_entities(recording.description)["subject"]
        for recording in recordings
    }
    missing_subjects = sorted(set(CONFIG.included_subjects) - loaded_subjects)

    if missing_subjects:
        raise RuntimeError(
            "The configured patient split includes subjects with no loaded "
            f"EEG recordings: {missing_subjects}"
        )

    all_decision_metadata: list[pd.DataFrame] = []
    all_seizure_metadata: list[pd.DataFrame] = []
    reused_recordings = 0
    rebuilt_recordings = 0

    # ------------------------------------------------------------------
    # Pass 1: Filter at 256 Hz, create decisions, and save recordings.
    # ------------------------------------------------------------------

    print_header("PASS 1: NATIVE-RATE FILTERING AND WINDOW GENERATION")

    for recording_number, recording in enumerate(
        recordings,
        start=1,
    ):
        entities = extract_recording_entities(
            recording.description
        )

        recording_id = recording_id_from_entities(
            entities
        )

        print(
            f"[{recording_number:03d}/"
            f"{len(recordings):03d}] "
            f"{recording_id}"
        )

        resumed_metadata = (
            load_resumable_checkpoint(recording_id)
            if arguments.resume
            else None
        )

        if resumed_metadata is not None:
            events_path = find_events_file(
                dataset_root=CONFIG.raw_data_dir,
                entities=entities,
            )
            seizure_events = read_seizure_events(
                events_path=events_path,
                entities=entities,
            )
            all_decision_metadata.append(resumed_metadata)
            if not seizure_events.empty:
                all_seizure_metadata.append(
                    seizure_metadata_from_resumed_checkpoint(
                        seizure_events,
                        resumed_metadata,
                    )
                )
            reused_recordings += 1
            print(
                "    Reused validated checkpoint: "
                f"{len(resumed_metadata)} decision points."
            )
            continue

        events_path = find_events_file(
            dataset_root=CONFIG.raw_data_dir,
            entities=entities,
        )

        seizure_events = read_seizure_events(
            events_path=events_path,
            entities=entities,
        )
        recording_events = read_recording_events(events_path)

        raw = recording.load_raw()
        bte_side = infer_bte_side(raw)
        availability = channel_availability_mask(raw, CONFIG)

        processed_raw = filter_and_prepare(
            raw=raw,
            config=CONFIG,
        )
        del raw

        metadata_recording, seizure_metadata_recording = (
            create_labeled_prediction_decisions(
                raw=processed_raw,
                seizure_events=seizure_events,
                recording_events=recording_events,
                entities=entities,
                bte_side=bte_side,
                config=CONFIG,
            )
        )

        if not seizure_metadata_recording.empty:
            seizure_metadata_recording["eligibility_evaluated"] = True
            all_seizure_metadata.append(seizure_metadata_recording)

        rebuilt_recordings += 1

        print(
            f"    Channels: {processed_raw.ch_names}"
        )
        print(f"    Availability mask: {availability.astype(int).tolist()}")
        print(
            f"    Sampling frequency: "
            f"{processed_raw.info['sfreq']} Hz"
        )
        print(
            f"    Seizures in recording: "
            f"{len(seizure_events)}"
        )
        print(
            f"    Retained decision points: "
            f"{len(metadata_recording)}"
        )

        if metadata_recording.empty:
            print(
                "    No usable decision points; recording skipped."
            )
            continue

        save_unscaled_recording(
            signal=processed_raw.get_data().astype(
                CONFIG.signal_dtype,
                copy=False,
            ),
            decision_metadata=metadata_recording,
            channel_names=processed_raw.ch_names,
            channel_availability=availability,
            output_directory=(
                CONFIG.unscaled_recordings_dir
            ),
            recording_id=recording_id,
        )

        all_decision_metadata.append(
            metadata_recording
        )

        # Explicitly release potentially large objects.
        del processed_raw
        del metadata_recording

    if arguments.resume:
        print(
            "\nResume result: "
            f"reused {reused_recordings} recordings; "
            f"processed {rebuilt_recordings} recordings."
        )

    if not all_decision_metadata:
        raise RuntimeError(
            "No usable prediction decision points were generated."
        )

    window_metadata = pd.concat(
        all_decision_metadata,
        ignore_index=True,
    )

    if all_seizure_metadata:
        seizure_metadata = pd.concat(
            all_seizure_metadata,
            ignore_index=True,
        )
    else:
        seizure_metadata = pd.DataFrame()

    # ------------------------------------------------------------------
    # Pass 2: Fixed patient-level splitting.
    # ------------------------------------------------------------------

    print_header("PASS 2: PATIENT-LEVEL SPLITTING")

    window_metadata = (
        assign_patient_splits(
            metadata=window_metadata,
            config=CONFIG,
        )
    )

    verify_patient_split_isolation(window_metadata, CONFIG)

    window_manifest_path = (
        CONFIG.manifests_dir
        / "decision_manifest.csv"
    )

    seizure_manifest_path = (
        CONFIG.manifests_dir
        / "seizure_manifest.csv"
    )

    window_metadata.to_csv(
        window_manifest_path,
        index=False,
    )

    seizure_metadata.to_csv(
        seizure_manifest_path,
        index=False,
    )

    print("\nDecision class counts:")
    print(
        class_summary(window_metadata).to_string(
            index=False
        )
    )

    print("\nPositive-decision seizure-scope counts:")
    scope_summary = seizure_scope_summary(
        window_metadata
    )

    if scope_summary.empty:
        print("No positive decisions were generated.")
    else:
        print(
            scope_summary.to_string(
                index=False
            )
        )

    unknown_positive_count = int(
        (
            (window_metadata["label"] == 1)
            & (
                window_metadata["seizure_scope"]
                == "unknown"
            )
        ).sum()
    )

    if unknown_positive_count > 0:
        print(
            "\nWARNING: "
            f"{unknown_positive_count} positive decisions have "
            "unknown local/cross scope. The event files did not "
            "contain a recognized authoritative scope column. "
            "Do not guess these labels; add them from the trusted "
            "SeizeIT2 annotation source."
        )

    # ------------------------------------------------------------------
    # Pass 3: Fit a global channel z-score using training patients only.
    # ------------------------------------------------------------------

    print_header("PASS 3: GLOBAL TRAIN-ONLY Z-SCORE FITTING")

    scaler = fit_global_channel_scaler(
        metadata=window_metadata,
        unscaled_recordings_dir=(
            CONFIG.unscaled_recordings_dir
        ),
        epsilon=CONFIG.zscore_epsilon,
        config=CONFIG,
    )

    patient_summary = patient_class_summary(window_metadata)
    patient_summary_path = (
        CONFIG.manifests_dir
        / "patient_class_summary.csv"
    )
    patient_summary.to_csv(patient_summary_path, index=False)

    print("\nPatient-level class counts:")
    print(patient_summary.to_string(index=False))

    scaler_path = (
        CONFIG.scaler_parameters_dir
        / "global_channel_zscore.json"
    )

    save_scaler(
        scaler=scaler,
        output_path=scaler_path,
    )

    print(
        "Fitted one global scaler using "
        f"{len(scaler['training_subjects'])} training patients."
    )
    print(f"Scaler file: {scaler_path}")

    # ------------------------------------------------------------------
    # Pass 4: Apply train-fitted scaling and save final shards.
    # ------------------------------------------------------------------

    print_header("PASS 4: FINAL STANDARDIZED DATASET")

    processed_manifest = write_standardized_shards(
        metadata=window_metadata,
        unscaled_recordings_dir=(
            CONFIG.unscaled_recordings_dir
        ),
        processed_data_dir=(
            CONFIG.processed_data_dir
        ),
        scaler=scaler,
        output_dtype=CONFIG.signal_dtype,
    )

    processed_manifest_path = (
        CONFIG.manifests_dir
        / "processed_shard_manifest.csv"
    )

    processed_manifest.to_csv(
        processed_manifest_path,
        index=False,
    )

    print("\nProcessed shard summary:")

    summary = (
        processed_manifest.groupby("split")
        .agg(
            shards=("recording_id", "count"),
            decisions=("number_of_decisions", "sum"),
            positive_decisions=(
                "number_of_positive_decisions",
                "sum",
            ),
            negative_decisions=(
                "number_of_negative_decisions",
                "sum",
            ),
        )
        .reset_index()
    )

    print(summary.to_string(index=False))

    print_header("PREPROCESSING COMPLETE")

    print(f"Decision manifest:  {window_manifest_path}")
    print(f"Patient summary:    {patient_summary_path}")
    print(f"Seizure manifest:   {seizure_manifest_path}")
    print(f"Scaler parameters:  {scaler_path}")
    print(
        f"Processed manifest: "
        f"{processed_manifest_path}"
    )
    print(
        f"Processed data:     "
        f"{CONFIG.processed_data_dir}"
    )

    print(
        "\nNext command:\n"
        "    python scripts/validate_dataset.py"
    )


if __name__ == "__main__":
    main()
