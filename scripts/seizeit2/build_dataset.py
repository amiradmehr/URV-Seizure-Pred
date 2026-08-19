"""
Build the filtered, labeled, patient-level split SeizeIT2 dataset.

Run from the repository root:

    python scripts/seizeit2/build_dataset.py

Pass --window-minutes/--horizon-minutes to build a different training
window or prediction horizon; the output nests under a tagged
subdirectory (e.g. data/seizeit2/processed/w30_h5/) so combinations can
coexist. Never pass --resume across two different combinations — resumed
checkpoints reuse a prior run's decision labels as-is.

Filtering each raw EDF (bandpass, notch, channel canonicalization) never
depends on the window/horizon configuration, so its output is cached once
under data/seizeit2/interim/_shared/filtered_recordings and reused by every
combination in a sweep — only the much cheaper decision-labeling, scaler,
and shard-writing passes are repeated per combination.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# Allow the script to work before or after `pip install -e .`.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIRECTORY = PROJECT_ROOT / "src"

if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))


from seizure_prediction.seizeit2.config import (  # noqa: E402
    PreprocessingConfig,
    build_config,
)
from seizure_prediction.seizeit2.preprocessing import (  # noqa: E402
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
    load_filtered_recording_cache,
    patient_class_summary,
    read_seizure_events,
    read_recording_events,
    recording_id_from_entities,
    save_decision_checkpoint,
    save_filtered_recording_cache,
    save_scaler,
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
            "Reuse validated decision checkpoints and process only "
            "missing or incomplete recordings. Never combine with a "
            "--window-minutes/--horizon-minutes value different from the "
            "run being resumed."
        ),
    )
    parser.add_argument(
        "--window-minutes",
        type=float,
        default=None,
        help="Override the training window length (default: 45 minutes).",
    )
    parser.add_argument(
        "--horizon-minutes",
        type=float,
        default=None,
        help=(
            "Override the seizure-occurrence prediction horizon "
            "(default: 10 minutes)."
        ),
    )
    return parser.parse_args()


def checkpoint_paths(
    recording_id: str,
    config: PreprocessingConfig,
) -> tuple[Path, Path, Path]:
    """Return the three required decision-checkpoint paths.

    Unlike an earlier version of this pipeline, there is no signal `.npy`
    checkpoint here -- the continuous EEG signal is never duplicated per
    combination; it lives once in the shared `filtered_recordings_dir`
    cache and `load_resumable_checkpoint` checks that directly instead.
    """
    directory = config.decision_checkpoints_dir
    return (
        directory / f"{recording_id}.csv",
        directory / f"{recording_id}_channels.json",
        directory / f"{recording_id}_channel_availability.json",
    )


def discard_checkpoint(recording_id: str, config: PreprocessingConfig) -> None:
    """Remove only the known generated files for one incomplete recording."""
    for checkpoint_path in checkpoint_paths(recording_id, config):
        checkpoint_path.unlink(missing_ok=True)


def load_resumable_checkpoint(
    recording_id: str,
    config: PreprocessingConfig,
) -> pd.DataFrame | None:
    """Return validated saved decision metadata, or discard a bad checkpoint."""
    metadata_path, channels_path, availability_path = checkpoint_paths(
        recording_id, config
    )
    paths = (metadata_path, channels_path, availability_path)

    if not any(path.exists() for path in paths):
        return None

    try:
        if not all(path.exists() for path in paths):
            raise ValueError("one or more required checkpoint files are missing")

        # The continuous signal itself is never duplicated per combination
        # -- it lives once in the shared filtered-recording cache. A
        # resumed decision checkpoint is only useful if that shared entry
        # still exists; if it was never written or was cleaned up between
        # runs, discard this checkpoint so Pass 1 regenerates it from a
        # fresh filter/decision pass.
        filtered_array_path = (
            config.filtered_recordings_dir / f"{recording_id}.npy"
        )
        if not filtered_array_path.exists():
            raise ValueError(
                "shared filtered-recording cache is missing for this recording"
            )

        with channels_path.open("r", encoding="utf-8") as channel_file:
            channel_names = json.load(channel_file)
        if channel_names != list(config.canonical_channel_names):
            raise ValueError(f"unexpected channel layout {channel_names}")

        with availability_path.open("r", encoding="utf-8") as availability_file:
            availability = np.asarray(json.load(availability_file), dtype=np.int8)
        if availability.shape != (len(config.canonical_channel_names),) or not np.isin(
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
        discard_checkpoint(recording_id, config)
        print(f"    Discarded incomplete checkpoint: {error}")
        return None
    except OSError as error:
        raise RuntimeError(
            "Could not read an existing checkpoint; it was left untouched. "
            f"Resolve the filesystem error and retry: {metadata_path} ({error})"
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
    config = build_config(arguments.window_minutes, arguments.horizon_minutes)

    print_header("SEIZEIT2 PREPROCESSING PIPELINE")

    print(f"Project root:       {config.project_root}")
    print(f"Raw dataset:        {config.raw_data_dir}")
    print(f"Interim output:     {config.interim_data_dir}")
    print(f"Processed output:   {config.processed_data_dir}")
    print(
        "Subjects:           "
        f"{len(config.included_subjects)} configured "
        "patient-level subjects"
    )

    if arguments.resume:
        config.decision_checkpoints_dir.mkdir(parents=True, exist_ok=True)
        print("Resume mode: validated existing recordings will be reused.")
    else:
        clear_generated_directory(config.decision_checkpoints_dir)

    # Final shards and aggregate manifests are rebuilt from the complete
    # checkpoint set after pass 1, whether this is a fresh run or a resume.
    clear_generated_directory(
        config.processed_data_dir
    )

    clear_generated_directory(config.manifests_dir)

    config.scaler_parameters_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    recordings = load_bids_recordings(
        config=config,
        subjects=config.included_subjects,
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
    missing_subjects = sorted(set(config.included_subjects) - loaded_subjects)

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
            load_resumable_checkpoint(recording_id, config)
            if arguments.resume
            else None
        )

        if resumed_metadata is not None:
            events_path = find_events_file(
                dataset_root=config.raw_data_dir,
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
            dataset_root=config.raw_data_dir,
            entities=entities,
        )

        seizure_events = read_seizure_events(
            events_path=events_path,
            entities=entities,
        )
        recording_events = read_recording_events(events_path)

        cached_filtered_recording = load_filtered_recording_cache(
            recording_id,
            config.filtered_recordings_dir,
            config,
        )

        if cached_filtered_recording is not None:
            processed_raw, bte_side, availability = cached_filtered_recording
            print("    Reused shared filtered-recording cache.")
        else:
            raw = recording.load_raw()
            bte_side = infer_bte_side(raw)
            availability = channel_availability_mask(raw, config)

            processed_raw = filter_and_prepare(
                raw=raw,
                config=config,
            )
            del raw

            save_filtered_recording_cache(
                raw=processed_raw,
                bte_side=bte_side,
                channel_availability=availability,
                output_directory=config.filtered_recordings_dir,
                recording_id=recording_id,
                output_dtype=config.signal_dtype,
            )

        metadata_recording, seizure_metadata_recording = (
            create_labeled_prediction_decisions(
                raw=processed_raw,
                seizure_events=seizure_events,
                recording_events=recording_events,
                entities=entities,
                bte_side=bte_side,
                config=config,
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

        save_decision_checkpoint(
            decision_metadata=metadata_recording,
            channel_names=processed_raw.ch_names,
            channel_availability=availability,
            output_directory=(
                config.decision_checkpoints_dir
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
            config=config,
        )
    )

    verify_patient_split_isolation(window_metadata, config)

    window_manifest_path = (
        config.manifests_dir
        / "decision_manifest.csv"
    )

    seizure_manifest_path = (
        config.manifests_dir
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
        filtered_recordings_dir=config.filtered_recordings_dir,
        epsilon=config.zscore_epsilon,
        config=config,
    )

    patient_summary = patient_class_summary(window_metadata)
    patient_summary_path = (
        config.manifests_dir
        / "patient_class_summary.csv"
    )
    patient_summary.to_csv(patient_summary_path, index=False)

    print("\nPatient-level class counts:")
    print(patient_summary.to_string(index=False))

    scaler_path = (
        config.scaler_parameters_dir
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

    already_cached = (
        config.standardized_recordings_dir.exists()
        and any(config.standardized_recordings_dir.iterdir())
    )
    print(f"Standardized recordings:    {config.standardized_recordings_dir}")
    print(
        "  "
        + (
            "Reusing standardized recordings already cached by a prior "
            "combination in this sweep where available; only recordings "
            "not yet cached will be standardized now."
            if already_cached
            else "No existing cache yet; standardizing every recording now."
        )
    )

    processed_manifest = write_standardized_shards(
        metadata=window_metadata,
        filtered_recordings_dir=(
            config.filtered_recordings_dir
        ),
        processed_data_dir=(
            config.processed_data_dir
        ),
        scaler=scaler,
        output_dtype=config.signal_dtype,
        project_root=config.project_root,
        standardized_recordings_dir=config.standardized_recordings_dir,
    )

    processed_manifest_path = (
        config.manifests_dir
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
        f"{config.processed_data_dir}"
    )

    sweep_flags = ""
    if arguments.window_minutes is not None:
        sweep_flags += f" --window-minutes {arguments.window_minutes:g}"
    if arguments.horizon_minutes is not None:
        sweep_flags += f" --horizon-minutes {arguments.horizon_minutes:g}"

    print(
        "\nNext command:\n"
        f"    python scripts/seizeit2/validate_dataset.py{sweep_flags}"
    )


if __name__ == "__main__":
    main()
