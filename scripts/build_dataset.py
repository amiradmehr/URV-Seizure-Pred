"""
Build the filtered, labeled, patient-level split SeizeIT2 dataset.

Run from the repository root:

    python scripts/build_dataset.py

By default this builds every combination of `INPUT_WINDOW_MINUTE_CHOICES` and
`SEIZURE_OCCURRENCE_PERIOD_MINUTE_CHOICES`; `--windows` and `--horizons`
narrow that.  Each combination gets its own tagged manifests and shards under
`data/seizeit2/{interim,processed}/w{window}_h{horizon}/`.

Only the decision indices and their labels differ between combinations, so
each EDF is opened and filtered exactly once and every combination is labeled
against that one in-memory copy.  The filtered EEG is written once to
`data/seizeit2/interim/_shared/unscaled_recordings/` and shared by all of
them, as are the fitted scalers.  Building twelve combinations therefore
costs roughly the wall time and disk of building one.
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd


# Allow the script to work before or after `pip install -e .`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIRECTORY = PROJECT_ROOT / "src"

if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))


from seizure_prediction.config import (  # noqa: E402
    CONFIG,
    INPUT_WINDOW_MINUTE_CHOICES,
    SEIZURE_OCCURRENCE_PERIOD_MINUTE_CHOICES,
    PreprocessingConfig,
    sweep_configurations,
)
from seizure_prediction.normalization import (  # noqa: E402
    NORMALIZATION_MODES,
    STATISTICS,
    build_scaler_document,
    fit_channel_scaler,
    save_scaler_document,
    scaler_key_for,
)
from seizure_prediction.preprocessing import (  # noqa: E402
    assign_patient_splits,
    channel_availability_mask,
    class_summary,
    clear_generated_directory,
    extract_recording_entities,
    filter_and_prepare,
    filtered_recording_paths,
    find_events_file,
    infer_bte_side,
    load_bids_recordings,
    load_filtered_recording,
    patient_class_summary,
    read_seizure_events,
    read_recording_events,
    recording_id_from_entities,
    normalize_entity,
    save_filtered_recording,
    seizure_scope_summary,
    verify_patient_split_isolation,
    write_decision_shards,
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
            "Reuse validated per-combination decision checkpoints and label "
            "only the recordings still missing one. Cached filtered "
            "recordings are reused either way, since they do not depend on "
            "the window or horizon."
        ),
    )
    parser.add_argument(
        "--normalization",
        choices=NORMALIZATION_MODES,
        default="patient",
        help=(
            "Scope of the channel z-score. 'patient' gives every patient their "
            "own center/scale, which stops between-patient amplitude "
            "differences from dominating the input. 'global' reproduces the "
            "original single train-fitted transform."
        ),
    )
    parser.add_argument(
        "--statistic",
        choices=STATISTICS,
        default="meanstd",
        help=(
            "meanstd is the classic z-score. robust uses median/IQR, which "
            "resists the movement and electrode-pop artifacts common in "
            "wearable EEG."
        ),
    )
    parser.add_argument(
        "--windows",
        type=float,
        nargs="+",
        default=None,
        metavar="MINUTES",
        help=(
            "Input windows to build, in minutes (default: "
            f"{' '.join(f'{value:g}' for value in INPUT_WINDOW_MINUTE_CHOICES)})."
        ),
    )
    parser.add_argument(
        "--horizons",
        type=float,
        nargs="+",
        default=None,
        metavar="MINUTES",
        help=(
            "Seizure occurrence periods to build, in minutes (default: "
            f"{' '.join(f'{value:g}' for value in SEIZURE_OCCURRENCE_PERIOD_MINUTE_CHOICES)})."
        ),
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        metavar="ID",
        help=(
            "Restrict the build to these subject IDs, without the 'sub-' "
            "prefix (for example: 001 007 042). The full configured split is "
            "used when omitted."
        ),
    )
    return parser.parse_args()


def decision_checkpoint_paths(
    recording_id: str,
    config: PreprocessingConfig,
) -> tuple[Path, Path]:
    """
    Return the decision-checkpoint paths for one recording and label definition.

    There is no signal checkpoint here.  The filtered EEG lives once in the
    shared cache and is validated by `load_filtered_recording`; what is
    checkpointed per combination is only the decision table it produced, which
    is the part a different input window or horizon actually changes.  Because
    these paths sit under the combination's own tag, resuming can never mix
    decision rows built at one geometry into a build at another.
    """
    directory = config.decision_checkpoints_dir
    return (
        directory / f"{recording_id}.csv",
        directory / f"{recording_id}_seizures.csv",
    )


def save_decision_checkpoint(
    recording_id: str,
    config: PreprocessingConfig,
    decision_metadata: pd.DataFrame,
    seizure_metadata: pd.DataFrame,
) -> None:
    """Save one recording's decisions and seizure eligibility for one tag."""
    decision_path, seizure_path = decision_checkpoint_paths(recording_id, config)
    decision_path.parent.mkdir(parents=True, exist_ok=True)
    decision_metadata.to_csv(decision_path, index=False)
    seizure_metadata.to_csv(seizure_path, index=False)


def discard_decision_checkpoint(
    recording_id: str,
    config: PreprocessingConfig,
) -> None:
    """Remove only the known generated files for one incomplete checkpoint."""
    for checkpoint_path in decision_checkpoint_paths(recording_id, config):
        checkpoint_path.unlink(missing_ok=True)


def load_resumable_checkpoint(
    recording_id: str,
    config: PreprocessingConfig,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """
    Return validated saved decision and seizure metadata for one tag.

    The geometry columns are checked against ``config``, so a checkpoint that
    somehow predates a geometry change is discarded rather than silently
    contributing decisions whose history length no longer matches the window
    the model will read.
    """
    decision_path, seizure_path = decision_checkpoint_paths(recording_id, config)
    paths = (decision_path, seizure_path)

    if not any(path.exists() for path in paths):
        return None

    entity_dtypes = {
        "recording_id": str,
        "subject": str,
        "session": str,
        "task": str,
        "run": str,
    }

    try:
        if not all(path.exists() for path in paths):
            raise ValueError("one or more required checkpoint files are missing")

        metadata = pd.read_csv(decision_path, dtype=entity_dtypes)
        seizure_metadata = pd.read_csv(seizure_path, dtype=entity_dtypes)

        required_columns = {
            "recording_id",
            "label",
            "target_seizure_id",
            "history_start_sample",
            "decision_end_sample",
            "chunk_samples",
            "chunks_per_history",
        }
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

        expected_chunk_samples = int(
            round(config.chunk_window_seconds * config.target_sfreq)
        )
        expected_history_samples = int(
            round(config.input_window_seconds * config.target_sfreq)
        )
        expected_chunks = expected_history_samples // expected_chunk_samples

        if not metadata["chunk_samples"].eq(expected_chunk_samples).all():
            raise ValueError("checkpoint chunk_samples does not match the config")
        if not metadata["chunks_per_history"].eq(expected_chunks).all():
            raise ValueError("checkpoint chunks_per_history does not match the config")

        history_lengths = (
            metadata["decision_end_sample"] - metadata["history_start_sample"]
        )
        if not history_lengths.eq(expected_history_samples).all():
            raise ValueError(
                "checkpoint decision history length does not match the "
                f"configured {config.input_window_seconds / 60.0:g}-minute window"
            )

        return metadata, seizure_metadata
    except (ValueError, KeyError, json.JSONDecodeError, EOFError) as error:
        discard_decision_checkpoint(recording_id, config)
        print(f"      Discarded incomplete checkpoint: {error}")
        return None
    except OSError as error:
        raise RuntimeError(
            "Could not read an existing checkpoint; it was left untouched. "
            f"Resolve the filesystem error and retry: {decision_path} ({error})"
        ) from error


def fit_scalers_for_build(
    recordings: pd.DataFrame,
    mode: str,
    statistic: str,
) -> dict:
    """Fit every scaler the requested normalization mode needs.

    ``global`` fits one transform from training patients only, preserving the
    original train/validation/test asymmetry. ``patient`` and ``recording`` fit
    each group from its own recordings, so validation and test patients are
    normalized by their own statistics and never by another patient's.

    The fit reads whole filtered recordings and so does not depend on the
    input window or the seizure occurrence period.  ``recordings`` is
    therefore the union of every recording any combination in this build
    kept, and the resulting document is shared by all of them.
    """
    recordings = recordings.drop_duplicates(subset="recording_id").copy()
    recordings["subject"] = recordings["subject"].map(normalize_entity)

    if mode == "global":
        contributing = recordings[recordings["split"] == "train"]
        if contributing.empty:
            raise ValueError("There are no training recordings to fit a scaler.")
    else:
        contributing = recordings

    availability_by_path: dict[Path, np.ndarray] = {}
    for recording_id in contributing["recording_id"]:
        array_path, _, availability_path, _ = filtered_recording_paths(
            CONFIG.unscaled_recordings_dir,
            str(recording_id),
        )
        with availability_path.open("r", encoding="utf-8") as availability_file:
            availability_by_path[array_path] = np.asarray(
                json.load(availability_file),
                dtype=bool,
            )

    contributing = contributing.assign(
        scaler_key=[
            scaler_key_for(
                mode,
                subject=row.subject,
                recording_id=row.recording_id,
            )
            for row in contributing.itertuples(index=False)
        ]
    )

    scalers: dict[str, dict] = {}
    groups = list(contributing.groupby("scaler_key", sort=True))
    for group_number, (scaler_key, group) in enumerate(groups, start=1):
        array_paths = [
            filtered_recording_paths(
                CONFIG.unscaled_recordings_dir,
                str(recording_id),
            )[0]
            for recording_id in group["recording_id"]
        ]
        scalers[scaler_key] = fit_channel_scaler(
            array_paths,
            availability_by_path,
            channel_names=list(CONFIG.canonical_channel_names),
            statistic=statistic,
            epsilon=CONFIG.zscore_epsilon,
        )
        print(
            f"[{group_number:04d}/{len(groups):04d}] {scaler_key}: "
            f"{len(array_paths)} recording(s)"
        )

    if mode != "global":
        # Every recording must be covered, including validation and test.
        missing = [
            scaler_key_for(
                mode,
                subject=row.subject,
                recording_id=row.recording_id,
            )
            for row in recordings.itertuples(index=False)
        ]
        uncovered = sorted(set(missing) - set(scalers))
        if uncovered:
            raise ValueError(
                f"No {mode} scaler was fitted for: {uncovered[:5]}"
            )

    document = build_scaler_document(
        mode=mode,
        statistic=statistic,
        channel_names=list(CONFIG.canonical_channel_names),
        scalers=scalers,
        epsilon=CONFIG.zscore_epsilon,
        training_subjects=(
            sorted(set(contributing["subject"])) if mode == "global" else None
        ),
    )
    # Recorded so a later build can tell whether this shared document already
    # covers the recordings it needs, or has to be refitted because a shorter
    # window kept a recording that no earlier combination did.
    document["covered_recording_ids"] = sorted(
        str(recording_id) for recording_id in recordings["recording_id"]
    )
    return document


def load_reusable_scaler_document(
    scaler_path: Path,
    required_recording_ids: set[str],
    mode: str,
    statistic: str,
) -> dict | None:
    """
    Return a previously fitted shared scaler document, if it still applies.

    Fitting reads every filtered recording end to end, so reusing the document
    across combinations saves a full pass over the whole dataset per
    combination.  It is only safe when the saved document covers every
    recording this build needs; a shorter window keeps recordings a longer one
    was too short for, so coverage genuinely can grow between combinations.
    """
    if not scaler_path.exists():
        return None

    try:
        with scaler_path.open("r", encoding="utf-8") as scaler_file:
            document = json.load(scaler_file)
    except (OSError, json.JSONDecodeError):
        return None

    if (
        document.get("normalization_mode") != mode
        or document.get("statistic") != statistic
    ):
        return None

    covered = set(document.get("covered_recording_ids", []))

    if not required_recording_ids.issubset(covered):
        return None

    return document


def label_recording_for_combination(
    processed_raw,
    seizure_events: pd.DataFrame,
    recording_events: pd.DataFrame,
    entities: dict[str, str],
    bte_side: str,
    config: PreprocessingConfig,
    recording_id: str,
    resume: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Produce one combination's decisions for an already filtered recording.

    Returns the decision table and the seizure table carrying this
    combination's eligibility verdicts.  Both are checkpointed under the
    combination's tag so a later resume skips the labeling scan.
    """
    if resume:
        resumed = load_resumable_checkpoint(recording_id, config)
        if resumed is not None:
            return resumed

    decision_metadata, seizure_metadata = create_labeled_prediction_decisions(
        raw=processed_raw,
        seizure_events=seizure_events,
        recording_events=recording_events,
        entities=entities,
        bte_side=bte_side,
        config=config,
    )

    if not seizure_metadata.empty:
        seizure_metadata = seizure_metadata.copy()
        seizure_metadata["eligibility_evaluated"] = True

    save_decision_checkpoint(
        recording_id=recording_id,
        config=config,
        decision_metadata=decision_metadata,
        seizure_metadata=seizure_metadata,
    )

    return decision_metadata, seizure_metadata


def build_combination_outputs(
    config: PreprocessingConfig,
    decision_metadata: pd.DataFrame,
    seizure_metadata: pd.DataFrame,
    scaler_document: dict,
    scaler_document_path: Path,
) -> dict[str, Path]:
    """
    Write one combination's splits, manifests, and decision shards.

    No EEG is written here.  Every shard references the shared filtered
    recording, and standardization happens when a decision is loaded.
    """
    decision_metadata = assign_patient_splits(
        metadata=decision_metadata,
        config=config,
    )

    verify_patient_split_isolation(decision_metadata, config)

    decision_manifest_path = config.manifests_dir / "decision_manifest.csv"
    seizure_manifest_path = config.manifests_dir / "seizure_manifest.csv"
    patient_summary_path = config.manifests_dir / "patient_class_summary.csv"
    processed_manifest_path = config.manifests_dir / "processed_shard_manifest.csv"

    decision_metadata.to_csv(decision_manifest_path, index=False)
    seizure_metadata.to_csv(seizure_manifest_path, index=False)

    patient_summary = patient_class_summary(decision_metadata)
    patient_summary.to_csv(patient_summary_path, index=False)

    processed_manifest = write_decision_shards(
        metadata=decision_metadata,
        unscaled_recordings_dir=config.unscaled_recordings_dir,
        processed_data_dir=config.processed_data_dir,
        scaler_document=scaler_document,
        scaler_document_path=scaler_document_path,
        project_root=config.project_root,
    )

    processed_manifest.to_csv(processed_manifest_path, index=False)

    print("\n  Decision class counts:")
    print(
        textwrap.indent(
            class_summary(decision_metadata).to_string(index=False),
            "    ",
        )
    )

    scope_summary = seizure_scope_summary(decision_metadata)

    print("\n  Positive-decision seizure-scope counts:")
    if scope_summary.empty:
        print("    No positive decisions were generated.")
    else:
        print(textwrap.indent(scope_summary.to_string(index=False), "    "))

    unknown_positive_count = int(
        (
            (decision_metadata["label"] == 1)
            & (decision_metadata["seizure_scope"] == "unknown")
        ).sum()
    )

    if unknown_positive_count > 0:
        print(
            "\n  WARNING: "
            f"{unknown_positive_count} positive decisions have unknown "
            "local/cross scope. The event files did not contain a recognized "
            "authoritative scope column. Do not guess these labels; add them "
            "from the trusted SeizeIT2 annotation source."
        )

    summary = (
        processed_manifest.groupby("split")
        .agg(
            shards=("recording_id", "count"),
            decisions=("number_of_decisions", "sum"),
            positive_decisions=("number_of_positive_decisions", "sum"),
            negative_decisions=("number_of_negative_decisions", "sum"),
        )
        .reset_index()
    )

    print("\n  Decision shard summary:")
    print(textwrap.indent(summary.to_string(index=False), "    "))

    return {
        "decision_manifest": decision_manifest_path,
        "seizure_manifest": seizure_manifest_path,
        "patient_summary": patient_summary_path,
        "processed_manifest": processed_manifest_path,
    }


def main() -> None:
    """Execute the complete preprocessing workflow for every combination."""
    arguments = parse_arguments()
    CONFIG.validate()

    configs = sweep_configurations(
        input_window_minutes=arguments.windows,
        seizure_occurrence_period_minutes=arguments.horizons,
    )

    subjects = (
        tuple(CONFIG.included_subjects)
        if arguments.subjects is None
        else tuple(normalize_entity(subject) for subject in arguments.subjects)
    )

    print_header("SEIZEIT2 PREPROCESSING PIPELINE")

    print(f"Project root:       {CONFIG.project_root}")
    print(f"Raw dataset:        {CONFIG.raw_data_dir}")
    print(f"Shared recordings:  {CONFIG.unscaled_recordings_dir}")
    print(f"Shared scalers:     {CONFIG.scaler_parameters_dir}")
    print(f"Subjects:           {len(subjects)} patient-level subjects")
    print(
        "Combinations:       "
        f"{len(configs)} "
        f"({', '.join(config.experiment_tag for config in configs)})"
    )
    print(
        "\nThe filtered EEG is written once and shared by every combination; "
        "only\ndecision indices and labels are built per combination."
    )

    CONFIG.unscaled_recordings_dir.mkdir(parents=True, exist_ok=True)
    CONFIG.scaler_parameters_dir.mkdir(parents=True, exist_ok=True)

    for config in configs:
        # Manifests and shards are always rebuilt from the complete checkpoint
        # set, whether this is a fresh run or a resume. The shared filtered
        # recordings and scalers are never cleared.
        clear_generated_directory(config.processed_data_dir)
        clear_generated_directory(config.manifests_dir)

        if arguments.resume:
            config.decision_checkpoints_dir.mkdir(parents=True, exist_ok=True)
        else:
            clear_generated_directory(config.decision_checkpoints_dir)

    if arguments.resume:
        print("\nResume mode: validated existing checkpoints will be reused.")

    recordings = load_bids_recordings(
        config=CONFIG,
        subjects=subjects,
        preload=False,
    )

    print(f"\nEDF recordings found: {len(recordings)}")

    loaded_subjects = {
        extract_recording_entities(recording.description)["subject"]
        for recording in recordings
    }
    missing_subjects = sorted(set(subjects) - loaded_subjects)

    if missing_subjects:
        raise RuntimeError(
            "The requested subjects include ones with no loaded EEG "
            f"recordings: {missing_subjects}"
        )

    decision_metadata_by_tag: dict[str, list[pd.DataFrame]] = {
        config.experiment_tag: [] for config in configs
    }
    seizure_metadata_by_tag: dict[str, list[pd.DataFrame]] = {
        config.experiment_tag: [] for config in configs
    }
    kept_recordings: list[dict[str, str]] = []
    reused_recordings = 0
    filtered_recordings = 0

    # ------------------------------------------------------------------
    # Pass 1: Filter each EDF once, then label it for every combination.
    # ------------------------------------------------------------------

    print_header("PASS 1: NATIVE-RATE FILTERING AND DECISION LABELING")

    for recording_number, recording in enumerate(recordings, start=1):
        entities = extract_recording_entities(recording.description)
        recording_id = recording_id_from_entities(entities)

        print(f"[{recording_number:04d}/{len(recordings):04d}] {recording_id}")

        events_path = find_events_file(
            dataset_root=CONFIG.raw_data_dir,
            entities=entities,
        )
        seizure_events = read_seizure_events(
            events_path=events_path,
            entities=entities,
        )
        recording_events = read_recording_events(events_path)

        cached = load_filtered_recording(
            CONFIG.unscaled_recordings_dir,
            recording_id,
            CONFIG,
        )

        if cached is not None:
            processed_raw, bte_side, availability = cached
            reused_recordings += 1
            print("    Reused shared filtered recording.")
        else:
            raw = recording.load_raw()
            bte_side = infer_bte_side(raw)
            availability = channel_availability_mask(raw, CONFIG)

            processed_raw = filter_and_prepare(raw=raw, config=CONFIG)
            del raw

            save_filtered_recording(
                raw=processed_raw,
                channel_availability=availability,
                bte_side=bte_side,
                output_directory=CONFIG.unscaled_recordings_dir,
                recording_id=recording_id,
                output_dtype=CONFIG.signal_dtype,
            )
            filtered_recordings += 1

            print(f"    Channels: {processed_raw.ch_names}")
            print(f"    Availability mask: {availability.astype(int).tolist()}")
            print(f"    Sampling frequency: {processed_raw.info['sfreq']} Hz")

        print(f"    Seizures in recording: {len(seizure_events)}")

        kept_for_any_combination = False

        for config in configs:
            decision_metadata, seizure_metadata = label_recording_for_combination(
                processed_raw=processed_raw,
                seizure_events=seizure_events,
                recording_events=recording_events,
                entities=entities,
                bte_side=bte_side,
                config=config,
                recording_id=recording_id,
                resume=arguments.resume,
            )

            eligible_count = (
                int(
                    seizure_metadata["eligible_for_prediction"]
                    .fillna(False)
                    .astype(bool)
                    .sum()
                )
                if "eligible_for_prediction" in seizure_metadata.columns
                and not seizure_metadata.empty
                else 0
            )

            print(
                f"    {config.experiment_tag:>8}: "
                f"{len(decision_metadata):6d} decisions, "
                f"{int(decision_metadata['label'].sum()) if not decision_metadata.empty else 0:4d} positive, "
                f"{eligible_count} eligible seizure(s)"
            )

            if decision_metadata.empty:
                continue

            kept_for_any_combination = True
            decision_metadata_by_tag[config.experiment_tag].append(decision_metadata)

            if not seizure_metadata.empty:
                seizure_metadata_by_tag[config.experiment_tag].append(seizure_metadata)

        if kept_for_any_combination:
            kept_recordings.append(
                {
                    "recording_id": recording_id,
                    "subject": entities["subject"],
                }
            )
        else:
            print("    No usable decision points for any combination.")

        # Explicitly release potentially large objects.
        del processed_raw

    print(
        f"\nFiltering result: reused {reused_recordings} cached recording(s); "
        f"filtered {filtered_recordings} recording(s)."
    )

    if not kept_recordings:
        raise RuntimeError("No usable prediction decision points were generated.")

    # ------------------------------------------------------------------
    # Pass 2: Fit the channel scalers once for the whole sweep.
    # ------------------------------------------------------------------

    print_header(
        f"PASS 2: {arguments.normalization.upper()} "
        f"{arguments.statistic.upper()} SCALER FITTING"
    )

    kept_frame = pd.DataFrame(kept_recordings).drop_duplicates(
        subset="recording_id"
    )
    kept_frame["split"] = kept_frame["subject"].map(CONFIG.subject_split_map)

    if kept_frame["split"].isna().any():
        unmapped = sorted(
            kept_frame.loc[kept_frame["split"].isna(), "subject"].unique()
        )
        raise ValueError(
            f"These subjects are not in any configured split: {unmapped}"
        )

    scaler_path = CONFIG.scaler_document_path(
        arguments.normalization,
        arguments.statistic,
    )

    scaler_document = load_reusable_scaler_document(
        scaler_path=scaler_path,
        required_recording_ids=set(kept_frame["recording_id"]),
        mode=arguments.normalization,
        statistic=arguments.statistic,
    )

    if scaler_document is None:
        scaler_document = fit_scalers_for_build(
            recordings=kept_frame,
            mode=arguments.normalization,
            statistic=arguments.statistic,
        )
        save_scaler_document(scaler_document, scaler_path)
        print(
            f"\nFitted {scaler_document['scaler_count']} "
            f"{arguments.normalization} scaler(s) "
            f"({arguments.statistic}) over {scaler_document['fitted_on']}."
        )
    else:
        print(
            f"\nReused {scaler_document['scaler_count']} cached "
            f"{arguments.normalization} scaler(s) ({arguments.statistic}); "
            "the saved document already covers every recording in this build."
        )

    if scaler_document["non_causal_statistics"]:
        print(
            "Note: these statistics summarize whole recordings, so the "
            "scaling of an early decision reflects later samples. A deployed "
            "system would calibrate on a prefix instead."
        )

    print(f"Scaler file: {scaler_path}")

    # ------------------------------------------------------------------
    # Pass 3: Split, manifest, and shard each combination.
    # ------------------------------------------------------------------

    output_paths_by_tag: dict[str, dict[str, Path]] = {}

    for config in configs:
        print_header(f"PASS 3: DECISION SHARDS FOR {config.experiment_tag}")

        tag_decisions = decision_metadata_by_tag[config.experiment_tag]

        if not tag_decisions:
            raise RuntimeError(
                "No usable prediction decision points were generated for "
                f"{config.experiment_tag}."
            )

        tag_seizures = seizure_metadata_by_tag[config.experiment_tag]

        output_paths_by_tag[config.experiment_tag] = build_combination_outputs(
            config=config,
            decision_metadata=pd.concat(tag_decisions, ignore_index=True),
            seizure_metadata=(
                pd.concat(tag_seizures, ignore_index=True)
                if tag_seizures
                else pd.DataFrame()
            ),
            scaler_document=scaler_document,
            scaler_document_path=scaler_path,
        )

    print_header("PREPROCESSING COMPLETE")

    print(f"Shared recordings:  {CONFIG.unscaled_recordings_dir}")
    print(f"Scaler parameters:  {scaler_path}")
    print("\nPer-combination output:")

    for config in configs:
        print(f"  {config.experiment_tag:>8}: {config.processed_data_dir}")

    print(
        "\nNext command:\n"
        "    python scripts/validate_dataset.py "
        f"--window-minutes {configs[0].input_window_seconds / 60.0:g} "
        f"--horizon-minutes {configs[0].seizure_occurrence_period_minutes:g}"
    )


if __name__ == "__main__":
    main()
