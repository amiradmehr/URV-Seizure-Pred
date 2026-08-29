r"""Cache robust AVA-style features for whole-minute EEG windows.

PowerShell example:

    .\.venv-old\Scripts\python.exe scripts\cache_handcrafted_features.py
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIRECTORY = PROJECT_ROOT / "src"
if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

from seizure_prediction.config import (  # noqa: E402
    CONFIG,
    add_label_definition_arguments,
    resolve_label_definition,
)
from seizure_prediction.datasets import (  # noqa: E402
    PatientEventBalancedEpochSampler,
    load_decision_examples,
    load_scaler_document,
    resolve_stored_path,
)
from seizure_prediction.normalization import (  # noqa: E402
    apply_channel_scaler,
    select_scaler,
)
from seizure_prediction.handcrafted_features import (  # noqa: E402
    FEATURE_NAMES,
    extract_ava_feature_batch,
    feature_cache_path,
    feature_coverage_path,
    load_channel_availability,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=(
            CONFIG.handcrafted_feature_cache_dir
            / "ava_minute_selected_v1"
        ),
        help=(
            "Features are extracted per whole minute of a recording, so the "
            "cache does not depend on the input window or horizon and is "
            "shared by every label definition."
        ),
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "validation", "test"),
        default=("train", "validation"),
    )
    parser.add_argument("--minute-batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--entropy-bins", type=int, default=64)
    parser.add_argument(
        "--selection-sampling-epochs",
        type=int,
        default=5,
        help=(
            "For training recordings, cache only histories selected by this "
            "many patient/event-balanced epochs. Validation remains complete."
        ),
    )
    parser.add_argument("--negative-to-positive-ratio", type=float, default=10.0)
    parser.add_argument("--max-events-per-patient", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--subjects",
        nargs="*",
        default=(),
        help=(
            "Restrict caching to these subjects. Without it every "
            "recording in the selected splits is processed."
        ),
    )
    parser.add_argument(
        "--complete-subjects",
        nargs="*",
        default=(),
        help=(
            "Cache every minute of these subjects' recordings instead of "
            "only the sampler-selected histories. Required for per-patient "
            "analyses that use all of a patient's decisions."
        ),
    )
    add_label_definition_arguments(parser)
    return parser.parse_args()


def cache_is_valid(
    path: Path,
    coverage_path: Path,
    expected_coverage: np.ndarray,
    channels: int,
) -> bool:
    try:
        cached = np.load(path, mmap_mode="r")
        coverage = np.load(coverage_path, mmap_mode="r")
    except (OSError, ValueError):
        return False
    return (
        cached.shape == (len(expected_coverage), channels, len(FEATURE_NAMES))
        and cached.dtype == np.float32
        and coverage.shape == expected_coverage.shape
        and np.array_equal(coverage, expected_coverage)
        and np.isfinite(cached[expected_coverage]).all()
    )


def cache_recording(
    *,
    signal_path: Path,
    availability_path: Path,
    output_path: Path,
    coverage_path: Path,
    required_coverage: np.ndarray,
    minute_batch_size: int,
    entropy_bins: int,
    overwrite: bool,
    sampling_frequency: float,
    channel_names: list[str],
    scaler: dict,
    scaler_document: dict,
) -> tuple[int, bool]:
    """Extract and cache per-minute features for one recording.

    The stored recording is filtered but not standardized, so each minute
    batch is standardized here with the same per-channel affine map the
    training loader applies.
    """
    eeg = np.load(signal_path, mmap_mode="r")
    availability = load_channel_availability(availability_path)
    if eeg.ndim != 2 or availability.shape != (eeg.shape[0],):
        raise ValueError(f"Invalid signal or availability shape for {signal_path}")
    samples_per_minute = int(round(sampling_frequency * 60.0))
    minute_count = eeg.shape[1] // samples_per_minute
    if minute_count == 0:
        raise ValueError(f"Recording is shorter than one minute: {signal_path}")
    if required_coverage.shape != (minute_count,):
        raise ValueError(f"Required feature coverage has the wrong shape: {signal_path}")
    required_minute_count = int(required_coverage.sum())
    if required_minute_count == 0:
        return 0, True
    if output_path.exists() and coverage_path.exists() and not overwrite:
        if cache_is_valid(
            output_path, coverage_path, required_coverage, eeg.shape[0]
        ):
            return required_minute_count, True
        raise ValueError(
            f"Existing cache is incompatible: {output_path}. Pass --overwrite."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".tmp.npy")
    cached = np.lib.format.open_memmap(
        temporary_path,
        mode="w+",
        dtype=np.float32,
        shape=(minute_count, eeg.shape[0], len(FEATURE_NAMES)),
    )
    cached[:] = 0.0
    try:
        required_indices = np.flatnonzero(required_coverage)
        run_starts = np.concatenate(
            [np.array([0]), np.flatnonzero(np.diff(required_indices) > 1) + 1]
        )
        run_stops = np.concatenate([run_starts[1:], np.array([len(required_indices)])])
        for run_start_position, run_stop_position in zip(run_starts, run_stops):
            run = required_indices[run_start_position:run_stop_position]
            first_minute = int(run[0])
            stop_minute = int(run[-1]) + 1
            for start in range(first_minute, stop_minute, minute_batch_size):
                stop = min(start + minute_batch_size, stop_minute)
                sample_start = start * samples_per_minute
                sample_stop = stop * samples_per_minute
                batch = apply_channel_scaler(
                    X=np.asarray(
                        eeg[:, sample_start:sample_stop],
                        dtype=np.float32,
                    ),
                    channel_names=channel_names,
                    channel_availability=availability,
                    scaler=scaler,
                    document=scaler_document,
                    output_dtype="float32",
                )
                windows = np.ascontiguousarray(
                    batch.reshape(eeg.shape[0], stop - start, samples_per_minute)
                    .transpose(1, 0, 2)
                )
                cached[start:stop] = extract_ava_feature_batch(
                    windows,
                    availability,
                    sampling_frequency=sampling_frequency,
                    entropy_bins=entropy_bins,
                )
        cached.flush()
        del cached
        temporary_path.replace(output_path)
        coverage_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(coverage_path, required_coverage.astype(bool, copy=False))
    except BaseException:
        del cached
        temporary_path.unlink(missing_ok=True)
        raise
    return required_minute_count, False


def run_cache_job(job: dict[str, object]) -> tuple[int, bool]:
    """Execute one picklable per-recording cache job."""
    return cache_recording(**job)  # type: ignore[arg-type]


def required_minute_coverage(
    examples: pd.DataFrame,
    manifest: pd.DataFrame,
    config,
    complete_subjects: set[str] | None = None,
) -> dict[str, np.ndarray]:
    """Return exact whole-minute coverage needed by the selected decisions.

    ``complete_subjects`` forces every minute of a subject's recordings to be
    cached, rather than only the minutes the balanced sampler happened to
    select. Per-patient analyses such as leave-one-recording-out need all of a
    patient's decisions, not a training subsample.
    """
    samples_per_minute = int(round(config.target_sfreq * 60.0))
    history_minutes = int(round(config.input_window_seconds / 60.0))
    complete_subjects = {
        str(subject).zfill(3) for subject in (complete_subjects or set())
    }
    signal_minutes = {
        str(resolve_stored_path(row.X_path, config.project_root)): (
            np.load(
                resolve_stored_path(row.X_path, config.project_root),
                mmap_mode="r",
            ).shape[1]
            // samples_per_minute
        )
        for row in manifest.itertuples(index=False)
    }
    coverage = {
        path: np.zeros(minutes, dtype=bool) for path, minutes in signal_minutes.items()
    }
    for row in examples.itertuples(index=False):
        path = str(resolve_stored_path(row.X_path, config.project_root))
        start = int(row.history_start_sample) // samples_per_minute
        stop = int(row.decision_end_sample) // samples_per_minute
        if path not in coverage or stop - start != history_minutes:
            raise ValueError(
                "Selected decision does not match a cache recording, or its "
                f"history is not the configured {history_minutes} minutes."
            )
        coverage[path][start:stop] = True

    if complete_subjects:
        for row in manifest.itertuples(index=False):
            if str(row.subject).zfill(3) not in complete_subjects:
                continue
            path = str(resolve_stored_path(row.X_path, config.project_root))
            coverage[path][:] = True

    return coverage


def main() -> None:
    arguments = parse_arguments()
    config = resolve_label_definition(arguments)
    config.validate()
    if arguments.minute_batch_size <= 0:
        raise ValueError("minute-batch-size must be positive.")
    if arguments.entropy_bins < 2:
        raise ValueError("entropy-bins must be at least two.")
    if arguments.selection_sampling_epochs <= 0:
        raise ValueError("selection-sampling-epochs must be positive.")
    if arguments.workers <= 0:
        raise ValueError("workers must be positive.")

    manifest_path = config.manifests_dir / "processed_shard_manifest.csv"
    manifest = pd.read_csv(
        manifest_path,
        dtype={"subject": str, "recording_id": str},
    )
    scaler_document = load_scaler_document(manifest_path, config.project_root)
    manifest = manifest[manifest["split"].isin(arguments.splits)].copy()
    if manifest.empty:
        raise ValueError(f"No recordings found for splits {arguments.splits}.")
    recordings = manifest.drop_duplicates(subset=["X_path"]).reset_index(drop=True)
    selected_examples: list[pd.DataFrame] = []
    if "train" in arguments.splits:
        train_examples = load_decision_examples(
            manifest_path,
            split="train",
            negative_to_positive_ratio=None,
            seed=arguments.seed,
            project_root=config.project_root,
        )
        sampler = PatientEventBalancedEpochSampler(
            train_examples,
            negative_to_positive_ratio=arguments.negative_to_positive_ratio,
            max_events_per_patient=arguments.max_events_per_patient,
            seed=arguments.seed,
        )
        selected_indices = np.concatenate(
            [
                sampler.indices_for_epoch(epoch)
                for epoch in range(arguments.selection_sampling_epochs)
            ]
        )
        selected_examples.append(train_examples.iloc[selected_indices])
    if "validation" in arguments.splits:
        selected_examples.append(
            load_decision_examples(
                manifest_path,
                split="validation",
                negative_to_positive_ratio=None,
                seed=arguments.seed,
                project_root=config.project_root,
            )
        )
    if "test" in arguments.splits:
        selected_examples.append(
            load_decision_examples(
                manifest_path,
                split="test",
                negative_to_positive_ratio=None,
                seed=arguments.seed,
                project_root=config.project_root,
            )
        )
    selected = pd.concat(selected_examples, ignore_index=True)
    if arguments.subjects:
        wanted = {str(subject).zfill(3) for subject in arguments.subjects}
        recordings = recordings[
            recordings["subject"].astype(str).str.zfill(3).isin(wanted)
        ].reset_index(drop=True)
        selected = selected[
            selected["subject"].astype(str).str.zfill(3).isin(wanted)
        ]
        if recordings.empty:
            raise ValueError(f"No recordings match subjects {sorted(wanted)}.")
        print(f"Restricted to {len(wanted)} subject(s), "
              f"{len(recordings)} recording(s).")
    coverage_by_path = required_minute_coverage(
        selected,
        recordings,
        config,
        complete_subjects=set(arguments.complete_subjects),
    )
    if arguments.complete_subjects:
        print(
            f"Complete coverage requested for "
            f"{len(set(arguments.complete_subjects))} subject(s)."
        )
    total_minutes = 0
    verified = 0
    print(f"Recordings to cache: {len(recordings):,}")
    print(f"Splits: {list(arguments.splits)}")
    print(f"Cache directory: {arguments.cache_dir.resolve()}")
    jobs: list[dict[str, object]] = []
    recording_names: list[str] = []
    for _, row in recordings.iterrows():
        signal_path = resolve_stored_path(row["X_path"], config.project_root)
        availability_path = resolve_stored_path(
            row["channel_availability_path"],
            config.project_root,
        )
        output_path = feature_cache_path(signal_path, arguments.cache_dir)
        coverage_path = feature_coverage_path(signal_path, arguments.cache_dir)
        jobs.append(
            {
                "signal_path": signal_path,
                "availability_path": availability_path,
                "output_path": output_path,
                "coverage_path": coverage_path,
                "required_coverage": coverage_by_path[str(signal_path)],
                "minute_batch_size": arguments.minute_batch_size,
                "entropy_bins": arguments.entropy_bins,
                "overwrite": arguments.overwrite,
                "sampling_frequency": config.target_sfreq,
                "channel_names": list(config.canonical_channel_names),
                "scaler": select_scaler(
                    scaler_document,
                    subject=str(row["subject"]).zfill(3),
                    recording_id=str(row["recording_id"]),
                ),
                "scaler_document": scaler_document,
            }
        )
        recording_names.append(signal_path.name)

    with ProcessPoolExecutor(max_workers=arguments.workers) as executor:
        for index, (minutes, skipped) in enumerate(executor.map(run_cache_job, jobs)):
            total_minutes += minutes
            verified += int(skipped)
            if (index + 1) % 10 == 0 or index + 1 == len(recordings):
                print(
                    f"[{index + 1:04d}/{len(recordings):04d}] completed through "
                    f"{recording_names[index]}; cumulative minutes={total_minutes:,}",
                    flush=True,
                )

    metadata = {
        "extractor": "AVA-style minute features v1",
        "feature_names": list(FEATURE_NAMES),
        "sampling_frequency": config.target_sfreq,
        "window_seconds": 60.0,
        "welch_segment_seconds": 2.0,
        "entropy_bins": arguments.entropy_bins,
        "workers": arguments.workers,
        "selection_sampling_epochs": arguments.selection_sampling_epochs,
        "negative_to_positive_ratio": arguments.negative_to_positive_ratio,
        "max_events_per_patient": arguments.max_events_per_patient,
        "seed": arguments.seed,
        "splits": list(arguments.splits),
        "recordings": int(len(recordings)),
        "verified_existing_recordings": int(verified),
        "cached_required_minutes": int(total_minutes),
        "storage_dtype": "float32",
        "missing_channels_zeroed": True,
    }
    arguments.cache_dir.mkdir(parents=True, exist_ok=True)
    stale_marker = arguments.cache_dir / "STALE.txt"
    if stale_marker.exists():
        stale_marker.unlink()
        print(f"Cleared staleness marker: {stale_marker}")
    metadata_path = arguments.cache_dir / "cache_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    print("Handcrafted feature cache is ready.")


if __name__ == "__main__":
    main()
