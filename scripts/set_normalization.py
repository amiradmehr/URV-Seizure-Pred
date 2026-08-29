"""Switch the normalization every label definition standardizes with.

This replaces the earlier ``restandardize_processed.py``, which rewrote the
whole standardized dataset in place.  There is nothing left to rewrite: the
pipeline stores one shared copy of the *filtered* EEG in raw filtered units,
and standardization is applied when a decision is loaded.  Changing
normalization is therefore fitting a new scaler document and repointing the
manifests at it -- seconds of work against a few hundred megabytes of JSON and
CSV, rather than a rewrite of every recording.

Run from the repository root:

    python scripts/set_normalization.py --mode patient --statistic meanstd

The new document is fitted from the shared filtered recordings, so it is
expressed in raw filtered units directly and no prior standardization has to
be undone.  Modes can be switched any number of times, in either direction,
with no drift.

Derived caches do bake in the standardization -- an embedding or a handcrafted
feature is computed from standardized EEG -- so any cache built under the old
normalization is stale afterwards.  They are reported, and removed with
``--clear-derived-caches``.
"""

from __future__ import annotations

import argparse
import json
import shutil
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
    filtered_recording_paths,
    relative_to_project_root,
)


SEPARATOR = "=" * 90


def print_header(title: str) -> None:
    """Print a console section header."""
    print()
    print(SEPARATOR)
    print(title.center(90))
    print(SEPARATOR)


def parse_arguments() -> argparse.Namespace:
    """Parse the target normalization and which label definitions to update."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=NORMALIZATION_MODES,
        default="patient",
        help="Target normalization scope.",
    )
    parser.add_argument(
        "--statistic",
        choices=STATISTICS,
        default="meanstd",
        help=(
            "meanstd is the classic z-score. robust uses median/IQR, which is "
            "far less sensitive to the movement and electrode-pop artifacts "
            "common in wearable EEG."
        ),
    )
    parser.add_argument(
        "--windows",
        type=float,
        nargs="+",
        default=None,
        metavar="MINUTES",
        help="Label definitions to repoint (default: every built one).",
    )
    parser.add_argument(
        "--horizons",
        type=float,
        nargs="+",
        default=None,
        metavar="MINUTES",
        help="Label definitions to repoint (default: every built one).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fit and report the scalers without writing anything.",
    )
    parser.add_argument(
        "--clear-derived-caches",
        action="store_true",
        help=(
            "Delete the embedding and handcrafted-feature caches, which were "
            "computed from the previous standardization and are stale once it "
            "changes."
        ),
    )
    return parser.parse_args()


def built_configurations(
    arguments: argparse.Namespace,
) -> list[PreprocessingConfig]:
    """Return the requested label definitions that exist on disk."""
    candidates = sweep_configurations(
        input_window_minutes=arguments.windows,
        seizure_occurrence_period_minutes=arguments.horizons,
    )
    built = [
        config
        for config in candidates
        if (config.manifests_dir / "processed_shard_manifest.csv").exists()
    ]

    if not built:
        raise FileNotFoundError(
            "None of the requested label definitions has been built. Run "
            "scripts/build_dataset.py first."
        )

    return built


def recordings_to_cover(
    configs: list[PreprocessingConfig],
) -> pd.DataFrame:
    """Return every recording referenced by the label definitions being updated.

    A shorter input window keeps recordings a longer one was too short for, so
    the union across label definitions is what the new document has to cover.
    """
    frames = []

    for config in configs:
        manifest = pd.read_csv(
            config.manifests_dir / "processed_shard_manifest.csv",
            dtype={"subject": str, "recording_id": str},
        )
        frames.append(manifest[["subject", "recording_id", "split"]])

    recordings = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset="recording_id"
    )
    recordings["subject"] = recordings["subject"].astype(str).str.zfill(3)
    return recordings.reset_index(drop=True)


def fit_document(
    recordings: pd.DataFrame,
    mode: str,
    statistic: str,
    config: PreprocessingConfig,
) -> dict:
    """Fit the target scaler document from the shared filtered recordings.

    The shared recordings are in raw filtered units, so the fitted parameters
    are expressed against raw units directly. Nothing has to be inverted out
    of a previous standardization, which is what made the earlier in-place
    conversion delicate.
    """
    if mode == "global":
        contributing = recordings[recordings["split"] == "train"]
        if contributing.empty:
            raise ValueError("There are no training recordings to fit a scaler.")
    else:
        contributing = recordings

    availability_by_path: dict[Path, np.ndarray] = {}

    for recording_id in contributing["recording_id"]:
        array_path, _, availability_path, _ = filtered_recording_paths(
            config.unscaled_recordings_dir,
            str(recording_id),
        )
        if not array_path.exists():
            raise FileNotFoundError(
                f"Shared filtered recording not found: {array_path}"
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
                config.unscaled_recordings_dir,
                str(recording_id),
            )[0]
            for recording_id in group["recording_id"]
        ]
        scalers[scaler_key] = fit_channel_scaler(
            array_paths,
            availability_by_path,
            channel_names=list(config.canonical_channel_names),
            statistic=statistic,
            epsilon=config.zscore_epsilon,
        )
        print(
            f"[{group_number:04d}/{len(groups):04d}] {scaler_key}: "
            f"{len(array_paths)} recording(s)"
        )

    if mode != "global":
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
        channel_names=list(config.canonical_channel_names),
        scalers=scalers,
        epsilon=config.zscore_epsilon,
        training_subjects=(
            sorted(set(contributing["subject"])) if mode == "global" else None
        ),
    )
    document["covered_recording_ids"] = sorted(
        str(recording_id) for recording_id in recordings["recording_id"]
    )
    return document


def repoint_manifest(
    config: PreprocessingConfig,
    document: dict,
    scaler_path: Path,
) -> int:
    """Point one label definition's shards at the new scaler document."""
    manifest_path = config.manifests_dir / "processed_shard_manifest.csv"
    manifest = pd.read_csv(
        manifest_path,
        dtype={"subject": str, "recording_id": str},
    )
    mode = document["normalization_mode"]

    manifest["normalization_mode"] = mode
    manifest["scaler_key"] = [
        scaler_key_for(
            mode,
            subject=str(row.subject).zfill(3),
            recording_id=str(row.recording_id),
        )
        for row in manifest.itertuples(index=False)
    ]
    manifest["scaler_document_path"] = relative_to_project_root(
        scaler_path,
        config.project_root,
    )

    uncovered = sorted(set(manifest["scaler_key"]) - set(document["scalers"]))

    if uncovered:
        raise ValueError(
            f"{len(uncovered)} shard(s) in {config.experiment_tag} have no "
            f"{mode} scaler, first: {uncovered[:5]}"
        )

    manifest.to_csv(manifest_path, index=False)
    return len(manifest)


def derived_cache_directories(config: PreprocessingConfig) -> list[Path]:
    """Return the caches that bake in the current standardization."""
    return [
        directory
        for directory in (
            config.embedding_cache_dir,
            config.handcrafted_feature_cache_dir,
        )
        if directory.exists() and any(directory.iterdir())
    ]


def main() -> None:
    """Fit the target normalization and repoint every built label definition."""
    arguments = parse_arguments()
    CONFIG.validate()

    configs = built_configurations(arguments)
    scaler_path = CONFIG.scaler_document_path(arguments.mode, arguments.statistic)

    print_header(
        f"SET NORMALIZATION: {arguments.mode.upper()} "
        f"{arguments.statistic.upper()}"
    )

    print(f"Shared recordings: {CONFIG.unscaled_recordings_dir}")
    print(f"Scaler document:   {scaler_path}")
    print(
        "Label definitions: "
        f"{', '.join(config.experiment_tag for config in configs)}"
    )

    recordings = recordings_to_cover(configs)
    print(f"Recordings to cover: {len(recordings)}")

    print_header("FITTING SCALERS")

    document = fit_document(
        recordings=recordings,
        mode=arguments.mode,
        statistic=arguments.statistic,
        config=CONFIG,
    )

    print(
        f"\nFitted {document['scaler_count']} {arguments.mode} scaler(s) "
        f"({arguments.statistic}) over {document['fitted_on']}."
    )

    if arguments.dry_run:
        print("\nDry run: nothing was written.")
        return

    save_scaler_document(document, scaler_path)
    print(f"Wrote {scaler_path}")

    print_header("REPOINTING LABEL DEFINITIONS")

    for config in configs:
        shard_count = repoint_manifest(config, document, scaler_path)
        print(f"  {config.experiment_tag:>8}: {shard_count} shard(s) repointed")

    stale_caches = derived_cache_directories(CONFIG)

    if stale_caches:
        print_header("DERIVED CACHES")
        print(
            "These caches were computed from the previous standardization and "
            "are now stale:"
        )
        for directory in stale_caches:
            print(f"  {directory}")

        if arguments.clear_derived_caches:
            for directory in stale_caches:
                shutil.rmtree(directory)
                print(f"  removed {directory}")
        else:
            print(
                "\nRebuild them, or rerun with --clear-derived-caches to "
                "delete them now."
            )

    print_header("NORMALIZATION UPDATED")
    print("No EEG was rewritten; only the scaler document and the manifests.")


if __name__ == "__main__":
    main()
