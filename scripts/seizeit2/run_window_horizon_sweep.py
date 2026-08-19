"""
Sweep training-window and horizon combinations for the SeizeIT2 baseline.

For each (window, horizon) combination, runs build_dataset.py ->
validate_dataset.py -> train_eegnet_baseline.py in sequence, as separate
subprocesses, then collects each combo's best validation average precision
into one comparison table.

Run from the repository root:

    python scripts/seizeit2/run_window_horizon_sweep.py

By default sweeps window in {30, 15, 10} minutes x horizon in {2, 5, 10}
minutes (9 combinations). Each combo's heavy interim/processed data is
deleted immediately after that combo's training succeeds, keeping only its
model checkpoint, metrics.json, and logs; pass --keep-data to retain it.

The first combination's build_dataset.py run filters every raw EDF once
and caches the result under data/seizeit2/interim/_shared/filtered_recordings
(shared across combos, since filtering never depends on window/horizon).
Every later combination reuses that cache instead of re-filtering, so only
the first combo pays the full raw-EDF-processing cost. That shared cache
is deleted at the end of the sweep unless --keep-data is passed.

--------------------------------------------------------------------------
Preprocessing on one machine, training on another (e.g. Google Colab)
--------------------------------------------------------------------------

CPU-bound preprocessing (filtering, labeling, standardization) and GPU-bound
training don't need to happen on the same machine. Pass --preprocess-only
to run just build_dataset.py -> validate_dataset.py for every combination
-- skipping train_eegnet_baseline.py -- and keep every combo's processed
data and manifests on disk instead of deleting them.

The standardized continuous EEG for a recording depends only on the fitted
global scaler, never on window/horizon -- and the scaler itself is fit from
the (also window/horizon-independent) filtered-recording cache, so every
combination fits the identical scaler. It is therefore stored exactly once,
ever, under data/seizeit2/interim/_shared/standardized_recordings/ and
reused by every combination in the sweep, instead of each combination
duplicating that far larger array. Only the small window/horizon-specific
decision labels/metadata are kept per combination. Without this sharing, a
9-combination sweep over SeizeIT2 would need roughly 9x the ~120GB one
combination's standardized EEG takes up; with it, the whole sweep costs
about the same ~120GB total, once.

Add --package to zip this data for upload to Google Drive: one small
<tag>.zip per combination (labels, manifests, scaler) plus one
shared_standardized_recordings.zip for the whole sweep, all under
outputs/colab_packages/<sweep-name>/.

Every path recorded in a combo's processed_shard_manifest.csv is relative
to the repository root, so these zips can be unzipped directly into
another clone of this repository (e.g. one `git clone`d inside Colab) and
they reproduce the exact data/seizeit2/... layout the manifest expects --
no path editing needed. On the training machine, unzip the shared
recordings archive once, unzip each combination's small archive, then run:

    python scripts/seizeit2/train_eegnet_baseline.py \\
        --window-minutes 30 --horizon-minutes 5 --device cuda

for each combination, then compare each run's outputs/models/.../metrics.json.
"""

from __future__ import annotations

import argparse
import itertools
import json
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIRECTORY = PROJECT_ROOT / "src"

if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))


from seizure_prediction.seizeit2.config import (  # noqa: E402
    PreprocessingConfig,
    build_config,
)


BUILD_SCRIPT = PROJECT_ROOT / "scripts" / "seizeit2" / "build_dataset.py"
VALIDATE_SCRIPT = PROJECT_ROOT / "scripts" / "seizeit2" / "validate_dataset.py"
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "seizeit2" / "train_eegnet_baseline.py"


def parse_arguments() -> argparse.Namespace:
    """Parse the combination grid and sweep-runner options."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--windows",
        type=float,
        nargs="+",
        default=[30.0, 15.0, 10.0],
        help="Training window lengths in minutes.",
    )
    parser.add_argument(
        "--horizons",
        type=float,
        nargs="+",
        default=[2.0, 5.0, 10.0],
        help="Seizure-occurrence prediction horizons in minutes.",
    )
    parser.add_argument(
        "--sweep-name",
        type=str,
        default=None,
        help="Defaults to a timestamp-free name derived from the grid size.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        default=True,
        help="Continue to the next combination if one fails (default: on).",
    )
    parser.add_argument(
        "--stop-on-failure",
        dest="keep_going",
        action="store_false",
        help="Abort the sweep on the first failed combination.",
    )
    parser.add_argument(
        "--keep-data",
        action="store_true",
        help=(
            "Do not delete each combination's tagged interim/processed "
            "directories after training, nor the shared standardized-"
            "recording cache at the end of the sweep. Implied for a "
            "combination's own processed data by --preprocess-only, which "
            "always keeps the shared cache too regardless of this flag."
        ),
    )
    parser.add_argument(
        "--preprocess-only",
        action="store_true",
        help=(
            "Run build_dataset.py and validate_dataset.py for every "
            "combination and stop there -- skip train_eegnet_baseline.py. "
            "Each combination's processed data and manifests are kept on "
            "disk (never deleted, regardless of --keep-data) so they can "
            "be uploaded elsewhere -- e.g. Google Drive -- and trained on "
            "a different machine. See --package to bundle them for upload."
        ),
    )
    parser.add_argument(
        "--package",
        action="store_true",
        help=(
            "After each combination succeeds, zip its small combo-specific "
            "data (labels, manifests, scaler parameters) into "
            "outputs/colab_packages/<sweep-name>/<tag>.zip, and zip the "
            "shared standardized-recording cache into "
            "shared_standardized_recordings.zip -- once for the whole "
            "sweep, not once per combination. Most useful with "
            "--preprocess-only."
        ),
    )
    parser.add_argument(
        "--package-dir",
        type=Path,
        default=None,
        help=(
            "Where --package writes archives. Defaults to "
            "outputs/colab_packages/<sweep-name>/."
        ),
    )
    parser.add_argument(
        "--extra-train-args",
        nargs=argparse.REMAINDER,
        default=[],
        help=(
            "Extra arguments forwarded verbatim to every combination's "
            "train_eegnet_baseline.py invocation (e.g. --epochs 10). "
            "Ignored with --preprocess-only."
        ),
    )
    return parser.parse_args()


def run_stage(
    command: list[str],
    log_path: Path,
) -> bool:
    """Run one subprocess stage, streaming and logging its output.

    Returns True on success, False on a non-zero exit.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"    $ {' '.join(command)}")

    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(f"    | {line}", end="")
            log_file.write(line)
        process.wait()

    return process.returncode == 0


def directory_size_bytes(directory: Path) -> int:
    """Return the total size of every file under `directory`, or 0 if absent."""
    if not directory.exists():
        return 0
    return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())


def zip_directory(directory: Path, project_root: Path, archive_path: Path) -> None:
    """Zip every file under `directory`, keeping paths relative to `project_root`."""
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for file_path in sorted(directory.rglob("*")):
            if file_path.is_file():
                archive.write(file_path, file_path.relative_to(project_root))


def package_combo_manifest(config: PreprocessingConfig, package_dir: Path) -> Path:
    """Zip one combination's small, combo-specific data for upload.

    Bundles the (now tiny) per-combo processed directory -- labels,
    decision metadata, channel layout/availability -- the manifests
    directory (including processed_shard_manifest.csv, whose paths are
    relative to the project root), and the fitted scaler. Deliberately
    excludes the shared standardized recordings, which
    `package_shared_standardized_recordings` bundles once for the whole
    sweep instead of once per combination.
    """
    package_dir.mkdir(parents=True, exist_ok=True)
    archive_path = package_dir / f"{config.experiment_tag}.zip"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for directory in (
            config.processed_data_dir,
            config.manifests_dir,
            config.scaler_parameters_dir,
        ):
            for file_path in sorted(directory.rglob("*")):
                if file_path.is_file():
                    archive.write(
                        file_path,
                        file_path.relative_to(config.project_root),
                    )
    return archive_path


def package_shared_standardized_recordings(
    config: PreprocessingConfig,
    package_dir: Path,
    already_packaged: list[bool],
) -> Path | None:
    """Zip the shared standardized-recording cache, once for the whole sweep.

    Every combination in a window/horizon sweep fits the identical scaler
    (see `fit_global_channel_scaler`) and therefore shares this exact cache,
    so it only needs to be packaged once across the whole sweep -- not once
    per combination -- to avoid re-uploading the same 100+GB of
    standardized EEG for every combination. Returns None without doing
    anything once already packaged earlier in the sweep (tracked via the
    single-element `already_packaged` flag, mutated in place).
    """
    if already_packaged[0]:
        return None
    already_packaged[0] = True
    archive_path = package_dir / "shared_standardized_recordings.zip"
    zip_directory(
        config.standardized_recordings_dir,
        config.project_root,
        archive_path,
    )
    return archive_path


def split_decision_counts(config: PreprocessingConfig) -> dict[str, object]:
    """Summarize decision counts per split from the processed shard manifest."""
    manifest_path = config.manifests_dir / "processed_shard_manifest.csv"
    manifest = pd.read_csv(manifest_path)
    totals = manifest.groupby("split")[
        [
            "number_of_decisions",
            "number_of_positive_decisions",
            "number_of_negative_decisions",
        ]
    ].sum()

    counts: dict[str, object] = {}
    for split_name in ("train", "validation", "test"):
        row = totals.loc[split_name] if split_name in totals.index else None
        counts[f"{split_name}_decisions"] = int(row["number_of_decisions"]) if row is not None else 0
        counts[f"{split_name}_positive_decisions"] = (
            int(row["number_of_positive_decisions"]) if row is not None else 0
        )
    return counts


def run_combination(
    window_minutes: float,
    horizon_minutes: float,
    sweep_dir: Path,
    extra_train_args: list[str],
    keep_data: bool,
    preprocess_only: bool,
    package_dir: Path | None,
    shared_data_packaged: list[bool],
) -> dict[str, object]:
    """Run build -> validate [-> train] for one combination and return its result."""
    tag = f"w{int(round(window_minutes))}_h{int(round(horizon_minutes))}"
    log_dir = sweep_dir / "logs"
    start_time = time.monotonic()

    print(f"\n=== {tag}: window={window_minutes:g}min, horizon={horizon_minutes:g}min ===")

    sweep_flags = [
        "--window-minutes",
        f"{window_minutes:g}",
        "--horizon-minutes",
        f"{horizon_minutes:g}",
    ]

    config = build_config(window_minutes, horizon_minutes)

    build_ok = run_stage(
        [sys.executable, "-u", str(BUILD_SCRIPT), *sweep_flags],
        log_dir / f"{tag}_build.log",
    )
    if not build_ok:
        return _failed_result(
            window_minutes, horizon_minutes, tag, "build_dataset failed",
            config, keep_data,
        )

    # Decision checkpoints are now tiny (per-recording CSV/JSON only -- the
    # continuous EEG signal is never duplicated per combination; Pass 4
    # reads it straight from the shared filtered-recording cache instead,
    # see write_standardized_shards). Still tidied up immediately after a
    # successful build so a stale checkpoint from an old run is never
    # mistaken for this run's, regardless of --keep-data/--preprocess-only.
    shutil.rmtree(config.decision_checkpoints_dir, ignore_errors=True)

    validate_ok = run_stage(
        [sys.executable, "-u", str(VALIDATE_SCRIPT), *sweep_flags],
        log_dir / f"{tag}_validate.log",
    )
    if not validate_ok:
        return _failed_result(
            window_minutes, horizon_minutes, tag, "validate_dataset failed",
            config, keep_data,
        )

    if preprocess_only:
        result: dict[str, object] = {
            "window_minutes": window_minutes,
            "horizon_minutes": horizon_minutes,
            "experiment_tag": tag,
            "status": "ok",
            "wall_clock_seconds": round(time.monotonic() - start_time, 1),
            # This combination's own processed directory now holds only
            # labels/metadata, not the (shared) standardized EEG -- see
            # shared_cache_size_mb for that.
            "combo_data_size_mb": round(
                directory_size_bytes(config.processed_data_dir) / (1024 * 1024), 1
            ),
            "shared_cache_size_mb": round(
                directory_size_bytes(config.standardized_recordings_dir)
                / (1024 * 1024),
                1,
            ),
            **split_decision_counts(config),
        }
        if package_dir is not None:
            combo_archive_path = package_combo_manifest(config, package_dir)
            result["combo_package_path"] = str(combo_archive_path)
            print(f"    Packaged combo data: {combo_archive_path}")

            shared_archive_path = package_shared_standardized_recordings(
                config, package_dir, shared_data_packaged
            )
            if shared_archive_path is not None:
                result["shared_package_path"] = str(shared_archive_path)
                print(
                    f"    Packaged shared standardized recordings: "
                    f"{shared_archive_path}"
                )
            else:
                print(
                    "    Shared standardized recordings already packaged "
                    "earlier in this sweep; skipped."
                )
        # The whole point of --preprocess-only is to keep this combination's
        # processed data and manifests on disk for upload, so they are never
        # deleted here regardless of --keep-data.
        return result

    output_dir = sweep_dir / "models" / tag
    train_ok = run_stage(
        [
            sys.executable,
            "-u",
            str(TRAIN_SCRIPT),
            *sweep_flags,
            "--output-dir",
            str(output_dir),
            *extra_train_args,
        ],
        log_dir / f"{tag}_train.log",
    )
    if not train_ok:
        return _failed_result(
            window_minutes, horizon_minutes, tag, "train_eegnet_baseline failed",
            config, keep_data,
        )

    metrics_path = output_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    if package_dir is not None:
        combo_archive_path = package_combo_manifest(config, package_dir)
        print(f"    Packaged combo data: {combo_archive_path}")
        shared_archive_path = package_shared_standardized_recordings(
            config, package_dir, shared_data_packaged
        )
        if shared_archive_path is not None:
            print(
                f"    Packaged shared standardized recordings: "
                f"{shared_archive_path}"
            )

    if not keep_data:
        shutil.rmtree(config.interim_data_dir, ignore_errors=True)
        shutil.rmtree(config.processed_data_dir, ignore_errors=True)

    return {
        "window_minutes": window_minutes,
        "horizon_minutes": horizon_minutes,
        "experiment_tag": tag,
        "best_validation_average_precision": metrics["best_validation_average_precision"],
        "best_epoch": metrics["best_epoch"],
        "status": "ok",
        "wall_clock_seconds": round(time.monotonic() - start_time, 1),
    }


def _failed_result(
    window_minutes: float,
    horizon_minutes: float,
    tag: str,
    reason: str,
    config: PreprocessingConfig,
    keep_data: bool,
) -> dict[str, object]:
    """Build a failure row and reclaim that combo's disk so the next one isn't starved."""
    print(f"    FAILED: {reason}")
    if not keep_data:
        shutil.rmtree(config.interim_data_dir, ignore_errors=True)
        shutil.rmtree(config.processed_data_dir, ignore_errors=True)
    return {
        "window_minutes": window_minutes,
        "horizon_minutes": horizon_minutes,
        "experiment_tag": tag,
        "best_validation_average_precision": None,
        "best_epoch": None,
        "status": f"failed: {reason}",
        "wall_clock_seconds": None,
    }


def write_results(results: list[dict[str, object]], sweep_dir: Path) -> Path:
    """Write the sorted comparison table and return its path."""
    results_path = sweep_dir / "sweep_results.csv"
    frame = pd.DataFrame(results)
    if "best_validation_average_precision" in frame.columns:
        frame = frame.sort_values(
            "best_validation_average_precision",
            ascending=False,
            na_position="last",
        )
    else:
        frame = frame.sort_values(["window_minutes", "horizon_minutes"])
    frame.to_csv(results_path, index=False)
    return results_path


def main() -> None:
    """Run every (window, horizon) combination and aggregate the results."""
    arguments = parse_arguments()
    combinations = list(itertools.product(arguments.windows, arguments.horizons))

    sweep_name = arguments.sweep_name or (
        f"windows_{'-'.join(f'{w:g}' for w in arguments.windows)}"
        f"_horizons_{'-'.join(f'{h:g}' for h in arguments.horizons)}"
    )
    sweep_dir = PROJECT_ROOT / "outputs" / "sweeps" / sweep_name
    sweep_dir.mkdir(parents=True, exist_ok=True)

    package_dir: Path | None = None
    if arguments.package:
        package_dir = arguments.package_dir or (
            PROJECT_ROOT / "outputs" / "colab_packages" / sweep_name
        )

    (sweep_dir / "sweep_manifest.json").write_text(
        json.dumps(
            {
                "windows": arguments.windows,
                "horizons": arguments.horizons,
                "extra_train_args": arguments.extra_train_args,
                "keep_data": arguments.keep_data,
                "preprocess_only": arguments.preprocess_only,
                "package_dir": str(package_dir) if package_dir else None,
                "combinations": [
                    {"window_minutes": w, "horizon_minutes": h}
                    for w, h in combinations
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Sweeping {len(combinations)} combinations into {sweep_dir}")
    if arguments.preprocess_only:
        print("Mode: --preprocess-only (build + validate only, no training).")
    if package_dir is not None:
        print(f"Packaging each combination's portable data into {package_dir}")

    results: list[dict[str, object]] = []
    shared_data_packaged = [False]
    for window_minutes, horizon_minutes in combinations:
        try:
            result = run_combination(
                window_minutes,
                horizon_minutes,
                sweep_dir,
                arguments.extra_train_args,
                arguments.keep_data,
                arguments.preprocess_only,
                package_dir,
                shared_data_packaged,
            )
        except Exception as error:  # noqa: BLE001 - record and keep going
            result = _failed_result(
                window_minutes,
                horizon_minutes,
                f"w{int(round(window_minutes))}_h{int(round(horizon_minutes))}",
                f"unexpected error: {error}",
                build_config(window_minutes, horizon_minutes),
                arguments.keep_data,
            )
        results.append(result)

        if result["status"] != "ok" and not arguments.keep_going:
            print("\nStopping sweep after failed combination (--stop-on-failure).")
            break

    results_path = write_results(results, sweep_dir)

    if not arguments.keep_data:
        shutil.rmtree(build_config().filtered_recordings_dir, ignore_errors=True)
        # --preprocess-only exists specifically to keep this data on disk for
        # upload, so it is never swept up here regardless of --keep-data.
        if not arguments.preprocess_only:
            shutil.rmtree(
                build_config().standardized_recordings_dir,
                ignore_errors=True,
            )

    print(f"\nSweep results: {results_path}")

    if arguments.preprocess_only:
        total_shared_mb = round(
            directory_size_bytes(build_config().standardized_recordings_dir)
            / (1024 * 1024),
            1,
        )
        print(
            f"Every combination fits the identical scaler, so the "
            f"standardized EEG is stored once for all {len(results)} "
            f"combination(s), not once each. Shared cache total: "
            f"{total_shared_mb:.1f} MB "
            f"({build_config().standardized_recordings_dir})."
        )
        for result in sorted(
            results, key=lambda row: (row["window_minutes"], row["horizon_minutes"])
        ):
            combo_size_text = (
                f"{result['combo_data_size_mb']:.1f} MB"
                if result.get("combo_data_size_mb") is not None
                else "N/A"
            )
            print(
                f"  {result['experiment_tag']:>10}  combo_data={combo_size_text}  "
                f"status={result['status']}"
            )
    else:
        for result in sorted(
            results,
            key=lambda row: (
                row["best_validation_average_precision"] is None,
                -(row["best_validation_average_precision"] or 0.0),
            ),
        ):
            ap = result["best_validation_average_precision"]
            ap_text = f"{ap:.6f}" if ap is not None else "N/A"
            print(f"  {result['experiment_tag']:>10}  AP={ap_text}  status={result['status']}")


if __name__ == "__main__":
    main()
