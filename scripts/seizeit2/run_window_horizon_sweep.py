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
"""

from __future__ import annotations

import argparse
import itertools
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIRECTORY = PROJECT_ROOT / "src"

if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))


from seizure_prediction.seizeit2.config import build_config  # noqa: E402


BUILD_SCRIPT = PROJECT_ROOT / "scripts" / "seizeit2" / "build_dataset.py"
VALIDATE_SCRIPT = PROJECT_ROOT / "scripts" / "seizeit2" / "validate_dataset.py"
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "seizeit2" / "train_eegnet_baseline.py"


def parse_arguments() -> argparse.Namespace:
    """Parse the combination grid and sweep-runner options."""
    parser = argparse.ArgumentParser(description=__doc__)
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
            "directories after training. Requires enough free disk for "
            "every combination's processed data simultaneously."
        ),
    )
    parser.add_argument(
        "--extra-train-args",
        nargs=argparse.REMAINDER,
        default=[],
        help=(
            "Extra arguments forwarded verbatim to every combination's "
            "train_eegnet_baseline.py invocation (e.g. --epochs 10)."
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


def run_combination(
    window_minutes: float,
    horizon_minutes: float,
    sweep_dir: Path,
    extra_train_args: list[str],
    keep_data: bool,
) -> dict[str, object]:
    """Run build -> validate -> train for one combination and return its result."""
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

    build_ok = run_stage(
        [sys.executable, str(BUILD_SCRIPT), *sweep_flags],
        log_dir / f"{tag}_build.log",
    )
    if not build_ok:
        return _failed_result(window_minutes, horizon_minutes, tag, "build_dataset failed")

    validate_ok = run_stage(
        [sys.executable, str(VALIDATE_SCRIPT), *sweep_flags],
        log_dir / f"{tag}_validate.log",
    )
    if not validate_ok:
        return _failed_result(window_minutes, horizon_minutes, tag, "validate_dataset failed")

    output_dir = sweep_dir / "models" / tag
    train_ok = run_stage(
        [
            sys.executable,
            str(TRAIN_SCRIPT),
            *sweep_flags,
            "--output-dir",
            str(output_dir),
            *extra_train_args,
        ],
        log_dir / f"{tag}_train.log",
    )
    if not train_ok:
        return _failed_result(window_minutes, horizon_minutes, tag, "train_eegnet_baseline failed")

    metrics_path = output_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    if not keep_data:
        config = build_config(window_minutes, horizon_minutes)
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
) -> dict[str, object]:
    """Build a failure row so one bad combination doesn't lose the others."""
    print(f"    FAILED: {reason}")
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
    import pandas as pd

    results_path = sweep_dir / "sweep_results.csv"
    frame = pd.DataFrame(results).sort_values(
        "best_validation_average_precision",
        ascending=False,
        na_position="last",
    )
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

    (sweep_dir / "sweep_manifest.json").write_text(
        json.dumps(
            {
                "windows": arguments.windows,
                "horizons": arguments.horizons,
                "extra_train_args": arguments.extra_train_args,
                "keep_data": arguments.keep_data,
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

    results: list[dict[str, object]] = []
    for window_minutes, horizon_minutes in combinations:
        try:
            result = run_combination(
                window_minutes,
                horizon_minutes,
                sweep_dir,
                arguments.extra_train_args,
                arguments.keep_data,
            )
        except Exception as error:  # noqa: BLE001 - record and keep going
            result = _failed_result(
                window_minutes,
                horizon_minutes,
                f"w{int(round(window_minutes))}_h{int(round(horizon_minutes))}",
                f"unexpected error: {error}",
            )
        results.append(result)

        if result["status"] != "ok" and not arguments.keep_going:
            print("\nStopping sweep after failed combination (--stop-on-failure).")
            break

    results_path = write_results(results, sweep_dir)

    print(f"\nSweep results: {results_path}")
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
