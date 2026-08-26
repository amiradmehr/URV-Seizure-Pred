r"""Build a seizure-detection dataset from the existing processed recordings.

Why detection
-------------
Two questions have been tangled together all along: whether the pipeline is
sound, and whether a preictal signal exists.  Prediction cannot separate them,
because nobody knows what the right answer is -- a null result is equally
consistent with a broken pipeline and with absent physiology.

Detection has a known answer.  SeizeIT2 was collected and published to validate
wearable seizure *detection*, so a pipeline that loads, filters and labels this
data correctly must be able to do it.  That makes detection a positive control:
failure convicts the pipeline, success clears it.

It is also the better pretraining task.  Prediction offers 253 eligible
training seizures; detection offers every annotated seizure in the cohort,
because it needs no clear pre-seizure history, no postictal exclusion and no
ten-minute lookahead.  An encoder trained on the larger, easier task is a far
better starting point than one learned from scratch on 253 events.

What this builds
----------------
Short windows labeled by whether they overlap an annotated seizure, drawn from
the same ``data/processed`` arrays, the same channel layout and the same
patient-level splits as the prediction task.  Nothing is re-filtered and no EEG
is copied: the output is a manifest of window indices, exactly as the
prediction pipeline indexes its decisions.

PowerShell example:

    .\.venv-old\Scripts\python.exe scripts\build_detection_dataset.py
"""

from __future__ import annotations

import argparse
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
from seizure_prediction.datasets import resolve_stored_path  # noqa: E402


SEPARATOR = "=" * 92


def parse_arguments() -> argparse.Namespace:
    """Parse windowing and output options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=10.0,
        help="Detection window length. Short windows are the standard choice.",
    )
    parser.add_argument(
        "--stride-seconds",
        type=float,
        default=10.0,
        help="Step between windows. Equal to the window means no overlap.",
    )
    parser.add_argument(
        "--minimum-overlap-seconds",
        type=float,
        default=5.0,
        help=(
            "Seizure overlap needed for a positive label. Windows that touch a "
            "seizure by less than this are dropped rather than labeled, since "
            "they are neither clearly ictal nor clearly interictal."
        ),
    )
    parser.add_argument(
        "--guard-seconds",
        type=float,
        default=60.0,
        help=(
            "Interval around every seizure excluded from negatives, so the "
            "negative class is unambiguously interictal."
        ),
    )
    parser.add_argument(
        "--negatives-per-positive",
        type=float,
        default=20.0,
        help="Negative windows retained per positive, sampled per recording.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "validation", "test"),
        default=("train", "validation"),
        help="The held-out test split is excluded by default.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Manifest path. Defaults to the interim manifests directory.",
    )
    return parser.parse_args()


def validate_arguments(arguments: argparse.Namespace) -> None:
    """Reject window settings that cannot produce a clean dataset."""
    if arguments.window_seconds <= 0 or arguments.stride_seconds <= 0:
        raise ValueError("window-seconds and stride-seconds must be positive.")
    if not 0 < arguments.minimum_overlap_seconds <= arguments.window_seconds:
        raise ValueError(
            "minimum-overlap-seconds must be positive and at most the window."
        )
    if arguments.guard_seconds < 0:
        raise ValueError("guard-seconds cannot be negative.")
    if arguments.negatives_per_positive <= 0:
        raise ValueError("negatives-per-positive must be positive.")


def load_seizures() -> pd.DataFrame:
    """Load every annotated seizure, not only the prediction-eligible ones."""
    path = CONFIG.manifests_dir / "seizure_manifest.csv"
    if not path.exists():
        raise FileNotFoundError(f"Seizure manifest not found: {path}")
    seizures = pd.read_csv(path, dtype={"subject": str})
    seizures["recording_id"] = (
        seizures["seizure_id"].astype(str).str.rsplit("_seizure-", n=1).str[0]
    )
    seizures["onset_seconds"] = pd.to_numeric(
        seizures["onset_seconds"], errors="raise"
    )
    seizures["duration_seconds"] = pd.to_numeric(
        seizures["duration_seconds"], errors="raise"
    )
    return seizures


def overlap_seconds(
    starts: np.ndarray,
    stops: np.ndarray,
    interval_start: float,
    interval_stop: float,
) -> np.ndarray:
    """Return the overlap of each window with one interval."""
    return np.clip(
        np.minimum(stops, interval_stop) - np.maximum(starts, interval_start),
        0.0,
        None,
    )


def windows_for_recording(
    *,
    recording_id: str,
    subject: str,
    split: str,
    signal_path: Path,
    availability_path: Path,
    sample_count: int,
    seizures: pd.DataFrame,
    arguments: argparse.Namespace,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Label detection windows for one processed recording."""
    window_samples = int(round(arguments.window_seconds * CONFIG.target_sfreq))
    stride_samples = int(round(arguments.stride_seconds * CONFIG.target_sfreq))
    if sample_count < window_samples:
        return pd.DataFrame()

    start_samples = np.arange(
        0,
        sample_count - window_samples + 1,
        stride_samples,
        dtype=np.int64,
    )
    starts = start_samples / CONFIG.target_sfreq
    stops = starts + arguments.window_seconds

    ictal_overlap = np.zeros(len(starts), dtype=np.float64)
    near_seizure = np.zeros(len(starts), dtype=bool)
    for seizure in seizures.itertuples(index=False):
        onset = float(seizure.onset_seconds)
        offset = onset + float(seizure.duration_seconds)
        ictal_overlap = np.maximum(
            ictal_overlap,
            overlap_seconds(starts, stops, onset, offset),
        )
        near_seizure |= (
            overlap_seconds(
                starts,
                stops,
                onset - arguments.guard_seconds,
                offset + arguments.guard_seconds,
            )
            > 0.0
        )

    positive = ictal_overlap >= arguments.minimum_overlap_seconds
    # A window that clips a seizure edge is neither clearly ictal nor clearly
    # interictal, so it is discarded instead of being forced into a class.
    ambiguous = (ictal_overlap > 0.0) & ~positive
    negative = ~positive & ~ambiguous & ~near_seizure

    positive_indices = np.flatnonzero(positive)
    negative_indices = np.flatnonzero(negative)
    if len(positive_indices) == 0:
        # Recordings without a seizure still supply interictal windows, but
        # capping them keeps one long recording from swamping the split.
        wanted = int(round(arguments.negatives_per_positive))
    else:
        wanted = int(round(len(positive_indices) * arguments.negatives_per_positive))
    if len(negative_indices) > wanted:
        negative_indices = rng.choice(negative_indices, size=wanted, replace=False)

    selected = np.concatenate([positive_indices, negative_indices])
    if len(selected) == 0:
        return pd.DataFrame()
    selected.sort()

    return pd.DataFrame(
        {
            "recording_id": recording_id,
            "subject": subject,
            "split": split,
            "window_start_sample": start_samples[selected],
            "window_stop_sample": start_samples[selected] + window_samples,
            "window_start_seconds": starts[selected],
            "label": positive[selected].astype(np.int64),
            "ictal_overlap_seconds": ictal_overlap[selected],
            "X_path": str(signal_path),
            "channel_availability_path": str(availability_path),
        }
    )


def main() -> None:
    """Write the detection window manifest."""
    arguments = parse_arguments()
    validate_arguments(arguments)
    CONFIG.validate()

    print(SEPARATOR)
    print("SEIZURE DETECTION DATASET".center(92))
    print(SEPARATOR)

    manifest_path = CONFIG.manifests_dir / "processed_shard_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Processed manifest not found: {manifest_path}")
    shards = pd.read_csv(manifest_path, dtype={"subject": str})
    shards = shards[shards["split"].isin(arguments.splits)].reset_index(drop=True)
    if shards.empty:
        raise ValueError(f"No processed shards for splits {arguments.splits}.")

    seizures = load_seizures()
    rng = np.random.default_rng(arguments.seed)
    frames: list[pd.DataFrame] = []

    print(
        f"\nWindow {arguments.window_seconds:g}s, stride "
        f"{arguments.stride_seconds:g}s, guard {arguments.guard_seconds:g}s, "
        f"{arguments.negatives_per_positive:g} negatives per positive"
    )
    print(f"Recordings: {len(shards)}  splits: {list(arguments.splits)}\n")

    for number, shard in enumerate(shards.itertuples(index=False), start=1):
        signal_path = resolve_stored_path(shard.X_path)
        signal = np.load(signal_path, mmap_mode="r")
        recording_seizures = seizures[
            seizures["recording_id"] == str(shard.recording_id)
        ]
        frame = windows_for_recording(
            recording_id=str(shard.recording_id),
            subject=str(shard.subject).zfill(3),
            split=str(shard.split),
            signal_path=signal_path,
            availability_path=resolve_stored_path(shard.channel_availability_path),
            sample_count=int(signal.shape[1]),
            seizures=recording_seizures,
            arguments=arguments,
            rng=rng,
        )
        del signal
        if not frame.empty:
            frames.append(frame)
        if number % 200 == 0:
            print(f"  processed {number}/{len(shards)} recordings")

    if not frames:
        raise RuntimeError("No detection windows were produced.")
    windows = pd.concat(frames, ignore_index=True)

    output_path = arguments.output or (
        CONFIG.manifests_dir / "detection_window_manifest.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    windows.to_csv(output_path, index=False)

    summary = {
        "window_seconds": arguments.window_seconds,
        "stride_seconds": arguments.stride_seconds,
        "minimum_overlap_seconds": arguments.minimum_overlap_seconds,
        "guard_seconds": arguments.guard_seconds,
        "negatives_per_positive": arguments.negatives_per_positive,
        "splits": list(arguments.splits),
        "held_out_test_used": "test" in arguments.splits,
        "windows": int(len(windows)),
        "positive_windows": int((windows["label"] == 1).sum()),
        "negative_windows": int((windows["label"] == 0).sum()),
        "recordings": int(windows["recording_id"].nunique()),
        "subjects": int(windows["subject"].nunique()),
        "seizures_represented": int(
            seizures[
                seizures["recording_id"].isin(windows["recording_id"].unique())
            ].shape[0]
        ),
        "seed": arguments.seed,
    }
    summary_path = output_path.with_name(
        output_path.stem + "_summary.json"
    )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + SEPARATOR)
    print("DETECTION DATASET COMPLETE".center(92))
    print(SEPARATOR)
    per_split = windows.groupby(["split", "label"]).size().unstack(fill_value=0)
    print("\n" + per_split.to_string())
    print(
        f"\nwindows {summary['windows']:,}  "
        f"positives {summary['positive_windows']:,}  "
        f"subjects {summary['subjects']}"
    )
    print(f"\nManifest: {output_path}")
    print(f"Summary : {summary_path}")


if __name__ == "__main__":
    main()
