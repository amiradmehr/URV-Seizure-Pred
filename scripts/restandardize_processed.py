"""Re-standardize data/processed in place under a different normalization.

Every standardization this pipeline uses is a per-channel affine map, so the
processed dataset can be moved between normalization modes exactly, without
re-reading the raw EDFs.  This matters because ``data/interim`` is disposable
and a full rebuild from BIDS takes hours.

How it stays exact
------------------
``data/interim/scaler_parameters/channel_scalers.json`` records, for every
recording, the ``center``/``scale`` that maps *raw filtered* units to what is
currently stored on disk::

    stored = (raw - center) / scale

Each recording is returned to raw units as ``raw = stored * scale + center``
while its statistics are accumulated, so the new scaler is expressed in raw
units directly.  Doing it per recording rather than per group matters: when
moving from a finer scope to a coarser one -- per-patient back to global -- the
recordings being pooled do *not* share a stored scaler, and assuming they did
would corrupt the result.

The rewrite is then a single affine pass over the stored array.  Because the
chain is always expressed against raw units, modes can be switched any number of
times, in either direction, with no drift and no one-way door.

PowerShell example:

    .\.venv-old\Scripts\python.exe scripts\restandardize_processed.py --mode patient
"""

from __future__ import annotations

import argparse
import json
import os
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
from seizure_prediction.normalization import (  # noqa: E402
    BLOCK_SAMPLES,
    NORMALIZATION_MODES,
    STATISTICS,
    build_scaler_document,
    fit_channel_scaler,
    load_scaler_document,
    save_scaler_document,
    scaler_key_for,
    select_scaler,
)


SEPARATOR = "=" * 90
SCALER_DOCUMENT_NAME = "channel_scalers.json"
STATE_NAME = "normalization_state.json"
DERIVED_CACHE_DIRECTORIES = (
    PROJECT_ROOT / "data" / "embedding_cache",
    PROJECT_ROOT / "data" / "handcrafted_feature_cache",
)


def parse_arguments() -> argparse.Namespace:
    """Parse conversion options."""
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
        "--dry-run",
        action="store_true",
        help="Fit and report the scalers without rewriting any array.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Convert even when the dataset already uses the target mode.",
    )
    parser.add_argument(
        "--limit-groups",
        type=int,
        default=None,
        help="Convert only the first N scaler groups. For smoke tests.",
    )
    return parser.parse_args()


def print_header(title: str) -> None:
    """Print a console section header."""
    print()
    print(SEPARATOR)
    print(title.center(90))
    print(SEPARATOR)


def legacy_scaler_path() -> Path:
    """Return the original global scaler written by build_dataset.py."""
    return CONFIG.scaler_parameters_dir / "global_channel_zscore.json"


def scaler_document_path() -> Path:
    """Return the authoritative scaler document for data/processed."""
    return CONFIG.scaler_parameters_dir / SCALER_DOCUMENT_NAME


def state_path() -> Path:
    """Return the conversion-state marker stored beside the processed data."""
    return CONFIG.processed_data_dir / STATE_NAME


def load_current_document() -> dict:
    """Return the scaler document describing data/processed as it exists now."""
    document_path = scaler_document_path()
    if document_path.exists():
        return load_scaler_document(document_path)

    legacy_path = legacy_scaler_path()
    if not legacy_path.exists():
        raise FileNotFoundError(
            f"Neither {document_path} nor {legacy_path} exists. Run "
            "scripts/build_dataset.py before re-standardizing."
        )
    print(f"Reading legacy global scaler: {legacy_path}")
    return load_scaler_document(legacy_path)


def load_manifest() -> pd.DataFrame:
    """Return the processed shard manifest with resolved local paths."""
    manifest_path = CONFIG.manifests_dir / "processed_shard_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Processed manifest not found: {manifest_path}")

    manifest = pd.read_csv(manifest_path, dtype={"subject": str})
    manifest["subject"] = manifest["subject"].astype(str).str.zfill(3)
    manifest["recording_id"] = manifest["recording_id"].astype(str)
    manifest["array_path"] = manifest["X_path"].map(
        lambda value: resolve_stored_path(value)
    )
    manifest["availability_path"] = manifest["channel_availability_path"].map(
        lambda value: resolve_stored_path(value)
    )
    missing = [
        str(path) for path in manifest["array_path"] if not Path(path).exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} processed arrays are missing, first: {missing[0]}"
        )
    return manifest


def load_availability(path: Path) -> np.ndarray:
    """Load one recording's canonical channel-availability mask."""
    with Path(path).open("r", encoding="utf-8") as mask_file:
        return np.asarray(json.load(mask_file), dtype=bool)


def current_raw_transform(
    document: dict,
    *,
    subject: str,
    recording_id: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the ``(center, scale)`` recovering raw units for one recording.

    The stored array satisfies ``stored = (raw - center) / scale``, so raw
    units come back as ``raw = stored * scale + center``.
    """
    scaler = select_scaler(
        document,
        subject=subject,
        recording_id=recording_id,
    )
    center = np.asarray(
        scaler.get("center", scaler.get("mean")),
        dtype=np.float64,
    )
    scale = np.asarray(
        scaler.get("scale", scaler.get("std")),
        dtype=np.float64,
    )
    return center, scale


def rewrite_array(
    array_path: Path,
    availability: np.ndarray,
    multiplier: np.ndarray,
    offset: np.ndarray,
    output_dtype: str,
) -> None:
    """Apply ``stored * multiplier + offset`` to one array, crash-safely."""
    source = np.load(array_path, mmap_mode="r")
    temporary_path = array_path.with_suffix(".npy.tmp")
    converted = np.lib.format.open_memmap(
        temporary_path,
        mode="w+",
        dtype=np.dtype(output_dtype),
        shape=source.shape,
    )
    multiplier_column = multiplier.astype(np.float32)[:, None]
    offset_column = offset.astype(np.float32)[:, None]
    unavailable = ~availability

    for start in range(0, source.shape[1], BLOCK_SAMPLES):
        block = np.asarray(
            source[:, start : start + BLOCK_SAMPLES],
            dtype=np.float32,
        )
        block = block * multiplier_column + offset_column
        block[unavailable, :] = 0.0
        converted[:, start : start + BLOCK_SAMPLES] = block

    converted.flush()
    del converted
    del source
    os.replace(temporary_path, array_path)


def mark_derived_caches_stale(mode: str, statistic: str) -> list[Path]:
    """Flag caches derived from data/processed without deleting anything."""
    marked: list[Path] = []
    for cache_root in DERIVED_CACHE_DIRECTORIES:
        if not cache_root.exists():
            continue
        for cache_directory in sorted(
            path for path in cache_root.iterdir() if path.is_dir()
        ):
            marker = cache_directory / "STALE.txt"
            marker.write_text(
                "This cache was derived from data/processed BEFORE it was "
                f"re-standardized to mode={mode}, statistic={statistic}.\n"
                "Its contents no longer correspond to the processed data and "
                "must be regenerated before use:\n"
                "  scripts/cache_eegnet_embeddings.py\n"
                "  scripts/cache_handcrafted_features.py\n",
                encoding="utf-8",
            )
            marked.append(cache_directory)
    return marked


def main() -> None:
    """Fit the target scalers and rewrite the processed arrays in place."""
    arguments = parse_arguments()
    CONFIG.validate()

    print_header("RE-STANDARDIZE PROCESSED DATASET")

    current_document = load_current_document()
    current_mode = current_document["normalization_mode"]
    current_statistic = current_document.get("statistic", "meanstd")
    print(f"Current normalization: mode={current_mode}, statistic={current_statistic}")
    print(f"Target normalization:  mode={arguments.mode}, statistic={arguments.statistic}")

    if (
        current_mode == arguments.mode
        and current_statistic == arguments.statistic
        and not arguments.force
    ):
        print(
            "\nThe processed dataset already uses the target normalization. "
            "Nothing to do. Pass --force to refit and rewrite anyway."
        )
        return

    manifest = load_manifest()
    channel_names = list(CONFIG.canonical_channel_names)
    print(
        f"\nRecordings: {len(manifest)}  "
        f"patients: {manifest['subject'].nunique()}"
    )

    manifest["scaler_key"] = [
        scaler_key_for(
            arguments.mode,
            subject=row.subject,
            recording_id=row.recording_id,
        )
        for row in manifest.itertuples(index=False)
    ]
    groups = list(manifest.groupby("scaler_key", sort=True))
    if arguments.limit_groups is not None:
        groups = groups[: arguments.limit_groups]
        print(f"Limiting to the first {len(groups)} scaler group(s).")

    print_header("PASS 1: FIT TARGET SCALERS")

    scalers: dict[str, dict] = {}
    raw_parameters: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for group_number, (scaler_key, group) in enumerate(groups, start=1):
        array_paths = [Path(path) for path in group["array_path"]]
        availability_by_path = {
            Path(row.array_path): load_availability(row.availability_path)
            for row in group.itertuples(index=False)
        }
        # Recordings in one group need not currently share a scaler -- going
        # back from per-patient to global is exactly that case -- so each is
        # returned to raw units before its statistics are accumulated.
        raw_transform_by_path = {
            Path(row.array_path): current_raw_transform(
                current_document,
                subject=row.subject,
                recording_id=row.recording_id,
            )
            for row in group.itertuples(index=False)
        }
        fitted = fit_channel_scaler(
            array_paths,
            availability_by_path,
            channel_names=channel_names,
            statistic=arguments.statistic,
            epsilon=CONFIG.zscore_epsilon,
            raw_transform_by_path=raw_transform_by_path,
        )

        raw_center = np.asarray(fitted["center"], dtype=np.float64)
        raw_scale = np.asarray(fitted["scale"], dtype=np.float64)
        raw_parameters[scaler_key] = (raw_center, raw_scale)
        scalers[scaler_key] = fitted
        print(
            f"[{group_number:04d}/{len(groups):04d}] {scaler_key}: "
            f"{len(array_paths)} recording(s), "
            f"scale={np.array2string(raw_scale, precision=3, suppress_small=False)}"
        )

    document = build_scaler_document(
        mode=arguments.mode,
        statistic=arguments.statistic,
        channel_names=channel_names,
        scalers=scalers,
        epsilon=CONFIG.zscore_epsilon,
        training_subjects=(
            sorted(set(manifest.loc[manifest["split"] == "train", "subject"]))
            if arguments.mode == "global"
            else None
        ),
    )

    if arguments.dry_run:
        print_header("DRY RUN COMPLETE")
        print(f"Fitted {len(scalers)} scaler(s); no array was modified.")
        return

    print_header("PASS 2: REWRITE PROCESSED ARRAYS")

    state = {
        "normalization_mode": arguments.mode,
        "statistic": arguments.statistic,
        "scaler_document": str(scaler_document_path()),
        "in_progress": True,
        "converted_recordings": [],
    }
    state_file = state_path()
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps(state, indent=2), encoding="utf-8")

    converted = 0
    for scaler_key, group in groups:
        raw_center, raw_scale = raw_parameters[scaler_key]
        for row in group.itertuples(index=False):
            current_center, current_scale = current_raw_transform(
                current_document,
                subject=row.subject,
                recording_id=row.recording_id,
            )
            multiplier = current_scale / raw_scale
            offset = (current_center - raw_center) / raw_scale
            availability = load_availability(row.availability_path)
            rewrite_array(
                Path(row.array_path),
                availability,
                multiplier,
                offset,
                CONFIG.signal_dtype,
            )
            converted += 1
            state["converted_recordings"].append(row.recording_id)
            if converted % 50 == 0:
                print(f"    rewritten {converted}/{len(manifest)} recordings")
                state_file.write_text(
                    json.dumps(state, indent=2),
                    encoding="utf-8",
                )

    save_scaler_document(document, scaler_document_path())
    state["in_progress"] = False
    state_file.write_text(json.dumps(state, indent=2), encoding="utf-8")

    print_header("RE-STANDARDIZATION COMPLETE")
    print(f"Rewrote {converted} recording(s).")
    print(f"Scaler document: {scaler_document_path()}")

    stale = mark_derived_caches_stale(arguments.mode, arguments.statistic)
    if stale:
        print("\n" + "!" * 90)
        print("DERIVED CACHES ARE NOW STALE and were marked with STALE.txt:")
        for cache_directory in stale:
            print(f"  {cache_directory}")
        print(
            "\nRegenerate them before training or evaluating:\n"
            "  scripts/cache_eegnet_embeddings.py\n"
            "  scripts/cache_handcrafted_features.py"
        )
        print("!" * 90)


if __name__ == "__main__":
    main()
