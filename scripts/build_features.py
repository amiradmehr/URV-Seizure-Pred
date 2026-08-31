r"""Precompute chunk-level spectral features for every processed recording.

Turns each recording into a compact (n_chunks, n_features) float32 bank aligned
to the 5-second chunk grid, written beside the processed shards.

One decision then costs 540 x 27 floats (58 KB) instead of 540 x 3 x 1280
(8.3 MB), which is what makes patient-wise cross-validation affordable.

Runs as a SLURM array; each task takes a disjoint stripe of recordings:

    python scripts/build_features.py --shard-index 0 --shard-count 20
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIRECTORY = PROJECT_ROOT / "src"
if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

from seizure_prediction.config import CONFIG  # noqa: E402
from seizure_prediction.features import (  # noqa: E402
    chunk_features,
    feature_names,
    features_per_channel,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CONFIG.interim_data_dir / "chunk_features",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    CONFIG.validate()
    if not 0 <= arguments.shard_index < arguments.shard_count:
        raise ValueError("shard-index must be within [0, shard-count).")

    output = arguments.output_dir
    output.mkdir(parents=True, exist_ok=True)

    manifest_path = CONFIG.manifests_dir / "processed_shard_manifest.csv"
    manifest = pd.read_csv(manifest_path, dtype={"subject": str})
    mine = manifest.iloc[arguments.shard_index :: arguments.shard_count]

    chunk_samples = int(round(CONFIG.chunk_window_seconds * CONFIG.target_sfreq))
    print(
        f"shard {arguments.shard_index+1}/{arguments.shard_count}: "
        f"{len(mine)} of {len(manifest)} recordings",
        flush=True,
    )

    if arguments.shard_index == 0:
        (output / "feature_names.json").write_text(
            json.dumps(feature_names(CONFIG.canonical_channel_names), indent=2),
            encoding="utf-8",
        )

    started = time.perf_counter()
    written = skipped = 0
    for number, row in enumerate(mine.itertuples(index=False), start=1):
        destination = output / f"{row.recording_id}_features.npy"
        availability_destination = output / f"{row.recording_id}_availability.npy"
        if destination.exists() and availability_destination.exists():
            skipped += 1
            continue

        signal = np.load(row.X_path, mmap_mode="r")
        with open(row.channel_availability_path, encoding="utf-8") as handle:
            availability = np.asarray(json.load(handle), dtype=bool)

        features = chunk_features(
            np.asarray(signal, dtype=np.float32),
            sampling_frequency=CONFIG.target_sfreq,
            chunk_samples=chunk_samples,
            availability=availability,
        )
        np.save(destination, features)

        # Per-feature-column availability, so the model can mask absent channels.
        per_channel = features_per_channel()
        columns = np.repeat(availability, per_channel)
        np.save(availability_destination, columns)
        written += 1

        if number % 25 == 0:
            elapsed = time.perf_counter() - started
            print(
                f"  [{number}/{len(mine)}] {row.recording_id} "
                f"{features.shape}  {elapsed/number:.2f} s/recording",
                flush=True,
            )

    elapsed = time.perf_counter() - started
    print(
        f"done: wrote {written}, skipped {skipped}, {elapsed/60:.1f} min",
        flush=True,
    )


if __name__ == "__main__":
    main()
