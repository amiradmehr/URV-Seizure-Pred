"""Download the SeizeIT2 BIDS dataset into the pipeline's raw data directory.

Run from the repository root:

    python scripts/download_seizeit2.py

The download target is `CONFIG.raw_data_dir`, which is where `build_dataset.py`
looks for it. Only the EEG modality is fetched; ECG, EMG, and movement are
excluded.
"""

from __future__ import annotations

import sys
from pathlib import Path

import openneuro

from braindecode.datasets import BIDSDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIRECTORY = PROJECT_ROOT / "src"

if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))


from seizure_prediction.config import CONFIG  # noqa: E402


DATASET_NAME = "ds005873"


def main() -> None:
    """Download the dataset unless it is already present."""
    dataset_root = CONFIG.raw_data_dir

    if not (dataset_root / "dataset_description.json").exists():
        dataset_root.mkdir(parents=True, exist_ok=True)
        openneuro.download(
            dataset=DATASET_NAME,
            target_dir=dataset_root,
            exclude=[  # Exclude the other modalities
                "sub-*/ses-*/ecg",
                "sub-*/ses-*/emg",
                "sub-*/ses-*/mov",
            ],
        )
    else:
        print(f"Dataset already present: {dataset_root}")

    # Confirm the download is a readable BIDS root before the pipeline runs.
    BIDSDataset(dataset_root)
    print(f"SeizeIT2 is ready at {dataset_root}")


if __name__ == "__main__":
    main()
