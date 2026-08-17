from pathlib import Path

import openneuro

from braindecode.datasets import BIDSDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
dataset_name = "ds005873"
dataset_root = PROJECT_ROOT / "data" / "seizeit2" / "raw"

if not dataset_root.exists():
    openneuro.download(
        dataset=dataset_name,
        target_dir=dataset_root,
        exclude=[ # Exclude the other modalities
            "sub-*/ses-*/ecg",
            "sub-*/ses-*/emg",
            "sub-*/ses-*/mov",
        ],
    )

# Now, loading the dataset is simply a one-line command:
bids_ds = BIDSDataset(dataset_root)
print(bids_ds.datasets[0].raw.annotations)