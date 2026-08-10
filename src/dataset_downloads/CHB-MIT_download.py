from pathlib import Path

from mne.datasets import fetch_dataset

from braindecode.datasets import CHBMIT
from braindecode.datasets.chb_mit import CHB_MIT_dataset_params
from braindecode.datasets.utils import _correct_dataset_path

data_dir = Path("~/mne_data/openneuro/").expanduser() # Downloads to the user directory
dataset_root = data_dir / "BIDS_CHB-MIT"

if not dataset_root.exists():
    path_root = fetch_dataset(
        dataset_params=CHB_MIT_dataset_params,
        path=data_dir,
        processor="unzip",
        force_update=False,
    )
    dataset_root = _correct_dataset_path(
        path_root, CHB_MIT_dataset_params["archive_name"], "BIDS_CHB-MIT"
    )

bids_ds = CHBMIT(root=dataset_root)
