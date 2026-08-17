from pathlib import Path

from braindecode.datasets import BIDSDataset


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = PROJECT_ROOT / "data" / "seizeit2" / "raw"

SUBJECT_ID = "001"


def main() -> None:
    print("=" * 80)
    print("SEIZEIT2 BRAINDECODE DATASET TEST")
    print("=" * 80)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Dataset root: {DATA_ROOT}")
    print(f"Dataset root exists: {DATA_ROOT.exists()}")

    description_file = DATA_ROOT / "dataset_description.json"
    participants_file = DATA_ROOT / "participants.tsv"
    subject_folder = DATA_ROOT / f"sub-{SUBJECT_ID}"

    print(f"dataset_description.json exists: {description_file.exists()}")
    print(f"participants.tsv exists: {participants_file.exists()}")
    print(f"Subject folder exists: {subject_folder.exists()}")

    if not DATA_ROOT.exists():
        raise FileNotFoundError(
            f"Dataset root does not exist: {DATA_ROOT}"
        )

    print("\nLoading one subject through Braindecode...")

    dataset = BIDSDataset(
        root=DATA_ROOT,
        subjects=[SUBJECT_ID],
        datatypes=["eeg"],
        preload=False,
        n_jobs=1,
    )

    print(f"\nNumber of recordings found: {len(dataset.datasets)}")

    if len(dataset.datasets) == 0:
        raise RuntimeError(
            f"Braindecode found no EEG recordings for sub-{SUBJECT_ID}."
        )

    first_dataset = dataset.datasets[0]
    raw = first_dataset.raw

    print("\nFirst recording description:")
    print(first_dataset.description)

    print("\nEEG information:")
    print(f"Channels: {raw.ch_names}")
    print(f"Channel types: {raw.get_channel_types()}")
    print(f"Sampling frequency: {raw.info['sfreq']} Hz")
    print(f"Number of samples: {raw.n_times:,}")
    print(f"Duration: {raw.times[-1]:.2f} seconds")
    print(f"Duration: {raw.times[-1] / 3600:.2f} hours")
    print(f"Bad channels: {raw.info['bads']}")

    print("\nAnnotations:")
    print(f"Number of annotations: {len(raw.annotations)}")

    if len(raw.annotations) > 0:
        for index, annotation in enumerate(raw.annotations[:10]):
            print(
                f"{index}: "
                f"onset={annotation['onset']:.2f}, "
                f"duration={annotation['duration']:.2f}, "
                f"description={annotation['description']}"
            )
    else:
        print("No annotations were attached to this recording.")

    print("\nReading 10 seconds of signal data...")

    sfreq = float(raw.info["sfreq"])
    stop_sample = int(10 * sfreq)

    signal = raw.get_data(
        start=0,
        stop=stop_sample,
    )

    print(f"Loaded signal shape: {signal.shape}")
    print(f"Signal dtype: {signal.dtype}")
    print(f"All values finite: {bool((signal == signal).all())}")

    expected_samples = int(10 * sfreq)

    print(
        "Expected shape approximately: "
        f"({len(raw.ch_names)}, {expected_samples})"
    )

    print("\nDataset loading test completed.")


if __name__ == "__main__":
    main()