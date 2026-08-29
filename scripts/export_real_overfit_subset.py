from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIRECTORY = PROJECT_ROOT / "src"

if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))


from seizure_prediction.config import (
    add_label_definition_arguments,
    resolve_label_definition,
)
from seizure_prediction.datasets import (
    StreamingDecisionDataset,
    load_decision_examples,
    load_scaler_document,
)


SEED = 42
NUMBER_OF_SEIZURES = 2

OUTPUT_DIRECTORY = PROJECT_ROOT / "outputs" / "colab_smoke"
OUTPUT_TENSOR_PATH = OUTPUT_DIRECTORY / "real_overfit_subset.pt"
OUTPUT_METADATA_PATH = OUTPUT_DIRECTORY / "real_overfit_subset_metadata.csv"


def parse_arguments() -> argparse.Namespace:
    """Parse the label definition to export a subset from."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_label_definition_arguments(parser)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    config = resolve_label_definition(arguments)
    config.validate()
    rng = np.random.default_rng(SEED)

    manifest_path = (
        config.manifests_dir
        / "processed_shard_manifest.csv"
    )

    scaler_document = load_scaler_document(manifest_path, config.project_root)

    examples = load_decision_examples(
        manifest_path,
        split="train",
        negative_to_positive_ratio=None,
        seed=SEED,
        project_root=config.project_root,
    )

    positives = examples[
        examples["label"] == 1
    ].copy()

    # Prefer complete seizure events containing all ten lead-time decisions.
    decisions_per_seizure = (
        positives.groupby("target_seizure_id")
        .size()
    )

    complete_seizure_ids = decisions_per_seizure[
        decisions_per_seizure == 10
    ].index.to_numpy()

    if len(complete_seizure_ids) < NUMBER_OF_SEIZURES:
        raise RuntimeError(
            "Not enough complete ten-decision seizure events."
        )

    selected_seizure_ids = rng.choice(
        complete_seizure_ids,
        size=NUMBER_OF_SEIZURES,
        replace=False,
    )

    selected_positives = positives[
        positives["target_seizure_id"].isin(
            selected_seizure_ids
        )
    ].copy()

    selected_recordings = set(
        selected_positives["recording_id"]
    )

    selected_subjects = set(
        selected_positives["subject"]
    )

    # First try to match negatives from the exact same recordings.
    negative_candidates = examples[
        (examples["label"] == 0)
        & examples["recording_id"].isin(selected_recordings)
    ].copy()

    # Fall back to the same patients if the recordings do not contain enough.
    if len(negative_candidates) < len(selected_positives):
        negative_candidates = examples[
            (examples["label"] == 0)
            & examples["subject"].isin(selected_subjects)
        ].copy()

    if len(negative_candidates) < len(selected_positives):
        raise RuntimeError(
            "Not enough matched negative decisions."
        )

    selected_negatives = negative_candidates.sample(
        n=len(selected_positives),
        replace=False,
        random_state=SEED,
    )

    subset = pd.concat(
        [selected_positives, selected_negatives],
        ignore_index=True,
    ).sample(
        frac=1.0,
        random_state=SEED,
    ).reset_index(drop=True)

    dataset = StreamingDecisionDataset(
        subset,
        config,
        scaler_document,
    )

    sequence_chunks = int(
        config.input_window_seconds
        / config.chunk_window_seconds
    )

    chunk_samples = int(
        config.chunk_window_seconds
        * config.target_sfreq
    )

    number_of_channels = len(
        config.canonical_channel_names
    )

    # Float16 makes the 40-example bundle approximately 160 MiB.
    signals = torch.empty(
        (
            len(dataset),
            sequence_chunks,
            number_of_channels,
            chunk_samples,
        ),
        dtype=torch.float16,
    )

    availability = torch.empty(
        (len(dataset), number_of_channels),
        dtype=torch.float32,
    )

    targets = torch.empty(
        len(dataset),
        dtype=torch.float32,
    )

    for index in range(len(dataset)):
        signal, channel_mask, target = dataset[index]

        signals[index].copy_(
            signal.to(dtype=torch.float16)
        )
        availability[index].copy_(channel_mask)
        targets[index] = target

        print(
            f"Materialized {index + 1}/{len(dataset)}",
            flush=True,
        )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "signals": signals,
            "availability": availability,
            "targets": targets,
        },
        OUTPUT_TENSOR_PATH,
    )

    metadata_columns = [
        column
        for column in (
            "subject",
            "recording_id",
            "decision_time_seconds",
            "label",
            "target_seizure_id",
            "bte_side",
        )
        if column in subset.columns
    ]

    subset[metadata_columns].to_csv(
        OUTPUT_METADATA_PATH,
        index=False,
    )

    print()
    print("Selected seizures:", selected_seizure_ids.tolist())
    print("Examples:", len(subset))
    print("Positive examples:", int(targets.sum().item()))
    print("Negative examples:", int((targets == 0).sum().item()))
    print("Tensor bundle:", OUTPUT_TENSOR_PATH)
    print("Metadata:", OUTPUT_METADATA_PATH)
    print(
        "Bundle size:",
        OUTPUT_TENSOR_PATH.stat().st_size / 2**20,
        "MiB",
    )


if __name__ == "__main__":
    main()