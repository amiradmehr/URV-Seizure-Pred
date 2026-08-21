r"""Train the unchanged EEGNet with patient/event-balanced exposure.

This is a controlled sampling experiment: architecture, preprocessing,
natural-prevalence validation, and AP checkpoint selection match the selected
baseline. Training caps one patient at four distinct seizure events per epoch,
uses one rotating positive window per selected event, and distributes negative
examples evenly across patients and their recordings.

PowerShell example:

    .\.venv-old\Scripts\python.exe scripts\train_eegnet_patient_event_balanced.py --device cuda
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def add_default_option(name: str, value: str) -> None:
    """Add a wrapper default while allowing explicit CLI overrides."""
    if not any(
        argument == name or argument.startswith(f"{name}=")
        for argument in sys.argv[1:]
    ):
        sys.argv.extend([name, value])


def main() -> None:
    """Apply controlled defaults and delegate to the baseline trainer."""
    add_default_option("--sampling-strategy", "patient-event-balanced")
    add_default_option("--max-events-per-patient", "4")
    add_default_option("--negative-to-positive-ratio", "10")
    add_default_option("--learning-rate", "1e-4")
    add_default_option("--weight-decay", "1e-3")
    add_default_option("--dropout", "0.4")
    add_default_option("--batch-size", "2")
    add_default_option("--gradient-accumulation-steps", "8")
    add_default_option("--early-stopping-patience", "6")
    add_default_option("--epochs", "20")
    add_default_option(
        "--output-dir",
        str(
            PROJECT_ROOT
            / "outputs"
            / "models"
            / "eegnet_patient_event_balanced_ratio10_lr1e4"
        ),
    )

    from train_eegnet_baseline import main as train_baseline

    train_baseline()


if __name__ == "__main__":
    main()
