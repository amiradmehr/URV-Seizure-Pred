r"""Train the EEGNet plus mean-pooling seizure-risk baseline.

The model receives the 45 minutes preceding each one-minute decision point,
represented as 540 five-second EEG chunks. EEGNet encodes each chunk and mean
pooling combines all chunk embeddings into one probability of seizure onset in
the following 10 minutes.

Example (start with a one-epoch smoke test):

    .venv\Scripts\python.exe scripts\train_eegnet_baseline.py --epochs 1
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIRECTORY = PROJECT_ROOT / "src"

if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))


from seizure_prediction.config import CONFIG  # noqa: E402
from seizure_prediction.datasets import (  # noqa: E402
    StreamingDecisionDataset,
    load_decision_examples,
)
from seizure_prediction.models import (  # noqa: E402
    EEGNetMeanPoolConfig,
    EEGNetMeanPoolRiskModel,
)


def parse_arguments() -> argparse.Namespace:
    """Parse training settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--encoder-chunk-batch-size", type=int, default=64)
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.25,
        help="EEGNet dropout probability; try values such as 0.25, 0.4, or 0.5.",
    )
    parser.add_argument(
        "--train-negative-ratio",
        type=float,
        default=4.0,
        help="Retain this many training negatives per positive decision.",
    )
    parser.add_argument(
        "--validation-negative-ratio",
        type=float,
        default=None,
        help=(
            "Optional validation negatives per positive. Omit to evaluate all "
            "validation decisions; use a value only for faster development runs."
        ),
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "models" / "eegnet_mean_pool",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="Print a running training summary every N batches (0 disables).",
    )
    return parser.parse_args()


def log(message: str) -> None:
    """Print immediately so piped/background runs show live progress."""
    print(message, flush=True)


def set_seed(seed: int) -> None:
    """Set reproducible random seeds for the baseline experiment."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested_device: str) -> torch.device:
    """Select the requested training device with a clear CUDA failure."""
    if requested_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but no CUDA device is available.")
        return torch.device("cuda")
    if requested_device == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_loader(
    examples,
    *,
    shuffle: bool,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    """Build a lazy loader over continuous recordings."""
    dataset = StreamingDecisionDataset(examples, CONFIG)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )


def binary_metrics(labels: list[float], probabilities: list[float]) -> dict[str, float]:
    """Return probability-quality metrics for one evaluation split."""
    targets = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(probabilities, dtype=np.float64)
    if len(np.unique(targets)) != 2:
        raise ValueError("Evaluation split must contain both classes.")
    return {
        "accuracy": float(np.mean((scores >= 0.5) == targets)),
        "average_precision": float(average_precision_score(targets, scores)),
        "roc_auc": float(roc_auc_score(targets, scores)),
        "brier_score": float(brier_score_loss(targets, scores)),
    }


def save_training_curves(
    history: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Save loss and accuracy learning curves for a completed training run."""
    if not history:
        raise ValueError("Cannot plot an empty training history.")

    epochs = [metrics["epoch"] for metrics in history]
    figure, (loss_axis, accuracy_axis) = plt.subplots(1, 2, figsize=(12, 5))

    loss_axis.plot(
        epochs,
        [metrics["train_loss"] for metrics in history],
        marker="o",
        label="Training loss",
    )
    loss_axis.plot(
        epochs,
        [metrics["validation_loss"] for metrics in history],
        marker="o",
        label="Validation loss",
    )
    loss_axis.set(title="Loss", xlabel="Epoch", ylabel="Binary cross-entropy")
    loss_axis.grid(alpha=0.3)
    loss_axis.legend()

    accuracy_axis.plot(
        epochs,
        [metrics["train_accuracy"] for metrics in history],
        marker="o",
        label="Training accuracy",
    )
    accuracy_axis.plot(
        epochs,
        [metrics["validation_accuracy"] for metrics in history],
        marker="o",
        label="Validation accuracy",
    )
    accuracy_axis.set(
        title="Accuracy at a 0.5 threshold",
        xlabel="Epoch",
        ylabel="Accuracy",
        ylim=(0.0, 1.0),
    )
    accuracy_axis.grid(alpha=0.3)
    accuracy_axis.legend()

    figure.suptitle("EEGNet mean-pooling learning curves")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    description: str = "Validating",
) -> dict[str, float]:
    """Evaluate loss and discrimination/calibration metrics without gradients."""
    model.eval()
    total_loss = 0.0
    total_examples = 0
    labels: list[float] = []
    probabilities: list[float] = []

    with torch.no_grad():
        progress = tqdm(loader, desc=description, unit="batch", leave=True)
        for signal, availability, target in progress:
            signal = signal.to(device, non_blocking=True)
            availability = availability.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            logits = model(signal, availability)
            loss = criterion(logits, target)

            total_loss += loss.item() * len(target)
            total_examples += len(target)
            labels.extend(target.cpu().tolist())
            probabilities.extend(torch.sigmoid(logits).cpu().tolist())
            progress.set_postfix(loss=f"{total_loss / total_examples:.4f}")

    return {
        "loss": total_loss / total_examples,
        **binary_metrics(labels, probabilities),
    }


def main() -> None:
    """Train and checkpoint the EEGNet mean-pooling baseline."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    arguments = parse_arguments()
    CONFIG.validate()
    if arguments.epochs <= 0 or arguments.batch_size <= 0:
        raise ValueError("epochs and batch-size must be positive.")
    if arguments.num_workers < 0:
        raise ValueError("num-workers cannot be negative.")
    if not 0.0 <= arguments.dropout < 1.0:
        raise ValueError("dropout must be at least 0 and less than 1.")

    set_seed(arguments.seed)
    device = resolve_device(arguments.device)
    manifest_path = CONFIG.manifests_dir / "processed_shard_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Processed manifest not found: {manifest_path}. "
            "Run build_dataset.py and validate_dataset.py first."
        )

    log(f"Device: {device}")
    if device.type == "cuda":
        log(f"GPU: {torch.cuda.get_device_name(device)}")

    log("Loading training decision metadata...")
    train_examples = load_decision_examples(
        manifest_path,
        split="train",
        negative_to_positive_ratio=arguments.train_negative_ratio,
        seed=arguments.seed,
    )
    log("Loading validation decision metadata...")
    validation_examples = load_decision_examples(
        manifest_path,
        split="validation",
        negative_to_positive_ratio=arguments.validation_negative_ratio,
        seed=arguments.seed,
    )
    log("Building data loaders...")
    train_loader = build_loader(
        train_examples,
        shuffle=True,
        batch_size=arguments.batch_size,
        num_workers=arguments.num_workers,
        device=device,
    )
    validation_loader = build_loader(
        validation_examples,
        shuffle=False,
        batch_size=arguments.batch_size,
        num_workers=arguments.num_workers,
        device=device,
    )

    model_config = EEGNetMeanPoolConfig(
        n_chans=len(CONFIG.canonical_channel_names),
        chunk_samples=int(CONFIG.chunk_window_seconds * CONFIG.target_sfreq),
        embedding_dim=arguments.embedding_dim,
        encoder_chunk_batch_size=arguments.encoder_chunk_batch_size,
        dropout=arguments.dropout,
    )
    model = EEGNetMeanPoolRiskModel(model_config).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
    )

    train_positive_count = int((train_examples["label"] == 1).sum())
    train_negative_count = int((train_examples["label"] == 0).sum())
    positive_weight = train_negative_count / train_positive_count
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(positive_weight, device=device)
    )

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = arguments.output_dir / "best_model.pt"
    metrics_path = arguments.output_dir / "metrics.json"
    curves_path = arguments.output_dir / "learning_curves.png"
    history: list[dict[str, Any]] = []
    best_average_precision = float("-inf")

    log(
        "Train decisions: "
        f"{len(train_examples)} "
        f"({train_positive_count} positive, {train_negative_count} negative)"
    )
    log(
        "Validation decisions: "
        f"{len(validation_examples)} "
        f"({int((validation_examples['label'] == 1).sum())} positive, "
        f"{int((validation_examples['label'] == 0).sum())} negative)"
    )
    log(
        "Each decision contains "
        f"{CONFIG.input_window_seconds / 60:.0f} minutes / "
        f"{CONFIG.input_window_seconds / CONFIG.chunk_window_seconds:.0f} chunks."
    )

    for epoch in range(1, arguments.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        cumulative_loss = 0.0
        cumulative_correct = 0
        examples_seen = 0

        log(f"\n=== Epoch {epoch}/{arguments.epochs} ===")
        train_progress = tqdm(
            train_loader,
            desc=f"Train epoch {epoch}",
            unit="batch",
            leave=True,
        )
        for batch_index, (signal, availability, target) in enumerate(
            train_progress,
            start=1,
        ):
            signal = signal.to(device, non_blocking=True)
            availability = availability.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(signal, availability)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()

            cumulative_loss += loss.item() * len(target)
            cumulative_correct += int(
                ((logits >= 0.0) == (target >= 0.5)).sum().item()
            )
            examples_seen += len(target)

            running_loss = cumulative_loss / examples_seen
            running_accuracy = cumulative_correct / examples_seen
            train_progress.set_postfix(
                loss=f"{running_loss:.4f}",
                acc=f"{running_accuracy:.3f}",
            )

            if (
                arguments.log_interval > 0
                and batch_index % arguments.log_interval == 0
            ):
                log(
                    f"Epoch {epoch} batch {batch_index}/{len(train_loader)} "
                    f"loss={running_loss:.4f} acc={running_accuracy:.3f}"
                )

        validation_metrics = evaluate(
            model,
            validation_loader,
            criterion,
            device,
            description=f"Validate epoch {epoch}",
        )
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": cumulative_loss / examples_seen,
            "train_accuracy": cumulative_correct / examples_seen,
            **{f"validation_{name}": value for name, value in validation_metrics.items()},
        }
        history.append(epoch_metrics)
        elapsed_minutes = (time.perf_counter() - epoch_start) / 60.0
        log(
            f"Epoch {epoch} finished in {elapsed_minutes:.1f} min: "
            f"train_loss={epoch_metrics['train_loss']:.4f} "
            f"val_auc={epoch_metrics['validation_roc_auc']:.4f} "
            f"val_ap={epoch_metrics['validation_average_precision']:.4f}"
        )
        log(json.dumps(epoch_metrics, sort_keys=True))

        if validation_metrics["average_precision"] > best_average_precision:
            best_average_precision = validation_metrics["average_precision"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": asdict(model_config),
                    "config": {
                        "target_sfreq": CONFIG.target_sfreq,
                        "input_window_seconds": CONFIG.input_window_seconds,
                        "chunk_window_seconds": CONFIG.chunk_window_seconds,
                        "canonical_channel_names": list(CONFIG.canonical_channel_names),
                        "seizure_occurrence_period_minutes": (
                            CONFIG.seizure_occurrence_period_minutes
                        ),
                    },
                    "training_arguments": vars(arguments),
                    "best_validation_average_precision": best_average_precision,
                },
                checkpoint_path,
            )

        metrics_path.write_text(
            json.dumps(
                {
                    "train_examples": len(train_examples),
                    "validation_examples": len(validation_examples),
                    "train_negative_to_positive_ratio": arguments.train_negative_ratio,
                    "validation_negative_to_positive_ratio": (
                        arguments.validation_negative_ratio
                    ),
                    "history": history,
                    "best_validation_average_precision": best_average_precision,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    save_training_curves(history, curves_path)
    log(f"Best checkpoint: {checkpoint_path}")
    log(f"Training metrics: {metrics_path}")
    log(f"Learning curves: {curves_path}")
    log("The held-out test split was not used.")


if __name__ == "__main__":
    main()
