r"""Train the single active EEGNet seizure-risk baseline.

The model uses 45 minutes of EEG to predict seizure onset in the next
10 minutes. Training sees every positive decision and a new 4:1 sample of
negative decisions each epoch. Validation always uses the complete natural,
patient-held-out validation split.

The deliberately minimal reporting protocol is:

* training: binary cross-entropy loss;
* validation: binary cross-entropy loss and average precision (AP);
* model selection/comparison: validation AP (higher is better).

Example:

    .venv/bin/python scripts/train_eegnet_baseline.py --device cuda
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import average_precision_score, precision_recall_curve
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader, Sampler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIRECTORY = PROJECT_ROOT / "src"

if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))


from seizure_prediction.config import CONFIG  # noqa: E402
from seizure_prediction.datasets import (  # noqa: E402
    BalancedEpochSampler,
    StreamingDecisionDataset,
    load_decision_examples,
)
from seizure_prediction.models import (  # noqa: E402
    BaselineEEGNet,
    BaselineEEGNetConfig,
)


@dataclass(frozen=True)
class EvaluationResult:
    """Minimal validation results plus values needed to draw the figures."""

    loss: float
    average_precision: float
    labels: np.ndarray
    probabilities: np.ndarray


def parse_arguments() -> argparse.Namespace:
    """Return conservative starting settings for the first clean baseline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--encoder-chunk-batch-size", type=int, default=128)
    parser.add_argument(
        "--negative-to-positive-ratio",
        type=float,
        default=4.0,
        help=(
            "Training negatives sampled per positive each epoch. The default "
            "4:1 retains useful imbalance while giving positives enough signal."
        ),
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=8,
        help="Accumulate small GPU batches into a more stable effective batch.",
    )
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=6,
        help="Stop after this many epochs without better validation AP.",
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "models" / "eegnet_baseline",
        help="A separate directory is recommended for every experimental run.",
    )
    return parser.parse_args()


def validate_arguments(arguments: argparse.Namespace) -> None:
    """Fail before data loading when a run setting is invalid."""
    positive_integer_names = (
        "epochs",
        "batch_size",
        "embedding_dim",
        "encoder_chunk_batch_size",
        "gradient_accumulation_steps",
        "early_stopping_patience",
    )
    for name in positive_integer_names:
        if getattr(arguments, name) <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be positive.")
    if arguments.num_workers < 0:
        raise ValueError("num-workers cannot be negative.")
    if arguments.learning_rate <= 0 or arguments.weight_decay < 0:
        raise ValueError("learning-rate must be positive and weight-decay nonnegative.")
    if arguments.negative_to_positive_ratio <= 0:
        raise ValueError("negative-to-positive-ratio must be positive.")
    if arguments.max_grad_norm <= 0:
        raise ValueError("max-grad-norm must be positive.")
    if not 0.0 <= arguments.dropout < 1.0:
        raise ValueError("dropout must be at least zero and less than one.")


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for a reproducible baseline run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested_device: str) -> torch.device:
    """Select CUDA when available unless the caller explicitly requests CPU."""
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available.")
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested_device)


def build_loader(
    examples,
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    sampler: Sampler[int] | None = None,
) -> DataLoader:
    """Build a lazy loader without copying complete recordings into memory."""
    return DataLoader(
        StreamingDecisionDataset(examples, CONFIG),
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )


def sampling_prior_logit_correction(
    population_positive_fraction: float,
    sampled_positive_fraction: float,
) -> float:
    """Correct probabilities after negative examples are undersampled."""
    for name, fraction in (
        ("population_positive_fraction", population_positive_fraction),
        ("sampled_positive_fraction", sampled_positive_fraction),
    ):
        if not 0.0 < fraction < 1.0:
            raise ValueError(f"{name} must be strictly between zero and one.")

    def log_odds(probability: float) -> float:
        return math.log(probability / (1.0 - probability))

    return log_odds(population_positive_fraction) - log_odds(
        sampled_positive_fraction
    )


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    logit_correction: float,
) -> EvaluationResult:
    """Evaluate only validation loss and the primary comparison metric, AP."""
    model.eval()
    total_loss = 0.0
    total_examples = 0
    labels: list[float] = []
    probabilities: list[float] = []

    with torch.no_grad():
        for signal, availability, target in loader:
            signal = signal.to(device, non_blocking=True)
            availability = availability.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            corrected_logits = model(signal, availability) + logit_correction
            loss = criterion(corrected_logits, target)

            total_loss += loss.item() * len(target)
            total_examples += len(target)
            labels.extend(target.cpu().tolist())
            probabilities.extend(torch.sigmoid(corrected_logits).cpu().tolist())

    label_values = np.asarray(labels, dtype=np.int64)
    probability_values = np.asarray(probabilities, dtype=np.float64)
    if total_examples == 0 or len(np.unique(label_values)) != 2:
        raise ValueError("Validation must contain examples from both classes.")

    return EvaluationResult(
        loss=total_loss / total_examples,
        average_precision=float(
            average_precision_score(label_values, probability_values)
        ),
        labels=label_values,
        probabilities=probability_values,
    )


def save_learning_curves(
    history: list[dict[str, float | int]],
    best_epoch: int,
    prevalence: float,
    output_path: Path,
) -> None:
    """Plot the minimal training and validation metrics for this run."""
    epochs = [int(row["epoch"]) for row in history]
    figure, (loss_axis, ap_axis) = plt.subplots(1, 2, figsize=(12, 5))

    loss_axis.plot(
        epochs,
        [float(row["train_loss"]) for row in history],
        marker="o",
        label="Training loss (sampled data)",
    )
    loss_axis.plot(
        epochs,
        [float(row["validation_loss"]) for row in history],
        marker="o",
        label="Validation loss (natural data)",
    )
    loss_axis.axvline(best_epoch, color="black", linestyle=":", label="Best epoch")
    loss_axis.set(
        title="Optimization and generalization",
        xlabel="Epoch",
        ylabel="Binary cross-entropy",
    )
    loss_axis.grid(alpha=0.3)
    loss_axis.legend()

    ap_axis.plot(
        epochs,
        [float(row["validation_average_precision"]) for row in history],
        marker="o",
        label="Validation average precision",
    )
    ap_axis.axhline(
        prevalence,
        color="black",
        linestyle="--",
        label=f"Random baseline ({prevalence:.4f})",
    )
    ap_axis.axvline(best_epoch, color="black", linestyle=":", label="Best epoch")
    ap_axis.set(
        title="Primary model-comparison metric",
        xlabel="Epoch",
        ylabel="Average precision (higher is better)",
        ylim=(0.0, 1.0),
    )
    ap_axis.grid(alpha=0.3)
    ap_axis.legend()

    figure.suptitle("Baseline EEGNet learning curves")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_model_overview(
    config: BaselineEEGNetConfig,
    output_path: Path,
) -> None:
    """Draw a presentation-ready overview of the active model architecture."""
    figure, axis = plt.subplots(figsize=(15, 4))
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.axis("off")

    box_width = 0.135
    box_height = 0.28
    box_y = 0.48
    box_x_values = (0.015, 0.18, 0.345, 0.51, 0.675, 0.84)
    labels = (
        "45-min EEG input\n540 × 3 × 1,280",
        "Shared EEGNet\non each 5-s chunk",
        f"Chunk features\n540 × {config.embedding_dim}",
        f"Mean pooling\n{config.embedding_dim} features",
        "Add electrode mask\n+ 3 availability values",
        "Linear classifier\n1 risk probability",
    )

    for index, (box_x, label) in enumerate(zip(box_x_values, labels)):
        box = FancyBboxPatch(
            (box_x, box_y),
            box_width,
            box_height,
            boxstyle="round,pad=0.015",
            facecolor="#eaf2f8",
            edgecolor="#2c3e50",
            linewidth=1.5,
        )
        axis.add_patch(box)
        axis.text(
            box_x + box_width / 2,
            box_y + box_height / 2,
            label,
            ha="center",
            va="center",
            fontsize=10,
        )
        if index < len(box_x_values) - 1:
            axis.annotate(
                "",
                xy=(box_x_values[index + 1] - 0.006, box_y + box_height / 2),
                xytext=(box_x + box_width + 0.006, box_y + box_height / 2),
                arrowprops={"arrowstyle": "->", "linewidth": 1.5},
            )

    axis.text(
        0.5,
        0.25,
        "One output per decision: seizure onset within the next 10 minutes",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
    )
    axis.text(
        0.5,
        0.12,
        "Baseline limitation: mean pooling gives all chunks equal weight and "
        "does not retain their order.",
        ha="center",
        va="center",
        fontsize=10,
    )
    figure.suptitle("Baseline EEGNet model overview", fontsize=15)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_validation_summary(
    result: EvaluationResult,
    output_path: Path,
) -> None:
    """Plot the best model's precision-recall curve and risk distributions."""
    prevalence = float(np.mean(result.labels))
    precision, recall, _ = precision_recall_curve(
        result.labels,
        result.probabilities,
    )
    positive_scores = result.probabilities[result.labels == 1]
    negative_scores = result.probabilities[result.labels == 0]

    figure, (pr_axis, score_axis) = plt.subplots(1, 2, figsize=(12, 5))
    pr_axis.plot(
        recall,
        precision,
        label=f"EEGNet (AP = {result.average_precision:.4f})",
    )
    pr_axis.axhline(
        prevalence,
        color="black",
        linestyle="--",
        label=f"Random baseline ({prevalence:.4f})",
    )
    pr_axis.set(
        title="Precision-recall curve",
        xlabel="Recall (sensitivity)",
        ylabel="Precision",
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.0),
    )
    pr_axis.grid(alpha=0.3)
    pr_axis.legend()

    maximum_score = max(0.05, float(np.max(result.probabilities)))
    bins = np.linspace(0.0, maximum_score, 51)
    score_axis.hist(
        negative_scores,
        bins=bins,
        density=True,
        alpha=0.55,
        label="No seizure in next 10 min",
    )
    score_axis.hist(
        positive_scores,
        bins=bins,
        density=True,
        alpha=0.55,
        label="Seizure in next 10 min",
    )
    score_axis.set(
        title="Validation risk-score distributions",
        xlabel="Prior-corrected predicted probability",
        ylabel="Density",
        xlim=(0.0, maximum_score),
    )
    score_axis.grid(alpha=0.3)
    score_axis.legend()

    figure.suptitle("Best baseline EEGNet validation summary")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_metrics(
    *,
    output_path: Path,
    arguments: argparse.Namespace,
    counts: dict[str, int],
    population_positive_fraction: float,
    sampled_positive_fraction: float,
    logit_correction: float,
    history: list[dict[str, float | int]],
    best_epoch: int,
    best_average_precision: float,
) -> None:
    """Write one concise, machine-readable record of the baseline run."""
    output_path.write_text(
        json.dumps(
            {
                "model": "BaselineEEGNet",
                "primary_comparison_metric": "validation_average_precision",
                "training_metric": "binary_cross_entropy",
                "validation_metrics": [
                    "binary_cross_entropy",
                    "average_precision",
                ],
                "counts": counts,
                "negative_to_positive_ratio_per_epoch": (
                    arguments.negative_to_positive_ratio
                ),
                "training_population_positive_fraction": (
                    population_positive_fraction
                ),
                "training_sampled_positive_fraction": sampled_positive_fraction,
                "sampling_prior_logit_correction": logit_correction,
                "hyperparameters": {
                    "learning_rate": arguments.learning_rate,
                    "weight_decay": arguments.weight_decay,
                    "dropout": arguments.dropout,
                    "batch_size": arguments.batch_size,
                    "gradient_accumulation_steps": (
                        arguments.gradient_accumulation_steps
                    ),
                    "effective_batch_size": (
                        arguments.batch_size
                        * arguments.gradient_accumulation_steps
                    ),
                    "embedding_dim": arguments.embedding_dim,
                    "seed": arguments.seed,
                },
                "history": history,
                "best_epoch": best_epoch,
                "best_validation_average_precision": best_average_precision,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    """Train, select, checkpoint, and visualize the baseline EEGNet."""
    arguments = parse_arguments()
    validate_arguments(arguments)
    CONFIG.validate()
    set_seed(arguments.seed)
    device = resolve_device(arguments.device)

    manifest_path = CONFIG.manifests_dir / "processed_shard_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Processed manifest not found: {manifest_path}. "
            "Run build_dataset.py and validate_dataset.py first."
        )

    train_examples = load_decision_examples(
        manifest_path,
        split="train",
        negative_to_positive_ratio=None,
        seed=arguments.seed,
    )
    validation_examples = load_decision_examples(
        manifest_path,
        split="validation",
        negative_to_positive_ratio=None,
        seed=arguments.seed,
    )
    train_sampler = BalancedEpochSampler(
        train_examples["label"],
        negative_to_positive_ratio=arguments.negative_to_positive_ratio,
        seed=arguments.seed,
    )
    train_loader = build_loader(
        train_examples,
        batch_size=arguments.batch_size,
        num_workers=arguments.num_workers,
        device=device,
        sampler=train_sampler,
    )
    validation_loader = build_loader(
        validation_examples,
        batch_size=arguments.batch_size,
        num_workers=arguments.num_workers,
        device=device,
    )

    train_positive_count = int((train_examples["label"] == 1).sum())
    train_negative_count = int((train_examples["label"] == 0).sum())
    validation_positive_count = int((validation_examples["label"] == 1).sum())
    validation_negative_count = int((validation_examples["label"] == 0).sum())
    population_positive_fraction = train_positive_count / len(train_examples)
    sampled_positive_fraction = train_positive_count / len(train_sampler)
    logit_correction = sampling_prior_logit_correction(
        population_positive_fraction,
        sampled_positive_fraction,
    )

    model_config = BaselineEEGNetConfig(
        n_chans=len(CONFIG.canonical_channel_names),
        chunk_samples=int(CONFIG.chunk_window_seconds * CONFIG.target_sfreq),
        sequence_chunks=int(
            CONFIG.input_window_seconds / CONFIG.chunk_window_seconds
        ),
        embedding_dim=arguments.embedding_dim,
        encoder_chunk_batch_size=arguments.encoder_chunk_batch_size,
        dropout=arguments.dropout,
        sampling_prior_logit_correction=logit_correction,
    )
    model = BaselineEEGNet(model_config).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
    )
    criterion = nn.BCEWithLogitsLoss()

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = arguments.output_dir / "best_model.pt"
    metrics_path = arguments.output_dir / "metrics.json"
    model_overview_path = arguments.output_dir / "model_overview.png"
    learning_curves_path = arguments.output_dir / "learning_curves.png"
    validation_summary_path = arguments.output_dir / "validation_summary.png"
    counts = {
        "all_training_decisions": len(train_examples),
        "training_positive_decisions": train_positive_count,
        "training_negative_decisions": train_negative_count,
        "training_decisions_per_epoch": len(train_sampler),
        "sampled_negatives_per_epoch": train_sampler.negative_count,
        "validation_decisions": len(validation_examples),
        "validation_positive_decisions": validation_positive_count,
        "validation_negative_decisions": validation_negative_count,
    }
    history: list[dict[str, float | int]] = []
    best_average_precision = float("-inf")
    best_epoch = 0
    epochs_without_improvement = 0

    print(f"Device: {device}")
    print(json.dumps(counts, indent=2))
    print(
        "Starting protocol: "
        f"negative:positive={arguments.negative_to_positive_ratio:g}:1, "
        f"learning_rate={arguments.learning_rate:g}, "
        f"dropout={arguments.dropout:g}, "
        f"effective_batch_size="
        f"{arguments.batch_size * arguments.gradient_accumulation_steps}."
    )
    print(
        "Primary model-comparison metric: average precision on the full "
        "natural-prevalence validation split."
    )
    save_model_overview(model_config, model_overview_path)

    for epoch in range(1, arguments.epochs + 1):
        train_sampler.set_epoch(epoch - 1)
        model.train()
        cumulative_train_loss = 0.0
        examples_seen = 0
        accumulated_batches = 0
        optimizer.zero_grad(set_to_none=True)

        for batch_index, (signal, availability, target) in enumerate(
            train_loader,
            start=1,
        ):
            signal = signal.to(device, non_blocking=True)
            availability = availability.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            logits = model(signal, availability)
            loss = criterion(logits, target)
            loss.backward()
            accumulated_batches += 1

            should_step = (
                accumulated_batches == arguments.gradient_accumulation_steps
                or batch_index == len(train_loader)
            )
            if should_step:
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(accumulated_batches)
                clip_grad_norm_(model.parameters(), arguments.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                accumulated_batches = 0

            cumulative_train_loss += loss.item() * len(target)
            examples_seen += len(target)

        validation_result = evaluate(
            model,
            validation_loader,
            criterion,
            device,
            logit_correction,
        )
        epoch_metrics: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": cumulative_train_loss / examples_seen,
            "validation_loss": validation_result.loss,
            "validation_average_precision": (
                validation_result.average_precision
            ),
        }
        history.append(epoch_metrics)
        print(json.dumps(epoch_metrics, sort_keys=True))

        if validation_result.average_precision > best_average_precision:
            best_average_precision = validation_result.average_precision
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": asdict(model_config),
                    "training_arguments": {
                        name: str(value) if isinstance(value, Path) else value
                        for name, value in vars(arguments).items()
                    },
                    "best_epoch": best_epoch,
                    "best_validation_average_precision": best_average_precision,
                    "primary_comparison_metric": "validation_average_precision",
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        save_metrics(
            output_path=metrics_path,
            arguments=arguments,
            counts=counts,
            population_positive_fraction=population_positive_fraction,
            sampled_positive_fraction=sampled_positive_fraction,
            logit_correction=logit_correction,
            history=history,
            best_epoch=best_epoch,
            best_average_precision=best_average_precision,
        )

        if epochs_without_improvement >= arguments.early_stopping_patience:
            print(
                "Early stopping: validation AP did not improve for "
                f"{arguments.early_stopping_patience} epochs."
            )
            break

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    best_validation_result = evaluate(
        model,
        validation_loader,
        criterion,
        device,
        logit_correction,
    )
    validation_prevalence = validation_positive_count / len(validation_examples)
    save_learning_curves(
        history,
        best_epoch,
        validation_prevalence,
        learning_curves_path,
    )
    save_validation_summary(best_validation_result, validation_summary_path)

    print(f"Best validation AP: {best_average_precision:.6f} at epoch {best_epoch}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Metrics: {metrics_path}")
    print(f"Model overview: {model_overview_path}")
    print(f"Learning curves: {learning_curves_path}")
    print(f"Validation summary: {validation_summary_path}")
    print("The held-out test split was not used.")


if __name__ == "__main__":
    main()
