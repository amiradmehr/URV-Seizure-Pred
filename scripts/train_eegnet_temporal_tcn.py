r"""Train EEGNet plus a causal temporal TCN for seizure-risk prediction.

EEGNet converts each five-second EEG chunk into a local embedding. A causal,
dilated temporal convolutional network then processes all 540 embeddings in
chronological order before predicting seizure onset in the next 10 minutes.

Training uses every positive decision and a configurable, non-repeating slice
of negatives in each epoch. Validation always uses the full natural split.

Example:

    .venv/bin/python scripts/train_eegnet_temporal_tcn.py --device cuda
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
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
from seizure_prediction.models_old import (  # noqa: E402
    EEGNetTemporalTCNConfig,
    EEGNetTemporalTCNRiskModel,
)


def parse_arguments() -> argparse.Namespace:
    """Parse training settings, matching the latest EEGNet baseline run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--tcn-channels", type=int, default=16)
    parser.add_argument("--encoder-chunk-batch-size", type=int, default=128)
    parser.add_argument(
        "--negative-to-positive-ratio",
        type=float,
        default=10.0,
        help="Number of different negative decisions sampled per positive each epoch.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=8,
        help="Accumulate this many decision batches before each optimizer step.",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Clip the global gradient norm to this value before optimizer steps.",
    )
    parser.add_argument(
        "--temporal-norm-groups",
        type=int,
        default=4,
        help="GroupNorm groups in the TCN; independent of the decision batch size.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.4,
        help="Dropout probability used by both EEGNet and the temporal TCN.",
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
        default=(
            PROJECT_ROOT
            / "outputs"
            / "models"
            / "eegnet_temporal_tcn_ratio10_prior_corrected"
        ),
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Set reproducible random seeds."""
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
    batch_size: int,
    num_workers: int,
    device: torch.device,
    sampler: Sampler[int] | None = None,
) -> DataLoader:
    """Build a lazy loader over continuous recordings."""
    dataset = StreamingDecisionDataset(examples, CONFIG)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
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
    predicted_positive = scores >= 0.5
    positive = targets == 1
    negative = ~positive
    true_positive = int(np.sum(predicted_positive & positive))
    false_positive = int(np.sum(predicted_positive & negative))
    true_negative = int(np.sum(~predicted_positive & negative))
    false_negative = int(np.sum(~predicted_positive & positive))
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    specificity_denominator = true_negative + false_positive
    precision = (
        true_positive / precision_denominator if precision_denominator else 0.0
    )
    recall = true_positive / recall_denominator
    specificity = true_negative / specificity_denominator
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    prevalence = float(np.mean(positive))
    average_precision = float(average_precision_score(targets, scores))
    return {
        "accuracy": float(np.mean(predicted_positive == positive)),
        "precision_at_0_5": float(precision),
        "recall_at_0_5": float(recall),
        "specificity_at_0_5": float(specificity),
        "f1_at_0_5": float(f1),
        "predicted_positive_rate_at_0_5": float(np.mean(predicted_positive)),
        "prevalence": prevalence,
        "average_precision_baseline": prevalence,
        "average_precision": average_precision,
        "average_precision_lift": average_precision / prevalence,
        "roc_auc": float(roc_auc_score(targets, scores)),
        "brier_score": float(brier_score_loss(targets, scores)),
    }


def sampling_prior_logit_correction(
    population_positive_fraction: float,
    sampled_positive_fraction: float,
) -> float:
    """Return the intercept shift from sampled to population class odds.

    Negative subsampling changes the positive class prior seen during training.
    Under case-control sampling, adding this constant to a raw model logit
    restores the original population prior without changing ranking metrics.
    """
    for name, fraction in (
        ("population_positive_fraction", population_positive_fraction),
        ("sampled_positive_fraction", sampled_positive_fraction),
    ):
        if not 0.0 < fraction < 1.0:
            raise ValueError(f"{name} must be strictly between zero and one.")

    population_log_odds = math.log(
        population_positive_fraction / (1.0 - population_positive_fraction)
    )
    sampled_log_odds = math.log(
        sampled_positive_fraction / (1.0 - sampled_positive_fraction)
    )
    return population_log_odds - sampled_log_odds


def save_training_curves(
    history: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Save loss and accuracy learning curves for a completed run."""
    if not history:
        raise ValueError("Cannot plot an empty training history.")

    epochs = [metrics["epoch"] for metrics in history]
    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    loss_axis, ap_axis, auc_axis, calibration_axis = axes.flat

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

    ap_axis.plot(
        epochs,
        [metrics["validation_average_precision"] for metrics in history],
        marker="o",
        label="Validation AP",
    )
    ap_axis.axhline(
        history[0]["validation_average_precision_baseline"],
        color="black",
        linestyle="--",
        label="Random AP (prevalence)",
    )
    ap_axis.set(
        title="Average precision on natural prevalence",
        xlabel="Epoch",
        ylabel="Average precision",
    )
    ap_axis.grid(alpha=0.3)
    ap_axis.legend()

    auc_axis.plot(
        epochs,
        [metrics["validation_roc_auc"] for metrics in history],
        marker="o",
        label="Validation ROC-AUC",
    )
    auc_axis.axhline(0.5, color="black", linestyle="--", label="Random ROC-AUC")
    auc_axis.set(title="Ranking quality", xlabel="Epoch", ylabel="ROC-AUC")
    auc_axis.grid(alpha=0.3)
    auc_axis.legend()

    calibration_axis.plot(
        epochs,
        [metrics["validation_brier_score"] for metrics in history],
        marker="o",
        label="Prior-corrected Brier score",
    )
    calibration_axis.plot(
        epochs,
        [metrics["validation_uncorrected_brier_score"] for metrics in history],
        marker="o",
        label="Uncorrected Brier score",
    )
    calibration_axis.set(
        title="Probability calibration",
        xlabel="Epoch",
        ylabel="Brier score (lower is better)",
    )
    calibration_axis.grid(alpha=0.3)
    calibration_axis.legend()

    figure.suptitle("EEGNet plus temporal TCN learning curves")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    logit_correction: float,
) -> dict[str, float]:
    """Evaluate raw and population-prior-corrected validation probabilities."""
    model.eval()
    corrected_total_loss = 0.0
    uncorrected_total_loss = 0.0
    total_examples = 0
    labels: list[float] = []
    corrected_probabilities: list[float] = []
    uncorrected_probabilities: list[float] = []

    with torch.no_grad():
        for signal, availability, target in loader:
            signal = signal.to(device, non_blocking=True)
            availability = availability.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            logits = model(signal, availability)
            corrected_logits = logits + logit_correction
            uncorrected_loss = criterion(logits, target)
            corrected_loss = criterion(corrected_logits, target)

            corrected_total_loss += corrected_loss.item() * len(target)
            uncorrected_total_loss += uncorrected_loss.item() * len(target)
            total_examples += len(target)
            labels.extend(target.cpu().tolist())
            corrected_probabilities.extend(
                torch.sigmoid(corrected_logits).cpu().tolist()
            )
            uncorrected_probabilities.extend(torch.sigmoid(logits).cpu().tolist())

    corrected_metrics = binary_metrics(labels, corrected_probabilities)
    uncorrected_metrics = binary_metrics(labels, uncorrected_probabilities)

    return {
        "loss": corrected_total_loss / total_examples,
        **corrected_metrics,
        "uncorrected_loss": uncorrected_total_loss / total_examples,
        **{
            f"uncorrected_{name}": value
            for name, value in uncorrected_metrics.items()
        },
    }


def main() -> None:
    """Train and checkpoint the EEGNet plus temporal TCN model."""
    arguments = parse_arguments()
    CONFIG.validate()
    if arguments.epochs <= 0 or arguments.batch_size <= 0:
        raise ValueError("epochs and batch-size must be positive.")
    if arguments.num_workers < 0:
        raise ValueError("num-workers cannot be negative.")
    if arguments.embedding_dim <= 0 or arguments.tcn_channels <= 0:
        raise ValueError("embedding-dim and tcn-channels must be positive.")
    if arguments.negative_to_positive_ratio <= 0:
        raise ValueError("negative-to-positive-ratio must be positive.")
    if arguments.gradient_accumulation_steps <= 0:
        raise ValueError("gradient-accumulation-steps must be positive.")
    if arguments.max_grad_norm <= 0:
        raise ValueError("max-grad-norm must be positive.")
    if arguments.temporal_norm_groups <= 0:
        raise ValueError("temporal-norm-groups must be positive.")
    if arguments.tcn_channels % arguments.temporal_norm_groups != 0:
        raise ValueError(
            "tcn-channels must be divisible by temporal-norm-groups."
        )
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

    all_train_examples = load_decision_examples(
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
        all_train_examples["label"],
        negative_to_positive_ratio=arguments.negative_to_positive_ratio,
        seed=arguments.seed,
    )
    train_loader = build_loader(
        all_train_examples,
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

    sequence_chunks = int(
        round(CONFIG.input_window_seconds / CONFIG.chunk_window_seconds)
    )
    train_positive_count = int((all_train_examples["label"] == 1).sum())
    train_negative_count = int((all_train_examples["label"] == 0).sum())
    population_positive_fraction = train_positive_count / len(all_train_examples)
    sampled_positive_fraction = train_positive_count / len(train_sampler)
    logit_correction = sampling_prior_logit_correction(
        population_positive_fraction,
        sampled_positive_fraction,
    )

    model_config = EEGNetTemporalTCNConfig(
        n_chans=len(CONFIG.canonical_channel_names),
        chunk_samples=int(CONFIG.chunk_window_seconds * CONFIG.target_sfreq),
        sequence_chunks=sequence_chunks,
        embedding_dim=arguments.embedding_dim,
        tcn_channels=arguments.tcn_channels,
        encoder_chunk_batch_size=arguments.encoder_chunk_batch_size,
        dropout=arguments.dropout,
        temporal_norm_groups=arguments.temporal_norm_groups,
        sampling_prior_logit_correction=logit_correction,
    )
    model = EEGNetTemporalTCNRiskModel(model_config).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
    )

    # One end-to-end loss trains the classifier, TCN, and EEGNet encoder.
    # Negative subsampling is handled by the inference-time prior correction.
    # The sampled training objective itself remains standard unweighted BCE.
    criterion = nn.BCEWithLogitsLoss()

    validation_positive_count = int((validation_examples["label"] == 1).sum())
    validation_negative_count = int((validation_examples["label"] == 0).sum())

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = arguments.output_dir / "best_model.pt"
    metrics_path = arguments.output_dir / "metrics.json"
    curves_path = arguments.output_dir / "learning_curves.png"
    history: list[dict[str, Any]] = []
    best_average_precision = float("-inf")

    print(f"Device: {device}")
    print(
        "Train decisions per epoch: "
        f"{len(train_sampler)} "
        f"({train_positive_count} positive, "
        f"{train_sampler.negative_count} newly selected negative)"
    )
    print(
        "Full validation decisions: "
        f"{len(validation_examples)} "
        f"({validation_positive_count} positive, "
        f"{validation_negative_count} negative)"
    )
    print(
        "Temporal context: "
        f"{sequence_chunks} chunks; TCN receptive field: "
        f"{model_config.temporal_receptive_field_chunks} chunks."
    )
    print(
        "Trainable parameters: "
        f"{sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad):,}"
    )
    print(
        "Sampling-prior correction: "
        f"population prevalence={population_positive_fraction:.6f}, "
        f"sampled prevalence={sampled_positive_fraction:.6f}, "
        f"logit shift={logit_correction:.6f}."
    )
    print(
        "Optimization: "
        f"batch size={arguments.batch_size}, "
        f"gradient accumulation={arguments.gradient_accumulation_steps}, "
        f"effective decision batch="
        f"{arguments.batch_size * arguments.gradient_accumulation_steps}, "
        f"max gradient norm={arguments.max_grad_norm}, "
        f"TCN normalization=GroupNorm({arguments.temporal_norm_groups} groups)."
    )
    print("Loss: unweighted BCEWithLogitsLoss, trained end-to-end.")
    print("The complete validation split will make each validation pass substantially longer.")

    for epoch in range(1, arguments.epochs + 1):
        train_sampler.set_epoch(epoch - 1)
        model.train()
        cumulative_loss = 0.0
        cumulative_correct = 0
        examples_seen = 0
        accumulated_batches = 0
        gradient_norms: list[float] = []
        optimizer_steps = 0
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
                gradient_norm = clip_grad_norm_(
                    model.parameters(),
                    arguments.max_grad_norm,
                )
                gradient_norms.append(float(gradient_norm.item()))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                accumulated_batches = 0
                optimizer_steps += 1

            cumulative_loss += loss.item() * len(target)
            cumulative_correct += int(
                ((logits >= 0.0) == (target >= 0.5)).sum().item()
            )
            examples_seen += len(target)

        validation_metrics = evaluate(
            model,
            validation_loader,
            criterion,
            device,
            logit_correction,
        )
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": cumulative_loss / examples_seen,
            "train_accuracy": cumulative_correct / examples_seen,
            "train_positive_examples": train_positive_count,
            "train_negative_examples": train_sampler.negative_count,
            "optimizer_steps": optimizer_steps,
            "mean_gradient_norm_before_clipping": float(np.mean(gradient_norms)),
            "max_gradient_norm_before_clipping": float(np.max(gradient_norms)),
            **{
                f"validation_{name}": value
                for name, value in validation_metrics.items()
            },
        }
        history.append(epoch_metrics)
        print(json.dumps(epoch_metrics, sort_keys=True))

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
                        "canonical_channel_names": list(
                            CONFIG.canonical_channel_names
                        ),
                        "seizure_occurrence_period_minutes": (
                            CONFIG.seizure_occurrence_period_minutes
                        ),
                    },
                    "training_arguments": vars(arguments),
                    "loss": "BCEWithLogitsLoss(pos_weight=1.0)",
                    "training_negative_sampling": (
                        "all positives plus a fresh negative slice per epoch"
                    ),
                    "training_population_positive_fraction": (
                        population_positive_fraction
                    ),
                    "training_sampled_positive_fraction": sampled_positive_fraction,
                    "sampling_prior_logit_correction": logit_correction,
                    "inference_probability": (
                        "sigmoid(raw_logit + sampling_prior_logit_correction)"
                    ),
                    "validation_sampling": "full natural validation split",
                    "best_validation_average_precision": best_average_precision,
                },
                checkpoint_path,
            )

        metrics_path.write_text(
            json.dumps(
                {
                    "all_train_examples": len(all_train_examples),
                    "all_train_positive_examples": train_positive_count,
                    "all_train_negative_examples": train_negative_count,
                    "train_examples_per_epoch": len(train_sampler),
                    "train_negative_to_positive_ratio": (
                        arguments.negative_to_positive_ratio
                    ),
                    "training_population_positive_fraction": (
                        population_positive_fraction
                    ),
                    "training_sampled_positive_fraction": sampled_positive_fraction,
                    "training_negatives_resampled_each_epoch": True,
                    "validation_examples": len(validation_examples),
                    "validation_positive_examples": validation_positive_count,
                    "validation_negative_examples": validation_negative_count,
                    "validation_positive_fraction": (
                        validation_positive_count / len(validation_examples)
                    ),
                    "validation_negative_to_positive_ratio": None,
                    "validation_uses_full_split": True,
                    "loss": "BCEWithLogitsLoss(pos_weight=1.0)",
                    "sampling_prior_logit_correction": logit_correction,
                    "inference_probability": (
                        "sigmoid(raw_logit + sampling_prior_logit_correction)"
                    ),
                    "temporal_normalization": (
                        f"GroupNorm(groups={arguments.temporal_norm_groups})"
                    ),
                    "gradient_accumulation_steps": (
                        arguments.gradient_accumulation_steps
                    ),
                    "effective_decision_batch_size": (
                        arguments.batch_size
                        * arguments.gradient_accumulation_steps
                    ),
                    "max_gradient_norm": arguments.max_grad_norm,
                    "history": history,
                    "best_validation_average_precision": best_average_precision,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    save_training_curves(history, curves_path)
    print(f"Best checkpoint: {checkpoint_path}")
    print(f"Training metrics: {metrics_path}")
    print(f"Learning curves: {curves_path}")
    print("The held-out test split was not used.")


if __name__ == "__main__":
    main()
