r"""Train a mean-pooling EEGNet under the temporal TCN comparison protocol.

This run deliberately matches the temporal TCN's data sampling, optimization,
validation, prior correction, metrics, and checkpoint selection. The only
substantive model difference is that the ordered chunk embeddings are averaged
instead of being processed by the temporal TCN.

PowerShell example:

    python .\scripts\train_eegnet_mean_pool_fair.py --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIRECTORY = PROJECT_ROOT / "src"

if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))


from seizure_prediction.config import (  # noqa: E402
    add_label_definition_arguments,
    resolve_label_definition,
)
from seizure_prediction.datasets import (  # noqa: E402
    BalancedEpochSampler,
    load_decision_examples,
)
from seizure_prediction.models_old import (  # noqa: E402
    EEGNetMeanPoolConfig,
    EEGNetMeanPoolRiskModel,
)
from train_eegnet_temporal_tcn import (  # noqa: E402
    build_loader,
    evaluate,
    resolve_device,
    sampling_prior_logit_correction,
    set_seed,
)


def parse_arguments() -> argparse.Namespace:
    """Parse settings matched to the fair temporal TCN run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--embedding-dim", type=int, default=32)
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
        "--dropout",
        type=float,
        default=0.4,
        help="EEGNet dropout probability, matched to the temporal TCN run.",
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
            / "eegnet_mean_pool_ratio10_prior_corrected"
        ),
    )
    add_label_definition_arguments(parser)
    return parser.parse_args()


def save_training_curves(
    history: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Save the same comparison curves used for the temporal TCN."""
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

    figure.suptitle("Fair EEGNet mean-pooling learning curves")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def validate_arguments(arguments: argparse.Namespace) -> None:
    """Reject invalid training settings before loading the dataset."""
    if arguments.epochs <= 0 or arguments.batch_size <= 0:
        raise ValueError("epochs and batch-size must be positive.")
    if arguments.num_workers < 0:
        raise ValueError("num-workers cannot be negative.")
    if arguments.embedding_dim <= 0:
        raise ValueError("embedding-dim must be positive.")
    if arguments.negative_to_positive_ratio <= 0:
        raise ValueError("negative-to-positive-ratio must be positive.")
    if arguments.gradient_accumulation_steps <= 0:
        raise ValueError("gradient-accumulation-steps must be positive.")
    if arguments.max_grad_norm <= 0:
        raise ValueError("max-grad-norm must be positive.")
    if not 0.0 <= arguments.dropout < 1.0:
        raise ValueError("dropout must be at least 0 and less than 1.")


def main() -> None:
    """Train and checkpoint the fair EEGNet mean-pooling comparison."""
    arguments = parse_arguments()
    config = resolve_label_definition(arguments)
    config.validate()
    validate_arguments(arguments)
    set_seed(arguments.seed)
    device = resolve_device(arguments.device)

    manifest_path = config.manifests_dir / "processed_shard_manifest.csv"
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
        project_root=config.project_root,
    )
    validation_examples = load_decision_examples(
        manifest_path,
        split="validation",
        negative_to_positive_ratio=None,
        seed=arguments.seed,
        project_root=config.project_root,
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

    train_positive_count = int((all_train_examples["label"] == 1).sum())
    train_negative_count = int((all_train_examples["label"] == 0).sum())
    validation_positive_count = int((validation_examples["label"] == 1).sum())
    validation_negative_count = int((validation_examples["label"] == 0).sum())
    population_positive_fraction = train_positive_count / len(all_train_examples)
    sampled_positive_fraction = train_positive_count / len(train_sampler)
    logit_correction = sampling_prior_logit_correction(
        population_positive_fraction,
        sampled_positive_fraction,
    )

    model_config = EEGNetMeanPoolConfig(
        n_chans=len(config.canonical_channel_names),
        chunk_samples=int(config.chunk_window_seconds * config.target_sfreq),
        embedding_dim=arguments.embedding_dim,
        encoder_chunk_batch_size=arguments.encoder_chunk_batch_size,
        dropout=arguments.dropout,
        sampling_prior_logit_correction=logit_correction,
    )
    model = EEGNetMeanPoolRiskModel(model_config).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
    )
    criterion = nn.BCEWithLogitsLoss()

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = arguments.output_dir / "best_model.pt"
    metrics_path = arguments.output_dir / "metrics.json"
    curves_path = arguments.output_dir / "learning_curves.png"
    history: list[dict[str, Any]] = []
    best_average_precision = float("-inf")
    best_epoch = 0

    print(f"Device: {device}")
    print(
        "Train decisions per epoch: "
        f"{len(train_sampler)} ({train_positive_count} positive, "
        f"{train_sampler.negative_count} newly selected negative)"
    )
    print(
        "Full validation decisions: "
        f"{len(validation_examples)} ({validation_positive_count} positive, "
        f"{validation_negative_count} negative)"
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
        f"max gradient norm={arguments.max_grad_norm}."
    )
    print(
        "Trainable parameters: "
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    for epoch in range(1, arguments.epochs + 1):
        train_sampler.set_epoch(epoch - 1)
        model.train()
        cumulative_loss = 0.0
        cumulative_correct = 0
        examples_seen = 0
        accumulated_batches = 0
        optimizer_steps = 0
        gradient_norms: list[float] = []
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
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": asdict(model_config),
                    "config": {
                        "target_sfreq": config.target_sfreq,
                        "input_window_seconds": config.input_window_seconds,
                        "chunk_window_seconds": config.chunk_window_seconds,
                        "canonical_channel_names": list(
                            config.canonical_channel_names
                        ),
                        "seizure_occurrence_period_minutes": (
                            config.seizure_occurrence_period_minutes
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
                    "best_epoch": best_epoch,
                },
                checkpoint_path,
            )

        metrics_path.write_text(
            json.dumps(
                {
                    "model": "EEGNetMeanPoolRiskModel",
                    "comparison_protocol": (
                        "matched to eegnet_temporal_tcn_ratio10_prior_corrected"
                    ),
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
                    "best_epoch": best_epoch,
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
