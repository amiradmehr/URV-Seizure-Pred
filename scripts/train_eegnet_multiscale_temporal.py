r"""Train the residual multi-scale temporal head on cached EEGNet features.

The selected baseline EEGNet encoder and classifier remain frozen.  The new
head learns causal contrasts between 1-, 5-, 15-, and 45-minute embedding
means.  Its zero initialization makes epoch zero equivalent to the baseline,
and model selection uses average precision on the complete natural-prevalence,
patient-held-out validation split.

Run ``scripts/cache_eegnet_embeddings.py`` once before this script.

PowerShell example:

    .\.venv-win\Scripts\python.exe scripts\train_eegnet_multiscale_temporal.py --device cuda
"""

from __future__ import annotations

import argparse
import hashlib
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
    CachedEmbeddingDecisionDataset,
    load_decision_examples,
)
from seizure_prediction.models import (  # noqa: E402
    EEGNetMultiScaleTemporalConfig,
    EEGNetMultiScaleTemporalRiskModel,
)


@dataclass(frozen=True)
class EvaluationResult:
    """Validation metrics plus arrays used by the final plots."""

    loss: float
    average_precision: float
    labels: np.ndarray
    probabilities: np.ndarray


def parse_arguments() -> argparse.Namespace:
    """Return conservative settings for the first temporal-head experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--temporal-hidden-dim", type=int, default=16)
    parser.add_argument("--temporal-dropout", type=float, default=0.2)
    parser.add_argument(
        "--negative-to-positive-ratio",
        type=float,
        default=10.0,
        help="Different training negatives sampled per positive each epoch.",
    )
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
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
        "--baseline-checkpoint",
        type=Path,
        default=(
            PROJECT_ROOT
            / "outputs"
            / "models"
            / "eegnet_baseline_ratio10_lr1e4"
            / "best_model.pt"
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "embedding_cache"
            / "eegnet_baseline_ratio10_lr1e4"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "outputs"
            / "models"
            / "eegnet_multiscale_temporal_ratio10"
        ),
    )
    return parser.parse_args()


def validate_arguments(arguments: argparse.Namespace) -> None:
    """Reject invalid run settings before the large metadata load."""
    for name in (
        "epochs",
        "batch_size",
        "temporal_hidden_dim",
        "early_stopping_patience",
    ):
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
    if not 0.0 <= arguments.temporal_dropout < 1.0:
        raise ValueError("temporal-dropout must be at least zero and less than one.")


def set_seed(seed: int) -> None:
    """Seed all randomness used by sampling and temporal-head training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested_device: str) -> torch.device:
    """Select CUDA when available unless CPU was explicitly requested."""
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available.")
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested_device)


def sha256_file(path: Path) -> str:
    """Return the encoder-checkpoint fingerprint used by the cache."""
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint_file:
        for block in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_embedding_cache(
    cache_dir: Path,
    checkpoint_path: Path,
    embedding_dim: int,
    sequence_chunks: int,
) -> None:
    """Ensure cached embeddings came from this exact baseline checkpoint."""
    metadata_path = cache_dir / "cache_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Embedding-cache metadata not found: {metadata_path}. "
            "Run scripts/cache_eegnet_embeddings.py first."
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("baseline_checkpoint_sha256") != sha256_file(checkpoint_path):
        raise ValueError(
            "The embedding cache was created with a different baseline checkpoint."
        )
    if int(metadata.get("embedding_dim", -1)) != embedding_dim:
        raise ValueError("The embedding cache has a different embedding dimension.")
    chunk_samples = int(metadata.get("chunk_samples", -1))
    expected_chunk_samples = int(
        round(CONFIG.chunk_window_seconds * CONFIG.target_sfreq)
    )
    if chunk_samples != expected_chunk_samples or sequence_chunks <= 0:
        raise ValueError("The embedding cache uses an incompatible chunk size.")
    cached_splits = set(metadata.get("splits", []))
    if not {"train", "validation"}.issubset(cached_splits):
        raise ValueError("The embedding cache must contain train and validation splits.")


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


def build_loader(
    examples,
    *,
    cache_dir: Path,
    embedding_dim: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    sampler: Sampler[int] | None = None,
) -> DataLoader:
    """Build a loader over compact cached decision histories."""
    dataset = CachedEmbeddingDecisionDataset(
        examples,
        CONFIG,
        cache_root=cache_dir,
        embedding_dim=embedding_dim,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )


def evaluate(
    model: EEGNetMultiScaleTemporalRiskModel,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    logit_correction: float,
) -> EvaluationResult:
    """Evaluate loss and AP on the full patient-held-out validation split."""
    model.eval()
    total_loss = 0.0
    total_examples = 0
    labels: list[float] = []
    probabilities: list[float] = []
    with torch.inference_mode():
        for embeddings, availability, target in loader:
            embeddings = embeddings.to(device, non_blocking=True)
            availability = availability.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            corrected_logits = (
                model.forward_from_chunk_embeddings(embeddings, availability)
                + logit_correction
            )
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


def train_epoch(
    model: EEGNetMultiScaleTemporalRiskModel,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: AdamW,
    device: torch.device,
    max_grad_norm: float,
) -> float:
    """Train only the residual temporal head for one balanced epoch."""
    model.train()
    model.keep_frozen_baseline_in_eval_mode()
    cumulative_loss = 0.0
    examples_seen = 0
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    for embeddings, availability, target in loader:
        embeddings = embeddings.to(device, non_blocking=True)
        availability = availability.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model.forward_from_chunk_embeddings(embeddings, availability)
        loss = criterion(logits, target)
        loss.backward()
        clip_grad_norm_(trainable_parameters, max_grad_norm)
        optimizer.step()
        cumulative_loss += loss.item() * len(target)
        examples_seen += len(target)
    return cumulative_loss / examples_seen


def save_checkpoint(
    *,
    path: Path,
    model: EEGNetMultiScaleTemporalRiskModel,
    arguments: argparse.Namespace,
    epoch: int,
    validation_average_precision: float,
    initial_baseline_average_precision: float,
) -> None:
    """Save a complete deployable model, including its frozen encoder."""
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": asdict(model.config),
            "training_arguments": {
                name: str(value) if isinstance(value, Path) else value
                for name, value in vars(arguments).items()
            },
            "baseline_checkpoint": str(arguments.baseline_checkpoint.resolve()),
            "baseline_parameters_frozen": True,
            "best_epoch": epoch,
            "best_validation_average_precision": validation_average_precision,
            "initial_baseline_validation_average_precision": (
                initial_baseline_average_precision
            ),
            "primary_comparison_metric": "validation_average_precision",
        },
        path,
    )


def save_metrics(
    *,
    path: Path,
    arguments: argparse.Namespace,
    model: EEGNetMultiScaleTemporalRiskModel,
    counts: dict[str, int],
    population_positive_fraction: float,
    sampled_positive_fraction: float,
    logit_correction: float,
    history: list[dict[str, float | int | str | None]],
    best_epoch: int,
    best_average_precision: float,
) -> None:
    """Write a concise, reproducible record after every epoch."""
    path.write_text(
        json.dumps(
            {
                "model": "EEGNetMultiScaleTemporalRiskModel",
                "baseline_checkpoint": str(
                    arguments.baseline_checkpoint.resolve()
                ),
                "embedding_cache": str(arguments.cache_dir.resolve()),
                "baseline_parameters_frozen": True,
                "primary_comparison_metric": "validation_average_precision",
                "counts": counts,
                "negative_to_positive_ratio_per_epoch": (
                    arguments.negative_to_positive_ratio
                ),
                "training_population_positive_fraction": (
                    population_positive_fraction
                ),
                "training_sampled_positive_fraction": sampled_positive_fraction,
                "sampling_prior_logit_correction": logit_correction,
                "model_config": asdict(model.config),
                "parameters": {
                    "total": sum(p.numel() for p in model.parameters()),
                    "trainable": sum(
                        p.numel() for p in model.parameters() if p.requires_grad
                    ),
                },
                "hyperparameters": {
                    "learning_rate": arguments.learning_rate,
                    "weight_decay": arguments.weight_decay,
                    "batch_size": arguments.batch_size,
                    "temporal_hidden_dim": arguments.temporal_hidden_dim,
                    "temporal_dropout": arguments.temporal_dropout,
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


def save_learning_curves(
    history: list[dict[str, float | int | str | None]],
    best_epoch: int,
    prevalence: float,
    output_path: Path,
) -> None:
    """Plot head optimization and validation AP, including epoch zero."""
    trained_rows = [row for row in history if int(row["epoch"]) > 0]
    figure, (loss_axis, ap_axis) = plt.subplots(1, 2, figsize=(12, 5))
    loss_axis.plot(
        [int(row["epoch"]) for row in trained_rows],
        [float(row["train_loss"]) for row in trained_rows],
        marker="o",
    )
    loss_axis.set(
        title="Temporal-head optimization",
        xlabel="Epoch",
        ylabel="Sampled training BCE",
    )
    loss_axis.grid(alpha=0.3)

    epochs = [int(row["epoch"]) for row in history]
    ap_axis.plot(
        epochs,
        [float(row["validation_average_precision"]) for row in history],
        marker="o",
        label="Validation AP",
    )
    ap_axis.axhline(
        prevalence,
        color="black",
        linestyle="--",
        label=f"Random AP ({prevalence:.4f})",
    )
    ap_axis.axvline(best_epoch, color="black", linestyle=":", label="Best epoch")
    ap_axis.set(
        title="Full natural-prevalence validation",
        xlabel="Epoch (0 = unchanged baseline)",
        ylabel="Average precision",
        ylim=(0.0, max(0.05, 1.1 * max(float(row["validation_average_precision"]) for row in history))),
    )
    ap_axis.grid(alpha=0.3)
    ap_axis.legend()
    figure.suptitle("EEGNet multi-scale temporal learning curves")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_validation_summary(
    result: EvaluationResult,
    output_path: Path,
) -> None:
    """Save the best model's precision-recall curve."""
    precision, recall, _ = precision_recall_curve(
        result.labels,
        result.probabilities,
    )
    prevalence = float(np.mean(result.labels))
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.plot(recall, precision, label=f"Temporal model (AP={result.average_precision:.4f})")
    axis.axhline(
        prevalence,
        color="black",
        linestyle="--",
        label=f"Random AP ({prevalence:.4f})",
    )
    axis.set(
        title="Best multi-scale temporal validation result",
        xlabel="Recall (sensitivity)",
        ylabel="Precision",
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.0),
    )
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """Initialize from the best baseline and fit the temporal residual."""
    arguments = parse_arguments()
    validate_arguments(arguments)
    CONFIG.validate()
    set_seed(arguments.seed)
    device = resolve_device(arguments.device)
    if not arguments.baseline_checkpoint.exists():
        raise FileNotFoundError(
            f"Baseline checkpoint not found: {arguments.baseline_checkpoint}"
        )

    baseline_checkpoint = torch.load(
        arguments.baseline_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    baseline_config = baseline_checkpoint.get("model_config")
    baseline_state_dict = baseline_checkpoint.get("model_state_dict")
    if not isinstance(baseline_config, dict) or not isinstance(
        baseline_state_dict,
        dict,
    ):
        raise ValueError("The checkpoint does not contain a BaselineEEGNet model.")

    chunk_seconds = CONFIG.chunk_window_seconds
    chunks_per_minute = int(round(60.0 / chunk_seconds))
    sequence_chunks = int(round(CONFIG.input_window_seconds / chunk_seconds))
    model_config = EEGNetMultiScaleTemporalConfig(
        n_chans=int(baseline_config["n_chans"]),
        chunk_samples=int(baseline_config["chunk_samples"]),
        sequence_chunks=sequence_chunks,
        chunks_per_minute=chunks_per_minute,
        embedding_dim=int(baseline_config["embedding_dim"]),
        temporal_hidden_dim=arguments.temporal_hidden_dim,
        temporal_windows_minutes=(1, 5, 15, 45),
        encoder_chunk_batch_size=int(
            baseline_config["encoder_chunk_batch_size"]
        ),
        encoder_dropout=float(baseline_config["dropout"]),
        temporal_dropout=arguments.temporal_dropout,
        sampling_prior_logit_correction=0.0,
    )
    for name in ("n_chans", "chunk_samples", "sequence_chunks"):
        if int(baseline_config[name]) != getattr(model_config, name):
            raise ValueError(f"Baseline checkpoint has incompatible {name}.")

    validate_embedding_cache(
        arguments.cache_dir,
        arguments.baseline_checkpoint,
        model_config.embedding_dim,
        model_config.sequence_chunks,
    )
    manifest_path = CONFIG.manifests_dir / "processed_shard_manifest.csv"
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
        cache_dir=arguments.cache_dir,
        embedding_dim=model_config.embedding_dim,
        batch_size=arguments.batch_size,
        num_workers=arguments.num_workers,
        device=device,
        sampler=train_sampler,
    )
    validation_loader = build_loader(
        validation_examples,
        cache_dir=arguments.cache_dir,
        embedding_dim=model_config.embedding_dim,
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
    model_config = EEGNetMultiScaleTemporalConfig(
        **{
            **asdict(model_config),
            "sampling_prior_logit_correction": logit_correction,
        }
    )
    model = EEGNetMultiScaleTemporalRiskModel(model_config)
    model.initialize_from_baseline_state_dict(baseline_state_dict)
    model.set_baseline_trainable(False)
    model.to(device)

    optimizer = AdamW(
        model.temporal_head.parameters(),
        lr=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
    )
    criterion = nn.BCEWithLogitsLoss()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = arguments.output_dir / "best_model.pt"
    metrics_path = arguments.output_dir / "metrics.json"
    curves_path = arguments.output_dir / "learning_curves.png"
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
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(json.dumps(counts, indent=2))
    print(
        "Parameters: "
        f"{sum(p.numel() for p in model.parameters()):,} total; "
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable."
    )
    print("Frozen baseline: EEGNet encoder plus original mean-pool classifier.")
    print("Trainable branch: adjacent 1/5/15/45-minute temporal contrasts.")

    initial_result = evaluate(
        model,
        validation_loader,
        criterion,
        device,
        logit_correction,
    )
    history: list[dict[str, float | int | str | None]] = [
        {
            "epoch": 0,
            "stage": "unchanged_baseline",
            "train_loss": None,
            "validation_loss": initial_result.loss,
            "validation_average_precision": initial_result.average_precision,
        }
    ]
    best_average_precision = initial_result.average_precision
    best_epoch = 0
    epochs_without_improvement = 0
    save_checkpoint(
        path=checkpoint_path,
        model=model,
        arguments=arguments,
        epoch=0,
        validation_average_precision=best_average_precision,
        initial_baseline_average_precision=initial_result.average_precision,
    )
    print(
        f"epoch=00 stage=unchanged_baseline "
        f"validation_loss={initial_result.loss:.6f} "
        f"validation_ap={initial_result.average_precision:.6f}"
    )

    for epoch in range(1, arguments.epochs + 1):
        train_sampler.set_epoch(epoch - 1)
        train_loss = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            arguments.max_grad_norm,
        )
        validation_result = evaluate(
            model,
            validation_loader,
            criterion,
            device,
            logit_correction,
        )
        row: dict[str, float | int | str | None] = {
            "epoch": epoch,
            "stage": "temporal_head",
            "train_loss": train_loss,
            "validation_loss": validation_result.loss,
            "validation_average_precision": validation_result.average_precision,
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True))

        if validation_result.average_precision > best_average_precision:
            best_average_precision = validation_result.average_precision
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                path=checkpoint_path,
                model=model,
                arguments=arguments,
                epoch=epoch,
                validation_average_precision=best_average_precision,
                initial_baseline_average_precision=(
                    initial_result.average_precision
                ),
            )
        else:
            epochs_without_improvement += 1

        save_metrics(
            path=metrics_path,
            arguments=arguments,
            model=model,
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

    best_checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(best_checkpoint["model_state_dict"], strict=True)
    best_result = evaluate(
        model,
        validation_loader,
        criterion,
        device,
        logit_correction,
    )
    save_metrics(
        path=metrics_path,
        arguments=arguments,
        model=model,
        counts=counts,
        population_positive_fraction=population_positive_fraction,
        sampled_positive_fraction=sampled_positive_fraction,
        logit_correction=logit_correction,
        history=history,
        best_epoch=best_epoch,
        best_average_precision=best_average_precision,
    )
    validation_prevalence = validation_positive_count / len(validation_examples)
    save_learning_curves(history, best_epoch, validation_prevalence, curves_path)
    save_validation_summary(best_result, validation_summary_path)

    improvement = best_average_precision - initial_result.average_precision
    print(
        f"Best validation AP: {best_average_precision:.6f} at epoch {best_epoch} "
        f"(change from unchanged baseline: {improvement:+.6f})"
    )
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Metrics: {metrics_path}")
    print(f"Learning curves: {curves_path}")
    print(f"Validation summary: {validation_summary_path}")
    print("The held-out test split was not used.")


if __name__ == "__main__":
    main()
