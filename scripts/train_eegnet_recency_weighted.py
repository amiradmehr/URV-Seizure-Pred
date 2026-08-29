r"""Train constrained minute-recency weights on the selected EEGNet baseline.

The EEGNet encoder and the original classifier remain frozen. The only
trainable values are one pooling logit for each minute of the configured
input window.
Softmax keeps the weights nonnegative and normalized, while a fixed uniform
mixture and KL penalty keep the result close to the working mean-pool baseline.

Run ``scripts/cache_eegnet_embeddings.py`` once before this script.

PowerShell example:

    .\.venv-old\Scripts\python.exe scripts\train_eegnet_recency_weighted.py --device cuda
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


from seizure_prediction.config import (  # noqa: E402
    CONFIG,
    add_label_definition_arguments,
    resolve_label_definition,
)
from seizure_prediction.datasets import (  # noqa: E402
    BalancedEpochSampler,
    CachedEmbeddingDecisionDataset,
    load_decision_examples,
)
from seizure_prediction.models import (  # noqa: E402
    EEGNetRecencyWeightedConfig,
    EEGNetRecencyWeightedRiskModel,
)


@dataclass(frozen=True)
class EvaluationResult:
    """Validation metrics plus arrays used by the final plots."""

    loss: float
    average_precision: float
    labels: np.ndarray
    probabilities: np.ndarray


@dataclass(frozen=True)
class TrainResult:
    """Mean sampled losses for one training epoch."""

    objective: float
    binary_cross_entropy: float
    uniform_weight_kl: float


def parse_arguments() -> argparse.Namespace:
    """Return conservative settings for the constrained temporal experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=5e-2)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument(
        "--uniform-kl-weight",
        type=float,
        default=0.1,
        help="Penalty for departing from uniform mean pooling.",
    )
    parser.add_argument(
        "--max-temporal-strength",
        type=float,
        default=0.25,
        help=(
            "Maximum fraction assigned to learned rather than uniform "
            "minute weights."
        ),
    )
    parser.add_argument("--softmax-temperature", type=float, default=1.0)
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
        default=8,
        help="Stop after this many epochs without better validation AP.",
    )
    parser.add_argument(
        "--minimum-ap-improvement",
        type=float,
        default=1e-6,
        help="Minimum validation-AP gain required to replace the checkpoint.",
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
            CONFIG.embedding_cache_dir
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
            / "eegnet_recency_weighted_ratio10"
        ),
    )
    add_label_definition_arguments(parser)
    return parser.parse_args()


def validate_arguments(arguments: argparse.Namespace) -> None:
    """Reject invalid run settings before loading the large manifest."""
    for name in (
        "epochs",
        "batch_size",
        "early_stopping_patience",
    ):
        if getattr(arguments, name) <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be positive.")
    if arguments.num_workers < 0:
        raise ValueError("num-workers cannot be negative.")
    if arguments.learning_rate <= 0 or arguments.weight_decay < 0:
        raise ValueError("learning-rate must be positive and weight-decay nonnegative.")
    if arguments.uniform_kl_weight < 0:
        raise ValueError("uniform-kl-weight cannot be negative.")
    if not 0.0 < arguments.max_temporal_strength <= 1.0:
        raise ValueError("max-temporal-strength must be in (0, 1].")
    if arguments.softmax_temperature <= 0:
        raise ValueError("softmax-temperature must be positive.")
    if arguments.negative_to_positive_ratio <= 0:
        raise ValueError("negative-to-positive-ratio must be positive.")
    if arguments.max_grad_norm <= 0:
        raise ValueError("max-grad-norm must be positive.")
    if arguments.minimum_ap_improvement < 0:
        raise ValueError("minimum-ap-improvement cannot be negative.")


def set_seed(seed: int) -> None:
    """Seed sampling and optimization for reproducible comparison."""
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
    """Return the baseline-checkpoint fingerprint recorded in the cache."""
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint_file:
        for block in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_embedding_cache(
    cache_dir: Path,
    checkpoint_path: Path,
    embedding_dim: int,
    config,
) -> None:
    """Ensure cached embeddings came from this exact baseline encoder."""
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
    expected_chunk_samples = int(
        round(config.chunk_window_seconds * config.target_sfreq)
    )
    if int(metadata.get("chunk_samples", -1)) != expected_chunk_samples:
        raise ValueError("The embedding cache uses an incompatible chunk size.")
    if not {"train", "validation"}.issubset(set(metadata.get("splits", []))):
        raise ValueError("The cache must contain train and validation splits.")


def sampling_prior_logit_correction(
    population_positive_fraction: float,
    sampled_positive_fraction: float,
) -> float:
    """Correct probabilities after training negatives are undersampled."""
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
    config,
    cache_dir: Path,
    embedding_dim: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    sampler: Sampler[int] | None = None,
) -> DataLoader:
    """Build a loader over cached embedding histories for the configured window."""
    dataset = CachedEmbeddingDecisionDataset(
        examples,
        config,
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
    model: EEGNetRecencyWeightedRiskModel,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    logit_correction: float,
) -> EvaluationResult:
    """Evaluate on every natural-prevalence validation decision."""
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
    model: EEGNetRecencyWeightedRiskModel,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: AdamW,
    device: torch.device,
    max_grad_norm: float,
    uniform_kl_weight: float,
) -> TrainResult:
    """Optimize only the 45 recency logits for one sampled epoch."""
    model.train()
    model.keep_frozen_baseline_in_eval_mode()
    objective_sum = 0.0
    bce_sum = 0.0
    examples_seen = 0
    for embeddings, availability, target in loader:
        embeddings = embeddings.to(device, non_blocking=True)
        availability = availability.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model.forward_from_chunk_embeddings(embeddings, availability)
        binary_cross_entropy = criterion(logits, target)
        uniform_weight_kl = model.uniform_weight_kl()
        objective = (
            binary_cross_entropy + uniform_kl_weight * uniform_weight_kl
        )
        objective.backward()
        clip_grad_norm_([model.recency_logits], max_grad_norm)
        optimizer.step()
        batch_size = len(target)
        objective_sum += objective.item() * batch_size
        bce_sum += binary_cross_entropy.item() * batch_size
        examples_seen += batch_size
    return TrainResult(
        objective=objective_sum / examples_seen,
        binary_cross_entropy=bce_sum / examples_seen,
        uniform_weight_kl=float(model.uniform_weight_kl().detach().cpu()),
    )


def recency_summary(
    model: EEGNetRecencyWeightedRiskModel,
) -> dict[str, float]:
    """Return interpretable summaries of chronological minute weights."""
    weights = model.effective_recency_weights().detach().cpu().numpy()
    minutes_before_decision = np.arange(
        model.config.history_minutes,
        0,
        -1,
        dtype=np.float64,
    )
    return {
        "oldest_minute_weight": float(weights[0]),
        "newest_minute_weight": float(weights[-1]),
        "minimum_minute_weight": float(weights.min()),
        "maximum_minute_weight": float(weights.max()),
        "weight_center_minutes_before_decision": float(
            np.sum(weights * minutes_before_decision)
        ),
        "uniform_weight_kl": float(model.uniform_weight_kl().detach().cpu()),
    }


def save_checkpoint(
    *,
    path: Path,
    model: EEGNetRecencyWeightedRiskModel,
    arguments: argparse.Namespace,
    epoch: int,
    validation_average_precision: float,
    initial_baseline_average_precision: float,
) -> None:
    """Save the complete raw-signal model for later deployment/export."""
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
            "trainable_parameter_names": ["recency_logits"],
            "best_epoch": epoch,
            "best_validation_average_precision": validation_average_precision,
            "initial_baseline_validation_average_precision": (
                initial_baseline_average_precision
            ),
            "primary_comparison_metric": "validation_average_precision",
            "effective_recency_weights_oldest_to_newest": (
                model.effective_recency_weights().detach().cpu()
            ),
        },
        path,
    )


def save_metrics(
    *,
    path: Path,
    arguments: argparse.Namespace,
    model: EEGNetRecencyWeightedRiskModel,
    counts: dict[str, int],
    population_positive_fraction: float,
    sampled_positive_fraction: float,
    logit_correction: float,
    history: list[dict[str, float | int | str | None]],
    best_epoch: int,
    best_average_precision: float,
) -> None:
    """Write a reproducible record after every epoch."""
    path.write_text(
        json.dumps(
            {
                "model": "EEGNetRecencyWeightedRiskModel",
                "baseline_checkpoint": str(
                    arguments.baseline_checkpoint.resolve()
                ),
                "embedding_cache": str(arguments.cache_dir.resolve()),
                "baseline_parameters_frozen": True,
                "trainable_parameter_names": ["recency_logits"],
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
                    "uniform_kl_weight": arguments.uniform_kl_weight,
                    "batch_size": arguments.batch_size,
                    "seed": arguments.seed,
                },
                "recency_weight_summary": recency_summary(model),
                "effective_recency_weights_oldest_to_newest": (
                    model.effective_recency_weights().detach().cpu().tolist()
                ),
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
    """Plot the regularized objective and full-validation AP."""
    trained_rows = [row for row in history if int(row["epoch"]) > 0]
    figure, (loss_axis, ap_axis) = plt.subplots(1, 2, figsize=(12, 5))
    trained_epochs = [int(row["epoch"]) for row in trained_rows]
    loss_axis.plot(
        trained_epochs,
        [float(row["train_bce"]) for row in trained_rows],
        marker="o",
        label="BCE",
    )
    loss_axis.plot(
        trained_epochs,
        [float(row["train_objective"]) for row in trained_rows],
        linestyle="--",
        label="BCE + uniform penalty",
    )
    loss_axis.set(
        title="Constrained recency optimization",
        xlabel="Epoch",
        ylabel="Sampled training loss",
    )
    loss_axis.grid(alpha=0.3)
    loss_axis.legend()

    epochs = [int(row["epoch"]) for row in history]
    ap_values = [float(row["validation_average_precision"]) for row in history]
    ap_axis.plot(epochs, ap_values, marker="o", label="Validation AP")
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
        ylim=(0.0, max(0.05, 1.1 * max(ap_values))),
    )
    ap_axis.grid(alpha=0.3)
    ap_axis.legend()
    figure.suptitle("EEGNet constrained recency-weighted learning curves")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_recency_weights(
    model: EEGNetRecencyWeightedRiskModel,
    output_path: Path,
) -> None:
    """Plot best chronological pooling weights against the uniform baseline."""
    weights = model.effective_recency_weights().detach().cpu().numpy()
    minutes_before_decision = np.arange(model.config.history_minutes, 0, -1)
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(
        minutes_before_decision,
        weights,
        marker="o",
        markersize=3,
        label="Learned constrained weight",
    )
    axis.axhline(
        1.0 / model.config.history_minutes,
        color="black",
        linestyle="--",
        label="Uniform baseline",
    )
    axis.invert_xaxis()
    axis.set(
        title="Best minute recency weights",
        xlabel="Minutes before decision (oldest to newest)",
        ylabel="Pooling weight",
    )
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_validation_summary(
    result: EvaluationResult,
    output_path: Path,
) -> None:
    """Save the best model's full-validation precision-recall curve."""
    precision, recall, _ = precision_recall_curve(
        result.labels,
        result.probabilities,
    )
    prevalence = float(np.mean(result.labels))
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.plot(
        recall,
        precision,
        label=f"Recency model (AP={result.average_precision:.4f})",
    )
    axis.axhline(
        prevalence,
        color="black",
        linestyle="--",
        label=f"Random AP ({prevalence:.4f})",
    )
    axis.set(
        title="Best constrained-recency validation result",
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
    """Initialize from the best baseline and learn constrained recency."""
    arguments = parse_arguments()
    validate_arguments(arguments)
    config = resolve_label_definition(arguments)
    config.validate()
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

    chunk_seconds = config.chunk_window_seconds
    chunks_per_minute = int(round(60.0 / chunk_seconds))
    sequence_chunks = int(round(config.input_window_seconds / chunk_seconds))
    model_config = EEGNetRecencyWeightedConfig(
        n_chans=int(baseline_config["n_chans"]),
        chunk_samples=int(baseline_config["chunk_samples"]),
        sequence_chunks=sequence_chunks,
        chunks_per_minute=chunks_per_minute,
        embedding_dim=int(baseline_config["embedding_dim"]),
        encoder_chunk_batch_size=int(
            baseline_config["encoder_chunk_batch_size"]
        ),
        encoder_dropout=float(baseline_config["dropout"]),
        softmax_temperature=arguments.softmax_temperature,
        max_temporal_strength=arguments.max_temporal_strength,
    )
    for name in ("n_chans", "chunk_samples", "sequence_chunks"):
        if int(baseline_config[name]) != getattr(model_config, name):
            raise ValueError(f"Baseline checkpoint has incompatible {name}.")

    validate_embedding_cache(
        arguments.cache_dir,
        arguments.baseline_checkpoint,
        model_config.embedding_dim,
        config,
    )
    manifest_path = config.manifests_dir / "processed_shard_manifest.csv"
    train_examples = load_decision_examples(
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
        train_examples["label"],
        negative_to_positive_ratio=arguments.negative_to_positive_ratio,
        seed=arguments.seed,
    )
    train_loader = build_loader(
        train_examples,
        config=config,
        cache_dir=arguments.cache_dir,
        embedding_dim=model_config.embedding_dim,
        batch_size=arguments.batch_size,
        num_workers=arguments.num_workers,
        device=device,
        sampler=train_sampler,
    )
    validation_loader = build_loader(
        validation_examples,
        config=config,
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
    model_config = EEGNetRecencyWeightedConfig(
        **{
            **asdict(model_config),
            "sampling_prior_logit_correction": logit_correction,
        }
    )
    model = EEGNetRecencyWeightedRiskModel(model_config)
    model.initialize_from_baseline_state_dict(baseline_state_dict)
    model.set_baseline_trainable(False)
    model.to(device)

    optimizer = AdamW(
        [model.recency_logits],
        lr=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
    )
    criterion = nn.BCEWithLogitsLoss()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = arguments.output_dir / "best_model.pt"
    metrics_path = arguments.output_dir / "metrics.json"
    curves_path = arguments.output_dir / "learning_curves.png"
    recency_weights_path = arguments.output_dir / "recency_weights.png"
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
    print("Frozen baseline: EEGNet encoder plus original classifier.")
    print(
        f"Trainable temporal component: {model.config.history_minutes} "
        "constrained minute-recency logits."
    )

    initial_result = evaluate(
        model,
        validation_loader,
        criterion,
        device,
        logit_correction,
    )
    initial_summary = recency_summary(model)
    history: list[dict[str, float | int | str | None]] = [
        {
            "epoch": 0,
            "stage": "unchanged_baseline",
            "train_bce": None,
            "train_objective": None,
            "uniform_weight_kl": initial_summary["uniform_weight_kl"],
            "minimum_minute_weight": initial_summary["minimum_minute_weight"],
            "maximum_minute_weight": initial_summary["maximum_minute_weight"],
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
        train_result = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            arguments.max_grad_norm,
            arguments.uniform_kl_weight,
        )
        validation_result = evaluate(
            model,
            validation_loader,
            criterion,
            device,
            logit_correction,
        )
        weight_summary = recency_summary(model)
        row: dict[str, float | int | str | None] = {
            "epoch": epoch,
            "stage": "constrained_recency",
            "train_bce": train_result.binary_cross_entropy,
            "train_objective": train_result.objective,
            "uniform_weight_kl": weight_summary["uniform_weight_kl"],
            "minimum_minute_weight": weight_summary["minimum_minute_weight"],
            "maximum_minute_weight": weight_summary["maximum_minute_weight"],
            "weight_center_minutes_before_decision": weight_summary[
                "weight_center_minutes_before_decision"
            ],
            "validation_loss": validation_result.loss,
            "validation_average_precision": validation_result.average_precision,
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True))

        improved = (
            validation_result.average_precision
            > best_average_precision + arguments.minimum_ap_improvement
        )
        if improved:
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
    save_recency_weights(model, recency_weights_path)
    save_validation_summary(best_result, validation_summary_path)

    improvement = best_average_precision - initial_result.average_precision
    print(
        f"Best validation AP: {best_average_precision:.6f} at epoch {best_epoch} "
        f"(change from unchanged baseline: {improvement:+.6f})"
    )
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Metrics: {metrics_path}")
    print(f"Learning curves: {curves_path}")
    print(f"Recency weights: {recency_weights_path}")
    print(f"Validation summary: {validation_summary_path}")
    print("The held-out test split was not used.")


if __name__ == "__main__":
    main()
