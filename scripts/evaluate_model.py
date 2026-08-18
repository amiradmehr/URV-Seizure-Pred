r"""Evaluate a trained seizure-risk checkpoint and write the result figures.

Training selects a checkpoint on a *subsampled* validation split so that
per-epoch scoring stays affordable. This script produces the reportable numbers
instead: it scores every retained decision in a split at its true prevalence,
which is the only setting in which precision, calibration, and false-alarm rate
mean anything.

Two views of sensitivity are reported, because they answer different questions:

* **Decision-level** — of all one-minute decisions that precede a seizure by
  under 10 minutes, how many raised an alarm?
* **Seizure-level** — of all target seizures, how many had *at least one*
  alarm during their pre-onset window? This is the clinically relevant one; a
  patient needs one warning per seizure, not ten.

False alarms are reported as alarms per interictal hour. Decisions are spaced
``input_stride_seconds`` apart, so each negative decision stands for that much
interictal time.

Example:

    .venv/bin/python scripts/evaluate_model.py --split validation
    .venv/bin/python scripts/evaluate_model.py --split test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
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
    EEGNetAttentionConfig,
    EEGNetAttentionRiskModel,
    EEGNetMeanPoolConfig,
    EEGNetMeanPoolRiskModel,
)


# Categorical slots 1-3 of the validated reference palette. These three clear
# the all-pairs colour-vision and normal-vision separation floors in both
# light and dark rendering, which the full eight-hue order does not.
SERIES_BLUE = "#2a78d6"
SERIES_ORANGE = "#eb6834"
SERIES_AQUA = "#1baf7a"
INK_MUTED = "#52514e"

# Alarm budgets to tabulate. Anything above roughly one false alarm per hour is
# generally considered unusable in ambulatory seizure prediction.
REPORTED_ALARM_BUDGETS = (0.1, 0.25, 0.5, 1.0, 2.0)


def parse_arguments() -> argparse.Namespace:
    """Parse evaluation settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "models" / "eegnet_mean_pool" / "best_model.pt",
    )
    parser.add_argument(
        "--split",
        choices=("validation", "test"),
        default="validation",
        help="Evaluate the held-out test split only after model selection is final.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "evaluation",
    )
    parser.add_argument(
        "--negative-ratio",
        type=float,
        default=None,
        help=(
            "Optional negatives per positive. Omit to evaluate every decision "
            "at true prevalence, which is what the reported metrics assume."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def log(message: str) -> None:
    """Print immediately so batch runs show live progress."""
    print(message, flush=True)


def resolve_device(requested_device: str) -> torch.device:
    """Select the requested device with a clear CUDA failure."""
    if requested_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but no CUDA device is available.")
        return torch.device("cuda")
    if requested_device == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    """Rebuild the trained model exactly as it was checkpointed.

    Older checkpoints predate the architecture field and are always mean-pool.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Run training first."
        )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    architecture = checkpoint.get("architecture", "meanpool")
    if architecture == "attention":
        model_config = EEGNetAttentionConfig(**checkpoint["model_config"])
    else:
        model_config = EEGNetMeanPoolConfig(**checkpoint["model_config"])

    saved_config = checkpoint.get("config", {})
    expected_chunk_samples = int(CONFIG.chunk_window_seconds * CONFIG.target_sfreq)
    if model_config.chunk_samples != expected_chunk_samples:
        raise ValueError(
            "The checkpoint was trained on a different chunk length: "
            f"{model_config.chunk_samples} versus {expected_chunk_samples}. "
            "The current config cannot evaluate it."
        )
    if saved_config.get("input_window_seconds", CONFIG.input_window_seconds) != (
        CONFIG.input_window_seconds
    ):
        raise ValueError(
            "The checkpoint was trained with a different history length than "
            "the current config defines."
        )

    if architecture == "attention":
        model = EEGNetAttentionRiskModel(model_config).to(device)
    else:
        model = EEGNetMeanPoolRiskModel(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


@torch.no_grad()
def score_split(
    model: nn.Module,
    examples: pd.DataFrame,
    device: torch.device,
    *,
    batch_size: int,
    num_workers: int,
    window_normalize: bool = False,
) -> np.ndarray:
    """Return one seizure-risk probability per decision, in ``examples`` order."""
    dataset = StreamingDecisionDataset(
        examples, CONFIG, window_normalize=window_normalize
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )

    probabilities: list[float] = []
    for signal, availability, _ in tqdm(loader, desc="Scoring", unit="batch"):
        signal = signal.to(device, non_blocking=True)
        availability = availability.to(device, non_blocking=True)
        probabilities.extend(
            model.predict_proba(signal, availability).float().cpu().tolist()
        )

    return np.asarray(probabilities, dtype=np.float64)


def alarms_per_hour(
    false_positive_count: np.ndarray | int,
    negative_count: int,
) -> np.ndarray | float:
    """Convert a false-positive count into false alarms per interictal hour."""
    interictal_hours = negative_count * CONFIG.input_stride_seconds / 3600.0
    return false_positive_count / interictal_hours


def seizure_level_sensitivity(
    predictions: pd.DataFrame,
    threshold: float,
) -> tuple[float, int, int]:
    """Return the share of target seizures that raised at least one alarm."""
    positives = predictions[predictions["label"] == 1]
    if positives.empty:
        return float("nan"), 0, 0

    alarmed_by_seizure = (
        positives.assign(alarm=positives["probability"] >= threshold)
        .groupby("target_seizure_id")["alarm"]
        .any()
    )
    detected = int(alarmed_by_seizure.sum())
    total = int(len(alarmed_by_seizure))
    return detected / total, detected, total


def operating_point_table(predictions: pd.DataFrame) -> pd.DataFrame:
    """Sweep thresholds and record decision- and seizure-level behaviour."""
    labels = predictions["label"].to_numpy(dtype=np.int64)
    scores = predictions["probability"].to_numpy(dtype=np.float64)
    positive_count = int((labels == 1).sum())
    negative_count = int((labels == 0).sum())

    # Sweep the observed score distribution rather than a fixed grid, so every
    # achievable operating point appears exactly once.
    candidate_thresholds = np.unique(scores)
    if len(candidate_thresholds) > 2000:
        candidate_thresholds = np.quantile(
            candidate_thresholds,
            np.linspace(0.0, 1.0, 2000),
        )

    rows: list[dict[str, float]] = []
    for threshold in candidate_thresholds:
        alarm = scores >= threshold
        true_positives = int((alarm & (labels == 1)).sum())
        false_positives = int((alarm & (labels == 0)).sum())
        seizure_sensitivity, detected, total = seizure_level_sensitivity(
            predictions,
            float(threshold),
        )
        rows.append(
            {
                "threshold": float(threshold),
                "decision_sensitivity": true_positives / max(positive_count, 1),
                "specificity": 1.0 - false_positives / max(negative_count, 1),
                "precision": true_positives / max(true_positives + false_positives, 1),
                "false_alarms_per_hour": float(
                    alarms_per_hour(false_positives, negative_count)
                ),
                "seizure_sensitivity": seizure_sensitivity,
                "seizures_detected": detected,
                "seizures_total": total,
                "true_positives": true_positives,
                "false_positives": false_positives,
            }
        )

    table = pd.DataFrame(rows)
    table["f1"] = (
        2.0
        * table["precision"]
        * table["decision_sensitivity"]
        / (table["precision"] + table["decision_sensitivity"]).replace(0.0, np.nan)
    ).fillna(0.0)
    return table


def summarize_at_alarm_budgets(table: pd.DataFrame) -> list[dict[str, float]]:
    """Report the most sensitive operating point under each alarm budget."""
    summaries: list[dict[str, float]] = []
    for budget in REPORTED_ALARM_BUDGETS:
        affordable = table[table["false_alarms_per_hour"] <= budget]
        if affordable.empty:
            summaries.append(
                {
                    "false_alarm_budget_per_hour": budget,
                    "achievable": False,
                }
            )
            continue
        best = affordable.loc[affordable["seizure_sensitivity"].idxmax()]
        summaries.append(
            {
                "false_alarm_budget_per_hour": budget,
                "achievable": True,
                "threshold": float(best["threshold"]),
                "seizure_sensitivity": float(best["seizure_sensitivity"]),
                "seizures_detected": int(best["seizures_detected"]),
                "seizures_total": int(best["seizures_total"]),
                "decision_sensitivity": float(best["decision_sensitivity"]),
                "precision": float(best["precision"]),
                "achieved_false_alarms_per_hour": float(
                    best["false_alarms_per_hour"]
                ),
            }
        )
    return summaries


def style_axis(axis: plt.Axes) -> None:
    """Apply the shared recessive grid and frame treatment."""
    axis.grid(alpha=0.25, linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(INK_MUTED)
        axis.spines[side].set_linewidth(0.8)


def plot_discrimination(
    predictions: pd.DataFrame,
    metrics: dict[str, Any],
    output_path: Path,
    split: str,
) -> None:
    """ROC, precision-recall, calibration, and the score distribution."""
    labels = predictions["label"].to_numpy(dtype=np.int64)
    scores = predictions["probability"].to_numpy(dtype=np.float64)
    prevalence = float(labels.mean())

    figure, axes = plt.subplots(2, 2, figsize=(12, 10))

    false_positive_rate, true_positive_rate, _ = roc_curve(labels, scores)
    roc_axis = axes[0, 0]
    roc_axis.plot(
        false_positive_rate,
        true_positive_rate,
        color=SERIES_BLUE,
        linewidth=2,
    )
    roc_axis.plot([0, 1], [0, 1], color=INK_MUTED, linewidth=1, linestyle="--")
    roc_axis.annotate(
        f"AUC {metrics['roc_auc']:.3f}",
        xy=(0.55, 0.12),
        xycoords="axes fraction",
        color=SERIES_BLUE,
        fontsize=12,
        fontweight="bold",
    )
    roc_axis.set(
        title="ROC — decision level",
        xlabel="False positive rate",
        ylabel="True positive rate",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    style_axis(roc_axis)

    precision, recall, _ = precision_recall_curve(labels, scores)
    pr_axis = axes[0, 1]
    pr_axis.plot(recall, precision, color=SERIES_BLUE, linewidth=2)
    pr_axis.axhline(
        prevalence,
        color=INK_MUTED,
        linewidth=1,
        linestyle="--",
    )
    pr_axis.annotate(
        f"chance {prevalence:.4f}",
        xy=(0.02, prevalence),
        xytext=(0.02, prevalence * 1.6 + 0.01),
        color=INK_MUTED,
        fontsize=9,
    )
    pr_axis.annotate(
        f"AP {metrics['average_precision']:.3f}",
        xy=(0.55, 0.8),
        xycoords="axes fraction",
        color=SERIES_BLUE,
        fontsize=12,
        fontweight="bold",
    )
    pr_axis.set(
        title="Precision-recall — decision level",
        xlabel="Recall",
        ylabel="Precision",
        xlim=(0, 1),
    )
    style_axis(pr_axis)

    calibration_axis = axes[1, 0]
    # Quantile bins keep every point supported by the same number of decisions,
    # which matters at a prevalence this low.
    observed, predicted = calibration_curve(
        labels,
        scores,
        n_bins=10,
        strategy="quantile",
    )
    calibration_axis.plot(
        [0, max(predicted.max(), observed.max()) * 1.05],
        [0, max(predicted.max(), observed.max()) * 1.05],
        color=INK_MUTED,
        linewidth=1,
        linestyle="--",
    )
    calibration_axis.plot(
        predicted,
        observed,
        color=SERIES_ORANGE,
        linewidth=2,
        marker="o",
        markersize=8,
        markeredgecolor="white",
        markeredgewidth=1.5,
    )
    calibration_axis.set(
        title=f"Calibration (Brier {metrics['brier_score']:.5f})",
        xlabel="Predicted probability",
        ylabel="Observed seizure frequency",
    )
    style_axis(calibration_axis)

    distribution_axis = axes[1, 1]
    bins = np.linspace(0.0, 1.0, 51)
    distribution_axis.hist(
        scores[labels == 0],
        bins=bins,
        color=SERIES_BLUE,
        alpha=0.75,
        label="No seizure in next 10 min",
        density=True,
    )
    distribution_axis.hist(
        scores[labels == 1],
        bins=bins,
        color=SERIES_ORANGE,
        alpha=0.75,
        label="Seizure in next 10 min",
        density=True,
    )
    distribution_axis.set(
        title="Predicted risk by outcome",
        xlabel="Predicted probability",
        ylabel="Density",
    )
    distribution_axis.legend(frameon=False, fontsize=9)
    style_axis(distribution_axis)

    figure.suptitle(
        f"EEGNet mean-pool seizure risk — {split} split "
        f"({len(predictions):,} decisions, {int(labels.sum()):,} positive)",
        fontsize=13,
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def plot_operating_characteristics(
    table: pd.DataFrame,
    output_path: Path,
    split: str,
) -> None:
    """Sensitivity against the false-alarm budget, plus the threshold sweep."""
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    ordered = table.sort_values("false_alarms_per_hour")
    alarm_axis = axes[0]
    alarm_axis.plot(
        ordered["false_alarms_per_hour"],
        ordered["seizure_sensitivity"],
        color=SERIES_ORANGE,
        linewidth=2,
        label="Seizure level (any alarm before onset)",
    )
    alarm_axis.plot(
        ordered["false_alarms_per_hour"],
        ordered["decision_sensitivity"],
        color=SERIES_BLUE,
        linewidth=2,
        label="Decision level (every 1-min decision)",
    )
    # One false alarm per hour is the rough ceiling of what ambulatory use
    # tolerates; everything to the right of this line is academic.
    alarm_axis.axvline(1.0, color=INK_MUTED, linewidth=1, linestyle=":")
    alarm_axis.annotate(
        "1 alarm/h",
        xy=(1.0, 0.04),
        xycoords=("data", "axes fraction"),
        xytext=(4, 0),
        textcoords="offset points",
        color=INK_MUTED,
        fontsize=9,
    )
    alarm_axis.set(
        title="Sensitivity versus false-alarm rate",
        xlabel="False alarms per interictal hour",
        ylabel="Sensitivity",
        xscale="log",
        ylim=(0, 1.02),
    )
    alarm_axis.legend(frameon=False, fontsize=9, loc="lower right")
    style_axis(alarm_axis)

    threshold_axis = axes[1]
    threshold_axis.plot(
        table["threshold"],
        table["decision_sensitivity"],
        color=SERIES_BLUE,
        linewidth=2,
        label="Sensitivity",
    )
    threshold_axis.plot(
        table["threshold"],
        table["specificity"],
        color=SERIES_AQUA,
        linewidth=2,
        label="Specificity",
    )
    threshold_axis.plot(
        table["threshold"],
        table["f1"],
        color=SERIES_ORANGE,
        linewidth=2,
        label="F1",
    )
    threshold_axis.set(
        title="Decision-level metrics versus threshold",
        xlabel="Alarm threshold",
        ylabel="Score",
        ylim=(0, 1.02),
    )
    threshold_axis.legend(frameon=False, fontsize=9)
    style_axis(threshold_axis)

    figure.suptitle(
        f"Operating characteristics — {split} split", fontsize=13
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def per_patient_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    """Score every patient separately to expose between-patient variance."""
    rows: list[dict[str, Any]] = []
    for subject, subject_predictions in predictions.groupby("subject", sort=True):
        labels = subject_predictions["label"].to_numpy(dtype=np.int64)
        scores = subject_predictions["probability"].to_numpy(dtype=np.float64)
        positive_count = int((labels == 1).sum())
        has_both_classes = len(np.unique(labels)) == 2
        rows.append(
            {
                "subject": str(subject),
                "decisions": len(labels),
                "positive_decisions": positive_count,
                "negative_decisions": int((labels == 0).sum()),
                "target_seizures": int(
                    subject_predictions.loc[
                        subject_predictions["label"] == 1, "target_seizure_id"
                    ].nunique()
                ),
                "average_precision": (
                    float(average_precision_score(labels, scores))
                    if has_both_classes
                    else float("nan")
                ),
                "roc_auc": (
                    float(roc_auc_score(labels, scores))
                    if has_both_classes
                    else float("nan")
                ),
                "mean_predicted_risk": float(scores.mean()),
            }
        )
    return pd.DataFrame(rows)


def plot_per_patient(
    patient_summary: pd.DataFrame,
    overall_average_precision: float,
    output_path: Path,
    split: str,
) -> None:
    """Show how unevenly the baseline performs across held-out patients."""
    scored = patient_summary.dropna(subset=["average_precision"]).sort_values(
        "average_precision",
        ascending=False,
    )
    if scored.empty:
        return

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(13, max(4.0, 0.42 * len(scored) + 2.0)),
        gridspec_kw={"width_ratios": [3, 2]},
    )

    positions = np.arange(len(scored))
    metric_axis = axes[0]
    metric_axis.barh(
        positions,
        scored["average_precision"],
        color=SERIES_BLUE,
        height=0.62,
    )
    metric_axis.axvline(
        overall_average_precision,
        color=SERIES_ORANGE,
        linewidth=2,
        linestyle="--",
    )
    metric_axis.annotate(
        f"pooled AP {overall_average_precision:.3f}",
        xy=(overall_average_precision, 1.0),
        xycoords=("data", "axes fraction"),
        xytext=(6, -12),
        textcoords="offset points",
        color=SERIES_ORANGE,
        fontsize=9,
        fontweight="bold",
        va="center",
    )
    for position, value in zip(positions, scored["average_precision"], strict=True):
        metric_axis.annotate(
            f"{value:.3f}",
            xy=(value, position),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=INK_MUTED,
        )
    metric_axis.set(
        title="Average precision by held-out patient",
        xlabel="Average precision",
        yticks=positions,
        yticklabels=[f"sub-{subject}" for subject in scored["subject"]],
    )
    metric_axis.invert_yaxis()
    style_axis(metric_axis)

    count_axis = axes[1]
    count_axis.barh(
        positions,
        scored["target_seizures"],
        color=SERIES_AQUA,
        height=0.62,
    )
    for position, value in zip(positions, scored["target_seizures"], strict=True):
        count_axis.annotate(
            f"{int(value)}",
            xy=(value, position),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=INK_MUTED,
        )
    count_axis.set(
        title="Eligible target seizures",
        xlabel="Seizures",
        yticks=positions,
        yticklabels=[],
    )
    count_axis.invert_yaxis()
    style_axis(count_axis)

    figure.suptitle(
        f"Per-patient breakdown — {split} split "
        "(each patient is entirely unseen during training)",
        fontsize=13,
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """Score one split and write every result artefact."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    arguments = parse_arguments()
    CONFIG.validate()

    device = resolve_device(arguments.device)
    log(f"Device: {device}")
    if device.type == "cuda":
        log(f"GPU: {torch.cuda.get_device_name(device)}")

    manifest_path = CONFIG.manifests_dir / "processed_shard_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Processed manifest not found: {manifest_path}. "
            "Run build_dataset.py and validate_dataset.py first."
        )

    model, checkpoint = load_model(arguments.checkpoint, device)
    log(
        "Loaded checkpoint selected at validation average precision "
        f"{checkpoint.get('best_validation_average_precision', float('nan')):.4f}"
    )

    log(f"Loading {arguments.split} decisions...")
    examples = load_decision_examples(
        manifest_path,
        split=arguments.split,
        negative_to_positive_ratio=arguments.negative_ratio,
        seed=arguments.seed,
    )
    positive_count = int((examples["label"] == 1).sum())
    negative_count = int((examples["label"] == 0).sum())
    log(
        f"{arguments.split}: {len(examples):,} decisions "
        f"({positive_count:,} positive, {negative_count:,} negative, "
        f"prevalence {positive_count / len(examples):.5f})"
    )
    if positive_count == 0 or negative_count == 0:
        raise ValueError(f"Split {arguments.split!r} does not contain both classes.")

    window_normalize = bool(checkpoint.get("window_normalize", False))
    log(
        f"Architecture: {checkpoint.get('architecture', 'meanpool')} | "
        f"window_normalize={window_normalize}"
    )
    probabilities = score_split(
        model,
        examples,
        device,
        batch_size=arguments.batch_size,
        num_workers=arguments.num_workers,
        window_normalize=window_normalize,
    )

    predictions = examples.copy()
    predictions["probability"] = probabilities

    output_directory = arguments.output_dir / arguments.split
    output_directory.mkdir(parents=True, exist_ok=True)

    prediction_columns = [
        column
        for column in (
            "subject",
            "recording_id",
            "decision_time_seconds",
            "label",
            "target_seizure_id",
            "seizure_scope",
            "probability",
        )
        if column in predictions.columns
    ]
    predictions[prediction_columns].to_csv(
        output_directory / "predictions.csv",
        index=False,
    )

    labels = predictions["label"].to_numpy(dtype=np.int64)
    scores = predictions["probability"].to_numpy(dtype=np.float64)

    table = operating_point_table(predictions)
    table.to_csv(output_directory / "operating_points.csv", index=False)

    best_f1_row = table.loc[table["f1"].idxmax()]
    metrics: dict[str, Any] = {
        "split": arguments.split,
        "checkpoint": str(arguments.checkpoint),
        "decisions": int(len(predictions)),
        "positive_decisions": positive_count,
        "negative_decisions": negative_count,
        "prevalence": positive_count / len(predictions),
        "patients": int(predictions["subject"].nunique()),
        "target_seizures": int(
            predictions.loc[predictions["label"] == 1, "target_seizure_id"].nunique()
        ),
        "interictal_hours": negative_count * CONFIG.input_stride_seconds / 3600.0,
        "average_precision": float(average_precision_score(labels, scores)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "brier_score": float(brier_score_loss(labels, scores)),
        "accuracy_at_0.5": float(np.mean((scores >= 0.5) == labels)),
        "best_f1_operating_point": {
            key: float(best_f1_row[key])
            for key in (
                "threshold",
                "f1",
                "decision_sensitivity",
                "specificity",
                "precision",
                "seizure_sensitivity",
                "false_alarms_per_hour",
            )
        },
        "alarm_budgets": summarize_at_alarm_budgets(table),
    }

    patient_summary = per_patient_summary(predictions)
    patient_summary.to_csv(output_directory / "per_patient_metrics.csv", index=False)

    (output_directory / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    plot_discrimination(
        predictions,
        metrics,
        output_directory / "discrimination.png",
        arguments.split,
    )
    plot_operating_characteristics(
        table,
        output_directory / "operating_characteristics.png",
        arguments.split,
    )
    plot_per_patient(
        patient_summary,
        metrics["average_precision"],
        output_directory / "per_patient.png",
        arguments.split,
    )

    print()
    print("=" * 78)
    print(f"{arguments.split.upper()} RESULTS".center(78))
    print("=" * 78)
    print(f"Patients                : {metrics['patients']}")
    print(f"Decisions               : {metrics['decisions']:,}")
    print(f"Target seizures         : {metrics['target_seizures']}")
    print(f"Prevalence              : {metrics['prevalence']:.5f}")
    print(f"Average precision       : {metrics['average_precision']:.4f}")
    print(f"ROC AUC                 : {metrics['roc_auc']:.4f}")
    print(f"Brier score             : {metrics['brier_score']:.5f}")
    print()
    print("Sensitivity under a false-alarm budget:")
    for budget in metrics["alarm_budgets"]:
        if not budget["achievable"]:
            print(
                f"  <= {budget['false_alarm_budget_per_hour']:>4} alarms/h : "
                "not achievable at any threshold"
            )
            continue
        print(
            f"  <= {budget['false_alarm_budget_per_hour']:>4} alarms/h : "
            f"{budget['seizure_sensitivity']:.3f} seizure-level "
            f"({budget['seizures_detected']}/{budget['seizures_total']}), "
            f"{budget['decision_sensitivity']:.3f} decision-level"
        )
    print()
    print(f"Artefacts: {output_directory}")


if __name__ == "__main__":
    main()
