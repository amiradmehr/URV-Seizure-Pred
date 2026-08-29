r"""Evaluate the selected EEGNet baseline as a seizure-warning system.

This script never uses the held-out test split. It generates prior-corrected
probabilities for every validation decision, sweeps score thresholds, merges
nearby alerts into episodes, and reports seizure sensitivity against false
alarms per 24 hours and time in warning.

PowerShell example:

    .\.venv-old\Scripts\python.exe scripts\evaluate_eegnet_events.py --device cuda
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader


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
    CachedEmbeddingDecisionDataset,
    load_decision_examples,
)
from seizure_prediction.event_evaluation import (  # noqa: E402
    bootstrap_patient_metrics,
    evaluate_alarm_threshold,
    prepare_target_seizures,
    threshold_grid,
)
from seizure_prediction.models import (  # noqa: E402
    BaselineEEGNet,
    BaselineEEGNetConfig,
)


def parse_arguments() -> argparse.Namespace:
    """Return inference and operational-evaluation settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--number-of-thresholds",
        type=int,
        default=201,
        help="Number of score-rank thresholds in the operating sweep.",
    )
    parser.add_argument(
        "--refractory-minutes",
        type=float,
        default=10.0,
        help="Alerts no farther apart than this are one alarm episode.",
    )
    parser.add_argument(
        "--operating-false-alarm-budget",
        type=float,
        default=1.0,
        help=(
            "Validation operating point to inspect in detail, expressed as "
            "false alarm episodes per 24 interictal hours."
        ),
    )
    parser.add_argument(
        "--reported-false-alarm-budgets",
        type=float,
        nargs="+",
        default=(0.5, 1.0, 2.0, 4.0),
    )
    parser.add_argument(
        "--target-sensitivities",
        type=float,
        nargs="+",
        default=(0.65, 0.70, 0.75),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    parser.add_argument(
        "--checkpoint",
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
            / "evaluation"
            / "eegnet_baseline_ratio10_lr1e4_events"
        ),
    )
    add_label_definition_arguments(parser)
    return parser.parse_args()


def validate_arguments(arguments: argparse.Namespace) -> None:
    """Fail before inference when an evaluation setting is invalid."""
    if arguments.batch_size <= 0:
        raise ValueError("batch-size must be positive.")
    if arguments.num_workers < 0:
        raise ValueError("num-workers cannot be negative.")
    if arguments.number_of_thresholds < 2:
        raise ValueError("number-of-thresholds must be at least two.")
    if arguments.refractory_minutes < 0.0:
        raise ValueError("refractory-minutes cannot be negative.")
    if arguments.operating_false_alarm_budget < 0.0:
        raise ValueError("operating-false-alarm-budget cannot be negative.")
    if any(value < 0.0 for value in arguments.reported_false_alarm_budgets):
        raise ValueError("reported false-alarm budgets cannot be negative.")
    if any(
        not 0.0 <= value <= 1.0 for value in arguments.target_sensitivities
    ):
        raise ValueError("target sensitivities must be in [0, 1].")
    if arguments.bootstrap_samples <= 0:
        raise ValueError("bootstrap-samples must be positive.")


def set_seed(seed: int) -> None:
    """Seed inference helpers and patient bootstrap resampling."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def resolve_device(requested_device: str) -> torch.device:
    """Select the requested accelerator, preferring CUDA then MPS then CPU under 'auto'."""
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available.")
    if requested_device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested, but no MPS device is available.")
    if requested_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested_device)


def sha256_file(path: Path) -> str:
    """Return a stable fingerprint for checkpoint/cache matching."""
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_embedding_cache(
    cache_dir: Path,
    checkpoint_path: Path,
    config: BaselineEEGNetConfig,
) -> None:
    """Ensure inference features came from the evaluated checkpoint."""
    metadata_path = cache_dir / "cache_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Embedding-cache metadata not found: {metadata_path}. "
            "Run scripts/cache_eegnet_embeddings.py first."
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("baseline_checkpoint_sha256") != sha256_file(checkpoint_path):
        raise ValueError(
            "The embedding cache was created from a different checkpoint."
        )
    expected_values = {
        "embedding_dim": config.embedding_dim,
        "chunk_samples": config.chunk_samples,
    }
    for name, expected_value in expected_values.items():
        if int(metadata.get(name, -1)) != expected_value:
            raise ValueError(f"Embedding cache has incompatible {name}.")
    if "validation" not in set(metadata.get("splits", [])):
        raise ValueError("Embedding cache does not contain validation features.")


def load_baseline(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[BaselineEEGNet, BaselineEEGNetConfig, dict[str, Any]]:
    """Load and validate the selected BaselineEEGNet checkpoint."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    config_values = checkpoint.get("model_config")
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(config_values, dict) or not isinstance(state_dict, dict):
        raise ValueError("Checkpoint does not contain a complete baseline model.")
    config = BaselineEEGNetConfig(**config_values)
    model = BaselineEEGNet(config)
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    return model, config, checkpoint


def generate_validation_predictions(
    model: BaselineEEGNet,
    config: BaselineEEGNetConfig,
    examples: pd.DataFrame,
    *,
    preprocessing_config,
    cache_dir: Path,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> pd.DataFrame:
    """Run cached baseline inference without changing decision order."""
    dataset = CachedEmbeddingDecisionDataset(
        examples,
        preprocessing_config,
        cache_root=cache_dir,
        embedding_dim=config.embedding_dim,
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
    observed_labels: list[int] = []
    with torch.inference_mode():
        for embeddings, availability, target in loader:
            embeddings = embeddings.to(device, non_blocking=True)
            availability = availability.to(device, non_blocking=True)
            logits = model.forward_from_chunk_embeddings(
                embeddings,
                availability,
            )
            corrected_logits = (
                logits + config.sampling_prior_logit_correction
            )
            probabilities.extend(
                torch.sigmoid(corrected_logits).cpu().tolist()
            )
            observed_labels.extend(target.to(torch.int64).tolist())

    expected_labels = examples["label"].to_numpy(dtype=np.int64)
    if not np.array_equal(
        np.asarray(observed_labels, dtype=np.int64),
        expected_labels,
    ):
        raise RuntimeError("Inference output order does not match metadata order.")
    predictions = examples.copy()
    predictions["probability"] = np.asarray(probabilities, dtype=np.float64)
    return predictions


def validate_positive_timing(
    predictions: pd.DataFrame,
    seizures: pd.DataFrame,
) -> None:
    """Verify exact onsets reproduce every positive decision label."""
    positives = predictions[predictions["label"] == 1].copy()
    onset_lookup = seizures.set_index("seizure_id")["onset_seconds"]
    positives["onset_seconds"] = positives["target_seizure_id"].map(onset_lookup)
    if positives["onset_seconds"].isna().any():
        raise ValueError("A positive decision has no matching target onset.")
    valid = (
        (positives["decision_time_seconds"] < positives["onset_seconds"])
        & (
            positives["prediction_start_seconds"]
            <= positives["onset_seconds"]
        )
        & (
            positives["onset_seconds"]
            <= positives["prediction_stop_seconds"]
        )
    )
    if not valid.all():
        raise ValueError("Positive decision timing disagrees with seizure onsets.")


def select_for_false_alarm_budget(
    sweep: pd.DataFrame,
    false_alarm_budget: float,
) -> pd.Series:
    """Select maximum sensitivity under a validation false-alarm budget."""
    candidates = sweep[
        sweep["false_alarms_per_24h"] <= false_alarm_budget + 1e-12
    ]
    if candidates.empty:
        raise RuntimeError("Threshold sweep contains no feasible operating point.")
    return candidates.sort_values(
        [
            "event_sensitivity",
            "macro_patient_sensitivity",
            "time_in_warning_fraction",
            "false_alarms_per_24h",
            "threshold",
        ],
        ascending=[False, False, True, True, False],
    ).iloc[0]


def select_for_target_sensitivity(
    sweep: pd.DataFrame,
    target_sensitivity: float,
) -> pd.Series | None:
    """Select the lowest-FAR row meeting an event-sensitivity target."""
    candidates = sweep[
        sweep["event_sensitivity"] + 1e-12 >= target_sensitivity
    ]
    if candidates.empty:
        return None
    return candidates.sort_values(
        [
            "false_alarms_per_24h",
            "time_in_warning_fraction",
            "threshold",
        ],
        ascending=[True, True, False],
    ).iloc[0]


def row_to_dict(row: pd.Series) -> dict[str, float | int]:
    """Convert a metric row to JSON-safe Python scalars."""
    result: dict[str, float | int] = {}
    for name, value in row.items():
        if isinstance(value, (np.integer, int)):
            result[str(name)] = int(value)
        else:
            result[str(name)] = float(value)
    return result


def save_operating_curve(
    sweep: pd.DataFrame,
    selected: pd.Series,
    output_path: Path,
) -> None:
    """Plot the event-level tradeoff used to choose an operating point."""
    finite = sweep.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["false_alarms_per_24h"]
    )
    figure, (far_axis, warning_axis) = plt.subplots(1, 2, figsize=(12, 5))
    far_axis.plot(
        finite["false_alarms_per_24h"],
        finite["event_sensitivity"],
        marker=".",
        linewidth=1.0,
    )
    far_axis.scatter(
        [selected["false_alarms_per_24h"]],
        [selected["event_sensitivity"]],
        color="red",
        label="Detailed operating point",
        zorder=3,
    )
    far_axis.set_xscale("symlog", linthresh=0.1)
    far_axis.set(
        title="Seizure sensitivity vs false alarms",
        xlabel="False alarm episodes per 24 interictal hours",
        ylabel="Event sensitivity",
        ylim=(-0.02, 1.02),
    )
    far_axis.grid(alpha=0.3)
    far_axis.legend()

    warning_axis.plot(
        finite["time_in_warning_fraction"],
        finite["event_sensitivity"],
        marker=".",
        linewidth=1.0,
    )
    warning_axis.scatter(
        [selected["time_in_warning_fraction"]],
        [selected["event_sensitivity"]],
        color="red",
        zorder=3,
    )
    warning_axis.set(
        title="Seizure sensitivity vs warning burden",
        xlabel="Fraction of valid time in warning",
        ylabel="Event sensitivity",
        xlim=(-0.01, 1.01),
        ylim=(-0.02, 1.02),
    )
    warning_axis.grid(alpha=0.3)
    figure.suptitle("Baseline EEGNet validation operating curve")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """Generate validation scores and report event-level alarm performance."""
    arguments = parse_arguments()
    validate_arguments(arguments)
    config = resolve_label_definition(arguments)
    config.validate()
    set_seed(arguments.seed)
    device = resolve_device(arguments.device)
    model, model_config, checkpoint = load_baseline(
        arguments.checkpoint,
        device,
    )
    validate_embedding_cache(
        arguments.cache_dir,
        arguments.checkpoint,
        model_config,
    )

    manifest_path = config.manifests_dir / "processed_shard_manifest.csv"
    validation_examples = load_decision_examples(
        manifest_path,
        split="validation",
        negative_to_positive_ratio=None,
        seed=arguments.seed,
        project_root=config.project_root,
    )
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    elif device.type == "mps":
        print("GPU: Apple Metal (MPS)")
    print(f"Validation decisions: {len(validation_examples):,}")
    print("Generating checkpoint-matched cached predictions...")
    predictions = generate_validation_predictions(
        model,
        model_config,
        validation_examples,
        preprocessing_config=config,
        cache_dir=arguments.cache_dir,
        batch_size=arguments.batch_size,
        num_workers=arguments.num_workers,
        device=device,
    )

    target_ids = set(
        predictions.loc[
            predictions["label"] == 1,
            "target_seizure_id",
        ].dropna().astype(str)
    )
    seizure_manifest_path = config.manifests_dir / "seizure_manifest.csv"
    seizure_manifest = pd.read_csv(
        seizure_manifest_path,
        dtype={"subject": str},
    )
    target_seizures = prepare_target_seizures(seizure_manifest, target_ids)
    validate_positive_timing(predictions, target_seizures)

    validation_ap = float(
        average_precision_score(
            predictions["label"].to_numpy(dtype=np.int64),
            predictions["probability"].to_numpy(dtype=np.float64),
        )
    )
    thresholds = threshold_grid(
        predictions["probability"],
        arguments.number_of_thresholds,
    )
    horizon_seconds = config.prediction_horizon_minutes * 60.0
    occurrence_seconds = config.seizure_occurrence_period_minutes * 60.0
    refractory_seconds = arguments.refractory_minutes * 60.0
    print(
        f"Validation AP: {validation_ap:.6f}; "
        f"target seizures: {len(target_seizures)}; "
        f"thresholds: {len(thresholds)}"
    )
    print("Sweeping event-level alarm thresholds...")
    sweep_rows: list[dict[str, float | int]] = []
    for threshold_index, threshold in enumerate(thresholds, start=1):
        evaluation = evaluate_alarm_threshold(
            predictions,
            target_seizures,
            threshold=float(threshold),
            prediction_horizon_seconds=horizon_seconds,
            occurrence_period_seconds=occurrence_seconds,
            refractory_seconds=refractory_seconds,
            decision_stride_seconds=config.input_stride_seconds,
        )
        sweep_rows.append(evaluation.metrics)
        if threshold_index % 25 == 0 or threshold_index == len(thresholds):
            print(f"Evaluated {threshold_index}/{len(thresholds)} thresholds")
    sweep = pd.DataFrame(sweep_rows).sort_values(
        "threshold",
        ascending=False,
    ).reset_index(drop=True)

    selected_row = select_for_false_alarm_budget(
        sweep,
        arguments.operating_false_alarm_budget,
    )
    selected_evaluation = evaluate_alarm_threshold(
        predictions,
        target_seizures,
        threshold=float(selected_row["threshold"]),
        prediction_horizon_seconds=horizon_seconds,
        occurrence_period_seconds=occurrence_seconds,
        refractory_seconds=refractory_seconds,
        decision_stride_seconds=config.input_stride_seconds,
    )
    bootstrap_intervals = bootstrap_patient_metrics(
        selected_evaluation.per_subject,
        samples=arguments.bootstrap_samples,
        seed=arguments.seed,
    )

    budget_results = {
        str(budget): row_to_dict(
            select_for_false_alarm_budget(sweep, budget)
        )
        for budget in arguments.reported_false_alarm_budgets
    }
    target_results: dict[str, dict[str, float | int] | None] = {}
    for target in arguments.target_sensitivities:
        row = select_for_target_sensitivity(sweep, target)
        target_results[str(target)] = (
            row_to_dict(row) if row is not None else None
        )

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = arguments.output_dir / "validation_predictions.csv"
    sweep_path = arguments.output_dir / "threshold_sweep.csv"
    event_path = arguments.output_dir / "selected_seizure_events.csv"
    episode_path = arguments.output_dir / "selected_alarm_episodes.csv"
    patient_path = arguments.output_dir / "selected_per_patient.csv"
    summary_path = arguments.output_dir / "event_summary.json"
    curve_path = arguments.output_dir / "event_operating_curve.png"
    predictions.to_csv(predictions_path, index=False)
    sweep.to_csv(sweep_path, index=False)
    selected_evaluation.seizure_events.to_csv(event_path, index=False)
    selected_evaluation.alarm_episodes.to_csv(episode_path, index=False)
    selected_evaluation.per_subject.to_csv(patient_path, index=False)
    save_operating_curve(sweep, selected_row, curve_path)

    summary = {
        "model": "BaselineEEGNet",
        "split": "validation",
        "held_out_test_used": False,
        "checkpoint": str(arguments.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(arguments.checkpoint),
        "best_training_epoch": checkpoint.get("best_epoch"),
        "model_config": asdict(model_config),
        "validation": {
            "decisions": int(len(predictions)),
            "positive_decisions": int((predictions["label"] == 1).sum()),
            "negative_decisions": int((predictions["label"] == 0).sum()),
            "target_seizures": int(len(target_seizures)),
            "patients": int(predictions["subject"].nunique()),
            "average_precision": validation_ap,
        },
        "alarm_definition": {
            "prediction_horizon_minutes": config.prediction_horizon_minutes,
            "occurrence_period_minutes": (
                config.seizure_occurrence_period_minutes
            ),
            "refractory_minutes": arguments.refractory_minutes,
            "false_alarm_denominator": (
                "valid negative-decision exposure at the configured stride"
            ),
            "time_in_warning_denominator": "all valid decision exposure",
        },
        "detailed_operating_budget_false_alarms_per_24h": (
            arguments.operating_false_alarm_budget
        ),
        "detailed_operating_point": selected_evaluation.metrics,
        "patient_bootstrap_95_intervals": bootstrap_intervals,
        "best_points_within_false_alarm_budgets": budget_results,
        "lowest_false_alarm_points_meeting_sensitivity_targets": target_results,
        "bootstrap_samples": arguments.bootstrap_samples,
        "seed": arguments.seed,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    selected = selected_evaluation.metrics
    print()
    print(
        "Detailed validation operating point "
        f"(budget <= {arguments.operating_false_alarm_budget:g} false alarms/24h):"
    )
    print(f"  threshold: {selected['threshold']:.8f}")
    print(
        "  seizures detected: "
        f"{selected['detected_seizures']}/{selected['total_seizures']}"
    )
    print(f"  event sensitivity: {selected['event_sensitivity']:.3f}")
    print(
        "  false alarm episodes/24h: "
        f"{selected['false_alarms_per_24h']:.3f}"
    )
    print(
        "  time in warning: "
        f"{100.0 * selected['time_in_warning_fraction']:.2f}%"
    )
    print(f"Summary: {summary_path}")
    print(f"Threshold sweep: {sweep_path}")
    print(f"Per-seizure results: {event_path}")
    print(f"Per-patient results: {patient_path}")
    print(f"Operating curve: {curve_path}")
    print("The held-out test split was not used.")


if __name__ == "__main__":
    main()
