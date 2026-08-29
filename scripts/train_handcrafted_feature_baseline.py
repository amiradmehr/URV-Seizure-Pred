r"""Train and event-evaluate a feature-only AVA-style risk baseline.

The representation contains no learned EEG encoder. A regularized logistic
regression receives robust minute-level handcrafted features summarized across
the same configured input-window histories used by EEGNet. Training exposure uses the same
patient/event-balanced sampler as the current best EEGNet experiment, while
validation remains complete, patient-held-out, and at natural prevalence.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIRECTORY = PROJECT_ROOT / "src"
if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

from evaluate_eegnet_events import (  # noqa: E402
    row_to_dict,
    select_for_false_alarm_budget,
    select_for_target_sensitivity,
    validate_positive_timing,
)
from seizure_prediction.config import (  # noqa: E402
    CONFIG,
    add_label_definition_arguments,
    resolve_label_definition,
)
from seizure_prediction.datasets import (  # noqa: E402
    PatientEventBalancedEpochSampler,
    load_decision_examples,
)
from seizure_prediction.event_evaluation import (  # noqa: E402
    bootstrap_patient_metrics,
    evaluate_alarm_threshold,
    prepare_target_seizures,
    threshold_grid,
)
from seizure_prediction.handcrafted_features import (  # noqa: E402
    FEATURE_NAMES,
    SUMMARY_NAMES,
    build_decision_feature_matrix,
    decision_feature_names,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=(
            CONFIG.handcrafted_feature_cache_dir
            / "ava_minute_selected_v1"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "outputs"
            / "models"
            / "handcrafted_ava_logistic_patient_event_balanced"
        ),
    )
    parser.add_argument("--sampling-epochs", type=int, default=5)
    parser.add_argument("--negative-to-positive-ratio", type=float, default=10.0)
    parser.add_argument("--max-events-per-patient", type=int, default=4)
    parser.add_argument("--regularization-c", type=float, default=1.0)
    parser.add_argument("--max-iterations", type=int, default=2000)
    parser.add_argument(
        "--classifier",
        choices=("logistic", "hist-gradient-boosting"),
        default="logistic",
    )
    parser.add_argument("--number-of-thresholds", type=int, default=201)
    parser.add_argument("--refractory-minutes", type=float, default=10.0)
    parser.add_argument("--operating-false-alarm-budget", type=float, default=1.0)
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
    add_label_definition_arguments(parser)
    return parser.parse_args()


def validate_arguments(arguments: argparse.Namespace) -> None:
    if arguments.sampling_epochs <= 0:
        raise ValueError("sampling-epochs must be positive.")
    if arguments.negative_to_positive_ratio <= 0.0:
        raise ValueError("negative-to-positive-ratio must be positive.")
    if arguments.max_events_per_patient <= 0:
        raise ValueError("max-events-per-patient must be positive.")
    if arguments.regularization_c <= 0.0 or arguments.max_iterations <= 0:
        raise ValueError("Classifier regularization and iterations must be positive.")
    if arguments.number_of_thresholds < 2 or arguments.bootstrap_samples <= 0:
        raise ValueError("Threshold and bootstrap counts are invalid.")


def sampling_prior_logit_correction(
    population_positive_fraction: float,
    sampled_positive_fraction: float,
) -> float:
    if not 0.0 < population_positive_fraction < 1.0:
        raise ValueError("Population prevalence must be strictly between zero and one.")
    if not 0.0 < sampled_positive_fraction < 1.0:
        raise ValueError("Sampled prevalence must be strictly between zero and one.")
    return float(
        math.log(population_positive_fraction / (1.0 - population_positive_fraction))
        - math.log(sampled_positive_fraction / (1.0 - sampled_positive_fraction))
    )


def corrected_probabilities(
    model: Pipeline,
    features: np.ndarray,
    correction: float,
) -> np.ndarray:
    raw = model.predict_proba(features)[:, 1]
    epsilon = np.finfo(np.float64).eps
    raw = np.clip(raw, epsilon, 1.0 - epsilon)
    logits = np.log(raw) - np.log1p(-raw) + correction
    return 1.0 / (1.0 + np.exp(-logits))


def save_operating_curve(
    sweep: pd.DataFrame,
    selected: pd.Series,
    output_path: Path,
) -> None:
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
    figure.suptitle("AVA handcrafted-feature validation operating curve")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    arguments = parse_arguments()
    validate_arguments(arguments)
    config = resolve_label_definition(arguments)
    config.validate()
    random.seed(arguments.seed)
    np.random.seed(arguments.seed)
    cache_metadata_path = arguments.cache_dir / "cache_metadata.json"
    if not cache_metadata_path.exists():
        raise FileNotFoundError(
            f"Missing feature cache metadata: {cache_metadata_path}. "
            "Run scripts/cache_handcrafted_features.py first."
        )
    cache_metadata = json.loads(cache_metadata_path.read_text(encoding="utf-8"))
    cached_splits = set(cache_metadata.get("splits", []))
    if not {"train", "validation"}.issubset(cached_splits):
        raise ValueError("Feature cache must contain train and validation splits.")

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
    sampler = PatientEventBalancedEpochSampler(
        train_examples,
        negative_to_positive_ratio=arguments.negative_to_positive_ratio,
        max_events_per_patient=arguments.max_events_per_patient,
        seed=arguments.seed,
    )
    sampled_indices = np.concatenate(
        [sampler.indices_for_epoch(epoch) for epoch in range(arguments.sampling_epochs)]
    )
    sampled_examples = train_examples.iloc[sampled_indices].reset_index(drop=True)
    train_labels = sampled_examples["label"].to_numpy(dtype=np.int64)
    validation_labels = validation_examples["label"].to_numpy(dtype=np.int64)
    population_fraction = float(train_examples["label"].mean())
    sampled_fraction = float(train_labels.mean())
    correction = sampling_prior_logit_correction(population_fraction, sampled_fraction)

    print(f"Sampled training decisions: {len(sampled_examples):,}")
    print(f"Validation decisions: {len(validation_examples):,}")
    print("Building sampled training feature matrix...")
    history_minutes = int(round(config.input_window_seconds / 60.0))
    train_features = build_decision_feature_matrix(
        sampled_examples,
        arguments.cache_dir,
        sampling_frequency=config.target_sfreq,
        history_minutes=history_minutes,
    )
    print("Building complete validation feature matrix...")
    validation_features = build_decision_feature_matrix(
        validation_examples,
        arguments.cache_dir,
        sampling_frequency=config.target_sfreq,
        history_minutes=history_minutes,
    )
    names = decision_feature_names(config.canonical_channel_names)
    if train_features.shape[1] != len(names):
        raise RuntimeError("Feature matrix width does not match feature names.")

    if arguments.classifier == "logistic":
        model = Pipeline(
            [
                ("standardizer", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=arguments.regularization_c,
                        max_iter=arguments.max_iterations,
                        solver="lbfgs",
                        random_state=arguments.seed,
                    ),
                ),
            ]
        )
        model_name = "AVAHandcraftedLogisticRegression"
    else:
        model = Pipeline(
            [
                (
                    "classifier",
                    HistGradientBoostingClassifier(
                        learning_rate=0.05,
                        max_iter=min(arguments.max_iterations, 300),
                        max_leaf_nodes=15,
                        min_samples_leaf=20,
                        l2_regularization=1.0,
                        early_stopping=True,
                        validation_fraction=0.15,
                        random_state=arguments.seed,
                    ),
                )
            ]
        )
        model_name = "AVAHandcraftedHistGradientBoosting"
    print(f"Fitting {model_name}...")
    model.fit(train_features, train_labels)
    train_probabilities = corrected_probabilities(model, train_features, correction)
    validation_probabilities = corrected_probabilities(
        model, validation_features, correction
    )
    train_ap = float(average_precision_score(train_labels, train_probabilities))
    validation_ap = float(
        average_precision_score(validation_labels, validation_probabilities)
    )
    validation_loss = float(
        log_loss(validation_labels, validation_probabilities, labels=[0, 1])
    )
    print(f"Sampled train AP: {train_ap:.6f}")
    print(f"Validation AP: {validation_ap:.6f}")

    prediction_columns = [
        "recording_id",
        "subject",
        "decision_time_seconds",
        "prediction_start_seconds",
        "prediction_stop_seconds",
        "label",
        "target_seizure_id",
    ]
    predictions = validation_examples[prediction_columns].copy()
    predictions["probability"] = validation_probabilities
    target_ids = set(
        predictions.loc[predictions["label"] == 1, "target_seizure_id"]
        .dropna()
        .astype(str)
    )
    seizure_manifest = pd.read_csv(
        config.manifests_dir / "seizure_manifest.csv",
        dtype={"subject": str},
    )
    target_seizures = prepare_target_seizures(seizure_manifest, target_ids)
    validate_positive_timing(predictions, target_seizures)
    thresholds = threshold_grid(
        predictions["probability"], arguments.number_of_thresholds
    )
    horizon_seconds = config.prediction_horizon_minutes * 60.0
    occurrence_seconds = config.seizure_occurrence_period_minutes * 60.0
    refractory_seconds = arguments.refractory_minutes * 60.0
    print(f"Sweeping {len(thresholds)} event thresholds...")
    sweep_rows: list[dict[str, float | int]] = []
    for index, threshold in enumerate(thresholds, start=1):
        result = evaluate_alarm_threshold(
            predictions,
            target_seizures,
            threshold=float(threshold),
            prediction_horizon_seconds=horizon_seconds,
            occurrence_period_seconds=occurrence_seconds,
            refractory_seconds=refractory_seconds,
            decision_stride_seconds=config.input_stride_seconds,
        )
        sweep_rows.append(result.metrics)
        if index % 25 == 0 or index == len(thresholds):
            print(f"Evaluated {index}/{len(thresholds)} thresholds", flush=True)
    sweep = pd.DataFrame(sweep_rows).sort_values(
        "threshold", ascending=False
    ).reset_index(drop=True)
    selected_row = select_for_false_alarm_budget(
        sweep, arguments.operating_false_alarm_budget
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
        str(budget): row_to_dict(select_for_false_alarm_budget(sweep, budget))
        for budget in arguments.reported_false_alarm_budgets
    }
    target_results: dict[str, dict[str, float | int] | None] = {}
    for target in arguments.target_sensitivities:
        row = select_for_target_sensitivity(sweep, target)
        target_results[str(target)] = row_to_dict(row) if row is not None else None

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(arguments.output_dir / "validation_predictions.csv", index=False)
    sweep.to_csv(arguments.output_dir / "threshold_sweep.csv", index=False)
    selected_evaluation.seizure_events.to_csv(
        arguments.output_dir / "selected_seizure_events.csv", index=False
    )
    selected_evaluation.alarm_episodes.to_csv(
        arguments.output_dir / "selected_alarm_episodes.csv", index=False
    )
    selected_evaluation.per_subject.to_csv(
        arguments.output_dir / "selected_per_patient.csv", index=False
    )
    save_operating_curve(
        sweep,
        selected_row,
        arguments.output_dir / "event_operating_curve.png",
    )
    joblib.dump(model, arguments.output_dir / "model.joblib")
    classifier = model.named_steps["classifier"]
    if hasattr(classifier, "coef_"):
        coefficients = classifier.coef_.reshape(-1)
        pd.DataFrame(
            {"feature": names, "standardized_coefficient": coefficients}
        ).assign(absolute_coefficient=np.abs(coefficients)).sort_values(
            "absolute_coefficient", ascending=False
        ).to_csv(arguments.output_dir / "feature_coefficients.csv", index=False)

    summary = {
        "model": model_name,
        "representation": {
            "minute_feature_names": list(FEATURE_NAMES),
            "history_summary_names": list(SUMMARY_NAMES),
            "history_minutes": history_minutes,
            "recent_and_early_minutes": 5,
            "classifier_features": len(names),
            "learned_raw_eeg_encoder": False,
            "channel_availability_included": True,
        },
        "split": "validation",
        "held_out_test_used": False,
        "training": {
            "sampling_strategy": "patient-event-balanced",
            "sampling_epochs": arguments.sampling_epochs,
            "sampled_decisions_with_repeats": int(len(sampled_examples)),
            "sampled_positive_decisions_with_repeats": int(train_labels.sum()),
            "sampled_negative_decisions_with_repeats": int((train_labels == 0).sum()),
            "unique_sampled_decisions": int(len(np.unique(sampled_indices))),
            "population_positive_fraction": population_fraction,
            "sampled_positive_fraction": sampled_fraction,
            "sampling_prior_logit_correction": correction,
            "regularization_c": arguments.regularization_c,
            "max_iterations": arguments.max_iterations,
            "classifier": arguments.classifier,
            "sampled_train_average_precision": train_ap,
        },
        "validation": {
            "decisions": int(len(predictions)),
            "positive_decisions": int(validation_labels.sum()),
            "negative_decisions": int((validation_labels == 0).sum()),
            "target_seizures": int(len(target_seizures)),
            "patients": int(predictions["subject"].nunique()),
            "average_precision": validation_ap,
            "binary_cross_entropy": validation_loss,
        },
        "alarm_definition": {
            "prediction_horizon_minutes": config.prediction_horizon_minutes,
            "occurrence_period_minutes": config.seizure_occurrence_period_minutes,
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
    summary_path = arguments.output_dir / "event_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    selected = selected_evaluation.metrics
    print()
    print(
        f"Selected <= {arguments.operating_false_alarm_budget:g} false alarms/24h:"
    )
    print(
        f"  detected {selected['detected_seizures']}/{selected['total_seizures']} "
        f"seizures; sensitivity={selected['event_sensitivity']:.3f}"
    )
    print(
        f"  false alarms/24h={selected['false_alarms_per_24h']:.3f}; "
        f"time in warning={100.0 * selected['time_in_warning_fraction']:.2f}%"
    )
    print(f"Summary: {summary_path}")
    print("The held-out test split was not used.")


if __name__ == "__main__":
    main()
