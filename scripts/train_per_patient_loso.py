r"""Per-patient seizure-risk models with leave-one-recording-out validation.

This is a diagnostic, not a deployment candidate.  It asks whether a
*patient-specific* preictal signature exists at all, by removing the
cross-patient generalization burden entirely: one model per patient, trained
and evaluated only on that patient's own EEG.

Interpretation, decided before running
--------------------------------------
* If per-patient models clear their own prevalence by a wide margin, the
  pipeline is sound and cross-patient transfer was the blocker.  That would be
  the most important result in the project.
* If they do not, the result is **ambiguous** on its own.  A null here is
  consistent with a broken pipeline, with no preictal signal, and with the
  event counts simply being too small.  It does not, by itself, convict the
  pipeline.  Only a positive control on a task with a known answer -- seizure
  detection -- can do that.

Holdout scheme
--------------
The default holds out one whole **recording** per fold.  Decisions inside a
recording overlap heavily and share electrode seating and vigilance state, so
training on negatives from the same recording as the held-out seizure lets a
model memorize the recording rather than learn a preictal state.  That is a
well-known source of inflated seizure-prediction results.

``--holdout seizure`` reproduces the looser leave-one-seizure-out scheme, which
keeps same-recording negatives in training.  Running both is informative: the
gap between them measures how much apparent skill is recording memorization.

Every reported score is accompanied by a within-patient label-permutation null,
because with a handful of events per patient the sampling noise on average
precision is large enough that a bare number cannot be read.

PowerShell example:

    .\.venv-old\Scripts\python.exe scripts\train_per_patient_loso.py
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIRECTORY = PROJECT_ROOT / "src"
if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

from seizure_prediction.config import (  # noqa: E402
    CONFIG,
    add_label_definition_arguments,
    resolve_label_definition,
)
from seizure_prediction.datasets import load_decision_examples  # noqa: E402
from seizure_prediction.handcrafted_features import (  # noqa: E402
    build_decision_feature_matrix,
    decision_feature_names,
)


SEPARATOR = "=" * 90


@dataclass(frozen=True)
class PatientResult:
    """Pooled out-of-fold performance for one patient."""

    subject: str
    split: str
    events: int
    folds: int
    decisions: int
    positives: int
    prevalence: float
    average_precision: float
    lift_over_prevalence: float
    null_median_average_precision: float
    null_p95_average_precision: float
    permutation_p_value: float
    beats_null: bool


def parse_arguments() -> argparse.Namespace:
    """Parse the diagnostic's options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=(
            CONFIG.handcrafted_feature_cache_dir
            / "ava_minute_selected_v1"
        ),
        help="Handcrafted minute-feature cache with complete per-patient coverage.",
    )
    parser.add_argument(
        "--minimum-events",
        type=int,
        default=5,
        help=(
            "Patients with fewer eligible seizures than this are skipped. Below "
            "about five events a per-patient estimate is not interpretable."
        ),
    )
    parser.add_argument(
        "--holdout",
        choices=("recording", "seizure"),
        default="recording",
        help=(
            "recording: hold out the whole recording containing the target "
            "seizure (conservative). seizure: hold out only that seizure's "
            "windows, keeping same-recording negatives in training (optimistic)."
        ),
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "validation"),
        default=("train", "validation"),
        help="The held-out test split is deliberately unavailable here.",
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=100,
        help="Within-patient label shuffles used to build the null distribution.",
    )
    parser.add_argument(
        "--regularization-c",
        type=float,
        default=1.0,
        help="Inverse L2 strength for the per-patient logistic regression.",
    )
    parser.add_argument("--max-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "analysis" / "per_patient_loso",
    )
    add_label_definition_arguments(parser)
    return parser.parse_args()


def validate_arguments(arguments: argparse.Namespace) -> None:
    """Reject settings that cannot produce an interpretable result."""
    if arguments.minimum_events < 2:
        raise ValueError("minimum-events must be at least two to form folds.")
    if arguments.permutations < 0:
        raise ValueError("permutations cannot be negative.")
    if arguments.regularization_c <= 0:
        raise ValueError("regularization-c must be positive.")
    if arguments.max_iterations <= 0:
        raise ValueError("max-iterations must be positive.")
    if not arguments.cache_dir.exists():
        raise FileNotFoundError(
            f"Feature cache not found: {arguments.cache_dir}. Run "
            "scripts/cache_handcrafted_features.py --complete-subjects ... first."
        )
    stale_marker = arguments.cache_dir / "STALE.txt"
    if stale_marker.exists():
        raise RuntimeError(
            f"{stale_marker} is present: this cache predates the current "
            "data/processed normalization. Regenerate it with "
            "scripts/cache_handcrafted_features.py --overwrite before trusting "
            "any result from it."
        )


def load_all_examples(
    splits: tuple[str, ...],
    seed: int,
    config,
) -> pd.DataFrame:
    """Load every decision from the requested splits, at natural prevalence."""
    manifest_path = config.manifests_dir / "processed_shard_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Processed manifest not found: {manifest_path}")

    frames = []
    for split in splits:
        split_examples = load_decision_examples(
            manifest_path,
            split=split,
            negative_to_positive_ratio=None,
            seed=seed,
            project_root=config.project_root,
        )
        split_examples["split"] = split
        frames.append(split_examples)

    examples = pd.concat(frames, ignore_index=True)
    examples["subject"] = examples["subject"].astype(str).str.zfill(3)
    examples["recording_id"] = examples["recording_id"].astype(str)
    return examples


def eligible_patients(
    examples: pd.DataFrame,
    minimum_events: int,
) -> list[str]:
    """Return patients with enough distinct seizures to support folding."""
    positives = examples[examples["label"] == 1]
    events = positives.groupby("subject")["target_seizure_id"].nunique()
    return sorted(events[events >= minimum_events].index.astype(str))


def build_folds(
    patient_examples: pd.DataFrame,
    holdout: str,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Return ``(fold_name, train_positions, test_positions)`` for one patient.

    Positions index into ``patient_examples`` after ``reset_index(drop=True)``.
    """
    positives = patient_examples[patient_examples["label"] == 1]
    event_ids = sorted(positives["target_seizure_id"].astype(str).unique())
    folds: list[tuple[str, np.ndarray, np.ndarray]] = []
    positions = np.arange(len(patient_examples))

    if holdout == "recording":
        # One fold per recording that contains at least one target seizure.
        recordings = sorted(
            positives["recording_id"].astype(str).unique()
        )
        for recording_id in recordings:
            test_mask = (
                patient_examples["recording_id"].astype(str) == recording_id
            ).to_numpy()
            train_mask = ~test_mask
            folds.append(
                (recording_id, positions[train_mask], positions[test_mask])
            )
    else:
        for event_id in event_ids:
            test_mask = (
                patient_examples["target_seizure_id"].astype(str) == event_id
            ).to_numpy()
            # A seizure's own negatives cannot be assigned to it, so the test
            # side holds that event's positive windows plus a disjoint slice of
            # this patient's negatives, chosen deterministically by position.
            negative_positions = positions[
                (patient_examples["label"] == 0).to_numpy()
            ]
            share = np.array_split(negative_positions, len(event_ids))
            index = event_ids.index(event_id)
            test_positions = np.concatenate([positions[test_mask], share[index]])
            train_positions = np.setdiff1d(positions, test_positions)
            folds.append((event_id, train_positions, test_positions))

    return folds


def usable_folds(
    folds: list[tuple[str, np.ndarray, np.ndarray]],
    labels: np.ndarray,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Drop folds that cannot train or cannot be scored."""
    usable = []
    for name, train_positions, test_positions in folds:
        train_labels = labels[train_positions]
        test_labels = labels[test_positions]
        if train_labels.sum() == 0 or (train_labels == 0).sum() == 0:
            continue
        if test_labels.sum() == 0:
            continue
        usable.append((name, train_positions, test_positions))
    return usable


def out_of_fold_predictions(
    features: np.ndarray,
    labels: np.ndarray,
    folds: list[tuple[str, np.ndarray, np.ndarray]],
    *,
    regularization_c: float,
    max_iterations: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Fit one model per fold and return pooled held-out scores and labels."""
    scores: list[np.ndarray] = []
    held_out_labels: list[np.ndarray] = []
    fold_names: list[str] = []

    for name, train_positions, test_positions in folds:
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "logistic",
                    LogisticRegression(
                        C=regularization_c,
                        max_iter=max_iterations,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        )
        # The scaler is fitted on the training fold only.
        model.fit(features[train_positions], labels[train_positions])
        scores.append(model.predict_proba(features[test_positions])[:, 1])
        held_out_labels.append(labels[test_positions])
        fold_names.extend([name] * len(test_positions))

    return (
        np.concatenate(scores),
        np.concatenate(held_out_labels),
        fold_names,
    )


def permutation_null(
    features: np.ndarray,
    labels: np.ndarray,
    folds: list[tuple[str, np.ndarray, np.ndarray]],
    *,
    permutations: int,
    regularization_c: float,
    max_iterations: int,
    seed: int,
) -> np.ndarray:
    """Return average precisions from within-patient label shuffles.

    Labels are permuted across the patient's own decisions, so the null keeps
    the patient's feature distribution, class balance, and fold structure and
    destroys only the association with seizure timing.
    """
    if permutations == 0:
        return np.empty(0, dtype=np.float64)

    rng = np.random.default_rng(seed)
    null_scores = np.empty(permutations, dtype=np.float64)
    for index in range(permutations):
        shuffled = rng.permutation(labels)
        usable = usable_folds(folds, shuffled)
        if not usable:
            null_scores[index] = np.nan
            continue
        scores, held_out, _ = out_of_fold_predictions(
            features,
            shuffled,
            usable,
            regularization_c=regularization_c,
            max_iterations=max_iterations,
            seed=seed,
        )
        null_scores[index] = average_precision_score(held_out, scores)
    return null_scores[np.isfinite(null_scores)]


def save_figure(results: pd.DataFrame, output_path: Path) -> None:
    """Plot each patient's lift against their own permutation null."""
    if results.empty:
        return
    ordered = results.sort_values("lift_over_prevalence", ascending=True)
    positions = np.arange(len(ordered))
    figure, axes = plt.subplots(figsize=(10, max(4.0, 0.32 * len(ordered))))

    null_lift = ordered["null_p95_average_precision"] / ordered["prevalence"]
    axes.barh(
        positions,
        null_lift,
        color="0.85",
        label="permutation null, 95th percentile",
    )
    colors = ["#2b7bba" if beats else "#c0c0c0" for beats in ordered["beats_null"]]
    axes.scatter(
        ordered["lift_over_prevalence"],
        positions,
        color=colors,
        zorder=3,
        label="observed",
    )
    axes.axvline(1.0, color="black", linewidth=1.0, label="patient's own prevalence")
    axes.axvline(
        5.0,
        color="#c44e52",
        linestyle="--",
        linewidth=1.0,
        label="pre-registered 5x bar",
    )
    axes.set_yticks(positions)
    axes.set_yticklabels(ordered["subject"])
    axes.set_xlabel("out-of-fold average precision / patient's own prevalence")
    axes.set_ylabel("subject")
    axes.set_title("Per-patient leave-one-recording-out, versus within-patient null")
    axes.legend(loc="lower right", fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def main() -> None:
    """Run the per-patient diagnostic and write an auditable summary."""
    arguments = parse_arguments()
    validate_arguments(arguments)
    config = resolve_label_definition(arguments)
    config.validate()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)

    print(SEPARATOR)
    print("PER-PATIENT LEAVE-ONE-RECORDING-OUT DIAGNOSTIC".center(90))
    print(SEPARATOR)

    examples = load_all_examples(tuple(arguments.splits), arguments.seed, config)
    patients = eligible_patients(examples, arguments.minimum_events)
    if not patients:
        raise ValueError(
            f"No patient has at least {arguments.minimum_events} eligible seizures."
        )
    print(f"Holdout scheme      : {arguments.holdout}")
    print(f"Splits              : {list(arguments.splits)} (test never used)")
    print(f"Eligible patients   : {len(patients)} (>= {arguments.minimum_events} events)")
    print(f"Permutations        : {arguments.permutations}")
    print(f"Feature cache       : {arguments.cache_dir}")

    feature_names = decision_feature_names(config.canonical_channel_names)
    results: list[PatientResult] = []
    fold_frames: list[pd.DataFrame] = []

    for number, subject in enumerate(patients, start=1):
        patient_examples = (
            examples[examples["subject"] == subject]
            .reset_index(drop=True)
        )
        labels = patient_examples["label"].to_numpy(dtype=np.int64)
        folds = usable_folds(
            build_folds(patient_examples, arguments.holdout),
            labels,
        )
        events = int(
            patient_examples.loc[
                patient_examples["label"] == 1, "target_seizure_id"
            ].nunique()
        )
        if len(folds) < 2:
            print(
                f"[{number:03d}/{len(patients):03d}] {subject}: skipped, "
                f"only {len(folds)} usable fold(s)."
            )
            continue

        features = build_decision_feature_matrix(
            patient_examples,
            arguments.cache_dir,
            sampling_frequency=config.target_sfreq,
            history_minutes=int(round(config.input_window_seconds / 60.0)),
        )
        scores, held_out_labels, fold_names = out_of_fold_predictions(
            features,
            labels,
            folds,
            regularization_c=arguments.regularization_c,
            max_iterations=arguments.max_iterations,
            seed=arguments.seed,
        )
        observed = float(average_precision_score(held_out_labels, scores))
        prevalence = float(held_out_labels.mean())
        null_scores = permutation_null(
            features,
            labels,
            folds,
            permutations=arguments.permutations,
            regularization_c=arguments.regularization_c,
            max_iterations=arguments.max_iterations,
            seed=arguments.seed,
        )
        if len(null_scores) > 0:
            null_median = float(np.median(null_scores))
            null_p95 = float(np.quantile(null_scores, 0.95))
            # Add-one smoothing keeps the p-value bounded away from zero.
            p_value = float(
                (1 + int((null_scores >= observed).sum())) / (1 + len(null_scores))
            )
        else:
            null_median = null_p95 = float("nan")
            p_value = float("nan")

        results.append(
            PatientResult(
                subject=subject,
                split=str(patient_examples["split"].iloc[0]),
                events=events,
                folds=len(folds),
                decisions=len(held_out_labels),
                positives=int(held_out_labels.sum()),
                prevalence=prevalence,
                average_precision=observed,
                lift_over_prevalence=observed / prevalence if prevalence > 0 else float("nan"),
                null_median_average_precision=null_median,
                null_p95_average_precision=null_p95,
                permutation_p_value=p_value,
                beats_null=bool(np.isfinite(null_p95) and observed > null_p95),
            )
        )
        fold_frames.append(
            pd.DataFrame(
                {
                    "subject": subject,
                    "fold": fold_names,
                    "label": held_out_labels,
                    "score": scores,
                }
            )
        )
        print(
            f"[{number:03d}/{len(patients):03d}] {subject}: "
            f"{events} events, {len(folds)} folds, "
            f"AP={observed:.4f} (prevalence {prevalence:.4f}, "
            f"lift {observed / prevalence:.2f}x), "
            f"null p95={null_p95:.4f}, p={p_value:.3f}"
        )

    if not results:
        raise RuntimeError("No patient produced a usable per-patient model.")

    frame = pd.DataFrame([asdict(result) for result in results])
    frame.to_csv(arguments.output_dir / "per_patient_results.csv", index=False)
    pd.concat(fold_frames, ignore_index=True).to_csv(
        arguments.output_dir / "fold_predictions.csv",
        index=False,
    )
    save_figure(frame, arguments.output_dir / "per_patient_lift.png")

    beat_null = int(frame["beats_null"].sum())
    cleared_bar = int((frame["lift_over_prevalence"] >= 5.0).sum())
    summary = {
        "diagnostic": "per-patient leave-one-recording-out",
        "holdout": arguments.holdout,
        "splits": list(arguments.splits),
        "held_out_test_used": False,
        "minimum_events": arguments.minimum_events,
        "permutations": arguments.permutations,
        "feature_cache": str(arguments.cache_dir),
        "classifier": "standardized L2 logistic regression, class_weight=balanced",
        "feature_count": len(feature_names),
        "patients_evaluated": len(frame),
        "patients_beating_their_own_null": beat_null,
        "patients_clearing_5x_prevalence": cleared_bar,
        "median_lift_over_prevalence": float(frame["lift_over_prevalence"].median()),
        "median_permutation_p_value": float(frame["permutation_p_value"].median()),
        "pooled_cross_patient_reference_lift": 7.48,
        "seed": arguments.seed,
    }
    (arguments.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print()
    print(SEPARATOR)
    print("RESULT".center(90))
    print(SEPARATOR)
    print(f"Patients evaluated                 : {len(frame)}")
    print(f"Beating their own permutation null : {beat_null} / {len(frame)}")
    print(f"Clearing the 5x prevalence bar     : {cleared_bar} / {len(frame)}")
    print(f"Median lift over prevalence        : {frame['lift_over_prevalence'].median():.2f}x")
    print(f"Pooled cross-patient model         : 7.48x (for comparison)")
    print(f"\nOutputs: {arguments.output_dir}")
    if beat_null <= max(1, int(0.1 * len(frame))):
        print(
            "\nAt this rate the patients beating their null are consistent with "
            "chance. Read this as no detectable patient-specific preictal signal "
            "in these features -- NOT as proof the pipeline is broken. Run the "
            "seizure-detection positive control to separate those."
        )


if __name__ == "__main__":
    main()
