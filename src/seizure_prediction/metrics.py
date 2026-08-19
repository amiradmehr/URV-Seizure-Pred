"""One shared definition of the binary seizure-risk evaluation metrics.

Both `train_eegnet_baseline.py` (validation, every epoch) and
`test_eegnet_baseline.py` (held-out test, once) score predictions through
`summarize_binary_predictions`, so a validation number and a test number always
mean exactly the same thing and can be read side by side. Defining them twice
invites the two from drifting apart, at which point comparing them is
misleading rather than informative.

Choosing metrics for this task
------------------------------

The decision labels are extremely imbalanced -- roughly 0.1% positive on
SeizeIT2 -- which rules out accuracy (a model predicting "no seizure" always
scores 99.9%) and makes several conventions actively misleading:

* **Average precision** stays the primary metric, and remains what model
  selection optimizes. It has no fixed baseline, though: a useless model scores
  about the positive rate, so `prevalence` and
  `average_precision_lift_over_prevalence` are reported next to it. AP of 0.05
  sounds terrible but is a 50x lift at 0.1% prevalence.
* **ROC AUC** is threshold-free and easy to read, but is well known to look
  flattering under heavy imbalance, because the enormous negative pool makes
  the false-positive rate move very little. Reported for comparability with
  other work, not as the metric to rank on.
* **Brier score** measures calibration rather than ranking (lower is better).
  Worth watching here because the probabilities are prior-corrected for
  negative undersampling, and that correction is exactly the kind of thing that
  can silently go wrong without changing the ranking at all.
* **Operating-point metrics** (F1, precision, recall, specificity) need a
  threshold. It is chosen as the one maximizing F1 on the data being scored,
  which is optimistic -- it is the best case for that set, not a threshold
  fixed in advance -- so it is reported alongside the threshold-free numbers
  rather than instead of them.
* **Recall at a capped false-positive rate** answers the question a clinician
  actually asks: if we accept this false-alarm rate, what fraction of seizures
  do we catch? Reported at 5% and 10% FPR.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def _recall_at_false_positive_rate(
    labels: np.ndarray,
    probabilities: np.ndarray,
    maximum_false_positive_rate: float,
) -> float:
    """Best recall achievable without exceeding a false-positive-rate budget.

    Returns 0.0 when no threshold stays within the budget.
    """
    false_positive_rate, true_positive_rate, _ = roc_curve(labels, probabilities)
    affordable = false_positive_rate <= maximum_false_positive_rate
    if not affordable.any():
        return 0.0
    return float(true_positive_rate[affordable].max())


def summarize_binary_predictions(
    labels: np.ndarray,
    probabilities: np.ndarray,
    loss: float | None = None,
    *,
    prefix: str = "",
) -> dict[str, float | int]:
    """Score pooled binary predictions.

    `prefix` is prepended to every key (e.g. "validation_" or "test_") so one
    run's validation and test numbers can live in the same record without
    colliding. `loss` is included as `<prefix>loss` when given; it is passed in
    rather than computed here because the caller already accumulates it over
    batches with the same criterion training used.

    Metrics that need both classes present return NaN when only one is, so a
    degenerate slice degrades to a missing number instead of an exception.
    """
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    both_classes_present = len(np.unique(labels)) == 2

    positive_count = int((labels == 1).sum())
    negative_count = int((labels == 0).sum())
    prevalence = positive_count / len(labels) if len(labels) else float("nan")

    summary: dict[str, float | int] = {
        f"{prefix}decisions": int(len(labels)),
        f"{prefix}positive_decisions": positive_count,
        f"{prefix}negative_decisions": negative_count,
        f"{prefix}prevalence": prevalence,
    }
    if loss is not None:
        summary[f"{prefix}loss"] = float(loss)

    if not both_classes_present:
        for name in (
            "average_precision",
            "average_precision_lift_over_prevalence",
            "roc_auc",
            "brier_score",
            "best_f1",
            "best_f1_threshold",
            "precision_at_best_f1",
            "recall_at_best_f1",
            "specificity_at_best_f1",
            "balanced_accuracy_at_best_f1",
            "recall_at_5pct_false_positive_rate",
            "recall_at_10pct_false_positive_rate",
        ):
            summary[f"{prefix}{name}"] = float("nan")
        return summary

    average_precision = float(average_precision_score(labels, probabilities))
    summary[f"{prefix}average_precision"] = average_precision
    summary[f"{prefix}average_precision_lift_over_prevalence"] = (
        average_precision / prevalence if prevalence > 0 else float("nan")
    )
    summary[f"{prefix}roc_auc"] = float(roc_auc_score(labels, probabilities))
    summary[f"{prefix}brier_score"] = float(brier_score_loss(labels, probabilities))

    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    # precision_recall_curve returns one more precision/recall point than it
    # does thresholds, so drop that trailing point to keep the arrays aligned.
    precision, recall = precision[:-1], recall[:-1]
    f1_scores = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )

    if len(f1_scores):
        best = int(np.argmax(f1_scores))
        best_threshold = float(thresholds[best])
        predicted_positive = probabilities >= best_threshold
        true_negative = int(((labels == 0) & ~predicted_positive).sum())
        specificity = true_negative / negative_count if negative_count else float("nan")

        summary[f"{prefix}best_f1"] = float(f1_scores[best])
        summary[f"{prefix}best_f1_threshold"] = best_threshold
        summary[f"{prefix}precision_at_best_f1"] = float(precision[best])
        summary[f"{prefix}recall_at_best_f1"] = float(recall[best])
        summary[f"{prefix}specificity_at_best_f1"] = specificity
        summary[f"{prefix}balanced_accuracy_at_best_f1"] = float(
            (recall[best] + specificity) / 2.0
        )
    else:
        for name in (
            "best_f1",
            "best_f1_threshold",
            "precision_at_best_f1",
            "recall_at_best_f1",
            "specificity_at_best_f1",
            "balanced_accuracy_at_best_f1",
        ):
            summary[f"{prefix}{name}"] = float("nan")

    summary[f"{prefix}recall_at_5pct_false_positive_rate"] = (
        _recall_at_false_positive_rate(labels, probabilities, 0.05)
    )
    summary[f"{prefix}recall_at_10pct_false_positive_rate"] = (
        _recall_at_false_positive_rate(labels, probabilities, 0.10)
    )
    return summary


#: Metrics worth plotting per epoch, as (metric name, axis label, fixed y-range).
EPOCH_CURVE_METRICS: tuple[tuple[str, str, tuple[float, float] | None], ...] = (
    ("average_precision", "Average precision", (0.0, 1.0)),
    ("roc_auc", "ROC AUC", (0.0, 1.0)),
    ("best_f1", "Best F1", (0.0, 1.0)),
)
