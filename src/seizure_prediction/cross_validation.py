"""Patient-level cross-validation and multi-seed aggregation.

Why this exists
---------------
Model selection on this project ran against a single 12-patient validation
split holding 33 seizures.  Per-epoch validation average precision swings by a
factor of five to six inside one run, so "best epoch" is a maximum over a noisy
series: it is biased upward, it rewards longer runs, and two configurations
cannot be told apart by it.  The global-versus-per-patient normalization
comparison died exactly there.

The fix is to stop reading single numbers.  Folds are cut across the *training*
patients only, so the real validation and test splits stay untouched for final
reporting, and every configuration is described by a mean and a spread across
folds and seeds rather than by its luckiest epoch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PatientFold:
    """One cross-validation fold, split by whole patient."""

    index: int
    train_subjects: tuple[str, ...]
    validation_subjects: tuple[str, ...]
    train_seizures: int
    validation_seizures: int


def seizures_by_subject(examples: pd.DataFrame) -> pd.Series:
    """Return the distinct eligible seizure count for every subject."""
    required = {"subject", "label", "target_seizure_id"}
    missing = required - set(examples.columns)
    if missing:
        raise ValueError(f"Examples are missing columns: {sorted(missing)}")

    subjects = examples["subject"].astype(str).str.zfill(3)
    positives = examples["label"] == 1
    counts = (
        examples.loc[positives]
        .assign(subject=subjects[positives])
        .groupby("subject")["target_seizure_id"]
        .nunique()
    )
    # Subjects with no eligible seizure still contribute negatives, and must
    # appear in exactly one fold rather than being silently dropped.
    all_subjects = pd.Index(sorted(subjects.unique()), name="subject")
    return counts.reindex(all_subjects, fill_value=0).astype(int)


def make_patient_folds(
    examples: pd.DataFrame,
    *,
    folds: int = 5,
    seed: int = 0,
) -> list[PatientFold]:
    """Partition patients into folds balanced on seizure count.

    Seizure counts are severely unequal -- the median patient has two eligible
    seizures and one has sixteen -- so a random split can leave a fold with
    almost no positive events and an uninterpretable score.  Patients are
    therefore dealt out in descending seizure order onto whichever fold is
    currently poorest, which keeps the per-fold event counts close.
    """
    if folds < 2:
        raise ValueError("folds must be at least two.")

    counts = seizures_by_subject(examples)
    if len(counts) < folds:
        raise ValueError(
            f"{len(counts)} patients cannot be split into {folds} folds."
        )

    rng = np.random.default_rng(seed)
    # Shuffle first so ties are broken differently for each seed.
    order = rng.permutation(len(counts))
    shuffled = counts.iloc[order]
    descending = shuffled.sort_values(ascending=False, kind="stable")

    fold_subjects: list[list[str]] = [[] for _ in range(folds)]
    fold_seizures = np.zeros(folds, dtype=np.int64)

    # Patients carrying seizures are dealt onto whichever fold is currently
    # poorest in events, which is what keeps the per-fold event counts close.
    with_seizures = descending[descending > 0]
    for subject, seizure_count in with_seizures.items():
        target = min(
            range(folds),
            key=lambda index: (fold_seizures[index], len(fold_subjects[index])),
        )
        fold_subjects[target].append(str(subject))
        fold_seizures[target] += int(seizure_count)

    # Patients with no eligible seizure never change the event balance, so
    # dealing them by the same rule would pile every one of them into the
    # single poorest fold. They go round-robin on patient count instead, which
    # keeps both the event counts and the training-set sizes even.
    without_seizures = descending[descending == 0]
    for subject in without_seizures.index:
        target = min(range(folds), key=lambda index: len(fold_subjects[index]))
        fold_subjects[target].append(str(subject))

    total_seizures = int(counts.sum())
    result: list[PatientFold] = []
    for index in range(folds):
        validation = tuple(sorted(fold_subjects[index]))
        train = tuple(
            sorted(
                subject
                for other in range(folds)
                if other != index
                for subject in fold_subjects[other]
            )
        )
        result.append(
            PatientFold(
                index=index,
                train_subjects=train,
                validation_subjects=validation,
                train_seizures=total_seizures - int(fold_seizures[index]),
                validation_seizures=int(fold_seizures[index]),
            )
        )
    return result


def split_examples_by_fold(
    examples: pd.DataFrame,
    fold: PatientFold,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(train_examples, validation_examples)`` for one fold."""
    subjects = examples["subject"].astype(str).str.zfill(3)
    validation_mask = subjects.isin(set(fold.validation_subjects))
    train = examples.loc[~validation_mask].reset_index(drop=True)
    validation = examples.loc[validation_mask].reset_index(drop=True)
    if train.empty or validation.empty:
        raise ValueError(f"Fold {fold.index} produced an empty side.")
    for name, side in (("train", train), ("validation", validation)):
        if side["label"].nunique() < 2:
            raise ValueError(
                f"Fold {fold.index} {name} side contains a single class; "
                "increase the fold count or rebalance."
            )
    return train, validation


def verify_fold_isolation(folds: list[PatientFold]) -> None:
    """Confirm no patient appears in two validation folds, or in both sides."""
    seen: set[str] = set()
    for fold in folds:
        overlap = seen & set(fold.validation_subjects)
        if overlap:
            raise ValueError(
                f"Patients appear in multiple validation folds: {sorted(overlap)}"
            )
        seen |= set(fold.validation_subjects)
        crossover = set(fold.train_subjects) & set(fold.validation_subjects)
        if crossover:
            raise ValueError(
                f"Fold {fold.index} has patients on both sides: {sorted(crossover)}"
            )


def aggregate_runs(
    values: list[float],
    *,
    confidence: float = 0.95,
) -> dict[str, float]:
    """Summarize a metric across folds and seeds.

    Reports the spread, not just the centre, because the entire point is that a
    single number from this pipeline is not interpretable.  ``n`` is small, so
    the interval is a normal approximation on the standard error and should be
    read as indicative rather than exact.
    """
    if not values:
        raise ValueError("No values to aggregate.")
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("Aggregated values must be finite.")

    count = len(array)
    mean = float(array.mean())
    deviation = float(array.std(ddof=1)) if count > 1 else 0.0
    standard_error = deviation / math.sqrt(count) if count > 1 else 0.0
    # 1.96 is the normal quantile for the default 95% level.
    quantile = 1.959963984540054 if abs(confidence - 0.95) < 1e-9 else None
    if quantile is None:
        from scipy import stats  # imported lazily; only needed off-default

        quantile = float(stats.norm.ppf(0.5 + confidence / 2.0))

    return {
        "runs": count,
        "mean": mean,
        "std": deviation,
        "standard_error": standard_error,
        "lower": mean - quantile * standard_error,
        "upper": mean + quantile * standard_error,
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def compare_configurations(
    baseline_values: list[float],
    candidate_values: list[float],
) -> dict[str, float]:
    """Report whether a candidate is distinguishable from a baseline.

    With a handful of folds the honest answer is usually "not distinguishable",
    and ``separated`` says so directly rather than leaving a reader to compare
    two means and assume the difference is real.
    """
    baseline = aggregate_runs(baseline_values)
    candidate = aggregate_runs(candidate_values)
    difference = candidate["mean"] - baseline["mean"]
    pooled_error = math.sqrt(
        baseline["standard_error"] ** 2 + candidate["standard_error"] ** 2
    )
    return {
        "baseline_mean": baseline["mean"],
        "candidate_mean": candidate["mean"],
        "difference": difference,
        "pooled_standard_error": pooled_error,
        "standardized_difference": (
            difference / pooled_error if pooled_error > 0 else float("nan")
        ),
        "separated": bool(pooled_error > 0 and abs(difference) > 1.96 * pooled_error),
    }
