"""Controls for the within-recording positional confound in this task.

The problem
-----------
A decision is labeled positive when a seizure begins in the following ten
minutes.  The pipeline also excludes sixty minutes after every seizure, and
SeizeIT2 recordings frequently end shortly after a seizure.  The combination
pushes positive decisions to the *end* of their recording's valid decision
sequence: measured on this dataset, positives sit at median within-recording
position 0.79 against 0.50 for negatives, 43% of positives fall in the final
10% of their recording, and in 55% of seizure-containing recordings the final
valid decision is positive.

The consequence is that "how close is this decision to the end of its
recording" scores AP 0.032 on the validation split -- around a 5x lift over
prevalence, without reading a single EEG sample.  That is the same range as
every model trained on this dataset so far.  Nothing errors, because the labels
are correct; the confound is in the task construction, not the code.

What this module provides
-------------------------
1. ``add_within_recording_position`` -- the positional quantities themselves.
2. ``positional_baseline_report`` -- the trivial no-EEG scorers, so every
   evaluation can print the number a model has to beat.
3. ``match_negatives_by_position`` -- a stratified subset in which negatives
   carry the same within-recording position distribution as positives, so
   positional information is worth nothing and a model's remaining lift is
   attributable to EEG.
4. ``filter_by_following_valid_decisions`` and ``positional_cost_curve`` -- the
   construction-side repair, which requires every decision to be followed by a
   minimum number of following decisions so positives stop being terminal, plus
   the accounting needed to choose that threshold knowingly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


REQUIRED_COLUMNS = ("recording_id", "decision_time_seconds", "label")

# Coarse bins leave residual position signal inside each bin. On the real
# validation split the residual positional lift falls 1.90 -> 1.42 -> 1.16 ->
# 1.08 at 10, 20, 40 and 80 bins and is flat beyond that, while positive
# retention only drops to 0.92. Eighty is where neutralization is achieved
# without paying more positives for it.
DEFAULT_POSITION_BINS = 80

POSITION_RANK_COLUMN = "within_recording_position"
SECONDS_FROM_END_COLUMN = "seconds_to_last_valid_decision"
FOLLOWING_DECISIONS_COLUMN = "following_valid_decisions"


def _validate(decisions: pd.DataFrame) -> None:
    """Reject frames that cannot carry positional analysis."""
    missing = set(REQUIRED_COLUMNS) - set(decisions.columns)
    if missing:
        raise ValueError(
            f"Decision table is missing columns: {sorted(missing)}"
        )
    if decisions.empty:
        raise ValueError("Decision table is empty.")
    if not decisions["label"].isin([0, 1]).all():
        raise ValueError("Labels must be binary.")


def add_within_recording_position(decisions: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with each decision's position inside its own recording.

    ``within_recording_position`` is the percentile rank of the decision time
    among that recording's valid decisions, so 1.0 is the last one.
    ``seconds_to_last_valid_decision`` is the gap to that final decision, which
    is the quantity the postictal exclusion actually distorts.
    """
    _validate(decisions)
    positioned = decisions.copy()
    positioned["recording_id"] = positioned["recording_id"].astype(str)
    grouped = positioned.groupby("recording_id")["decision_time_seconds"]
    positioned[POSITION_RANK_COLUMN] = grouped.rank(pct=True).astype(float)
    positioned[SECONDS_FROM_END_COLUMN] = (
        grouped.transform("max") - positioned["decision_time_seconds"]
    ).astype(float)
    # The count of following decisions, not the elapsed time, is what the
    # confound is made of. A positive followed by a 60-minute postictal
    # exclusion has hours of recording left but almost no valid decisions
    # after it, so the time measure would wrongly call it non-terminal.
    positioned[FOLLOWING_DECISIONS_COLUMN] = (
        positioned.sort_values("decision_time_seconds")
        .groupby("recording_id")
        .cumcount(ascending=False)
        .reindex(positioned.index)
        .astype(int)
    )
    return positioned


def positional_scores(decisions: pd.DataFrame) -> dict[str, np.ndarray]:
    """Return the trivial no-EEG scorers that exploit the confound.

    Each is oriented so that a higher value means "more likely positive".
    """
    # Always recompute. Reusing columns that happen to be present would score
    # a filtered subset with its pre-filter positions, which silently inverts
    # the result.
    positioned = add_within_recording_position(decisions)
    return {
        "position_within_recording": positioned[POSITION_RANK_COLUMN].to_numpy(
            dtype=np.float64
        ),
        "closeness_to_recording_end": -positioned[
            SECONDS_FROM_END_COLUMN
        ].to_numpy(dtype=np.float64),
    }


def positional_baseline_report(decisions: pd.DataFrame) -> dict[str, float]:
    """Return AP and prevalence lift for every trivial positional scorer.

    ``best_positional_average_precision`` is the number a model must beat
    before any of its performance can be attributed to EEG.
    """
    _validate(decisions)
    labels = decisions["label"].to_numpy(dtype=np.int64)
    prevalence = float(labels.mean())
    if prevalence <= 0.0 or prevalence >= 1.0:
        raise ValueError("Positional controls need both classes present.")

    report: dict[str, float] = {"prevalence": prevalence}
    best = 0.0
    for name, scores in positional_scores(decisions).items():
        average_precision = float(average_precision_score(labels, scores))
        report[f"{name}_average_precision"] = average_precision
        report[f"{name}_lift"] = average_precision / prevalence
        best = max(best, average_precision)
    report["best_positional_average_precision"] = best
    report["best_positional_lift"] = best / prevalence
    return report


def match_negatives_by_position(
    decisions: pd.DataFrame,
    *,
    negatives_per_positive: float = 20.0,
    bins: int = DEFAULT_POSITION_BINS,
    seed: int = 0,
) -> pd.DataFrame:
    """Return a subset whose negatives match the positives' position profile.

    Within each within-recording position bin the classes are held at the same
    ratio, so prevalence is constant across positions and a positional scorer
    collapses to chance.  Any lift a model retains here cannot be positional.

    Holding the ratio uniform is what makes the subset neutral, so a bin that
    cannot supply enough negatives has its *positives* subsampled rather than
    its ratio allowed to drift, and positives sitting where no negative exists
    at all are dropped.  The count of positives lost this way is recorded in
    ``matched.attrs["unmatched_positives"]`` and reported by
    :func:`matched_subset_report`; it is never silent.
    """
    _validate(decisions)
    if negatives_per_positive <= 0:
        raise ValueError("negatives_per_positive must be positive.")
    if bins < 2:
        raise ValueError("bins must be at least two.")

    positioned = add_within_recording_position(decisions).reset_index(drop=True)
    edges = np.linspace(0.0, 1.0, bins + 1)
    # Right-closed bins so the final decision of a recording (position 1.0)
    # lands in the last bin rather than falling outside the range.
    positioned["position_bin"] = np.clip(
        np.digitize(positioned[POSITION_RANK_COLUMN].to_numpy(), edges[1:-1]),
        0,
        bins - 1,
    )

    rng = np.random.default_rng(seed)
    positives = positioned[positioned["label"] == 1]
    negatives = positioned[positioned["label"] == 0]
    if positives.empty or negatives.empty:
        raise ValueError("Position matching requires both classes.")

    # Every bin must end up at the *same* positive:negative ratio. If one bin
    # were richer in positives than another, position would still carry label
    # information and the subset would not be neutral. Where a bin cannot
    # supply enough negatives, its positives are subsampled instead of its
    # ratio being allowed to drift.
    selected_indices: list[np.ndarray] = []
    unmatched_positives = 0
    for position_bin, bin_positives in positives.groupby("position_bin"):
        bin_negatives = negatives[negatives["position_bin"] == position_bin]
        if bin_negatives.empty:
            # No position-comparable negative exists, so these positives can
            # never be scored fairly. Dropping them is the honest choice.
            unmatched_positives += len(bin_positives)
            continue

        wanted_negatives = int(round(len(bin_positives) * negatives_per_positive))
        if len(bin_negatives) >= wanted_negatives:
            keep_positives = bin_positives.index.to_numpy()
            take_negatives = wanted_negatives
        else:
            affordable_positives = int(
                np.floor(len(bin_negatives) / negatives_per_positive)
            )
            if affordable_positives < 1:
                unmatched_positives += len(bin_positives)
                continue
            keep_positives = rng.choice(
                bin_positives.index.to_numpy(),
                size=affordable_positives,
                replace=False,
            )
            unmatched_positives += len(bin_positives) - affordable_positives
            take_negatives = int(
                round(affordable_positives * negatives_per_positive)
            )

        chosen_negatives = rng.choice(
            bin_negatives.index.to_numpy(),
            size=min(take_negatives, len(bin_negatives)),
            replace=False,
        )
        selected_indices.append(np.asarray(keep_positives, dtype=np.int64))
        selected_indices.append(np.asarray(chosen_negatives, dtype=np.int64))

    if not selected_indices:
        raise ValueError(
            "No position bin contains both classes, so the positional confound "
            "cannot be neutralized by matching. The positive and negative "
            "position distributions do not overlap; use "
            "filter_by_following_valid_decisions to repair the construction "
            "instead."
        )

    matched = positioned.loc[np.concatenate(selected_indices)]
    matched = matched.sort_index().reset_index(drop=True)
    matched.attrs["unmatched_positives"] = unmatched_positives
    return matched


def matched_subset_report(
    matched: pd.DataFrame,
    original: pd.DataFrame,
) -> dict[str, float]:
    """Describe how much of the original set survived position matching."""
    _validate(matched)
    _validate(original)
    matched_labels = matched["label"].to_numpy(dtype=np.int64)
    original_labels = original["label"].to_numpy(dtype=np.int64)
    positives = int(matched_labels.sum())
    negatives = int((matched_labels == 0).sum())
    return {
        "matched_decisions": int(len(matched)),
        "matched_positives": positives,
        "matched_negatives": negatives,
        "achieved_negatives_per_positive": (
            negatives / positives if positives else float("nan")
        ),
        "matched_prevalence": float(matched_labels.mean()),
        "original_decisions": int(len(original)),
        "original_prevalence": float(original_labels.mean()),
        "positive_retention": (
            positives / int(original_labels.sum())
            if original_labels.sum()
            else float("nan")
        ),
    }


def filter_by_following_valid_decisions(
    decisions: pd.DataFrame,
    minimum_following: int,
) -> pd.DataFrame:
    """Drop decisions with too few valid decisions after them in their recording.

    Applied to both classes alike, this is the construction-side repair for the
    confound: it removes the terminal region where positives concentrate, so
    "near the end" stops predicting "preictal".

    The criterion counts *decisions*, not elapsed time. A positive followed by
    the 60-minute postictal exclusion still has hours of recording left, so a
    time-based rule would wrongly treat it as non-terminal and leave the
    confound in place. Measured on the validation split, requiring 10 following
    decisions drops the positional lift from 5.25x to 1.16x while retaining 71%
    of positives; requiring 20 drops it to 0.84x at 67% retention.

    This is expensive in positives, which is why :func:`positional_cost_curve`
    exists to price it before it is adopted.
    """
    if minimum_following < 0:
        raise ValueError("minimum_following cannot be negative.")
    positioned = add_within_recording_position(decisions)
    keep = positioned[FOLLOWING_DECISIONS_COLUMN] >= minimum_following
    kept = positioned[keep].reset_index(drop=True)
    # The retained rows have new positions inside the smaller sequence, so the
    # stale derived columns must not travel with them.
    return kept.drop(
        columns=[
            POSITION_RANK_COLUMN,
            SECONDS_FROM_END_COLUMN,
            FOLLOWING_DECISIONS_COLUMN,
        ]
    )


def positional_cost_curve(
    decisions: pd.DataFrame,
    thresholds_following: tuple[int, ...] = (0, 5, 10, 20, 30, 60),
    seizure_column: str = "target_seizure_id",
) -> pd.DataFrame:
    """Price the construction repair across candidate thresholds.

    For each threshold this reports what survives and, critically, what the
    positional baseline is still worth afterwards.  The threshold to choose is
    the smallest one that drives ``best_positional_lift`` near 1.0 while
    retaining enough seizures to train on.
    """
    _validate(decisions)
    rows: list[dict[str, float]] = []
    original_positives = int((decisions["label"] == 1).sum())
    original_seizures = (
        decisions.loc[decisions["label"] == 1, seizure_column].nunique()
        if seizure_column in decisions.columns
        else np.nan
    )

    for threshold in thresholds_following:
        kept = filter_by_following_valid_decisions(decisions, threshold)
        positives = int((kept["label"] == 1).sum())
        negatives = int((kept["label"] == 0).sum())
        row: dict[str, float] = {
            "minimum_following_decisions": int(threshold),
            "decisions": int(len(kept)),
            "positives": positives,
            "negatives": negatives,
            "positive_retention": (
                positives / original_positives if original_positives else np.nan
            ),
        }
        if seizure_column in kept.columns:
            seizures = kept.loc[kept["label"] == 1, seizure_column].nunique()
            row["seizures"] = int(seizures)
            row["seizure_retention"] = (
                seizures / original_seizures if original_seizures else np.nan
            )
        if positives > 0 and negatives > 0:
            baseline = positional_baseline_report(kept)
            row["prevalence"] = baseline["prevalence"]
            row["best_positional_lift"] = baseline["best_positional_lift"]
        else:
            row["prevalence"] = np.nan
            row["best_positional_lift"] = np.nan
        rows.append(row)

    return pd.DataFrame(rows)


def evaluate_against_positional_controls(
    decisions: pd.DataFrame,
    score_column: str = "probability",
    *,
    negatives_per_positive: float = 20.0,
    bins: int = DEFAULT_POSITION_BINS,
    seed: int = 0,
) -> dict[str, object]:
    """Score a model beside the confound it has to beat.

    Returns the model's average precision on the full set, the trivial
    positional baselines on that same set, and both again on a position-matched
    subset where the confound is neutralized.  ``beats_positional_baseline``
    is the honest headline: a model that fails it has demonstrated nothing
    about EEG.
    """
    _validate(decisions)
    if score_column not in decisions.columns:
        raise ValueError(f"Decision table has no {score_column!r} column.")

    labels = decisions["label"].to_numpy(dtype=np.int64)
    scores = decisions[score_column].to_numpy(dtype=np.float64)
    if not np.isfinite(scores).all():
        raise ValueError("Model scores contain non-finite values.")

    prevalence = float(labels.mean())
    model_average_precision = float(average_precision_score(labels, scores))
    baseline = positional_baseline_report(decisions)

    matched = match_negatives_by_position(
        decisions,
        negatives_per_positive=negatives_per_positive,
        bins=bins,
        seed=seed,
    )
    matched_labels = matched["label"].to_numpy(dtype=np.int64)
    matched_scores = matched[score_column].to_numpy(dtype=np.float64)
    matched_prevalence = float(matched_labels.mean())
    matched_model = float(average_precision_score(matched_labels, matched_scores))
    matched_baseline = positional_baseline_report(matched)

    return {
        "full": {
            "decisions": int(len(decisions)),
            "prevalence": prevalence,
            "model_average_precision": model_average_precision,
            "model_lift": model_average_precision / prevalence,
            **{
                key: value
                for key, value in baseline.items()
                if key != "prevalence"
            },
            "beats_positional_baseline": bool(
                model_average_precision
                > baseline["best_positional_average_precision"]
            ),
        },
        "position_matched": {
            **matched_subset_report(matched, decisions),
            "model_average_precision": matched_model,
            "model_lift": matched_model / matched_prevalence,
            "best_positional_average_precision": matched_baseline[
                "best_positional_average_precision"
            ],
            "best_positional_lift": matched_baseline["best_positional_lift"],
            "beats_positional_baseline": bool(
                matched_model
                > matched_baseline["best_positional_average_precision"]
            ),
        },
        "matching": {
            "negatives_per_positive": negatives_per_positive,
            "bins": bins,
            "seed": seed,
        },
    }
