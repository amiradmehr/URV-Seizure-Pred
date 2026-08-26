"""Tests for the within-recording positional confound controls."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from seizure_prediction.positional_controls import (
    POSITION_RANK_COLUMN,
    SECONDS_FROM_END_COLUMN,
    add_within_recording_position,
    evaluate_against_positional_controls,
    FOLLOWING_DECISIONS_COLUMN,
    filter_by_following_valid_decisions,
    match_negatives_by_position,
    positional_baseline_report,
    positional_cost_curve,
)


def make_confounded_decisions(
    recordings: int = 40,
    decisions_per_recording: int = 100,
    preictal_decisions: int = 5,
    stride_seconds: float = 60.0,
    truncate_probability: float = 0.55,
    seed: int = 0,
) -> pd.DataFrame:
    """Build decisions that reproduce the real dataset's positional structure.

    A seizure is placed at a random point in each recording and the decisions
    immediately before it are labeled positive. With probability
    ``truncate_probability`` the recording then *ends* -- which is what the
    60-minute postictal exclusion does in practice -- so those positives become
    the final valid decisions. The rest continue past the seizure.

    This yields the measured real-data shape: positives skewed strongly late
    but still overlapping negatives, rather than perfectly separated.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for recording in range(recordings):
        onset = int(rng.integers(preictal_decisions + 5, decisions_per_recording))
        truncated = rng.random() < truncate_probability
        last_index = onset if truncated else decisions_per_recording
        for index in range(last_index):
            is_positive = onset - preictal_decisions <= index < onset
            rows.append(
                {
                    "recording_id": f"rec-{recording:02d}",
                    "subject": f"{recording % 8:03d}",
                    "decision_time_seconds": float(index) * stride_seconds,
                    "label": int(is_positive),
                    "target_seizure_id": (
                        f"rec-{recording:02d}_seizure" if is_positive else None
                    ),
                }
            )
    return pd.DataFrame(rows)


class PositionTests(unittest.TestCase):
    """Cover the positional quantities themselves."""

    def setUp(self) -> None:
        self.decisions = make_confounded_decisions()

    def test_position_rank_and_gap_to_end(self) -> None:
        """The last decision of a recording ranks 1.0 with a zero gap."""
        positioned = add_within_recording_position(self.decisions)
        for _, group in positioned.groupby("recording_id"):
            last = group.loc[group["decision_time_seconds"].idxmax()]
            self.assertAlmostEqual(last[POSITION_RANK_COLUMN], 1.0)
            self.assertAlmostEqual(last[SECONDS_FROM_END_COLUMN], 0.0)
            self.assertTrue((group[SECONDS_FROM_END_COLUMN] >= 0).all())

    def test_position_is_computed_per_recording_not_globally(self) -> None:
        """Recordings with different time offsets still rank independently."""
        shifted = self.decisions.copy()
        late = shifted["recording_id"] == "rec-01"
        shifted.loc[late, "decision_time_seconds"] += 1_000_000.0
        positioned = add_within_recording_position(shifted)
        for recording_id in ("rec-00", "rec-01"):
            group = positioned[positioned["recording_id"] == recording_id]
            self.assertAlmostEqual(group[POSITION_RANK_COLUMN].max(), 1.0)
            self.assertGreater(group[POSITION_RANK_COLUMN].min(), 0.0)
            self.assertLess(group[POSITION_RANK_COLUMN].min(), 0.1)

    def test_missing_columns_are_rejected(self) -> None:
        """A frame without the required columns fails loudly."""
        with self.assertRaises(ValueError):
            add_within_recording_position(self.decisions.drop(columns=["label"]))


class PositionalBaselineTests(unittest.TestCase):
    """Cover the trivial no-EEG scorers."""

    def test_confounded_data_gives_the_baseline_large_lift(self) -> None:
        """On end-loaded positives, position alone is a strong predictor."""
        report = positional_baseline_report(make_confounded_decisions())
        self.assertGreater(report["best_positional_lift"], 5.0)
        self.assertGreater(report["prevalence"], 0.0)

    def test_unconfounded_data_gives_the_baseline_no_lift(self) -> None:
        """With positives spread uniformly, position predicts nothing."""
        rng = np.random.default_rng(0)
        decisions = make_confounded_decisions()
        # Reassign the same number of positives uniformly at random.
        positive_count = int(decisions["label"].sum())
        decisions["label"] = 0
        chosen = rng.choice(len(decisions), size=positive_count, replace=False)
        decisions.loc[chosen, "label"] = 1
        report = positional_baseline_report(decisions)
        self.assertLess(report["best_positional_lift"], 2.0)


class PositionMatchingTests(unittest.TestCase):
    """Cover the matched-negative construction."""

    def setUp(self) -> None:
        self.decisions = make_confounded_decisions()

    def test_matching_neutralizes_the_positional_baseline(self) -> None:
        """This is the point of the whole module.

        Before matching, position is worth a large lift. After matching, the
        same scorer must collapse to roughly chance, otherwise a model
        evaluated on the matched subset could still be scoring positionally.
        """
        before = positional_baseline_report(self.decisions)
        matched = match_negatives_by_position(
            self.decisions,
            negatives_per_positive=5.0,
            bins=10,
            seed=0,
        )
        after = positional_baseline_report(matched)
        self.assertGreater(before["best_positional_lift"], 5.0)
        self.assertLess(after["best_positional_lift"], 1.6)

    def test_dropped_positives_are_accounted_for(self) -> None:
        """Positives with no comparable negative are dropped, and counted.

        A positive sitting at a position where no negative exists cannot be
        scored fairly, so keeping it would smuggle the confound back in. The
        contract is that every original positive is either matched or reported
        as unmatched -- never silently lost.
        """
        matched = match_negatives_by_position(
            self.decisions,
            negatives_per_positive=3.0,
            seed=0,
        )
        kept = int(matched["label"].sum())
        dropped = int(matched.attrs["unmatched_positives"])
        self.assertEqual(kept + dropped, int(self.decisions["label"].sum()))
        self.assertGreater(kept, 0)

    def test_matched_subset_has_uniform_prevalence_across_positions(self) -> None:
        """The invariant that makes the subset neutral.

        If any position bin were richer in positives than another, position
        would still predict the label. Every populated bin must therefore sit
        at the same prevalence, 1 / (1 + ratio).
        """
        ratio = 4.0
        matched = match_negatives_by_position(
            self.decisions,
            negatives_per_positive=ratio,
            bins=10,
            seed=0,
        )
        edges = np.linspace(0.0, 1.0, 11)
        bin_index = np.clip(
            np.digitize(matched[POSITION_RANK_COLUMN].to_numpy(), edges[1:-1]),
            0,
            9,
        )
        prevalence_by_bin = (
            pd.DataFrame({"bin": bin_index, "label": matched["label"].to_numpy()})
            .groupby("bin")["label"]
            .agg(["mean", "size"])
        )
        populated = prevalence_by_bin[prevalence_by_bin["size"] >= 10]
        expected = 1.0 / (1.0 + ratio)
        for _, row in populated.iterrows():
            self.assertAlmostEqual(row["mean"], expected, delta=0.08)

    def test_matching_is_reproducible(self) -> None:
        """The same seed selects the same negatives."""
        first = match_negatives_by_position(self.decisions, seed=7)
        second = match_negatives_by_position(self.decisions, seed=7)
        pd.testing.assert_frame_equal(first, second)


class ConstructionRepairTests(unittest.TestCase):
    """Cover the following-valid-minutes filter and its cost accounting."""

    def setUp(self) -> None:
        self.decisions = make_confounded_decisions()

    def test_filter_removes_the_terminal_region(self) -> None:
        """Every surviving decision keeps the required number after it."""
        kept = filter_by_following_valid_decisions(self.decisions, 10)
        original = add_within_recording_position(self.decisions)
        for recording_id, group in kept.groupby("recording_id"):
            following = original.loc[
                original["recording_id"] == recording_id,
                FOLLOWING_DECISIONS_COLUMN,
            ]
            self.assertGreaterEqual(len(following) - len(group), 10)

    def test_filter_counts_decisions_not_elapsed_time(self) -> None:
        """A postictal gap must not make a terminal positive look safe.

        Time-to-end and decisions-after disagree exactly where the confound
        lives, so the filter has to count decisions.
        """
        positioned = add_within_recording_position(self.decisions)
        terminal = positioned[positioned[FOLLOWING_DECISIONS_COLUMN] == 0]
        self.assertTrue((terminal[SECONDS_FROM_END_COLUMN] == 0).all())
        kept = filter_by_following_valid_decisions(self.decisions, 1)
        self.assertEqual(len(kept), len(positioned) - len(terminal))

    def test_filter_applies_to_both_classes(self) -> None:
        """Negatives near the end are dropped too, or the fix would bias."""
        kept = filter_by_following_valid_decisions(self.decisions, 3)
        original = add_within_recording_position(self.decisions)
        for recording_id, group in original.groupby("recording_id"):
            expected = int((group[FOLLOWING_DECISIONS_COLUMN] >= 3).sum())
            actual = int((kept["recording_id"] == recording_id).sum())
            self.assertEqual(actual, expected)
        self.assertLess(int(kept["label"].sum()), int(self.decisions["label"].sum()))

    def test_filter_reduces_the_positional_confound(self) -> None:
        """The whole point: filtering must shrink the positional lift.

        An earlier time-based implementation silently *raised* it, because the
        filtered frame carried stale pre-filter positions.
        """
        before = positional_baseline_report(self.decisions)["best_positional_lift"]
        after = positional_baseline_report(
            filter_by_following_valid_decisions(self.decisions, 10)
        )["best_positional_lift"]
        self.assertGreater(before, 3.0)
        self.assertLess(after, before)

    def test_zero_threshold_keeps_everything(self) -> None:
        """A zero requirement is a no-op."""
        kept = filter_by_following_valid_decisions(self.decisions, 0)
        self.assertEqual(len(kept), len(self.decisions))

    def test_cost_curve_trades_positives_against_the_confound(self) -> None:
        """Raising the threshold should cost positives and shrink the lift."""
        curve = positional_cost_curve(
            self.decisions,
            thresholds_following=(0, 5, 10),
        )
        self.assertEqual(len(curve), 3)
        self.assertAlmostEqual(curve.iloc[0]["positive_retention"], 1.0)
        self.assertTrue(
            curve["positive_retention"].is_monotonic_decreasing,
            "retention must fall as the requirement rises",
        )
        self.assertIn("seizure_retention", curve.columns)


class ModelEvaluationTests(unittest.TestCase):
    """Cover the combined model-versus-confound report."""

    def setUp(self) -> None:
        self.decisions = make_confounded_decisions()

    def test_a_purely_positional_model_is_caught(self) -> None:
        """A model that only knows position must fail the matched check."""
        decisions = add_within_recording_position(self.decisions)
        decisions["probability"] = decisions[POSITION_RANK_COLUMN]
        report = evaluate_against_positional_controls(
            decisions,
            negatives_per_positive=5.0,
            seed=0,
        )
        # It ties the baseline on the full set because it *is* the baseline.
        self.assertFalse(report["full"]["beats_positional_baseline"])
        # And on the matched subset its lift collapses toward chance.
        self.assertLess(report["position_matched"]["model_lift"], 1.6)

    def test_a_genuinely_informative_model_survives_matching(self) -> None:
        """A model reading real label information keeps its lift."""
        rng = np.random.default_rng(0)
        decisions = self.decisions.copy()
        labels = decisions["label"].to_numpy()
        decisions["probability"] = labels * 0.8 + rng.normal(0, 0.1, len(labels))
        report = evaluate_against_positional_controls(
            decisions,
            negatives_per_positive=5.0,
            seed=0,
        )
        self.assertTrue(report["full"]["beats_positional_baseline"])
        self.assertTrue(report["position_matched"]["beats_positional_baseline"])
        self.assertGreater(report["position_matched"]["model_lift"], 3.0)

    def test_non_finite_scores_are_rejected(self) -> None:
        """A NaN score would silently corrupt average precision."""
        decisions = self.decisions.copy()
        decisions["probability"] = 0.5
        decisions.loc[0, "probability"] = np.nan
        with self.assertRaises(ValueError):
            evaluate_against_positional_controls(decisions)


if __name__ == "__main__":
    unittest.main()
