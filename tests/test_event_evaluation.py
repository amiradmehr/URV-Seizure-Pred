"""Regression tests for event-level seizure-warning evaluation."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from seizure_prediction.event_evaluation import (
    bootstrap_patient_metrics,
    evaluate_alarm_threshold,
    prepare_target_seizures,
    threshold_grid,
)


class EventEvaluationTests(unittest.TestCase):
    """Protect alarm merging, event detection, and exposure denominators."""

    def test_alarm_episodes_detect_events_and_merge_adjacent_alerts(self) -> None:
        """Adjacent minute alerts must count as one true or false episode."""
        decisions = pd.DataFrame(
            {
                "recording_id": ["recording-a"] * 3 + ["recording-b"] * 3,
                "subject": ["101"] * 3 + ["102"] * 3,
                "decision_time_seconds": [0, 60, 120, 0, 60, 120],
                "label": [1, 1, 0, 0, 0, 0],
                "probability": [0.9, 0.8, 0.1, 0.7, 0.6, 0.1],
            }
        )
        seizures = pd.DataFrame(
            {
                "seizure_id": ["seizure-a"],
                "subject": ["101"],
                "recording_id": ["recording-a"],
                "onset_seconds": [500.0],
            }
        )
        result = evaluate_alarm_threshold(
            decisions,
            seizures,
            threshold=0.5,
            prediction_horizon_seconds=0.0,
            occurrence_period_seconds=600.0,
            refractory_seconds=600.0,
            decision_stride_seconds=60.0,
        )

        self.assertEqual(result.metrics["detected_seizures"], 1)
        self.assertEqual(result.metrics["total_seizures"], 1)
        self.assertEqual(result.metrics["total_alarm_episodes"], 2)
        self.assertEqual(result.metrics["true_alarm_episodes"], 1)
        self.assertEqual(result.metrics["false_alarm_episodes"], 1)
        self.assertAlmostEqual(result.metrics["event_sensitivity"], 1.0)
        self.assertAlmostEqual(result.metrics["time_in_warning_fraction"], 1.0)
        self.assertAlmostEqual(result.metrics["false_alarms_per_24h"], 360.0)
        self.assertEqual(
            result.alarm_episodes["number_of_raw_alerts"].tolist(),
            [2, 2],
        )
        event = result.seizure_events.iloc[0]
        self.assertTrue(event["detected"])
        self.assertAlmostEqual(event["warning_lead_minutes"], 500.0 / 60.0)

    def test_prediction_horizon_is_respected(self) -> None:
        """An onset before a nonzero horizon cannot be called detected."""
        decisions = pd.DataFrame(
            {
                "recording_id": ["recording-a"],
                "subject": ["101"],
                "decision_time_seconds": [0.0],
                "label": [0],
                "probability": [0.9],
            }
        )
        seizures = pd.DataFrame(
            {
                "seizure_id": ["seizure-a"],
                "subject": ["101"],
                "recording_id": ["recording-a"],
                "onset_seconds": [30.0],
            }
        )
        result = evaluate_alarm_threshold(
            decisions,
            seizures,
            threshold=0.5,
            prediction_horizon_seconds=60.0,
            occurrence_period_seconds=600.0,
            refractory_seconds=600.0,
            decision_stride_seconds=60.0,
        )
        self.assertEqual(result.metrics["detected_seizures"], 0)
        self.assertEqual(result.metrics["false_alarm_episodes"], 1)

    def test_prepare_target_seizures_requires_every_target(self) -> None:
        """Missing onset metadata must fail rather than reduce the denominator."""
        manifest = pd.DataFrame(
            {
                "seizure_id": ["recording-a_seizure-0001"],
                "subject": ["1"],
                "onset_seconds": [100.0],
            }
        )
        prepared = prepare_target_seizures(
            manifest,
            {"recording-a_seizure-0001"},
        )
        self.assertEqual(prepared.loc[0, "subject"], "001")
        self.assertEqual(prepared.loc[0, "recording_id"], "recording-a")
        with self.assertRaisesRegex(ValueError, "missing"):
            prepare_target_seizures(manifest, {"unknown-seizure"})

    def test_threshold_grid_contains_all_and_no_alarm_endpoints(self) -> None:
        """The operating sweep must include both boundary behaviors."""
        probabilities = np.array([0.1, 0.2, 0.3, 0.4])
        thresholds = threshold_grid(probabilities, 3)
        self.assertIn(0.0, thresholds)
        self.assertGreater(float(thresholds.max()), 0.4)
        self.assertTrue(np.all(np.diff(thresholds) <= 0.0))

    def test_patient_bootstrap_returns_finite_intervals(self) -> None:
        """Patient-resampled uncertainty must cover all operating metrics."""
        per_subject = pd.DataFrame(
            {
                "total_seizures": [2, 1],
                "detected_seizures": [1, 1],
                "event_sensitivity": [0.5, 1.0],
                "false_alarm_episodes": [2, 1],
                "interictal_hours": [24.0, 12.0],
                "valid_decisions": [100, 50],
                "warning_decisions": [10, 10],
            }
        )
        intervals = bootstrap_patient_metrics(
            per_subject,
            samples=100,
            seed=7,
        )
        self.assertEqual(
            set(intervals),
            {
                "event_sensitivity",
                "macro_patient_sensitivity",
                "false_alarms_per_24h",
                "time_in_warning_fraction",
            },
        )
        for interval in intervals.values():
            self.assertTrue(np.isfinite(list(interval.values())).all())
            self.assertLessEqual(interval["lower_95"], interval["upper_95"])


if __name__ == "__main__":
    unittest.main()
