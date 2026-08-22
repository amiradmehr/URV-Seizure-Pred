"""Regression tests for patient/event-balanced training exposure."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from seizure_prediction.datasets import PatientEventBalancedEpochSampler


def synthetic_examples() -> pd.DataFrame:
    """Return correlated positives plus multi-patient, multi-run negatives."""
    rows: list[dict[str, object]] = []
    for subject, event_ids in {
        "001": ("event-a", "event-b", "event-c"),
        "002": ("event-d",),
    }.items():
        for event_id in event_ids:
            for lead_window in range(2):
                rows.append(
                    {
                        "label": 1,
                        "subject": subject,
                        "recording_id": f"{subject}-positive",
                        "target_seizure_id": event_id,
                        "lead_window": lead_window,
                    }
                )
    for subject in ("001", "002", "003"):
        for recording in ("run-a", "run-b"):
            for negative_index in range(4):
                rows.append(
                    {
                        "label": 0,
                        "subject": subject,
                        "recording_id": f"{subject}-{recording}",
                        "target_seizure_id": np.nan,
                        "lead_window": negative_index,
                    }
                )
    return pd.DataFrame(rows)


class PatientEventBalancedSamplerTests(unittest.TestCase):
    """Protect event uniqueness and patient/recording negative balance."""

    def setUp(self) -> None:
        self.examples = synthetic_examples()
        self.sampler = PatientEventBalancedEpochSampler(
            self.examples,
            negative_to_positive_ratio=2.0,
            max_events_per_patient=2,
            seed=13,
        )

    def test_one_window_per_distinct_selected_event(self) -> None:
        """Correlated lead-time windows cannot all enter the same epoch."""
        positive_indices = self.sampler.positive_indices_for_epoch(0)
        selected = self.examples.iloc[positive_indices]
        self.assertEqual(len(selected), 3)
        self.assertEqual(selected["target_seizure_id"].nunique(), 3)
        self.assertEqual(
            selected.groupby("subject").size().to_dict(),
            {"001": 2, "002": 1},
        )
        self.assertEqual(self.sampler.unique_seizure_count, 4)

    def test_events_and_lead_windows_rotate_across_epochs(self) -> None:
        """Capped patients and each event's positive offset must rotate."""
        first_indices = self.sampler.positive_indices_for_epoch(0)
        second_indices = self.sampler.positive_indices_for_epoch(1)
        first = self.examples.iloc[first_indices].set_index("target_seizure_id")
        second = self.examples.iloc[second_indices].set_index("target_seizure_id")
        patient_one_events = set(
            first[first["subject"] == "001"].index
        ) | set(second[second["subject"] == "001"].index)
        self.assertEqual(patient_one_events, {"event-a", "event-b", "event-c"})

        common_events = set(first.index) & set(second.index)
        for event_id in common_events:
            self.assertNotEqual(
                int(first.loc[event_id, "lead_window"]),
                int(second.loc[event_id, "lead_window"]),
            )

    def test_negatives_are_even_across_patients_and_recordings(self) -> None:
        """Long patients or runs must not dominate sampled negatives."""
        negative_indices = self.sampler.negative_indices_for_epoch(0)
        selected = self.examples.iloc[negative_indices]
        self.assertEqual(len(selected), 6)
        self.assertEqual(
            selected.groupby("subject").size().to_dict(),
            {"001": 2, "002": 2, "003": 2},
        )
        recordings_per_patient = selected.groupby("subject")[
            "recording_id"
        ].nunique()
        self.assertTrue((recordings_per_patient == 2).all())
        next_epoch = set(self.sampler.negative_indices_for_epoch(1).tolist())
        self.assertTrue(set(negative_indices.tolist()).isdisjoint(next_epoch))

    def test_epoch_indices_are_unique_and_reproducible(self) -> None:
        """A seeded epoch must contain no duplicated decision indices."""
        first = self.sampler.indices_for_epoch(3)
        duplicate_sampler = PatientEventBalancedEpochSampler(
            self.examples,
            negative_to_positive_ratio=2.0,
            max_events_per_patient=2,
            seed=13,
        )
        second = duplicate_sampler.indices_for_epoch(3)
        self.assertEqual(len(first), len(self.sampler))
        self.assertEqual(len(np.unique(first)), len(first))
        np.testing.assert_array_equal(first, second)

    def test_invalid_positive_target_is_rejected(self) -> None:
        """Every positive must be groupable into a real seizure event."""
        invalid = self.examples.copy()
        invalid.loc[0, "target_seizure_id"] = np.nan
        with self.assertRaisesRegex(ValueError, "target seizure"):
            PatientEventBalancedEpochSampler(invalid)


if __name__ == "__main__":
    unittest.main()
