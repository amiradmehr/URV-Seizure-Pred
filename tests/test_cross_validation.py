"""Tests for patient-level cross-validation and multi-run aggregation."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from seizure_prediction.cross_validation import (
    aggregate_runs,
    compare_configurations,
    make_patient_folds,
    seizures_by_subject,
    split_examples_by_fold,
    verify_fold_isolation,
)


def make_examples(
    seizures_per_subject: dict[str, int],
    negatives_per_subject: int = 40,
) -> pd.DataFrame:
    """Build decisions with a controlled per-subject seizure count."""
    rows = []
    for subject, seizure_count in seizures_per_subject.items():
        for index in range(negatives_per_subject):
            rows.append(
                {
                    "subject": subject,
                    "recording_id": f"{subject}_run-01",
                    "decision_time_seconds": float(index) * 60.0,
                    "label": 0,
                    "target_seizure_id": None,
                }
            )
        for seizure in range(seizure_count):
            for window in range(3):
                rows.append(
                    {
                        "subject": subject,
                        "recording_id": f"{subject}_run-01",
                        "decision_time_seconds": 10_000.0 + seizure * 100 + window,
                        "label": 1,
                        "target_seizure_id": f"{subject}_seizure-{seizure}",
                    }
                )
    return pd.DataFrame(rows)


class SeizureCountTests(unittest.TestCase):
    """Cover the per-subject event count."""

    def test_counts_distinct_seizures_not_windows(self) -> None:
        """Ten correlated windows from one seizure are one event."""
        examples = make_examples({"001": 4})
        counts = seizures_by_subject(examples)
        self.assertEqual(int(counts.loc["001"]), 4)

    def test_subjects_without_seizures_are_retained(self) -> None:
        """A negatives-only patient still has to land in exactly one fold."""
        examples = make_examples({"001": 3, "002": 0})
        counts = seizures_by_subject(examples)
        self.assertEqual(int(counts.loc["002"]), 0)
        self.assertEqual(len(counts), 2)


class PatientFoldTests(unittest.TestCase):
    """Cover the fold partition."""

    def setUp(self) -> None:
        # Deliberately unequal, like the real cohort: a few patients hold most
        # of the events and many hold none.
        counts = {f"{index:03d}": index % 7 for index in range(1, 41)}
        self.examples = make_examples(counts)

    def test_every_patient_is_held_out_exactly_once(self) -> None:
        """The folds must partition the cohort, not sample it."""
        folds = make_patient_folds(self.examples, folds=5, seed=0)
        verify_fold_isolation(folds)
        held_out = [
            subject for fold in folds for subject in fold.validation_subjects
        ]
        self.assertEqual(len(held_out), len(set(held_out)))
        self.assertEqual(
            set(held_out),
            set(self.examples["subject"].astype(str).str.zfill(3)),
        )

    def test_no_patient_appears_on_both_sides_of_a_fold(self) -> None:
        """The whole point of a patient-level split."""
        folds = make_patient_folds(self.examples, folds=5, seed=0)
        for fold in folds:
            self.assertFalse(
                set(fold.train_subjects) & set(fold.validation_subjects)
            )

    def test_seizure_counts_are_balanced_across_folds(self) -> None:
        """An unlucky fold with no events would be uninterpretable."""
        folds = make_patient_folds(self.examples, folds=5, seed=0)
        counts = np.array([fold.validation_seizures for fold in folds])
        self.assertGreater(counts.min(), 0)
        self.assertLessEqual(counts.max() - counts.min(), 2)

    def test_patient_counts_are_balanced_across_folds(self) -> None:
        """Zero-seizure patients must not all pile into the poorest fold."""
        folds = make_patient_folds(self.examples, folds=5, seed=0)
        sizes = np.array([len(fold.validation_subjects) for fold in folds])
        self.assertLessEqual(sizes.max() - sizes.min(), 2)

    def test_different_seeds_give_different_partitions(self) -> None:
        """Multi-seed runs need the partition itself to vary."""
        first = make_patient_folds(self.examples, folds=5, seed=0)
        second = make_patient_folds(self.examples, folds=5, seed=1)
        self.assertNotEqual(
            [set(fold.validation_subjects) for fold in first],
            [set(fold.validation_subjects) for fold in second],
        )

    def test_splitting_reproduces_the_fold_membership(self) -> None:
        """The returned frames must match the fold's subject lists."""
        folds = make_patient_folds(self.examples, folds=5, seed=0)
        train, validation = split_examples_by_fold(self.examples, folds[0])
        self.assertEqual(
            set(validation["subject"].astype(str).str.zfill(3)),
            set(folds[0].validation_subjects),
        )
        self.assertEqual(
            set(train["subject"].astype(str).str.zfill(3)),
            set(folds[0].train_subjects),
        )

    def test_overlapping_folds_are_rejected(self) -> None:
        """The isolation check has to actually catch a violation."""
        folds = make_patient_folds(self.examples, folds=5, seed=0)
        broken = list(folds)
        broken[1] = type(broken[1])(
            index=1,
            train_subjects=broken[1].train_subjects,
            validation_subjects=broken[0].validation_subjects,
            train_seizures=broken[1].train_seizures,
            validation_seizures=broken[1].validation_seizures,
        )
        with self.assertRaises(ValueError):
            verify_fold_isolation(broken)

    def test_too_few_patients_is_rejected(self) -> None:
        """Three patients cannot make five folds."""
        small = make_examples({"001": 2, "002": 2, "003": 2})
        with self.assertRaises(ValueError):
            make_patient_folds(small, folds=5, seed=0)


class AggregationTests(unittest.TestCase):
    """Cover the multi-run summaries."""

    def test_aggregate_reports_spread_not_just_centre(self) -> None:
        """A single number from this pipeline is not interpretable."""
        summary = aggregate_runs([0.02, 0.04, 0.03, 0.05, 0.01])
        self.assertEqual(summary["runs"], 5)
        self.assertAlmostEqual(summary["mean"], 0.03)
        self.assertGreater(summary["std"], 0.0)
        self.assertLess(summary["lower"], summary["mean"])
        self.assertGreater(summary["upper"], summary["mean"])

    def test_single_run_has_no_spread(self) -> None:
        """One run cannot support an interval, and must not fake one."""
        summary = aggregate_runs([0.03])
        self.assertEqual(summary["std"], 0.0)
        self.assertEqual(summary["lower"], summary["upper"])

    def test_noisy_configurations_are_reported_as_indistinguishable(self) -> None:
        """The realistic case for this project, and it must say so."""
        rng = np.random.default_rng(0)
        baseline = list(rng.normal(0.03, 0.015, 5))
        candidate = list(rng.normal(0.035, 0.015, 5))
        comparison = compare_configurations(baseline, candidate)
        self.assertFalse(comparison["separated"])

    def test_clearly_better_configurations_are_separated(self) -> None:
        """A large, consistent gap must register."""
        baseline = [0.030, 0.031, 0.029, 0.030, 0.031]
        candidate = [0.090, 0.091, 0.089, 0.090, 0.091]
        comparison = compare_configurations(baseline, candidate)
        self.assertTrue(comparison["separated"])
        self.assertGreater(comparison["difference"], 0.0)

    def test_non_finite_values_are_rejected(self) -> None:
        """A NaN fold would silently poison the mean."""
        with self.assertRaises(ValueError):
            aggregate_runs([0.03, float("nan")])


if __name__ == "__main__":
    unittest.main()
