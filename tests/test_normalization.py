"""Tests for global, per-patient, and per-recording channel standardization."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from seizure_prediction.normalization import (
    GLOBAL_SCALER_KEY,
    apply_channel_scaler,
    build_scaler_document,
    fit_channel_scaler,
    load_scaler_document,
    save_scaler_document,
    scaler_key_for,
    select_scaler,
)


CHANNELS = ["BTE_LEFT", "BTE_RIGHT", "CROSS_HEAD"]


def write_recording(
    directory: Path,
    name: str,
    signal: np.ndarray,
) -> Path:
    """Save one continuous (channels, samples) recording."""
    path = directory / f"{name}.npy"
    np.save(path, signal.astype(np.float32))
    return path


class NormalizationTests(unittest.TestCase):
    """Cover fitting, application, and mode equivalence."""

    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.directory = Path(self._directory.name)
        self.rng = np.random.default_rng(0)
        # Two patients whose amplitudes differ by two orders of magnitude,
        # which is the situation per-patient scaling is meant to remove.
        # Each has a zero-filled CROSS_HEAD placeholder.
        self.availability = np.array([True, True, False])
        self.patient_a = self.make_patient(scale=1e-4, offset=2e-5, samples=6000)
        self.patient_b = self.make_patient(scale=1e-2, offset=-3e-3, samples=6000)

    def tearDown(self) -> None:
        self._directory.cleanup()

    def make_patient(
        self,
        *,
        scale: float,
        offset: float,
        samples: int,
    ) -> np.ndarray:
        """Return a 3-channel recording with one zero placeholder channel."""
        signal = np.zeros((3, samples), dtype=np.float64)
        signal[0] = self.rng.normal(offset, scale, samples)
        signal[1] = self.rng.normal(offset, scale, samples)
        return signal

    def test_patient_scaler_standardizes_each_patient_to_zero_mean_unit_scale(
        self,
    ) -> None:
        """Per-patient fitting removes each patient's own center and scale."""
        for name, signal in (("a", self.patient_a), ("b", self.patient_b)):
            path = write_recording(self.directory, name, signal)
            scaler = fit_channel_scaler(
                [path],
                {path: self.availability},
                channel_names=CHANNELS,
                statistic="meanstd",
                epsilon=1e-8,
            )
            document = build_scaler_document(
                mode="patient",
                statistic="meanstd",
                channel_names=CHANNELS,
                scalers={"001": scaler},
                epsilon=1e-8,
            )
            standardized = apply_channel_scaler(
                signal,
                CHANNELS,
                self.availability,
                scaler,
                document,
                "float32",
            )
            present = standardized[:2]
            self.assertTrue(np.allclose(present.mean(axis=1), 0.0, atol=1e-5))
            self.assertTrue(np.allclose(present.std(axis=1), 1.0, atol=1e-4))

    def test_unavailable_channel_is_excluded_from_fit_and_stays_zero(self) -> None:
        """The zero placeholder must never be treated as EEG."""
        path = write_recording(self.directory, "a", self.patient_a)
        scaler = fit_channel_scaler(
            [path],
            {path: self.availability},
            channel_names=CHANNELS,
            statistic="meanstd",
            epsilon=1e-8,
        )
        self.assertEqual(scaler["fitted_channels"], [1, 1, 0])
        self.assertEqual(scaler["samples_per_channel"][2], 0)
        # A channel with no data gets a neutral transform, never a division by
        # a degenerate scale.
        self.assertEqual(scaler["center"][2], 0.0)
        self.assertEqual(scaler["scale"][2], 1.0)

        document = build_scaler_document(
            mode="patient",
            statistic="meanstd",
            channel_names=CHANNELS,
            scalers={"001": scaler},
            epsilon=1e-8,
        )
        standardized = apply_channel_scaler(
            self.patient_a,
            CHANNELS,
            self.availability,
            scaler,
            document,
            "float32",
        )
        self.assertTrue(np.array_equal(standardized[2], np.zeros(6000, dtype=np.float32)))

    def test_patient_scaling_makes_different_patients_comparable(self) -> None:
        """A global scaler leaves an amplitude gap that patient scaling closes."""
        path_a = write_recording(self.directory, "a", self.patient_a)
        path_b = write_recording(self.directory, "b", self.patient_b)
        availability_by_path = {
            path_a: self.availability,
            path_b: self.availability,
        }

        global_scaler = fit_channel_scaler(
            [path_a, path_b],
            availability_by_path,
            channel_names=CHANNELS,
            statistic="meanstd",
            epsilon=1e-8,
        )
        global_document = build_scaler_document(
            mode="global",
            statistic="meanstd",
            channel_names=CHANNELS,
            scalers={GLOBAL_SCALER_KEY: global_scaler},
            epsilon=1e-8,
            training_subjects=["001", "002"],
        )
        globally_scaled = [
            apply_channel_scaler(
                signal,
                CHANNELS,
                self.availability,
                global_scaler,
                global_document,
                "float32",
            )
            for signal in (self.patient_a, self.patient_b)
        ]
        global_ratio = (
            globally_scaled[1][:2].std() / globally_scaled[0][:2].std()
        )

        patient_scaled = []
        for path, signal in ((path_a, self.patient_a), (path_b, self.patient_b)):
            scaler = fit_channel_scaler(
                [path],
                availability_by_path,
                channel_names=CHANNELS,
                statistic="meanstd",
                epsilon=1e-8,
            )
            document = build_scaler_document(
                mode="patient",
                statistic="meanstd",
                channel_names=CHANNELS,
                scalers={"001": scaler},
                epsilon=1e-8,
            )
            patient_scaled.append(
                apply_channel_scaler(
                    signal,
                    CHANNELS,
                    self.availability,
                    scaler,
                    document,
                    "float32",
                )
            )
        patient_ratio = (
            patient_scaled[1][:2].std() / patient_scaled[0][:2].std()
        )

        self.assertGreater(global_ratio, 10.0)
        self.assertAlmostEqual(patient_ratio, 1.0, places=3)

    def test_restandardizing_globally_scaled_data_equals_a_raw_patient_build(
        self,
    ) -> None:
        """The conversion script's core identity, verified numerically.

        Re-z-scoring already-standardized data by its own per-patient
        statistics must equal standardizing the raw data per patient, because
        the earlier per-channel affine transform cancels. This is what lets
        scripts/restandardize_processed.py skip the raw EDFs.
        """
        path_a = write_recording(self.directory, "a", self.patient_a)
        path_b = write_recording(self.directory, "b", self.patient_b)
        availability_by_path = {
            path_a: self.availability,
            path_b: self.availability,
        }

        # Path A: fit per patient directly on raw units.
        direct = {}
        for key, path, signal in (
            ("001", path_a, self.patient_a),
            ("002", path_b, self.patient_b),
        ):
            scaler = fit_channel_scaler(
                [path],
                availability_by_path,
                channel_names=CHANNELS,
                statistic="meanstd",
                epsilon=1e-8,
            )
            document = build_scaler_document(
                mode="patient",
                statistic="meanstd",
                channel_names=CHANNELS,
                scalers={key: scaler},
                epsilon=1e-8,
            )
            direct[key] = apply_channel_scaler(
                signal,
                CHANNELS,
                self.availability,
                scaler,
                document,
                "float32",
            )

        # Path B: global first, then per patient on the stored result.
        global_scaler = fit_channel_scaler(
            [path_a, path_b],
            availability_by_path,
            channel_names=CHANNELS,
            statistic="meanstd",
            epsilon=1e-8,
        )
        global_document = build_scaler_document(
            mode="global",
            statistic="meanstd",
            channel_names=CHANNELS,
            scalers={GLOBAL_SCALER_KEY: global_scaler},
            epsilon=1e-8,
            training_subjects=["001", "002"],
        )
        via_global = {}
        for key, name, signal in (
            ("001", "ga", self.patient_a),
            ("002", "gb", self.patient_b),
        ):
            stored = apply_channel_scaler(
                signal,
                CHANNELS,
                self.availability,
                global_scaler,
                global_document,
                "float32",
            )
            stored_path = write_recording(self.directory, name, stored)
            scaler = fit_channel_scaler(
                [stored_path],
                {stored_path: self.availability},
                channel_names=CHANNELS,
                statistic="meanstd",
                epsilon=1e-8,
            )
            document = build_scaler_document(
                mode="patient",
                statistic="meanstd",
                channel_names=CHANNELS,
                scalers={key: scaler},
                epsilon=1e-8,
            )
            via_global[key] = apply_channel_scaler(
                stored,
                CHANNELS,
                self.availability,
                scaler,
                document,
                "float32",
            )

        for key in ("001", "002"):
            self.assertTrue(
                np.allclose(direct[key], via_global[key], atol=1e-4),
                f"patient {key} diverged between build paths",
            )

    def test_refitting_a_coarser_scope_recovers_the_original_transform(
        self,
    ) -> None:
        """global -> patient -> global must return to the original scaling.

        Going back to a coarser scope pools recordings that no longer share a
        stored scaler, so each has to be returned to raw units individually.
        Reusing one group member's scaler for the whole group silently
        corrupts this direction.
        """
        path_a = write_recording(self.directory, "a", self.patient_a)
        path_b = write_recording(self.directory, "b", self.patient_b)
        availability_by_path = {
            path_a: self.availability,
            path_b: self.availability,
        }
        identity = {
            path: (np.zeros(3), np.ones(3)) for path in availability_by_path
        }

        original_global = fit_channel_scaler(
            [path_a, path_b],
            availability_by_path,
            channel_names=CHANNELS,
            statistic="meanstd",
            epsilon=1e-8,
            raw_transform_by_path=identity,
        )

        # Move each patient to its own scaling and store the result.
        patient_paths = {}
        patient_transforms = {}
        for key, path, signal in (
            ("001", path_a, self.patient_a),
            ("002", path_b, self.patient_b),
        ):
            patient_scaler = fit_channel_scaler(
                [path],
                availability_by_path,
                channel_names=CHANNELS,
                statistic="meanstd",
                epsilon=1e-8,
                raw_transform_by_path=identity,
            )
            document = build_scaler_document(
                mode="patient",
                statistic="meanstd",
                channel_names=CHANNELS,
                scalers={key: patient_scaler},
                epsilon=1e-8,
            )
            stored = apply_channel_scaler(
                signal,
                CHANNELS,
                self.availability,
                patient_scaler,
                document,
                "float32",
            )
            stored_path = write_recording(self.directory, f"p_{key}", stored)
            patient_paths[key] = stored_path
            patient_transforms[stored_path] = (
                np.asarray(patient_scaler["center"]),
                np.asarray(patient_scaler["scale"]),
            )

        # Now pool them back into one global scaler.
        refitted_global = fit_channel_scaler(
            list(patient_paths.values()),
            {
                path: self.availability
                for path in patient_paths.values()
            },
            channel_names=CHANNELS,
            statistic="meanstd",
            epsilon=1e-8,
            raw_transform_by_path=patient_transforms,
        )

        for channel in (0, 1):
            self.assertAlmostEqual(
                refitted_global["center"][channel],
                original_global["center"][channel],
                places=8,
                msg=f"center drifted on channel {channel}",
            )
            self.assertAlmostEqual(
                refitted_global["scale"][channel],
                original_global["scale"][channel],
                places=8,
                msg=f"scale drifted on channel {channel}",
            )

    def test_robust_statistic_resists_an_artifact_spike(self) -> None:
        """Median/IQR scaling is not dominated by a rare electrode pop."""
        contaminated = self.patient_a.copy()
        contaminated[0, :20] = 1.0  # a huge artifact relative to 1e-4 EEG
        path = write_recording(self.directory, "spike", contaminated)

        mean_std = fit_channel_scaler(
            [path],
            {path: self.availability},
            channel_names=CHANNELS,
            statistic="meanstd",
            epsilon=1e-8,
        )
        robust = fit_channel_scaler(
            [path],
            {path: self.availability},
            channel_names=CHANNELS,
            statistic="robust",
            epsilon=1e-8,
        )
        clean_scale = float(np.std(self.patient_a[0]))

        self.assertGreater(mean_std["scale"][0], 10.0 * clean_scale)
        self.assertLess(robust["scale"][0], 2.0 * clean_scale)

    def test_scaler_keys_follow_the_normalization_mode(self) -> None:
        """Each mode addresses its scalers by the right identifier."""
        self.assertEqual(
            scaler_key_for("global", subject="007", recording_id="rec-1"),
            GLOBAL_SCALER_KEY,
        )
        self.assertEqual(
            scaler_key_for("patient", subject="7", recording_id="rec-1"),
            "007",
        )
        self.assertEqual(
            scaler_key_for("recording", subject="007", recording_id="rec-1"),
            "rec-1",
        )
        with self.assertRaises(ValueError):
            scaler_key_for("session", subject="007", recording_id="rec-1")

    def test_missing_scaler_is_reported_rather_than_silently_substituted(
        self,
    ) -> None:
        """A recording with no fitted scaler must fail loudly."""
        document = build_scaler_document(
            mode="patient",
            statistic="meanstd",
            channel_names=CHANNELS,
            scalers={"001": {"center": [0, 0, 0], "scale": [1, 1, 1]}},
            epsilon=1e-8,
        )
        select_scaler(document, subject="001", recording_id="rec-1")
        with self.assertRaises(KeyError):
            select_scaler(document, subject="002", recording_id="rec-1")

    def test_legacy_global_scaler_file_is_upgraded_on_read(self) -> None:
        """The original mean/std JSON still loads and still applies."""
        legacy_path = self.directory / "global_channel_zscore.json"
        legacy_path.write_text(
            json.dumps(
                {
                    "channel_names": CHANNELS,
                    "mean": [0.0, 0.0, 0.0],
                    "std": [2.0, 4.0, 1.0],
                    "training_subjects": ["001"],
                    "training_samples_per_channel": [10, 10, 0],
                }
            ),
            encoding="utf-8",
        )
        document = load_scaler_document(legacy_path)
        self.assertEqual(document["normalization_mode"], "global")
        scaler = select_scaler(document, subject="001", recording_id="rec-1")
        signal = np.ones((3, 8), dtype=np.float32)
        standardized = apply_channel_scaler(
            signal,
            CHANNELS,
            self.availability,
            scaler,
            document,
            "float32",
        )
        self.assertTrue(np.allclose(standardized[0], 0.5))
        self.assertTrue(np.allclose(standardized[1], 0.25))
        self.assertTrue(np.allclose(standardized[2], 0.0))

    def test_document_round_trips_through_disk(self) -> None:
        """Saved scaler documents reload identically."""
        document = build_scaler_document(
            mode="recording",
            statistic="robust",
            channel_names=CHANNELS,
            scalers={"rec-1": {"center": [1, 2, 0], "scale": [3, 4, 1]}},
            epsilon=1e-8,
        )
        path = self.directory / "channel_scalers.json"
        save_scaler_document(document, path)
        self.assertEqual(load_scaler_document(path), document)

    def test_three_dimensional_windows_are_supported(self) -> None:
        """Windowed EEG uses the same scaler as continuous EEG."""
        scaler = {"center": [0.0, 0.0, 0.0], "scale": [2.0, 2.0, 1.0]}
        document = build_scaler_document(
            mode="patient",
            statistic="meanstd",
            channel_names=CHANNELS,
            scalers={"001": scaler},
            epsilon=1e-8,
        )
        windows = np.ones((4, 3, 10), dtype=np.float32)
        standardized = apply_channel_scaler(
            windows,
            CHANNELS,
            self.availability,
            scaler,
            document,
            "float32",
        )
        self.assertEqual(standardized.shape, (4, 3, 10))
        self.assertTrue(np.allclose(standardized[:, 0], 0.5))
        self.assertTrue(np.allclose(standardized[:, 2], 0.0))


if __name__ == "__main__":
    unittest.main()
