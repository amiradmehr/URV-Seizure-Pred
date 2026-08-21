import unittest

import numpy as np
import pandas as pd

from seizure_prediction.patient_relative_psd import (
    BANDS_HZ,
    compute_band_power_density,
    decibels_relative_to_baseline,
    interval_overlaps_any,
    preictal_bin_label,
    sample_evenly_across_recordings,
)


class PatientRelativePsdTests(unittest.TestCase):
    def test_alpha_power_tracks_signal_amplitude_in_decibels(self) -> None:
        sampling_frequency = 256.0
        time = np.arange(int(60 * sampling_frequency)) / sampling_frequency
        baseline = np.sin(2.0 * np.pi * 10.0 * time)
        doubled = 2.0 * baseline
        windows = np.stack([baseline, doubled])[:, None, :]
        density = compute_band_power_density(
            windows,
            np.array([True]),
            sampling_frequency=sampling_frequency,
        )
        alpha_index = [name for name, _, _ in BANDS_HZ].index("alpha")
        change_db = decibels_relative_to_baseline(
            density[1, 0, alpha_index], density[0, 0, alpha_index]
        )
        self.assertAlmostEqual(float(change_db), 6.0206, places=3)

    def test_unavailable_channels_are_nan(self) -> None:
        windows = np.ones((2, 3, 512), dtype=np.float32)
        density = compute_band_power_density(
            windows,
            np.array([True, False, True]),
        )
        self.assertTrue(np.isnan(density[:, 1, :]).all())
        self.assertTrue(np.isfinite(density[:, [0, 2], :]).all())

    def test_preictal_bins_approach_onset(self) -> None:
        self.assertEqual(preictal_bin_label(59.5), "60-50")
        self.assertEqual(preictal_bin_label(50.0), "50-40")
        self.assertEqual(preictal_bin_label(0.5), "10-0")

    def test_interval_overlap_uses_half_open_boundaries(self) -> None:
        exclusions = [(100.0, 200.0)]
        self.assertFalse(interval_overlaps_any(0.0, 100.0, exclusions))
        self.assertTrue(interval_overlaps_any(99.0, 101.0, exclusions))
        self.assertFalse(interval_overlaps_any(200.0, 201.0, exclusions))

    def test_baseline_sampling_spreads_across_recordings(self) -> None:
        candidates = pd.DataFrame(
            {
                "recording_id": ["a"] * 10 + ["b"] * 2,
                "value": np.arange(12),
            }
        )
        sampled = sample_evenly_across_recordings(candidates, 4, seed=42)
        self.assertEqual(len(sampled), 4)
        self.assertEqual(sampled["recording_id"].value_counts().to_dict(), {"a": 2, "b": 2})
        repeated = sample_evenly_across_recordings(candidates, 4, seed=42)
        pd.testing.assert_frame_equal(sampled, repeated)


if __name__ == "__main__":
    unittest.main()
