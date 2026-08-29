from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from seizure_prediction.handcrafted_features import (
    FEATURE_NAMES,
    build_decision_feature_matrix,
    decision_feature_names,
    extract_ava_feature_batch,
    feature_cache_path,
    load_channel_availability,
)


def test_missing_channel_features_are_finite_zeros() -> None:
    rng = np.random.default_rng(7)
    windows = rng.normal(size=(2, 3, 2560)).astype(np.float32)
    windows[:, 1, :] = 0.0
    features = extract_ava_feature_batch(
        windows,
        np.array([True, False, True]),
    )
    assert features.shape == (2, 3, len(FEATURE_NAMES))
    assert np.isfinite(features).all()
    np.testing.assert_array_equal(features[:, 1, :], 0.0)


def test_channel_availability_loads_json(tmp_path: Path) -> None:
    path = tmp_path / "availability.json"
    path.write_text("[1, 0, 1]", encoding="utf-8")
    np.testing.assert_array_equal(
        load_channel_availability(path), np.array([True, False, True])
    )


def test_alpha_sine_has_largest_alpha_bandpower() -> None:
    time = np.arange(2560, dtype=np.float64) / 256.0
    sine = np.sin(2.0 * np.pi * 10.0 * time)
    windows = np.zeros((1, 3, len(time)), dtype=np.float32)
    windows[0, 0] = sine
    features = extract_ava_feature_batch(
        windows,
        np.array([True, False, False]),
    )
    bandpowers = features[0, 0, 5:10]
    assert int(np.argmax(bandpowers)) == 2


def test_decision_summary_uses_the_configured_history(tmp_path: Path) -> None:
    signal_path = tmp_path / "processed" / "train" / "recording_X.npy"
    signal_path.parent.mkdir(parents=True)
    np.save(signal_path, np.zeros((3, 46), dtype=np.float32))
    availability_path = tmp_path / "availability.npy"
    np.save(availability_path, np.array([True, False, True]))
    cache_root = tmp_path / "cache"
    cache_path = feature_cache_path(signal_path, cache_root)
    cache_path.parent.mkdir(parents=True)
    minute_features = np.zeros((46, 3, len(FEATURE_NAMES)), dtype=np.float32)
    minute_features[:, 0, 0] = np.arange(46, dtype=np.float32)
    minute_features[:, 2, 0] = 2.0
    np.save(cache_path, minute_features)
    examples = pd.DataFrame(
        {
            "X_path": [str(signal_path), str(signal_path)],
            "channel_availability_path": [
                str(availability_path),
                str(availability_path),
            ],
            "history_start_sample": [0, 256 * 60],
            "decision_end_sample": [45 * 256 * 60, 46 * 256 * 60],
        },
        index=[10, 20],
    )
    matrix = build_decision_feature_matrix(
        examples,
        cache_root,
        history_minutes=45,
    )
    names = decision_feature_names(("BTE_LEFT", "BTE_RIGHT", "CROSS_HEAD"))
    assert matrix.shape == (2, len(names))
    np.testing.assert_allclose(matrix[:, 0], [22.0, 23.0])
    np.testing.assert_array_equal(matrix[:, -3:], [[1, 0, 1], [1, 0, 1]])
