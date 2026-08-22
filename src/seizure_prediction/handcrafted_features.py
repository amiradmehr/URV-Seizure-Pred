"""Robust AVA-style handcrafted features for seizure-risk decisions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

from seizure_prediction.datasets import resolve_stored_path


FEATURE_NAMES: tuple[str, ...] = (
    "rms",
    "variance",
    "skewness",
    "kurtosis",
    "histogram_entropy",
    "delta_power",
    "theta_power",
    "alpha_power",
    "beta_power",
    "gamma_power",
    "hjorth_activity",
    "hjorth_mobility",
    "hjorth_complexity",
)

BANDS_HZ: tuple[tuple[str, float, float], ...] = (
    ("delta", 0.5, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
    ("gamma", 30.0, 40.0),
)

SUMMARY_NAMES: tuple[str, ...] = ("mean", "std", "recent_minus_early")


def load_channel_availability(path: str | Path) -> np.ndarray:
    """Load a saved channel mask from the pipeline's JSON or NumPy format."""
    resolved = resolve_stored_path(path)
    if resolved.suffix.lower() == ".json":
        values = json.loads(resolved.read_text(encoding="utf-8"))
        return np.asarray(values, dtype=bool)
    return np.load(resolved).astype(bool, copy=False)


def feature_cache_path(signal_path: str | Path, cache_root: Path) -> Path:
    """Return the stable cache path for one processed recording."""
    resolved = resolve_stored_path(signal_path)
    return cache_root / resolved.parent.name / f"{resolved.stem}_features.npy"


def feature_coverage_path(signal_path: str | Path, cache_root: Path) -> Path:
    """Return the cached whole-minute coverage mask for one recording."""
    feature_path = feature_cache_path(signal_path, cache_root)
    return feature_path.with_name(f"{feature_path.stem}_coverage.npy")


def _histogram_entropy(values: np.ndarray, bins: int) -> float:
    """Return a finite amplitude-histogram entropy for one signal window."""
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    if not np.isfinite(minimum) or not np.isfinite(maximum):
        raise ValueError("Feature windows must contain only finite samples.")
    if maximum <= minimum:
        return 0.0
    counts, _ = np.histogram(values, bins=bins, range=(minimum, maximum))
    probabilities = counts[counts > 0].astype(np.float64)
    probabilities /= probabilities.sum()
    return float(-np.sum(probabilities * np.log(probabilities)))


def extract_ava_feature_batch(
    windows: np.ndarray,
    channel_availability: np.ndarray,
    *,
    sampling_frequency: float = 256.0,
    entropy_bins: int = 64,
) -> np.ndarray:
    """Extract AVA's 13 features from windows shaped (window, channel, sample).

    Constant or unavailable channels are represented by finite zeros. This is
    important for SeizeIT2, where every recording has one zero-filled canonical
    channel and a separate availability mask.
    """
    values = np.asarray(windows, dtype=np.float64)
    availability = np.asarray(channel_availability, dtype=bool)
    if values.ndim != 3:
        raise ValueError("windows must have shape (windows, channels, samples).")
    if availability.shape != (values.shape[1],):
        raise ValueError("channel_availability must contain one value per channel.")
    if values.shape[-1] < 3:
        raise ValueError("Feature windows must contain at least three samples.")
    if sampling_frequency <= 0.0:
        raise ValueError("sampling_frequency must be positive.")
    if entropy_bins < 2:
        raise ValueError("entropy_bins must be at least two.")
    if not np.isfinite(values).all():
        raise ValueError("Feature windows must contain only finite samples.")

    centered = values - values.mean(axis=-1, keepdims=True)
    variance = np.mean(np.square(centered), axis=-1)
    scale = np.sqrt(variance)
    stable = scale > np.finfo(np.float64).eps
    skewness = np.zeros_like(variance)
    kurtosis = np.zeros_like(variance)
    skewness[stable] = (
        np.mean(np.power(centered, 3), axis=-1)[stable]
        / np.power(scale[stable], 3)
    )
    kurtosis[stable] = (
        np.mean(np.power(centered, 4), axis=-1)[stable]
        / np.power(variance[stable], 2)
        - 3.0
    )
    rms = np.sqrt(np.mean(np.square(values), axis=-1))

    entropy = np.zeros_like(variance)
    for window_index in range(values.shape[0]):
        for channel_index in np.flatnonzero(availability):
            entropy[window_index, channel_index] = _histogram_entropy(
                values[window_index, channel_index],
                entropy_bins,
            )

    frequencies, power_spectral_density = signal.welch(
        values,
        fs=sampling_frequency,
        nperseg=min(int(round(2.0 * sampling_frequency)), values.shape[-1]),
        axis=-1,
    )
    band_powers: list[np.ndarray] = []
    for _, low_hz, high_hz in BANDS_HZ:
        selected = (frequencies >= low_hz) & (frequencies <= high_hz)
        band_powers.append(
            np.trapezoid(
                power_spectral_density[..., selected],
                frequencies[selected],
                axis=-1,
            )
        )

    first_difference = np.diff(values, axis=-1)
    second_difference = np.diff(first_difference, axis=-1)
    first_variance = np.var(first_difference, axis=-1)
    second_variance = np.var(second_difference, axis=-1)
    mobility = np.zeros_like(variance)
    complexity = np.zeros_like(variance)
    positive_variance = variance > np.finfo(np.float64).eps
    mobility[positive_variance] = np.sqrt(
        first_variance[positive_variance] / variance[positive_variance]
    )
    valid_complexity = (
        positive_variance
        & (first_variance > np.finfo(np.float64).eps)
        & (mobility > np.finfo(np.float64).eps)
    )
    complexity[valid_complexity] = (
        np.sqrt(
            second_variance[valid_complexity]
            / first_variance[valid_complexity]
        )
        / mobility[valid_complexity]
    )

    features = np.stack(
        [
            rms,
            variance,
            skewness,
            kurtosis,
            entropy,
            *band_powers,
            variance,
            mobility,
            complexity,
        ],
        axis=-1,
    )
    features[:, ~availability, :] = 0.0
    if features.shape[-1] != len(FEATURE_NAMES):
        raise RuntimeError("Handcrafted feature width is inconsistent.")
    if not np.isfinite(features).all():
        raise ValueError("Handcrafted feature extraction produced nonfinite values.")
    return features.astype(np.float32, copy=False)


def build_decision_feature_matrix(
    examples: pd.DataFrame,
    cache_root: Path,
    *,
    sampling_frequency: float = 256.0,
    minute_seconds: float = 60.0,
    history_minutes: int = 45,
    recent_minutes: int = 5,
) -> np.ndarray:
    """Summarize cached minute features for every 45-minute decision."""
    required = {
        "X_path",
        "channel_availability_path",
        "history_start_sample",
        "decision_end_sample",
    }
    missing = required - set(examples.columns)
    if missing:
        raise ValueError(f"Decision examples are missing columns: {sorted(missing)}")
    if examples.empty:
        raise ValueError("Decision examples cannot be empty.")
    if history_minutes <= 0 or not 0 < recent_minutes <= history_minutes:
        raise ValueError("Invalid history or recent summary duration.")

    normalized_examples = examples.reset_index(drop=True)
    samples_per_minute = int(round(sampling_frequency * minute_seconds))
    channel_count: int | None = None
    output: np.ndarray | None = None

    for signal_path, group in normalized_examples.groupby("X_path", sort=False):
        cache_path = feature_cache_path(signal_path, cache_root)
        if not cache_path.exists():
            raise FileNotFoundError(f"Missing handcrafted feature cache: {cache_path}")
        minute_features = np.load(cache_path, mmap_mode="r")
        if minute_features.ndim != 3 or minute_features.shape[-1] != len(FEATURE_NAMES):
            raise ValueError(f"Invalid handcrafted cache shape: {cache_path}")
        if channel_count is None:
            channel_count = int(minute_features.shape[1])
            output_width = (
                len(SUMMARY_NAMES) * channel_count * len(FEATURE_NAMES)
                + channel_count
            )
            output = np.empty((len(normalized_examples), output_width), dtype=np.float32)
        elif minute_features.shape[1] != channel_count:
            raise ValueError("Handcrafted caches use inconsistent channel counts.")

        start_samples = group["history_start_sample"].to_numpy(dtype=np.int64)
        end_samples = group["decision_end_sample"].to_numpy(dtype=np.int64)
        if (
            np.any(start_samples % samples_per_minute != 0)
            or np.any(end_samples % samples_per_minute != 0)
        ):
            raise ValueError("Decision histories must align to whole minutes.")
        starts = start_samples // samples_per_minute
        ends = end_samples // samples_per_minute
        if np.any(ends - starts != history_minutes):
            raise ValueError("Decision histories have the wrong minute duration.")
        if np.any(starts < 0) or np.any(ends > minute_features.shape[0]):
            raise ValueError("Decision history indexes outside its feature cache.")
        coverage_path = feature_coverage_path(signal_path, cache_root)
        if coverage_path.exists():
            coverage = np.load(coverage_path, mmap_mode="r").astype(bool, copy=False)
            if coverage.shape != (minute_features.shape[0],):
                raise ValueError(f"Invalid feature coverage shape: {coverage_path}")
            coverage_prefix = np.concatenate(
                [np.array([0], dtype=np.int64), np.cumsum(coverage, dtype=np.int64)]
            )
            if np.any(coverage_prefix[ends] - coverage_prefix[starts] != history_minutes):
                raise ValueError("A decision history is missing cached minute features.")

        flattened = np.asarray(minute_features, dtype=np.float64).reshape(
            minute_features.shape[0], -1
        )
        prefix = np.vstack(
            [
                np.zeros((1, flattened.shape[1]), dtype=np.float64),
                np.cumsum(flattened, axis=0),
            ]
        )
        square_prefix = np.vstack(
            [
                np.zeros((1, flattened.shape[1]), dtype=np.float64),
                np.cumsum(np.square(flattened), axis=0),
            ]
        )
        means = (prefix[ends] - prefix[starts]) / history_minutes
        second_moments = (
            square_prefix[ends] - square_prefix[starts]
        ) / history_minutes
        standard_deviations = np.sqrt(
            np.maximum(second_moments - np.square(means), 0.0)
        )
        early_means = (prefix[starts + recent_minutes] - prefix[starts]) / recent_minutes
        recent_means = (prefix[ends] - prefix[ends - recent_minutes]) / recent_minutes
        changes = recent_means - early_means

        availability = load_channel_availability(
            str(group["channel_availability_path"].iloc[0])
        ).astype(np.float32, copy=False)
        if availability.shape != (channel_count,):
            raise ValueError("Channel availability has the wrong shape.")
        availability_rows = np.broadcast_to(availability, (len(group), channel_count))
        group_features = np.concatenate(
            [means, standard_deviations, changes, availability_rows], axis=1
        )
        assert output is not None
        output[group.index.to_numpy(dtype=np.int64)] = group_features.astype(
            np.float32, copy=False
        )

    assert output is not None
    if not np.isfinite(output).all():
        raise ValueError("Decision feature summaries contain nonfinite values.")
    return output


def decision_feature_names(channel_names: tuple[str, ...]) -> list[str]:
    """Return stable column names for the summarized classifier input."""
    names = [
        f"{summary}__{channel}__{feature}"
        for summary in SUMMARY_NAMES
        for channel in channel_names
        for feature in FEATURE_NAMES
    ]
    names.extend(f"available__{channel}" for channel in channel_names)
    return names
