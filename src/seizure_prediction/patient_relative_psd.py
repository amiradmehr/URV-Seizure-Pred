"""Patient-relative spectral-density utilities for preictal EEG analysis."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy import signal


BANDS_HZ: tuple[tuple[str, float, float], ...] = (
    ("delta", 0.5, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
    ("gamma", 30.0, 40.0),
    ("broadband", 0.5, 40.0),
)


def compute_band_power_density(
    windows: np.ndarray,
    channel_availability: np.ndarray,
    *,
    sampling_frequency: float = 256.0,
    welch_segment_seconds: float = 2.0,
) -> np.ndarray:
    """Return mean Welch PSD per band for ``(window, channel, sample)`` EEG.

    Mean spectral density is integrated band power divided by band width. The
    result therefore retains PSD units and can be compared across differently
    sized bands. Unavailable canonical channels are represented by ``NaN``.
    """
    values = np.asarray(windows)
    availability = np.asarray(channel_availability, dtype=bool)
    if values.ndim != 3:
        raise ValueError("windows must have shape (windows, channels, samples).")
    if availability.shape != (values.shape[1],):
        raise ValueError("channel_availability must contain one value per channel.")
    if sampling_frequency <= 0.0:
        raise ValueError("sampling_frequency must be positive.")
    if welch_segment_seconds <= 0.0:
        raise ValueError("welch_segment_seconds must be positive.")
    if values.shape[-1] < 3:
        raise ValueError("PSD windows must contain at least three samples.")
    if not np.isfinite(values).all():
        raise ValueError("PSD windows must contain only finite samples.")

    nperseg = min(
        int(round(welch_segment_seconds * sampling_frequency)),
        values.shape[-1],
    )
    frequencies, psd = signal.welch(
        values,
        fs=sampling_frequency,
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend="constant",
        scaling="density",
        axis=-1,
    )
    band_densities: list[np.ndarray] = []
    for _, low_hz, high_hz in BANDS_HZ:
        selected = (frequencies >= low_hz) & (frequencies <= high_hz)
        if selected.sum() < 2:
            raise ValueError(
                f"Welch resolution cannot represent the {low_hz:g}-{high_hz:g} Hz band."
            )
        integrated_power = np.trapezoid(
            psd[..., selected],
            frequencies[selected],
            axis=-1,
        )
        band_densities.append(integrated_power / (high_hz - low_hz))

    output = np.stack(band_densities, axis=-1).astype(np.float64, copy=False)
    output[:, ~availability, :] = np.nan
    if not np.isfinite(output[:, availability, :]).all():
        raise ValueError("Welch analysis produced nonfinite power densities.")
    return output


def decibels_relative_to_baseline(
    power_density: np.ndarray,
    baseline_power_density: np.ndarray,
    *,
    epsilon: float | None = None,
) -> np.ndarray:
    """Return ``10*log10(power / baseline)`` with finite positive flooring."""
    power = np.asarray(power_density, dtype=np.float64)
    baseline = np.asarray(baseline_power_density, dtype=np.float64)
    if epsilon is None:
        epsilon = np.finfo(np.float64).tiny
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive.")
    if np.any(np.isfinite(power) & (power < 0.0)):
        raise ValueError("power_density cannot contain negative values.")
    if np.any(np.isfinite(baseline) & (baseline < 0.0)):
        raise ValueError("baseline_power_density cannot contain negative values.")
    return 10.0 * np.log10(
        np.maximum(power, epsilon) / np.maximum(baseline, epsilon)
    )


def preictal_bin_label(
    minutes_before_onset: float,
    *,
    preictal_minutes: int = 60,
    bin_minutes: int = 10,
) -> str:
    """Return an ordered label such as ``60-50`` for a preictal minute."""
    if preictal_minutes <= 0 or bin_minutes <= 0:
        raise ValueError("preictal_minutes and bin_minutes must be positive.")
    if preictal_minutes % bin_minutes != 0:
        raise ValueError("preictal_minutes must be divisible by bin_minutes.")
    value = float(minutes_before_onset)
    if not 0.0 < value <= float(preictal_minutes):
        raise ValueError("minutes_before_onset must lie in (0, preictal_minutes].")
    bin_index = int(np.ceil(value / bin_minutes)) - 1
    lower = bin_index * bin_minutes
    upper = lower + bin_minutes
    return f"{upper}-{lower}"


def interval_overlaps_any(
    start_seconds: float,
    stop_seconds: float,
    excluded_intervals: Iterable[tuple[float, float]],
) -> bool:
    """Return whether a half-open interval overlaps any excluded interval."""
    if stop_seconds <= start_seconds:
        raise ValueError("stop_seconds must be greater than start_seconds.")
    return any(
        start_seconds < excluded_stop and stop_seconds > excluded_start
        for excluded_start, excluded_stop in excluded_intervals
    )


def sample_evenly_across_recordings(
    candidates: pd.DataFrame,
    count: int,
    *,
    seed: int,
) -> pd.DataFrame:
    """Select deterministic baseline rows in round-robin recording order."""
    if "recording_id" not in candidates.columns:
        raise ValueError("candidates must contain recording_id.")
    if count < 0:
        raise ValueError("count cannot be negative.")
    if count == 0 or candidates.empty:
        return candidates.iloc[:0].copy()
    count = min(count, len(candidates))
    rng = np.random.default_rng(seed)
    queues = {
        str(recording_id): rng.permutation(group.index.to_numpy(dtype=np.int64))
        for recording_id, group in candidates.groupby("recording_id", sort=True)
    }
    recording_ids = list(queues)
    rng.shuffle(recording_ids)
    selected: list[int] = []
    position = 0
    while len(selected) < count:
        added = False
        for recording_id in recording_ids:
            indices = queues[recording_id]
            if position < len(indices):
                selected.append(int(indices[position]))
                added = True
                if len(selected) == count:
                    break
        if not added:
            break
        position += 1
    return candidates.loc[selected].reset_index(drop=True)
