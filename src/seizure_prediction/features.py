"""Chunk-level spectral features for the streaming seizure-risk task.

Why features rather than raw EEG
--------------------------------
The EEGNet baseline learns a filter bank from scratch on 2,484 training
positives drawn from a few hundred seizures. That is a very small sample for a
convolutional encoder, and the measured result was chance on held-out patients.
Band power, Hjorth parameters and line length are the representation the seizure
literature actually uses, and they cost no parameters at all.

Why this also removes the amplitude confound
--------------------------------------------
Measured on the built dataset, median amplitude per patient spans ~44x after the
single global z-score -- impedance and electrode seating rather than physiology.
A multiplicative gain ``g`` on the signal scales every band power by ``g**2``:

    log P(g x) = log(g**2 P(x)) = 2 log g + log P(x)

so in *log* space a per-patient gain is a constant offset, identical across
bands and chunks. Centring each 45-minute window on its own mean removes it
exactly. Log power plus per-window centring is therefore invariant to
per-patient amplitude scale by construction, not by estimation.

Storage
-------
One decision is 540 chunks x 27 features = 58 KB instead of 8.3 MB of raw
float32, a ~150x reduction. The whole corpus of chunk features is under 1 GB, so
training becomes compute-bound rather than I/O-bound, which is what makes
patient-wise cross-validation affordable.
"""

from __future__ import annotations

import numpy as np

# Classical EEG bands. The upper edge stops at the 40 Hz bandpass corner.
BANDS: tuple[tuple[str, float, float], ...] = (
    ("delta", 0.5, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
    ("gamma", 30.0, 40.0),
)

# Guards a log of zero on an all-zero (absent) channel.
LOG_EPSILON = 1e-12


def feature_names(channel_names: tuple[str, ...]) -> list[str]:
    """Return the feature column names, in the order ``chunk_features`` emits."""
    names: list[str] = []
    for channel in channel_names:
        for band, _, _ in BANDS:
            names.append(f"{channel}::log_power_{band}")
        names.append(f"{channel}::log_total_power")
        names.append(f"{channel}::hjorth_mobility")
        names.append(f"{channel}::hjorth_complexity")
        names.append(f"{channel}::log_line_length")
    return names


def features_per_channel() -> int:
    """Number of features emitted for one channel."""
    return len(BANDS) + 4


def chunk_features(
    signal: np.ndarray,
    sampling_frequency: float,
    chunk_samples: int,
    availability: np.ndarray | None = None,
) -> np.ndarray:
    """Compute per-chunk features for one continuous recording.

    Parameters
    ----------
    signal:
        ``(channels, samples)`` continuous EEG.
    sampling_frequency:
        Sampling rate in Hz.
    chunk_samples:
        Samples per chunk; trailing samples that do not fill a chunk are dropped.
    availability:
        Optional ``(channels,)`` boolean mask. Absent channels are emitted as
        exact zeros rather than as the log of an all-zero signal.

    Returns
    -------
    np.ndarray
        ``(n_chunks, channels * features_per_channel())`` float32.
    """
    channels, samples = signal.shape
    n_chunks = samples // chunk_samples
    if n_chunks == 0:
        return np.zeros((0, channels * features_per_channel()), dtype=np.float32)

    if availability is None:
        availability = np.ones(channels, dtype=bool)
    availability = np.asarray(availability, dtype=bool)

    # (channels, n_chunks, chunk_samples)
    blocks = signal[:, : n_chunks * chunk_samples].reshape(
        channels, n_chunks, chunk_samples
    ).astype(np.float64, copy=False)

    # Hann-windowed periodogram. A single FFT per chunk is enough at this
    # resolution (5 s at 256 Hz -> 0.2 Hz bins) and is far cheaper than Welch.
    window = np.hanning(chunk_samples)
    windowed = blocks * window
    spectrum = np.fft.rfft(windowed, axis=-1)
    power = (np.abs(spectrum) ** 2) / (np.sum(window**2) * sampling_frequency)
    power[..., 1:-1] *= 2.0
    frequencies = np.fft.rfftfreq(chunk_samples, d=1.0 / sampling_frequency)

    # First difference and second difference, for the Hjorth descriptors.
    first = np.diff(blocks, axis=-1)
    second = np.diff(first, axis=-1)
    variance = blocks.var(axis=-1)
    variance_first = first.var(axis=-1)
    variance_second = second.var(axis=-1)

    safe_variance = np.maximum(variance, LOG_EPSILON)
    safe_variance_first = np.maximum(variance_first, LOG_EPSILON)
    mobility = np.sqrt(variance_first / safe_variance)
    complexity = np.sqrt(variance_second / safe_variance_first) / np.maximum(
        mobility, LOG_EPSILON
    )
    line_length = np.abs(first).sum(axis=-1)

    columns: list[np.ndarray] = []
    for channel in range(channels):
        for _, low, high in BANDS:
            selected = (frequencies >= low) & (frequencies < high)
            band_power = power[channel][:, selected].sum(axis=-1)
            columns.append(np.log(band_power + LOG_EPSILON))
        in_band = (frequencies >= BANDS[0][1]) & (frequencies < BANDS[-1][2])
        columns.append(np.log(power[channel][:, in_band].sum(axis=-1) + LOG_EPSILON))
        columns.append(mobility[channel])
        columns.append(complexity[channel])
        columns.append(np.log(line_length[channel] + LOG_EPSILON))

    features = np.stack(columns, axis=1).astype(np.float32)

    # An absent channel is stored as exact zeros upstream; its "features" are
    # meaningless, so emit zeros and let the availability mask carry the fact.
    per_channel = features_per_channel()
    for channel in range(channels):
        if not availability[channel]:
            features[:, channel * per_channel : (channel + 1) * per_channel] = 0.0

    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def normalize_window(
    window: np.ndarray,
    availability_columns: np.ndarray,
) -> np.ndarray:
    """Centre and scale a feature window on its own robust statistics.

    ``window`` is ``(n_chunks, n_features)``. Centring in log-power space is what
    removes the per-patient amplitude gain (see the module docstring); the IQR
    scaling additionally puts every feature on a comparable range so the model
    does not have to learn one.

    Columns belonging to absent channels are left at exactly zero.
    """
    if window.size == 0:
        return window

    quartiles = np.percentile(window, [25, 50, 75], axis=0)
    median = quartiles[1]
    spread = np.maximum(quartiles[2] - quartiles[0], 1e-6)
    normalized = (window - median) / spread
    normalized[:, ~availability_columns] = 0.0
    return normalized.astype(np.float32, copy=False)
