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


# A present channel whose log total power sits at or near the log(1e-12) floor
# of -27.59 is a dead electrode, not EEG. The normal 1st-99th percentile range
# of log_total_power is -6.04 to +3.86, so -20 separates the two cleanly.
DEAD_CHANNEL_LOG_POWER = -20.0

# normalize_window divides by an IQR floored at ABSOLUTE_IQR_FLOOR. On a window
# where a present channel is flat, that produced values up to 3.0e7 (0.27% of
# real decision windows). Clipping the standardised output bounds the damage
# regardless of how degenerate the input is; +-20 robust IQRs is far outside any
# legitimate value (measured p99 of max|value| = 90.5 before clipping, but the
# legitimate median max was 14.5).
ABSOLUTE_IQR_FLOOR = 1e-3
OUTPUT_CLIP = 20.0


def dead_channel_columns(
    window: np.ndarray,
    availability_columns: np.ndarray,
    channel_count: int = 3,
) -> np.ndarray:
    """Return a per-column mask marking channels that are dead in this window.

    A dead electrode emits exact zeros, which is finite and unannotated, so it
    passes every cleanliness check in the pipeline: the non-finite test, the
    ``bad*`` annotation test, and the event-overlap tests. Measured consequence:
    0.28% of NEGATIVE decision windows contain a >=99% dead present channel
    versus 0.00% of positive windows, i.e. a small spurious cue pointing at
    "dead electrode implies no seizure".
    """
    per_channel = window.shape[1] // channel_count
    dead = np.zeros(window.shape[1], dtype=bool)
    for channel in range(channel_count):
        lo = channel * per_channel
        hi = lo + per_channel
        if not availability_columns[lo]:
            continue
        # log_total_power is the 6th feature of each channel block.
        total_power = window[:, lo + len(BANDS)]
        if np.median(total_power) < DEAD_CHANNEL_LOG_POWER:
            dead[lo:hi] = True
    return dead


def normalize_window(
    window: np.ndarray,
    availability_columns: np.ndarray,
    recording_median: np.ndarray | None = None,
) -> np.ndarray:
    """Centre and scale a feature window on its own robust statistics.

    Centring in log-power space is what removes the per-patient amplitude gain
    (see the module docstring); the IQR scaling puts every feature on a
    comparable range.

    Three corrections over the naive version, each from a measured defect:

    * The IQR floor is raised from 1e-6 to 1e-3 and the output is clipped. The
      old floor let a flat present channel produce values up to 3.0e7; with a
      batch of 64 and gradient clipping at 5.0, a single such sample dominated
      the direction of roughly one clipped gradient in four.
    * Channels that are dead in this window are zeroed, so a dead electrode
      cannot act as a label cue.
    * When ``recording_median`` is supplied, the window's own level relative to
      that baseline is returned alongside the standardised window. Median
      centring removes the level, and the level is where vigilance lives --
      measured, window normalisation costs 63% of the asleep-vs-awake
      discrimination (AUC 0.765 -> 0.599). Referencing to the recording rather
      than to a global constant keeps the state while still removing the
      per-patient gain.
    """
    if window.size == 0:
        return window

    quartiles = np.percentile(window, [25, 50, 75], axis=0)
    median = quartiles[1]
    spread = np.maximum(quartiles[2] - quartiles[0], ABSOLUTE_IQR_FLOOR)
    normalized = (window - median) / spread
    np.clip(normalized, -OUTPUT_CLIP, OUTPUT_CLIP, out=normalized)

    present = np.asarray(availability_columns, dtype=bool)
    dead = dead_channel_columns(window, present)
    normalized[:, ~present] = 0.0
    normalized[:, dead] = 0.0

    if recording_median is None:
        return normalized.astype(np.float32, copy=False)

    # Level relative to this recording's own baseline: patient gain cancels,
    # within-patient state (sleep/wake, drowsiness) survives.
    level = (median - recording_median)
    level[~present] = 0.0
    level[dead] = 0.0
    level = np.clip(level, -OUTPUT_CLIP, OUTPUT_CLIP)
    broadcast = np.repeat(level[None, :], normalized.shape[0], axis=0)
    return np.concatenate([normalized, broadcast], axis=1).astype(np.float32, copy=False)
