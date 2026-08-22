"""Channel standardization strategies for the SeizeIT2 pipeline.

The pipeline historically fitted a single global per-channel z-score from all
training patients.  In behind-the-ear wearable EEG, between-patient amplitude
differences (electrode impedance, skin, hair, device seating) are large and are
mostly nuisance, so one global scale lets a model key on patient identity rather
than on brain state.  This module adds per-patient and per-recording
alternatives.

Scope of the normalizations
---------------------------
``global``
    One center/scale per channel, fitted from training patients only.  This is
    the original behavior and the only mode with a train/validation/test
    asymmetry.
``patient``
    One center/scale per channel for every patient, fitted from that patient's
    own recordings.
``recording``
    One center/scale per channel for every recording.  This additionally
    removes drift between sessions, which matters here because SeizeIT2
    recordings span days and the wearable is re-applied between them.

Leakage
-------
``patient`` and ``recording`` scalers are fitted on the data they normalize,
including for validation and test patients.  No labels are used and no
information crosses between patients, so this is not train/test leakage in the
usual sense; it is standard subject-wise standardization.  It is, however,
*non-causal*: the statistics summarize a whole recording, so the scaling of an
early decision reflects samples that arrive later.  A deployed system would fit
these statistics on a calibration prefix instead.  This is the same class of
concern as the pipeline's zero-phase filtering and is recorded in the saved
scaler document so downstream analysis can account for it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


NORMALIZATION_MODES: tuple[str, ...] = ("global", "patient", "recording")
STATISTICS: tuple[str, ...] = ("meanstd", "robust")

GLOBAL_SCALER_KEY = "__global__"

# One block is (channels x BLOCK_SAMPLES) float64 during accumulation.
BLOCK_SAMPLES = 4_000_000

# Cap on samples retained per channel when estimating robust quantiles.
DEFAULT_MAX_QUANTILE_SAMPLES = 4_000_000

# Scale factor making the interquartile range consistent with the standard
# deviation of a normal distribution.
IQR_TO_SIGMA = 1.3489795003921634


def scaler_key_for(
    mode: str,
    *,
    subject: str,
    recording_id: str,
) -> str:
    """Return the scaler-document key that standardizes one recording."""
    if mode not in NORMALIZATION_MODES:
        raise ValueError(f"Unknown normalization mode: {mode!r}")
    if mode == "global":
        return GLOBAL_SCALER_KEY
    if mode == "patient":
        return str(subject).zfill(3)
    return str(recording_id)


def _iterate_blocks(
    signal: np.ndarray,
    block_samples: int,
) -> Iterable[np.ndarray]:
    """Yield contiguous time blocks of a ``(channels, samples)`` array."""
    total_samples = signal.shape[1]
    for start in range(0, total_samples, block_samples):
        yield np.asarray(
            signal[:, start : start + block_samples],
            dtype=np.float64,
        )


def fit_channel_scaler(
    array_paths: Iterable[Path],
    availability_by_path: dict[Path, np.ndarray],
    *,
    channel_names: list[str],
    statistic: str = "meanstd",
    epsilon: float = 1e-8,
    max_quantile_samples: int = DEFAULT_MAX_QUANTILE_SAMPLES,
    raw_transform_by_path: dict[Path, tuple[np.ndarray, np.ndarray]] | None = None,
) -> dict[str, Any]:
    """Fit one per-channel center/scale from a group of continuous recordings.

    Channels absent from a recording are stored as zero placeholders and carry
    no signal, so they are excluded from every accumulation via the recording's
    availability mask.  A channel that is unavailable in *all* of the supplied
    recordings receives a neutral center of zero and a scale of one; it stays
    zero after scaling.

    ``raw_transform_by_path`` maps each path to the ``(center, scale)`` that
    recovers raw filtered units from what is stored on disk, as
    ``raw = stored * scale + center``.  Supplying it makes the returned scaler
    describe raw units even when the inputs are already standardized, and it is
    required whenever the recordings in one group do not currently share a
    single scaler -- for example when re-fitting a global scaler over data that
    is presently normalized per patient.
    """
    if statistic not in STATISTICS:
        raise ValueError(f"Unknown statistic: {statistic!r}")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive.")
    if max_quantile_samples <= 0:
        raise ValueError("max_quantile_samples must be positive.")

    paths = [Path(path) for path in array_paths]
    if not paths:
        raise ValueError("At least one recording is required to fit a scaler.")

    channel_count = len(channel_names)
    signal_sum = np.zeros(channel_count, dtype=np.float64)
    signal_sum_squares = np.zeros(channel_count, dtype=np.float64)
    sample_count = np.zeros(channel_count, dtype=np.int64)
    reservoirs: list[list[np.ndarray]] = [[] for _ in range(channel_count)]

    # Reserve a per-recording share of the quantile budget so one long
    # recording cannot dominate a patient's robust statistics.
    per_path_quantile_budget = max(1, max_quantile_samples // len(paths))

    for path in paths:
        availability = np.asarray(
            availability_by_path[path],
            dtype=bool,
        )
        if availability.shape != (channel_count,):
            raise ValueError(
                f"Channel availability for {path} must contain one value per "
                f"canonical channel; found shape {availability.shape}."
            )

        signal = np.load(path, mmap_mode="r")
        if signal.ndim != 2 or signal.shape[0] != channel_count:
            raise ValueError(
                f"Expected shape ({channel_count}, samples) for {path}; "
                f"found {signal.shape}."
            )

        present = np.flatnonzero(availability)
        if len(present) == 0:
            raise ValueError(f"Recording {path} has no available channels.")

        raw_center = raw_scale = None
        if raw_transform_by_path is not None:
            if path not in raw_transform_by_path:
                raise KeyError(f"No raw transform was supplied for {path}.")
            transform_center, transform_scale = raw_transform_by_path[path]
            raw_center = np.asarray(transform_center, dtype=np.float64)[:, None]
            raw_scale = np.asarray(transform_scale, dtype=np.float64)[:, None]

        retained_per_channel = 0
        for block in _iterate_blocks(signal, BLOCK_SAMPLES):
            if raw_scale is not None:
                # Recover raw filtered units so groups whose members carry
                # different stored scalers still accumulate comparably.
                block = block * raw_scale + raw_center
            signal_sum[present] += block[present].sum(axis=1, dtype=np.float64)
            signal_sum_squares[present] += np.square(block[present]).sum(
                axis=1,
                dtype=np.float64,
            )
            sample_count[present] += block.shape[1]

            if statistic == "robust" and retained_per_channel < per_path_quantile_budget:
                # Systematic subsampling keeps quantile estimates accurate at a
                # small fraction of the memory of a full sort.
                stride = max(
                    1,
                    block.shape[1]
                    // max(1, per_path_quantile_budget // 8),
                )
                sampled = block[:, ::stride]
                for channel in present:
                    reservoirs[channel].append(
                        sampled[channel].astype(np.float32, copy=True)
                    )
                retained_per_channel += sampled.shape[1]

        del signal

    center = np.zeros(channel_count, dtype=np.float64)
    scale = np.ones(channel_count, dtype=np.float64)
    fitted_channels = sample_count > 0

    if statistic == "meanstd":
        mean = np.zeros(channel_count, dtype=np.float64)
        mean[fitted_channels] = (
            signal_sum[fitted_channels] / sample_count[fitted_channels]
        )
        variance = np.zeros(channel_count, dtype=np.float64)
        variance[fitted_channels] = (
            signal_sum_squares[fitted_channels] / sample_count[fitted_channels]
            - np.square(mean[fitted_channels])
        )
        center[fitted_channels] = mean[fitted_channels]
        scale[fitted_channels] = np.maximum(
            np.sqrt(np.maximum(variance[fitted_channels], 0.0)),
            epsilon,
        )
    else:
        for channel in range(channel_count):
            if not fitted_channels[channel]:
                continue
            values = np.concatenate(reservoirs[channel])
            lower, median, upper = np.quantile(values, [0.25, 0.5, 0.75])
            center[channel] = float(median)
            scale[channel] = max(
                float(upper - lower) / IQR_TO_SIGMA,
                epsilon,
            )

    return {
        "center": center.tolist(),
        "scale": scale.tolist(),
        "samples_per_channel": sample_count.tolist(),
        "fitted_channels": fitted_channels.astype(int).tolist(),
        "recordings": len(paths),
    }


def apply_channel_scaler(
    X: np.ndarray,
    channel_names: list[str],
    channel_availability: np.ndarray,
    scaler: dict[str, Any],
    document: dict[str, Any],
    output_dtype: str,
) -> np.ndarray:
    """Center and scale continuous or windowed EEG for one recording.

    Accepts both the ``center``/``scale`` keys written by this module and the
    legacy ``mean``/``std`` keys written by the original global scaler.
    """
    expected_channels = document["channel_names"]
    if list(channel_names) != list(expected_channels):
        raise ValueError(
            "Channel order does not match the fitted scaler. "
            f"Expected: {expected_channels}; found: {channel_names}"
        )

    center_values = scaler.get("center", scaler.get("mean"))
    scale_values = scaler.get("scale", scaler.get("std"))
    if center_values is None or scale_values is None:
        raise ValueError("Scaler is missing center/scale parameters.")

    values = np.asarray(X, dtype=np.float32)
    availability = np.asarray(channel_availability, dtype=bool)
    if availability.shape != (len(channel_names),):
        raise ValueError(
            "Channel availability must contain one value per canonical channel."
        )

    center = np.asarray(center_values, dtype=np.float32)
    scale = np.asarray(scale_values, dtype=np.float32)
    if np.any(scale <= 0.0):
        raise ValueError("Scaler contains a non-positive scale value.")

    if values.ndim == 2:
        standardized = (values - center[:, None]) / scale[:, None]
        standardized[~availability, :] = 0.0
    elif values.ndim == 3:
        standardized = (
            values - center[None, :, None]
        ) / scale[None, :, None]
        standardized[:, ~availability, :] = 0.0
    else:
        raise ValueError(
            "Scaling expects 2-D continuous EEG or 3-D windowed EEG, "
            f"found shape {values.shape}."
        )

    return standardized.astype(output_dtype, copy=False)


def build_scaler_document(
    *,
    mode: str,
    statistic: str,
    channel_names: list[str],
    scalers: dict[str, dict[str, Any]],
    epsilon: float,
    training_subjects: list[str] | None = None,
) -> dict[str, Any]:
    """Assemble the self-describing scaler document written to disk."""
    if mode not in NORMALIZATION_MODES:
        raise ValueError(f"Unknown normalization mode: {mode!r}")
    if statistic not in STATISTICS:
        raise ValueError(f"Unknown statistic: {statistic!r}")
    if not scalers:
        raise ValueError("A scaler document must contain at least one scaler.")

    document: dict[str, Any] = {
        "normalization_mode": mode,
        "statistic": statistic,
        "channel_names": list(channel_names),
        "epsilon": float(epsilon),
        "scaler_count": len(scalers),
        "fitted_on": (
            "training patients only"
            if mode == "global"
            else f"each {mode}'s own recordings"
        ),
        "non_causal_statistics": mode != "global",
        "scalers": scalers,
    }
    if mode == "global":
        if not training_subjects:
            raise ValueError("Global mode must record its training subjects.")
        document["training_subjects"] = sorted(training_subjects)
    return document


def save_scaler_document(document: dict[str, Any], output_path: Path) -> None:
    """Write a scaler document as indented JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as scaler_file:
        json.dump(document, scaler_file, indent=2)


def load_scaler_document(scaler_path: Path) -> dict[str, Any]:
    """Read a scaler document, upgrading the legacy global-only format."""
    with Path(scaler_path).open("r", encoding="utf-8") as scaler_file:
        document = json.load(scaler_file)

    if "scalers" in document:
        return document

    # Legacy data/interim/scaler_parameters/global_channel_zscore.json.
    return {
        "normalization_mode": "global",
        "statistic": "meanstd",
        "channel_names": document["channel_names"],
        "epsilon": None,
        "scaler_count": 1,
        "fitted_on": "training patients only",
        "non_causal_statistics": False,
        "training_subjects": document.get("training_subjects", []),
        "scalers": {
            GLOBAL_SCALER_KEY: {
                "center": document["mean"],
                "scale": document["std"],
                "samples_per_channel": document.get(
                    "training_samples_per_channel"
                ),
            }
        },
    }


def select_scaler(
    document: dict[str, Any],
    *,
    subject: str,
    recording_id: str,
) -> dict[str, Any]:
    """Return the scaler that standardizes one recording."""
    mode = document["normalization_mode"]
    key = scaler_key_for(mode, subject=subject, recording_id=recording_id)
    scalers = document["scalers"]
    if key not in scalers:
        raise KeyError(
            f"No {mode} scaler was fitted for {key!r}. The scaler document "
            "and the processed dataset are out of sync."
        )
    return scalers[key]
