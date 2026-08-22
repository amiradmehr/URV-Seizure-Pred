"""Event-level alarm metrics for streaming seizure-risk predictions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EventEvaluation:
    """One threshold's aggregate metrics and auditable detail tables."""

    metrics: dict[str, float | int]
    alarm_episodes: pd.DataFrame
    seizure_events: pd.DataFrame
    per_subject: pd.DataFrame


def prepare_target_seizures(
    seizure_manifest: pd.DataFrame,
    target_seizure_ids: set[str],
) -> pd.DataFrame:
    """Return exact onset metadata for the seizures represented by labels."""
    required_columns = {"seizure_id", "subject", "onset_seconds"}
    missing_columns = required_columns - set(seizure_manifest.columns)
    if missing_columns:
        raise ValueError(
            "Seizure manifest is missing columns: "
            f"{sorted(missing_columns)}"
        )
    if not target_seizure_ids:
        raise ValueError("No target seizures were found in the evaluation split.")

    seizures = seizure_manifest[
        seizure_manifest["seizure_id"].astype(str).isin(target_seizure_ids)
    ].copy()
    found_ids = set(seizures["seizure_id"].astype(str))
    missing_ids = sorted(target_seizure_ids - found_ids)
    if missing_ids:
        raise ValueError(
            "Target seizures are missing from seizure_manifest.csv: "
            f"{missing_ids}"
        )
    if seizures["seizure_id"].astype(str).duplicated().any():
        raise ValueError("The seizure manifest contains duplicate seizure IDs.")

    seizures["seizure_id"] = seizures["seizure_id"].astype(str)
    seizures["subject"] = seizures["subject"].astype(str).str.zfill(3)
    seizures["onset_seconds"] = pd.to_numeric(
        seizures["onset_seconds"],
        errors="raise",
    )
    seizures["recording_id"] = seizures["seizure_id"].str.rsplit(
        "_seizure-",
        n=1,
    ).str[0]
    return seizures.sort_values(
        ["subject", "recording_id", "onset_seconds"],
    ).reset_index(drop=True)


def threshold_grid(
    probabilities: np.ndarray | pd.Series,
    number_of_thresholds: int,
) -> np.ndarray:
    """Return score-rank thresholds plus explicit all/no-alarm endpoints."""
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("Probabilities must be a nonempty one-dimensional array.")
    if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("Probabilities must be finite values in [0, 1].")
    if number_of_thresholds < 2:
        raise ValueError("number_of_thresholds must be at least two.")

    unique_values = np.unique(values)
    indices = np.linspace(
        0,
        len(unique_values) - 1,
        min(number_of_thresholds, len(unique_values)),
    ).round().astype(np.int64)
    candidates = np.concatenate(
        [
            unique_values[indices],
            np.array(
                [
                    0.0,
                    np.nextafter(float(unique_values[-1]), np.inf),
                ]
            ),
        ]
    )
    return np.unique(candidates)[::-1]


def _validate_decisions(decisions: pd.DataFrame) -> pd.DataFrame:
    """Normalize the columns used by the event simulation."""
    required_columns = {
        "recording_id",
        "subject",
        "decision_time_seconds",
        "label",
        "probability",
    }
    missing_columns = required_columns - set(decisions.columns)
    if missing_columns:
        raise ValueError(
            "Prediction table is missing columns: "
            f"{sorted(missing_columns)}"
        )
    if decisions.empty:
        raise ValueError("The prediction table is empty.")

    normalized = decisions.copy()
    normalized["recording_id"] = normalized["recording_id"].astype(str)
    normalized["subject"] = normalized["subject"].astype(str).str.zfill(3)
    normalized["decision_time_seconds"] = pd.to_numeric(
        normalized["decision_time_seconds"],
        errors="raise",
    )
    normalized["label"] = pd.to_numeric(
        normalized["label"],
        errors="raise",
    ).astype(np.int64)
    normalized["probability"] = pd.to_numeric(
        normalized["probability"],
        errors="raise",
    )
    if not normalized["label"].isin([0, 1]).all():
        raise ValueError("Decision labels must be binary.")
    probabilities = normalized["probability"].to_numpy(dtype=np.float64)
    if not np.isfinite(probabilities).all() or np.any(
        (probabilities < 0.0) | (probabilities > 1.0)
    ):
        raise ValueError("Decision probabilities must be finite values in [0, 1].")
    if normalized.duplicated(["recording_id", "decision_time_seconds"]).any():
        raise ValueError("Decision times must be unique within each recording.")
    return normalized.sort_values(
        ["subject", "recording_id", "decision_time_seconds"]
    ).reset_index(drop=True)


def _merge_intervals(
    intervals: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Merge overlapping warning intervals."""
    if not intervals:
        return []
    merged: list[list[float]] = []
    for start, stop in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, stop])
        else:
            merged[-1][1] = max(merged[-1][1], stop)
    return [(start, stop) for start, stop in merged]


def _count_times_in_intervals(
    times: np.ndarray,
    intervals: list[tuple[float, float]],
) -> int:
    """Count decision instants covered by the union of warning intervals."""
    if len(times) == 0 or not intervals:
        return 0
    covered = np.zeros(len(times), dtype=bool)
    for start, stop in _merge_intervals(intervals):
        covered |= (times >= start) & (times < stop)
    return int(covered.sum())


def evaluate_alarm_threshold(
    decisions: pd.DataFrame,
    seizures: pd.DataFrame,
    *,
    threshold: float,
    prediction_horizon_seconds: float,
    occurrence_period_seconds: float,
    refractory_seconds: float,
    decision_stride_seconds: float,
) -> EventEvaluation:
    """Simulate alarm episodes and calculate event-level performance.

    Every above-threshold decision creates a prediction interval. Alerts less
    than ``refractory_seconds`` apart are reported as one episode, so adjacent
    minute-level warnings cannot inflate the false-alarm count. A seizure is
    detected only when its exact onset satisfies the same future-window rule
    used to create the binary labels.
    """
    if not 0.0 <= threshold <= 1.0 + np.finfo(np.float64).eps:
        raise ValueError("threshold must be between zero and one.")
    if prediction_horizon_seconds < 0.0:
        raise ValueError("prediction_horizon_seconds cannot be negative.")
    if occurrence_period_seconds <= 0.0:
        raise ValueError("occurrence_period_seconds must be positive.")
    if refractory_seconds < 0.0:
        raise ValueError("refractory_seconds cannot be negative.")
    if decision_stride_seconds <= 0.0:
        raise ValueError("decision_stride_seconds must be positive.")

    normalized_decisions = _validate_decisions(decisions)
    required_seizure_columns = {
        "seizure_id",
        "subject",
        "recording_id",
        "onset_seconds",
    }
    missing_columns = required_seizure_columns - set(seizures.columns)
    if missing_columns:
        raise ValueError(
            "Target seizure table is missing columns: "
            f"{sorted(missing_columns)}"
        )
    normalized_seizures = seizures.copy()
    normalized_seizures["seizure_id"] = normalized_seizures[
        "seizure_id"
    ].astype(str)
    normalized_seizures["subject"] = normalized_seizures["subject"].astype(
        str
    ).str.zfill(3)
    normalized_seizures["recording_id"] = normalized_seizures[
        "recording_id"
    ].astype(str)
    normalized_seizures["onset_seconds"] = pd.to_numeric(
        normalized_seizures["onset_seconds"],
        errors="raise",
    )

    seizure_groups = {
        recording_id: group.sort_values("onset_seconds")
        for recording_id, group in normalized_seizures.groupby(
            "recording_id",
            sort=False,
        )
    }
    episode_rows: list[dict[str, object]] = []
    event_detection: dict[str, list[tuple[float, int]]] = {
        seizure_id: []
        for seizure_id in normalized_seizures["seizure_id"].astype(str)
    }
    warning_decisions_by_subject: dict[str, int] = {}
    episode_index = 0

    for recording_id, recording_decisions in normalized_decisions.groupby(
        "recording_id",
        sort=False,
    ):
        subject = str(recording_decisions["subject"].iloc[0])
        alerts = recording_decisions[
            recording_decisions["probability"] >= threshold
        ]
        alert_times = alerts["decision_time_seconds"].to_numpy(
            dtype=np.float64
        )
        warning_intervals = [
            (
                float(alert_time + prediction_horizon_seconds),
                float(
                    alert_time
                    + prediction_horizon_seconds
                    + occurrence_period_seconds
                ),
            )
            for alert_time in alert_times
        ]
        recording_times = recording_decisions[
            "decision_time_seconds"
        ].to_numpy(dtype=np.float64)
        warning_decisions_by_subject[subject] = (
            warning_decisions_by_subject.get(subject, 0)
            + _count_times_in_intervals(recording_times, warning_intervals)
        )
        if len(alert_times) == 0:
            continue

        recording_seizures = seizure_groups.get(
            str(recording_id),
            normalized_seizures.iloc[0:0],
        )
        episode_starts = [0]
        for alert_position in range(1, len(alert_times)):
            if (
                alert_times[alert_position] - alert_times[alert_position - 1]
                > refractory_seconds
            ):
                episode_starts.append(alert_position)
        episode_starts.append(len(alert_times))

        for start_index, stop_index in zip(
            episode_starts[:-1],
            episode_starts[1:],
        ):
            episode_index += 1
            episode_alert_times = alert_times[start_index:stop_index]
            detected_ids: list[str] = []
            detecting_alert_by_seizure: dict[str, float] = {}
            for seizure in recording_seizures.itertuples(index=False):
                onset = float(seizure.onset_seconds)
                qualifying = episode_alert_times[
                    (episode_alert_times < onset)
                    & (
                        episode_alert_times + prediction_horizon_seconds
                        <= onset
                    )
                    & (
                        onset
                        <= episode_alert_times
                        + prediction_horizon_seconds
                        + occurrence_period_seconds
                    )
                ]
                if len(qualifying) > 0:
                    seizure_id = str(seizure.seizure_id)
                    detecting_time = float(qualifying.min())
                    detected_ids.append(seizure_id)
                    detecting_alert_by_seizure[seizure_id] = detecting_time
                    event_detection[seizure_id].append(
                        (detecting_time, episode_index)
                    )

            episode_rows.append(
                {
                    "episode_id": episode_index,
                    "subject": subject,
                    "recording_id": str(recording_id),
                    "first_alert_seconds": float(episode_alert_times[0]),
                    "last_alert_seconds": float(episode_alert_times[-1]),
                    "number_of_raw_alerts": int(len(episode_alert_times)),
                    "is_true_alarm": bool(detected_ids),
                    "detected_seizure_count": int(len(detected_ids)),
                    "detected_seizure_ids": ";".join(detected_ids),
                }
            )

    alarm_episodes = pd.DataFrame(
        episode_rows,
        columns=[
            "episode_id",
            "subject",
            "recording_id",
            "first_alert_seconds",
            "last_alert_seconds",
            "number_of_raw_alerts",
            "is_true_alarm",
            "detected_seizure_count",
            "detected_seizure_ids",
        ],
    )
    seizure_rows: list[dict[str, object]] = []
    for seizure in normalized_seizures.itertuples(index=False):
        detections = event_detection[str(seizure.seizure_id)]
        earliest_detection = min(detections, default=None)
        seizure_rows.append(
            {
                "seizure_id": str(seizure.seizure_id),
                "subject": str(seizure.subject),
                "recording_id": str(seizure.recording_id),
                "onset_seconds": float(seizure.onset_seconds),
                "detected": bool(detections),
                "detecting_episode_id": (
                    int(earliest_detection[1])
                    if earliest_detection is not None
                    else pd.NA
                ),
                "detecting_alert_seconds": (
                    float(earliest_detection[0])
                    if earliest_detection is not None
                    else np.nan
                ),
                "warning_lead_minutes": (
                    float(
                        (float(seizure.onset_seconds) - earliest_detection[0])
                        / 60.0
                    )
                    if earliest_detection is not None
                    else np.nan
                ),
            }
        )
    seizure_events = pd.DataFrame(seizure_rows)

    subject_rows: list[dict[str, float | int | str]] = []
    all_subjects = sorted(
        set(normalized_decisions["subject"].astype(str))
        | set(normalized_seizures["subject"].astype(str))
    )
    for subject in all_subjects:
        subject_decisions = normalized_decisions[
            normalized_decisions["subject"] == subject
        ]
        subject_events = seizure_events[seizure_events["subject"] == subject]
        if alarm_episodes.empty:
            subject_episodes = alarm_episodes
        else:
            subject_episodes = alarm_episodes[
                alarm_episodes["subject"] == subject
            ]
        total_events = len(subject_events)
        detected_events = int(subject_events["detected"].sum())
        false_alarms = int(
            (~subject_episodes["is_true_alarm"].astype(bool)).sum()
        )
        interictal_decisions = int((subject_decisions["label"] == 0).sum())
        interictal_hours = (
            interictal_decisions * decision_stride_seconds / 3600.0
        )
        total_valid_decisions = len(subject_decisions)
        warning_decisions = warning_decisions_by_subject.get(subject, 0)
        subject_rows.append(
            {
                "subject": subject,
                "total_seizures": total_events,
                "detected_seizures": detected_events,
                "event_sensitivity": (
                    detected_events / total_events
                    if total_events > 0
                    else np.nan
                ),
                "false_alarm_episodes": false_alarms,
                "interictal_hours": interictal_hours,
                "false_alarms_per_24h": (
                    false_alarms * 24.0 / interictal_hours
                    if interictal_hours > 0.0
                    else np.nan
                ),
                "valid_decisions": total_valid_decisions,
                "warning_decisions": warning_decisions,
                "time_in_warning_fraction": (
                    warning_decisions / total_valid_decisions
                    if total_valid_decisions > 0
                    else np.nan
                ),
            }
        )
    per_subject = pd.DataFrame(subject_rows)

    total_seizures = len(seizure_events)
    detected_seizures = int(seizure_events["detected"].sum())
    false_alarm_episodes = int(
        (~alarm_episodes["is_true_alarm"].astype(bool)).sum()
    ) if not alarm_episodes.empty else 0
    true_alarm_episodes = int(
        alarm_episodes["is_true_alarm"].astype(bool).sum()
    ) if not alarm_episodes.empty else 0
    interictal_hours = float(per_subject["interictal_hours"].sum())
    valid_decisions = int(per_subject["valid_decisions"].sum())
    warning_decisions = int(per_subject["warning_decisions"].sum())
    event_subject_sensitivities = per_subject.loc[
        per_subject["total_seizures"] > 0,
        "event_sensitivity",
    ]
    metrics: dict[str, float | int] = {
        "threshold": float(threshold),
        "total_seizures": total_seizures,
        "detected_seizures": detected_seizures,
        "event_sensitivity": (
            detected_seizures / total_seizures if total_seizures > 0 else np.nan
        ),
        "macro_patient_sensitivity": float(event_subject_sensitivities.mean()),
        "total_alarm_episodes": int(len(alarm_episodes)),
        "true_alarm_episodes": true_alarm_episodes,
        "false_alarm_episodes": false_alarm_episodes,
        "alarm_episode_precision": (
            true_alarm_episodes / len(alarm_episodes)
            if len(alarm_episodes) > 0
            else np.nan
        ),
        "interictal_hours": interictal_hours,
        "false_alarms_per_24h": (
            false_alarm_episodes * 24.0 / interictal_hours
            if interictal_hours > 0.0
            else np.nan
        ),
        "valid_hours": (
            valid_decisions * decision_stride_seconds / 3600.0
        ),
        "warning_hours": (
            warning_decisions * decision_stride_seconds / 3600.0
        ),
        "time_in_warning_fraction": (
            warning_decisions / valid_decisions
            if valid_decisions > 0
            else np.nan
        ),
    }
    return EventEvaluation(
        metrics=metrics,
        alarm_episodes=alarm_episodes,
        seizure_events=seizure_events,
        per_subject=per_subject,
    )


def bootstrap_patient_metrics(
    per_subject: pd.DataFrame,
    *,
    samples: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    """Return patient-resampled 95% intervals for an operating point."""
    if samples <= 0:
        raise ValueError("samples must be positive.")
    if per_subject.empty:
        raise ValueError("per_subject cannot be empty.")
    rng = np.random.default_rng(seed)
    rows = per_subject.reset_index(drop=True)
    bootstrap_values = {
        "event_sensitivity": [],
        "macro_patient_sensitivity": [],
        "false_alarms_per_24h": [],
        "time_in_warning_fraction": [],
    }
    for _ in range(samples):
        sampled = rows.iloc[rng.integers(0, len(rows), size=len(rows))]
        total_seizures = float(sampled["total_seizures"].sum())
        detected_seizures = float(sampled["detected_seizures"].sum())
        interictal_hours = float(sampled["interictal_hours"].sum())
        false_alarms = float(sampled["false_alarm_episodes"].sum())
        valid_decisions = float(sampled["valid_decisions"].sum())
        warning_decisions = float(sampled["warning_decisions"].sum())
        patient_sensitivity = sampled.loc[
            sampled["total_seizures"] > 0,
            "event_sensitivity",
        ]
        bootstrap_values["event_sensitivity"].append(
            detected_seizures / total_seizures
        )
        bootstrap_values["macro_patient_sensitivity"].append(
            float(patient_sensitivity.mean())
        )
        bootstrap_values["false_alarms_per_24h"].append(
            false_alarms * 24.0 / interictal_hours
        )
        bootstrap_values["time_in_warning_fraction"].append(
            warning_decisions / valid_decisions
        )

    intervals: dict[str, dict[str, float]] = {}
    for metric_name, values in bootstrap_values.items():
        metric_values = np.asarray(values, dtype=np.float64)
        intervals[metric_name] = {
            "lower_95": float(np.quantile(metric_values, 0.025)),
            "median": float(np.quantile(metric_values, 0.5)),
            "upper_95": float(np.quantile(metric_values, 0.975)),
        }
    return intervals
