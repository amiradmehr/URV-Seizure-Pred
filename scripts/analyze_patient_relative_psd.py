#!/usr/bin/env python3
"""Compare preictal PSD with each patient's own interictal baseline.

The analysis uses filtered, unstandardized continuous EEG. Eligible target
seizures come from positive decisions in the train/validation manifests. The
held-out test split is intentionally unavailable from this command.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from seizure_prediction.config import CONFIG
from seizure_prediction.patient_relative_psd import (
    BANDS_HZ,
    compute_band_power_density,
    decibels_relative_to_baseline,
    interval_overlaps_any,
    preictal_bin_label,
    sample_evenly_across_recordings,
)


CHANNEL_COLORS = {
    "delta": "#4C78A8",
    "theta": "#59A14F",
    "alpha": "#F28E2B",
    "beta": "#E15759",
    "gamma": "#B07AA1",
    "broadband": "#222222",
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "validation"),
        default=("train", "validation"),
        help="Patient splits to analyze. The held-out test split is not allowed.",
    )
    parser.add_argument("--preictal-minutes", type=int, default=60)
    parser.add_argument("--bin-minutes", type=int, default=10)
    parser.add_argument("--minute-seconds", type=int, default=60)
    parser.add_argument("--welch-segment-seconds", type=float, default=2.0)
    parser.add_argument("--postictal-exclusion-minutes", type=float, default=60.0)
    parser.add_argument(
        "--baseline-ratio",
        type=float,
        default=1.0,
        help="Interictal baseline minutes sampled per available preictal minute.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--decision-manifest",
        type=Path,
        default=CONFIG.manifests_dir / "decision_manifest.csv",
    )
    parser.add_argument(
        "--seizure-manifest",
        type=Path,
        default=CONFIG.manifests_dir / "seizure_manifest.csv",
    )
    parser.add_argument(
        "--recordings-dir",
        type=Path,
        default=CONFIG.unscaled_recordings_dir,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "outputs"
            / "analysis"
            / "patient_relative_psd_train_validation"
        ),
    )
    arguments = parser.parse_args()
    if arguments.preictal_minutes <= 0 or arguments.bin_minutes <= 0:
        parser.error("Preictal and bin durations must be positive.")
    if arguments.preictal_minutes % arguments.bin_minutes != 0:
        parser.error("--preictal-minutes must be divisible by --bin-minutes.")
    if arguments.minute_seconds <= 0 or arguments.batch_size <= 0:
        parser.error("Minute duration and batch size must be positive.")
    if arguments.baseline_ratio <= 0.0:
        parser.error("--baseline-ratio must be positive.")
    return arguments


def _normalize_subject(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(3)


def load_manifests(
    decision_path: Path,
    seizure_path: Path,
    splits: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load selected decisions, all seizure exclusions, and eligible targets."""
    decisions = pd.read_csv(
        decision_path,
        low_memory=False,
        dtype={
            "subject": str,
            "recording_id": str,
            "target_seizure_id": str,
            "split": str,
        },
    )
    required_decisions = {
        "subject",
        "recording_id",
        "decision_time_seconds",
        "label",
        "target_seizure_id",
        "split",
    }
    missing = required_decisions - set(decisions.columns)
    if missing:
        raise ValueError(f"Decision manifest is missing: {sorted(missing)}")
    decisions["subject"] = _normalize_subject(decisions["subject"])
    decisions = decisions[decisions["split"].isin(splits)].copy()
    if decisions.empty:
        raise ValueError(f"No decisions found for splits {list(splits)}.")

    seizures = pd.read_csv(
        seizure_path,
        low_memory=False,
        dtype={"subject": str, "seizure_id": str},
    )
    required_seizures = {
        "seizure_id",
        "subject",
        "onset_seconds",
        "duration_seconds",
    }
    missing = required_seizures - set(seizures.columns)
    if missing:
        raise ValueError(f"Seizure manifest is missing: {sorted(missing)}")
    seizures["subject"] = _normalize_subject(seizures["subject"])
    seizures["recording_id"] = seizures["seizure_id"].str.rsplit(
        "_seizure-", n=1
    ).str[0]

    target_ids = set(
        decisions.loc[decisions["label"].eq(1), "target_seizure_id"]
        .dropna()
        .astype(str)
    )
    targets = seizures[seizures["seizure_id"].isin(target_ids)].copy()
    missing_targets = sorted(target_ids - set(targets["seizure_id"]))
    if missing_targets:
        raise ValueError(f"Missing target seizures: {missing_targets[:5]}")
    if targets.empty:
        raise ValueError("The selected splits contain no eligible target seizures.")
    target_subjects = set(targets["subject"])
    seizures = seizures[seizures["subject"].isin(target_subjects)].copy()
    decisions = decisions[decisions["subject"].isin(target_subjects)].copy()
    return decisions, seizures, targets


def build_preictal_segments(
    targets: pd.DataFrame,
    *,
    preictal_minutes: int,
    minute_seconds: int,
    bin_minutes: int,
) -> pd.DataFrame:
    """Build non-overlapping one-minute segments preceding each target onset."""
    rows: list[dict[str, object]] = []
    segment_count = int(round(preictal_minutes * 60.0 / minute_seconds))
    for seizure in targets.itertuples(index=False):
        onset = float(seizure.onset_seconds)
        for offset_index in range(segment_count):
            stop_seconds = onset - offset_index * minute_seconds
            start_seconds = stop_seconds - minute_seconds
            if start_seconds < 0.0:
                continue
            minutes_before = (onset - (start_seconds + stop_seconds) / 2.0) / 60.0
            rows.append(
                {
                    "subject": str(seizure.subject),
                    "recording_id": str(seizure.recording_id),
                    "split": str(getattr(seizure, "split", "unknown")),
                    "phase": "preictal",
                    "seizure_id": str(seizure.seizure_id),
                    "segment_start_seconds": start_seconds,
                    "segment_stop_seconds": stop_seconds,
                    "minutes_before_onset": minutes_before,
                    "preictal_bin": preictal_bin_label(
                        minutes_before,
                        preictal_minutes=preictal_minutes,
                        bin_minutes=bin_minutes,
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_baseline_segments(
    decisions: pd.DataFrame,
    seizures: pd.DataFrame,
    preictal_counts: pd.Series,
    *,
    preictal_minutes: int,
    minute_seconds: int,
    postictal_exclusion_minutes: float,
    baseline_ratio: float,
    seed: int,
) -> pd.DataFrame:
    """Select patient-balanced negative minutes outside seizure-adjacent time."""
    negative = decisions[decisions["label"].eq(0)].copy()
    negative["segment_stop_seconds"] = negative["decision_time_seconds"].astype(float)
    negative["segment_start_seconds"] = (
        negative["segment_stop_seconds"] - minute_seconds
    )
    negative = negative[negative["segment_start_seconds"] >= 0.0]

    exclusions: dict[str, list[tuple[float, float]]] = {}
    for recording_id, group in seizures.groupby("recording_id", sort=False):
        exclusions[str(recording_id)] = [
            (
                float(row.onset_seconds) - preictal_minutes * 60.0,
                float(row.onset_seconds)
                + float(row.duration_seconds)
                + postictal_exclusion_minutes * 60.0,
            )
            for row in group.itertuples(index=False)
        ]

    keep = [
        not interval_overlaps_any(
            float(row.segment_start_seconds),
            float(row.segment_stop_seconds),
            exclusions.get(str(row.recording_id), ()),
        )
        for row in negative.itertuples(index=False)
    ]
    negative = negative.loc[keep]
    selected_groups: list[pd.DataFrame] = []
    for subject, preictal_count in preictal_counts.items():
        candidates = negative[negative["subject"].eq(subject)].copy()
        requested = int(round(float(preictal_count) * baseline_ratio))
        selected = sample_evenly_across_recordings(
            candidates,
            requested,
            seed=seed + int(subject),
        )
        if selected.empty:
            continue
        selected["phase"] = "interictal_baseline"
        selected["seizure_id"] = pd.NA
        selected["minutes_before_onset"] = np.nan
        selected["preictal_bin"] = "baseline"
        selected_groups.append(selected)
    if not selected_groups:
        raise ValueError("No uncontaminated interictal baseline segments were found.")
    columns = [
        "subject",
        "recording_id",
        "split",
        "phase",
        "seizure_id",
        "segment_start_seconds",
        "segment_stop_seconds",
        "minutes_before_onset",
        "preictal_bin",
    ]
    return pd.concat(selected_groups, ignore_index=True)[columns]


def load_recording_metadata(
    recordings_dir: Path,
    recording_id: str,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    array_path = recordings_dir / f"{recording_id}.npy"
    channels_path = recordings_dir / f"{recording_id}_channels.json"
    availability_path = recordings_dir / f"{recording_id}_channel_availability.json"
    for path in (array_path, channels_path, availability_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing recording artifact: {path}")
    values = np.load(array_path, mmap_mode="r")
    channels = json.loads(channels_path.read_text(encoding="utf-8"))
    availability = np.asarray(
        json.loads(availability_path.read_text(encoding="utf-8")), dtype=bool
    )
    if values.ndim != 2 or values.shape[0] != len(channels):
        raise ValueError(f"Unexpected recording shape for {recording_id}: {values.shape}")
    if availability.shape != (values.shape[0],):
        raise ValueError(f"Invalid channel mask for {recording_id}: {availability.shape}")
    return values, [str(channel) for channel in channels], availability


def extract_segment_power(
    segments: pd.DataFrame,
    recordings_dir: Path,
    *,
    sampling_frequency: float,
    welch_segment_seconds: float,
    batch_size: int,
) -> pd.DataFrame:
    """Extract band densities for every segment and physically present channel."""
    output: list[pd.DataFrame] = []
    band_names = [name for name, _, _ in BANDS_HZ]
    for recording_number, (recording_id, group) in enumerate(
        segments.groupby("recording_id", sort=True), start=1
    ):
        values, channel_names, availability = load_recording_metadata(
            recordings_dir, str(recording_id)
        )
        group = group.reset_index(drop=True)
        start_samples = np.rint(
            group["segment_start_seconds"].to_numpy(float) * sampling_frequency
        ).astype(np.int64)
        stop_samples = np.rint(
            group["segment_stop_seconds"].to_numpy(float) * sampling_frequency
        ).astype(np.int64)
        valid = (start_samples >= 0) & (stop_samples <= values.shape[1])
        if not valid.all():
            group = group.loc[valid].reset_index(drop=True)
            start_samples = start_samples[valid]
            stop_samples = stop_samples[valid]
        if group.empty:
            continue
        expected_samples = int(round(
            (group["segment_stop_seconds"].iloc[0]
             - group["segment_start_seconds"].iloc[0])
            * sampling_frequency
        ))
        for batch_start in range(0, len(group), batch_size):
            batch_stop = min(batch_start + batch_size, len(group))
            windows = np.stack(
                [
                    np.asarray(values[:, start:stop], dtype=np.float32)
                    for start, stop in zip(
                        start_samples[batch_start:batch_stop],
                        stop_samples[batch_start:batch_stop],
                    )
                ],
                axis=0,
            )
            if windows.shape[-1] != expected_samples:
                raise ValueError(f"Inconsistent segment length in {recording_id}.")
            densities = compute_band_power_density(
                windows,
                availability,
                sampling_frequency=sampling_frequency,
                welch_segment_seconds=welch_segment_seconds,
            )
            metadata = group.iloc[batch_start:batch_stop].reset_index(drop=True)
            for channel_index, channel_name in enumerate(channel_names):
                if not availability[channel_index]:
                    continue
                channel_frame = metadata.loc[
                    metadata.index.repeat(len(band_names))
                ].reset_index(drop=True)
                channel_frame["channel"] = channel_name
                channel_frame["band"] = np.tile(band_names, len(metadata))
                channel_frame["power_density"] = densities[
                    :, channel_index, :
                ].reshape(-1)
                output.append(channel_frame)
        if recording_number % 100 == 0:
            print(f"Processed {recording_number} recordings...", flush=True)
    if not output:
        raise ValueError("No valid PSD segments were extracted.")
    return pd.concat(output, ignore_index=True)


def add_patient_baselines(power: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["subject", "channel", "band"]
    baseline = (
        power[power["phase"].eq("interictal_baseline")]
        .groupby(keys, as_index=False)["power_density"]
        .median()
        .rename(columns={"power_density": "baseline_median_power_density"})
    )
    if baseline.empty:
        raise ValueError("Baseline PSD table is empty.")
    merged = power.merge(baseline, on=keys, how="left", validate="many_to_one")
    if merged.loc[merged["phase"].eq("preictal"), "baseline_median_power_density"].isna().any():
        raise ValueError("Some preictal channels lack a patient baseline.")
    merged["db_relative_to_patient_baseline"] = decibels_relative_to_baseline(
        merged["power_density"].to_numpy(float),
        merged["baseline_median_power_density"].to_numpy(float),
    )
    return merged, baseline


def summarize_preictal(power: pd.DataFrame) -> pd.DataFrame:
    preictal = power[power["phase"].eq("preictal")].copy()
    keys = ["subject", "split", "channel", "band", "preictal_bin"]
    grouped = preictal.groupby(keys, observed=True)["db_relative_to_patient_baseline"]
    summary = grouped.agg(
        segments="size",
        median_db="median",
        mean_db="mean",
        q25_db=lambda values: values.quantile(0.25),
        q75_db=lambda values: values.quantile(0.75),
    ).reset_index()
    return summary


def plot_patient(
    subject: str,
    subject_summary: pd.DataFrame,
    output_path: Path,
    *,
    preictal_minutes: int,
    bin_minutes: int,
) -> None:
    channels = list(CONFIG.canonical_channel_names)
    centers = np.arange(
        preictal_minutes - bin_minutes / 2.0,
        0.0,
        -bin_minutes,
    )
    labels = [
        preictal_bin_label(
            center,
            preictal_minutes=preictal_minutes,
            bin_minutes=bin_minutes,
        )
        for center in centers
    ]
    figure, axes = plt.subplots(1, len(channels), figsize=(16, 4.8), sharey=True)
    for axis, channel in zip(axes, channels):
        channel_rows = subject_summary[subject_summary["channel"].eq(channel)]
        if channel_rows.empty:
            axis.text(0.5, 0.5, "Channel unavailable", ha="center", va="center")
        else:
            for band, _, _ in BANDS_HZ:
                rows = channel_rows[channel_rows["band"].eq(band)].set_index(
                    "preictal_bin"
                )
                y = rows.reindex(labels)["median_db"].to_numpy(float)
                axis.plot(
                    centers,
                    y,
                    marker="o",
                    linewidth=1.8 if band == "broadband" else 1.2,
                    markersize=4,
                    label=band,
                    color=CHANNEL_COLORS[band],
                )
        axis.axhline(0.0, color="#777777", linewidth=1.0, linestyle="--")
        axis.set_title(channel.replace("_", " ").title())
        axis.set_xlabel("Minutes before seizure onset")
        axis.set_xlim(preictal_minutes, 0.0)
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Power density relative to patient baseline (dB)")
    handles, legend_labels = axes[-1].get_legend_handles_labels()
    if handles:
        figure.legend(
            handles,
            legend_labels,
            loc="lower center",
            ncol=len(BANDS_HZ),
            frameon=False,
        )
    figure.suptitle(f"Patient {subject}: preictal spectral-density change")
    figure.tight_layout(rect=(0.0, 0.09, 1.0, 0.93))
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_cohort_broadband(summary: pd.DataFrame, output_path: Path) -> None:
    broadband = summary[summary["band"].eq("broadband")].copy()
    patient_bin = (
        broadband.groupby(["subject", "preictal_bin"], observed=True)["median_db"]
        .median()
        .reset_index()
    )
    bin_order = sorted(
        patient_bin["preictal_bin"].unique(),
        key=lambda value: int(str(value).split("-")[0]),
        reverse=True,
    )
    matrix = patient_bin.pivot(index="subject", columns="preictal_bin", values="median_db")
    matrix = matrix.reindex(columns=bin_order)
    x = np.arange(len(bin_order))
    figure, axis = plt.subplots(figsize=(10.5, 6.0))
    for _, row in matrix.iterrows():
        axis.plot(x, row.to_numpy(float), color="#7F7F7F", alpha=0.22, linewidth=0.8)
    axis.plot(
        x,
        matrix.median(axis=0).to_numpy(float),
        color="#C23B22",
        marker="o",
        linewidth=3.0,
        label="Across-patient median",
    )
    axis.axhline(0.0, color="#222222", linestyle="--", linewidth=1.0)
    axis.set_xticks(x, bin_order)
    axis.set_xlabel("Minutes before seizure onset")
    axis.set_ylabel("Broadband power density relative to patient baseline (dB)")
    axis.set_title("Patient-specific preictal broadband PSD changes")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    arguments = parse_arguments()
    splits = tuple(dict.fromkeys(arguments.splits))
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    patient_plot_dir = arguments.output_dir / "patients"
    patient_plot_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading manifests for {list(splits)}...", flush=True)
    decisions, seizures, targets = load_manifests(
        arguments.decision_manifest,
        arguments.seizure_manifest,
        splits,
    )
    split_lookup = decisions[["subject", "split"]].drop_duplicates()
    if split_lookup["subject"].duplicated().any():
        raise ValueError("A patient appears in multiple selected splits.")
    targets = targets.merge(split_lookup, on="subject", how="left", validate="many_to_one")

    preictal = build_preictal_segments(
        targets,
        preictal_minutes=arguments.preictal_minutes,
        minute_seconds=arguments.minute_seconds,
        bin_minutes=arguments.bin_minutes,
    )
    preictal_counts = preictal.groupby("subject").size()
    baseline = build_baseline_segments(
        decisions,
        seizures,
        preictal_counts,
        preictal_minutes=arguments.preictal_minutes,
        minute_seconds=arguments.minute_seconds,
        postictal_exclusion_minutes=arguments.postictal_exclusion_minutes,
        baseline_ratio=arguments.baseline_ratio,
        seed=arguments.seed,
    )
    segments = pd.concat([preictal, baseline], ignore_index=True)
    segments.to_csv(arguments.output_dir / "selected_segments.csv", index=False)
    print(
        f"Extracting PSD for {len(preictal):,} preictal and "
        f"{len(baseline):,} interictal baseline minutes...",
        flush=True,
    )
    power = extract_segment_power(
        segments,
        arguments.recordings_dir,
        sampling_frequency=CONFIG.target_sfreq,
        welch_segment_seconds=arguments.welch_segment_seconds,
        batch_size=arguments.batch_size,
    )
    power, patient_baselines = add_patient_baselines(power)
    summary = summarize_preictal(power)

    power.to_csv(
        arguments.output_dir / "minute_band_power.csv.gz",
        index=False,
        compression="gzip",
    )
    patient_baselines.to_csv(
        arguments.output_dir / "patient_interictal_baselines.csv", index=False
    )
    summary.to_csv(arguments.output_dir / "patient_preictal_psd_summary.csv", index=False)

    for subject, subject_rows in summary.groupby("subject", sort=True):
        plot_patient(
            str(subject),
            subject_rows,
            patient_plot_dir / f"patient_{subject}_relative_psd.png",
            preictal_minutes=arguments.preictal_minutes,
            bin_minutes=arguments.bin_minutes,
        )
    plot_cohort_broadband(
        summary,
        arguments.output_dir / "cohort_broadband_relative_psd.png",
    )

    last_bin = f"{arguments.bin_minutes}-0"
    last_bin_summary = summary[summary["preictal_bin"].eq(last_bin)].copy()
    last_bin_summary.to_csv(
        arguments.output_dir / "patient_last_bin_psd_change.csv", index=False
    )
    analysis_summary = {
        "splits": list(splits),
        "held_out_test_used": False,
        "sampling_frequency_hz": CONFIG.target_sfreq,
        "preictal_minutes": arguments.preictal_minutes,
        "bin_minutes": arguments.bin_minutes,
        "minute_seconds": arguments.minute_seconds,
        "welch_segment_seconds": arguments.welch_segment_seconds,
        "postictal_exclusion_minutes": arguments.postictal_exclusion_minutes,
        "baseline_ratio": arguments.baseline_ratio,
        "baseline_statistic": "within-patient/channel/band median",
        "normalization": "10*log10(preictal PSD / patient interictal median PSD)",
        "target_patients": int(targets["subject"].nunique()),
        "target_seizures": int(targets["seizure_id"].nunique()),
        "preictal_segments": int(len(preictal)),
        "baseline_segments": int(len(baseline)),
        "patients_with_output": int(summary["subject"].nunique()),
        "bands_hz": [
            {"name": name, "low_hz": low, "high_hz": high}
            for name, low, high in BANDS_HZ
        ],
    }
    (arguments.output_dir / "analysis_summary.json").write_text(
        json.dumps(analysis_summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(analysis_summary, indent=2), flush=True)
    print(f"Saved analysis to {arguments.output_dir}", flush=True)


if __name__ == "__main__":
    main()

