r"""Render an exploratory picture of the processed SeizeIT2 dataset.

Produces one PNG per topic in ``outputs/dataset_figures/``:

    01_cohort.png        EDF files, EEG duration and decisions per patient
    02_labels.png        class balance, retention, decisions per seizure
    03_seizures.png      seizure counts, durations and why most are ineligible
    04_signals.png       real 45-minute histories, positive versus negative
    05_chunks.png        the 5-second chunk the encoder actually sees
    06_amplitude.png     per-patient amplitude spread after global z-scoring
    07_spectra.png       Welch PSD and band power, pre-ictal versus interictal

Units used throughout, stated on every axis:

    count      a number of things (EDF files, decisions, seizures, patients)
    h / min / s   wall-clock duration of EEG
    Hz         frequency
    z          dimensionless amplitude in units of the global per-channel
               standard deviation. The stored shards are z-scored, so nothing
               downstream is in volts; the conversion back to volts is printed
               on the amplitude figure and is channel-specific.

Everything is read from the manifests plus a sample of the standardized
shards; nothing is recomputed from the raw EDFs.

    python scripts/visualize_dataset.py --recording-sample 60
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import welch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIRECTORY = PROJECT_ROOT / "src"
if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

from seizure_prediction.config import CONFIG  # noqa: E402

# Validated categorical slots 1-3, plus a muted ink for reference marks.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK = "#52514e"
SPLIT_COLOR = {"train": BLUE, "validation": ORANGE, "test": AQUA}

# Amplitudes are dimensionless multiples of the global per-channel sigma.
Z_UNIT = "z  (1 z = 1 global channel σ)"

BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 40.0),
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "dataset_figures",
    )
    parser.add_argument(
        "--recording-sample",
        type=int,
        default=60,
        help="Recordings to open for amplitude/spectral statistics.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def log(message: str) -> None:
    print(message, flush=True)


def style(axis: plt.Axes) -> None:
    axis.grid(alpha=0.22, linewidth=0.7)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(INK)
        axis.spines[side].set_linewidth(0.8)


def save(figure: plt.Figure, path: Path, title: str) -> None:
    figure.suptitle(title, fontsize=14, y=0.995)
    figure.tight_layout()
    figure.savefig(path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    log(f"  wrote {path.name}")


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------


def load_manifests() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the decision, seizure and shard manifests."""
    manifests = CONFIG.manifests_dir
    decisions = pd.read_csv(
        manifests / "decision_manifest.csv",
        dtype={"subject": str, "session": str, "task": str, "run": str},
    )
    seizures = pd.read_csv(manifests / "seizure_manifest.csv", dtype={"subject": str})
    shards = pd.read_csv(
        manifests / "processed_shard_manifest.csv", dtype={"subject": str}
    )
    return decisions, seizures, shards


def load_global_sigma() -> dict[str, float]:
    """Return the fitted global sigma per channel, in volts."""
    path = CONFIG.scaler_parameters_dir / "global_channel_zscore.json"
    if not path.exists():
        return {}
    scaler = json.loads(path.read_text(encoding="utf-8"))
    return dict(zip(scaler["channel_names"], scaler["std"], strict=True))


def shard_durations_hours(shards: pd.DataFrame) -> pd.Series:
    """EEG duration of every shard in hours, from the .npy header only."""
    hours = {}
    for row in shards.itertuples(index=False):
        array = np.load(row.X_path, mmap_mode="r")
        hours[row.recording_id] = array.shape[1] / CONFIG.target_sfreq / 3600.0
    return pd.Series(hours, name="hours")


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------


def figure_cohort(
    decisions: pd.DataFrame,
    shards: pd.DataFrame,
    durations: pd.Series,
    path: Path,
) -> None:
    """Who is in the dataset, how much EEG each patient contributes."""
    figure, axes = plt.subplots(2, 3, figsize=(18, 9))

    shards = shards.assign(hours=shards["recording_id"].map(durations))

    # 1. EDF files per patient
    axis = axes[0, 0]
    per_patient = shards.groupby(["split", "subject"]).size().rename("files").reset_index()
    for split, group in per_patient.groupby("split"):
        axis.scatter(group["subject"].astype(int), group["files"], s=18,
                     color=SPLIT_COLOR[split], label=split, alpha=0.85)
    axis.set(title="EDF files per patient",
             xlabel="Subject ID", ylabel="EDF files (count)", yscale="log")
    axis.legend(frameon=False, fontsize=9, title="split", title_fontsize=9)
    style(axis)

    # 2. EEG duration per patient
    axis = axes[0, 1]
    hours_pp = shards.groupby(["split", "subject"])["hours"].sum().reset_index()
    for split, group in hours_pp.groupby("split"):
        axis.scatter(group["subject"].astype(int), group["hours"], s=18,
                     color=SPLIT_COLOR[split], label=split, alpha=0.85)
    total = hours_pp["hours"].sum()
    axis.set(title=f"EEG duration per patient  (total {total:,.0f} h)",
             xlabel="Subject ID", ylabel="EEG duration (h)", yscale="log")
    axis.legend(frameon=False, fontsize=9, title="split", title_fontsize=9)
    style(axis)

    # 3. Decisions per patient
    axis = axes[0, 2]
    dec_pp = decisions.groupby(["split", "subject"]).size().rename("n").reset_index()
    for split, group in dec_pp.groupby("split"):
        axis.scatter(group["subject"].astype(int), group["n"], s=18,
                     color=SPLIT_COLOR[split], label=split, alpha=0.85)
    axis.set(title="Decisions per patient  (1 decision per 60 s of usable EEG)",
             xlabel="Subject ID", ylabel="Decisions (count)", yscale="log")
    axis.legend(frameon=False, fontsize=9, title="split", title_fontsize=9)
    style(axis)

    # 4. Split totals
    axis = axes[1, 0]
    order = ["train", "validation", "test"]
    totals = decisions.groupby("split").agg(
        n=("label", "size"), pos=("label", "sum"), pat=("subject", "nunique")
    ).reindex(order)
    positions = np.arange(len(order))
    axis.bar(positions, totals["n"], color=[SPLIT_COLOR[s] for s in order], width=0.6)
    for x, (n, p, pat) in enumerate(
        zip(totals["n"], totals["pos"], totals["pat"], strict=True)
    ):
        axis.annotate(f"{n:,}\n{p:,} pos\n{pat} patients", xy=(x, n), xytext=(0, 5),
                      textcoords="offset points", ha="center", fontsize=9, color=INK)
    axis.set(title="Decisions by split", xticks=positions, xticklabels=order,
             xlabel="Split", ylabel="Decisions (count)",
             ylim=(0, totals["n"].max() * 1.3))
    style(axis)

    # 5. Recording duration distribution
    axis = axes[1, 1]
    axis.hist(durations.values, bins=60, color=BLUE)
    minimum_hours = (
        CONFIG.input_window_seconds
        + 60.0 * (CONFIG.prediction_horizon_minutes
                  + CONFIG.seizure_occurrence_period_minutes)
    ) / 3600.0
    axis.axvline(minimum_hours, color=ORANGE, linestyle="--", linewidth=1.5)
    axis.annotate(f"{minimum_hours*60:.0f} min minimum\nfor one decision",
                  xy=(minimum_hours, 0.72), xycoords=("data", "axes fraction"),
                  xytext=(6, 0), textcoords="offset points",
                  fontsize=9, color=ORANGE)
    axis.set(title=f"Recording duration  (median {np.median(durations):.1f} h)",
             xlabel="EEG duration per EDF file (h)",
             ylabel="EDF files (count)", yscale="log")
    style(axis)

    # 6. Electrode pairs
    axis = axes[1, 2]
    combos: dict[str, int] = {}
    for row in shards.itertuples(index=False):
        with open(row.channel_availability_path, encoding="utf-8") as handle:
            mask = tuple(json.load(handle))
        names = [n for n, a in zip(CONFIG.canonical_channel_names, mask, strict=True) if a]
        key = " + ".join(names)
        combos[key] = combos.get(key, 0) + 1
    labels = sorted(combos, key=lambda k: -combos[k])
    values = [combos[k] for k in labels]
    axis.barh(np.arange(len(labels)), values, color=BLUE, height=0.6)
    for i, v in enumerate(values):
        axis.annotate(f"{v:,}", xy=(v, i), xytext=(4, 0), textcoords="offset points",
                      va="center", fontsize=9, color=INK)
    axis.set(title="Electrode pair present  (2 of 3 per file)",
             yticks=np.arange(len(labels)), yticklabels=labels,
             xlabel="EDF files (count)", xlim=(0, max(values) * 1.2))
    axis.invert_yaxis()
    style(axis)

    save(figure, path, "Cohort — 125 configured patients, whole-patient holdout")


def figure_labels(decisions: pd.DataFrame, path: Path) -> None:
    """Class balance and how positives are distributed."""
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    axis = axes[0]
    order = ["train", "validation", "test"]
    neg = [int((decisions[decisions.split == s].label == 0).sum()) for s in order]
    pos = [int((decisions[decisions.split == s].label == 1).sum()) for s in order]
    positions = np.arange(len(order))
    axis.bar(positions - 0.19, neg, width=0.36, color=BLUE, label="negative")
    axis.bar(positions + 0.19, pos, width=0.36, color=ORANGE, label="positive")
    for x, (n, p) in enumerate(zip(neg, pos, strict=True)):
        axis.annotate(f"{n:,}", (x - 0.19, n), xytext=(0, 4),
                      textcoords="offset points", ha="center", fontsize=8, color=INK)
        axis.annotate(f"{p:,}\n{100*p/(n+p):.2f}%", (x + 0.19, p), xytext=(0, 4),
                      textcoords="offset points", ha="center", fontsize=8, color=INK)
    axis.set(title="Class balance", yscale="log", xticks=positions, xticklabels=order,
             xlabel="Split", ylabel="Decisions (count, log scale)")
    axis.legend(frameon=False, fontsize=9, title="label", title_fontsize=9)
    style(axis)

    axis = axes[1]
    per_seizure = decisions[decisions.label == 1].groupby("target_seizure_id").size()
    axis.hist(per_seizure, bins=np.arange(0.5, per_seizure.max() + 1.5), color=ORANGE)
    expected = CONFIG.seizure_occurrence_period_minutes * 60 / CONFIG.input_stride_seconds
    axis.axvline(expected, color=INK, linestyle="--", linewidth=1.2)
    axis.annotate(f"SOP / stride\n= 600 s / 60 s = {expected:.0f}",
                  xy=(0.5, 0.82), xycoords="axes fraction", fontsize=9, color=INK)
    axis.set(title=f"Positive decisions per target seizure  (n = {len(per_seizure)})",
             xlabel="Positive decisions for one seizure (count)",
             ylabel="Seizures (count)")
    style(axis)

    axis = axes[2]
    retained = decisions.groupby("recording_id").size()
    axis.hist(retained, bins=60, color=BLUE)
    axis.set(title=f"Decisions kept per EDF file  (n = {len(retained)} files)",
             xlabel="Decisions retained (count)",
             ylabel="EDF files (count, log scale)", yscale="log")
    style(axis)

    save(figure, path,
         "Labels — 3,113 of 494,283 decisions are positive (0.63 %)")


def figure_seizures(seizures: pd.DataFrame, decisions: pd.DataFrame, path: Path) -> None:
    """Seizure inventory and the eligibility funnel."""
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    eligible = seizures["eligible_for_prediction"].astype(str).str.lower().eq("true")

    axis = axes[0]
    targeted = decisions.loc[decisions.label == 1, "target_seizure_id"].nunique()
    stages = ["annotated", "60 min clear\nbefore onset", "used as\ntargets"]
    values = [len(seizures), int(eligible.sum()), int(targeted)]
    axis.bar(np.arange(3), values, color=[BLUE, BLUE, ORANGE], width=0.6)
    for i, v in enumerate(values):
        axis.annotate(f"{v}\n({100*v/values[0]:.0f} %)", (i, v), xytext=(0, 4),
                      textcoords="offset points", ha="center", fontsize=10,
                      color=INK, fontweight="bold")
    axis.set(title="Seizure eligibility funnel", xticks=np.arange(3),
             xticklabels=stages, xlabel="Stage", ylabel="Seizures (count)",
             ylim=(0, max(values) * 1.28))
    style(axis)

    axis = axes[1]
    dur = seizures["duration_seconds"].dropna()
    axis.hist(dur[~eligible], bins=50, color=BLUE, alpha=0.75, label="ineligible")
    axis.hist(dur[eligible], bins=50, color=ORANGE, alpha=0.8, label="eligible")
    axis.set(title=f"Seizure duration  (median {dur.median():.0f} s)",
             xlabel="Seizure duration (s)",
             ylabel="Seizures (count, log scale)", yscale="log")
    axis.legend(frameon=False, fontsize=9)
    style(axis)

    axis = axes[2]
    per_patient = seizures.groupby("subject").size().sort_values(ascending=False)
    elig_pp = seizures[eligible].groupby("subject").size()
    axis.bar(np.arange(len(per_patient)), per_patient.values, color=BLUE,
             width=1.0, label="annotated")
    axis.bar(np.arange(len(per_patient)),
             [elig_pp.get(s, 0) for s in per_patient.index],
             color=ORANGE, width=1.0, label="eligible")
    axis.set(title=f"Seizures per patient  ({len(per_patient)} patients have any)",
             xlabel="Patients (rank, sorted by seizure count)",
             ylabel="Seizures (count)")
    axis.legend(frameon=False, fontsize=9)
    style(axis)

    save(figure, path,
         "Seizures — 883 annotated, 317 usable as prediction targets (36 %)")


def pick_example(
    decisions: pd.DataFrame,
    shards: pd.DataFrame,
    label: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, pd.Series, np.ndarray] | None:
    """Return (history, decision row, availability) for one random decision."""
    candidates = decisions[(decisions.label == label) & (decisions.split == "train")]
    if candidates.empty:
        return None
    shard_by_id = shards.set_index("recording_id")
    for _ in range(40):
        row = candidates.iloc[int(rng.integers(len(candidates)))]
        if row["recording_id"] not in shard_by_id.index:
            continue
        shard = shard_by_id.loc[row["recording_id"]]
        array = np.load(shard["X_path"], mmap_mode="r")
        start, stop = int(row["history_start_sample"]), int(row["decision_end_sample"])
        if stop > array.shape[1]:
            continue
        with open(shard["channel_availability_path"], encoding="utf-8") as handle:
            mask = np.asarray(json.load(handle), dtype=int)
        return np.asarray(array[:, start:stop], dtype=np.float32), row, mask
    return None


def figure_signals(decisions: pd.DataFrame, shards: pd.DataFrame,
                   path: Path, rng: np.random.Generator) -> None:
    """Two real 45-minute histories, one before a seizure and one not."""
    figure, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    minutes = np.arange(
        int(CONFIG.input_window_seconds * CONFIG.target_sfreq)
    ) / CONFIG.target_sfreq / 60.0

    for axis, label, name, color in (
        (axes[0], 1, "POSITIVE — seizure onset within 10 min after the decision", ORANGE),
        (axes[1], 0, "NEGATIVE — no seizure in the 10 min after the decision", BLUE),
    ):
        picked = pick_example(decisions, shards, label, rng)
        if picked is None:
            continue
        history, row, mask = picked
        for c, (channel, offset) in enumerate(
            zip(CONFIG.canonical_channel_names, [8, 0, -8], strict=True)
        ):
            present = bool(mask[c])
            axis.plot(minutes, history[c] + offset, linewidth=0.25,
                      color=color if present else "#BBBBBB",
                      alpha=0.9 if present else 0.7)
            axis.annotate(
                f"{channel}{'' if present else '  (absent → zero-filled)'}"
                f"     peak |x| = {np.abs(history[c]).max():.1f} z",
                xy=(0.002, offset + 3.4), xycoords=("axes fraction", "data"),
                fontsize=9, color=INK,
            )
        axis.set(
            title=f"{name}   ·   {row['recording_id']}   decision at t = "
                  f"{row['decision_time_seconds']:.0f} s",
            ylabel=f"Amplitude, {Z_UNIT}\nchannels offset for display",
            ylim=(-16, 16), xlim=(0, 45),
        )
        style(axis)

    axes[1].set_xlabel(
        "Time within the 45-minute history (min) — the decision is made at 45 min"
    )
    save(figure, path,
         "One decision = 45 min × 3 channels = 691,200 samples → reshaped to 540 × 3 × 1280")


def figure_chunks(decisions: pd.DataFrame, shards: pd.DataFrame,
                  path: Path, rng: np.random.Generator) -> None:
    """What the encoder is actually handed: single 5-second chunks."""
    picked = pick_example(decisions, shards, 1, rng)
    if picked is None:
        return
    history, row, mask = picked
    chunk_samples = int(CONFIG.chunk_window_seconds * CONFIG.target_sfreq)
    total = history.shape[1] // chunk_samples
    picks = [0, total // 2, total - 1]
    names = ["chunk 0\n45.0 min before the decision",
             f"chunk {total//2}\n22.5 min before the decision",
             f"chunk {total-1}\n0.1 min before the decision"]

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
    seconds = np.arange(chunk_samples) / CONFIG.target_sfreq
    for axis, index, name in zip(axes, picks, names, strict=True):
        chunk = history[:, index * chunk_samples:(index + 1) * chunk_samples]
        for c, offset in enumerate([6, 0, -6]):
            present = bool(mask[c])
            axis.plot(seconds, chunk[c] + offset, linewidth=0.8,
                      color=[BLUE, ORANGE, AQUA][c] if present else "#BBBBBB")
        axis.set(title=name, xlabel="Time within chunk (s)", xlim=(0, 5), ylim=(-12, 12))
        style(axis)
    for c, (channel, offset) in enumerate(
        zip(CONFIG.canonical_channel_names, [6, 0, -6], strict=True)
    ):
        axes[0].annotate(channel, xy=(0.05, offset + 1.6), fontsize=9, color=INK)
    axes[0].set_ylabel(f"Amplitude, {Z_UNIT}\nchannels offset for display")
    save(figure, path,
         f"Encoder input: 3 channels × {chunk_samples} samples (5 s at 256 Hz), "
         f"applied {total}× per decision")


def sample_recording_stats(
    shards: pd.DataFrame,
    decisions: pd.DataFrame,
    n_sample: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Per-recording amplitude stats plus pooled PSDs, from a random sample."""
    sample = shards.sample(n=min(n_sample, len(shards)),
                           random_state=int(rng.integers(1 << 30)))
    rows = []
    psd_pre: list[np.ndarray] = []
    psd_inter: list[np.ndarray] = []
    freqs = None
    positives = decisions[decisions.label == 1]
    negatives = decisions[decisions.label == 0]
    history_samples = int(CONFIG.input_window_seconds * CONFIG.target_sfreq)

    for row in sample.itertuples(index=False):
        array = np.load(row.X_path, mmap_mode="r")
        with open(row.channel_availability_path, encoding="utf-8") as handle:
            mask = np.asarray(json.load(handle), dtype=bool)
        step = max(1, array.shape[1] // 400_000)
        block = np.asarray(array[:, ::step], dtype=np.float32)
        for c, channel in enumerate(CONFIG.canonical_channel_names):
            if not mask[c]:
                continue
            rows.append({
                "recording_id": row.recording_id, "subject": row.subject,
                "split": row.split, "channel": channel,
                "mean_z": float(block[c].mean()),
                "std_z": float(block[c].std()),
                "p99_abs_z": float(np.percentile(np.abs(block[c]), 99)),
            })
        for pool, source in ((psd_pre, positives), (psd_inter, negatives)):
            subset = source[source.recording_id == row.recording_id]
            if subset.empty:
                continue
            pick = subset.iloc[int(rng.integers(len(subset)))]
            stop = int(pick["decision_end_sample"])
            start = stop - history_samples
            if start < 0 or stop > array.shape[1]:
                continue
            seg = np.asarray(array[:, max(start, stop - 5 * 60 * 256):stop],
                             dtype=np.float64)
            for c in range(len(CONFIG.canonical_channel_names)):
                if not mask[c]:
                    continue
                f, p = welch(seg[c], fs=CONFIG.target_sfreq, nperseg=1024)
                pool.append(p)
                freqs = f

    return pd.DataFrame(rows), {
        "freqs": freqs,
        "pre": np.asarray(psd_pre) if psd_pre else np.empty((0, 0)),
        "inter": np.asarray(psd_inter) if psd_inter else np.empty((0, 0)),
    }


def figure_amplitude(stats: pd.DataFrame, sigma_volts: dict[str, float],
                     path: Path) -> None:
    """Residual per-patient amplitude spread after the single global z-score."""
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    axis = axes[0]
    for i, channel in enumerate(CONFIG.canonical_channel_names):
        sub = stats[stats.channel == channel]
        if sub.empty:
            continue
        volts = sigma_volts.get(channel)
        label = channel if volts is None else f"{channel}  (1 z = {volts*1e6:.0f} µV)"
        axis.hist(sub["std_z"], bins=50, alpha=0.65,
                  color=[BLUE, ORANGE, AQUA][i], label=label)
    axis.axvline(1.0, color=INK, linestyle="--", linewidth=1.2)
    axis.annotate("intended σ = 1 z", xy=(0.42, 0.9), xycoords="axes fraction",
                  fontsize=9, color=INK)
    axis.set(title="Per-file amplitude spread after global z-score",
             xlabel=f"Standard deviation of one EDF file ({Z_UNIT})",
             ylabel="Channel × file pairs (count)", xscale="log")
    axis.legend(frameon=False, fontsize=8)
    style(axis)

    axis = axes[1]
    by_patient = stats.groupby("subject")["std_z"].median().sort_values()
    axis.bar(np.arange(len(by_patient)), by_patient.values, color=BLUE, width=1.0)
    axis.axhline(1.0, color=ORANGE, linestyle="--", linewidth=1.4)
    ratio = by_patient.max() / max(by_patient.min(), 1e-9)
    axis.set(title=f"Median amplitude per patient — spans {ratio:.0f}×",
             xlabel="Patients (rank, sorted by median σ)",
             ylabel=f"Median σ across that patient's files ({Z_UNIT})", yscale="log")
    style(axis)

    axis = axes[2]
    for split in ["train", "validation", "test"]:
        sub = stats[stats.split == split]
        if sub.empty:
            continue
        axis.hist(sub["p99_abs_z"], bins=40, alpha=0.6, color=SPLIT_COLOR[split],
                  label=split, density=True)
    axis.set(title="99th-percentile amplitude by split",
             xlabel=f"99th percentile of |amplitude| ({Z_UNIT})",
             ylabel="Probability density (1/z)", xscale="log")
    axis.legend(frameon=False, fontsize=9, title="split", title_fontsize=9)
    style(axis)

    save(figure, path,
         "Amplitude — one global scaler leaves large per-patient differences in the input")


def figure_spectra(psd: dict[str, np.ndarray], path: Path) -> None:
    """Average spectra and band power, pre-ictal versus interictal."""
    if psd["freqs"] is None or psd["pre"].size == 0 or psd["inter"].size == 0:
        return
    freqs = psd["freqs"]
    band = (freqs >= 0.3) & (freqs <= 45)

    figure, axes = plt.subplots(1, 2, figsize=(14, 4.8))

    axis = axes[0]
    for data, name, color in ((psd["inter"], "interictal (label 0)", BLUE),
                              (psd["pre"], "pre-ictal (label 1)", ORANGE)):
        median = np.median(data, axis=0)
        q1, q3 = np.percentile(data, [25, 75], axis=0)
        axis.plot(freqs[band], median[band], color=color, linewidth=2,
                  label=f"{name}, n = {len(data)}")
        axis.fill_between(freqs[band], q1[band], q3[band], color=color, alpha=0.18)
    axis.set(title="Welch PSD, final 5 min of the history (median, IQR band)",
             xlabel="Frequency (Hz)",
             ylabel="Power spectral density (z²/Hz)", yscale="log")
    axis.legend(frameon=False, fontsize=9)
    style(axis)

    axis = axes[1]
    names = list(BANDS)
    width = 0.36
    positions = np.arange(len(names))
    for offset, data, name, color in ((-width/2, psd["inter"], "interictal", BLUE),
                                      (width/2, psd["pre"], "pre-ictal", ORANGE)):
        powers = []
        for lo, hi in BANDS.values():
            sel = (freqs >= lo) & (freqs < hi)
            powers.append(np.median(data[:, sel].sum(axis=1)))
        axis.bar(positions + offset, powers, width=width, color=color, label=name)
    axis.set(title="Median band power", xticks=positions,
             xticklabels=[f"{n}\n{BANDS[n][0]:g}–{BANDS[n][1]:g} Hz" for n in names],
             xlabel="Frequency band",
             ylabel="Band power (z², summed over band)", yscale="log")
    axis.legend(frameon=False, fontsize=9)
    style(axis)

    save(figure, path, "Spectra — where a pre-ictal signature would have to live")


# ----------------------------------------------------------------------
# Self-contained HTML report
# ----------------------------------------------------------------------

FIGURE_NOTES: list[dict[str, str]] = [
    {
        "file": "01_cohort.png",
        "num": "01",
        "title": "Cohort",
        "reads": "Who is in the dataset and how much EEG each patient contributes. "
                 "Left to right, top row: number of EDF files per patient, the wall-clock "
                 "hours those files contain, and the decisions they yield. Bottom row: "
                 "split totals, the distribution of file lengths, and which electrode "
                 "pair each file physically recorded.",
        "take": "File count and data volume are different things, and the first panel "
                "alone is misleading. Subject 081 has ~100 EDF files while subject 124 has "
                "one, yet the duration panel shows most patients land between 50 and 200 h "
                "regardless. Median file length is 1.6 h and the mode sits just above the "
                "55-minute floor needed to produce a single decision, so the dataset is "
                "many short recordings rather than a few long ones. "
                "CROSS_HEAD appears in 2,075 of 2,494 files (83 %); genuinely bilateral "
                "behind-the-ear recordings are the minority at 419 files (17 %).",
    },
    {
        "file": "02_labels.png",
        "num": "02",
        "title": "Labels",
        "reads": "How the binary target is distributed. Class balance per split on a log "
                 "axis, how many positive decisions each target seizure generates, and how "
                 "many decisions survive the clean-history filter in each file.",
        "take": "Prevalence is 0.63 % and, importantly, near-identical across train, "
                "validation and test (0.63 / 0.62 / 0.62 %) — the splits are comparable, so "
                "a metric shift between them is a model property, not a data artefact. "
                "The middle panel is the sharpest constraint in the whole project: 300 of "
                "317 seizures produce exactly 10 positives, because SOP / stride = "
                "600 s / 60 s. The positive class size is set by arithmetic, not by how "
                "much EEG exists. The retention panel is bimodal — a large spike near zero "
                "from files too short to yield decisions at all.",
    },
    {
        "file": "03_seizures.png",
        "num": "03",
        "title": "Seizures",
        "reads": "The seizure inventory and the eligibility filter that decides which "
                 "seizures can serve as prediction targets at all.",
        "take": "883 seizures are annotated but only 317 (36 %) have the 60 minutes of "
                "continuous clean EEG before onset that eligibility requires — the single "
                "largest source of data loss in the pipeline. Every eligible seizure is "
                "used, so the funnel's second and third bars are identical. "
                "Median seizure duration is 33 s. The per-patient distribution is severely "
                "skewed: the top patient contributes 75 seizures while most contribute "
                "fewer than five, so pooled metrics are dominated by a handful of people.",
    },
    {
        "file": "04_signals.png",
        "num": "04",
        "title": "Signals",
        "reads": "Two real model inputs at full length: one 45-minute history ending "
                 "10 minutes before a seizure onset, and one that is not. Both are drawn "
                 "from the stored, globally z-scored shards, so this is exactly what the "
                 "network receives. Channels are vertically offset for legibility.",
        "take": "This is the clearest picture of why the baseline failed. The positive "
                "example peaks near 4 z; the negative example peaks past 15 z and is full "
                "of high-amplitude artefact transients. The difference between these two "
                "windows is dominated by which patient they came from, not by whether a "
                "seizure follows. Note also that they are missing different electrodes — "
                "CROSS_HEAD in one, BTE_LEFT in the other — and the absent channel is a "
                "flat zero-filled line the encoder still processes.",
    },
    {
        "file": "05_chunks.png",
        "num": "05",
        "title": "Chunks",
        "reads": "The actual tensor handed to EEGNet: 3 channels x 1,280 samples, five "
                 "seconds at 256 Hz. Three chunks from a single positive history — the "
                 "earliest, the middle, and the one ending at the decision instant.",
        "take": "Each decision is 540 of these, encoded by shared weights and then pooled. "
                "Nothing in the chunk tells the encoder where in the 45 minutes it sits, "
                "which is why the mean-pooling baseline is permutation-invariant: the same "
                "540 chunks in any order produce an identical logit. Visually the chunk at "
                "the decision instant is not obviously different from the one 45 minutes "
                "earlier, which is the honest difficulty of the task.",
    },
    {
        "file": "06_amplitude.png",
        "num": "06",
        "title": "Amplitude",
        "reads": "What the single global z-score actually achieved. Per-file standard "
                 "deviation, median amplitude per patient, and the 99th-percentile "
                 "amplitude by split. Amplitude is in z units; the legend gives the "
                 "conversion back to microvolts per channel.",
        "take": "The scaler is badly calibrated for the typical recording. It targets "
                "sigma = 1, but the median file sits at 0.386 z and 57 % fall below 0.5 z: "
                "a small number of very high-amplitude recordings dominated the fit, so "
                "most patients arrive compressed toward zero. Median amplitude per patient "
                "spans 44x, from 0.09 z to 4.08 z. That residual spread is impedance, skin "
                "contact and electrode seating — patient identity rather than physiology — "
                "and it is the most likely thing the baseline learned that could not "
                "transfer to unseen patients. This figure is the direct motivation for "
                "per-window normalisation.",
    },
    {
        "file": "07_spectra.png",
        "num": "07",
        "title": "Spectra",
        "reads": "Welch power spectral density over the final five minutes before a "
                 "decision, pre-ictal versus interictal, plus median power in the five "
                 "classical EEG bands.",
        "take": "Pre-ictal windows show 2-3x more power than interictal ones in every "
                "band, delta through gamma. Read that with suspicion rather than "
                "enthusiasm: a uniform broadband lift is the signature of an amplitude "
                "difference, not of a change in spectral shape, and it is exactly what "
                "figure 06 predicts if the two groups happen to come from different "
                "patients. The sample is also unbalanced and not patient-matched — 28 "
                "pre-ictal against 240 interictal — so this comparison is confounded and "
                "should not be treated as a discovered pre-ictal feature. A patient-matched "
                "version is the experiment worth running.",
    },
]


def build_html_report(
    output: Path,
    summary: dict[str, object],
    stats: pd.DataFrame,
) -> Path:
    """Write a self-contained HTML report with the figures embedded as data URIs."""
    def embed(name: str) -> str:
        data = (output / name).read_bytes()
        return "data:image/png;base64," + base64.b64encode(data).decode("ascii")

    by_split = summary["by_split"]
    sigma = summary["units"]["global_sigma_volts"]
    per_file_sigma = stats["std_z"].median() if not stats.empty else float("nan")
    patient_medians = stats.groupby("subject")["std_z"].median() if not stats.empty else None
    sigma_ratio = (
        patient_medians.max() / patient_medians.min()
        if patient_medians is not None and len(patient_medians) and patient_medians.min() > 0
        else float("nan")
    )

    stat_cards = [
        ("Patients with usable data", f"{summary['patients_with_data']}", "of 125 configured"),
        ("EDF files", f"{summary['edf_files']:,}", "behind-the-ear recordings"),
        ("EEG duration", f"{summary['eeg_hours']:,.0f} h", "median 1.6 h per file"),
        ("Decisions", f"{summary['decisions']:,}", "one per 60 s of usable EEG"),
        ("Positive decisions", f"{summary['positive_decisions']:,}", f"{summary['prevalence']*100:.2f} % prevalence"),
        ("Seizures annotated", f"{summary['seizures_annotated']}", "in the event files"),
        ("Seizures usable", f"{summary['seizures_targeted']}", "have 60 min clean history"),
        ("Amplitude spread", f"{sigma_ratio:.0f}×", "median σ, patient to patient"),
    ]

    cards_html = "\n".join(
        f'      <div class="stat"><span class="stat-label">{label}</span>'
        f'<span class="stat-value">{value}</span>'
        f'<span class="stat-note">{note}</span></div>'
        for label, value, note in stat_cards
    )

    splits_html = "\n".join(
        f"          <tr><td>{name}</td><td class='m'>{d['patients']}</td>"
        f"<td class='m'>{d['decisions']:,}</td><td class='m'>{d['positive']:,}</td>"
        f"<td class='m'>{d['positive']/d['decisions']*100:.2f} %</td></tr>"
        for name, d in (
            ("train", by_split["train"]),
            ("validation", by_split["validation"]),
            ("test", by_split["test"]),
        )
    )

    figures_html = "\n".join(
        f'''  <section class="fig">
    <div class="fig-head">
      <span class="fig-num">{note["num"]}</span>
      <h2>{note["title"]}</h2>
    </div>
    <img src="{embed(note["file"])}" alt="{note["title"]} figure">
    <div class="fig-body">
      <div class="fig-col"><h3>What it shows</h3><p>{note["reads"]}</p></div>
      <div class="fig-col"><h3>What to take from it</h3><p>{note["take"]}</p></div>
    </div>
    <p class="fig-file">{note["file"]}</p>
  </section>'''
        for note in FIGURE_NOTES
    )

    sigma_rows = " · ".join(f"{k} 1 z = {v*1e6:.0f} µV" for k, v in sigma.items())

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SeizeIT2 Dataset Atlas</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500&display=swap">
<style>
:root {{
  color-scheme: light;
  --paper:#F5F6F8; --surface:#FFFFFF; --surface-2:#EEF1F5;
  --ink:#10141A; --ink-2:#4C5568; --ink-3:#6E7788;
  --rule:#DCE0E6; --rule-firm:#C3CAD4;
  --accent:#1E5FA8; --accent-bg:#E7EEF8; --hot:#C2542F;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    color-scheme: dark;
    --paper:#0F1216; --surface:#171B21; --surface-2:#1E242C;
    --ink:#EDF0F4; --ink-2:#A8B2C0; --ink-3:#7E8797;
    --rule:#272D36; --rule-firm:#3A424E;
    --accent:#6BA5EC; --accent-bg:#16222F; --hot:#E58468;
  }}
}}
:root[data-theme="dark"] {{
  color-scheme: dark;
  --paper:#0F1216; --surface:#171B21; --surface-2:#1E242C;
  --ink:#EDF0F4; --ink-2:#A8B2C0; --ink-3:#7E8797;
  --rule:#272D36; --rule-firm:#3A424E;
  --accent:#6BA5EC; --accent-bg:#16222F; --hot:#E58468;
}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--paper);color:var(--ink);
  font-family:"IBM Plex Sans",ui-sans-serif,system-ui,sans-serif;
  font-size:15.5px;line-height:1.62;-webkit-font-smoothing:antialiased}}
.wrap{{max-width:1180px;margin:0 auto;padding:56px 26px 96px;
  display:flex;flex-direction:column;gap:44px}}
.eyebrow{{font-family:"IBM Plex Mono",monospace;font-size:11px;font-weight:500;
  letter-spacing:.15em;text-transform:uppercase;color:var(--ink-3);margin:0 0 12px}}
h1{{font-family:Archivo,sans-serif;font-weight:700;font-size:clamp(32px,5vw,50px);
  line-height:1.04;letter-spacing:-.022em;margin:0 0 16px;text-wrap:balance}}
.standfirst{{font-size:18px;color:var(--ink-2);max-width:64ch;margin:0}}
.units{{margin-top:22px;padding:16px 18px;background:var(--surface);border:1px solid var(--rule);
  border-left:2px solid var(--accent);border-radius:2px;
  font-family:"IBM Plex Mono",monospace;font-size:12.5px;line-height:1.95;color:var(--ink-2)}}
.units b{{color:var(--accent);font-weight:600}}
.stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px}}
.stat{{background:var(--surface);border:1px solid var(--rule);border-radius:2px;
  padding:15px 17px;display:flex;flex-direction:column;gap:3px}}
.stat-label{{font-size:12.5px;color:var(--ink-3)}}
.stat-value{{font-family:"IBM Plex Mono",monospace;font-size:25px;font-weight:600;
  letter-spacing:-.01em;font-variant-numeric:tabular-nums}}
.stat-note{{font-size:12px;color:var(--ink-3)}}
h2.sec{{font-family:Archivo,sans-serif;font-size:12.5px;font-weight:600;letter-spacing:.11em;
  text-transform:uppercase;color:var(--ink-3);margin:0 0 14px;padding-bottom:9px;
  border-bottom:1px solid var(--rule-firm)}}
.tw{{overflow-x:auto;border:1px solid var(--rule);border-radius:2px;background:var(--surface)}}
table{{border-collapse:collapse;width:100%;font-size:14px}}
th,td{{text-align:left;padding:10px 15px;border-bottom:1px solid var(--rule);white-space:nowrap}}
thead th{{font-family:"IBM Plex Mono",monospace;font-size:10px;font-weight:600;
  letter-spacing:.1em;text-transform:uppercase;color:var(--ink-3);background:var(--surface-2)}}
tbody tr:last-child td{{border-bottom:none}}
td.m{{font-family:"IBM Plex Mono",monospace;font-variant-numeric:tabular-nums}}
.fig{{background:var(--surface);border:1px solid var(--rule);border-radius:3px;
  padding:22px 24px 18px;display:flex;flex-direction:column;gap:16px}}
.fig-head{{display:flex;align-items:baseline;gap:13px;
  border-bottom:1px solid var(--rule);padding-bottom:12px}}
.fig-num{{font-family:"IBM Plex Mono",monospace;font-size:12px;font-weight:600;
  color:var(--paper);background:var(--accent);border-radius:2px;padding:3px 8px}}
.fig h2{{font-family:Archivo,sans-serif;font-size:21px;font-weight:600;margin:0;
  letter-spacing:-.012em}}
.fig img{{display:block;width:100%;height:auto;border:1px solid var(--rule);border-radius:2px;
  background:#fff}}
.fig-body{{display:grid;grid-template-columns:1fr 1fr;gap:26px}}
@media (max-width:820px){{.fig-body{{grid-template-columns:1fr;gap:16px}}}}
.fig-col h3{{font-family:Archivo,sans-serif;font-size:11px;font-weight:600;letter-spacing:.1em;
  text-transform:uppercase;color:var(--ink-3);margin:0 0 7px}}
.fig-col p{{margin:0;color:var(--ink-2);font-size:14.5px}}
.fig-file{{margin:0;font-family:"IBM Plex Mono",monospace;font-size:11.5px;color:var(--ink-3);
  border-top:1px solid var(--rule);padding-top:11px}}
.closing{{background:var(--accent-bg);border:1px solid var(--accent);border-radius:3px;
  padding:24px 26px;display:flex;flex-direction:column;gap:12px}}
.closing h2{{font-family:Archivo,sans-serif;font-size:20px;font-weight:600;margin:0;
  letter-spacing:-.012em}}
.closing p{{margin:0;color:var(--ink-2);max-width:74ch}}
.closing ol{{margin:4px 0 0;padding-left:20px;color:var(--ink-2)}}
.closing li{{margin-bottom:7px}}
footer{{border-top:1px solid var(--rule-firm);padding-top:20px;
  font-family:"IBM Plex Mono",monospace;font-size:11.5px;color:var(--ink-3);line-height:1.9}}
code{{font-family:"IBM Plex Mono",monospace;font-size:.9em;background:var(--surface-2);
  padding:1px 5px;border-radius:2px}}
</style>
</head>
<body>
<div class="wrap">

  <header>
    <p class="eyebrow">SeizeIT2 · behind-the-ear EEG · dataset atlas</p>
    <h1>What the seizure-prediction data actually looks like</h1>
    <p class="standfirst">Seven figures over {summary['eeg_hours']:,.0f} hours of EEG from
    {summary['patients_with_data']} patients, generated by
    <code>scripts/visualize_dataset.py</code>. Each is paired with what it shows and what
    it implies for modelling — including the two that explain why the first baseline
    scored at chance on held-out patients.</p>
    <div class="units">
<b>Units used on every axis</b>
count   a number of things — EDF files, decisions, seizures, patients
h/min/s wall-clock duration of EEG
Hz      frequency
z       dimensionless amplitude; 1 z = one global channel σ
        ({sigma_rows})</div>
  </header>

  <section>
    <h2 class="sec">At a glance</h2>
    <div class="stats">
{cards_html}
    </div>
  </section>

  <section>
    <h2 class="sec">Splits — whole patients, never mixed</h2>
    <div class="tw"><table>
      <thead><tr><th>Split</th><th>Patients</th><th>Decisions</th><th>Positive</th><th>Prevalence</th></tr></thead>
      <tbody>
{splits_html}
      </tbody>
    </table></div>
  </section>

{figures_html}

  <section class="closing">
    <h2>What the data says about the modelling problem</h2>
    <p>Three constraints are visible in the figures and none of them are fixable by
    training longer.</p>
    <ol>
      <li><strong>The positive class is set by arithmetic.</strong> SOP / stride = 10
      decisions per seizure, and only 317 seizures qualify, so 3,113 positives is the
      ceiling until the stride or the eligibility rule changes.</li>
      <li><strong>Between-patient amplitude dwarfs the pre-ictal signal.</strong> Median σ
      spans {sigma_ratio:.0f}× across patients while the global scaler leaves the typical file at
      {per_file_sigma:.2f} z. Separating patients is easy; separating pre-ictal states is not.</li>
      <li><strong>The obvious spectral difference is confounded.</strong> Pre-ictal power is
      uniformly higher across all five bands, which is an amplitude effect, and the samples
      are not patient-matched.</li>
    </ol>
    <p>Figures 04 and 06 together are the argument for normalising each window on its own
    robust statistics rather than on one global constant, and figure 02 is the argument for
    densifying positives near onset.</p>
  </section>

  <footer>
    Generated by scripts/visualize_dataset.py · figures embedded, file is self-contained<br>
    Source figures and recording_statistics.csv sit beside this file in outputs/dataset_figures/<br>
    Amplitude statistics sampled from {len(stats)} channel-file pairs
  </footer>

</div>
</body>
</html>"""

    path = output / "dataset_report.html"
    path.write_text(html, encoding="utf-8")
    return path


def main() -> None:
    arguments = parse_arguments()
    CONFIG.validate()
    rng = np.random.default_rng(arguments.seed)
    output = arguments.output_dir
    output.mkdir(parents=True, exist_ok=True)

    log("Loading manifests...")
    decisions, seizures, shards = load_manifests()
    sigma_volts = load_global_sigma()
    log(f"  decisions {len(decisions):,} | seizures {len(seizures)} | shards {len(shards)}")

    log("Reading shard durations from .npy headers...")
    durations = shard_durations_hours(shards)
    log(f"  {durations.sum():,.0f} h of EEG across {len(durations)} files")

    log("Figures...")
    figure_cohort(decisions, shards, durations, output / "01_cohort.png")
    figure_labels(decisions, output / "02_labels.png")
    figure_seizures(seizures, decisions, output / "03_seizures.png")
    figure_signals(decisions, shards, output / "04_signals.png", rng)
    figure_chunks(decisions, shards, output / "05_chunks.png", rng)

    log(f"Sampling {arguments.recording_sample} files for signal statistics...")
    stats, psd = sample_recording_stats(shards, decisions, arguments.recording_sample, rng)
    log(f"  {len(stats)} channel-file rows | {len(psd['pre'])} pre-ictal PSDs")
    figure_amplitude(stats, sigma_volts, output / "06_amplitude.png")
    figure_spectra(psd, output / "07_spectra.png")

    stats.to_csv(output / "recording_statistics.csv", index=False)

    summary = {
        "units": {
            "count": "number of things (EDF files, decisions, seizures, patients)",
            "h/min/s": "wall-clock duration of EEG",
            "Hz": "frequency",
            "z": "dimensionless amplitude, 1 z = 1 global channel sigma",
            "global_sigma_volts": sigma_volts,
        },
        "patients_with_data": int(decisions["subject"].nunique()),
        "edf_files": int(shards["recording_id"].nunique()),
        "eeg_hours": float(durations.sum()),
        "decisions": int(len(decisions)),
        "positive_decisions": int((decisions["label"] == 1).sum()),
        "prevalence": float((decisions["label"] == 1).mean()),
        "seizures_annotated": int(len(seizures)),
        "seizures_eligible": int(
            seizures["eligible_for_prediction"].astype(str).str.lower().eq("true").sum()
        ),
        "seizures_targeted": int(
            decisions.loc[decisions.label == 1, "target_seizure_id"].nunique()
        ),
        "by_split": {
            split: {
                "patients": int(group["subject"].nunique()),
                "decisions": int(len(group)),
                "positive": int((group["label"] == 1).sum()),
            }
            for split, group in decisions.groupby("split")
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report = build_html_report(output, summary, stats)
    log(f"  wrote {report.name}  ({report.stat().st_size/1e6:.1f} MB, self-contained)")
    log(f"\nFigures, summary.json and dataset_report.html in {output}")


if __name__ == "__main__":
    main()
