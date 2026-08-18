r"""Render an exploratory picture of the processed SeizeIT2 dataset.

Produces one PNG per topic in ``outputs/dataset_figures/``:

    01_cohort.png        recordings, hours and decisions per patient/split
    02_labels.png        class balance, retention, decisions per seizure
    03_seizures.png      seizure counts, durations and why most are ineligible
    04_signals.png       real 45-minute histories, positive versus negative
    05_chunks.png        the 5-second chunk the encoder actually sees
    06_amplitude.png     per-patient amplitude spread after global z-scoring
    07_spectra.png       Welch PSD and band power, pre-ictal versus interictal

Everything is read from the manifests plus a sample of the standardized
shards; nothing is recomputed from the raw EDFs.

    python scripts/visualize_dataset.py --recording-sample 60
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

BANDS = {
    "delta 0.5-4": (0.5, 4.0),
    "theta 4-8": (4.0, 8.0),
    "alpha 8-13": (8.0, 13.0),
    "beta 13-30": (13.0, 30.0),
    "gamma 30-40": (30.0, 40.0),
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
    shards = pd.read_csv(manifests / "processed_shard_manifest.csv", dtype={"subject": str})
    return decisions, seizures, shards


def recording_hours(shards: pd.DataFrame, sample: pd.DataFrame) -> pd.Series:
    """Duration in hours for each sampled shard, from the array header only."""
    hours = {}
    for row in sample.itertuples(index=False):
        array = np.load(row.X_path, mmap_mode="r")
        hours[row.recording_id] = array.shape[1] / CONFIG.target_sfreq / 3600.0
    return pd.Series(hours)


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------


def figure_cohort(decisions: pd.DataFrame, shards: pd.DataFrame, path: Path) -> None:
    """Who is in the dataset and how much data each patient contributes."""
    figure, axes = plt.subplots(2, 2, figsize=(14, 9))

    # Recordings per patient
    per_patient = (
        shards.groupby(["split", "subject"])
        .size()
        .rename("recordings")
        .reset_index()
    )
    axis = axes[0, 0]
    for split, group in per_patient.groupby("split"):
        axis.scatter(
            group["subject"].astype(int),
            group["recordings"],
            s=18,
            color=SPLIT_COLOR[split],
            label=split,
            alpha=0.85,
        )
    axis.set(
        title="Recordings per patient",
        xlabel="Subject",
        ylabel="Recordings",
        yscale="log",
    )
    axis.legend(frameon=False, fontsize=9)
    style(axis)

    # Decisions per patient
    dec_patient = (
        decisions.groupby(["split", "subject"]).size().rename("decisions").reset_index()
    )
    axis = axes[0, 1]
    for split, group in dec_patient.groupby("split"):
        axis.scatter(
            group["subject"].astype(int),
            group["decisions"],
            s=18,
            color=SPLIT_COLOR[split],
            label=split,
            alpha=0.85,
        )
    axis.set(
        title="Decisions per patient (1 per minute of usable EEG)",
        xlabel="Subject",
        ylabel="Decisions",
        yscale="log",
    )
    axis.legend(frameon=False, fontsize=9)
    style(axis)

    # Split totals
    axis = axes[1, 0]
    totals = decisions.groupby("split").agg(
        decisions=("label", "size"),
        positives=("label", "sum"),
        patients=("subject", "nunique"),
    )
    order = ["train", "validation", "test"]
    totals = totals.reindex(order)
    positions = np.arange(len(order))
    axis.bar(
        positions,
        totals["decisions"],
        color=[SPLIT_COLOR[s] for s in order],
        width=0.6,
    )
    for x, (n, p, pat) in enumerate(
        zip(totals["decisions"], totals["positives"], totals["patients"], strict=True)
    ):
        axis.annotate(
            f"{n:,}\n{p:,} pos\n{pat} patients",
            xy=(x, n),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color=INK,
        )
    axis.set(
        title="Decisions by split",
        xticks=positions,
        xticklabels=order,
        ylabel="Decisions",
        ylim=(0, totals["decisions"].max() * 1.28),
    )
    style(axis)

    # Channel availability combinations
    axis = axes[1, 1]
    combos: dict[str, int] = {}
    for row in shards.itertuples(index=False):
        with open(row.channel_availability_path, encoding="utf-8") as handle:
            mask = tuple(json.load(handle))
        names = [
            n for n, a in zip(CONFIG.canonical_channel_names, mask, strict=True) if a
        ]
        combos["+".join(names)] = combos.get("+".join(names), 0) + 1
    labels = sorted(combos, key=lambda k: -combos[k])
    values = [combos[k] for k in labels]
    axis.barh(np.arange(len(labels)), values, color=BLUE, height=0.6)
    for i, v in enumerate(values):
        axis.annotate(
            f"{v:,}",
            xy=(v, i),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=INK,
        )
    axis.set(
        title="Electrode pairs present (2 of 3 per recording)",
        yticks=np.arange(len(labels)),
        yticklabels=labels,
        xlabel="Recordings",
        xlim=(0, max(values) * 1.18),
    )
    axis.invert_yaxis()
    style(axis)

    save(figure, path, "Cohort — 125 patients, whole-patient holdout")


def figure_labels(decisions: pd.DataFrame, path: Path) -> None:
    """Class balance and how positives are distributed."""
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    # Class balance, log scale
    axis = axes[0]
    order = ["train", "validation", "test"]
    neg = [int((decisions[decisions.split == s].label == 0).sum()) for s in order]
    pos = [int((decisions[decisions.split == s].label == 1).sum()) for s in order]
    positions = np.arange(len(order))
    axis.bar(positions - 0.19, neg, width=0.36, color=BLUE, label="negative")
    axis.bar(positions + 0.19, pos, width=0.36, color=ORANGE, label="positive")
    for x, (n, p) in enumerate(zip(neg, pos, strict=True)):
        axis.annotate(f"{n:,}", (x - 0.19, n), xytext=(0, 4), textcoords="offset points",
                      ha="center", fontsize=8, color=INK)
        axis.annotate(f"{p:,}\n{100*p/(n+p):.2f}%", (x + 0.19, p), xytext=(0, 4),
                      textcoords="offset points", ha="center", fontsize=8, color=INK)
    axis.set(title="Class balance (log)", yscale="log", xticks=positions,
             xticklabels=order, ylabel="Decisions")
    axis.legend(frameon=False, fontsize=9)
    style(axis)

    # Positives per target seizure
    axis = axes[1]
    per_seizure = (
        decisions[decisions.label == 1].groupby("target_seizure_id").size()
    )
    axis.hist(per_seizure, bins=np.arange(0.5, per_seizure.max() + 1.5),
              color=ORANGE, alpha=0.9)
    axis.axvline(
        CONFIG.seizure_occurrence_period_minutes * 60 / CONFIG.input_stride_seconds,
        color=INK, linestyle="--", linewidth=1.2,
    )
    axis.annotate(
        f"SOP/stride = {int(CONFIG.seizure_occurrence_period_minutes*60/CONFIG.input_stride_seconds)}",
        xy=(0.55, 0.9), xycoords="axes fraction", fontsize=9, color=INK,
    )
    axis.set(title=f"Positive decisions per seizure (n={len(per_seizure)})",
             xlabel="Decisions", ylabel="Seizures")
    style(axis)

    # Why decisions were dropped is not recorded per candidate, so show retention
    axis = axes[2]
    retained = decisions.groupby("recording_id").size()
    axis.hist(retained, bins=60, color=BLUE, alpha=0.9)
    axis.set(title=f"Decisions retained per recording (n={len(retained)})",
             xlabel="Decisions", ylabel="Recordings", yscale="log")
    style(axis)

    save(figure, path, "Labels — 0.63 % of decisions are positive")


def figure_seizures(seizures: pd.DataFrame, decisions: pd.DataFrame, path: Path) -> None:
    """Seizure inventory and the eligibility funnel."""
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    eligible = seizures["eligible_for_prediction"].astype(str).str.lower().eq("true")

    # Funnel
    axis = axes[0]
    targeted = decisions.loc[decisions.label == 1, "target_seizure_id"].nunique()
    stages = ["annotated", "60-min clear\n(eligible)", "actually\ntargeted"]
    values = [len(seizures), int(eligible.sum()), int(targeted)]
    axis.bar(np.arange(3), values, color=[BLUE, BLUE, ORANGE], width=0.6)
    for i, v in enumerate(values):
        axis.annotate(f"{v}", (i, v), xytext=(0, 4), textcoords="offset points",
                      ha="center", fontsize=10, color=INK, fontweight="bold")
    axis.set(title="Seizure eligibility funnel", xticks=np.arange(3),
             xticklabels=stages, ylabel="Seizures",
             ylim=(0, max(values) * 1.18))
    style(axis)

    # Durations
    axis = axes[1]
    dur = seizures["duration_seconds"].dropna()
    axis.hist(dur[~eligible], bins=50, color=BLUE, alpha=0.75, label="ineligible")
    axis.hist(dur[eligible], bins=50, color=ORANGE, alpha=0.8, label="eligible")
    axis.set(title="Seizure duration", xlabel="Seconds", ylabel="Seizures", yscale="log")
    axis.legend(frameon=False, fontsize=9)
    style(axis)

    # Seizures per patient
    axis = axes[2]
    per_patient = seizures.groupby("subject").size().sort_values(ascending=False)
    elig_pp = seizures[eligible].groupby("subject").size()
    axis.bar(np.arange(len(per_patient)), per_patient.values, color=BLUE,
             width=1.0, label="annotated")
    axis.bar(np.arange(len(per_patient)),
             [elig_pp.get(s, 0) for s in per_patient.index],
             color=ORANGE, width=1.0, label="eligible")
    axis.set(title=f"Seizures per patient ({len(per_patient)} patients with any)",
             xlabel="Patients, sorted", ylabel="Seizures")
    axis.legend(frameon=False, fontsize=9)
    style(axis)

    save(figure, path, "Seizures — 883 annotated, 317 usable as prediction targets")


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
        (axes[0], 1, "POSITIVE — seizure onset within 10 min of the decision", ORANGE),
        (axes[1], 0, "NEGATIVE — no seizure in the following 10 min", BLUE),
    ):
        picked = pick_example(decisions, shards, label, rng)
        if picked is None:
            continue
        history, row, mask = picked
        offsets = [8, 0, -8]
        for c, (channel, offset) in enumerate(
            zip(CONFIG.canonical_channel_names, offsets, strict=True)
        ):
            present = bool(mask[c])
            axis.plot(
                minutes,
                history[c] + offset,
                linewidth=0.25,
                color=color if present else "#BBBBBB",
                alpha=0.9 if present else 0.7,
            )
            axis.annotate(
                f"{channel}{'' if present else '  (absent → zero-filled)'}",
                xy=(0.002, offset + 3.2),
                xycoords=("axes fraction", "data"),
                fontsize=9,
                color=INK,
            )
        axis.set(
            title=f"{name}   ·   {row['recording_id']}  t={row['decision_time_seconds']:.0f}s",
            ylabel="z-scored amplitude (offset)",
            ylim=(-16, 16),
            xlim=(0, 45),
        )
        style(axis)

    axes[1].set_xlabel("Minutes within the 45-minute history  (decision at 45)")
    save(figure, path,
         "One decision = 45 min x 3 channels = 691,200 samples, reshaped to 540 x 3 x 1280")


def figure_chunks(decisions: pd.DataFrame, shards: pd.DataFrame,
                  path: Path, rng: np.random.Generator) -> None:
    """What the encoder is actually handed: single 5-second chunks."""
    picked = pick_example(decisions, shards, 1, rng)
    if picked is None:
        return
    history, row, mask = picked
    chunk_samples = int(CONFIG.chunk_window_seconds * CONFIG.target_sfreq)
    total_chunks = history.shape[1] // chunk_samples
    # First chunk, middle chunk, and the last chunk before the decision.
    picks = [0, total_chunks // 2, total_chunks - 1]
    names = ["chunk 0\n(45 min before)", f"chunk {total_chunks//2}\n(22.5 min before)",
             f"chunk {total_chunks-1}\n(at the decision)"]

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
    seconds = np.arange(chunk_samples) / CONFIG.target_sfreq
    for axis, index, name in zip(axes, picks, names, strict=True):
        chunk = history[:, index * chunk_samples:(index + 1) * chunk_samples]
        for c, (channel, offset) in enumerate(
            zip(CONFIG.canonical_channel_names, [6, 0, -6], strict=True)
        ):
            present = bool(mask[c])
            axis.plot(seconds, chunk[c] + offset, linewidth=0.8,
                      color=[BLUE, ORANGE, AQUA][c] if present else "#BBBBBB")
        axis.set(title=name, xlabel="Seconds", xlim=(0, 5), ylim=(-12, 12))
        style(axis)
    axes[0].set_ylabel("z-scored amplitude (offset)")
    save(figure, path,
         f"The encoder input: (3 channels x {chunk_samples} samples), applied {total_chunks}x per decision")


def sample_recording_stats(
    shards: pd.DataFrame,
    decisions: pd.DataFrame,
    n_sample: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Per-recording amplitude stats plus pooled PSDs, from a random sample."""
    sample = shards.sample(n=min(n_sample, len(shards)), random_state=int(rng.integers(1 << 30)))
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
        # Subsample along time to keep this cheap.
        step = max(1, array.shape[1] // 400_000)
        block = np.asarray(array[:, ::step], dtype=np.float32)
        for c, channel in enumerate(CONFIG.canonical_channel_names):
            if not mask[c]:
                continue
            rows.append({
                "recording_id": row.recording_id,
                "subject": row.subject,
                "split": row.split,
                "channel": channel,
                "mean": float(block[c].mean()),
                "std": float(block[c].std()),
                "p99_abs": float(np.percentile(np.abs(block[c]), 99)),
            })

        # PSDs: last chunk before a positive decision versus a negative one.
        for pool, source in ((psd_pre, positives), (psd_inter, negatives)):
            subset = source[source.recording_id == row.recording_id]
            if subset.empty:
                continue
            pick = subset.iloc[int(rng.integers(len(subset)))]
            stop = int(pick["decision_end_sample"])
            start = stop - history_samples
            if start < 0 or stop > array.shape[1]:
                continue
            # Final 5 minutes of the history — closest to the decision.
            seg = np.asarray(array[:, max(start, stop - 5 * 60 * 256):stop], dtype=np.float64)
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


def figure_amplitude(stats: pd.DataFrame, path: Path) -> None:
    """Residual per-patient amplitude spread after the single global z-score."""
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    axis = axes[0]
    for i, channel in enumerate(CONFIG.canonical_channel_names):
        sub = stats[stats.channel == channel]
        axis.hist(sub["std"], bins=50, alpha=0.65,
                  color=[BLUE, ORANGE, AQUA][i], label=channel)
    axis.axvline(1.0, color=INK, linestyle="--", linewidth=1.2)
    axis.annotate("global target σ = 1", xy=(0.4, 0.88), xycoords="axes fraction",
                  fontsize=9, color=INK)
    axis.set(title="Per-recording σ after global z-score",
             xlabel="Standard deviation", ylabel="Recordings", xscale="log")
    axis.legend(frameon=False, fontsize=9)
    style(axis)

    axis = axes[1]
    by_patient = stats.groupby("subject")["std"].median().sort_values()
    axis.bar(np.arange(len(by_patient)), by_patient.values, color=BLUE, width=1.0)
    axis.axhline(1.0, color=ORANGE, linestyle="--", linewidth=1.4)
    ratio = by_patient.max() / max(by_patient.min(), 1e-9)
    axis.set(title=f"Median σ per patient — spans {ratio:.0f}x",
             xlabel="Patients, sorted", ylabel="Median σ", yscale="log")
    style(axis)

    axis = axes[2]
    for i, split in enumerate(["train", "validation", "test"]):
        sub = stats[stats.split == split]
        if sub.empty:
            continue
        axis.hist(sub["p99_abs"], bins=40, alpha=0.6,
                  color=SPLIT_COLOR[split], label=split, density=True)
    axis.set(title="99th-percentile |amplitude| by split",
             xlabel="z-scored amplitude", ylabel="Density", xscale="log")
    axis.legend(frameon=False, fontsize=9)
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
    for data, name, color in ((psd["inter"], "interictal (negative)", BLUE),
                              (psd["pre"], "pre-ictal (positive)", ORANGE)):
        median = np.median(data, axis=0)
        q1, q3 = np.percentile(data, [25, 75], axis=0)
        axis.plot(freqs[band], median[band], color=color, linewidth=2, label=name)
        axis.fill_between(freqs[band], q1[band], q3[band], color=color, alpha=0.18)
    axis.set(title="Welch PSD, last 5 min of history (median, IQR)",
             xlabel="Hz", ylabel="Power / Hz", yscale="log")
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
             xticklabels=[n.split()[0] for n in names], ylabel="Power", yscale="log")
    axis.legend(frameon=False, fontsize=9)
    style(axis)

    save(figure, path,
         "Spectra — where a pre-ictal signature would have to live")


def main() -> None:
    arguments = parse_arguments()
    CONFIG.validate()
    rng = np.random.default_rng(arguments.seed)
    output = arguments.output_dir
    output.mkdir(parents=True, exist_ok=True)

    log("Loading manifests...")
    decisions, seizures, shards = load_manifests()
    log(f"  decisions {len(decisions):,} | seizures {len(seizures)} | shards {len(shards)}")

    log("Figures...")
    figure_cohort(decisions, shards, output / "01_cohort.png")
    figure_labels(decisions, output / "02_labels.png")
    figure_seizures(seizures, decisions, output / "03_seizures.png")
    figure_signals(decisions, shards, output / "04_signals.png", rng)
    figure_chunks(decisions, shards, output / "05_chunks.png", rng)

    log(f"Sampling {arguments.recording_sample} recordings for signal statistics...")
    stats, psd = sample_recording_stats(shards, decisions, arguments.recording_sample, rng)
    log(f"  {len(stats)} channel-recording rows | {len(psd['pre'])} pre-ictal PSDs")
    figure_amplitude(stats, output / "06_amplitude.png")
    figure_spectra(psd, output / "07_spectra.png")

    stats.to_csv(output / "recording_statistics.csv", index=False)

    # A compact text digest next to the figures.
    summary = {
        "patients": int(decisions["subject"].nunique()),
        "recordings": int(shards["recording_id"].nunique()),
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

    log(f"\nFigures and summary.json in {output}")


if __name__ == "__main__":
    main()
