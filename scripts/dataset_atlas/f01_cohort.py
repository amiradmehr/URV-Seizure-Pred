"""Who is in the dataset, and how unevenly the EEG is spread across them."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from . import common as C
from .context import Atlas

NUMBER = "01"
SLUG = "cohort"
TITLE = "Cohort"
QUESTION = "Who is in the dataset, and how much EEG does each patient contribute?"
READS = (
    "Seven views of the cohort, all measured from the shard manifest plus the header of "
    "every stored .npy. The top row puts one dot per patient against subject ID, on a log "
    "axis, for EDF files, EEG hours and decisions; the tinted bands behind them are the "
    "three whole-patient splits, so the holdout is visible as a cut in subject ID rather "
    "than described in a caption, and a ring marks a patient who never contributes a "
    "positive decision. Below: the split totals with decisions, positives, patients and "
    "hours written on the bars; the distribution of file lengths against the 55-minute "
    "floor a file has to clear before it can yield a single decision; which two of the "
    "three canonical electrodes each file physically holds; and a Lorenz curve of how "
    "concentrated the EEG hours are across patients."
)
TAKE = (
    "The cohort is lopsided in ways that matter for how any result should be read. 121 of "
    "the 125 configured subjects have a usable file at all, and the median patient gives "
    "9 files and 93 h while subject 078 alone gives 397 h, so the top 10 per cent of "
    "patients hold 22 per cent of all EEG and half of the 11,367 h comes from 41 people. "
    "Any metric pooled over decisions is therefore mostly a statement about a few long "
    "recordings. The splits are thin where it counts: validation and test are 12 patients "
    "each with 330 and 299 positive decisions, and 20 of the 121 patients contribute no "
    "positive at all, which is why held-out numbers in this project move by several points "
    "between runs and why the near-chance baseline is hard to argue with in either "
    "direction. The montage is a second confound sitting in plain sight. No file has all "
    "three canonical channels, and the pair varies file by file: CROSS_HEAD is present in "
    "83 per cent of files, BTE_LEFT in 67 per cent, BTE_RIGHT in only 50 per cent. The "
    "availability mask that records this is fed to the model, so it can be read as a "
    "patient fingerprint instead of as EEG."
)

SPLIT_ORDER = ("train", "validation", "test")
# The top row keeps a narrow gutter to the right of subject 125 so the median
# line can be labelled without a label landing on top of a patient.
ID_LIMIT = (0.5, 133.0)
GUTTER_X = 126.5


def _shard_hours(shards: pd.DataFrame) -> np.ndarray:
    """Duration in hours of every stored recording, from the .npy headers only."""
    hours = np.empty(len(shards), dtype=np.float64)
    for index, path in enumerate(shards["X_path"]):
        array = np.load(path, mmap_mode="r")
        hours[index] = array.shape[1] / C.CONFIG.target_sfreq / 3600.0
    return hours


def _availability_pairs(shards: pd.DataFrame) -> list[tuple[tuple[bool, ...], int]]:
    """Count the electrode combinations actually stored, commonest first."""
    counts: dict[tuple[bool, ...], int] = {}
    for path in shards["channel_availability_path"]:
        with open(path, encoding="utf-8") as handle:
            key = tuple(bool(value) for value in json.load(handle))
        counts[key] = counts.get(key, 0) + 1
    return sorted(counts.items(), key=lambda item: -item[1])


def _split_bands(axis: plt.Axes, label: bool = False) -> None:
    """Tint the subject-ID axis by split, so the holdout is a visible cut."""
    edges = {
        "train": (0.5, 100.5),
        "validation": (100.5, 112.5),
        "test": (112.5, 125.5),
    }
    # The three label boxes are wider than the two narrow bands, so they are
    # staggered vertically rather than allowed to collide.
    label_y = {"train": 0.975, "validation": 0.975, "test": 0.855}
    for split, (low, high) in edges.items():
        axis.axvspan(low, high, facecolor=C.SPLIT_COLOR[split], alpha=0.06, zorder=0)
    for boundary in (100.5, 112.5):
        axis.axvline(boundary, color=C.MUTED, linewidth=0.9, linestyle=(0, (4, 3)), zorder=1)
    if label:
        for split, (low, high) in edges.items():
            axis.annotate(
                f"{split}\n{int(low + 0.5):03d}-{int(high - 0.5):03d}",
                xy=((low + high) / 2, label_y[split]),
                xycoords=("data", "axes fraction"),
                ha="center",
                va="top",
                fontsize=7.8,
                color=C.SPLIT_COLOR[split],
                fontweight="bold",
                zorder=6,
                bbox=dict(
                    boxstyle="round,pad=0.18",
                    facecolor=C.SURFACE,
                    edgecolor=C.SPLIT_COLOR[split],
                    linewidth=0.6,
                    alpha=0.94,
                ),
            )
    axis.set_xlim(*ID_LIMIT)


def _per_patient_scatter(
    axis: plt.Axes, values: pd.Series, splits: pd.Series, size: float = 20.0
) -> None:
    """One dot per patient at its subject ID, coloured by split."""
    for split in SPLIT_ORDER:
        keep = splits == split
        if not keep.any():
            continue
        axis.scatter(
            values.index[keep].astype(int),
            values[keep].to_numpy(),
            s=size,
            color=C.SPLIT_COLOR[split],
            alpha=0.88,
            linewidths=0.0,
            zorder=4,
        )


def _median_line(axis: plt.Axes, value: float, text: str) -> None:
    """A dotted median with its label parked in the right-hand gutter."""
    axis.axhline(value, color=C.INK_2, linewidth=0.9, linestyle=(0, (1, 2)), zorder=3)
    axis.annotate(
        text,
        xy=(GUTTER_X, value),
        ha="left",
        va="center",
        fontsize=7.6,
        color=C.INK_2,
        zorder=7,
    )


def _montage_chip(axis: plt.Axes, mask: tuple[bool, ...], row: float, unit: float) -> None:
    """Three little squares showing which canonical channels this file holds."""
    for position, present in enumerate(mask):
        name = C.CONFIG.canonical_channel_names[position]
        axis.add_patch(
            Rectangle(
                (unit * (0.6 + 1.3 * position), row - 0.15),
                unit,
                0.30,
                facecolor=C.CHANNEL_COLOR[name] if present else C.ABSENT_COLOR,
                edgecolor=C.INK_2 if not present else "none",
                hatch=None if present else "xxx",
                linewidth=0.7,
                zorder=5,
            )
        )


def render(atlas: Atlas) -> Path | None:
    shards = atlas.manifests.shards.copy()
    decisions = atlas.manifests.decisions
    if shards.empty:
        return None
    shards["hours"] = _shard_hours(shards)

    files_pp = shards.groupby("subject").size()
    hours_pp = shards.groupby("subject")["hours"].sum()
    decisions_pp = decisions.groupby("subject").size()
    split_pp = shards.groupby("subject")["split"].first()
    positives_pp = decisions[decisions["label"] == 1].groupby("subject").size()
    zero_positive = [s for s in hours_pp.index if int(positives_pp.get(s, 0)) == 0]

    total_hours = float(hours_pp.sum())
    n_patients = int(len(hours_pp))
    missing = sorted(set(C.CONFIG.included_subjects) - set(hours_pp.index))
    raw_files = len(list(C.CONFIG.raw_data_dir.glob("sub-*/ses-*/eeg/*_eeg.edf")))

    figure = plt.figure(figsize=(18, 10.4))
    grid = figure.add_gridspec(
        2,
        3,
        height_ratios=[1.0, 1.22],
        hspace=0.34,
        wspace=0.235,
        top=0.900,
        bottom=0.068,
        left=0.052,
        right=0.985,
    )

    # ---- 1. EDF files per patient --------------------------------------
    axis = figure.add_subplot(grid[0, 0])
    _split_bands(axis, label=True)
    _per_patient_scatter(axis, files_pp, split_pp)
    axis.set(yscale="log", ylim=(0.45, 900.0))
    axis.set(
        xlabel="Subject ID  (001-125, one dot per patient)",
        ylabel="EDF files per patient (count, log)",
    )
    axis.set_title(f"1  EDF files per patient   ·   {len(shards):,} files in total", loc="left")
    _median_line(axis, float(files_pp.median()), f"median\n{files_pp.median():.0f}")
    axis.annotate(
        f"{n_patients} of the {len(C.CONFIG.included_subjects)} configured subjects have a\n"
        f"usable file; sub-{', sub-'.join(missing)} have none",
        xy=(0.015, 0.78),
        xycoords="axes fraction",
        ha="left",
        va="top",
        fontsize=7.8,
        color=C.INK_2,
        bbox=dict(boxstyle="round,pad=0.24", facecolor=C.SURFACE, edgecolor=C.GRID,
                  linewidth=0.7, alpha=0.93),
    )
    C.style(axis)

    # ---- 2. EEG hours per patient --------------------------------------
    axis = figure.add_subplot(grid[0, 1])
    _split_bands(axis)
    _per_patient_scatter(axis, hours_pp, split_pp)
    axis.set(yscale="log", ylim=(0.6, 2600.0))
    axis.set(
        xlabel="Subject ID  (001-125, one dot per patient)",
        ylabel="EEG recorded per patient (h, log)",
    )
    axis.set_title(
        f"2  EEG hours per patient   ·   {total_hours:,.0f} h in total", loc="left"
    )
    _median_line(axis, float(hours_pp.median()), f"median\n{hours_pp.median():.0f} h")
    top_subject = str(hours_pp.idxmax())
    axis.annotate(
        f"sub-{top_subject}: {hours_pp.max():.0f} h\n"
        f"{hours_pp.max() / total_hours * 100:.1f} % of all the EEG, from one patient",
        xy=(int(top_subject), float(hours_pp.max())),
        xytext=(-6, 30),
        textcoords="offset points",
        ha="center",
        fontsize=7.8,
        color=C.INK_2,
        arrowprops=dict(arrowstyle="-", color=C.INK_2, linewidth=0.8),
        bbox=dict(boxstyle="round,pad=0.24", facecolor=C.SURFACE, edgecolor=C.GRID,
                  linewidth=0.7, alpha=0.93),
    )
    C.style(axis)

    # ---- 3. decisions per patient --------------------------------------
    axis = figure.add_subplot(grid[0, 2])
    _split_bands(axis)
    _per_patient_scatter(axis, decisions_pp, split_pp.reindex(decisions_pp.index))
    axis.scatter(
        [int(s) for s in zero_positive],
        [float(decisions_pp.get(s, np.nan)) for s in zero_positive],
        s=62,
        facecolors="none",
        edgecolors=C.INK_2,
        linewidths=0.9,
        zorder=5,
    )
    axis.set(yscale="log", ylim=(3.0, 60_000.0))
    axis.set(
        xlabel="Subject ID  (001-125, one dot per patient)",
        ylabel="Decisions per patient (count, log)",
    )
    axis.set_title(
        f"3  Decisions per patient   ·   {len(decisions):,} in total, one per 60 s", loc="left"
    )
    _median_line(axis, float(decisions_pp.median()), f"median\n{decisions_pp.median():,.0f}")
    axis.annotate(
        f"ring = patient never contributes a positive\n"
        f"decision ({len(zero_positive)} of {n_patients} patients)",
        xy=(0.015, 0.155),
        xycoords="axes fraction",
        ha="left",
        va="top",
        fontsize=7.8,
        color=C.INK_2,
        bbox=dict(boxstyle="round,pad=0.24", facecolor=C.SURFACE, edgecolor=C.GRID,
                  linewidth=0.7, alpha=0.93),
    )
    C.style(axis)

    # ---- 4. split totals ------------------------------------------------
    axis = figure.add_subplot(grid[1, 0])
    totals = (
        decisions.groupby("split")
        .agg(n=("label", "size"), positive=("label", "sum"), patients=("subject", "nunique"))
        .reindex(SPLIT_ORDER)
    )
    hours_split = shards.groupby("split")["hours"].sum().reindex(SPLIT_ORDER)
    files_split = shards.groupby("split").size().reindex(SPLIT_ORDER)
    positions = np.arange(len(SPLIT_ORDER))
    axis.bar(
        positions,
        totals["n"].to_numpy(),
        color=[C.SPLIT_COLOR[s] for s in SPLIT_ORDER],
        width=0.62,
        zorder=3,
    )
    for x, split in enumerate(SPLIT_ORDER):
        row = totals.loc[split]
        axis.annotate(
            f"{int(row['n']):,} decisions\n{int(row['positive']):,} positive "
            f"({row['positive'] / row['n'] * 100:.2f} %)\n"
            f"{int(row['patients'])} patients · {int(files_split[split]):,} files\n"
            f"{hours_split[split]:,.0f} h EEG",
            xy=(x, float(row["n"])),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.2,
            color=C.INK,
        )
    axis.set(
        xticks=positions,
        ylim=(0, float(totals["n"].max()) * 1.55),
        xlabel="Split  (whole-patient holdout, by subject-ID index)",
        ylabel="Decisions (count)",
    )
    axis.set_xticklabels(
        [f"{s}\n{'001-100' if s == 'train' else '101-112' if s == 'validation' else '113-125'}"
         for s in SPLIT_ORDER]
    )
    axis.yaxis.set_major_formatter(lambda value, _: f"{int(value):,}")
    axis.set_title("4  What each split actually contains", loc="left")
    C.style(axis, grid="y")

    # ---- 5. recording-duration histogram --------------------------------
    axis = figure.add_subplot(grid[1, 1])
    minimum_seconds = C.CONFIG.input_window_seconds + 60.0 * (
        C.CONFIG.prediction_horizon_minutes + C.CONFIG.seizure_occurrence_period_minutes
    )
    minimum_hours = minimum_seconds / 3600.0
    longest = float(shards["hours"].max())
    axis.hist(
        shards["hours"].to_numpy(),
        bins=np.arange(0.0, longest + 0.5, 0.5),
        color=C.BLUE,
        alpha=0.9,
        zorder=3,
    )
    axis.axvline(minimum_hours, color=C.ORANGE, linestyle="--", linewidth=1.6, zorder=5)
    axis.annotate(
        f"{minimum_seconds / 60:.0f} min floor = 45 min history + 10 min occurrence\n"
        f"period; a shorter EDF yields no decision at all. The\n"
        f"shortest stored file is {shards['hours'].min() * 60:.1f} min, right on the line.",
        xy=(0.115, 0.985),
        xycoords="axes fraction",
        ha="left",
        va="top",
        fontsize=8.0,
        color=C.ORANGE,
        bbox=dict(boxstyle="round,pad=0.26", facecolor=C.SURFACE, edgecolor=C.ORANGE,
                  linewidth=0.7, alpha=0.95),
    )
    median_hours = float(shards["hours"].median())
    axis.axvline(median_hours, color=C.INK_2, linestyle=(0, (1, 2)), linewidth=1.1, zorder=5)
    axis.annotate(
        f"median file {median_hours:.2f} h",
        xy=(median_hours, 0.03),
        xycoords=("data", "axes fraction"),
        xytext=(4, 0),
        textcoords="offset points",
        rotation=90,
        ha="left",
        va="bottom",
        fontsize=7.8,
        color=C.INK_2,
        zorder=8,
        bbox=dict(boxstyle="round,pad=0.16", facecolor=C.SURFACE, edgecolor="none", alpha=0.85),
    )
    axis.set(
        yscale="log",
        xlabel="Length of one EDF file (h)",
        ylabel="EDF files (count, log)",
        xlim=(0, longest * 1.03),
        ylim=(0.7, 2.0e5),
    )
    axis.set_title(
        f"5  File length   ·   {len(shards):,} of the {raw_files:,} EDF files "
        f"yield a decision", loc="left"
    )
    axis.annotate(
        f"{raw_files - len(shards):,} of the {raw_files:,} EDF files in the download produced no\n"
        f"decision and have no shard, so they are absent here.\n"
        f"Longest stored file {longest:.1f} h.",
        xy=(0.115, 0.855),
        xycoords="axes fraction",
        ha="left",
        va="top",
        fontsize=7.8,
        color=C.INK_2,
        bbox=dict(boxstyle="round,pad=0.26", facecolor=C.SURFACE, edgecolor=C.GRID,
                  linewidth=0.7, alpha=0.93),
    )
    C.style(axis)

    # ---- 6 + 7. montage and concentration --------------------------------
    lower = grid[1, 2].subgridspec(2, 1, height_ratios=[1.0, 1.12], hspace=1.12)

    axis = figure.add_subplot(lower[0])
    pairs = _availability_pairs(shards)
    biggest = float(pairs[0][1])
    unit = biggest * 0.034
    for row, (mask, count) in enumerate(pairs):
        axis.barh(
            row,
            count,
            height=0.56,
            facecolor=C.MUTED,
            alpha=0.26,
            edgecolor=C.INK_2,
            linewidth=0.7,
            zorder=3,
        )
        _montage_chip(axis, mask, float(row), unit)
        axis.annotate(
            f"{count:,} files   {count / len(shards) * 100:.1f} %",
            xy=(count, row),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=8.2,
            color=C.INK,
        )
    present_rate = {
        name: sum(count for mask, count in pairs if mask[index]) / len(shards) * 100
        for index, name in enumerate(C.CONFIG.canonical_channel_names)
    }
    axis.set(
        yticks=np.arange(len(pairs)),
        yticklabels=[
            " + ".join(
                n for n, p in zip(C.CONFIG.canonical_channel_names, mask, strict=True) if p
            )
            for mask, _ in pairs
        ],
        xlim=(0, biggest * 1.36),
        xlabel="EDF files (count)",
    )
    axis.tick_params(axis="y", labelsize=8.0)
    axis.set_ylim(len(pairs) + 0.55, -0.62)
    axis.set_title("6  Which two electrodes the file actually holds", loc="left")
    axis.annotate(
        "each channel is present in:  "
        + "  ·  ".join(
            f"{n} {v:.0f} %"
            for n, v in sorted(present_rate.items(), key=lambda kv: -kv[1])
        ),
        xy=(unit * 0.6, len(pairs) + 0.05),
        ha="left",
        va="center",
        fontsize=7.6,
        color=C.INK_2,
    )
    axis.annotate(
        "chip = the 3 canonical channels in order (BTE_LEFT, BTE_RIGHT, CROSS_HEAD);\n"
        "the crossed grey square is the one zero-filled and flagged absent.",
        xy=(0.0, -0.60),
        xycoords="axes fraction",
        ha="left",
        va="top",
        fontsize=7.5,
        color=C.INK_2,
    )
    C.style(axis, grid="x")

    axis = figure.add_subplot(lower[1])
    sorted_hours = np.sort(hours_pp.to_numpy())
    cumulative = np.concatenate([[0.0], np.cumsum(sorted_hours)]) / total_hours * 100.0
    patient_axis = np.arange(len(sorted_hours) + 1) / len(sorted_hours) * 100.0
    gini = 1.0 - 2.0 * np.trapezoid(cumulative / 100.0, patient_axis / 100.0)
    top_k = int(np.ceil(0.10 * len(sorted_hours)))
    top_share = float(sorted_hours[-top_k:].sum() / total_hours * 100.0)
    half_patients = int(np.searchsorted(np.cumsum(sorted_hours[::-1]) / total_hours, 0.5) + 1)

    axis.plot([0, 100], [0, 100], color=C.MUTED, linewidth=1.0, linestyle=(0, (4, 3)), zorder=3)
    axis.plot(patient_axis, cumulative, color=C.BLUE, linewidth=1.8, zorder=5)
    axis.fill_between(patient_axis, cumulative, patient_axis, color=C.BLUE, alpha=0.13, zorder=2)
    axis.axvspan(100.0 - top_k / len(sorted_hours) * 100.0, 100.0,
                 facecolor=C.MUTED, alpha=0.16, hatch="///", edgecolor=C.MUTED, zorder=1)
    axis.annotate(
        f"top 10 % of patients\n({top_k} of {n_patients}) hold\n{top_share:.0f} % of all EEG hours",
        xy=(88.0, 4.0),
        xytext=(-6, 0),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=7.8,
        color=C.INK,
        bbox=dict(boxstyle="round,pad=0.24", facecolor=C.SURFACE, edgecolor=C.GRID,
                  linewidth=0.7, alpha=0.93),
    )
    axis.annotate(
        "equal contribution",
        xy=(34.0, 34.0),
        xytext=(-3, 5),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=7.6,
        color=C.MUTED,
        rotation=38,
        rotation_mode="anchor",
    )
    axis.annotate(
        f"Gini = {gini:.2f}",
        xy=(52.0, 34.0),
        ha="center",
        va="center",
        fontsize=7.8,
        color=C.BLUE,
        fontweight="bold",
    )
    axis.annotate(
        f"half of all {total_hours:,.0f} h comes from {half_patients} of the "
        f"{n_patients} patients",
        xy=(0.025, 0.97),
        xycoords="axes fraction",
        ha="left",
        va="top",
        fontsize=7.8,
        color=C.INK_2,
    )
    axis.set(
        xlim=(0, 100),
        ylim=(0, 100),
        xticks=[0, 20, 40, 60, 80, 100],
        yticks=[0, 25, 50, 75, 100],
        xlabel="Patients ranked by EEG contributed, lowest first (cumulative %)",
        ylabel="EEG hours (cumulative %)",
    )
    axis.set_title("7  How concentrated the hours are", loc="left")
    C.style(axis)

    atlas.fact("cohort_patients_with_data", n_patients)
    atlas.fact("cohort_eeg_hours", round(total_hours, 1))
    atlas.fact("cohort_median_hours_per_patient", round(float(hours_pp.median()), 1))
    atlas.fact("cohort_top10pct_hour_share", round(top_share, 1))
    atlas.fact("cohort_hours_gini", round(float(gini), 3))
    atlas.fact("cohort_patients_for_half_the_hours", half_patients)
    atlas.fact("cohort_median_file_hours", round(median_hours, 2))
    atlas.fact("cohort_files_without_decisions", int(raw_files - len(shards)))
    atlas.fact("cohort_channel_present_percent", {k: round(v, 1) for k, v in present_rate.items()})
    atlas.fact("cohort_patients_without_positive", len(zero_positive))

    return C.save(
        figure,
        atlas.path(NUMBER, SLUG),
        f"Cohort — {n_patients} patients with usable EEG, {total_hours:,.0f} h, "
        f"{len(decisions):,} decisions, {int((decisions['label'] == 1).sum()):,} of them positive",
        "One dot per patient. The tinted bands are the whole-patient splits: subjects 001-100 "
        "train, 101-112 validation, 113-125 test, and no subject appears in two of them.",
        tight=False,
    )
