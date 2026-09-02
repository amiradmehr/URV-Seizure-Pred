"""Every stored recording in the dataset, drawn to scale in one picture."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from . import common as C
from .context import Atlas

NUMBER = "02"
SLUG = "dataset_mosaic"
TITLE = "The whole dataset at once"
QUESTION = "What do eleven thousand hours of behind-the-ear EEG actually look like?"
READS = (
    "The mosaic gives every patient one row, sorted by how much EEG they contributed, "
    "longest at the top. Along a row the patient's EDF files are laid end to end and "
    "drawn to scale, each bar coloured by the split that patient belongs to and separated "
    "from the next by a thin gap, so x is cumulative recorded hours for that patient and "
    "x = 0 is the start of their first file. Above each row every annotated seizure is a "
    "violet tick at its cumulative-hours position: full height when the seizure passed the "
    "eligibility rules and became a prediction target, half height when it did not. One "
    "row is magnified below right, where the file boundaries and the two tick heights are "
    "large enough to name directly. Bottom left, the same hours as a concentration curve "
    "against patient rank."
)
TAKE = (
    "The corpus is not 121 comparable subjects, it is a handful of long stays plus a long "
    "tail of short ones. The twelve largest patients (10 per cent of the cohort) hold "
    "21 per cent of the 11,367 hours, and the single largest holds 397 h against a median "
    "of 93 h and a minimum of 1 h. Eligible seizures are more concentrated still: the top "
    "twelve patients by target count hold 34 per cent of the 317 targets, and 20 of the "
    "121 patients supply none at all, so a fifth of the cohort contributes only negatives. "
    "That matters for the held-out numbers, because validation is 12 patients and 1,070 h "
    "and test is 12 patients and 1,305 h; a per-patient result on those splits is an "
    "average over very few people, which is one reason the baseline's held-out score is "
    "hard to separate from chance. The ticks also show the labelling loss: only 713 of the "
    "883 annotated seizures sit in a file that produced any decision at all, and of those "
    "only 317 have the 60 clear minutes before onset that make them usable."
)

# Row geometry, in units of the one-row pitch: the bar, then the seizure tick
# rising out of its top edge at one of two heights.
BAR_HALF = 0.20
TICK_FULL = 0.52
TICK_HALF = 0.26


# ----------------------------------------------------------------------
# Geometry, measured off the stored shards
# ----------------------------------------------------------------------


def _header_samples(path: str) -> int:
    """Sample count of one shard from its .npy header alone (no data read)."""
    readers = {
        (1, 0): np.lib.format.read_array_header_1_0,
        (2, 0): np.lib.format.read_array_header_2_0,
    }
    with open(path, "rb") as handle:
        version = np.lib.format.read_magic(handle)
        reader = readers.get(version)
        if reader is None:
            return int(np.load(path, mmap_mode="r").shape[1])
        shape = reader(handle)[0]
    return int(shape[1])


def _layout(atlas: Atlas) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-file cumulative offsets within a patient, and the per-patient totals."""
    shards = atlas.manifests.shards.copy()
    shards["hours"] = [
        _header_samples(path) / C.CONFIG.target_sfreq / 3600.0 for path in shards["X_path"]
    ]
    shards = shards.sort_values(["subject", "recording_id"]).reset_index(drop=True)
    shards["start_hours"] = shards.groupby("subject")["hours"].cumsum() - shards["hours"]

    patients = shards.groupby("subject").agg(
        hours=("hours", "sum"),
        files=("recording_id", "size"),
        split=("split", "first"),
    )
    patients = patients.sort_values("hours", ascending=False)
    patients["rank"] = np.arange(len(patients), dtype=int)
    patients["y"] = -patients["rank"].to_numpy(dtype=float)
    return shards, patients


def _seizure_positions(atlas: Atlas, shards: pd.DataFrame, patients: pd.DataFrame) -> pd.DataFrame:
    """Every annotated seizure placed on the mosaic, where its file was stored."""
    seizures = atlas.manifests.seizures.copy()
    seizures["recording_id"] = (
        "sub-" + seizures["subject"]
        + "_ses-" + seizures["session"].astype(str).str.zfill(2)
        + "_task-" + C.CONFIG.bids_task
        + "_run-" + seizures["run"].astype(str).str.zfill(2)
    )
    offsets = shards.set_index("recording_id")["start_hours"]
    seizures["start_hours"] = seizures["recording_id"].map(offsets)
    placed = seizures.dropna(subset=["start_hours"]).copy()
    placed["x_hours"] = placed["start_hours"] + placed["onset_seconds"] / 3600.0
    placed["y"] = placed["subject"].map(patients["y"])
    return placed.dropna(subset=["y"])


def _tick_segments(x: np.ndarray, y: np.ndarray, length: float) -> list:
    """Vertical tick segments rising out of the top edge of each row's bar."""
    base = y + BAR_HALF
    return [[(xi, bi), (xi, bi + length)] for xi, bi in zip(x, base, strict=True)]


# ----------------------------------------------------------------------
# Panels
# ----------------------------------------------------------------------


def _draw_mosaic(
    axis: plt.Axes,
    figure: plt.Figure,
    shards: pd.DataFrame,
    patients: pd.DataFrame,
    placed: pd.DataFrame,
    n_annotated: int,
) -> dict[str, float]:
    total_hours = float(patients["hours"].sum())
    span = float(patients["hours"].max()) * 1.21
    axis.set_xlim(0.0, span)

    # A visible gap between consecutive files, sized in pixels, but never allowed
    # to eat a short file: a quarter of the shortest bars is the floor.
    width_pixels = figure.get_size_inches()[0] * axis.get_position().width * C.DPI
    gap = 2.0 * span / max(width_pixels, 1.0)

    for subject, row in patients.iterrows():
        mine = shards[shards["subject"] == subject]
        bars = [
            (float(start), max(float(hours) - min(gap, 0.22 * float(hours)), gap * 0.15))
            for start, hours in zip(mine["start_hours"], mine["hours"], strict=True)
        ]
        axis.broken_barh(
            bars,
            (float(row["y"]) - BAR_HALF, 2 * BAR_HALF),
            facecolors=C.SPLIT_COLOR[str(row["split"])],
            edgecolors="none",
            zorder=3,
        )

    eligible = placed[placed["eligible"]]
    ineligible = placed[~placed["eligible"]]
    for subset, length, width in (
        (ineligible, TICK_HALF, 0.75),
        (eligible, TICK_FULL, 1.1),
    ):
        if subset.empty:
            continue
        axis.add_collection(
            LineCollection(
                _tick_segments(
                    subset["x_hours"].to_numpy(float), subset["y"].to_numpy(float), length
                ),
                colors=C.VIOLET,
                linewidths=width,
                zorder=5,
            ),
            autolim=False,
        )

    n_top = max(1, int(round(0.10 * len(patients))))
    top_share = float(patients["hours"].head(n_top).sum()) / total_hours
    axis.axhline(
        float(patients["y"].iloc[n_top - 1]) - 0.5,
        color=C.INK_2,
        linestyle=(0, (5, 4)),
        linewidth=0.9,
        zorder=6,
    )
    axis.annotate(
        f"top {n_top} patients (10 % of the cohort) hold {100 * top_share:.0f} % of all the EEG",
        xy=(span * 0.50, float(patients["y"].iloc[n_top - 1]) - 0.5),
        xytext=(0, 4),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=8.8,
        color=C.INK_2,
        zorder=7,
    )

    middle = len(patients) // 2
    axis.annotate(
        f"median patient: {patients['hours'].iloc[middle]:.0f} h",
        xy=(float(patients["hours"].iloc[middle]), float(patients["y"].iloc[middle])),
        xytext=(26, 0),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=8.8,
        color=C.INK_2,
        arrowprops=dict(arrowstyle="-", color=C.INK_2, linewidth=0.7,
                        shrinkA=0.0, shrinkB=2.0),
        zorder=7,
    )

    # One direct in-plot label per split, hung just below and right of the end of
    # that split's longest row: every row below it is shorter, so the label covers
    # no bar.  Aqua alone is below the relief threshold on white, so it needs this.
    for split in ("train", "validation", "test"):
        rows = patients[patients["split"] == split]
        if rows.empty:
            continue
        first = rows.iloc[0]
        axis.annotate(
            f"{split}  ({len(rows)} patients, {rows['hours'].sum():,.0f} h)",
            xy=(float(first["hours"]), float(first["y"])),
            xytext=(6, -3),
            textcoords="offset points",
            ha="left",
            va="top",
            fontsize=8.6,
            color=C.SPLIT_COLOR[split],
            fontweight="bold",
            zorder=8,
            bbox=dict(boxstyle="round,pad=0.20", facecolor=C.SURFACE,
                      edgecolor="none", alpha=0.90),
        )

    ranks = sorted({0, n_top, len(patients) // 3, middle, 2 * len(patients) // 3, len(patients) - 1})
    axis.set_yticks([-r for r in ranks])
    axis.set_yticklabels(
        [f"sub-{patients.index[r]}   {patients['hours'].iloc[r]:.0f} h" for r in ranks]
    )
    axis.set_ylim(-(len(patients) - 1) - 0.85, 1.15)
    axis.set_xlabel(
        "Cumulative EEG per patient (h)   ·   x = 0 is the start of that patient's first file"
    )
    axis.set_ylabel(
        f"Patients (one row each, {len(patients)} rows)\n"
        "sorted by hours recorded, rank 0 at the top"
    )
    axis.set_title(
        f"{len(shards):,} EDF files   ·   {total_hours:,.0f} h of behind-the-ear EEG   ·   "
        f"{len(patients)} patients with stored data   ·   every bar drawn to scale",
        loc="left",
    )

    handles = [
        Rectangle((0, 0), 1, 1, facecolor=C.SPLIT_COLOR[s], edgecolor="none", label=f"{s} split")
        for s in ("train", "validation", "test")
    ]
    handles += [
        Line2D([], [], color=C.VIOLET, marker="|", markersize=11, markeredgewidth=1.2,
               linestyle="none", label=f"seizure, eligible target ({len(eligible)})"),
        Line2D([], [], color=C.VIOLET, marker="|", markersize=5.5, markeredgewidth=1.0,
               linestyle="none", label=f"seizure, not eligible ({len(ineligible)})"),
    ]
    legend = axis.legend(
        handles=handles,
        loc="center right",
        bbox_to_anchor=(0.995, 0.44),
        frameon=True,
        framealpha=0.95,
        edgecolor=C.GRID,
        fontsize=8.8,
        title="bar colour = split   ·   tick height = eligibility",
        title_fontsize=8.8,
    )
    legend.get_frame().set_linewidth(0.7)

    axis.annotate(
        "One bar = one EDF file, drawn to scale, files laid end to end in recording order.\n"
        "Violet ticks above a row are annotated seizures, placed at the hour they occur.\n"
        f"{len(placed)} of the {n_annotated} annotated seizures are drawn; the other "
        f"{n_annotated - len(placed)} sit in files that\nproduced no decision at all, "
        "so they have no bar to stand on.",
        xy=(0.995, 0.30),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=8.6,
        color=C.INK_2,
        bbox=dict(boxstyle="round,pad=0.35", facecolor=C.SURFACE, edgecolor=C.GRID,
                  linewidth=0.7, alpha=0.95),
        zorder=8,
    )
    C.style(axis, grid="x")
    return {
        "total_hours": total_hours,
        "top_share": top_share,
        "n_top": n_top,
        "gap_hours": gap,
    }


def _draw_concentration(
    axis: plt.Axes, patients: pd.DataFrame, placed: pd.DataFrame
) -> dict[str, float]:
    """A concentration curve: how much of the corpus the biggest patients hold."""
    note = dict(
        boxstyle="round,pad=0.24", facecolor=C.SURFACE, edgecolor="none", alpha=0.92
    )
    n = len(patients)
    rank_percent = np.concatenate([[0.0], (np.arange(n) + 1) / n * 100.0])

    hours = patients["hours"].to_numpy(float)
    hours_share = np.concatenate([[0.0], np.cumsum(hours) / hours.sum() * 100.0])

    targets = (
        placed[placed["eligible"]].groupby("subject").size().reindex(patients.index).fillna(0.0)
    )
    targets = np.sort(targets.to_numpy(float))[::-1]
    target_share = np.concatenate([[0.0], np.cumsum(targets) / max(targets.sum(), 1.0) * 100.0])

    axis.plot([0, 100], [0, 100], color=C.MUTED, linestyle=(0, (4, 3)), linewidth=1.0)
    axis.annotate(
        "if every patient\ncontributed equally",
        xy=(36, 36),
        xytext=(7, -7),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=8.4,
        color=C.MUTED,
        bbox=note,
        zorder=6,
    )
    axis.plot(rank_percent, hours_share, color=C.INK, linewidth=1.8, zorder=4)
    axis.plot(rank_percent, target_share, color=C.VIOLET, linewidth=1.8, zorder=4)

    n_top = max(1, int(round(0.10 * n)))
    top_hours = hours_share[n_top]
    top_targets = target_share[n_top]
    half = int(round(n / 2))
    for x, y, colour in (
        (rank_percent[n_top], top_hours, C.INK),
        (rank_percent[n_top], top_targets, C.VIOLET),
        (rank_percent[half], hours_share[half], C.INK),
    ):
        axis.plot([x], [y], marker="o", markersize=4.5, color=colour, zorder=5)

    axis.annotate(
        f"top {n_top} patients: {top_hours:.0f} % of the hours",
        xy=(rank_percent[n_top], top_hours),
        xytext=(10, -18),
        textcoords="offset points",
        fontsize=8.6,
        color=C.INK,
        bbox=note,
        zorder=6,
        arrowprops=dict(arrowstyle="-", color=C.INK, linewidth=0.7, shrinkA=0, shrinkB=3),
    )
    axis.annotate(
        f"and {top_targets:.0f} % of the {int(targets.sum())} eligible targets",
        xy=(rank_percent[n_top], top_targets),
        xytext=(12, 12),
        textcoords="offset points",
        fontsize=8.6,
        color=C.VIOLET,
        bbox=note,
        zorder=6,
        arrowprops=dict(arrowstyle="-", color=C.VIOLET, linewidth=0.7, shrinkA=0, shrinkB=3),
    )
    axis.annotate(
        f"half the patients: {hours_share[half]:.0f} % of the hours",
        xy=(rank_percent[half], hours_share[half]),
        xytext=(14, -30),
        textcoords="offset points",
        fontsize=8.6,
        color=C.INK,
        bbox=note,
        zorder=6,
        arrowprops=dict(arrowstyle="-", color=C.INK, linewidth=0.7, shrinkA=0, shrinkB=3),
    )

    empty = int((targets == 0).sum())
    if empty:
        axis.annotate(
            f"the violet curve is flat over the last fifth of the cohort:\n"
            f"{empty} of the {n} patients have no eligible seizure at all",
            xy=(99, 25),
            xycoords="data",
            ha="right",
            va="top",
            fontsize=8.4,
            color=C.VIOLET,
            bbox=note,
            zorder=6,
        )

    axis.annotate(
        "cumulative share\nof EEG hours",
        xy=(74, np.interp(74, rank_percent, hours_share)),
        xytext=(99, 45),
        textcoords="data",
        ha="right",
        va="top",
        fontsize=8.8,
        color=C.INK,
        fontweight="bold",
        bbox=note,
        zorder=6,
        arrowprops=dict(arrowstyle="-", color=C.INK, linewidth=0.7, shrinkA=2, shrinkB=3),
    )
    axis.annotate(
        "cumulative share of\neligible seizures",
        xy=(24, np.interp(24, rank_percent, target_share)),
        xytext=(1.5, 103),
        textcoords="data",
        ha="left",
        va="top",
        fontsize=8.8,
        color=C.VIOLET,
        fontweight="bold",
        bbox=note,
        zorder=6,
        arrowprops=dict(arrowstyle="-", color=C.VIOLET, linewidth=0.7, shrinkA=2, shrinkB=3),
    )

    axis.set(
        xlim=(0, 100),
        ylim=(0, 106),
        xlabel="Patients ranked by their own contribution, largest first (% of patients)",
        ylabel="Cumulative share of the total (%)",
    )
    axis.set_title("How unevenly the corpus is spread across patients", loc="left")
    C.style(axis)
    return {
        "top_hours_share": float(top_hours),
        "top_target_share": float(top_targets),
        "half_hours_share": float(hours_share[half]),
        "patients_without_target": empty,
    }


def _pick_zoom_subject(shards: pd.DataFrame, patients: pd.DataFrame, placed: pd.DataFrame) -> str | None:
    """A mid-sized patient whose row shows files and both tick heights."""
    counts = placed.groupby("subject")["eligible"].agg(["sum", "size"])
    table = patients.join(counts).fillna(0.0)
    table = table[
        (table["files"] >= 5)
        & (table["files"] <= 12)
        & (table["hours"] >= 30)
        & (table["hours"] <= 100)
        & (table["sum"] >= 2)
        & (table["size"] - table["sum"] >= 2)
    ]
    if table.empty:
        return None
    table = table.assign(
        imbalance=(2 * table["sum"] - table["size"]).abs(),
        distance=(table["size"] - 10).abs(),
    )
    table = table.sort_values(
        ["imbalance", "distance", "files"], ascending=[True, True, False]
    )
    return str(table.index[0])


def _draw_zoom(
    axis: plt.Axes,
    figure: plt.Figure,
    subject: str,
    shards: pd.DataFrame,
    patients: pd.DataFrame,
    placed: pd.DataFrame,
) -> None:
    mine = shards[shards["subject"] == subject].reset_index(drop=True)
    row = patients.loc[subject]
    span = float(row["hours"]) * 1.02
    axis.set_xlim(0.0, span)
    width_pixels = figure.get_size_inches()[0] * axis.get_position().width * C.DPI
    gap = 2.0 * span / max(width_pixels, 1.0)

    axis.broken_barh(
        [
            (float(s), max(float(h) - min(gap, 0.22 * float(h)), gap * 0.15))
            for s, h in zip(mine["start_hours"], mine["hours"], strict=True)
        ],
        (-BAR_HALF, 2 * BAR_HALF),
        facecolors=C.SPLIT_COLOR[str(row["split"])],
        edgecolors="none",
        zorder=3,
    )

    ticks = placed[placed["subject"] == subject]
    for subset, length, width in (
        (ticks[~ticks["eligible"]], TICK_HALF, 1.1),
        (ticks[ticks["eligible"]], TICK_FULL, 1.5),
    ):
        if subset.empty:
            continue
        axis.add_collection(
            LineCollection(
                _tick_segments(
                    subset["x_hours"].to_numpy(float), np.zeros(len(subset)), length
                ),
                colors=C.VIOLET,
                linewidths=width,
                zorder=5,
            ),
            autolim=False,
        )

    widest = int(np.argmax(mine["hours"].to_numpy()))
    centre = float(mine["start_hours"].iloc[widest]) + float(mine["hours"].iloc[widest]) / 2
    axis.annotate(
        f"one bar = one EDF file ({mine['hours'].iloc[widest]:.1f} h here)",
        xy=(centre, -BAR_HALF),
        xytext=(centre, -0.85),
        textcoords="data",
        ha="center",
        va="top",
        fontsize=8.5,
        color=C.INK_2,
        arrowprops=dict(arrowstyle="->", color=C.INK_2, linewidth=0.8, shrinkB=1),
    )
    boundaries = [float(v) for v in mine["start_hours"].to_numpy()[1:]]
    if boundaries:
        boundary = boundaries[len(boundaries) // 2]
        axis.annotate(
            "gap = file boundary",
            xy=(boundary, -BAR_HALF),
            xytext=(min(boundary + span * 0.10, span * 0.92), -1.35),
            textcoords="data",
            ha="center",
            va="top",
            fontsize=8.5,
            color=C.INK_2,
            arrowprops=dict(arrowstyle="->", color=C.INK_2, linewidth=0.8, shrinkB=1),
        )

    everything = ticks["x_hours"].to_numpy(float)
    for subset, length, height, text in (
        (
            ticks[ticks["eligible"]],
            TICK_FULL,
            1.05,
            "full tick = eligible prediction target",
        ),
        (
            ticks[~ticks["eligible"]],
            TICK_HALF,
            2.10,
            "half tick = ineligible (no 60 clear minutes before onset)",
        ),
    ):
        if subset.empty:
            continue
        positions = subset["x_hours"].to_numpy(float)
        # The most isolated tick of its class, so the leader is unambiguous.
        gaps = np.abs(positions[:, None] - everything[None, :])
        gaps[gaps == 0.0] = np.inf
        pick = float(positions[int(np.argmax(gaps.min(axis=1)))])
        axis.annotate(
            text,
            xy=(pick, BAR_HALF + length),
            xytext=(float(np.clip(pick, span * 0.26, span * 0.74)), height),
            textcoords="data",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=C.VIOLET,
            arrowprops=dict(arrowstyle="->", color=C.VIOLET, linewidth=0.8, shrinkB=2),
        )

    n_eligible = int(ticks["eligible"].sum())
    axis.set(
        ylim=(-1.85, 2.95),
        yticks=[0.0],
        yticklabels=[f"sub-{subject}"],
        xlabel="Cumulative EEG for this one patient (h)",
    )
    axis.set_title(
        f"Row detail — sub-{subject} magnified   ·   {len(mine)} files, "
        f"{row['hours']:.0f} h, {n_eligible} of {len(ticks)} seizures eligible",
        loc="left",
    )
    C.style(axis, grid="x")


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------


def render(atlas: Atlas) -> Path | None:
    shards, patients = _layout(atlas)
    if patients.empty:
        return None
    placed = _seizure_positions(atlas, shards, patients)

    figure = plt.figure(figsize=(17, 11.4))
    grid = figure.add_gridspec(
        2,
        2,
        height_ratios=[3.05, 1.0],
        width_ratios=[1.22, 1.0],
        hspace=0.30,
        wspace=0.17,
        top=0.905,
        bottom=0.058,
        left=0.088,
        right=0.988,
    )

    n_annotated = int(len(atlas.manifests.seizures))
    mosaic = _draw_mosaic(
        figure.add_subplot(grid[0, :]), figure, shards, patients, placed, n_annotated
    )
    concentration = _draw_concentration(figure.add_subplot(grid[1, 0]), patients, placed)

    subject = _pick_zoom_subject(shards, patients, placed)
    if subject is not None:
        _draw_zoom(figure.add_subplot(grid[1, 1]), figure, subject, shards, patients, placed)

    atlas.fact("mosaic_total_hours", round(mosaic["total_hours"], 1))
    atlas.fact("mosaic_patients", int(len(patients)))
    atlas.fact("mosaic_files", int(len(shards)))
    atlas.fact("mosaic_hours_median", round(float(patients["hours"].median()), 1))
    atlas.fact("mosaic_hours_max", round(float(patients["hours"].max()), 1))
    atlas.fact("mosaic_hours_min", round(float(patients["hours"].min()), 2))
    atlas.fact("mosaic_top10pct_hours_share", round(concentration["top_hours_share"], 1))
    atlas.fact("mosaic_top10pct_target_share", round(concentration["top_target_share"], 1))
    atlas.fact("mosaic_patients_without_target", concentration["patients_without_target"])
    atlas.fact("mosaic_seizures_placed", int(len(placed)))
    atlas.fact("mosaic_seizures_unplaceable", n_annotated - int(len(placed)))
    atlas.fact("mosaic_zoom_subject", subject)

    return C.save(
        figure,
        atlas.path(NUMBER, SLUG),
        f"The whole dataset at once — {len(shards):,} EDF files, "
        f"{mosaic['total_hours']:,.0f} h, {len(patients)} patients, one row each",
        "Bars are files drawn to scale and coloured by split; violet ticks are annotated "
        "seizures, full height when eligible as a prediction target and half height when not.",
        tight=False,
    )
