"""How the binary target is produced, and why the positive class is this small."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from . import common as C
from .context import Atlas

NUMBER = "04"
SLUG = "labels"
TITLE = "How the label is made"
QUESTION = "Where do the positive decisions come from, and why are there so few?"
READS = (
    "Top left, the class balance of every one-minute decision, split by split, on a log "
    "count axis with the positive share printed on each pair of bars. Top middle, the "
    "number of positive decisions each target seizure produces, with the 600 s / 60 s = 10 "
    "ceiling marked. Top right, how many decisions survive per EDF file, log on both axes. "
    "The wide panel underneath is the labelling rule drawn from real manifest rows for one "
    "eligible seizure: every candidate one-minute slot in the two hours before onset and "
    "the half hour after it, coloured by the label it received or left grey where the "
    "pipeline dropped it, with the 45-minute history and 10-minute occurrence period of "
    "the last negative, the first positive and the last positive drawn underneath, and the "
    "60-minute clean-EEG eligibility bracket below those."
)
TAKE = (
    "The positive class is not rare because seizures are rare in the recordings; it is "
    "rare because the rule mints exactly ten positives per usable seizure. 600 s of "
    "occurrence period divided by a 60 s stride is 10, and 301 of the 317 target seizures "
    "hit that number exactly, so 3,113 of 494,283 decisions (0.63 %) are positive and the "
    "whole positive class is really 317 events wearing 3,113 labels. Those ten histories "
    "overlap heavily: consecutive ones share 44 of their 45 minutes, and the first and the "
    "last of a run still share 80 % of their samples, so ten positives is closer to one "
    "independent example than to ten. Prevalence is almost identical across the three "
    "splits (0.633 / 0.618 / 0.620 %), which is reassuring for calibration but means a "
    "validation split with 33 target seizures carries very little resolution. The "
    "geometry panel also shows where the negatives go: in this recording 26 slots before "
    "onset and every slot after it are dropped because their history touches an "
    "ictal or postictal span, and across the cohort 47 % of EDF files keep 10 decisions or "
    "fewer while the busiest 5 % of files hold 29 % of them all."
)


# ----------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------

STRIDE = C.CONFIG.input_stride_seconds
HISTORY = C.CONFIG.input_window_seconds
SOP = C.CONFIG.seizure_occurrence_period_minutes * 60.0
HORIZON = C.CONFIG.prediction_horizon_minutes * 60.0
POSTICTAL = C.CONFIG.postictal_exclusion_minutes * 60.0
CLEAR = C.CONFIG.minimum_preseizure_clear_minutes * 60.0
SLOTS_PER_SEIZURE = int(round(SOP / STRIDE))


@dataclass
class GeometryPick:
    """One eligible seizure plus everything needed to draw its labelling."""

    seizure_id: str
    recording_id: str
    subject: str
    split: str
    onset: float
    duration: float
    event_type: str
    vigilance: str
    decisions: pd.DataFrame
    grid: np.ndarray  # candidate decision times, seconds
    n_dropped_before: int
    n_dropped_after: int


def _candidate_grid(manifests: C.Manifests, recording_id: str) -> np.ndarray:
    """Every decision instant the loop in ``preprocessing`` visits, in seconds.

    The loop starts with ``decision_end_sample = history_samples`` and advances by
    one stride until the occurrence period no longer fits inside the recording,
    so the grid is fixed by the recording length alone and does not depend on
    which slots survived.
    """
    samples = int(np.load(manifests.shard(recording_id)["X_path"], mmap_mode="r").shape[1])
    sfreq = C.CONFIG.target_sfreq
    last = int(np.floor((samples - (HORIZON + SOP) * sfreq - HISTORY * sfreq) / (STRIDE * sfreq)))
    if last < 0:
        return np.zeros(0)
    return HISTORY + np.arange(last + 1) * STRIDE


def _drop_reason(
    t: float, seizures: pd.DataFrame, events: pd.DataFrame, eligible: set[str]
) -> str:
    """Why the pipeline kept no decision at instant ``t``, reconstructed from disk."""
    start = t - HISTORY
    for row in seizures.itertuples(index=False):
        onset = float(row.onset_seconds)
        stop = onset + float(row.duration_seconds)
        if onset <= t < stop + POSTICTAL:
            return "decision instant is ictal or postictal"
    for row in seizures.itertuples(index=False):
        onset = float(row.onset_seconds)
        stop = onset + float(row.duration_seconds) + POSTICTAL
        if start < stop and onset < t:
            return "45-min history overlaps an ictal or postictal span"
    for row in events.itertuples(index=False):
        kind = str(row.eventType).strip().lower()
        if kind == "bckg" or kind.startswith("sz_"):
            continue
        if start < float(row.offset) and float(row.onset) < t:
            return f"45-min history overlaps a {row.eventType} annotation"
    for row in seizures.itertuples(index=False):
        onset = float(row.onset_seconds)
        if t < onset <= t + HORIZON + SOP and str(row.seizure_id) not in eligible:
            return "the seizure in the occurrence period is not eligible"
    return "non-finite samples or a bad* annotation in the history"


def _runs(times: np.ndarray, reasons: list[str]) -> list[tuple[float, float, str]]:
    """Group consecutive dropped slots that share a reason into labelled runs."""
    grouped: list[tuple[float, float, str]] = []
    for t, reason in zip(times, reasons, strict=True):
        if grouped and grouped[-1][2] == reason and abs(t - grouped[-1][1] - STRIDE) < 1e-6:
            grouped[-1] = (grouped[-1][0], t, reason)
        else:
            grouped.append((t, t, reason))
    return grouped


def _pick_geometry_seizure(manifests: C.Manifests, span: float) -> GeometryPick | None:
    """The eligible seizure whose surroundings show the rule most completely.

    Wanted: a full run of ten positives, the whole ``span`` of candidate grid on
    both sides of onset so nothing is cut off by the file boundary, plenty of
    surviving negatives, and at least one stretch of dropped slots so the panel
    shows what the eligibility rules remove as well as what they keep.
    """
    seizures = manifests.seizures[manifests.seizures["eligible"]].copy()
    seizures["recording_id"] = (
        "sub-" + seizures["subject"]
        + "_ses-" + seizures["session"].astype(str).str.zfill(2)
        + "_task-" + C.CONFIG.bids_task
        + "_run-" + seizures["run"].astype(str).str.zfill(2)
    )
    positives = manifests.decisions[manifests.decisions["label"] == 1]
    per_target = positives.groupby("target_seizure_id").size()
    seizures["n_positive"] = (
        seizures["seizure_id"].map(per_target).fillna(0).astype(int)
    )
    seizures = seizures[seizures["n_positive"] == SLOTS_PER_SEIZURE]
    seizures = seizures[seizures["onset_seconds"] >= HISTORY + span]
    seizures = seizures[seizures["recording_id"].isin(manifests.shard_by_id.index)]

    by_recording = dict(tuple(manifests.decisions.groupby("recording_id")))
    best: tuple[float, GeometryPick] | None = None
    for row in seizures.itertuples(index=False):
        table = by_recording.get(row.recording_id)
        if table is None:
            continue
        onset = float(row.onset_seconds)
        grid = _candidate_grid(manifests, row.recording_id)
        window = grid[(grid >= onset - span) & (grid <= onset + span / 4.0)]
        if window.size == 0 or window.min() > onset - span + STRIDE:
            continue
        kept = np.isin(window, table["decision_time_seconds"].to_numpy())
        before = int((~kept & (window < onset)).sum())
        after = int((~kept & (window >= onset)).sum())
        negatives = int(((table["label"] == 0) & table["decision_time_seconds"].between(
            onset - span, onset + span / 4.0
        )).sum())
        if after < int(span / 4.0 / STRIDE) - 1:  # the file must run past the onset
            continue
        score = negatives * 0.02 + min(before, 30) * 1.0 + (12.0 if before else 0.0)
        if best is None or score > best[0]:
            best = (
                score,
                GeometryPick(
                    seizure_id=str(row.seizure_id),
                    recording_id=str(row.recording_id),
                    subject=str(row.subject),
                    split=str(table["split"].iloc[0]),
                    onset=onset,
                    duration=float(row.duration_seconds),
                    event_type=str(row.event_type),
                    vigilance=str(row.vigilance),
                    decisions=table,
                    grid=window,
                    n_dropped_before=before,
                    n_dropped_after=after,
                ),
            )
    return None if best is None else best[1]


# ----------------------------------------------------------------------
# Top row
# ----------------------------------------------------------------------


def _class_balance(axis: plt.Axes, decisions: pd.DataFrame) -> dict[str, float]:
    """Negatives and positives per split, log counts, prevalence printed."""
    order = [s for s in ("train", "validation", "test") if (decisions["split"] == s).any()]
    prevalence: dict[str, float] = {}
    positions = np.arange(len(order), dtype=float)
    for offset, label in ((-0.19, 0), (0.19, 1)):
        heights = [int(((decisions["split"] == s) & (decisions["label"] == label)).sum())
                   for s in order]
        axis.bar(
            positions + offset,
            heights,
            width=0.36,
            color=C.LABEL_COLOR[label],
            label=f"{label} = {C.LABEL_NAME[label]}",
            zorder=3,
        )
        for x, height in zip(positions + offset, heights, strict=True):
            axis.annotate(
                f"{height:,}",
                xy=(x, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=8.4,
                color=C.INK,
                fontweight="bold" if label else "normal",
            )
    tick_labels: list[str] = []
    for split in order:
        group = decisions[decisions["split"] == split]
        share = float((group["label"] == 1).mean())
        prevalence[split] = share
        tick_labels.append(
            f"{split}\n{100 * share:.3f} % positive\n{group['subject'].nunique()} patients\n"
            f"{group.loc[group['label'] == 1, 'target_seizure_id'].nunique()} target seizures"
        )
    axis.set(
        yscale="log",
        ylim=(1.0, 3.0e6),
        xticks=positions,
        xticklabels=tick_labels,
        ylabel="Decisions (count, log scale)",
    )
    axis.set_title("1  Class balance, per whole-patient split", loc="left")
    legend = axis.legend(
        loc="upper right", frameon=True, framealpha=0.94, edgecolor=C.GRID, fontsize=8.5,
        title="label", title_fontsize=8.5,
    )
    legend.get_frame().set_linewidth(0.7)
    C.style(axis, grid="y")
    axis.tick_params(axis="x", labelsize=8.2, length=0)
    return prevalence


def _per_seizure(axis: plt.Axes, decisions: pd.DataFrame) -> tuple[int, int]:
    """Positives per target seizure: the 600 s / 60 s = 10 ceiling."""
    per_target = decisions[decisions["label"] == 1].groupby("target_seizure_id").size()
    counts = per_target.to_numpy()
    n_full = int((counts == SLOTS_PER_SEIZURE).sum())
    edges = np.arange(0.5, SLOTS_PER_SEIZURE + 1.5)
    heights, _ = np.histogram(counts, bins=edges)
    centres = np.arange(1, SLOTS_PER_SEIZURE + 1)
    for centre, height in zip(centres, heights, strict=True):
        if height == 0:
            continue
        full = centre == SLOTS_PER_SEIZURE
        axis.bar(
            centre,
            height,
            width=0.82,
            color=C.ORANGE,
            alpha=1.0 if full else 0.42,
            hatch=None if full else "///",
            edgecolor=C.ORANGE,
            linewidth=0.8,
            zorder=3,
        )
        axis.annotate(
            f"{height}",
            xy=(centre, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=8.4,
            color=C.INK,
            fontweight="bold" if full else "normal",
        )
    axis.axvline(SLOTS_PER_SEIZURE, color=C.INK, linestyle="--", linewidth=1.3, zorder=5)
    axis.annotate(
        f"occurrence period / stride\n= {SOP:.0f} s / {STRIDE:.0f} s = {SLOTS_PER_SEIZURE}\n"
        f"{n_full} of {len(counts)} target seizures\nreach the ceiling exactly",
        xy=(SLOTS_PER_SEIZURE - 0.72, 26.0),
        ha="right",
        va="center",
        fontsize=9.0,
        color=C.INK,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.32", facecolor=C.SURFACE, edgecolor=C.INK,
                  linewidth=0.9, alpha=0.95),
        zorder=6,
    )
    axis.annotate(
        f"{len(counts) - n_full} seizures fall short:\nan earlier event blocks part of\n"
        f"the run-up, so fewer slots survive",
        xy=(0.7, 190.0),
        ha="left",
        va="center",
        fontsize=8.2,
        color=C.INK_2,
        bbox=dict(boxstyle="round,pad=0.24", facecolor=C.SURFACE, edgecolor=C.GRID,
                  linewidth=0.7, alpha=0.92),
        zorder=6,
    )
    axis.set(
        yscale="log",
        ylim=(0.6, 1.2e3),
        xlim=(0.2, SLOTS_PER_SEIZURE + 0.9),
        xticks=centres,
        xlabel="Positive decisions for one target seizure (count)",
        ylabel="Target seizures (count, log scale)",
    )
    axis.set_title(
        f"2  The sharpest constraint: {SLOTS_PER_SEIZURE} positives per seizure", loc="left"
    )
    C.style(axis, grid="y")
    return n_full, int(len(counts))


def _per_file(axis: plt.Axes, decisions: pd.DataFrame) -> dict[str, float]:
    """Decisions retained per EDF file, log on both axes."""
    retained = decisions.groupby("recording_id").size()
    values = retained.to_numpy()
    edges = np.logspace(0, np.log10(values.max() * 1.05), 44)
    axis.hist(values, bins=edges, color=C.BLUE, zorder=3)
    median = float(np.median(values))
    axis.axvline(median, color=C.INK, linestyle="--", linewidth=1.3, zorder=5)
    top = values[values >= np.quantile(values, 0.95)]
    share = float(top.sum() / values.sum())
    small = int((values <= SLOTS_PER_SEIZURE).sum())
    axis.annotate(
        f"median {median:.0f}",
        xy=(median, 1.9),
        xytext=(5, 0),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=8.6,
        color=C.INK,
        fontweight="bold",
    )
    axis.annotate(
        f"{small:,} of {len(values):,} files ({100 * small / len(values):.0f} %) keep\n"
        f"{SLOTS_PER_SEIZURE} decisions or fewer; the busiest 5 %\n"
        f"of files hold {100 * share:.0f} % of all {len(decisions):,} decisions",
        xy=(0.985, 0.965),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=8.2,
        color=C.INK_2,
        bbox=dict(boxstyle="round,pad=0.26", facecolor=C.SURFACE, edgecolor=C.GRID,
                  linewidth=0.7, alpha=0.94),
    )
    axis.set(
        xscale="log",
        yscale="log",
        ylim=(0.6, 2.2e3),
        xlabel="Decisions retained by one EDF file (count, log scale)",
        ylabel="EDF files (count, log scale)",
    )
    axis.set_title(f"3  Yield per file  (n = {len(values):,} files)", loc="left")
    C.style(axis, grid="both")
    return {"median": median, "top5_share": share, "small": small}


# ----------------------------------------------------------------------
# The labelling geometry
# ----------------------------------------------------------------------

BAND_LOW, BAND_HIGH = 6.00, 6.80
ROW_Y = {"rule": 5.05, "neg": 4.05, "first": 3.05, "last": 2.05}
ROW_HALF = 0.20


def _decision_row(
    axis: plt.Axes,
    y: float,
    t: float,
    onset: float,
    *,
    label: int,
    text: str,
    leader_to: float,
) -> None:
    """One decision drawn as its history bar plus its occurrence-period bar."""
    rel = (t - onset) / 60.0
    axis.add_patch(
        Rectangle(
            (rel - HISTORY / 60.0, y - ROW_HALF),
            HISTORY / 60.0,
            2 * ROW_HALF,
            facecolor=C.BLUE,
            alpha=0.20,
            edgecolor=C.BLUE,
            linewidth=0.9,
            zorder=3,
        )
    )
    axis.add_patch(
        Rectangle(
            (rel, y - ROW_HALF),
            (HORIZON + SOP) / 60.0,
            2 * ROW_HALF,
            facecolor=C.LABEL_COLOR[label],
            alpha=0.30 if label else 0.16,
            edgecolor=C.LABEL_COLOR[label],
            linewidth=0.9,
            hatch=None if label else "///",
            zorder=3,
        )
    )
    axis.plot([rel, rel], [y - ROW_HALF - 0.10, y + ROW_HALF + 0.10],
              color=C.INK, linestyle="--", linewidth=1.1, zorder=6)
    axis.annotate(
        "45-min input history",
        xy=(rel - HISTORY / 120.0, y),
        ha="center", va="center", fontsize=7.8, color=C.INK_2, zorder=7,
    )
    axis.annotate(
        f"{SOP / 60:.0f} min",
        xy=(rel + (HORIZON + SOP) / 120.0, y),
        ha="center", va="center", fontsize=7.6, color=C.INK_2, zorder=7,
    )
    axis.plot(
        [rel + (HORIZON + SOP) / 60.0, leader_to],
        [y, y],
        color=C.MUTED, linewidth=0.6, linestyle=":", zorder=3,
    )
    axis.annotate(
        text,
        xy=(leader_to + 0.8, y),
        ha="left", va="center", fontsize=8.2, color=C.INK, zorder=7,
    )


def _geometry(axis: plt.Axes, atlas: Atlas, pick: GeometryPick, span: float) -> dict[str, float]:
    """The labelling rule drawn from the manifest rows of one eligible seizure."""
    onset = pick.onset
    rel = lambda t: (t - onset) / 60.0  # noqa: E731
    table = pick.decisions
    kept = table.set_index("decision_time_seconds")["label"].to_dict()

    events = C.read_events(pick.recording_id)
    seizures = atlas.manifests.seizures.copy()
    seizures["recording_id"] = (
        "sub-" + seizures["subject"]
        + "_ses-" + seizures["session"].astype(str).str.zfill(2)
        + "_task-" + C.CONFIG.bids_task
        + "_run-" + seizures["run"].astype(str).str.zfill(2)
    )
    mine = seizures[seizures["recording_id"] == pick.recording_id]
    eligible_ids = set(mine.loc[mine["eligible"], "seizure_id"].astype(str))

    # ---- the candidate slot band --------------------------------------
    axis.add_patch(
        Rectangle((rel(pick.grid.min()) - 0.5, BAND_LOW), 0, 0, facecolor="none")
    )
    dropped_times: list[float] = []
    for t in pick.grid:
        label = kept.get(t)
        x = rel(t)
        if label is None:
            dropped_times.append(t)
            axis.add_patch(Rectangle((x - 0.30, BAND_LOW + 0.06), 0.60, 0.26,
                                     facecolor=C.ABSENT_COLOR, edgecolor="none", zorder=4))
        elif label == 1:
            axis.add_patch(Rectangle((x - 0.34, BAND_LOW + 0.02), 0.68, 0.76,
                                     facecolor=C.ORANGE, edgecolor=C.ORANGE,
                                     linewidth=0.5, zorder=6))
        else:
            axis.add_patch(Rectangle((x - 0.30, BAND_LOW + 0.04), 0.60, 0.52,
                                     facecolor=C.BLUE, edgecolor="none", zorder=5))

    reasons = [_drop_reason(t, mine, events, eligible_ids) for t in dropped_times]
    for start, stop, reason in _runs(np.asarray(dropped_times), reasons):
        n = int(round((stop - start) / STRIDE)) + 1
        axis.add_patch(
            Rectangle(
                (rel(start) - 0.5, BAND_LOW),
                (stop - start) / 60.0 + 1.0,
                BAND_HIGH - BAND_LOW,
                facecolor=C.MUTED, alpha=0.14, hatch="///", edgecolor=C.MUTED,
                linewidth=0.0, zorder=3,
            )
        )
        if n < 6:
            continue
        # Anchor a run that touches an edge of the grid inward, so a long label
        # never runs off the axes.
        if start <= pick.grid.min() + STRIDE:
            x, align = rel(start) - 0.5, "left"
        elif stop >= pick.grid.max() - STRIDE:
            x, align = rel(stop) + 0.5, "right"
        else:
            x, align = (rel(start) + rel(stop)) / 2.0, "center"
        axis.annotate(
            f"{n} slots dropped: {reason}",
            xy=(x, (BAND_LOW + BAND_HIGH) / 2.0),
            ha=align, va="center", fontsize=7.8, color=C.INK_2, zorder=8,
            bbox=dict(boxstyle="round,pad=0.20", facecolor=C.SURFACE,
                      edgecolor=C.MUTED, linewidth=0.6, alpha=0.90),
        )

    # ---- the three worked decisions -----------------------------------
    in_window = table[
        table["decision_time_seconds"].between(pick.grid.min(), pick.grid.max())
    ]
    positives = np.sort(
        in_window.loc[in_window["label"] == 1, "decision_time_seconds"].to_numpy()
    )
    negatives = in_window.loc[in_window["label"] == 0, "decision_time_seconds"].to_numpy()
    first_positive, last_positive = float(positives[0]), float(positives[-1])
    before = negatives[negatives < first_positive]
    last_negative = float(before.max()) if before.size else None

    leader = rel(onset) + 11.4
    if last_negative is not None:
        gap = onset - (last_negative + HORIZON + SOP)
        _decision_row(
            axis, ROW_Y["neg"], last_negative, onset, label=0, leader_to=leader,
            text=f"t = onset - {onset - last_negative:.0f} s.  The window closes "
                 f"{gap:.0f} s before onset,\nso the answer is no:  label 0",
        )
    _decision_row(
        axis, ROW_Y["first"], first_positive, onset, label=1, leader_to=leader,
        text=f"t = onset - {onset - first_positive:.0f} s <= {SOP:.0f} s.  "
             f"First of the {SLOTS_PER_SEIZURE}:  label 1",
    )
    _decision_row(
        axis, ROW_Y["last"], last_positive, onset, label=1, leader_to=leader,
        text=f"t = onset - {onset - last_positive:.0f} s.  Last of the "
             f"{SLOTS_PER_SEIZURE}:  label 1.\nThe next slot is inside the seizure and is dropped",
    )

    # how the history slides between the first and the last positive
    slide_y = 1.36
    x0, x1 = rel(first_positive - HISTORY), rel(last_positive - HISTORY)
    axis.annotate(
        "", xy=(x1, slide_y), xytext=(x0, slide_y),
        arrowprops=dict(arrowstyle="<|-|>", color=C.INK_2, linewidth=1.0,
                        shrinkA=0, shrinkB=0),
    )
    shared = (HISTORY - (last_positive - first_positive)) / HISTORY
    axis.annotate(
        f"the {SLOTS_PER_SEIZURE} histories slide "
        f"{(last_positive - first_positive) / 60:.0f} min end to end; consecutive ones share "
        f"{100 * (HISTORY - STRIDE) / HISTORY:.1f} % of their samples, first and last "
        f"share {100 * shared:.0f} %",
        xy=((x0 + x1) / 2.0, slide_y - 0.12),
        ha="center", va="top", fontsize=8.2, color=C.INK_2,
        bbox=dict(boxstyle="round,pad=0.24", facecolor=C.SURFACE, edgecolor=C.GRID,
                  linewidth=0.7, alpha=0.94),
    )

    axis.annotate(
        f"{SLOTS_PER_SEIZURE} positive slots, one every {STRIDE:.0f} s",
        xy=(rel((first_positive + last_positive) / 2.0), BAND_LOW - 0.06),
        ha="center", va="top", fontsize=8.2, color=C.ORANGE, fontweight="bold", zorder=9,
        bbox=dict(boxstyle="round,pad=0.22", facecolor=C.SURFACE, edgecolor=C.ORANGE,
                  linewidth=0.8, alpha=0.94),
    )

    # ---- the eligibility bracket ---------------------------------------
    axis.add_patch(
        Rectangle(
            (rel(onset - CLEAR), ROW_Y["rule"] - ROW_HALF),
            CLEAR / 60.0,
            2 * ROW_HALF,
            facecolor=C.AQUA, alpha=0.18, edgecolor=C.AQUA, linewidth=1.0, zorder=3,
        )
    )
    axis.annotate(
        f"ELIGIBILITY: {CLEAR / 60:.0f} min of continuous clean EEG before onset",
        xy=(rel(onset - CLEAR * 0.63), ROW_Y["rule"]),
        ha="center", va="center", fontsize=8.4, color=C.INK, fontweight="bold", zorder=7,
    )
    axis.annotate(
        f"no non-finite sample, no bad* annotation, no other ictal or postictal\n"
        f"span, no non-background event.  566 of 883 annotated seizures fail it",
        xy=(leader + 0.8, ROW_Y["rule"]),
        ha="left", va="center", fontsize=8.2, color=C.INK, zorder=7,
    )

    # ---- the seizure itself --------------------------------------------
    axis.add_patch(
        Rectangle(
            (0.0, 0.0), max(pick.duration / 60.0, 0.45), 8.4,
            facecolor=C.VIOLET, alpha=0.20, edgecolor="none", zorder=2,
        )
    )
    axis.axvline(0.0, color=C.VIOLET, linewidth=1.6, zorder=9)
    axis.annotate(
        f"SEIZURE ONSET  t = {C.fmt_hms(onset)}  ({onset:,.0f} s into the file)\n"
        f"{pick.event_type}, {pick.duration:.0f} s long, {pick.vigilance}, ictal span shaded",
        xy=(max(pick.duration / 60.0, 0.45) + 0.8, 8.30),
        ha="left", va="top", fontsize=8.6, color=C.VIOLET, fontweight="bold", zorder=10,
        bbox=dict(boxstyle="round,pad=0.24", facecolor=C.SURFACE, edgecolor=C.VIOLET,
                  linewidth=0.8, alpha=0.94),
    )

    # ---- the arithmetic -------------------------------------------------
    axis.annotate(
        "label 1        if   decision_time < onset  and  onset <= decision_time + "
        f"{SOP:.0f} s   ->   {SLOTS_PER_SEIZURE} slots per eligible seizure\n"
        "label 0        if   no annotated onset falls in that 10-minute window\n"
        "no decision  if   the decision instant is itself ictal or postictal, or the 45-min "
        "history touches\n                       an ictal or postictal span, a bad* "
        "annotation, a non-background event or a\n                       non-finite sample, "
        "or the only onset in the window is an ineligible seizure",
        xy=(0.003, 0.995),
        xycoords="axes fraction",
        ha="left", va="top", fontsize=8.3, color=C.INK, zorder=10,
        bbox=dict(boxstyle="round,pad=0.32", facecolor=C.SURFACE, edgecolor=C.GRID,
                  linewidth=0.8, alpha=0.95),
    )

    n_neg = int(len(negatives))
    n_drop = len(dropped_times)
    axis.set(
        xlim=(rel(pick.grid.min()) - 1.0, rel(pick.grid.max()) + 20.0),
        ylim=(0.55, 8.4),
        yticks=[(BAND_LOW + BAND_HIGH) / 2.0, ROW_Y["rule"], ROW_Y["neg"],
                ROW_Y["first"], ROW_Y["last"]],
        yticklabels=["candidate\nslots", "eligibility\nrule", "last negative\nbefore the run",
                     "first positive", "last positive"],
        xlabel="Time relative to the annotated seizure onset (min).  "
               "One candidate slot every 60 s.",
    )
    axis.set_title(
        f"4  The rule, drawn from the manifest rows of {pick.seizure_id}   ·   "
        f"subject {pick.subject} ({pick.split} split)   ·   in this window: "
        f"{len(positives)} positive, {n_neg} negative, {n_drop} slots with no decision",
        loc="left",
    )
    axis.tick_params(axis="y", length=0)
    C.style(axis, grid="x")
    for side in ("left",):
        axis.spines[side].set_visible(False)
    return {
        "first_positive_lead_seconds": onset - first_positive,
        "last_positive_lead_seconds": onset - last_positive,
        "shared_fraction_first_last": shared,
        "n_dropped_before": pick.n_dropped_before,
        "n_dropped_after": pick.n_dropped_after,
    }


# ----------------------------------------------------------------------
# Render
# ----------------------------------------------------------------------


def render(atlas: Atlas) -> Path | None:
    decisions = atlas.manifests.decisions
    span = 120.0 * 60.0

    figure = plt.figure(figsize=(17, 9.6))
    grid = figure.add_gridspec(
        2, 3,
        height_ratios=[1.0, 1.32],
        hspace=0.42,
        wspace=0.24,
        top=0.885,
        bottom=0.062,
        left=0.052,
        right=0.988,
    )

    prevalence = _class_balance(figure.add_subplot(grid[0, 0]), decisions)
    n_full, n_targets = _per_seizure(figure.add_subplot(grid[0, 1]), decisions)
    per_file = _per_file(figure.add_subplot(grid[0, 2]), decisions)

    pick = _pick_geometry_seizure(atlas.manifests, span)
    if pick is not None:
        measured = _geometry(figure.add_subplot(grid[1, :]), atlas, pick, span)
        atlas.fact("labels_geometry_seizure", pick.seizure_id)
        for key, value in measured.items():
            atlas.fact(f"labels_{key}", value)

    n_positive = int((decisions["label"] == 1).sum())
    atlas.fact("labels_positive_decisions", n_positive)
    atlas.fact("labels_prevalence", round(float((decisions["label"] == 1).mean()), 6))
    atlas.fact("labels_targets_at_ceiling", f"{n_full} of {n_targets}")
    atlas.fact("labels_prevalence_by_split", {k: round(v, 5) for k, v in prevalence.items()})
    atlas.fact("labels_decisions_per_file_median", per_file["median"])
    atlas.fact("labels_top5_file_share", round(per_file["top5_share"], 4))

    return C.save(
        figure,
        atlas.path(NUMBER, SLUG),
        f"How the label is made — {n_positive:,} of {len(decisions):,} decisions are "
        f"positive (0.63 %), and they come from only {n_targets} seizures",
        f"A decision is positive when an eligible seizure onset falls in the {SOP / 60:.0f} "
        f"minutes after it. At a {STRIDE:.0f} s stride that is exactly "
        f"{SLOTS_PER_SEIZURE} decisions per seizure, so the positive class is "
        f"{n_targets} events counted {SLOTS_PER_SEIZURE} times each.",
        tight=False,
    )
