"""The seizure inventory, and the eligibility filter that chooses the targets."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import common as C
from .context import Atlas

NUMBER = "03"
SLUG = "seizures"
TITLE = "The seizures"
QUESTION = "How many seizures are there, what kind are they, and which ones can be used as targets at all?"
READS = (
    "Six views of seizure_manifest.csv. Top row: the eligibility funnel, drawn as the "
    "three filters actually applied in sequence, with the count and the share of the 883 "
    "annotated events surviving each one; the duration distribution of the seizures that "
    "survive against the ones that do not; and the annotated and eligible counts for each "
    "of the 125 patients who have any seizure at all. Bottom row: every value of the "
    "SeizeIT2 event_type string with its own count, the vigilance state at onset crossed "
    "with eligibility, and the lateralization and localization fields as the reviewers "
    "filled them in. Orange is the part of each bar that becomes a prediction target, "
    "blue the part the eligibility rules drop."
)
TAKE = (
    "Two thirds of the seizure annotations are unusable as prediction targets, and almost "
    "none of that loss is about signal quality: 201 events start less than 60 minutes "
    "into their recording and 286 more have another seizure inside the hour before them, "
    "so 487 of the 566 losses are pure clock arithmetic. Only 79 fail on artefact or "
    "missing data. The filter is not neutral about which seizures it keeps. Surviving "
    "seizures are longer (median 44 s against 26 s for the dropped ones) because "
    "clustered brief events are exactly what the 60-minute rule removes, and 45 % of "
    "awake seizures survive against 31 % of asleep ones, so the target set leans awake "
    "even though the recordings do not. Anything the model appears to learn about "
    "vigilance or seizure length has this selection sitting underneath it. The remaining "
    "317 targets are also concentrated: 24 of the 125 patients contribute none at all and "
    "the ten best-represented patients supply 29 % of them, which is why held-out-patient "
    "scores move so much when the split changes."
)

# The eligibility rule, restated from config so the funnel stages are honest.
CLEAR_SECONDS = C.CONFIG.minimum_preseizure_clear_minutes * 60.0


def _annotated(atlas: Atlas) -> pd.DataFrame:
    """The seizure manifest plus the two structural eligibility causes."""
    table = atlas.manifests.seizures.copy()
    table["recording_id"] = (
        "sub-" + table["subject"]
        + "_ses-" + table["session"].astype(str).str.zfill(2)
        + "_task-" + C.CONFIG.bids_task
        + "_run-" + table["run"].astype(str).str.zfill(2)
    )
    table = table.sort_values(["recording_id", "onset_seconds"]).reset_index(drop=True)
    grouped = table.groupby("recording_id")
    previous_offset = grouped["onset_seconds"].shift(1) + grouped["duration_seconds"].shift(1)
    # A NaN previous offset compares False, which is what we want for the first
    # seizure of a recording.
    table["onset_too_early"] = table["onset_seconds"] < CLEAR_SECONDS
    table["seizure_in_hour"] = (table["onset_seconds"] - previous_offset) < CLEAR_SECONDS
    return table


def _counts(table: pd.DataFrame, column: str) -> pd.DataFrame:
    """Per-category annotated total and eligible count, biggest first."""
    filled = table[column].astype(str).fillna("un").replace({"nan": "un"})
    grouped = table.assign(**{column: filled}).groupby(column)["eligible"]
    return grouped.agg(total="size", eligible="sum").sort_values("total", ascending=False)


def _note(axis: plt.Axes, text: str, *, xy: tuple[float, float], fontsize: float = 7.6,
          ha: str = "left", va: str = "bottom") -> None:
    """A small boxed annotation in axes coordinates."""
    axis.annotate(
        text,
        xy=xy,
        xycoords="axes fraction",
        ha=ha,
        va=va,
        fontsize=fontsize,
        color=C.INK_2,
        linespacing=1.45,
        zorder=9,
        bbox=dict(boxstyle="round,pad=0.32", facecolor=C.SURFACE,
                  edgecolor=C.GRID, linewidth=0.7, alpha=0.95),
    )


# ----------------------------------------------------------------------
# Panels
# ----------------------------------------------------------------------


def _funnel(axis: plt.Axes, table: pd.DataFrame, targeted: int, positives: int,
            full_targets: int) -> list[int]:
    """The three filters, applied in sequence, with what each one costs."""
    total = int(len(table))
    kept_early = ~table["onset_too_early"]
    kept_isolated = kept_early & ~table["seizure_in_hour"]
    stages = [
        ("annotated in the events.tsv files", total),
        ("onset at least 60 min into its recording", int(kept_early.sum())),
        ("no earlier seizure inside that hour", int(kept_isolated.sum())),
        ("that hour is clean, and the file was stored", int(table["eligible"].sum())),
        ("used as a prediction target", int(targeted)),
    ]
    top = len(stages) - 1
    previous = total
    for index, (name, count) in enumerate(stages):
        y = top - index
        axis.barh(y, previous, height=0.50, color=C.GRID, alpha=0.85, zorder=1)
        axis.barh(
            y,
            count,
            height=0.50,
            color=C.ORANGE if index == top else C.BLUE,
            edgecolor="none",
            zorder=3,
        )
        axis.annotate(
            name,
            xy=(total * 0.006, y + 0.33),
            ha="left",
            va="bottom",
            fontsize=8.4,
            color=C.INK,
            fontweight="bold" if index in (0, top) else "normal",
        )
        # Counts live in their own right-hand column so nothing lands on a bar.
        axis.annotate(
            f"{count:,}   ({100.0 * count / total:.0f} %)",
            xy=(total * 1.05, y),
            ha="left",
            va="center",
            fontsize=9.0,
            color=C.ORANGE if index == top else C.INK,
            fontweight="bold",
        )
        drop = previous - count
        if drop > 0:
            axis.annotate(
                f"−{drop}",
                xy=((count + previous) / 2.0, y),
                ha="center",
                va="center",
                fontsize=8.2,
                color=C.INK_2,
            )
        previous = count
    _note(
        axis,
        f"The two clock filters cost {total - stages[2][1]} of the {total - int(table['eligible'].sum())} "
        f"losses; only {stages[2][1] - int(table['eligible'].sum())} fall to artefact,\n"
        f"impedance or bad* annotations inside the hour (8 of those have no stored file).\n"
        f"No eligible seizure goes unused: all {targeted} carry the {positives:,} positive\n"
        f"decisions between them, {full_targets} of them getting the full 10.",
        xy=(0.0, 0.015),
    )
    axis.set(
        xlim=(0, total * 1.42),
        ylim=(-1.55, top + 0.95),
        yticks=[],
        xlabel="Seizures surviving the filter (count)",
    )
    axis.set_title("1.  Eligibility funnel: 883 annotated, 317 usable", loc="left")
    C.style(axis, grid="x")
    return [count for _, count in stages]


def _durations(axis: plt.Axes, table: pd.DataFrame) -> tuple[float, float]:
    """Duration distribution, kept against dropped, on log bins and a log count."""
    eligible = table.loc[table["eligible"], "duration_seconds"].dropna()
    dropped = table.loc[~table["eligible"], "duration_seconds"].dropna()
    bins = np.logspace(np.log10(1.8), np.log10(1100.0), 32)
    axis.hist(
        [eligible.to_numpy(), dropped.to_numpy()],
        bins=bins,
        stacked=True,
        color=[C.ORANGE, C.BLUE],
        edgecolor=C.SURFACE,
        linewidth=0.4,
        label=[f"eligible ({len(eligible)})", f"dropped ({len(dropped)})"],
        zorder=3,
    )
    median_eligible = float(eligible.median())
    median_dropped = float(dropped.median())
    for value, color, name, offset in (
        (median_dropped, C.BLUE, "dropped", 0.855),
        (median_eligible, C.ORANGE, "eligible", 0.96),
    ):
        axis.axvline(value, color=color, linewidth=1.4, linestyle="--", zorder=6)
        axis.annotate(
            f"median {name}\n{value:.0f} s",
            xy=(value, offset),
            xycoords=("data", "axes fraction"),
            xytext=(-4, 0),
            textcoords="offset points",
            ha="right",
            va="top",
            fontsize=8.0,
            color=color,
            fontweight="bold",
            zorder=7,
            bbox=dict(boxstyle="round,pad=0.2", facecolor=C.SURFACE,
                      edgecolor=color, linewidth=0.7, alpha=0.92),
        )
    _note(
        axis,
        "The filter keeps the longer seizures: brief events\n"
        "cluster, and clustering is what the 60-min rule cuts.",
        xy=(0.985, 0.02),
        ha="right",
        fontsize=7.8,
    )
    axis.set(
        xscale="log",
        yscale="log",
        xlim=(1.8, 1100.0),
        ylim=(0.7, 500.0),
        xlabel="Annotated seizure duration (s), log scale",
        ylabel="Seizures (count), log scale",
    )
    axis.set_title("2.  Duration, kept against dropped", loc="left")
    legend = axis.legend(loc="upper right", frameon=True, framealpha=0.95,
                         edgecolor=C.GRID, fontsize=8.2)
    legend.get_frame().set_linewidth(0.7)
    C.style(axis)
    return median_eligible, median_dropped


def _per_patient(axis: plt.Axes, table: pd.DataFrame) -> dict[str, int]:
    """Annotated and eligible seizure counts for every patient who has one."""
    per_patient = table.groupby("subject").size().sort_values(ascending=False)
    eligible_per = table[table["eligible"]].groupby("subject").size()
    order = list(per_patient.index)
    annotated = per_patient.to_numpy()
    eligible = np.array([int(eligible_per.get(name, 0)) for name in order])
    x = np.arange(len(order))
    axis.bar(x, annotated, width=1.0, color=C.BLUE, edgecolor="none",
             label="annotated", zorder=3)
    axis.bar(x, eligible, width=1.0, color=C.ORANGE, edgecolor="none",
             label="eligible (becomes a target)", zorder=4)

    top_ten = float(np.sort(eligible)[::-1][:10].sum() / max(eligible.sum(), 1) * 100.0)
    zero = int((eligible == 0).sum())
    axis.annotate(
        f"subject {order[0]}:  {annotated[0]} annotated,\n{eligible[0]} of them eligible",
        xy=(0.6, annotated[0] * 0.94),
        xytext=(len(order) * 0.07, annotated[0] * 0.72),
        ha="left",
        va="center",
        fontsize=8.0,
        color=C.INK_2,
        arrowprops=dict(arrowstyle="-", color=C.MUTED, linewidth=0.8),
    )
    _note(
        axis,
        f"{len(order)} patients have an annotated seizure, {len(order) - zero} an eligible one.\n"
        f"{zero} contribute no target at all; the ten best-represented\n"
        f"patients supply {top_ten:.0f} % of the {int(eligible.sum())} targets.\n"
        f"Median patient: {int(np.median(annotated))} annotated seizures.",
        xy=(0.985, 0.63),
        ha="right",
        va="bottom",
        fontsize=7.8,
    )
    axis.set(
        xlim=(-0.6, len(order) - 0.4),
        ylim=(0, annotated[0] * 1.30),
        xlabel="Patients (rank, sorted by annotated seizure count)",
        ylabel="Seizures per patient (count)",
    )
    axis.set_title("3.  Seizures per patient", loc="left")
    legend = axis.legend(loc="upper right", frameon=True, framealpha=0.95,
                         edgecolor=C.GRID, fontsize=8.2)
    legend.get_frame().set_linewidth(0.7)
    C.style(axis, grid="y")
    return {"patients_with_seizures": len(order), "patients_without_target": zero,
            "top_ten_share_percent": round(top_ten, 1)}


def _category_bars(axis: plt.Axes, counts: pd.DataFrame, *, title: str,
                   fontsize: float = 7.4, bar_height: float = 0.76,
                   pad: float = 1.42) -> None:
    """Horizontal bars per category: eligible from zero, dropped stacked on top."""
    names = [str(name) for name in counts.index]
    total = counts["total"].to_numpy(dtype=float)
    eligible = counts["eligible"].to_numpy(dtype=float)
    y = np.arange(len(names))[::-1]
    axis.barh(y, eligible, height=bar_height, color=C.ORANGE, edgecolor="none", zorder=3)
    axis.barh(y, total - eligible, left=eligible, height=bar_height, color=C.BLUE,
              edgecolor="none", alpha=0.85, zorder=3)
    limit = float(total.max()) * pad
    for row, name in enumerate(names):
        axis.annotate(
            f"{int(total[row])}  ({int(eligible[row])} elig.)",
            xy=(total[row] + limit * 0.012, y[row]),
            ha="left",
            va="center",
            fontsize=fontsize,
            color=C.INK_2,
        )
    axis.set(
        xlim=(0, limit),
        ylim=(-0.75, len(names) - 0.25),
        yticks=y,
        yticklabels=names,
        xlabel="Seizures (count)",
    )
    axis.tick_params(axis="y", labelsize=fontsize + 0.2, length=0)
    axis.set_title(title, loc="left")
    C.style(axis, grid="x")


def _vigilance(axis: plt.Axes, table: pd.DataFrame) -> dict[str, float]:
    """Vigilance at onset crossed with eligibility, as grouped bars."""
    counts = _counts(table, "vigilance")
    order = [name for name in ("awake", "asleep", "un") if name in counts.index]
    order += [name for name in counts.index if name not in order]
    counts = counts.loc[order]
    eligible = counts["eligible"].to_numpy(dtype=float)
    dropped = (counts["total"] - counts["eligible"]).to_numpy(dtype=float)
    x = np.arange(len(order), dtype=float)
    width = 0.36
    axis.bar(x - width / 2, eligible, width, color=C.ORANGE, edgecolor="none",
             label="eligible", zorder=3)
    axis.bar(x + width / 2, dropped, width, color=C.BLUE, edgecolor="none",
             label="dropped", zorder=3)
    ceiling = float(max(eligible.max(), dropped.max()))
    for index in range(len(order)):
        for offset, value in ((-width / 2, eligible[index]), (width / 2, dropped[index])):
            axis.annotate(
                f"{int(value)}",
                xy=(x[index] + offset, value),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.0,
                color=C.INK_2,
            )
        share = 100.0 * eligible[index] / float(counts["total"].to_numpy()[index])
        axis.annotate(
            f"{share:.0f} % survive",
            xy=(x[index], ceiling * 1.14),
            ha="center",
            va="bottom",
            fontsize=8.2,
            color=C.INK,
            fontweight="bold",
        )
    axis.set(
        xticks=x,
        xticklabels=[f"{name}\n({int(value)})" for name, value in
                     zip(order, counts["total"].to_numpy(), strict=True)],
        xlim=(-0.6, len(order) - 0.4),
        ylim=(0, ceiling * 1.46),
        ylabel="Seizures (count)",
    )
    axis.set_title("5.  Vigilance at onset x eligibility   (un = not scored)", loc="left")
    legend = axis.legend(loc="upper right", frameon=True, framealpha=0.95,
                         edgecolor=C.GRID, fontsize=8.0, ncol=2)
    legend.get_frame().set_linewidth(0.7)
    C.style(axis, grid="y")
    shares = {
        str(name): round(100.0 * float(counts.loc[name, "eligible"]) /
                         float(counts.loc[name, "total"]), 1)
        for name in order
    }
    return shares


# ----------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------


def render(atlas: Atlas) -> Path | None:
    table = _annotated(atlas)
    if table.empty:
        return None
    decisions = atlas.manifests.decisions
    positives = decisions[decisions["label"] == 1]
    targeted = int(positives["target_seizure_id"].nunique())
    per_target = positives.groupby("target_seizure_id").size()
    full_targets = int((per_target == 10).sum())

    figure = plt.figure(figsize=(18.4, 10.8))
    grid = figure.add_gridspec(
        2,
        3,
        height_ratios=[1.0, 1.46],
        hspace=0.30,
        wspace=0.26,
        top=0.905,
        bottom=0.048,
        left=0.088,
        right=0.982,
    )

    stages = _funnel(figure.add_subplot(grid[0, 0]), table, targeted,
                     int(len(positives)), full_targets)
    median_eligible, median_dropped = _durations(figure.add_subplot(grid[0, 1]), table)
    patient_facts = _per_patient(figure.add_subplot(grid[0, 2]), table)

    # ---- 4. every event_type string, verbatim -------------------------
    axis = figure.add_subplot(grid[1, 0])
    event_counts = _counts(table, "event_type")
    _category_bars(
        axis,
        event_counts,
        title=f"4.  event_type, verbatim ({len(event_counts)} distinct strings)",
        fontsize=7.2,
        pad=1.58,
    )
    _note(
        axis,
        "SeizeIT2 naming\n"
        "sz    seizure                     foc   focal onset\n"
        "uo    unknown onset               f2b   focal to bilateral\n"
        "a / ia / ua   aware / impaired awareness / unknown awareness\n"
        "m / nm / um   motor / non-motor / unknown motor\n"
        "trailing word  the motor pattern (hyperkinetic, automatisms,\n"
        "               tonic, clonic, myoclonic, behavior)",
        xy=(0.985, 0.02),
        ha="right",
        fontsize=7.2,
    )
    axis.annotate(
        "orange = eligible, becomes a prediction target\nblue = dropped by the eligibility rules",
        xy=(0.985, 0.42),
        xycoords="axes fraction",
        ha="right",
        va="bottom",
        fontsize=7.8,
        color=C.INK_2,
        bbox=dict(boxstyle="round,pad=0.3", facecolor=C.SURFACE,
                  edgecolor=C.GRID, linewidth=0.7, alpha=0.95),
    )

    # ---- 5 and 6a. vigilance, then lateralization ---------------------
    nested = grid[1, 1].subgridspec(2, 1, height_ratios=[1.0, 0.92], hspace=0.42)
    vigilance_shares = _vigilance(figure.add_subplot(nested[0, 0]), table)

    axis = figure.add_subplot(nested[1, 0])
    lateralization = _counts(table, "lateralization")
    _category_bars(
        axis,
        lateralization,
        title="6a.  Lateralization   (un = unspecified, bi = bilateral)",
        fontsize=8.0,
        bar_height=0.66,
        pad=1.40,
    )

    # ---- 6b. localization ---------------------------------------------
    axis = figure.add_subplot(grid[1, 2])
    localization = _counts(table, "localization")
    _category_bars(
        axis,
        localization,
        title=f"6b.  Localization ({len(localization)} distinct strings)",
        fontsize=7.4,
        pad=1.50,
    )
    _note(
        axis,
        "temp temporal   front frontal   cen central\n"
        "par parietal    occ occipital   ins insular\n"
        "un unspecified. Compound strings are the\n"
        "reviewer's own multi-region labels, kept verbatim.",
        xy=(0.985, 0.02),
        ha="right",
        fontsize=7.2,
    )

    atlas.fact("seizures_annotated", stages[0])
    atlas.fact("seizures_eligible", stages[3])
    atlas.fact("seizures_used_as_targets", targeted)
    atlas.fact("seizures_lost_to_clock_rules", stages[0] - stages[2])
    atlas.fact("seizures_lost_to_data_quality", stages[2] - stages[3])
    atlas.fact("seizure_median_duration_eligible_seconds", median_eligible)
    atlas.fact("seizure_median_duration_dropped_seconds", median_dropped)
    atlas.fact("seizure_eligibility_by_vigilance_percent", vigilance_shares)
    for key, value in patient_facts.items():
        atlas.fact(f"seizure_{key}", value)

    return C.save(
        figure,
        atlas.path(NUMBER, SLUG),
        f"The seizures — {stages[0]} annotated, {targeted} usable as prediction targets "
        f"({100.0 * targeted / stages[0]:.0f} %)",
        "Orange is the part of every bar that survives the eligibility rules and becomes a "
        "target; blue is what the rules drop. Most of the loss is clock arithmetic, not "
        "signal quality, and it is not evenly spread across seizure types.",
        tight=False,
    )
