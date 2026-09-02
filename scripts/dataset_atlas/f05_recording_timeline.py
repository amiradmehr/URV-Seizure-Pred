"""One whole recording, end to end, with every annotation drawn on it."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from . import common as C
from .context import Atlas

NUMBER = "05"
SLUG = "recording_timeline"
TITLE = "One recording, end to end"
QUESTION = "What does a single night of behind-the-ear EEG actually look like?"
READS = (
    "One complete EDF file drawn from the stored, globally z-scored shard: the two "
    "electrodes that were physically recorded as a min/max envelope across the whole "
    "night, the per-minute amplitude beneath it, and the decision track showing which "
    "one-minute slots became a positive decision, a negative decision, or no decision at "
    "all. Every annotation in the sibling events.tsv is shaded and named in place: "
    "seizures, impedance checks, anything else the reviewer marked."
)
TAKE = (
    "This is the unit the pipeline actually consumes. Almost the entire night is one "
    "long negative, punctuated by a handful of seconds of seizure and by artefact bursts "
    "that are far larger than any ictal change. The decision track makes the arithmetic "
    "visible: each seizure contributes exactly ten positive decisions while the same "
    "recording contributes hundreds of negatives, and the grey stretches are candidates "
    "the eligibility rules threw away."
)


def _decision_track(axis: plt.Axes, atlas: Atlas, recording_id: str, duration: float) -> None:
    """A one-row strip showing what each 60-second slot became."""
    stride = C.CONFIG.input_stride_seconds
    rows = atlas.manifests.decisions
    rows = rows[rows["recording_id"] == recording_id]

    axis.axhspan(0, 1, facecolor=C.MUTED, alpha=0.10, zorder=0)
    for label, color in ((0, C.BLUE), (1, C.ORANGE)):
        times = rows.loc[rows["label"] == label, "decision_time_seconds"].to_numpy()
        if times.size == 0:
            continue
        axis.broken_barh(
            [(t / 3600.0, stride / 3600.0) for t in times],
            (0.0, 1.0),
            facecolors=color,
            edgecolors="none",
            zorder=2,
        )
    n_pos = int((rows["label"] == 1).sum())
    n_neg = int((rows["label"] == 0).sum())
    slots = int(duration // stride)
    axis.set(xlim=(0, duration / 3600.0), ylim=(0, 1), yticks=[])
    axis.set_ylabel("decision track\n(one per 60 s)", fontsize=9, color=C.INK_2)
    axis.annotate(
        f"decision track:   {n_neg:,} negative (blue)   ·   {n_pos} positive (orange)   ·   "
        f"{max(slots - n_neg - n_pos, 0):,} of {slots:,} one-minute slots dropped by the "
        f"eligibility rules (grey)",
        xy=(0.5, 1.30),
        xycoords="axes fraction",
        ha="center",
        va="bottom",
        fontsize=9,
        color=C.INK_2,
    )
    C.style(axis, grid="x")


def render(atlas: Atlas) -> Path | None:
    if not atlas.exemplars.hero_recordings:
        return None
    hero = atlas.exemplars.hero
    recording = C.open_recording(hero.recording_id, atlas.manifests)
    events = recording.events()
    duration = recording.duration_seconds
    present = recording.present_channels

    figure, axes = plt.subplots(
        3,
        1,
        figsize=(17, 9.2),
        sharex=True,
        gridspec_kw=dict(height_ratios=[3.4, 1.15, 0.42], hspace=0.16),
    )

    # ---- 1. the signal itself -----------------------------------------
    axis = axes[0]
    segment = recording.slice(0.0, duration)
    spacing = 2.4 * C.robust_limits(segment[present], percentile=99.9, pad=1.0)
    offsets = C.channel_offsets(spacing)
    C.draw_signal(
        axis,
        segment,
        t0=0.0,
        mask=recording.mask,
        offsets=offsets,
        time_unit="h",
        max_points=2600,
        label_dy=spacing * 0.26,
        label_fontsize=9.0,
    )
    kinds = C.annotate_events(
        axis,
        events,
        0.0,
        duration,
        divisor=3600.0,
        min_visible_seconds=duration / 260.0,
        label=False,
    )
    seizures = events[events["span_class"] == "seizure"]
    for row in seizures.itertuples(index=False):
        C.add_marker(
            axis,
            float(row.onset),
            f"{row.eventType}\n{C.fmt_hms(float(row.onset))}  ·  {float(row.duration):.0f} s",
            divisor=3600.0,
            label_y=0.02,
            linewidth=1.3,
            fontsize=8.2,
        )
    axis.set(
        xlim=(0, duration / 3600.0),
        ylim=(-spacing * 1.55, spacing * 1.55),
        ylabel=f"{C.Z_AXIS_OFFSET}\n1 z = one global channel sigma",
    )
    axis.set_title(
        f"{hero.recording_id}   ·   subject {hero.subject}   ·   {hero.split} split   ·   "
        f"{C.fmt_duration(duration)}   ·   electrodes recorded: "
        f"{' + '.join(hero.available_channels)}"
    )
    C.span_legend(
        axis,
        kinds,
        extra=[(C.VIOLET, "seizure onset")],
        loc="lower right",
        title="annotations from events.tsv",
    )
    C.style(axis, grid="x")

    # ---- 2. per-minute amplitude --------------------------------------
    axis = axes[1]
    minutes = int(duration // 60)
    per_minute = np.abs(
        np.asarray(
            recording.array[:, : minutes * int(60 * recording.sfreq) : 16],
            dtype=np.float32,
        )
    )
    per_minute = per_minute.reshape(3, minutes, -1)
    hours = (np.arange(minutes) + 0.5) / 60.0
    for index in present:
        name = C.CONFIG.canonical_channel_names[index]
        axis.plot(
            hours,
            np.median(per_minute[index], axis=1),
            color=C.CHANNEL_COLOR[name],
            linewidth=1.0,
            label=name,
        )
    axis.set(
        yscale="log",
        ylabel="Median |amplitude|\nper minute (z, log)",
        xlim=(0, duration / 3600.0),
    )
    for row in seizures.itertuples(index=False):
        axis.axvline(float(row.onset) / 3600.0, color=C.VIOLET, linewidth=1.2, zorder=5)
    legend = axis.legend(
        loc="upper right", frameon=True, framealpha=0.94, edgecolor=C.GRID, fontsize=8.5
    )
    legend.get_frame().set_linewidth(0.7)
    axis.annotate(
        "amplitude drifts across the night; the seizures (violet lines) "
        "are not where it peaks",
        xy=(0.012, 0.06),
        xycoords="axes fraction",
        fontsize=8.5,
        color=C.INK_2,
        bbox=dict(boxstyle="round,pad=0.25", facecolor=C.SURFACE,
                  edgecolor=C.GRID, linewidth=0.7, alpha=0.92),
    )
    C.style(axis)

    # ---- 3. the decision track ----------------------------------------
    _decision_track(axes[2], atlas, hero.recording_id, duration)
    axes[2].set_xlabel("Time from the start of the recording (h)")

    atlas.fact("hero_recording", hero.recording_id)
    atlas.fact("hero_duration_hours", round(duration / 3600.0, 2))
    atlas.fact("hero_positive_decisions", hero.n_positive_decisions)
    atlas.fact("hero_decisions", hero.n_decisions)

    n_seizures = int(len(seizures))
    return C.save(
        figure,
        atlas.path(NUMBER, SLUG),
        f"One recording, end to end — {duration / 3600:.0f} hours of EEG, "
        f"{n_seizures} seizure{'s' if n_seizures != 1 else ''}, "
        f"{hero.n_positive_decisions} positive decisions out of {hero.n_decisions:,}",
        "The two traces are the electrodes this file physically recorded; the third canonical "
        "channel is absent and zero-filled. Shaded spans are the reviewer's own annotations, "
        "named in place.",
        rect=(0, 0, 1, 0.935),
    )
