"""One seizure at five zoom levels, from the whole night down to four seconds."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ConnectionPatch, Rectangle

from . import common as C
from .context import Atlas

NUMBER = "06"
SLUG = "scale_ladder"
TITLE = "The scale ladder"
QUESTION = "How far apart are the timescales this dataset spans?"
READS = (
    "The same seizure at five magnifications, each panel zooming into the bracketed "
    "region of the one above it: the whole recording in hours, forty minutes, four "
    "minutes, thirty seconds, and finally four seconds at the full 256 Hz sample rate. "
    "The ictal span is shaded and the annotated onset is marked in every panel where it "
    "is wide enough to see. Each panel is scaled to its own content, and the scale bar "
    "gives the amplitude in z."
)
TAKE = (
    "Five orders of magnitude separate the recording from the sample. A seizure that is "
    "unmistakable in the bottom panel is a single hairline in the top one, and the "
    "45-minute history the model reads is itself only about a thirtieth of the night. "
    "This is the core difficulty of the framing: the evidence, if any exists, has to "
    "survive being averaged over 540 five-second chunks. Two caveats belong with the "
    "bottom panels. The stored signal has been through a zero-phase FIR bandpass whose "
    "impulse response is 1,691 samples (6.6 s) long, so energy from the discharge is "
    "smeared roughly three seconds either side of the annotated onset and the fifth "
    "panel cannot be read as sample-exact timing. And this seizure was chosen as the "
    "clearest one in the dataset: across all 317 eligible seizures the median ictal "
    "amplitude is only about twice the interictal baseline, and on roughly a fifth of "
    "channel-observations the ictal window is no larger at all."
)

# (half-width in seconds around the annotated onset, x-axis unit, label)
LEVELS = [
    (None, "h", "the whole recording"),
    (1200.0, "min", "40 minutes"),
    (120.0, "min", "4 minutes"),
    (15.0, "s", "30 seconds"),
    (2.0, "s", "4 seconds — every sample drawn"),
]


def _zoom_bracket(axis: plt.Axes, t0: float, t1: float, divisor: float, text: str) -> None:
    """Mark, at the foot of a panel, the window the next panel expands."""
    low, high = axis.get_ylim()
    height = (high - low) * 0.10
    width = max((t1 - t0) / divisor, (axis.get_xlim()[1] - axis.get_xlim()[0]) * 0.0025)
    centre = (t0 + t1) / 2 / divisor
    axis.add_patch(
        Rectangle(
            (centre - width / 2, low),
            width,
            height,
            facecolor=C.INK,
            edgecolor=C.INK,
            alpha=0.85,
            zorder=10,
        )
    )
    axis.annotate(
        text,
        xy=(centre + width / 2, low + height / 2),
        xytext=(5, 0),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=8.2,
        color=C.INK,
        fontweight="bold",
        zorder=11,
    )


def render(atlas: Atlas) -> Path | None:
    if not atlas.exemplars.seizures:
        return None
    seizure = atlas.exemplars.seizures[0]
    recording = C.open_recording(seizure.recording_id, atlas.manifests)
    events = recording.events()
    onset = seizure.onset_seconds
    duration = max(seizure.duration_seconds, 1.0)
    offset = onset + duration
    present = recording.present_channels

    # One sensitivity for the whole ladder, set by the minute of EEG just before
    # onset, so every panel is directly comparable and the ictal burst saturates
    # exactly as it would in a clinical viewer at a fixed gain.
    baseline = recording.slice(max(onset - 65.0, 0.0), max(onset - 5.0, 1.0))
    sensitivity = C.nice_number(
        C.robust_limits(baseline[present], percentile=99.0, pad=1.0)
    )
    spacing = 8.0 * sensitivity

    figure, axes = plt.subplots(len(LEVELS), 1, figsize=(15.5, 13.0))
    figure.subplots_adjust(top=0.893, bottom=0.055, left=0.10, right=0.985, hspace=0.52)

    windows: list[tuple[float, float, float]] = []
    for index, (half, unit, name) in enumerate(LEVELS):
        axis = axes[index]
        divisor = {"s": 1.0, "min": 60.0, "h": 3600.0}[unit]
        if half is None:
            t0, t1 = 0.0, recording.duration_seconds
        else:
            # Put the annotated onset 30 % in, so pre-ictal EEG is always on screen.
            t0, t1 = onset - 0.6 * half, onset + 1.4 * half
            t0 = max(t0, 0.0)
            t1 = min(t1, recording.duration_seconds)
        windows.append((t0, t1, divisor))

        segment = recording.slice(t0, t1)
        # Panels that sit inside the seizure need the gain turned down, exactly
        # as a reviewer would turn it down to keep the discharge on screen.
        needed = 2.4 * C.robust_limits(segment[present], percentile=99.5, pad=1.0)
        panel_spacing = max(spacing, needed)
        panel_clip = panel_spacing * 0.46
        C.draw_signal(
            axis,
            segment,
            t0=t0,
            mask=recording.mask,
            offsets=C.channel_offsets(panel_spacing),
            time_unit=unit,
            max_points=9000,
            linewidth=0.6 if half and half <= 15 else 0.35,
            label_dy=panel_spacing * 0.30,
            label_fontsize=8.2,
            show_peak=True,
            band=None if (half is not None and half <= 15) else (2.0, 98.0),
            clip=panel_clip,
        )
        if panel_spacing > spacing * 1.05:
            axis.annotate(
                f"gain reduced {panel_spacing / spacing:.0f}x to keep the discharge "
                f"on screen",
                xy=(0.985, 0.03),
                xycoords="axes fraction",
                ha="right",
                va="bottom",
                fontsize=8.2,
                color=C.INK_2,
                bbox=dict(boxstyle="round,pad=0.22", facecolor=C.SURFACE,
                          edgecolor=C.GRID, linewidth=0.7, alpha=0.92),
            )
        axis.set(
            xlim=(t0 / divisor, t1 / divisor),
            ylim=(-panel_spacing * 1.62, panel_spacing * 1.55),
            ylabel=C.Z_AXIS_OFFSET,
        )
        span = t1 - t0
        covers_panel = (min(offset, t1) - max(onset, t0)) > 0.55 * span
        if not covers_panel:
            C.add_span(
                axis,
                onset,
                offset,
                "seizure",
                label=f"ictal  {duration:.0f} s" if span < 4 * 3600 else "",
                divisor=divisor,
                label_y=0.985,
            )
        else:
            axis.annotate(
                f"entirely inside the {duration:.0f}-second ictal span",
                xy=(0.5, 0.985),
                xycoords="axes fraction",
                ha="center",
                va="top",
                fontsize=8.5,
                color=C.VIOLET,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.22", facecolor=C.SURFACE,
                          edgecolor=C.VIOLET, linewidth=0.7, alpha=0.92),
            )
        if span <= 4000:
            C.add_marker(
                axis,
                onset,
                f"annotated onset  t = {C.fmt_hms(onset)}",
                divisor=divisor,
                label_y=0.03,
                fontsize=8.2,
            )
        C.annotate_events(
            axis, events, t0, t1, divisor=divisor, label=False, min_visible_seconds=span / 320
        )
        axis.set_xlabel(f"Time from the start of the recording ({unit})")
        axis.set_title(
            f"{index + 1}.  {name}   ·   {C.fmt_duration(span)} on screen   ·   "
            f"{int(span * recording.sfreq):,} samples per channel",
            loc="left",
        )
        C.style(axis)
        if half is not None and half <= 15:
            C.add_scale_bar(
                axis,
                x=t0 + span * 0.03,
                y=-panel_spacing * 1.38,
                seconds=C.nice_number(span * 0.12),
                z=C.nice_number(panel_spacing / 8.0 * 2),
                divisor=divisor,
            )

    # Brackets and connectors tying each panel to the one below it.
    for index in range(len(LEVELS) - 1):
        upper, lower = axes[index], axes[index + 1]
        t0, t1, divisor = windows[index + 1]
        _zoom_bracket(upper, t0, t1, windows[index][2], f"panel {index + 2} ↓")
        for side, x_next in ((0, t0 / divisor), (1, t1 / divisor)):
            x_here = (t0 if side == 0 else t1) / windows[index][2]
            connector = ConnectionPatch(
                xyA=(x_here, upper.get_ylim()[0]),
                coordsA=upper.transData,
                xyB=(x_next, lower.get_ylim()[1]),
                coordsB=lower.transData,
                color=C.INK,
                linewidth=0.8,
                linestyle=(0, (4, 3)),
                alpha=0.5,
                zorder=0,
            )
            connector.set_annotation_clip(False)
            figure.add_artist(connector)

    atlas.fact("ladder_seizure", seizure.seizure_id)
    atlas.fact("ladder_contrast", round(seizure.contrast, 1))

    return C.save(
        figure,
        atlas.path(NUMBER, SLUG),
        f"The scale ladder — one {seizure.event_type} seizure from "
        f"{C.fmt_duration(recording.duration_seconds)} down to 4 seconds",
        f"{seizure.seizure_id}  ·  subject {seizure.subject} ({seizure.split})  ·  "
        f"{seizure.vigilance}, {seizure.lateralization} {seizure.localization}\n"
        f"Panels share one sensitivity, set from the minute before onset; where the "
        f"discharge would not fit, the gain is reduced and the panel says so.",
        tight=False,
        bbox=None,
    )
