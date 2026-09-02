"""One decision, fully annotated: what the model reads and what it is asked."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrow, Rectangle

from . import common as C
from .context import Atlas

NUMBER = "09"
SLUG = "decision_anatomy"
TITLE = "One decision, taken apart"
QUESTION = "What exactly is the model shown, and what exactly is it asked?"
READS = (
    "A single positive decision drawn continuously from its recording, so both halves of "
    "the question are on the same axis: the 45-minute history the model reads (left of "
    "the dashed line) and the 10-minute occurrence period the label refers to (right of "
    "it), with the target seizure's annotated onset and ictal span marked inside it. "
    "Below, the same history cut into the 540 five-second chunks the encoder is applied "
    "to, three of those chunks at full sample resolution, and the tensor arithmetic that "
    "turns 691,200 samples into one number."
)
TAKE = (
    "The model never sees the seizure. Everything right of the dashed line is the "
    "question, not the evidence, and at prediction_horizon_minutes = 0 the onset can fall "
    "anywhere in that ten-minute window. On the evidence side there is no visible "
    "landmark: the chunk ending at the decision instant looks like the chunk from "
    "45 minutes earlier, which is the honest difficulty of the task. Mean pooling over "
    "the 540 chunk embeddings also throws away their order, so the same history shuffled "
    "produces an identical logit."
)


def _chunk_strip(axis: plt.Axes, highlights: list[int]) -> None:
    """A strip standing in for the 540 chunk slots, with three picked out."""
    total = C.CHUNKS_PER_HISTORY
    axis.add_patch(
        Rectangle((0, 0), total, 1, facecolor=C.BLUE, alpha=0.16, edgecolor="none")
    )
    for index in range(0, total, 20):
        axis.axvline(index, color=C.BLUE, linewidth=0.4, alpha=0.55)
    for index in highlights:
        axis.add_patch(
            Rectangle(
                (index, 0), 1, 1, facecolor=C.ORANGE, edgecolor=C.ORANGE, linewidth=1.4
            )
        )
        axis.annotate(
            f"chunk {index}",
            xy=(index + 0.5, 1.05),
            ha="center",
            va="bottom",
            fontsize=8.2,
            color=C.ORANGE,
            fontweight="bold",
        )
    axis.set(xlim=(0, total), ylim=(0, 1), yticks=[])
    axis.set_xlabel(
        f"Chunk index within the history  (0 = oldest, {total - 1} = ending at the "
        f"decision instant)"
    )
    axis.set_ylabel("chunk slots\n(count)", fontsize=9, color=C.INK_2)
    C.style(axis, grid="none")


def _tensor_diagram(axis: plt.Axes) -> None:
    """The shape arithmetic from raw window to a single logit."""
    axis.set(xlim=(0, 10), ylim=(0, 10))
    axis.axis("off")
    blocks = [
        (
            "45-minute history",
            f"(3, {C.HISTORY_SAMPLES:,}) = {3 * C.HISTORY_SAMPLES:,} numbers",
            C.BLUE,
        ),
        (
            "reshape into chunks",
            f"({C.CHUNKS_PER_HISTORY}, 3, {C.CHUNK_SAMPLES:,})",
            C.BLUE,
        ),
        (
            "EEGNet encoder, shared weights",
            f"applied {C.CHUNKS_PER_HISTORY}x -> ({C.CHUNKS_PER_HISTORY}, embedding)",
            C.AQUA,
        ),
        ("mean pool over chunks", "(embedding,) — chunk order discarded", C.AQUA),
        (
            "+ availability mask -> linear head",
            "1 logit -> P(onset within 10 min)",
            C.ORANGE,
        ),
    ]
    slot = 10.0 / len(blocks)
    box = slot - 0.62
    for index, (title, detail, color) in enumerate(blocks):
        top = 10.0 - index * slot
        axis.add_patch(
            Rectangle(
                (0.25, top - box),
                9.5,
                box,
                facecolor=color,
                alpha=0.13,
                edgecolor=color,
                linewidth=1.0,
            )
        )
        axis.annotate(
            title,
            xy=(0.6, top - box * 0.34),
            fontsize=9.0,
            color=C.INK,
            fontweight="bold",
            va="center",
        )
        axis.annotate(
            detail,
            xy=(0.6, top - box * 0.72),
            fontsize=8.3,
            color=C.INK_2,
            va="center",
        )
        if index < len(blocks) - 1:
            axis.add_patch(
                FancyArrow(
                    5.0,
                    top - box - 0.05,
                    0,
                    -0.45,
                    width=0.025,
                    head_width=0.20,
                    head_length=0.16,
                    color=C.INK_2,
                    length_includes_head=True,
                )
            )
    axis.set_title("What one decision becomes", loc="left")


def render(atlas: Atlas) -> Path | None:
    positives = [d for d in atlas.exemplars.positives if d.seizure_onset_seconds is not None]
    if not positives:
        return None
    # A mid-amplitude example: the extremes are shown in the gallery figure.
    pick = positives[len(positives) // 2]
    recording = C.open_recording(pick.recording_id, atlas.manifests)
    events = recording.events()
    decision_t = pick.decision_time_seconds
    onset = pick.seizure_onset_seconds
    sop_end = decision_t + C.CONFIG.seizure_occurrence_period_minutes * 60.0

    figure = plt.figure(figsize=(17, 11.4))
    grid = figure.add_gridspec(
        3,
        4,
        height_ratios=[2.5, 0.42, 1.6],
        hspace=0.55,
        wspace=0.26,
        top=0.905,
        bottom=0.055,
        left=0.062,
        right=0.985,
    )

    # ---- history + occurrence period ----------------------------------
    axis = figure.add_subplot(grid[0, :])
    t0 = decision_t - C.CONFIG.input_window_seconds
    t1 = min(sop_end + 120.0, recording.duration_seconds)
    segment = recording.slice(t0, t1)
    present = recording.present_channels
    spacing = 2.6 * C.robust_limits(segment[present], percentile=99.97, pad=1.0)
    relative = lambda t: (t - decision_t) / 60.0  # noqa: E731

    C.draw_signal(
        axis,
        segment,
        t0=t0 - decision_t,
        mask=recording.mask,
        offsets=C.channel_offsets(spacing),
        time_unit="min",
        max_points=5200,
        label_dy=spacing * 0.30,
        label_fontsize=8.8,
        clip=spacing * 0.48,
    )
    C.add_span(
        axis,
        t0 - decision_t,
        0.0,
        "history",
        label=f"THE EVIDENCE — 45-minute input history\n{C.CHUNKS_PER_HISTORY} chunks x "
        f"{C.CONFIG.chunk_window_seconds:.0f} s, all the model ever sees",
        divisor=60.0,
        label_y=0.985,
    )
    C.add_span(
        axis,
        0.0,
        sop_end - decision_t,
        "sop",
        label="THE QUESTION — 10-minute occurrence period\ndoes an onset fall in here?",
        divisor=60.0,
        label_y=0.985,
    )
    C.add_marker(
        axis,
        0.0,
        f"decision instant  t = {C.fmt_hms(decision_t)}",
        color=C.INK,
        divisor=60.0,
        linestyle="--",
        linewidth=1.6,
        ha="right",
        label_y=0.10,
    )
    C.add_marker(
        axis,
        relative(onset) * 60.0,
        f"SEIZURE ONSET  +{(onset - decision_t) / 60:.1f} min",
        divisor=60.0,
        label_y=0.10,
    )
    C.annotate_events(
        axis, events, t0, t1, divisor=60.0, label=False, min_visible_seconds=25.0
    )
    axis.set(
        xlim=(relative(t0), relative(t1)),
        ylim=(-spacing * 1.55, spacing * 1.60),
        xlabel="Time relative to the decision instant (min)",
        ylabel=C.Z_AXIS_OFFSET,
    )
    axis.set_title(
        f"{pick.recording_id}   ·   subject {pick.subject} ({pick.split} split)   ·   "
        f"label = 1   ·   electrodes recorded: {' + '.join(pick.available_channels)}",
        loc="left",
    )
    C.style(axis, grid="x")

    # ---- the chunk strip ----------------------------------------------
    highlights = [0, C.CHUNKS_PER_HISTORY // 2, C.CHUNKS_PER_HISTORY - 1]
    _chunk_strip(figure.add_subplot(grid[1, :]), highlights)

    # ---- three chunks at full resolution -------------------------------
    history = np.asarray(
        recording.array[:, pick.history_start_sample : pick.decision_end_sample],
        dtype=np.float32,
    )
    chunk_spacing = 2.6 * C.robust_limits(history[present], percentile=99.0, pad=1.0)
    for column, index in enumerate(highlights):
        axis = figure.add_subplot(grid[2, column])
        chunk = history[:, index * C.CHUNK_SAMPLES : (index + 1) * C.CHUNK_SAMPLES]
        C.draw_signal(
            axis,
            chunk,
            t0=0.0,
            mask=recording.mask,
            offsets=C.channel_offsets(chunk_spacing),
            time_unit="s",
            linewidth=0.8,
            max_points=4000,
            label_channels=column == 0,
            label_dy=chunk_spacing * 0.28,
            label_fontsize=8.0,
            show_peak=False,
            clip=chunk_spacing * 0.46,
        )
        minutes_before = (C.CHUNKS_PER_HISTORY - index) * C.CONFIG.chunk_window_seconds / 60.0
        axis.set(
            xlim=(0, C.CONFIG.chunk_window_seconds),
            ylim=(-chunk_spacing * 1.5, chunk_spacing * 1.5),
            xlabel="Time within the chunk (s)",
        )
        if column == 0:
            axis.set_ylabel(C.Z_AXIS_OFFSET)
        axis.set_title(
            f"chunk {index}  ·  {minutes_before:.1f} min before the decision\n"
            f"tensor (3, {C.CHUNK_SAMPLES:,})",
            loc="left",
        )
        C.style(axis)

    _tensor_diagram(figure.add_subplot(grid[2, 3]))

    atlas.fact("anatomy_decision", f"{pick.recording_id}#{pick.decision_index_in_shard}")
    atlas.fact("anatomy_lead_seconds", pick.lead_seconds)

    return C.save(
        figure,
        atlas.path(NUMBER, SLUG),
        "One decision, taken apart — 691,200 numbers in, one probability out",
        "Everything left of the dashed line is what the model reads; everything right of it "
        "is what the label asks about. The seizure itself is always on the right.",
        tight=False,
    )
