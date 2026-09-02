r"""Shared plumbing for the dataset atlas figures.

Every ``fNN_*.py`` module in this package imports from here so that twenty
figures do not each reinvent loading, styling, waveform drawing, or event
annotation.  Nothing in this module reads the raw EDFs: the atlas is built from
the manifests plus the stored continuous recordings.

Time base
---------
The stored recordings are *continuous*: ``<rec>.npy`` is ``(3, n_samples)`` and
covers the whole EDF from ``t = 0`` with no trimming, so

    sample_index = round(t_seconds * CONFIG.target_sfreq)

is exact.  Verified: ``sub-001_..._run-03`` has 16,634,624 samples = 64,979 s,
which is exactly the ``recordingDuration`` column of its ``_events.tsv``.  The
same identity makes ``events.tsv`` onsets, ``seizure_manifest.onset_seconds``
and ``decision_manifest.*_sample`` all directly comparable.

Units
-----
``count``    a number of things (files, decisions, seizures, patients)
``h/min/s``  wall-clock duration of EEG
``Hz``       frequency
``z``        dimensionless amplitude; 1 z = one *global* per-channel sigma.  The
             stored shards are z-scored, so nothing downstream is in volts;
             :func:`microvolts_per_z` gives the channel-specific conversion.

Colour
------
Categorical slots 1-4 of the validated reference palette, in fixed order::

    slot 1 blue    #2a78d6   BTE_LEFT   / train / negative
    slot 2 orange  #eb6834   BTE_RIGHT  / validation / positive
    slot 3 aqua    #1baf7a   CROSS_HEAD / test
    slot 4 violet  #4a3aa7   the seizure itself (onset lines, ictal spans)

Those four are the *entire* hue budget.  Checked with the skill's validator
(``validate_palette.py "#2a78d6,#eb6834,#1baf7a,#4a3aa7" --mode light
--surface "#ffffff" --pairs all``): worst all-pairs CVD dE 9.2, worst
normal-vision dE 16.3, both above the gates.  Adding a fifth documented slot
fails: red vs orange measures dE 5.6 under deuteranopia.  So every further
annotation class -- postictal exclusion, impedance checks, bad segments,
dropped candidates -- is drawn as a neutral wash separated by *hatch angle* and
a direct in-plot label, never by a new hue.  Aqua sits at 2.82:1 on white, below
the 3:1 relief threshold, so it always carries a visible direct label too.
"""

from __future__ import annotations

import json
import sys
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIRECTORY = PROJECT_ROOT / "src"
if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

from seizure_prediction.config import CONFIG  # noqa: E402

# ----------------------------------------------------------------------
# Palette
# ----------------------------------------------------------------------

BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#ffffff"

CHANNEL_COLOR = {"BTE_LEFT": BLUE, "BTE_RIGHT": ORANGE, "CROSS_HEAD": AQUA}
SPLIT_COLOR = {"train": BLUE, "validation": ORANGE, "test": AQUA}
LABEL_COLOR = {0: BLUE, 1: ORANGE}
LABEL_NAME = {0: "negative", 1: "positive"}
ABSENT_COLOR = "#c3c2b7"

Z_UNIT = "z  (1 z = 1 global channel sigma)"
Z_AXIS = f"Amplitude, {Z_UNIT}"
# Short form for stacked panels, where the full unit string collides across axes.
Z_AXIS_SHORT = "Amplitude (z)"
Z_AXIS_OFFSET = "Amplitude (z), channels offset"

CHUNKS_PER_HISTORY = int(CONFIG.input_window_seconds / CONFIG.chunk_window_seconds)
CHUNK_SAMPLES = int(CONFIG.chunk_window_seconds * CONFIG.target_sfreq)
HISTORY_SAMPLES = int(CONFIG.input_window_seconds * CONFIG.target_sfreq)

BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 40.0),
}

# Annotation classes drawn on a signal axis.  Only the seizure gets a hue; the
# rest are neutral washes separated by hatch angle plus a direct label.
SPAN_STYLE: dict[str, dict[str, object]] = {
    "seizure": dict(color=VIOLET, alpha=0.20, hatch=None, label="seizure (ictal)"),
    "postictal": dict(color=MUTED, alpha=0.16, hatch="///", label="postictal exclusion"),
    "impedance": dict(color=MUTED, alpha=0.16, hatch="\\\\\\", label="impedance check"),
    "bad": dict(color=MUTED, alpha=0.22, hatch="xxx", label="bad / unusable"),
    "other": dict(color=MUTED, alpha=0.12, hatch="...", label="other annotation"),
    "history": dict(color=BLUE, alpha=0.09, hatch=None, label="45-min input history"),
    "sop": dict(color=ORANGE, alpha=0.14, hatch=None, label="10-min occurrence period"),
    "dropped": dict(color=MUTED, alpha=0.10, hatch="///", label="no decisions"),
}

# events.tsv eventType -> annotation class.  Anything unmatched falls to "other".
EVENT_CLASS_PREFIX = (
    ("sz", "seizure"),
    ("impd", "impedance"),
    ("bad", "bad"),
    ("bckg", "background"),
)

DPI = 150

matplotlib.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "font.size": 10.0,
        "axes.titlesize": 11.5,
        "axes.labelsize": 10.0,
        "xtick.labelsize": 9.0,
        "ytick.labelsize": 9.0,
        "legend.fontsize": 9.0,
        "axes.titlepad": 8.0,
        "hatch.linewidth": 0.6,
        "path.simplify": True,
        "path.simplify_threshold": 1.0,
        "agg.path.chunksize": 20000,
    }
)


# ----------------------------------------------------------------------
# Axis chrome
# ----------------------------------------------------------------------


def style(axis: plt.Axes, grid: str = "both") -> plt.Axes:
    """Apply the house axis chrome: recessive grid, two spines, muted ticks."""
    if grid in ("both", "x", "y"):
        axis.grid(
            True,
            axis=grid if grid != "both" else "both",
            color=GRID,
            linewidth=0.7,
            alpha=0.9,
        )
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color("#c3c2b7")
        axis.spines[side].set_linewidth(0.8)
    axis.tick_params(colors=MUTED, labelcolor=INK_2, length=3, width=0.8)
    axis.title.set_color(INK)
    axis.title.set_fontweight("bold")
    return axis


# Tokens that count as an explicit unit on an axis label.  Every axis in the
# atlas must carry one; save() warns when it does not, which is how the rule
# survives twenty figure modules written independently.
UNIT_TOKENS = (
    "(s)", " s)", "(ms)", "(min)", " min)", "(h)", " h)", "(z", "z)", "(count", "count)", "(hz", "hz)",
    "%", "(db", "db)", "(uv", "uv)", "µv", "rank", "index", "fraction",
    "per hour", "per minute", "per patient", "probability", "log", "(1/",
)


def check_units(figure: plt.Figure, name: str) -> list[str]:
    """Warn about axis labels that carry no explicit unit.  Never raises."""
    problems: list[str] = []
    for axis in figure.get_axes():
        if not axis.get_visible():
            continue
        for which, label in (("x", axis.get_xlabel()), ("y", axis.get_ylabel())):
            text = label.strip().lower()
            if not text:
                continue
            if not any(token in text for token in UNIT_TOKENS):
                problems.append(f"{name}: {which} label has no unit -> {label!r}")
    return problems


def save(
    figure: plt.Figure,
    path: Path,
    title: str,
    subtitle: str | None = None,
    rect: tuple[float, float, float, float] | None = None,
    tight: bool = True,
    bbox: str | None = "tight",
) -> Path:
    """Title, lay out and write one figure.  Returns the path written.

    ``tight=False`` is for figures that set their own margins with
    ``subplots_adjust``; ``tight_layout`` would undo them.
    """
    top = 0.985
    figure.suptitle(title, fontsize=14.5, y=top, color=INK, fontweight="bold")
    if subtitle:
        figure.text(
            0.5,
            top - 0.030,
            subtitle,
            ha="center",
            va="top",
            fontsize=10.5,
            color=INK_2,
        )
    for problem in check_units(figure, path.stem):
        print(f"  UNITS? {problem}", flush=True)
    if tight:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*not compatible with tight_layout.*")
            figure.tight_layout(rect=rect or (0, 0, 1, 0.955 if subtitle else 0.975))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=DPI, bbox_inches=bbox, facecolor=SURFACE)
    plt.close(figure)
    print(f"  wrote {path.name}", flush=True)
    return path


def fmt_hms(seconds: float) -> str:
    """Seconds -> ``h:mm:ss`` for wall-clock positions inside a recording."""
    seconds = int(round(seconds))
    return f"{seconds // 3600:d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


def fmt_duration(seconds: float) -> str:
    """Seconds -> a compact human duration with an explicit unit."""
    if seconds < 10:
        return f"{seconds:.3g} s"
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 5400:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.1f} h"


# ----------------------------------------------------------------------
# Manifests and scaler
# ----------------------------------------------------------------------


@dataclass
class Manifests:
    """The three manifest tables, loaded once and shared by every figure."""

    decisions: pd.DataFrame
    seizures: pd.DataFrame
    shards: pd.DataFrame

    def __post_init__(self) -> None:
        self._shard_by_id = self.shards.set_index("recording_id")

    @property
    def shard_by_id(self) -> pd.DataFrame:
        return self._shard_by_id

    def shard(self, recording_id: str) -> pd.Series:
        return self._shard_by_id.loc[recording_id]

    def has(self, recording_id: str) -> bool:
        return recording_id in self._shard_by_id.index


@lru_cache(maxsize=1)
def load_manifests() -> Manifests:
    """Load decision / seizure / shard manifests with stable dtypes."""
    root = CONFIG.manifests_dir
    decisions = pd.read_csv(
        root / "decision_manifest.csv",
        dtype={
            "subject": str,
            "session": str,
            "task": str,
            "run": str,
            "target_seizure_id": str,
        },
    )
    seizures = pd.read_csv(
        root / "seizure_manifest.csv",
        dtype={"subject": str, "session": str, "run": str},
    )
    shards = pd.read_csv(
        root / "processed_shard_manifest.csv", dtype={"subject": str}
    )
    seizures["eligible"] = (
        seizures["eligible_for_prediction"].astype(str).str.lower().eq("true")
    )
    return Manifests(decisions, seizures, shards)


@lru_cache(maxsize=1)
def microvolts_per_z() -> dict[str, float]:
    """Return the global per-channel sigma in microvolts (1 z = this many uV)."""
    path = CONFIG.scaler_parameters_dir / "global_channel_zscore.json"
    if not path.exists():
        return {}
    scaler = json.loads(path.read_text(encoding="utf-8"))
    return {
        name: float(std) * 1e6
        for name, std in zip(scaler["channel_names"], scaler["std"], strict=True)
    }


def channel_legend_label(channel: str, present: bool = True) -> str:
    """``BTE_LEFT  (1 z = 121 uV)`` -- the label drawn beside each trace."""
    micro = microvolts_per_z().get(channel)
    suffix = "" if micro is None else f"   1 z = {micro:.0f} uV"
    if not present:
        return f"{channel}   ABSENT -> zero-filled{suffix and '  ' + suffix}"
    return f"{channel}{suffix}"


# ----------------------------------------------------------------------
# Recordings
# ----------------------------------------------------------------------


@dataclass
class Recording:
    """One continuous stored recording, opened lazily as a memory map."""

    recording_id: str
    subject: str
    split: str
    array: np.ndarray  # (3, n_samples) memmap, float32, z units
    mask: np.ndarray  # (3,) bool, True where the electrode was recorded
    sfreq: float

    @property
    def n_samples(self) -> int:
        return int(self.array.shape[1])

    @property
    def duration_seconds(self) -> float:
        return self.n_samples / self.sfreq

    @property
    def present_channels(self) -> list[int]:
        return [c for c in range(3) if bool(self.mask[c])]

    def sample(self, t_seconds: float) -> int:
        return int(round(t_seconds * self.sfreq))

    def slice(self, t0: float, t1: float) -> np.ndarray:
        """Return ``(3, n)`` in z units for ``[t0, t1)`` seconds, clipped in range."""
        a = max(0, self.sample(t0))
        b = min(self.n_samples, self.sample(t1))
        if b <= a:
            return np.zeros((3, 0), dtype=np.float32)
        return np.asarray(self.array[:, a:b], dtype=np.float32)

    def events(self) -> pd.DataFrame:
        return read_events(self.recording_id)


def open_recording(recording_id: str, manifests: Manifests | None = None) -> Recording:
    """Memory-map the standardized shard for one recording."""
    manifests = manifests or load_manifests()
    row = manifests.shard(recording_id)
    array = np.load(row["X_path"], mmap_mode="r")
    with open(row["channel_availability_path"], encoding="utf-8") as handle:
        mask = np.asarray(json.load(handle), dtype=bool)
    return Recording(
        recording_id=recording_id,
        subject=str(row["subject"]),
        split=str(row["split"]),
        array=array,
        mask=mask,
        sfreq=CONFIG.target_sfreq,
    )


def events_path(recording_id: str) -> Path:
    """``sub-001_ses-01_task-szMonitoring_run-03`` -> its raw ``_events.tsv``."""
    subject = recording_id.split("_")[0]
    session = recording_id.split("_")[1]
    return CONFIG.raw_data_dir / subject / session / "eeg" / f"{recording_id}_events.tsv"


@lru_cache(maxsize=512)
def read_events(recording_id: str) -> pd.DataFrame:
    """Raw annotations for one recording, with a normalized ``span_class`` column.

    Columns: onset, duration, eventType, lateralization, localization, vigilance,
    confidence, channels, dateTime, recordingDuration, plus ``offset`` and
    ``span_class`` (seizure / impedance / bad / background / other).
    """
    path = events_path(recording_id)
    if not path.exists():
        return pd.DataFrame(
            columns=["onset", "duration", "eventType", "offset", "span_class"]
        )
    table = pd.read_csv(path, sep="\t")
    table["onset"] = pd.to_numeric(table["onset"], errors="coerce")
    table["duration"] = pd.to_numeric(table["duration"], errors="coerce").fillna(0.0)
    table = table.dropna(subset=["onset"]).copy()
    table["offset"] = table["onset"] + table["duration"]
    kind = table["eventType"].astype(str).str.strip().str.lower()
    span_class = pd.Series("other", index=table.index, dtype=object)
    for prefix, name in EVENT_CLASS_PREFIX:
        span_class = span_class.mask(kind.str.startswith(prefix), name)
    table["span_class"] = span_class
    return table.sort_values("onset").reset_index(drop=True)


# ----------------------------------------------------------------------
# Waveform drawing
# ----------------------------------------------------------------------


def envelope(signal: np.ndarray, columns: int) -> tuple[np.ndarray, np.ndarray]:
    """Min/max decimate a 1-D signal to ``columns`` pixel columns.

    Plotting 5 million points is both slow and a lie -- matplotlib drops
    samples arbitrarily.  A per-column min/max envelope keeps every transient
    visible at any zoom level.  Returns ``(lo, hi)``, each length ``columns``.
    """
    signal = np.asarray(signal, dtype=np.float32)
    if signal.size == 0:
        return np.zeros(columns), np.zeros(columns)
    columns = max(1, min(columns, signal.size))
    step = signal.size // columns
    if step < 2:
        # Fewer than two samples per column: decimating would drop the tail.
        return signal.copy(), signal.copy()
    trimmed = signal[: step * columns].reshape(columns, step)
    return trimmed.min(axis=1), trimmed.max(axis=1)


def percentile_envelope(
    signal: np.ndarray, columns: int, band: tuple[float, float] = (2.0, 98.0)
) -> tuple[np.ndarray, np.ndarray]:
    """Per-column percentile band -- the body of the signal without its spikes."""
    signal = np.asarray(signal, dtype=np.float32)
    if signal.size == 0:
        return np.zeros(columns), np.zeros(columns)
    columns = max(1, min(columns, signal.size))
    step = signal.size // columns
    if step < 2:
        return signal.copy(), signal.copy()
    trimmed = signal[: step * columns].reshape(columns, step)
    lo, hi = np.percentile(trimmed, band, axis=1)
    return lo, hi


def draw_signal(
    axis: plt.Axes,
    data: np.ndarray,
    *,
    t0: float = 0.0,
    sfreq: float | None = None,
    mask: np.ndarray | None = None,
    offsets: tuple[float, ...] | np.ndarray = (8.0, 0.0, -8.0),
    time_unit: str = "s",
    colors: dict[str, str] | str | None = None,
    linewidth: float = 0.5,
    max_points: int = 3000,
    channel_names: tuple[str, ...] | None = None,
    label_channels: bool = True,
    label_x: float = 0.004,
    label_dy: float = 2.6,
    label_fontsize: float = 8.5,
    show_peak: bool = True,
    band: tuple[float, float] | None = (2.0, 98.0),
    clip: float | None = None,
    zorder: int = 3,
) -> dict[str, float]:
    """Draw a multi-channel z-scored segment with per-channel vertical offsets.

    Long segments are drawn as min/max envelopes so no transient is lost.  A
    pure min/max band over a whole night is a solid block, though, so a second
    inner fill at ``band`` percentiles carries the body of the signal while the
    outer min/max tint keeps the transients visible.  Returns
    ``{channel: peak_abs_z}``.

    ``time_unit`` scales the x axis: ``s``, ``min`` or ``h``.
    """
    sfreq = sfreq or CONFIG.target_sfreq
    names = channel_names or CONFIG.canonical_channel_names
    mask = np.ones(len(names), dtype=bool) if mask is None else np.asarray(mask, bool)
    divisor = {"s": 1.0, "min": 60.0, "h": 3600.0}[time_unit]
    n = data.shape[1]
    if n == 0:
        return {}

    peaks: dict[str, float] = {}
    for index, (name, offset) in enumerate(zip(names, offsets, strict=False)):
        trace = np.asarray(data[index], dtype=np.float32)
        present = bool(mask[index])
        if isinstance(colors, str):
            color = colors
        elif isinstance(colors, dict):
            color = colors.get(name, CHANNEL_COLOR.get(name, BLUE))
        else:
            color = CHANNEL_COLOR.get(name, BLUE)
        if not present:
            color = ABSENT_COLOR
        peaks[name] = float(np.abs(trace).max()) if trace.size else 0.0
        if clip is not None:
            trace = np.clip(trace, -clip, clip)

        if n > max_points:
            lo, hi = envelope(trace, max_points)
            edges = t0 + np.arange(lo.size) * (n / lo.size) / sfreq
            axis.fill_between(
                edges / divisor,
                lo + offset,
                hi + offset,
                color=color,
                linewidth=0.0,
                alpha=(0.32 if band else 0.95) if present else 0.55,
                zorder=zorder,
            )
            if band is not None:
                inner_lo, inner_hi = percentile_envelope(trace, lo.size, band)
                axis.fill_between(
                    edges / divisor,
                    inner_lo + offset,
                    inner_hi + offset,
                    color=color,
                    linewidth=0.0,
                    alpha=0.95 if present else 0.75,
                    zorder=zorder + 1,
                )
        else:
            times = (t0 + np.arange(n) / sfreq) / divisor
            axis.plot(
                times,
                trace + offset,
                color=color,
                linewidth=linewidth,
                alpha=0.95 if present else 0.8,
                solid_capstyle="round",
                zorder=zorder,
            )

        if label_channels:
            text = channel_legend_label(name, present)
            if show_peak and present:
                text = f"{text}     peak |x| = {peaks[name]:.1f} z"
                if clip is not None and peaks[name] > clip:
                    text = f"{text}  (clipped at {clip:.0f} z)"
            axis.annotate(
                text,
                xy=(label_x, offset + label_dy),
                xycoords=("axes fraction", "data"),
                fontsize=label_fontsize,
                color=INK_2 if present else MUTED,
                fontweight="bold" if present else "normal",
                zorder=zorder + 2,
                bbox=dict(
                    boxstyle="round,pad=0.18",
                    facecolor=SURFACE,
                    edgecolor="none",
                    alpha=0.78,
                ),
            )
    return peaks


def channel_offsets(spacing: float, n: int = 3) -> tuple[float, ...]:
    """Evenly spaced vertical offsets, first channel on top."""
    return tuple(spacing * (n - 1 - 2 * i) / 2 for i in range(n))


def add_span(
    axis: plt.Axes,
    t0: float,
    t1: float,
    kind: str,
    *,
    label: str | None = None,
    divisor: float = 1.0,
    label_y: float = 0.965,
    fontsize: float = 8.5,
    edge: bool = True,
    zorder: int = 1,
) -> None:
    """Shade ``[t0, t1)`` as one annotation class, with a direct in-plot label.

    Hue alone never carries the class: everything except the seizure is a
    neutral wash distinguished by hatch angle, and every span is labelled.
    """
    spec = SPAN_STYLE.get(kind, SPAN_STYLE["other"])
    width = (t1 - t0) / divisor
    if width <= 0:
        return
    axis.axvspan(
        t0 / divisor,
        t1 / divisor,
        facecolor=spec["color"],
        alpha=float(spec["alpha"]),
        hatch=spec["hatch"],
        edgecolor=spec["color"] if spec["hatch"] else "none",
        linewidth=0.0,
        zorder=zorder,
    )
    if edge:
        for x in (t0 / divisor, t1 / divisor):
            axis.axvline(x, color=spec["color"], linewidth=0.8, alpha=0.65, zorder=zorder + 1)
    text = label if label is not None else str(spec["label"])
    if text:
        axis.annotate(
            text,
            xy=((t0 + t1) / 2 / divisor, label_y),
            xycoords=("data", "axes fraction"),
            ha="center",
            va="top",
            fontsize=fontsize,
            color=INK_2,
            zorder=zorder + 6,
            bbox=dict(
                boxstyle="round,pad=0.22",
                facecolor=SURFACE,
                edgecolor=spec["color"],
                linewidth=0.7,
                alpha=0.92,
            ),
        )


def add_marker(
    axis: plt.Axes,
    t: float,
    label: str,
    *,
    color: str = VIOLET,
    divisor: float = 1.0,
    label_y: float = 0.06,
    linestyle: str = "-",
    linewidth: float = 1.6,
    ha: str = "left",
    fontsize: float = 8.8,
    zorder: int = 8,
) -> None:
    """A labelled vertical landmark: seizure onset, decision instant, and so on."""
    axis.axvline(
        t / divisor, color=color, linewidth=linewidth, linestyle=linestyle, zorder=zorder
    )
    axis.annotate(
        label,
        xy=(t / divisor, label_y),
        xycoords=("data", "axes fraction"),
        xytext=(4 if ha == "left" else -4, 0),
        textcoords="offset points",
        ha=ha,
        va="bottom",
        fontsize=fontsize,
        color=color,
        fontweight="bold",
        zorder=zorder + 1,
        bbox=dict(
            boxstyle="round,pad=0.22",
            facecolor=SURFACE,
            edgecolor=color,
            linewidth=0.7,
            alpha=0.92,
        ),
    )


def annotate_events(
    axis: plt.Axes,
    events: pd.DataFrame,
    t0: float,
    t1: float,
    *,
    divisor: float = 1.0,
    include_background: bool = False,
    label: bool = True,
    label_y: float = 0.965,
    min_visible_seconds: float = 0.0,
) -> list[str]:
    """Shade every annotation from ``events`` that overlaps ``[t0, t1]``.

    Returns the list of annotation classes actually drawn, for legend building.
    """
    drawn: list[str] = []
    if events.empty:
        return drawn
    window = events[(events["offset"] >= t0) & (events["onset"] <= t1)]
    for row in window.itertuples(index=False):
        kind = row.span_class
        if kind == "background" and not include_background:
            continue
        start, stop = max(float(row.onset), t0), min(float(row.offset), t1)
        if stop - start < min_visible_seconds:
            middle = (start + stop) / 2
            start, stop = middle - min_visible_seconds / 2, middle + min_visible_seconds / 2
        text = f"{row.eventType}  {fmt_duration(float(row.duration))}" if label else ""
        add_span(
            axis,
            start,
            stop,
            kind if kind != "background" else "other",
            label=text,
            divisor=divisor,
            label_y=label_y,
        )
        drawn.append(kind)
    return drawn


def span_legend(
    axis: plt.Axes,
    kinds: list[str],
    *,
    extra: list[tuple[str, str]] | None = None,
    loc: str = "upper right",
    title: str | None = None,
    ncol: int = 1,
) -> None:
    """A legend for the annotation classes present, matching hatch and hue."""
    handles: list[object] = []
    seen: set[str] = set()
    for kind in kinds:
        if kind in seen or kind not in SPAN_STYLE:
            continue
        seen.add(kind)
        spec = SPAN_STYLE[kind]
        handles.append(
            Rectangle(
                (0, 0),
                1,
                1,
                facecolor=spec["color"],
                alpha=float(spec["alpha"]) * 2.2,
                hatch=spec["hatch"],
                edgecolor=spec["color"],
                linewidth=0.7,
                label=str(spec["label"]),
            )
        )
    for color, text in extra or []:
        handles.append(Line2D([0], [0], color=color, linewidth=2.0, label=text))
    if not handles:
        return
    legend = axis.legend(
        handles=handles,
        loc=loc,
        frameon=True,
        framealpha=0.94,
        edgecolor=GRID,
        fontsize=8.5,
        ncol=ncol,
        title=title,
        title_fontsize=8.5,
    )
    legend.get_frame().set_linewidth(0.7)


def add_scale_bar(
    axis: plt.Axes,
    *,
    x: float,
    y: float,
    seconds: float,
    z: float,
    divisor: float = 1.0,
    color: str = INK_2,
    fontsize: float = 8.5,
) -> None:
    """An L-shaped time/amplitude scale bar anchored at data coordinates."""
    width = seconds / divisor
    axis.plot([x, x + width], [y, y], color=color, linewidth=1.6, zorder=9)
    axis.plot([x, x], [y, y + z], color=color, linewidth=1.6, zorder=9)
    axis.annotate(
        f"{fmt_duration(seconds)}",
        xy=(x + width / 2, y),
        xytext=(0, -3),
        textcoords="offset points",
        ha="center",
        va="top",
        fontsize=fontsize,
        color=color,
        zorder=9,
    )
    axis.annotate(
        f"{z:g} z",
        xy=(x, y + z / 2),
        xytext=(-4, 0),
        textcoords="offset points",
        ha="right",
        va="center",
        fontsize=fontsize,
        color=color,
        zorder=9,
    )


# ----------------------------------------------------------------------
# Decision geometry
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class DecisionGeometry:
    """Every timing landmark of one decision, in seconds from recording start."""

    history_start: float
    decision_time: float
    prediction_start: float
    prediction_stop: float
    label: int
    target_seizure_id: str | None
    recording_id: str
    subject: str
    split: str
    onset: float | None = None  # target seizure onset, when there is one
    onset_duration: float | None = None

    @property
    def lead_seconds(self) -> float | None:
        """Seconds from the decision instant to the target seizure onset."""
        return None if self.onset is None else self.onset - self.decision_time


def decision_geometry(row: pd.Series, manifests: Manifests | None = None) -> DecisionGeometry:
    """Build a :class:`DecisionGeometry` from one ``decision_manifest`` row."""
    manifests = manifests or load_manifests()
    target = row.get("target_seizure_id")
    onset = duration = None
    if isinstance(target, str) and target:
        match = manifests.seizures[manifests.seizures["seizure_id"] == target]
        if not match.empty:
            onset = float(match.iloc[0]["onset_seconds"])
            duration = float(match.iloc[0]["duration_seconds"])
    return DecisionGeometry(
        history_start=float(row["history_start_seconds"]),
        decision_time=float(row["decision_time_seconds"]),
        prediction_start=float(row["prediction_start_seconds"]),
        prediction_stop=float(row["prediction_stop_seconds"]),
        label=int(row["label"]),
        target_seizure_id=target if isinstance(target, str) and target else None,
        recording_id=str(row["recording_id"]),
        subject=str(row["subject"]),
        split=str(row["split"]),
        onset=onset,
        onset_duration=duration,
    )


def annotate_decision(
    axis: plt.Axes,
    geometry: DecisionGeometry,
    *,
    divisor: float = 60.0,
    show_history: bool = True,
    show_sop: bool = True,
    label_y: float = 0.965,
) -> list[str]:
    """Draw the labelling geometry of one decision onto a time axis."""
    drawn: list[str] = []
    if show_history:
        add_span(
            axis,
            geometry.history_start,
            geometry.decision_time,
            "history",
            label=f"45-min input history\n{CHUNKS_PER_HISTORY} chunks x {CONFIG.chunk_window_seconds:.0f} s",
            divisor=divisor,
            label_y=label_y,
        )
        drawn.append("history")
    if show_sop:
        add_span(
            axis,
            geometry.prediction_start,
            geometry.prediction_stop,
            "sop",
            label="10-min seizure occurrence period\nthe question being asked",
            divisor=divisor,
            label_y=label_y,
        )
        drawn.append("sop")
    add_marker(
        axis,
        geometry.decision_time,
        f"decision t = {fmt_hms(geometry.decision_time)}",
        color=INK,
        divisor=divisor,
        linestyle="--",
        linewidth=1.4,
        ha="right",
    )
    if geometry.onset is not None:
        add_marker(
            axis,
            geometry.onset,
            f"SEIZURE ONSET  (+{geometry.lead_seconds / 60:.1f} min)",
            color=VIOLET,
            divisor=divisor,
        )
    return drawn


# ----------------------------------------------------------------------
# Sampling helpers
# ----------------------------------------------------------------------


def usable_decisions(
    manifests: Manifests,
    *,
    label: int | None = None,
    split: str | None = None,
    one_per_subject: bool = False,
    limit: int | None = None,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Decisions whose shard exists, optionally one per subject, shuffled."""
    table = manifests.decisions
    if label is not None:
        table = table[table["label"] == label]
    if split is not None:
        table = table[table["split"] == split]
    table = table[table["recording_id"].isin(manifests.shard_by_id.index)]
    if rng is not None and len(table):
        table = table.sample(frac=1.0, random_state=int(rng.integers(1 << 30)))
    if one_per_subject:
        table = table.drop_duplicates(subset="subject")
    if limit is not None:
        table = table.head(limit)
    return table.reset_index(drop=True)


def history_of(row: pd.Series, manifests: Manifests | None = None) -> tuple[np.ndarray, np.ndarray] | None:
    """Return ``(history, mask)`` for one decision row, or None if out of range."""
    manifests = manifests or load_manifests()
    recording_id = str(row["recording_id"])
    if not manifests.has(recording_id):
        return None
    recording = open_recording(recording_id, manifests)
    start = int(row["history_start_sample"])
    stop = int(row["decision_end_sample"])
    if stop > recording.n_samples or start < 0:
        return None
    return np.asarray(recording.array[:, start:stop], dtype=np.float32), recording.mask


def nice_number(value: float) -> float:
    """Round to the nearest 1/2/5 x 10^n, so scale bars carry readable numbers."""
    if not np.isfinite(value) or value <= 0:
        return 1.0
    exponent = np.floor(np.log10(value))
    mantissa = value / 10.0**exponent
    for candidate in (1.0, 2.0, 5.0, 10.0):
        if mantissa <= candidate * 1.5:
            return float(candidate * 10.0**exponent)
    return float(10.0 ** (exponent + 1))


def robust_limits(data: np.ndarray, percentile: float = 99.5, pad: float = 1.25) -> float:
    """A symmetric y-limit that ignores the worst transients."""
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return 1.0
    return float(max(np.percentile(np.abs(finite), percentile) * pad, 0.5))


__all__ = [name for name in dir() if not name.startswith("_")]
