r"""Choose, once per run, the concrete recordings/seizures/decisions to plot.

Every signal figure needs examples.  Picking them independently in each figure
module would mean the atlas shows a different seizure in every panel and cannot
be reproduced.  This module scores real candidates from the manifests, verifies
each choice by actually loading the samples it proposes, and caches the result
as JSON next to the figures so the whole atlas talks about the same examples.

Nothing here is hard-coded to a subject: the selection is a deterministic
function of the manifests plus ``--seed``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .common import (
    CONFIG,
    HISTORY_SAMPLES,
    Manifests,
    load_manifests,
    open_recording,
    read_events,
)

SFREQ = CONFIG.target_sfreq


# ----------------------------------------------------------------------
# Records
# ----------------------------------------------------------------------


@dataclass
class SeizureExemplar:
    """One eligible seizure verified to sit inside its stored recording."""

    seizure_id: str
    recording_id: str
    subject: str
    split: str
    onset_seconds: float
    duration_seconds: float
    event_type: str
    vigilance: str
    lateralization: str
    localization: str
    available_channels: list[str]
    ictal_mean_abs_z: float
    baseline_mean_abs_z: float
    contrast: float
    recording_duration_seconds: float
    n_positive_decisions: int


@dataclass
class DecisionExemplar:
    """One decision whose full 45-minute history is inside the stored array."""

    recording_id: str
    subject: str
    split: str
    label: int
    decision_index_in_shard: int
    decision_time_seconds: float
    history_start_sample: int
    decision_end_sample: int
    target_seizure_id: str | None
    seizure_onset_seconds: float | None
    lead_seconds: float | None
    available_channels: list[str]
    peak_abs_z: float
    median_abs_z: float


@dataclass
class RecordingExemplar:
    """One whole recording chosen for the end-to-end timeline figure."""

    recording_id: str
    subject: str
    split: str
    duration_seconds: float
    n_decisions: int
    n_positive_decisions: int
    n_seizures: int
    n_eligible_seizures: int
    n_other_events: int
    event_types: list[str]
    available_channels: list[str]
    score: float


@dataclass
class Exemplars:
    """Everything the signal figures plot, chosen once and reused everywhere."""

    hero_recordings: list[RecordingExemplar] = field(default_factory=list)
    seizures: list[SeizureExemplar] = field(default_factory=list)
    positives: list[DecisionExemplar] = field(default_factory=list)
    negatives: list[DecisionExemplar] = field(default_factory=list)
    amplitude_extremes: dict[str, dict] = field(default_factory=dict)

    @property
    def hero(self) -> RecordingExemplar:
        return self.hero_recordings[0]

    def to_json(self) -> str:
        return json.dumps(
            {
                "hero_recordings": [asdict(r) for r in self.hero_recordings],
                "seizures": [asdict(s) for s in self.seizures],
                "positives": [asdict(d) for d in self.positives],
                "negatives": [asdict(d) for d in self.negatives],
                "amplitude_extremes": self.amplitude_extremes,
            },
            indent=2,
        )

    @classmethod
    def from_json(cls, text: str) -> "Exemplars":
        raw = json.loads(text)
        return cls(
            hero_recordings=[RecordingExemplar(**r) for r in raw["hero_recordings"]],
            seizures=[SeizureExemplar(**s) for s in raw["seizures"]],
            positives=[DecisionExemplar(**d) for d in raw["positives"]],
            negatives=[DecisionExemplar(**d) for d in raw["negatives"]],
            amplitude_extremes=raw.get("amplitude_extremes", {}),
        )


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _channel_names(mask: np.ndarray) -> list[str]:
    return [
        name
        for name, present in zip(CONFIG.canonical_channel_names, mask, strict=True)
        if bool(present)
    ]


def _mean_abs(recording, t0: float, t1: float) -> float:
    """Mean |z| over the available channels of one time span."""
    segment = recording.slice(t0, t1)
    if segment.shape[1] == 0:
        return float("nan")
    present = recording.present_channels
    if not present:
        return float("nan")
    return float(np.abs(segment[present]).mean())


# ----------------------------------------------------------------------
# Selection
# ----------------------------------------------------------------------


def rank_hero_recordings(manifests: Manifests, top: int = 5) -> list[RecordingExemplar]:
    """Recordings that show the whole labelling story in one timeline.

    Wanted: long enough to see structure, at least one eligible seizure that is
    not at the very start, some non-seizure annotation to shade, and a healthy
    number of decisions including positives.
    """
    decisions = manifests.decisions
    per_recording = decisions.groupby("recording_id").agg(
        n_decisions=("label", "size"),
        n_positive=("label", "sum"),
        subject=("subject", "first"),
        split=("split", "first"),
    )
    seizures = manifests.seizures.copy()
    seizures["recording_id"] = (
        "sub-" + seizures["subject"]
        + "_ses-" + seizures["session"].astype(str).str.zfill(2)
        + "_task-" + CONFIG.bids_task
        + "_run-" + seizures["run"].astype(str).str.zfill(2)
    )

    candidates: list[RecordingExemplar] = []
    for recording_id, row in per_recording.iterrows():
        if int(row["n_positive"]) < 5 or not manifests.has(recording_id):
            continue
        mine = seizures[seizures["recording_id"] == recording_id]
        if mine.empty or not bool(mine["eligible"].any()):
            continue
        events = read_events(recording_id)
        if events.empty:
            continue
        other = events[~events["span_class"].isin(["seizure", "background"])]
        shard = manifests.shard(recording_id)
        array = np.load(shard["X_path"], mmap_mode="r")
        duration = float(array.shape[1] / SFREQ)
        if duration < 4 * 3600:
            continue
        first_onset = float(mine["onset_seconds"].min())
        if first_onset < 1.5 * 3600:  # want history visible before the first seizure
            continue
        with open(shard["channel_availability_path"], encoding="utf-8") as handle:
            mask = np.asarray(json.load(handle), dtype=bool)

        score = (
            min(duration / 3600.0, 24.0) * 1.0
            + int(mine["eligible"].sum()) * 6.0
            + min(len(other), 6) * 4.0
            + min(int(row["n_positive"]), 40) * 0.4
            + min(int(row["n_decisions"]) / 100.0, 6.0)
        )
        candidates.append(
            RecordingExemplar(
                recording_id=str(recording_id),
                subject=str(row["subject"]),
                split=str(row["split"]),
                duration_seconds=duration,
                n_decisions=int(row["n_decisions"]),
                n_positive_decisions=int(row["n_positive"]),
                n_seizures=int(len(mine)),
                n_eligible_seizures=int(mine["eligible"].sum()),
                n_other_events=int(len(other)),
                event_types=sorted(events["eventType"].astype(str).unique().tolist()),
                available_channels=_channel_names(mask),
                score=float(score),
            )
        )

    candidates.sort(key=lambda c: -c.score)
    return candidates[:top]


def rank_seizures(
    manifests: Manifests,
    top: int = 8,
    max_scan: int = 340,
    baseline_offset_seconds: float = 1800.0,
) -> list[SeizureExemplar]:
    """Eligible seizures ranked by how visible the ictal change is.

    Contrast is ``mean |z| during the ictal span / mean |z| in an equally long
    span 30 minutes earlier`` on the available channels of the same recording,
    which keeps the comparison within one patient and one montage.

    Ranking by contrast deliberately selects the visible tail, not the typical
    seizure: measured over all 317 eligible seizures the median ictal window is
    about 2.1x the interictal baseline, and on roughly 18 % of channel
    observations it is no larger at all (a focal seizure is often invisible on
    the contralateral behind-the-ear pair).  Figures that plot these exemplars
    must say so rather than implying seizures usually look like this.
    """
    seizures = manifests.seizures[manifests.seizures["eligible"]].copy()
    seizures["recording_id"] = (
        "sub-" + seizures["subject"]
        + "_ses-" + seizures["session"].astype(str).str.zfill(2)
        + "_task-" + CONFIG.bids_task
        + "_run-" + seizures["run"].astype(str).str.zfill(2)
    )
    positives = manifests.decisions[manifests.decisions["label"] == 1]
    per_target = positives.groupby("target_seizure_id").size()

    found: list[SeizureExemplar] = []
    for row in seizures.head(max_scan).itertuples(index=False):
        recording_id = row.recording_id
        if not manifests.has(recording_id):
            continue
        recording = open_recording(recording_id, manifests)
        onset = float(row.onset_seconds)
        duration = max(float(row.duration_seconds), 10.0)
        if recording.sample(onset + duration) > recording.n_samples:
            continue
        if onset - baseline_offset_seconds - duration < 0:
            continue
        ictal = _mean_abs(recording, onset, onset + duration)
        baseline = _mean_abs(
            recording,
            onset - baseline_offset_seconds - duration,
            onset - baseline_offset_seconds,
        )
        # Guard against dividing by a dead or disconnected baseline segment.
        if not np.isfinite(ictal) or not np.isfinite(baseline) or baseline < 5e-3:
            continue
        found.append(
            SeizureExemplar(
                seizure_id=str(row.seizure_id),
                recording_id=recording_id,
                subject=str(row.subject),
                split=recording.split,
                onset_seconds=onset,
                duration_seconds=float(row.duration_seconds),
                event_type=str(row.event_type),
                vigilance=str(row.vigilance),
                lateralization=str(row.lateralization),
                localization=str(row.localization),
                available_channels=_channel_names(recording.mask),
                ictal_mean_abs_z=float(ictal),
                baseline_mean_abs_z=float(baseline),
                contrast=float(ictal / baseline),
                recording_duration_seconds=recording.duration_seconds,
                n_positive_decisions=int(per_target.get(str(row.seizure_id), 0)),
            )
        )

    found.sort(key=lambda s: -s.contrast)
    # One per patient, so the gallery is not eight views of the same person.
    seen: set[str] = set()
    unique: list[SeizureExemplar] = []
    for item in found:
        if item.subject in seen:
            continue
        seen.add(item.subject)
        unique.append(item)
        if len(unique) >= top:
            break
    return unique


def rank_decisions(
    manifests: Manifests,
    label: int,
    top: int = 8,
    rng: np.random.Generator | None = None,
    max_scan: int = 260,
) -> list[DecisionExemplar]:
    """Decisions from distinct patients spanning the amplitude range.

    The gallery is only honest if it shows the quiet recordings and the noisy
    ones, so candidates are scanned, measured, and then sampled across the
    quantiles of their own median |z| rather than taken at random.
    """
    rng = rng or np.random.default_rng(0)
    table = manifests.decisions
    table = table[(table["label"] == label)]
    table = table[table["recording_id"].isin(manifests.shard_by_id.index)]
    table = table.sample(frac=1.0, random_state=int(rng.integers(1 << 30)))
    table = table.drop_duplicates(subset="subject").head(max_scan)

    seizure_by_id = manifests.seizures.set_index("seizure_id")
    measured: list[DecisionExemplar] = []
    for row in table.itertuples(index=False):
        recording = open_recording(str(row.recording_id), manifests)
        start, stop = int(row.history_start_sample), int(row.decision_end_sample)
        if start < 0 or stop > recording.n_samples or stop - start != HISTORY_SAMPLES:
            continue
        present = recording.present_channels
        if not present:
            continue
        # Read a strided view: 1 in 64 samples is plenty for a summary statistic.
        block = np.asarray(recording.array[:, start:stop:64], dtype=np.float32)
        if not np.isfinite(block).all():
            continue
        target = getattr(row, "target_seizure_id", None)
        onset = None
        if isinstance(target, str) and target and target in seizure_by_id.index:
            onset = float(seizure_by_id.loc[target, "onset_seconds"])
        measured.append(
            DecisionExemplar(
                recording_id=str(row.recording_id),
                subject=str(row.subject),
                split=str(row.split),
                label=int(row.label),
                decision_index_in_shard=int(row.decision_index_in_shard),
                decision_time_seconds=float(row.decision_time_seconds),
                history_start_sample=start,
                decision_end_sample=stop,
                target_seizure_id=target if isinstance(target, str) and target else None,
                seizure_onset_seconds=onset,
                lead_seconds=(
                    None if onset is None else onset - float(row.decision_time_seconds)
                ),
                available_channels=_channel_names(recording.mask),
                peak_abs_z=float(np.abs(block[present]).max()),
                median_abs_z=float(np.median(np.abs(block[present]))),
            )
        )

    if not measured:
        return []
    measured.sort(key=lambda d: d.median_abs_z)
    # Even spread across the measured amplitude range, quietest to noisiest.
    picks = np.linspace(0, len(measured) - 1, num=min(top, len(measured)))
    return [measured[int(round(i))] for i in picks]


def find_amplitude_extremes(
    manifests: Manifests,
    n_sample: int = 260,
    rng: np.random.Generator | None = None,
) -> dict[str, dict]:
    """The quietest and loudest recordings in a random sample, with a clean slice."""
    rng = rng or np.random.default_rng(0)
    sample = manifests.shards.sample(
        n=min(n_sample, len(manifests.shards)), random_state=int(rng.integers(1 << 30))
    )
    rows = []
    for row in sample.itertuples(index=False):
        array = np.load(row.X_path, mmap_mode="r")
        with open(row.channel_availability_path, encoding="utf-8") as handle:
            mask = np.asarray(json.load(handle), dtype=bool)
        present = [c for c in range(3) if bool(mask[c])]
        if not present or array.shape[1] < 60 * 256:
            continue
        step = max(1, array.shape[1] // 120_000)
        block = np.asarray(array[:, ::step], dtype=np.float32)[present]
        if not np.isfinite(block).all():
            continue
        rows.append(
            {
                "recording_id": row.recording_id,
                "subject": row.subject,
                "split": row.split,
                "median_abs_z": float(np.median(np.abs(block))),
                "duration_seconds": float(array.shape[1] / SFREQ),
                "available_channels": _channel_names(mask),
            }
        )
    if not rows:
        return {}
    table = pd.DataFrame(rows).sort_values("median_abs_z")
    return {
        "quietest": table.iloc[0].to_dict(),
        "loudest": table.iloc[-1].to_dict(),
        "median": table.iloc[len(table) // 2].to_dict(),
        "n_sampled": len(table),
    }


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------


def select_exemplars(
    manifests: Manifests | None = None,
    *,
    seed: int = 0,
    cache_path: Path | None = None,
    refresh: bool = False,
) -> Exemplars:
    """Choose every exemplar the atlas plots, caching the result as JSON."""
    if cache_path is not None and cache_path.exists() and not refresh:
        return Exemplars.from_json(cache_path.read_text(encoding="utf-8"))

    manifests = manifests or load_manifests()
    rng = np.random.default_rng(seed)
    print("Selecting exemplars...", flush=True)
    heroes = rank_hero_recordings(manifests)
    print(f"  hero recordings   {len(heroes)}", flush=True)
    seizures = rank_seizures(manifests)
    print(f"  seizure exemplars {len(seizures)}", flush=True)
    positives = rank_decisions(manifests, 1, rng=rng)
    negatives = rank_decisions(manifests, 0, rng=rng)
    print(f"  decisions         {len(positives)} pos / {len(negatives)} neg", flush=True)
    extremes = find_amplitude_extremes(manifests, rng=rng)

    chosen = Exemplars(
        hero_recordings=heroes,
        seizures=seizures,
        positives=positives,
        negatives=negatives,
        amplitude_extremes=extremes,
    )
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(chosen.to_json(), encoding="utf-8")
    return chosen
