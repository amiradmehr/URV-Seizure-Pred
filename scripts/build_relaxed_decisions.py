r"""Re-derive eligibility and decisions with configurable timing constants.

Why
---
The 883 -> 317 seizure loss is not a data-quality filter. Reconstructing the
rule from preprocessing.py shows 400 of the 566 exclusions (70.7%) happen
because another seizure's ictal + postictal window overlaps the candidate's
clear-history requirement, and the non-finite and bad-annotation branches
exclude exactly zero seizures. Because ``postictal_exclusion_minutes`` (60) is
equal to ``minimum_preseizure_clear_minutes`` (60), any seizure whose
predecessor ended within ~120 min of its onset is deleted. The pipeline
therefore studies *isolated* seizures by construction.

That matters because no configuration in the 42-run sweep ever varied it: the
history axis trimmed the stored window and the SOP axis only relabelled
decisions among already-eligible seizures, so all 44 cross-validation runs used
the identical 317-seizure set. Eligibility is the one axis the sweep could not
have detected as the limiting factor.

This script decouples the constants and regenerates decisions directly from the
chunk-feature banks, which already cover every recording end to end. No 330 GB
rebuild is required.

Correctness
-----------
Running with the pipeline's own constants (--clear-minutes 60
--postictal-minutes 60) must reproduce exactly 317 eligible seizures. That
identity is asserted by --verify and is the acceptance test for this script.

Two pipeline checks are deliberately omitted, both verified never to fire:
non-finite samples (validate_dataset.py asserts np.isfinite over every shard and
passed) and ``bad*`` annotations (BIDS events are never converted to MNE
annotations, so that branch is unreachable).

    python scripts/build_relaxed_decisions.py --verify
    python scripts/build_relaxed_decisions.py --clear-minutes 25 \
        --postictal-minutes 10 --history-minutes 15 --out relaxed_h15_p10.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIRECTORY = PROJECT_ROOT / "src"
if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

from seizure_prediction.config import CONFIG  # noqa: E402
from seizure_prediction.preprocessing import (  # noqa: E402
    find_events_file,
    read_recording_events,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clear-minutes",
        type=float,
        default=60.0,
        help="Clean EEG required before a target onset (pipeline default 60).",
    )
    parser.add_argument(
        "--postictal-minutes",
        type=float,
        default=60.0,
        help="Postictal exclusion after seizure end (pipeline default 60).",
    )
    parser.add_argument(
        "--history-minutes",
        type=float,
        default=45.0,
        help="Model input length. Must not exceed --clear-minutes minus SOP.",
    )
    parser.add_argument("--sop-minutes", type=float, default=10.0)
    parser.add_argument("--stride-seconds", type=float, default=60.0)
    parser.add_argument(
        "--splits", nargs="+", default=["train", "validation"],
        help="Test patients are excluded by default.",
    )
    parser.add_argument(
        "--feature-dir", type=Path,
        default=CONFIG.interim_data_dir / "chunk_features",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Output CSV. Written under data/interim/relaxed/ if relative.",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Only check that the pipeline constants reproduce 317 eligible seizures.",
    )
    return parser.parse_args()


def log(message: str) -> None:
    print(message, flush=True)


def recording_id_of(row: pd.Series) -> str:
    parts = [f"sub-{row['subject']}"]
    for key, prefix in (("session", "ses"), ("task", "task"), ("run", "run")):
        value = row.get(key)
        if isinstance(value, str) and value:
            parts.append(f"{prefix}-{value}")
    return "_".join(parts)


def overlaps(a0: float, a1: float, b0: float, b1: float) -> bool:
    return a0 < b1 and a1 > b0


def interval_blocked(
    start: float,
    stop: float,
    seizures: np.ndarray,
    postictal_seconds: float,
    events: np.ndarray,
) -> bool:
    """True if [start, stop) hits ictal+postictal or a non-background event."""
    for onset, duration in seizures:
        if overlaps(start, stop, onset, onset + duration + postictal_seconds):
            return True
    for onset, duration in events:
        if overlaps(start, stop, onset, onset + duration):
            return True
    return False


def load_recording_events(recording_id: str, subject: str, session: str,
                          task: str, run: str) -> np.ndarray:
    """Non-background, non-seizure event intervals for one recording."""
    entities = {"subject": subject, "session": session, "task": task, "run": run}
    try:
        path = find_events_file(dataset_root=CONFIG.raw_data_dir, entities=entities)
        frame = read_recording_events(path)
    except (FileNotFoundError, RuntimeError, ValueError):
        return np.empty((0, 2), dtype=np.float64)
    keep = ~(
        frame["eventType"].str.strip().str.lower().eq("bckg")
        | frame["eventType"].str.strip().str.lower().str.startswith("sz_")
    )
    subset = frame.loc[keep, ["onset", "duration"]]
    if subset.empty:
        return np.empty((0, 2), dtype=np.float64)
    return subset.to_numpy(dtype=np.float64)


def main() -> None:
    arguments = parse_arguments()
    CONFIG.validate()

    clear_seconds = arguments.clear_minutes * 60.0
    postictal_seconds = arguments.postictal_minutes * 60.0
    history_seconds = arguments.history_minutes * 60.0
    sop_seconds = arguments.sop_minutes * 60.0
    chunk_samples = int(round(CONFIG.chunk_window_seconds * CONFIG.target_sfreq))

    if history_seconds + sop_seconds > clear_seconds + 1e-6:
        raise ValueError(
            f"history ({arguments.history_minutes} min) + SOP "
            f"({arguments.sop_minutes} min) exceeds clear window "
            f"({arguments.clear_minutes} min); eligibility would not guarantee a "
            "clean history."
        )

    manifest = pd.read_csv(
        CONFIG.manifests_dir / "processed_shard_manifest.csv", dtype={"subject": str}
    )
    manifest = manifest[manifest["split"].isin(arguments.splits)]
    kept_recordings = set(manifest["recording_id"])

    seizures = pd.read_csv(
        CONFIG.manifests_dir / "seizure_manifest.csv",
        dtype={"subject": str, "session": str, "task": str, "run": str},
    )
    seizures["recording_id"] = seizures.apply(recording_id_of, axis=1)

    log(f"clear={arguments.clear_minutes:g}m postictal={arguments.postictal_minutes:g}m "
        f"history={arguments.history_minutes:g}m sop={arguments.sop_minutes:g}m")
    log(f"{len(manifest)} recordings in splits {arguments.splits}")

    # ------------------------------------------------------------------
    # Pass 1: eligibility over ALL seizures (for the 317 identity check)
    # ------------------------------------------------------------------
    eligible_ids: set[str] = set()
    events_cache: dict[str, np.ndarray] = {}
    lengths: dict[str, int] = {}

    for recording_id, group in seizures.groupby("recording_id", sort=False):
        first = group.iloc[0]
        if recording_id not in events_cache:
            events_cache[recording_id] = load_recording_events(
                recording_id, first["subject"], first["session"],
                first["task"], first["run"],
            )
        events = events_cache[recording_id]
        table = group[["onset_seconds", "duration_seconds"]].to_numpy(dtype=np.float64)

        for index, row in enumerate(group.itertuples(index=False)):
            onset = float(row.onset_seconds)
            start = onset - clear_seconds
            if start < 0:
                continue
            others = np.delete(table, index, axis=0)
            if interval_blocked(start, onset, others, postictal_seconds, events):
                continue
            eligible_ids.add(str(row.seizure_id))

    log(f"eligible seizures (all recordings): {len(eligible_ids)}")

    if arguments.verify:
        expected = 317
        status = "PASS" if len(eligible_ids) == expected else "FAIL"
        log(f"\nverification with pipeline constants: {status} "
            f"(got {len(eligible_ids)}, expected {expected})")
        raise SystemExit(0 if status == "PASS" else 1)

    # ------------------------------------------------------------------
    # Pass 2: generate decisions on retained recordings
    # ------------------------------------------------------------------
    rows: list[dict] = []
    stride_samples = int(round(arguments.stride_seconds * CONFIG.target_sfreq))
    history_samples = int(round(history_seconds * CONFIG.target_sfreq))

    for number, manifest_row in enumerate(manifest.itertuples(index=False), start=1):
        recording_id = manifest_row.recording_id
        bank = arguments.feature_dir / f"{recording_id}_features.npy"
        if not bank.exists():
            continue
        n_chunks = int(np.load(bank, mmap_mode="r").shape[0])
        n_samples = n_chunks * chunk_samples
        lengths[recording_id] = n_samples

        group = seizures[seizures["recording_id"] == recording_id]
        table = (
            group[["onset_seconds", "duration_seconds"]].to_numpy(dtype=np.float64)
            if not group.empty else np.empty((0, 2), dtype=np.float64)
        )
        if recording_id not in events_cache:
            events_cache[recording_id] = load_recording_events(
                recording_id, manifest_row.subject, "", CONFIG.bids_task, "",
            ) if group.empty else np.empty((0, 2), dtype=np.float64)
        events = events_cache.get(recording_id, np.empty((0, 2), dtype=np.float64))

        onsets = table[:, 0] if len(table) else np.empty(0)
        durations = table[:, 1] if len(table) else np.empty(0)
        ids = group["seizure_id"].astype(str).to_numpy() if not group.empty else np.empty(0, dtype=object)

        decision_end = history_samples
        while decision_end + int(sop_seconds * CONFIG.target_sfreq) <= n_samples:
            t = decision_end / CONFIG.target_sfreq
            start = t - history_seconds

            # drop if the decision instant is ictal or postictal
            if len(onsets) and np.any(
                (onsets <= t) & (t < onsets + durations + postictal_seconds)
            ):
                decision_end += stride_samples
                continue

            if interval_blocked(start, t, table, postictal_seconds, events):
                decision_end += stride_samples
                continue

            label, target = 0, ""
            if len(onsets):
                inside = (onsets > t) & (onsets <= t + sop_seconds)
                if inside.any():
                    first = int(np.argmax(inside))
                    if str(ids[first]) in eligible_ids:
                        label, target = 1, str(ids[first])
                    else:
                        decision_end += stride_samples
                        continue

            rows.append({
                "recording_id": recording_id,
                "subject": manifest_row.subject,
                "split": manifest_row.split,
                "decision_time_seconds": t,
                "history_start_sample": decision_end - history_samples,
                "decision_end_sample": decision_end,
                "label": label,
                "target_seizure_id": target,
            })
            decision_end += stride_samples

        if number % 400 == 0:
            log(f"  [{number}/{len(manifest)}] {len(rows):,} decisions so far")

    decisions = pd.DataFrame(rows)
    if decisions.empty:
        raise SystemExit("No decisions generated.")

    positives = int((decisions["label"] == 1).sum())
    covered = decisions.loc[decisions["label"] == 1, "target_seizure_id"].nunique()
    log(f"\ndecisions   : {len(decisions):,}")
    log(f"positive    : {positives:,}  ({positives/len(decisions)*100:.3f}%)")
    log(f"seizures    : {covered}")
    log(f"patients    : {decisions['subject'].nunique()}")

    out = arguments.out or Path(
        f"relaxed_c{arguments.clear_minutes:g}_p{arguments.postictal_minutes:g}"
        f"_h{arguments.history_minutes:g}.csv"
    )
    if not out.is_absolute():
        out = CONFIG.interim_data_dir / "relaxed" / out
    out.parent.mkdir(parents=True, exist_ok=True)
    decisions.to_csv(out, index=False)
    log(f"\nwrote {out}")


if __name__ == "__main__":
    main()
