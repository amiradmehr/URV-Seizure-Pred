r"""Is the pre-onset window special, or is any local window special?

The one lead that survived the audit is an *unsigned* within-recording effect:
the 10 minutes before an eligible onset separate from that recording's
far-interictal baseline more than a length-matched interictal block does
(+0.0492 mean |AUC-0.5|, paired p=8.5e-12), and a within-recording classifier
reaches median AUC 0.65.

Both verifiers flagged the same untested confound. The far-interictal baseline
sits >2 h away by construction, so "close to a seizure" and "close to the end of
a long quiet stretch" are not separated. EEG is strongly non-stationary over
hours -- vigilance, electrode drift, activity -- so ANY local 10-minute window
may separate from a distant baseline without anything pre-ictal happening.

This script runs the control. For every eligible seizure it measures

    separability(window) = mean over available features of
                           |AUC(window vs that recording's baseline) - 0.5|

for the true pre-onset window and for N sham windows: 10-minute slices ending at
random interictal reference points in the same recording. The tested window's own
chunks are always removed from the baseline, so true and sham are constructed
identically and differ only in whether a seizure follows.

Decision rule, fixed before running: if the true pre-onset window does not beat
the sham by a margin whose patient-clustered 95% CI excludes zero, the lead is
dead and the project's negative result is complete.

    python scripts/preonset_sham_test.py --shams 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIRECTORY = PROJECT_ROOT / "src"
if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

from seizure_prediction.config import CONFIG  # noqa: E402

CHUNK_SECONDS = 5.0
WINDOW_SECONDS = 600.0      # the 10-min SOP: what a positive decision sees
FAR_SECONDS = 7200.0        # baseline must be >2 h from any seizure
WINDOW_CHUNKS = int(WINDOW_SECONDS / CHUNK_SECONDS)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shams", type=int, default=20)
    parser.add_argument("--min-baseline-chunks", type=int, default=200)
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--feature-dir", type=Path,
                        default=CONFIG.interim_data_dir / "chunk_features")
    parser.add_argument("--out", type=Path,
                        default=PROJECT_ROOT / "outputs" / "analysis" / "sham")
    return parser.parse_args()


def log(message: str) -> None:
    print(message, flush=True)


def auc_columns(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-column AUC of a vs b, ties averaged."""
    n_a, n_b = a.shape[0], b.shape[0]
    stacked = np.vstack([a, b])
    ranks = np.empty_like(stacked, dtype=np.float64)
    for column in range(stacked.shape[1]):
        values = stacked[:, column]
        order = np.argsort(values, kind="stable")
        raw = np.empty(len(values), dtype=np.float64)
        raw[order] = np.arange(len(values), dtype=np.float64)
        unique, inverse, counts = np.unique(
            values, return_inverse=True, return_counts=True)
        sums = np.zeros(len(unique))
        np.add.at(sums, inverse, raw)
        ranks[:, column] = (sums / counts)[inverse]
    rank_sum_a = ranks[:n_a].sum(axis=0)
    return (rank_sum_a - n_a * (n_a - 1) / 2.0) / (n_a * n_b)


def separability(window: np.ndarray, baseline: np.ndarray,
                 usable: np.ndarray) -> float:
    """Mean |AUC-0.5| over usable feature columns."""
    if window.shape[0] < 10 or baseline.shape[0] < 10 or not usable.any():
        return float("nan")
    auc = auc_columns(window[:, usable], baseline[:, usable])
    return float(np.mean(np.abs(auc - 0.5)))


def recording_id_of(row) -> str:
    parts = [f"sub-{row['subject']}"]
    for key, prefix in (("session", "ses"), ("task", "task"), ("run", "run")):
        value = row.get(key)
        if isinstance(value, str) and value:
            parts.append(f"{prefix}-{value}")
    return "_".join(parts)


def main() -> None:
    arguments = parse_arguments()
    CONFIG.validate()
    rng = np.random.default_rng(arguments.seed)
    arguments.out.mkdir(parents=True, exist_ok=True)

    seizures = pd.read_csv(
        CONFIG.manifests_dir / "seizure_manifest.csv",
        dtype={"subject": str, "session": str, "task": str, "run": str})
    seizures["recording_id"] = seizures.apply(recording_id_of, axis=1)
    eligible = seizures[
        seizures["eligible_for_prediction"].astype(str).str.lower().eq("true")
    ].copy()
    log(f"{len(eligible)} eligible seizures across "
        f"{eligible['recording_id'].nunique()} recordings")

    rows = []
    for number, (rec, group) in enumerate(eligible.groupby("recording_id"), start=1):
        bank_path = arguments.feature_dir / f"{rec}_features.npy"
        avail_path = arguments.feature_dir / f"{rec}_availability.npy"
        if not bank_path.exists():
            continue
        bank = np.asarray(np.load(bank_path, mmap_mode="r"), dtype=np.float64)
        usable = np.load(avail_path).astype(bool)
        n_chunks = bank.shape[0]
        if n_chunks < WINDOW_CHUNKS + arguments.min_baseline_chunks:
            continue

        # Per-recording median centring cancels the per-patient gain, so the
        # comparison is about shape and local level, not electrode seating.
        bank = bank - np.median(bank, axis=0, keepdims=True)

        in_recording = seizures[seizures["recording_id"] == rec]
        onsets = in_recording["onset_seconds"].to_numpy(float)
        ends = onsets + in_recording["duration_seconds"].to_numpy(float)

        times = (np.arange(n_chunks) + 0.5) * CHUNK_SECONDS
        far = np.ones(n_chunks, dtype=bool)
        for onset, end in zip(onsets, ends, strict=True):
            far &= (times < onset - FAR_SECONDS) | (times > end + FAR_SECONDS)
        if far.sum() < arguments.min_baseline_chunks:
            continue
        far_index = np.flatnonzero(far)

        for row in group.itertuples(index=False):
            onset = float(row.onset_seconds)
            stop = int(onset / CHUNK_SECONDS)
            start = stop - WINDOW_CHUNKS
            if start < 0 or stop > n_chunks:
                continue

            keep = (far_index < start) | (far_index >= stop)
            true_sep = separability(bank[start:stop], bank[far_index[keep]], usable)
            if not np.isfinite(true_sep):
                continue

            candidates = far_index[far_index >= WINDOW_CHUNKS]
            sham_values = []
            if len(candidates):
                picks = rng.choice(
                    candidates,
                    size=min(arguments.shams, len(candidates)),
                    replace=len(candidates) < arguments.shams)
                for pick in picks:
                    s_stop = int(pick)
                    s_start = s_stop - WINDOW_CHUNKS
                    keep_s = (far_index < s_start) | (far_index >= s_stop)
                    value = separability(
                        bank[s_start:s_stop], bank[far_index[keep_s]], usable)
                    if np.isfinite(value):
                        sham_values.append(value)
            if not sham_values:
                continue

            rows.append({
                "seizure_id": str(row.seizure_id),
                "subject": str(row.subject),
                "recording_id": rec,
                "vigilance": str(row.vigilance),
                "true_separability": true_sep,
                "sham_separability_mean": float(np.mean(sham_values)),
                "sham_separability_sd": float(np.std(sham_values)),
                "n_sham": len(sham_values),
            })

        if number % 40 == 0:
            log(f"  [{number}] {len(rows)} seizures measured")

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise SystemExit("No seizures measurable.")
    frame.to_csv(arguments.out / "sham_separability.csv", index=False)

    delta = frame["true_separability"] - frame["sham_separability_mean"]
    subjects = frame["subject"].to_numpy()
    unique = np.unique(subjects)
    by_subject = {s: delta.to_numpy()[subjects == s] for s in unique}
    draws = np.empty(arguments.bootstrap)
    for i in range(arguments.bootstrap):
        picked = rng.choice(unique, size=len(unique), replace=True)
        draws[i] = np.mean(np.concatenate([by_subject[s] for s in picked]))
    low, high = np.percentile(draws, [2.5, 97.5])

    summary = {
        "seizures": int(len(frame)),
        "patients": int(len(unique)),
        "shams_per_seizure": int(frame["n_sham"].median()),
        "true_separability_mean": float(frame["true_separability"].mean()),
        "sham_separability_mean": float(frame["sham_separability_mean"].mean()),
        "delta_mean": float(delta.mean()),
        "delta_ci_low": float(low),
        "delta_ci_high": float(high),
        "fraction_true_above_sham": float((delta > 0).mean()),
    }
    for state, sub in frame.groupby("vigilance"):
        d = sub["true_separability"] - sub["sham_separability_mean"]
        summary[f"delta_{state}"] = float(d.mean())
        summary[f"n_{state}"] = int(len(sub))

    (arguments.out / "sham_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    print()
    print("=" * 72)
    print("PRE-ONSET vs PROXIMITY-MATCHED SHAM".center(72))
    print("=" * 72)
    print(f"  seizures / patients      : {summary['seizures']} / {summary['patients']}")
    print(f"  shams per seizure        : {summary['shams_per_seizure']}")
    print(f"  true separability        : {summary['true_separability_mean']:.4f}")
    print(f"  sham separability        : {summary['sham_separability_mean']:.4f}")
    print(f"  delta (true - sham)      : {summary['delta_mean']:+.4f}")
    print(f"  95% patient-clustered CI : [{summary['delta_ci_low']:+.4f}, "
          f"{summary['delta_ci_high']:+.4f}]")
    print(f"  fraction true > sham     : {summary['fraction_true_above_sham']:.3f}")
    print()
    for state in ("awake", "asleep", "un"):
        if f"delta_{state}" in summary:
            print(f"    {state:7s} delta {summary[f'delta_{state}']:+.4f}  "
                  f"(n={summary[f'n_{state}']})")
    print()
    verdict = "LEAD SURVIVES" if summary["delta_ci_low"] > 0 else "LEAD IS DEAD"
    print(f"  DECISION RULE -> {verdict}")
    if summary["delta_ci_low"] <= 0:
        print("  The pre-onset window is no more separable from baseline than a")
        print("  random interictal window of the same length. The apparent effect")
        print("  was within-recording non-stationarity, not pre-ictal state.")
    print(f"\n  artefacts: {arguments.out}")


if __name__ == "__main__":
    main()
