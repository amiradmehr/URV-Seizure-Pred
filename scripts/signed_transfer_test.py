r"""Does the pre-onset shift point the same way in every patient?

The proximity-matched sham control established that the 10 minutes before an
eligible onset really are more separable from that recording's interictal
baseline than a random interictal window of the same length: +0.0444, patient-
clustered 95% CI [+0.0327, +0.0567], and it survives in awake seizures alone
(+0.0230, CI [+0.0110, +0.0348]).

But that measurement is UNSIGNED -- it uses |AUC - 0.5| per seizure. An unsigned
within-recording effect whose DIRECTION varies between patients would produce
exactly the patient-independent chance result the project has measured
everywhere: something happens before a seizure, but not the same something.

This script tests the direction. Each seizure yields one signed feature vector
(pre-onset window mean, referenced to its own recording's interictal mean) plus
matched interictal controls built identically. A logistic model is then trained
with patients held out, so it can only succeed if the shift generalises across
people.

Decision rule, fixed before running: held-out AUC <= 0.55 with a patient-
clustered CI containing 0.5 means the structure is idiosyncratic and cannot
support a patient-independent model on this cohort -- that is the mechanism of
the null, and the project stops. Above 0.55 with the CI excluding 0.5 is a
genuine finding.

    python scripts/signed_transfer_test.py --controls 8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIRECTORY = PROJECT_ROOT / "src"
if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

from seizure_prediction.config import CONFIG  # noqa: E402

CHUNK_SECONDS = 5.0
WINDOW_CHUNKS = 120          # 10 min
FAR_SECONDS = 7200.0


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controls", type=int, default=8)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--feature-dir", type=Path,
                        default=CONFIG.interim_data_dir / "chunk_features")
    parser.add_argument("--out", type=Path,
                        default=PROJECT_ROOT / "outputs" / "analysis" / "signed")
    return parser.parse_args()


def log(m: str) -> None:
    print(m, flush=True)


def recording_id_of(row) -> str:
    parts = [f"sub-{row['subject']}"]
    for key, prefix in (("session", "ses"), ("task", "task"), ("run", "run")):
        v = row.get(key)
        if isinstance(v, str) and v:
            parts.append(f"{prefix}-{v}")
    return "_".join(parts)


def main() -> None:
    a = parse_arguments()
    CONFIG.validate()
    rng = np.random.default_rng(a.seed)
    a.out.mkdir(parents=True, exist_ok=True)

    sz = pd.read_csv(CONFIG.manifests_dir / "seizure_manifest.csv",
                     dtype={"subject": str, "session": str, "task": str, "run": str})
    sz["recording_id"] = sz.apply(recording_id_of, axis=1)
    elig = sz[sz["eligible_for_prediction"].astype(str).str.lower().eq("true")].copy()

    X, y, groups, vig, montage, seiz_ids = [], [], [], [], [], []
    shifts = []          # per-seizure signed shift, for the direction analysis
    for n, (rec, group) in enumerate(elig.groupby("recording_id"), start=1):
        bp = a.feature_dir / f"{rec}_features.npy"
        ap = a.feature_dir / f"{rec}_availability.npy"
        if not bp.exists():
            continue
        bank = np.asarray(np.load(bp, mmap_mode="r"), dtype=np.float64)
        usable = np.load(ap).astype(bool)
        # Electrode pair actually recorded: a patient-level attribute that
        # explained 106% of the earlier apparent sleep effect.
        per_ch = usable.reshape(3, -1)[:, 0]
        pair = "+".join(n for n, f in zip(("L", "R", "X"), per_ch, strict=True) if f)
        n_chunks = bank.shape[0]
        if n_chunks < WINDOW_CHUNKS + 200:
            continue

        inrec = sz[sz["recording_id"] == rec]
        onsets = inrec["onset_seconds"].to_numpy(float)
        ends = onsets + inrec["duration_seconds"].to_numpy(float)
        times = (np.arange(n_chunks) + 0.5) * CHUNK_SECONDS
        far = np.ones(n_chunks, dtype=bool)
        for o, e in zip(onsets, ends, strict=True):
            far &= (times < o - FAR_SECONDS) | (times > e + FAR_SECONDS)
        if far.sum() < 200:
            continue
        far_index = np.flatnonzero(far)
        # Reference: this recording's own interictal mean. Removing it makes the
        # vector a signed SHIFT rather than an absolute level, so patient gain
        # and montage cancel while direction survives.
        reference = bank[far_index].mean(axis=0)

        for row in group.itertuples(index=False):
            stop = int(float(row.onset_seconds) / CHUNK_SECONDS)
            start = stop - WINDOW_CHUNKS
            if start < 0 or stop > n_chunks:
                continue
            vec = bank[start:stop].mean(axis=0) - reference
            vec = np.where(usable, vec, 0.0)
            X.append(vec); y.append(1); groups.append(str(row.subject))
            vig.append(str(row.vigilance)); montage.append(pair)
            seiz_ids.append(str(row.seizure_id))
            shifts.append({"subject": str(row.subject), "vec": vec, "usable": usable})

            cands = far_index[far_index >= WINDOW_CHUNKS]
            if len(cands) == 0:
                continue
            picks = rng.choice(cands, size=min(a.controls, len(cands)),
                               replace=len(cands) < a.controls)
            for p in picks:
                s_stop = int(p); s_start = s_stop - WINDOW_CHUNKS
                cvec = bank[s_start:s_stop].mean(axis=0) - reference
                X.append(np.where(usable, cvec, 0.0)); y.append(0)
                groups.append(str(row.subject)); vig.append(str(row.vigilance))
                montage.append(pair); seiz_ids.append(str(row.seizure_id))
        if n % 60 == 0:
            log(f"  [{n}] {int(np.sum(y))} pre-onset vectors")

    X = np.asarray(X); y = np.asarray(y); groups = np.asarray(groups)
    vig = np.asarray(vig); montage = np.asarray(montage)
    seiz_ids = np.asarray(seiz_ids)
    log(f"\n{len(y)} vectors | {int(y.sum())} pre-onset | "
        f"{len(np.unique(groups))} patients | {X.shape[1]} features")

    # Patient-held-out out-of-fold prediction
    oof = np.zeros(len(y))
    gkf = GroupKFold(n_splits=a.folds)
    for tr, te in gkf.split(X, y, groups):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, C=0.1, class_weight="balanced"))
        model.fit(X[tr], y[tr])
        oof[te] = model.predict_proba(X[te])[:, 1]

    pd.DataFrame({"seizure_id": seiz_ids, "subject": groups, "vigilance": vig,
                  "montage": montage, "label": y, "probability": oof}
                 ).to_csv(a.out / "signed_oof.csv", index=False)
    auc = roc_auc_score(y, oof)
    uniq = np.unique(groups)
    draws = []
    for _ in range(a.bootstrap):
        picked = rng.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([np.flatnonzero(groups == s) for s in picked])
        if len(np.unique(y[idx])) < 2:
            continue
        draws.append(roc_auc_score(y[idx], oof[idx]))
    lo, hi = np.percentile(draws, [2.5, 97.5])

    # Direction consistency: per feature, the fraction of patients whose mean
    # pre-onset shift agrees in sign with the population mean.
    per_patient = {}
    for s in shifts:
        per_patient.setdefault(s["subject"], []).append(s["vec"])
    matrix = np.array([np.mean(v, axis=0) for v in per_patient.values()])
    population = matrix.mean(axis=0)
    agree = (np.sign(matrix) == np.sign(population)).mean(axis=0)

    summary = {
        "vectors": int(len(y)), "pre_onset": int(y.sum()),
        "patients": int(len(uniq)),
        "held_out_auc": float(auc),
        "auc_ci_low": float(lo), "auc_ci_high": float(hi),
        "direction_agreement_mean": float(np.mean(agree[population != 0])),
        "direction_agreement_max": float(np.max(agree)),
        "features_above_70pct_agreement": int(np.sum(agree > 0.7)),
    }
    for state in ("awake", "asleep"):
        m = vig == state
        if m.sum() and len(np.unique(y[m])) == 2:
            summary[f"auc_{state}"] = float(roc_auc_score(y[m], oof[m]))
            summary[f"n_{state}"] = int(m.sum())

    (a.out / "signed_summary.json").write_text(json.dumps(summary, indent=2),
                                               encoding="utf-8")
    print()
    print("=" * 72)
    print("SIGNED PATIENT-HELD-OUT TRANSFER".center(72))
    print("=" * 72)
    print(f"  vectors / pre-onset / patients : {summary['vectors']} / "
          f"{summary['pre_onset']} / {summary['patients']}")
    print(f"  held-out AUC                   : {auc:.4f}")
    print(f"  95% patient-clustered CI       : [{lo:.4f}, {hi:.4f}]")
    for state in ("awake", "asleep"):
        if f"auc_{state}" in summary:
            print(f"    {state:7s} AUC {summary[f'auc_{state}']:.4f} "
                  f"(n={summary[f'n_{state}']})")
    print()
    print(f"  mean per-feature sign agreement across patients : "
          f"{summary['direction_agreement_mean']:.3f}  (0.5 = no consistency)")
    print(f"  features with >70% agreement                    : "
          f"{summary['features_above_70pct_agreement']} / {X.shape[1]}")
    print()
    survives = auc > 0.55 and lo > 0.5
    print(f"  DECISION RULE -> {'GENUINE TRANSFERABLE EFFECT' if survives else 'IDIOSYNCRATIC — STOP'}")
    if not survives:
        print("  Something happens before a seizure, but not the same something in")
        print("  different patients. That is the mechanism of the patient-independent")
        print("  null: an unsigned within-recording effect with no shared direction.")
    print(f"\n  artefacts: {a.out}")


if __name__ == "__main__":
    main()
