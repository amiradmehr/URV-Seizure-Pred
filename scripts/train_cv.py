r"""Patient-wise cross-validation for the seizure-risk task.

Why this exists
---------------
Every result so far was read off a single validation split holding 33 target
seizures. At that size a two-seizure change is indistinguishable from noise:
the 95% Wilson interval on 2/33 is [0.017, 0.196]. Four model variants were
compared through that keyhole and none of the differences survived a paired
test. The measurement instrument, not the model, was the binding constraint.

K-fold over *patients* reuses every patient for evaluation exactly once, so the
pooled out-of-fold predictions cover ~250 seizures instead of 33, and bootstrap
resampling over seizures gives the interval the single split could not.

The held-out test patients (113-125) are deliberately excluded. Cross-validation
runs on the train and validation patients only, so a final one-shot test
evaluation remains available for whichever configuration is chosen.

Model selection inside a fold uses an inner patient split carved out of that
fold's training patients, never the held-out fold.

    python scripts/train_cv.py --folds 5 --epochs 25
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIRECTORY = PROJECT_ROOT / "src"
if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

from seizure_prediction.config import CONFIG  # noqa: E402
from seizure_prediction.datasets import ChunkFeatureDataset  # noqa: E402
from seizure_prediction.models import (  # noqa: E402
    SpectralAttentionConfig,
    SpectralAttentionRiskModel,
    SpectralContrastRiskModel,
    SpectralGRURiskModel,
    SpectralMeanPoolRiskModel,
)

ALARM_BUDGETS = (0.1, 0.25, 0.5, 1.0, 2.0)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--train-negative-ratio", type=float, default=4.0)
    parser.add_argument("--num-workers", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--history-minutes",
        type=float,
        default=None,
        help=(
            "Use only the last N minutes of the stored 45-minute history. "
            "Trims the feature window; nothing is rebuilt."
        ),
    )
    parser.add_argument(
        "--sop-minutes",
        type=float,
        default=None,
        help=(
            "Relabel with a wider seizure-occurrence period. The stored labels "
            "use 10 min; a wider window only turns negatives into positives, so "
            "it needs no new EEG and no new eligibility check."
        ),
    )
    parser.add_argument(
        "--architecture",
        choices=("spectral-attention", "spectral-gru", "spectral-meanpool",
                 "spectral-contrast", "logistic-mean"),
        default="spectral-attention",
        help=(
            "logistic-mean is a linear control: the same features averaged over "
            "the 540 chunks. If the attention model cannot beat it, the temporal "
            "structure is not being used."
        ),
    )
    parser.add_argument("--tag", type=str, default="cv")
    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=CONFIG.interim_data_dir / "chunk_features",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "cv"
    )
    parser.add_argument(
        "--decisions-csv",
        type=Path,
        default=None,
        help=(
            "Use an externally generated decision set (see "
            "scripts/build_relaxed_decisions.py) instead of the pipeline's "
            "decision manifest. Lets eligibility be varied -- the one axis the "
            "42-config sweep could not reach."
        ),
    )
    parser.add_argument(
        "--recent-chunks", type=int, default=60,
        help="Chunks treated as 'recent' by spectral-contrast (60 = last 5 min).",
    )
    parser.add_argument(
        "--level-features", action="store_true",
        help=(
            "Append each window's level relative to its recording median. "
            "Window normalisation removes the level along with the patient gain, "
            "and the level is where vigilance lives."
        ),
    )
    parser.add_argument("--bootstrap", type=int, default=2000)
    return parser.parse_args()


def log(message: str) -> None:
    print(message, flush=True)


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LogisticMeanModel(nn.Module):
    """Linear control: mean-pool the features, then one linear layer."""

    def __init__(self, n_features: int, n_chans: int = 3) -> None:
        super().__init__()
        self.n_chans = n_chans
        self.head = nn.Sequential(
            nn.LayerNorm(n_features + n_chans),
            nn.Linear(n_features + n_chans, 1),
        )

    def forward(self, features: torch.Tensor, availability: torch.Tensor) -> torch.Tensor:
        pooled = features.mean(dim=1)
        per_channel = availability.shape[1] // self.n_chans
        flags = availability.reshape(-1, self.n_chans, per_channel)[:, :, 0]
        return self.head(torch.cat([pooled, flags.to(pooled.dtype)], dim=1)).squeeze(1)

    def predict_proba(self, features: torch.Tensor, availability: torch.Tensor):
        return torch.sigmoid(self(features, availability))


def load_all_decisions(feature_dir: Path) -> pd.DataFrame:
    """Every decision from the train and validation patients, test excluded."""
    manifest = pd.read_csv(
        CONFIG.manifests_dir / "processed_shard_manifest.csv", dtype={"subject": str}
    )
    manifest = manifest[manifest["split"].isin(["train", "validation"])]

    frames = []
    for row in manifest.itertuples(index=False):
        if not (feature_dir / f"{row.recording_id}_features.npy").exists():
            continue
        metadata = pd.read_csv(
            row.metadata_path,
            dtype={"recording_id": str, "subject": str, "session": str,
                   "task": str, "run": str},
        )
        frames.append(metadata)
    if not frames:
        raise RuntimeError(f"No feature banks found in {feature_dir}.")
    return pd.concat(frames, ignore_index=True)


def subsample_negatives(
    examples: pd.DataFrame, ratio: float, seed: int
) -> pd.DataFrame:
    positives = examples[examples["label"] == 1]
    negatives = examples[examples["label"] == 0]
    keep = min(int(round(len(positives) * ratio)), len(negatives))
    return pd.concat(
        [positives, negatives.sample(n=keep, random_state=seed)], ignore_index=True
    ).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def chunks_used(arguments) -> int:
    """Chunks per example after any history trimming."""
    full = int(CONFIG.input_window_seconds / CONFIG.chunk_window_seconds)
    if arguments.history_minutes is None:
        return full
    return int(round(arguments.history_minutes * 60.0 / CONFIG.chunk_window_seconds))


def build_model(arguments, n_features: int, device: torch.device) -> nn.Module:
    if arguments.architecture == "logistic-mean":
        return LogisticMeanModel(n_features).to(device)
    model_config = SpectralAttentionConfig(
        n_features=n_features,
        n_chunks=chunks_used(arguments),
        embedding_dim=arguments.embedding_dim,
        hidden_dim=arguments.hidden_dim,
        dropout=arguments.dropout,
    )
    if arguments.architecture == "spectral-contrast":
        return SpectralContrastRiskModel(
            model_config, recent_chunks=arguments.recent_chunks
        ).to(device)
    if arguments.architecture == "spectral-gru":
        return SpectralGRURiskModel(model_config).to(device)
    if arguments.architecture == "spectral-meanpool":
        return SpectralMeanPoolRiskModel(model_config).to(device)
    return SpectralAttentionRiskModel(model_config).to(device)


def relabel_for_sop(decisions: pd.DataFrame, sop_minutes: float) -> pd.DataFrame:
    """Widen the seizure-occurrence period by relabelling existing decisions.

    The stored labels use SOP = 10 min. Widening it can only turn negatives into
    positives: a decision's history, its cleanliness and its ictal/postictal
    exclusion are all independent of how far ahead the label looks. Only
    seizures already marked eligible may become targets, so the 60-minute
    clear-history guarantee still holds.
    """
    seizures = pd.read_csv(
        CONFIG.manifests_dir / "seizure_manifest.csv",
        dtype={"subject": str, "session": str, "task": str, "run": str},
    )
    eligible = seizures[
        seizures["eligible_for_prediction"].astype(str).str.lower().eq("true")
    ]
    onsets: dict[str, np.ndarray] = {}
    ids: dict[str, np.ndarray] = {}
    for _, row in eligible.iterrows():
        parts = [f"sub-{row['subject']}"]
        if isinstance(row["session"], str) and row["session"]:
            parts.append(f"ses-{row['session']}")
        if isinstance(row["task"], str) and row["task"]:
            parts.append(f"task-{row['task']}")
        if isinstance(row["run"], str) and row["run"]:
            parts.append(f"run-{row['run']}")
        key = "_".join(parts)
        onsets.setdefault(key, []).append(float(row["onset_seconds"]))
        ids.setdefault(key, []).append(str(row["seizure_id"]))

    window = sop_minutes * 60.0
    updated = decisions.copy()
    label = updated["label"].to_numpy(dtype=np.int64).copy()
    target = updated["target_seizure_id"].astype("object").to_numpy().copy()
    times = updated["decision_time_seconds"].to_numpy(dtype=np.float64)
    recordings = updated["recording_id"].astype(str).to_numpy()

    changed = 0
    for key in np.unique(recordings):
        if key not in onsets:
            continue
        mask = recordings == key
        recording_onsets = np.asarray(onsets[key], dtype=np.float64)
        recording_ids = np.asarray(ids[key], dtype=object)
        t = times[mask]
        # nearest future onset within the widened window
        delta = recording_onsets[None, :] - t[:, None]
        inside = (delta > 0) & (delta <= window)
        any_inside = inside.any(axis=1)
        first = np.where(any_inside, np.argmax(inside, axis=1), 0)

        block_label = label[mask]
        block_target = target[mask]
        promote = any_inside & (block_label == 0)
        changed += int(promote.sum())
        block_label[promote] = 1
        block_target[promote] = recording_ids[first[promote]]
        label[mask] = block_label
        target[mask] = block_target

    updated["label"] = label
    updated["target_seizure_id"] = target
    print(
        f"SOP {sop_minutes:g} min: promoted {changed:,} negatives to positive "
        f"({int((decisions['label']==1).sum()):,} -> {int(label.sum()):,})",
        flush=True,
    )
    return updated


@torch.no_grad()
def score(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    out: list[float] = []
    for features, availability, _ in loader:
        features = features.to(device, non_blocking=True)
        availability = availability.to(device, non_blocking=True)
        out.extend(model.predict_proba(features, availability).float().cpu().tolist())
    return np.asarray(out, dtype=np.float64)


def train_one_fold(
    arguments,
    train_examples: pd.DataFrame,
    inner_examples: pd.DataFrame,
    device: torch.device,
    n_features: int,
) -> nn.Module:
    """Train on a fold's patients, selecting the epoch on an inner patient split."""
    train_loader = DataLoader(
        ChunkFeatureDataset(train_examples, CONFIG, arguments.feature_dir,
                            history_minutes=arguments.history_minutes,
                            level_features=arguments.level_features),
        batch_size=arguments.batch_size, shuffle=True,
        num_workers=arguments.num_workers, pin_memory=device.type == "cuda",
        persistent_workers=arguments.num_workers > 0, drop_last=False,
    )
    inner_loader = DataLoader(
        ChunkFeatureDataset(inner_examples, CONFIG, arguments.feature_dir,
                            history_minutes=arguments.history_minutes,
                            level_features=arguments.level_features),
        batch_size=256, shuffle=False, num_workers=arguments.num_workers,
        persistent_workers=arguments.num_workers > 0,
    )

    model = build_model(arguments, n_features, device)
    optimizer = AdamW(model.parameters(), lr=arguments.learning_rate,
                      weight_decay=arguments.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(arguments.epochs, 1)
    )
    positives = int((train_examples["label"] == 1).sum())
    negatives = int((train_examples["label"] == 0).sum())
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(max(negatives / max(positives, 1), 1.0), device=device)
    )

    inner_labels = inner_examples["label"].to_numpy(dtype=np.int64)
    best_ap, best_state = -np.inf, None
    for epoch in range(1, arguments.epochs + 1):
        model.train()
        total = 0.0
        seen = 0
        for features, availability, target in train_loader:
            features = features.to(device, non_blocking=True)
            availability = availability.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(features, availability), target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += loss.item() * len(target)
            seen += len(target)
        scheduler.step()

        if len(np.unique(inner_labels)) == 2:
            inner_scores = score(model, inner_loader, device)
            inner_ap = average_precision_score(inner_labels, inner_scores)
            if inner_ap > best_ap:
                best_ap = inner_ap
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
        if epoch % 5 == 0 or epoch == arguments.epochs:
            log(f"      epoch {epoch:3d}  train_loss {total/max(seen,1):.4f}  "
                f"inner_AP {best_ap:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def seizure_sensitivity(
    frame: pd.DataFrame, threshold: float
) -> tuple[int, int]:
    positives = frame[frame["label"] == 1]
    if positives.empty:
        return 0, 0
    caught = (
        positives.assign(alarm=positives["probability"] >= threshold)
        .groupby("target_seizure_id")["alarm"].any()
    )
    return int(caught.sum()), int(len(caught))


def threshold_for_budget(frame: pd.DataFrame, budget: float) -> float:
    negatives = np.sort(frame.loc[frame["label"] == 0, "probability"].to_numpy())[::-1]
    hours = (frame["label"] == 0).sum() * CONFIG.input_stride_seconds / 3600.0
    allowed = int(budget * hours)
    if len(negatives) == 0:
        return 1.0
    return float(negatives[min(allowed, len(negatives) - 1)])


def bootstrap_seizure_sensitivity(
    frame: pd.DataFrame, budget: float, n_boot: int, seed: int
) -> tuple[float, float, float]:
    """Resample *seizures* to get an interval on seizure-level sensitivity."""
    threshold = threshold_for_budget(frame, budget)
    positives = frame[frame["label"] == 1]
    if positives.empty:
        return float("nan"), float("nan"), float("nan")
    caught = (
        positives.assign(alarm=positives["probability"] >= threshold)
        .groupby("target_seizure_id")["alarm"].any().to_numpy()
    )
    rng = np.random.default_rng(seed)
    draws = rng.choice(caught, size=(n_boot, len(caught)), replace=True).mean(axis=1)
    return float(caught.mean()), float(np.percentile(draws, 2.5)), float(
        np.percentile(draws, 97.5)
    )


def vigilance_breakdown(frame: pd.DataFrame, budget: float) -> list[dict]:
    """Seizure-level sensitivity split by the patient's state at onset.

    42% of annotated seizures occur asleep. Sleep and wake EEG differ profoundly,
    so a single model spanning both may be averaging over two different problems.
    This is reported rather than trained on, so it costs nothing.
    """
    seizures = pd.read_csv(
        CONFIG.manifests_dir / "seizure_manifest.csv", dtype={"subject": str}
    )
    state = dict(
        zip(seizures["seizure_id"].astype(str), seizures["vigilance"].astype(str))
    )
    threshold = threshold_for_budget(frame, budget)
    positives = frame[frame["label"] == 1].copy()
    if positives.empty:
        return []
    caught = (
        positives.assign(alarm=positives["probability"] >= threshold)
        .groupby("target_seizure_id")["alarm"].any()
    )
    rows = []
    labels = pd.Series(
        {s: state.get(str(s), "un") for s in caught.index}, dtype="object"
    )
    for value in sorted(set(labels.values)):
        selected = caught[labels == value]
        if len(selected) == 0:
            continue
        rows.append({
            "vigilance": value,
            "seizures": int(len(selected)),
            "detected": int(selected.sum()),
            "sensitivity": float(selected.mean()),
        })
    return rows


def main() -> None:
    arguments = parse_arguments()
    CONFIG.validate()
    torch.manual_seed(arguments.seed)
    np.random.seed(arguments.seed)
    device = resolve_device(arguments.device)

    output = arguments.output_dir / arguments.tag
    output.mkdir(parents=True, exist_ok=True)

    log(f"Device: {device} | architecture: {arguments.architecture}")
    if arguments.decisions_csv is not None:
        log(f"Loading external decision set {arguments.decisions_csv}")
        decisions = pd.read_csv(
            arguments.decisions_csv,
            dtype={"recording_id": str, "subject": str, "target_seizure_id": str},
        )
        decisions["target_seizure_id"] = decisions["target_seizure_id"].fillna("")
    else:
        log("Loading decisions (train + validation patients; test excluded)...")
        decisions = load_all_decisions(arguments.feature_dir)
    if arguments.sop_minutes is not None:
        decisions = relabel_for_sop(decisions, arguments.sop_minutes)
    patients = np.array(sorted(decisions["subject"].unique()))
    log(f"  {len(decisions):,} decisions | {len(patients)} patients | "
        f"{int((decisions['label']==1).sum()):,} positive")

    sample_bank = np.load(
        arguments.feature_dir
        / f"{decisions['recording_id'].iloc[0]}_features.npy",
        mmap_mode="r",
    )
    n_features = int(sample_bank.shape[1])
    if arguments.level_features:
        n_features *= 2
    log(f"  {n_features} features per chunk"
        f"{' (27 standardised + 27 recording-relative level)' if arguments.level_features else ''}")

    rng = np.random.default_rng(arguments.seed)
    shuffled = patients.copy()
    rng.shuffle(shuffled)
    folds = np.array_split(shuffled, arguments.folds)

    out_of_fold: list[pd.DataFrame] = []
    fold_rows: list[dict] = []
    started = time.perf_counter()

    for index, held_out in enumerate(folds, start=1):
        held = set(held_out.tolist())
        training_patients = np.array([p for p in shuffled if p not in held])
        inner_count = max(1, int(0.15 * len(training_patients)))
        inner_patients = set(training_patients[:inner_count].tolist())

        train_examples = decisions[
            decisions["subject"].isin(set(training_patients.tolist()) - inner_patients)
        ]
        inner_examples = decisions[decisions["subject"].isin(inner_patients)]
        test_examples = decisions[decisions["subject"].isin(held)]

        train_examples = subsample_negatives(
            train_examples, arguments.train_negative_ratio, arguments.seed + index
        )
        inner_examples = subsample_negatives(
            inner_examples, 20.0, arguments.seed + index
        )

        log(f"\n  fold {index}/{arguments.folds}: {len(held)} held-out patients, "
            f"{len(train_examples):,} train decisions "
            f"({int((train_examples['label']==1).sum())} pos), "
            f"{len(test_examples):,} scored")

        model = train_one_fold(
            arguments, train_examples, inner_examples, device, n_features
        )

        loader = DataLoader(
            ChunkFeatureDataset(test_examples, CONFIG, arguments.feature_dir,
                                history_minutes=arguments.history_minutes,
                                level_features=arguments.level_features),
            batch_size=256, shuffle=False, num_workers=arguments.num_workers,
            persistent_workers=arguments.num_workers > 0,
        )
        probabilities = score(model, loader, device)
        scored = test_examples.copy()
        scored["probability"] = probabilities
        scored["fold"] = index
        out_of_fold.append(scored)

        labels = scored["label"].to_numpy(dtype=np.int64)
        if len(np.unique(labels)) == 2:
            fold_rows.append({
                "fold": index,
                "patients": len(held),
                "decisions": int(len(scored)),
                "positive": int(labels.sum()),
                "average_precision": float(
                    average_precision_score(labels, probabilities)
                ),
                "roc_auc": float(roc_auc_score(labels, probabilities)),
            })
            log(f"    fold AP {fold_rows[-1]['average_precision']:.4f}  "
                f"AUC {fold_rows[-1]['roc_auc']:.4f}")

    pooled = pd.concat(out_of_fold, ignore_index=True)
    pooled.to_csv(output / "out_of_fold_predictions.csv", index=False)

    labels = pooled["label"].to_numpy(dtype=np.int64)
    scores = pooled["probability"].to_numpy(dtype=np.float64)
    prevalence = float(labels.mean())
    ap = float(average_precision_score(labels, scores))

    budgets = []
    for budget in ALARM_BUDGETS:
        point, low, high = bootstrap_seizure_sensitivity(
            pooled, budget, arguments.bootstrap, arguments.seed
        )
        caught, total = seizure_sensitivity(pooled, threshold_for_budget(pooled, budget))
        budgets.append({
            "false_alarm_budget_per_hour": budget,
            "seizure_sensitivity": point,
            "ci_low": low, "ci_high": high,
            "seizures_detected": caught, "seizures_total": total,
        })

    summary = {
        "tag": arguments.tag,
        "architecture": arguments.architecture,
        "folds": arguments.folds,
        "patients": int(len(patients)),
        "decisions": int(len(pooled)),
        "positive_decisions": int(labels.sum()),
        "target_seizures": int(
            pooled.loc[pooled["label"] == 1, "target_seizure_id"].nunique()
        ),
        "prevalence": prevalence,
        "average_precision": ap,
        "ap_over_chance": ap / prevalence,
        "roc_auc": float(roc_auc_score(labels, scores)),
        "brier_score": float(brier_score_loss(labels, scores)),
        "per_fold": fold_rows,
        "alarm_budgets": budgets,
        "vigilance_at_1_per_hour": vigilance_breakdown(pooled, 1.0),
        "history_minutes": arguments.history_minutes
        or CONFIG.input_window_seconds / 60.0,
        "sop_minutes": arguments.sop_minutes
        or CONFIG.seizure_occurrence_period_minutes,
        "chunks_used": chunks_used(arguments),
        "model_parameters": int(
            sum(p.numel() for p in build_model(arguments, n_features, "cpu").parameters())
        ),
        "arguments": {k: (str(v) if isinstance(v, Path) else v)
                      for k, v in vars(arguments).items()},
        "note": "test patients (113-125) were not used",
    }
    (output / "cv_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print()
    print("=" * 76)
    print(f"{arguments.tag.upper()} — POOLED OUT-OF-FOLD".center(76))
    print("=" * 76)
    print(f"Patients            : {summary['patients']}  ({arguments.folds} folds)")
    print(f"Decisions           : {summary['decisions']:,}")
    print(f"Target seizures     : {summary['target_seizures']}")
    print(f"Prevalence          : {prevalence:.5f}")
    print(f"Average precision   : {ap:.4f}  ({summary['ap_over_chance']:.2f}x chance)")
    print(f"ROC AUC             : {summary['roc_auc']:.4f}")
    print(f"Brier score         : {summary['brier_score']:.5f}")
    print()
    print("Seizure-level sensitivity, 95% bootstrap CI over seizures:")
    for b in budgets:
        print(f"  <= {b['false_alarm_budget_per_hour']:>4} /h : "
              f"{b['seizure_sensitivity']:.3f}  "
              f"[{b['ci_low']:.3f}, {b['ci_high']:.3f}]   "
              f"{b['seizures_detected']}/{b['seizures_total']}")
    print()
    if summary["vigilance_at_1_per_hour"]:
        print()
        print("By vigilance at onset (<= 1 alarm/h):")
        for row in summary["vigilance_at_1_per_hour"]:
            print(f"  {row['vigilance']:8s} {row['detected']:3d}/{row['seizures']:<4d} "
                  f"= {row['sensitivity']:.3f}")
    print()
    print(f"Elapsed: {(time.perf_counter()-started)/60:.1f} min")
    print(f"Artefacts: {output}")


if __name__ == "__main__":
    main()
