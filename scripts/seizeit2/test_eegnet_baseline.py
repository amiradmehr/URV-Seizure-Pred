r"""Evaluate a trained EEGNet baseline on the held-out test split.

`train_eegnet_baseline.py` never touches the test split -- it trains on
`train`, selects the best epoch by `validation` average precision, and says so
on its last line ("The held-out test split was not used."). That keeps the test
split genuinely held out: validation AP is used repeatedly for model selection
and comparison, so it is optimistically biased, while test AP is measured once
on data no training decision ever saw.

This script closes that loop. Point it at a checkpoint written by
`train_eegnet_baseline.py` and it reports that model's test-split metrics using
the same probability calibration training applied, so test AP is directly
comparable to the `best_validation_average_precision` in `metrics.json`.

Running out of disk on a training machine
-----------------------------------------

The standardized EEG for the test split is large (~15 GB for SeizeIT2), and on
a Colab runtime it usually will not fit alongside the already-extracted
training data. Pass `--shared-archive` one or more times to stream it instead:
recordings are extracted from those zips a batch at a time, evaluated, then
deleted before the next batch, so peak extra disk is one batch rather than the
whole split. Predictions are pooled across batches and the metrics are computed
once at the end, so batching changes nothing about the numbers.

Examples:

    # Recordings already extracted on disk.
    python scripts/seizeit2/test_eegnet_baseline.py \
        --window-minutes 30 --horizon-minutes 5 \
        --model-dir outputs/sweeps/<sweep>/models/w30_h5 --device cuda

    # Disk-constrained: stream the test recordings from Drive zips.
    python scripts/seizeit2/test_eegnet_baseline.py \
        --window-minutes 30 --horizon-minutes 5 \
        --model-dir outputs/sweeps/<sweep>/models/w30_h5 --device cuda \
        --shared-archive /content/drive/MyDrive/.../shared_standardized_recordings.zip \
        --shared-archive /content/drive/MyDrive/.../shared_standardized_recordings_supplement.zip
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIRECTORY = PROJECT_ROOT / "src"

if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))


from seizure_prediction.seizeit2.config import (  # noqa: E402
    PreprocessingConfig,
    build_config,
)
from seizure_prediction.datasets import (  # noqa: E402
    StreamingDecisionDataset,
    load_decision_examples,
)
from seizure_prediction.models import (  # noqa: E402
    BaselineEEGNet,
    BaselineEEGNetConfig,
)


def parse_arguments() -> argparse.Namespace:
    """Parse which checkpoint to evaluate and how to reach the test EEG."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--window-minutes",
        type=float,
        default=None,
        help="Must match the value the checkpoint was trained with.",
    )
    parser.add_argument(
        "--horizon-minutes",
        type=float,
        default=None,
        help="Must match the value the checkpoint was trained with.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help=(
            "Directory holding best_model.pt (as written by "
            "train_eegnet_baseline.py --output-dir). Ignored if --checkpoint "
            "is given."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to best_model.pt. Defaults to <model-dir>/best_model.pt.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where test_metrics.json and the figure go. Defaults to the checkpoint's directory.",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--shared-archive",
        type=Path,
        action="append",
        default=[],
        help=(
            "Zip holding standardized recordings, e.g. "
            "shared_standardized_recordings.zip. Repeatable; later archives "
            "win on duplicate members. When given, test recordings are "
            "extracted a batch at a time and deleted after use, so the whole "
            "test split never has to fit on disk at once."
        ),
    )
    parser.add_argument(
        "--recordings-per-batch",
        type=int,
        default=40,
        help=(
            "Recordings extracted per streaming batch. Larger batches mean "
            "fewer archive open/seek cycles but more peak disk."
        ),
    )
    parser.add_argument(
        "--keep-extracted",
        action="store_true",
        help=(
            "Do not delete streamed recordings after evaluating them. Only "
            "useful when disk is plentiful and you expect to re-run."
        ),
    )
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=2.0,
        help="Abort before extracting a batch that would leave less than this free.",
    )
    return parser.parse_args()


def resolve_device(requested_device: str) -> torch.device:
    """Select CUDA when available unless the caller explicitly requests CPU."""
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available.")
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested_device)


def load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[BaselineEEGNet, float, dict]:
    """Rebuild the trained model exactly as the checkpoint describes it.

    The architecture is taken from the checkpoint's own `model_config` rather
    than re-derived from the preprocessing config, so a mismatch between the
    two can never silently reshape the model under the saved weights.

    Returns the model, the sampling-prior logit correction to apply to its raw
    logits, and the rest of the checkpoint for provenance. `BaselineEEGNet`
    only applies that correction inside `predict_proba`; `forward` returns raw
    logits, so the caller must add it exactly once -- the same thing
    `train_eegnet_baseline.evaluate` does.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = BaselineEEGNetConfig(**checkpoint["model_config"])
    model = BaselineEEGNet(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, float(model_config.sampling_prior_logit_correction), checkpoint


def index_archives(archive_paths: list[Path]) -> dict[str, tuple[Path, int]]:
    """Map member name to (archive, uncompressed size) across every archive."""
    index: dict[str, tuple[Path, int]] = {}
    for archive_path in archive_paths:
        if not archive_path.exists():
            raise FileNotFoundError(f"Shared archive not found: {archive_path}")
        with zipfile.ZipFile(archive_path) as archive:
            for info in archive.infolist():
                index[info.filename] = (archive_path, info.file_size)
    return index


def extract_members(
    index: dict[str, tuple[Path, int]],
    member_names: list[str],
    destination: Path,
    minimum_free_gb: float,
) -> list[Path]:
    """Extract `member_names` under `destination`, returning what was written.

    Members already present at their full expected size are left alone, so a
    re-run after an interruption does not re-transfer them.
    """
    pending = [
        name
        for name in member_names
        if not (
            (destination / name).exists()
            and (destination / name).stat().st_size == index[name][1]
        )
    ]
    if not pending:
        return []

    required_bytes = sum(index[name][1] for name in pending)
    free_bytes = shutil.disk_usage(destination).free
    if free_bytes - required_bytes < minimum_free_gb * 1e9:
        raise RuntimeError(
            f"Extracting {len(pending)} recording(s) needs "
            f"{required_bytes / 1e9:.1f} GB but only {free_bytes / 1e9:.1f} GB "
            f"is free, which would leave less than the {minimum_free_gb:.1f} GB "
            "floor. Lower --recordings-per-batch, or free space."
        )

    written: list[Path] = []
    by_archive: dict[Path, list[str]] = {}
    for name in pending:
        by_archive.setdefault(index[name][0], []).append(name)

    for archive_path, names in by_archive.items():
        with zipfile.ZipFile(archive_path) as archive:
            for name in names:
                (destination / name).parent.mkdir(parents=True, exist_ok=True)
                archive.extract(name, destination)
                written.append(destination / name)
    return written


def predict_split(
    model: BaselineEEGNet,
    examples: pd.DataFrame,
    config: PreprocessingConfig,
    criterion: nn.Module,
    device: torch.device,
    logit_correction: float,
    *,
    batch_size: int,
    num_workers: int,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Run the model over `examples`, returning labels, probabilities, and loss.

    Deliberately computes no metrics: a streaming batch of recordings can
    easily contain a single class, and average precision is only meaningful
    over the pooled split. The caller accumulates these and scores once.
    """
    loader = DataLoader(
        StreamingDecisionDataset(examples, config),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
    )

    labels: list[float] = []
    probabilities: list[float] = []
    total_loss = 0.0
    total_examples = 0

    with torch.no_grad():
        for signal, availability, target in loader:
            signal = signal.to(device, non_blocking=True)
            availability = availability.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            corrected_logits = model(signal, availability) + logit_correction
            loss = criterion(corrected_logits, target)

            total_loss += loss.item() * len(target)
            total_examples += len(target)
            labels.extend(target.cpu().tolist())
            probabilities.extend(torch.sigmoid(corrected_logits).cpu().tolist())

    return (
        np.asarray(labels, dtype=np.int64),
        np.asarray(probabilities, dtype=np.float64),
        total_loss,
        total_examples,
    )


def summarize_predictions(
    labels: np.ndarray,
    probabilities: np.ndarray,
    mean_loss: float,
) -> dict[str, float | int]:
    """Score pooled test predictions.

    Average precision is the primary number, chosen to match the metric
    training selects on so test and validation AP can be read side by side.
    Prevalence is reported alongside it because AP has no fixed baseline: a
    useless model scores about the positive rate, so AP must be judged as a
    lift over that, not against zero.
    """
    positive_count = int((labels == 1).sum())
    negative_count = int((labels == 0).sum())
    prevalence = positive_count / len(labels)

    average_precision = float(average_precision_score(labels, probabilities))
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)

    # precision_recall_curve returns one more precision/recall point than
    # thresholds; drop that trailing point so the arrays line up.
    f1_scores = np.divide(
        2 * precision[:-1] * recall[:-1],
        precision[:-1] + recall[:-1],
        out=np.zeros_like(precision[:-1]),
        where=(precision[:-1] + recall[:-1]) > 0,
    )
    best_index = int(np.argmax(f1_scores)) if len(f1_scores) else 0

    return {
        "test_decisions": int(len(labels)),
        "test_positive_decisions": positive_count,
        "test_negative_decisions": negative_count,
        "test_prevalence": prevalence,
        "test_loss": mean_loss,
        "test_average_precision": average_precision,
        "test_average_precision_lift_over_prevalence": (
            average_precision / prevalence if prevalence > 0 else float("nan")
        ),
        "test_roc_auc": float(roc_auc_score(labels, probabilities)),
        "test_best_f1": float(f1_scores[best_index]) if len(f1_scores) else 0.0,
        "test_best_f1_threshold": (
            float(thresholds[best_index]) if len(thresholds) else float("nan")
        ),
        "test_precision_at_best_f1": (
            float(precision[best_index]) if len(f1_scores) else 0.0
        ),
        "test_recall_at_best_f1": (
            float(recall[best_index]) if len(f1_scores) else 0.0
        ),
    }


def save_test_summary(
    labels: np.ndarray,
    probabilities: np.ndarray,
    summary: dict[str, float | int],
    experiment_tag: str,
    output_path: Path,
) -> None:
    """Draw the test precision-recall curve against its prevalence baseline."""
    precision, recall, _ = precision_recall_curve(labels, probabilities)
    prevalence = float(summary["test_prevalence"])

    figure, axis = plt.subplots(figsize=(7, 5))
    axis.plot(
        recall,
        precision,
        color="#2c7fb8",
        label=f"AP = {summary['test_average_precision']:.4f}",
    )
    axis.axhline(
        prevalence,
        color="#999999",
        linestyle="--",
        label=f"Prevalence baseline = {prevalence:.4f}",
    )
    axis.set(
        title=f"Held-out test precision-recall ({experiment_tag or 'baseline'})",
        xlabel="Recall",
        ylabel="Precision",
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.0),
    )
    axis.grid(alpha=0.3)
    axis.legend(loc="upper right")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def main() -> None:
    """Evaluate one trained checkpoint on the held-out test split."""
    arguments = parse_arguments()

    if arguments.checkpoint is not None:
        checkpoint_path = arguments.checkpoint
    elif arguments.model_dir is not None:
        checkpoint_path = arguments.model_dir / "best_model.pt"
    else:
        raise ValueError("Pass either --checkpoint or --model-dir.")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    config = build_config(arguments.window_minutes, arguments.horizon_minutes)
    device = resolve_device(arguments.device)
    output_dir = arguments.output_dir or checkpoint_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = config.manifests_dir / "processed_shard_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Processed manifest not found: {manifest_path}. Unpack this "
            "combination's package first."
        )

    model, logit_correction, checkpoint = load_model(checkpoint_path, device)
    criterion = nn.BCEWithLogitsLoss()

    print(f"Checkpoint: {checkpoint_path}")
    print(f"  trained to epoch {checkpoint.get('best_epoch')} with validation AP "
          f"{checkpoint.get('best_validation_average_precision')}")
    print(f"  logit correction: {logit_correction:+.6f}")
    print(f"Device: {device}")

    examples = load_decision_examples(
        manifest_path,
        split="test",
        negative_to_positive_ratio=None,
        seed=0,
        project_root=config.project_root,
    )
    print(f"Test decisions: {len(examples)} across "
          f"{examples['X_path'].nunique()} recording(s)")

    # Group decisions by recording so a streaming batch extracts, evaluates,
    # and releases whole recordings -- a decision's EEG is a slice of one
    # recording, so a partially extracted recording is useless.
    recording_paths = list(dict.fromkeys(examples["X_path"].tolist()))

    archive_index: dict[str, tuple[Path, int]] = {}
    if arguments.shared_archive:
        archive_index = index_archives(arguments.shared_archive)
        print(f"Streaming test EEG from {len(arguments.shared_archive)} archive(s); "
              f"{arguments.recordings_per_batch} recording(s) per batch.")

    batch_size_recordings = (
        arguments.recordings_per_batch if archive_index else len(recording_paths)
    )

    all_labels: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []
    running_loss = 0.0
    running_examples = 0

    for batch_start in range(0, len(recording_paths), batch_size_recordings):
        batch_paths = recording_paths[batch_start : batch_start + batch_size_recordings]
        batch_examples = examples[examples["X_path"].isin(batch_paths)]

        extracted: list[Path] = []
        if archive_index:
            member_names = [
                Path(path).resolve().relative_to(config.project_root).as_posix()
                for path in batch_paths
            ]
            missing = [name for name in member_names if name not in archive_index]
            if missing:
                raise KeyError(
                    f"{len(missing)} test recording(s) are in none of the given "
                    f"archives, e.g. {missing[:3]}."
                )
            extracted = extract_members(
                archive_index,
                member_names,
                config.project_root,
                arguments.min_free_gb,
            )

        labels, probabilities, batch_loss, batch_count = predict_split(
            model,
            batch_examples,
            config,
            criterion,
            device,
            logit_correction,
            batch_size=arguments.batch_size,
            num_workers=arguments.num_workers,
        )
        all_labels.append(labels)
        all_probabilities.append(probabilities)
        running_loss += batch_loss
        running_examples += batch_count

        if extracted and not arguments.keep_extracted:
            for path in extracted:
                path.unlink(missing_ok=True)

        done = min(batch_start + batch_size_recordings, len(recording_paths))
        print(
            f"  {done}/{len(recording_paths)} recordings evaluated "
            f"({running_examples} decisions, "
            f"{shutil.disk_usage(config.project_root).free / 1e9:.1f} GB free)",
            flush=True,
        )

    labels = np.concatenate(all_labels)
    probabilities = np.concatenate(all_probabilities)

    if len(np.unique(labels)) != 2:
        raise ValueError(
            "The test split must contain both classes to score average "
            f"precision; found only label(s) {np.unique(labels).tolist()}."
        )

    summary = summarize_predictions(
        labels, probabilities, running_loss / running_examples
    )
    summary_record: dict[str, object] = {
        "experiment_tag": config.experiment_tag,
        "window_minutes": config.input_window_seconds / 60.0,
        # The sweep's "horizon" is the seizure-occurrence period -- the window
        # a seizure must fall in for a decision to count positive -- which is
        # what build_config overrides. `prediction_horizon_minutes` is a
        # separate (and here always zero) minimum-warning-time setting.
        "horizon_minutes": config.seizure_occurrence_period_minutes,
        "checkpoint": str(checkpoint_path),
        "best_epoch": checkpoint.get("best_epoch"),
        "best_validation_average_precision": checkpoint.get(
            "best_validation_average_precision"
        ),
        "sampling_prior_logit_correction": logit_correction,
        "primary_comparison_metric": "test_average_precision",
        **summary,
    }

    metrics_path = output_dir / "test_metrics.json"
    metrics_path.write_text(json.dumps(summary_record, indent=2), encoding="utf-8")

    figure_path = output_dir / "test_summary.png"
    save_test_summary(
        labels, probabilities, summary, config.experiment_tag or "", figure_path
    )

    print()
    print(f"Test AP:        {summary['test_average_precision']:.6f}")
    print(f"  prevalence:   {summary['test_prevalence']:.6f} "
          f"({summary['test_positive_decisions']}/{summary['test_decisions']} positive)")
    print(f"  lift over it: {summary['test_average_precision_lift_over_prevalence']:.2f}x")
    print(f"Test ROC AUC:   {summary['test_roc_auc']:.6f}")
    print(f"Test loss:      {summary['test_loss']:.6f}")
    print(f"Best F1:        {summary['test_best_f1']:.6f} at threshold "
          f"{summary['test_best_f1_threshold']:.6f}")
    validation_average_precision = checkpoint.get("best_validation_average_precision")
    if validation_average_precision is not None:
        print(
            f"Validation AP was {validation_average_precision:.6f}; "
            "a large drop here is the usual sign that selection overfit it."
        )
    print()
    print(f"Metrics: {metrics_path}")
    print(f"Figure:  {figure_path}")


if __name__ == "__main__":
    main()
