r"""Train an EEGNet seizure detector: the pipeline's positive control.

Prediction cannot tell a broken pipeline from an absent preictal signal,
because nobody knows what the right answer is.  Detection can.  SeizeIT2 was
collected and published to validate wearable seizure detection, so a pipeline
that loads, filters, aligns and labels this data correctly must be able to do
it.  Failure convicts the pipeline; success clears it and moves the open
question to whether a preictal state exists at all.

The encoder is deliberately the same shape as ``BaselineEEGNet``'s -- EEGNet
over 5-second, 3-channel chunks producing a 32-dimensional embedding -- so a
detector trained here can be loaded straight into the prediction model.  That
matters because prediction offers 253 eligible training seizures while
detection offers every annotated seizure in the cohort, which makes this the
better place to learn an EEG representation.

Run ``scripts/build_detection_dataset.py`` first.

PowerShell example:

    .\.venv-old\Scripts\python.exe scripts\train_detection_baseline.py --device cuda
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIRECTORY = PROJECT_ROOT / "src"
if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

from seizure_prediction.config import CONFIG  # noqa: E402
from seizure_prediction.datasets import resolve_stored_path  # noqa: E402
from seizure_prediction.models import (  # noqa: E402
    BaselineEEGNet,
    BaselineEEGNetConfig,
)


SEPARATOR = "=" * 92


@dataclass(frozen=True)
class EvaluationResult:
    """Validation metrics for one epoch."""

    loss: float
    average_precision: float
    roc_auc: float


class DetectionWindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Read short labeled windows lazily from the processed recordings."""

    def __init__(self, windows: pd.DataFrame, window_samples: int) -> None:
        self.windows = windows.reset_index(drop=True)
        self.window_samples = window_samples
        self._signals: dict[str, np.ndarray] = {}
        self._availability: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.windows)

    def __getstate__(self) -> dict[str, object]:
        """Memory maps must not travel to DataLoader workers."""
        state = self.__dict__.copy()
        state["_signals"] = {}
        state["_availability"] = {}
        return state

    def _signal(self, path: str) -> np.ndarray:
        if path not in self._signals:
            self._signals[path] = np.load(path, mmap_mode="r")
        return self._signals[path]

    def _mask(self, path: str) -> np.ndarray:
        if path not in self._availability:
            with Path(path).open("r", encoding="utf-8") as handle:
                self._availability[path] = np.asarray(
                    json.load(handle), dtype=np.float32
                )
        return self._availability[path]

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.windows.iloc[index]
        signal = self._signal(str(row["X_path"]))
        start = int(row["window_start_sample"])
        stop = start + self.window_samples
        if stop > signal.shape[1]:
            raise ValueError("Detection window indexes past the recording.")
        window = np.asarray(signal[:, start:stop], dtype=np.float32)
        return (
            torch.from_numpy(np.ascontiguousarray(window)),
            torch.from_numpy(self._mask(str(row["channel_availability_path"])).copy()),
            torch.tensor(float(row["label"]), dtype=torch.float32),
        )


class DetectionEEGNet(nn.Module):
    """EEGNet over one short window, with the availability mask concatenated.

    The encoder is byte-compatible with ``BaselineEEGNet.encoder``, so its
    weights transfer into the prediction model unchanged.
    """

    def __init__(self, config: BaselineEEGNetConfig) -> None:
        super().__init__()
        baseline = BaselineEEGNet(config)
        self.config = config
        self.encoder = baseline.encoder
        self.classifier = baseline.classifier

    def forward(
        self,
        window: torch.Tensor,
        channel_availability: torch.Tensor,
    ) -> torch.Tensor:
        """Return one detection logit per window."""
        embedding = self.encoder(window)
        features = torch.cat(
            [
                embedding,
                channel_availability.to(
                    dtype=embedding.dtype,
                    device=embedding.device,
                ),
            ],
            dim=1,
        )
        return self.classifier(features).squeeze(1)


def parse_arguments() -> argparse.Namespace:
    """Parse training options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=CONFIG.manifests_dir / "detection_window_manifest_5s.csv",
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--positive-oversampling",
        type=float,
        default=10.0,
        help=(
            "Relative sampling weight for ictal windows. Detection prevalence "
            "is about 3.9%, so plain shuffling starves the positive class."
        ),
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "models" / "eegnet_detection",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    """Select the compute device."""
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def load_windows(manifest_path: Path) -> pd.DataFrame:
    """Load the detection manifest with resolved local paths."""
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Detection manifest not found: {manifest_path}. Run "
            "scripts/build_detection_dataset.py first."
        )
    windows = pd.read_csv(manifest_path, dtype={"subject": str})
    windows["subject"] = windows["subject"].astype(str).str.zfill(3)
    windows["X_path"] = windows["X_path"].map(
        lambda value: str(resolve_stored_path(value))
    )
    windows["channel_availability_path"] = windows[
        "channel_availability_path"
    ].map(lambda value: str(resolve_stored_path(value)))
    return windows


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> EvaluationResult:
    """Score the held-out patients."""
    model.eval()
    total_loss = 0.0
    seen = 0
    labels: list[float] = []
    scores: list[float] = []
    with torch.inference_mode():
        for window, availability, target in loader:
            window = window.to(device, non_blocking=True)
            availability = availability.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            logits = model(window, availability)
            loss = criterion(logits, target)
            total_loss += loss.item() * len(target)
            seen += len(target)
            labels.extend(target.cpu().tolist())
            scores.extend(torch.sigmoid(logits).cpu().tolist())
    label_values = np.asarray(labels, dtype=np.int64)
    score_values = np.asarray(scores, dtype=np.float64)
    if len(np.unique(label_values)) < 2:
        raise ValueError("Validation must contain both classes.")
    return EvaluationResult(
        loss=total_loss / seen,
        average_precision=float(
            average_precision_score(label_values, score_values)
        ),
        roc_auc=float(roc_auc_score(label_values, score_values)),
    )


def main() -> None:
    """Train the detector and report whether the pipeline can do the easy task."""
    arguments = parse_arguments()
    CONFIG.validate()
    set_seed(arguments.seed)
    device = resolve_device(arguments.device)
    arguments.output_dir.mkdir(parents=True, exist_ok=True)

    print(SEPARATOR)
    print("SEIZURE DETECTION POSITIVE CONTROL".center(92))
    print(SEPARATOR)

    windows = load_windows(arguments.manifest)
    train_windows = windows[windows["split"] == "train"]
    validation_windows = windows[windows["split"] == "validation"]
    if train_windows.empty or validation_windows.empty:
        raise ValueError("The manifest must contain train and validation windows.")

    window_samples = int(
        train_windows["window_stop_sample"].iloc[0]
        - train_windows["window_start_sample"].iloc[0]
    )
    prevalence = float(validation_windows["label"].mean())
    print(
        f"\ntrain {len(train_windows):,} windows "
        f"({int(train_windows['label'].sum()):,} ictal)   "
        f"validation {len(validation_windows):,} windows "
        f"({int(validation_windows['label'].sum()):,} ictal)"
    )
    print(
        f"window {window_samples / CONFIG.target_sfreq:g}s   "
        f"validation prevalence {prevalence:.4f}   "
        f"patients {windows['subject'].nunique()}"
    )
    print(
        "\nPatients are split exactly as in the prediction task, so a detector "
        "that works\nhere shows the loading, filtering, alignment and labeling "
        "are all sound.\n"
    )

    labels = train_windows["label"].to_numpy(dtype=np.int64)
    weights = np.where(labels == 1, arguments.positive_oversampling, 1.0)
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(train_windows),
        replacement=True,
    )
    train_loader = DataLoader(
        DetectionWindowDataset(train_windows, window_samples),
        batch_size=arguments.batch_size,
        sampler=sampler,
        num_workers=arguments.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=arguments.num_workers > 0,
    )
    validation_loader = DataLoader(
        DetectionWindowDataset(validation_windows, window_samples),
        batch_size=arguments.batch_size,
        shuffle=False,
        num_workers=arguments.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=arguments.num_workers > 0,
    )

    model_config = BaselineEEGNetConfig(
        n_chans=len(CONFIG.canonical_channel_names),
        chunk_samples=window_samples,
        embedding_dim=arguments.embedding_dim,
        dropout=arguments.dropout,
    )
    model = DetectionEEGNet(model_config).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
    )
    criterion = nn.BCEWithLogitsLoss()

    history: list[dict[str, float]] = []
    best_average_precision = float("-inf")
    best_auc = float("nan")
    best_epoch = 0
    checkpoint_path = arguments.output_dir / "best_model.pt"

    for epoch in range(1, arguments.epochs + 1):
        model.train()
        cumulative = 0.0
        seen = 0
        for window, availability, target in train_loader:
            window = window.to(device, non_blocking=True)
            availability = availability.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(window, availability), target)
            loss.backward()
            clip_grad_norm_(model.parameters(), arguments.max_grad_norm)
            optimizer.step()
            cumulative += loss.item() * len(target)
            seen += len(target)
        train_loss = cumulative / seen
        result = evaluate(model, validation_loader, criterion, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": result.loss,
                "validation_average_precision": result.average_precision,
                "validation_roc_auc": result.roc_auc,
            }
        )
        print(
            f"epoch {epoch:2d}  train {train_loss:.4f}  "
            f"val {result.loss:.4f}  "
            f"AP {result.average_precision:.4f} "
            f"({result.average_precision / prevalence:5.2f}x)  "
            f"AUC {result.roc_auc:.4f}"
        )
        if result.average_precision > best_average_precision:
            best_average_precision = result.average_precision
            best_auc = result.roc_auc
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": vars(model_config),
                    "epoch": epoch,
                    "validation_average_precision": result.average_precision,
                    "validation_roc_auc": result.roc_auc,
                },
                checkpoint_path,
            )

    summary = {
        "task": "seizure detection (positive control)",
        "window_seconds": window_samples / CONFIG.target_sfreq,
        "encoder": "EEGNet, shape-compatible with BaselineEEGNet.encoder",
        "train_windows": int(len(train_windows)),
        "train_ictal_windows": int(train_windows["label"].sum()),
        "validation_windows": int(len(validation_windows)),
        "validation_ictal_windows": int(validation_windows["label"].sum()),
        "validation_prevalence": prevalence,
        "best_epoch": best_epoch,
        "best_validation_average_precision": best_average_precision,
        "best_validation_lift_over_prevalence": best_average_precision / prevalence,
        "best_validation_roc_auc": best_auc,
        "history": history,
        "seed": arguments.seed,
        "held_out_test_used": False,
    }
    (arguments.output_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n" + SEPARATOR)
    print("VERDICT".center(92))
    print(SEPARATOR)
    print(
        f"\nbest AP {best_average_precision:.4f} "
        f"({best_average_precision / prevalence:.2f}x prevalence)  "
        f"AUC {best_auc:.4f}"
    )
    if best_auc >= 0.85:
        print(
            "\nThe pipeline performs the task it is known to support. Loading, "
            "filtering,\nalignment and labeling are sound, and a failure to "
            "predict is a statement about\npreictal physiology rather than "
            "about the code."
        )
    elif best_auc >= 0.70:
        print(
            "\nDetection works but is not strong. The pipeline is not broken; "
            "behind-the-ear\nEEG simply captures focal seizures weakly, which "
            "is expected for this montage."
        )
    else:
        print(
            "\nDetection is weak on a task this data was collected to support. "
            "This is the\none result that would point at the pipeline itself; "
            "check channel mapping,\nfiltering and window alignment before "
            "drawing any conclusion about prediction."
        )
    print(f"\nCheckpoint: {checkpoint_path}")
    print(f"Outputs   : {arguments.output_dir}")


if __name__ == "__main__":
    main()
