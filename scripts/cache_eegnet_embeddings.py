r"""Cache continuous five-second EEGNet embeddings for temporal experiments.

The processed recordings stay unchanged.  Each cache file stores one compact
embedding per non-overlapping five-second chunk, so overlapping 45-minute
decision histories can reuse the same encoder output.  This makes temporal-head
training practical on a local GPU without creating a materialized window copy.

PowerShell example:

    .\.venv-win\Scripts\python.exe scripts\cache_eegnet_embeddings.py --device cuda
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIRECTORY = PROJECT_ROOT / "src"

if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))


from seizure_prediction.config import CONFIG  # noqa: E402
from seizure_prediction.datasets import (  # noqa: E402
    embedding_cache_path,
    resolve_stored_path,
)
from seizure_prediction.models import (  # noqa: E402
    BaselineEEGNet,
    BaselineEEGNetConfig,
)


def parse_arguments() -> argparse.Namespace:
    """Return cache-generation settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-checkpoint",
        type=Path,
        default=(
            PROJECT_ROOT
            / "outputs"
            / "models"
            / "eegnet_baseline_ratio10_lr1e4"
            / "best_model.pt"
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "embedding_cache"
            / "eegnet_baseline_ratio10_lr1e4"
        ),
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "validation", "test"),
        default=("train", "validation"),
    )
    parser.add_argument("--chunk-batch-size", type=int, default=512)
    parser.add_argument(
        "--storage-dtype",
        choices=("float16", "float32"),
        default="float32",
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use CUDA mixed precision while encoding. Disabled by default so "
            "the cache matches the float32 baseline evaluation."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    return parser.parse_args()


def resolve_device(requested_device: str) -> torch.device:
    """Select the requested accelerator and fail clearly when CUDA is absent."""
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available.")
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested_device)


def sha256_file(path: Path) -> str:
    """Return a stable fingerprint for the encoder checkpoint."""
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint_file:
        for block in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_baseline(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[BaselineEEGNet, dict[str, object]]:
    """Restore the exact encoder selected by the baseline validation run."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Baseline checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if "model_state_dict" not in checkpoint or "model_config" not in checkpoint:
        raise ValueError("The checkpoint does not contain a BaselineEEGNet model.")
    model_config = BaselineEEGNetConfig(**checkpoint["model_config"])
    model = BaselineEEGNet(model_config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model, checkpoint


def cache_is_valid(
    cache_path: Path,
    expected_chunks: int,
    embedding_dim: int,
    expected_dtype: np.dtype,
) -> bool:
    """Return whether an existing cache has the expected storage contract."""
    try:
        cached = np.load(cache_path, mmap_mode="r")
    except (OSError, ValueError):
        return False
    return (
        cached.shape == (expected_chunks, embedding_dim)
        and cached.dtype == expected_dtype
    )


def encode_recording(
    *,
    signal_path: Path,
    cache_path: Path,
    model: BaselineEEGNet,
    device: torch.device,
    chunk_batch_size: int,
    storage_dtype: np.dtype,
    amp_enabled: bool,
    overwrite: bool,
) -> tuple[int, bool]:
    """Encode one continuous recording and atomically publish its cache."""
    signal = np.load(signal_path, mmap_mode="r")
    expected_channels = model.config.n_chans
    if signal.ndim != 2 or signal.shape[0] != expected_channels:
        raise ValueError(
            f"Unexpected processed signal shape in {signal_path}: {signal.shape}."
        )

    chunk_samples = model.config.chunk_samples
    number_of_chunks = signal.shape[1] // chunk_samples
    if number_of_chunks == 0:
        raise ValueError(f"Recording is shorter than one chunk: {signal_path}")
    if cache_path.exists() and not overwrite:
        if cache_is_valid(
            cache_path,
            number_of_chunks,
            model.config.embedding_dim,
            storage_dtype,
        ):
            return number_of_chunks, True
        raise ValueError(
            f"Existing cache is incompatible: {cache_path}. "
            "Pass --overwrite to rebuild it."
        )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_path.with_name(f"{cache_path.stem}.partial.npy")
    cached_embeddings = np.lib.format.open_memmap(
        temporary_path,
        mode="w+",
        dtype=storage_dtype,
        shape=(number_of_chunks, model.config.embedding_dim),
    )

    try:
        with torch.inference_mode():
            for start_chunk in range(0, number_of_chunks, chunk_batch_size):
                end_chunk = min(start_chunk + chunk_batch_size, number_of_chunks)
                start_sample = start_chunk * chunk_samples
                end_sample = end_chunk * chunk_samples
                signal_batch = np.asarray(
                    signal[:, start_sample:end_sample],
                    dtype=np.float32,
                ).reshape(
                    expected_channels,
                    end_chunk - start_chunk,
                    chunk_samples,
                ).transpose(1, 0, 2)
                signal_tensor = torch.from_numpy(
                    np.ascontiguousarray(signal_batch)
                ).to(device, non_blocking=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=amp_enabled,
                ):
                    embedding_batch = model.encoder(signal_tensor)
                cached_embeddings[start_chunk:end_chunk] = (
                    embedding_batch.float().cpu().numpy().astype(
                        storage_dtype,
                        copy=False,
                    )
                )
        cached_embeddings.flush()
    except BaseException:
        del cached_embeddings
        temporary_path.unlink(missing_ok=True)
        raise
    del cached_embeddings
    try:
        temporary_path.replace(cache_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return number_of_chunks, False


def main() -> None:
    """Build or resume the compact embedding cache."""
    arguments = parse_arguments()
    if arguments.chunk_batch_size <= 0:
        raise ValueError("chunk-batch-size must be positive.")
    CONFIG.validate()
    device = resolve_device(arguments.device)
    amp_enabled = bool(arguments.amp and device.type == "cuda")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    model, checkpoint = load_baseline(arguments.baseline_checkpoint, device)
    manifest_path = CONFIG.manifests_dir / "processed_shard_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Processed manifest not found: {manifest_path}")
    manifest = pd.read_csv(manifest_path, dtype={"subject": str})
    manifest = manifest[manifest["split"].isin(arguments.splits)].copy()
    if manifest.empty:
        raise ValueError(f"No recordings found for splits {arguments.splits}.")

    storage_dtype = np.dtype(arguments.storage_dtype)
    unique_recordings = manifest.drop_duplicates(subset=["X_path"]).reset_index(
        drop=True
    )
    total_chunks = 0
    skipped_recordings = 0
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Recordings to cache: {len(unique_recordings)}")
    print(f"Cache directory: {arguments.cache_dir.resolve()}")
    print(f"Storage dtype: {storage_dtype}; AMP encoding: {amp_enabled}")

    for index, row in unique_recordings.iterrows():
        signal_path = resolve_stored_path(row["X_path"])
        cache_path = embedding_cache_path(signal_path, arguments.cache_dir)
        number_of_chunks, skipped = encode_recording(
            signal_path=signal_path,
            cache_path=cache_path,
            model=model,
            device=device,
            chunk_batch_size=arguments.chunk_batch_size,
            storage_dtype=storage_dtype,
            amp_enabled=amp_enabled,
            overwrite=arguments.overwrite,
        )
        total_chunks += number_of_chunks
        skipped_recordings += int(skipped)
        action = "verified" if skipped else "encoded"
        print(
            f"[{index + 1:04d}/{len(unique_recordings):04d}] "
            f"{action}: {signal_path.name} ({number_of_chunks:,} chunks)",
            flush=True,
        )

    metadata = {
        "baseline_checkpoint": str(arguments.baseline_checkpoint.resolve()),
        "baseline_checkpoint_sha256": sha256_file(
            arguments.baseline_checkpoint
        ),
        "baseline_best_validation_average_precision": checkpoint.get(
            "best_validation_average_precision"
        ),
        "splits": list(arguments.splits),
        "recordings": len(unique_recordings),
        "verified_existing_recordings": skipped_recordings,
        "chunks": total_chunks,
        "chunk_samples": model.config.chunk_samples,
        "embedding_dim": model.config.embedding_dim,
        "storage_dtype": str(storage_dtype),
    }
    arguments.cache_dir.mkdir(parents=True, exist_ok=True)
    # A rebuilt cache matches the current data/processed again, so the
    # staleness marker left by scripts/restandardize_processed.py no longer
    # applies. Leaving it would make consumers refuse a valid cache.
    stale_marker = arguments.cache_dir / "STALE.txt"
    if stale_marker.exists():
        stale_marker.unlink()
        print(f"Cleared staleness marker: {stale_marker}")
    (arguments.cache_dir / "cache_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2))
    print("Embedding cache is ready.")


if __name__ == "__main__":
    main()
