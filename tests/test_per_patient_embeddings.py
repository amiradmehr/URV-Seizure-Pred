"""Tests for the pooled-embedding representation used by per-patient models."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from seizure_prediction.config import CONFIG  # noqa: E402
from train_per_patient_loso import build_embedding_feature_matrix  # noqa: E402


EMBEDDING_DIM = 32
CHUNK_SAMPLES = int(round(CONFIG.chunk_window_seconds * CONFIG.target_sfreq))
HISTORY_CHUNKS = int(round(CONFIG.input_window_seconds / CONFIG.chunk_window_seconds))


class PooledEmbeddingTests(unittest.TestCase):
    """Cover the prefix-sum pooling against a direct mean."""

    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.root = Path(self._directory.name)
        self.cache_root = self.root / "cache"
        (self.cache_root / "train").mkdir(parents=True)

        self.signal_path = self.root / "train" / "rec-01_train_X.npy"
        self.signal_path.parent.mkdir(parents=True, exist_ok=True)
        self.signal_path.write_bytes(b"")

        rng = np.random.default_rng(0)
        self.embeddings = rng.normal(
            size=(HISTORY_CHUNKS * 3, EMBEDDING_DIM)
        ).astype(np.float32)
        cache_path = self.cache_root / "train" / "rec-01_train_X_eegnet_embeddings.npy"
        np.save(cache_path, self.embeddings)

        self.availability_path = self.root / "avail.json"
        self.availability_path.write_text(json.dumps([1, 0, 1]), encoding="utf-8")

    def tearDown(self) -> None:
        self._directory.cleanup()

    def make_examples(self, start_chunks: list[int]) -> pd.DataFrame:
        """Build decisions whose histories start at the given chunk offsets."""
        return pd.DataFrame(
            [
                {
                    "X_path": str(self.signal_path),
                    "channel_availability_path": str(self.availability_path),
                    "history_start_sample": start * CHUNK_SAMPLES,
                    "decision_end_sample": (start + HISTORY_CHUNKS) * CHUNK_SAMPLES,
                }
                for start in start_chunks
            ]
        )

    def test_pooling_matches_a_direct_mean(self) -> None:
        """The prefix sum must reproduce a plain mean over the history."""
        starts = [0, 5, HISTORY_CHUNKS, HISTORY_CHUNKS * 2]
        features = build_embedding_feature_matrix(
            self.make_examples(starts),
            self.cache_root,
            EMBEDDING_DIM,
        )
        self.assertEqual(features.shape, (len(starts), EMBEDDING_DIM + 3))
        for row, start in enumerate(starts):
            expected = self.embeddings[start : start + HISTORY_CHUNKS].mean(axis=0)
            np.testing.assert_allclose(
                features[row, :EMBEDDING_DIM],
                expected,
                rtol=1e-4,
                atol=1e-5,
            )

    def test_availability_is_appended(self) -> None:
        """The classifier needs the electrode mask, as the baseline does."""
        features = build_embedding_feature_matrix(
            self.make_examples([0]),
            self.cache_root,
            EMBEDDING_DIM,
        )
        np.testing.assert_array_equal(
            features[0, EMBEDDING_DIM:],
            np.array([1.0, 0.0, 1.0], dtype=np.float32),
        )

    def test_row_order_is_preserved(self) -> None:
        """Rows must line up with the examples frame, not the group order."""
        starts = [HISTORY_CHUNKS * 2, 0, HISTORY_CHUNKS]
        features = build_embedding_feature_matrix(
            self.make_examples(starts),
            self.cache_root,
            EMBEDDING_DIM,
        )
        for row, start in enumerate(starts):
            expected = self.embeddings[start : start + HISTORY_CHUNKS].mean(axis=0)
            np.testing.assert_allclose(
                features[row, :EMBEDDING_DIM],
                expected,
                rtol=1e-4,
                atol=1e-5,
            )

    def test_misaligned_history_is_rejected(self) -> None:
        """A history that does not land on chunk boundaries must fail."""
        examples = self.make_examples([0])
        examples.loc[0, "history_start_sample"] = 7
        with self.assertRaises(ValueError):
            build_embedding_feature_matrix(examples, self.cache_root, EMBEDDING_DIM)

    def test_history_past_the_cache_is_rejected(self) -> None:
        """Indexing beyond the cached chunks must fail rather than wrap."""
        examples = self.make_examples([HISTORY_CHUNKS * 3])
        with self.assertRaises(ValueError):
            build_embedding_feature_matrix(examples, self.cache_root, EMBEDDING_DIM)

    def test_missing_cache_is_reported(self) -> None:
        """A missing cache file names the script that produces it."""
        examples = self.make_examples([0])
        examples.loc[0, "X_path"] = str(self.root / "train" / "absent_train_X.npy")
        with self.assertRaises(FileNotFoundError):
            build_embedding_feature_matrix(examples, self.cache_root, EMBEDDING_DIM)


if __name__ == "__main__":
    unittest.main()
