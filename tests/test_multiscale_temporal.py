"""Regression tests for the multi-scale temporal EEGNet."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from seizure_prediction.config import CONFIG
from seizure_prediction.datasets import (
    CachedEmbeddingDecisionDataset,
    embedding_cache_path,
)
from seizure_prediction.models import (
    BaselineEEGNet,
    BaselineEEGNetConfig,
    EEGNetMultiScaleTemporalConfig,
    EEGNetMultiScaleTemporalRiskModel,
)


class MultiScaleTemporalModelTests(unittest.TestCase):
    """Protect the baseline bypass and causal temporal feature contract."""

    def setUp(self) -> None:
        torch.manual_seed(7)
        self.baseline_config = BaselineEEGNetConfig(
            n_chans=3,
            chunk_samples=1280,
            sequence_chunks=8,
            embedding_dim=8,
            encoder_chunk_batch_size=16,
            dropout=0.0,
        )
        self.temporal_config = EEGNetMultiScaleTemporalConfig(
            n_chans=3,
            chunk_samples=1280,
            sequence_chunks=8,
            chunks_per_minute=2,
            embedding_dim=8,
            temporal_hidden_dim=4,
            temporal_windows_minutes=(1, 2, 4),
            encoder_chunk_batch_size=16,
            encoder_dropout=0.0,
            temporal_dropout=0.0,
        )

    def test_baseline_checkpoint_initialization_preserves_logits(self) -> None:
        """A zero temporal residual must reproduce the baseline prediction."""
        baseline = BaselineEEGNet(self.baseline_config).eval()
        temporal = EEGNetMultiScaleTemporalRiskModel(self.temporal_config).eval()
        temporal.initialize_from_baseline_state_dict(baseline.state_dict())
        signal = torch.randn(2, 8, 3, 1280)
        availability = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
        with torch.inference_mode():
            baseline_logits = baseline(signal, availability)
            temporal_logits = temporal(signal, availability)
        torch.testing.assert_close(
            temporal_logits,
            baseline_logits,
            rtol=1e-5,
            atol=1e-5,
        )

    def test_temporal_contrasts_change_when_minute_order_changes(self) -> None:
        """The temporal branch must distinguish histories with equal means."""
        model = EEGNetMultiScaleTemporalRiskModel(self.temporal_config)
        minute_values = torch.arange(4, dtype=torch.float32).reshape(1, 4, 1)
        minute_embeddings = minute_values.expand(1, 4, 8)
        reversed_embeddings = minute_embeddings.flip(dims=(1,))
        global_mean, temporal_features = model._multi_scale_features(
            minute_embeddings
        )
        reversed_global_mean, reversed_features = model._multi_scale_features(
            reversed_embeddings
        )
        torch.testing.assert_close(global_mean, reversed_global_mean)
        self.assertFalse(torch.allclose(temporal_features, reversed_features))

    def test_frozen_baseline_leaves_only_temporal_parameters_trainable(self) -> None:
        """The first experiment must not overwrite the selected baseline."""
        baseline = BaselineEEGNet(self.baseline_config)
        model = EEGNetMultiScaleTemporalRiskModel(self.temporal_config)
        model.initialize_from_baseline_state_dict(baseline.state_dict())
        model.set_baseline_trainable(False)
        self.assertFalse(any(p.requires_grad for p in model.encoder.parameters()))
        self.assertFalse(
            any(p.requires_grad for p in model.baseline_classifier.parameters())
        )
        self.assertTrue(any(p.requires_grad for p in model.temporal_head.parameters()))

        embeddings = torch.randn(4, 8, 8)
        availability = torch.ones(4, 3)
        targets = torch.tensor([0.0, 1.0, 0.0, 1.0])
        loss = nn.BCEWithLogitsLoss()(
            model.forward_from_chunk_embeddings(embeddings, availability),
            targets,
        )
        loss.backward()
        final_layer = model.temporal_head[-1]
        self.assertIsNotNone(final_layer.weight.grad)
        self.assertGreater(float(final_layer.weight.grad.abs().sum()), 0.0)

    def test_temporal_head_learns_order_when_global_means_match(self) -> None:
        """A temporal-only signal must be learnable without changing EEGNet."""
        model = EEGNetMultiScaleTemporalRiskModel(self.temporal_config)
        model.set_baseline_trainable(False)
        positive_pattern = torch.cat(
            [
                -torch.ones(4, 8),
                torch.ones(4, 8),
            ],
            dim=0,
        )
        negative_pattern = positive_pattern.flip(dims=(0,))
        embeddings = torch.stack(
            [positive_pattern, negative_pattern] * 8,
            dim=0,
        )
        availability = torch.ones(16, 3)
        targets = torch.tensor([1.0, 0.0] * 8)
        optimizer = torch.optim.Adam(model.temporal_head.parameters(), lr=0.03)
        criterion = nn.BCEWithLogitsLoss()
        with torch.no_grad():
            initial_loss = criterion(
                model.forward_from_chunk_embeddings(embeddings, availability),
                targets,
            )
        for _ in range(80):
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(
                model.forward_from_chunk_embeddings(embeddings, availability),
                targets,
            )
            loss.backward()
            optimizer.step()
        final_loss = float(loss.detach())
        self.assertLess(final_loss, 0.1)
        self.assertLess(final_loss, float(initial_loss) / 5.0)

    def test_temporal_windows_must_end_at_full_history(self) -> None:
        """Reject a temporal pyramid that silently omits old context."""
        invalid_config = EEGNetMultiScaleTemporalConfig(
            sequence_chunks=8,
            chunks_per_minute=2,
            temporal_windows_minutes=(1, 2),
        )
        with self.assertRaisesRegex(ValueError, "complete input history"):
            EEGNetMultiScaleTemporalRiskModel(invalid_config)

    def test_cached_dataset_extracts_aligned_embedding_history(self) -> None:
        """The cache loader must map sample indices to chunk indices exactly."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            signal_path = temporary_root / "train" / "demo_X.npy"
            signal_path.parent.mkdir(parents=True)
            np.save(signal_path, np.zeros((3, 16), dtype=np.float32))
            availability_path = temporary_root / "train" / "availability.json"
            availability_path.write_text("[1, 1, 0]", encoding="utf-8")
            cache_root = temporary_root / "cache"
            cache_path = embedding_cache_path(signal_path, cache_root)
            cache_path.parent.mkdir(parents=True)
            expected_embeddings = np.arange(32, dtype=np.float32).reshape(8, 4)
            np.save(cache_path, expected_embeddings)

            examples = pd.DataFrame(
                [
                    {
                        "X_path": str(signal_path),
                        "channel_availability_path": str(availability_path),
                        "history_start_sample": 4,
                        "decision_end_sample": 12,
                        "label": 1,
                    }
                ]
            )
            test_config = replace(
                CONFIG,
                target_sfreq=2.0,
                chunk_window_seconds=1.0,
                input_window_seconds=4.0,
            )
            dataset = CachedEmbeddingDecisionDataset(
                examples,
                test_config,
                cache_root=cache_root,
                embedding_dim=4,
            )
            embeddings, availability, target = dataset[0]
            torch.testing.assert_close(
                embeddings,
                torch.from_numpy(expected_embeddings[2:6]),
            )
            torch.testing.assert_close(
                availability,
                torch.tensor([1.0, 1.0, 0.0]),
            )
            self.assertEqual(float(target), 1.0)
            dataset.close()


if __name__ == "__main__":
    unittest.main()
