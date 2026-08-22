"""Regression tests for constrained recency-weighted EEGNet pooling."""

from __future__ import annotations

import unittest

import torch
from torch import nn

from seizure_prediction.models import (
    BaselineEEGNet,
    BaselineEEGNetConfig,
    EEGNetRecencyWeightedConfig,
    EEGNetRecencyWeightedRiskModel,
)


class RecencyWeightedModelTests(unittest.TestCase):
    """Protect baseline equivalence and the recency constraints."""

    def setUp(self) -> None:
        torch.manual_seed(11)
        self.baseline_config = BaselineEEGNetConfig(
            n_chans=3,
            chunk_samples=1280,
            sequence_chunks=8,
            embedding_dim=8,
            encoder_chunk_batch_size=16,
            dropout=0.0,
        )
        self.recency_config = EEGNetRecencyWeightedConfig(
            n_chans=3,
            chunk_samples=1280,
            sequence_chunks=8,
            chunks_per_minute=2,
            embedding_dim=8,
            encoder_chunk_batch_size=16,
            encoder_dropout=0.0,
            max_temporal_strength=0.25,
        )

    def test_uniform_initialization_preserves_cached_baseline_logits(self) -> None:
        """Epoch zero must reproduce mean pooling before any optimization."""
        baseline = BaselineEEGNet(self.baseline_config).eval()
        recency = EEGNetRecencyWeightedRiskModel(self.recency_config).eval()
        recency.initialize_from_baseline_state_dict(baseline.state_dict())
        embeddings = torch.randn(5, 8, 8)
        availability = torch.tensor(
            [
                [1.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
                [1.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
            ]
        )
        with torch.inference_mode():
            pooled = embeddings.mean(dim=1)
            baseline_logits = baseline.classifier(
                torch.cat([pooled, availability], dim=1)
            ).squeeze(1)
            recency_logits = recency.forward_from_chunk_embeddings(
                embeddings,
                availability,
            )
        torch.testing.assert_close(
            recency_logits,
            baseline_logits,
            rtol=1e-5,
            atol=1e-5,
        )

    def test_effective_weights_are_normalized_and_bounded(self) -> None:
        """The learned branch cannot assign negative or unconstrained weights."""
        model = EEGNetRecencyWeightedRiskModel(self.recency_config)
        initial_weights = model.effective_recency_weights()
        torch.testing.assert_close(
            initial_weights,
            torch.full((4,), 0.25),
        )
        self.assertAlmostEqual(
            float(model.uniform_weight_kl().detach()),
            0.0,
            places=6,
        )

        with torch.no_grad():
            model.recency_logits.copy_(torch.tensor([-20.0, -20.0, -20.0, 20.0]))
        weights = model.effective_recency_weights().detach()
        uniform_floor = (1.0 - self.recency_config.max_temporal_strength) / 4.0
        maximum = uniform_floor + self.recency_config.max_temporal_strength
        self.assertGreaterEqual(float(weights.min()), uniform_floor - 1e-7)
        self.assertLessEqual(float(weights.max()), maximum + 1e-7)
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=6)

    def test_only_one_logit_per_minute_remains_trainable(self) -> None:
        """Freezing the baseline must leave exactly four test parameters."""
        baseline = BaselineEEGNet(self.baseline_config)
        model = EEGNetRecencyWeightedRiskModel(self.recency_config)
        model.initialize_from_baseline_state_dict(baseline.state_dict())
        model.set_baseline_trainable(False)
        trainable = {
            name: parameter
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        self.assertEqual(set(trainable), {"recency_logits"})
        self.assertEqual(trainable["recency_logits"].numel(), 4)

    def test_recency_logits_learn_order_when_global_means_match(self) -> None:
        """The constrained pooler must detect a recent-vs-old reversal."""
        config = EEGNetRecencyWeightedConfig(
            n_chans=3,
            chunk_samples=1280,
            sequence_chunks=8,
            chunks_per_minute=2,
            embedding_dim=4,
            encoder_chunk_batch_size=16,
            encoder_dropout=0.0,
            max_temporal_strength=1.0,
        )
        model = EEGNetRecencyWeightedRiskModel(config)
        model.set_baseline_trainable(False)
        with torch.no_grad():
            classifier = model.baseline_classifier[1]
            classifier.weight.zero_()
            classifier.weight[0, 0] = 2.0
            classifier.bias.zero_()

        positive_minutes = torch.tensor([-1.0, -1.0, 1.0, 1.0])
        positive_chunks = positive_minutes.repeat_interleave(2)
        positive = torch.zeros(8, 4)
        positive[:, 0] = positive_chunks
        negative = positive.flip(dims=(0,))
        embeddings = torch.stack([positive, negative] * 8)
        availability = torch.zeros(16, 3)
        targets = torch.tensor([1.0, 0.0] * 8)
        optimizer = torch.optim.Adam([model.recency_logits], lr=0.1)
        criterion = nn.BCEWithLogitsLoss()

        with torch.no_grad():
            initial_loss = criterion(
                model.forward_from_chunk_embeddings(embeddings, availability),
                targets,
            )
        for _ in range(60):
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(
                model.forward_from_chunk_embeddings(embeddings, availability),
                targets,
            )
            loss.backward()
            optimizer.step()

        final_loss = float(loss.detach())
        weights = model.effective_recency_weights().detach()
        self.assertLess(final_loss, float(initial_loss) / 2.0)
        self.assertGreater(float(weights[2:].sum()), float(weights[:2].sum()))


if __name__ == "__main__":
    unittest.main()
