"""The active seizure-prediction model: a minimal EEGNet baseline."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from braindecode.models import EEGNet
from torch import nn


@dataclass(frozen=True)
class BaselineEEGNetConfig:
    """Dimensions and regularization for the 45-minute EEGNet baseline."""

    n_chans: int = 3
    chunk_samples: int = 5 * 256
    sequence_chunks: int = 45 * 60 // 5
    embedding_dim: int = 32
    encoder_chunk_batch_size: int = 128
    dropout: float = 0.4
    sampling_prior_logit_correction: float = 0.0


class BaselineEEGNet(nn.Module):
    """Predict next-10-minute seizure risk from 45 minutes of EEG.

    EEGNet independently converts each five-second chunk into a compact feature
    vector. Mean pooling gives every chunk equal weight and intentionally
    discards chunk order, making this a simple reference model rather than a
    temporal sequence model. The pooled EEG representation and the three-value
    electrode-availability mask feed one binary classifier.
    """

    def __init__(self, config: BaselineEEGNetConfig) -> None:
        super().__init__()
        if config.n_chans <= 0:
            raise ValueError("n_chans must be positive.")
        if config.chunk_samples <= 0 or config.sequence_chunks <= 0:
            raise ValueError("chunk_samples and sequence_chunks must be positive.")
        if config.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive.")
        if config.encoder_chunk_batch_size <= 0:
            raise ValueError("encoder_chunk_batch_size must be positive.")
        if not 0.0 <= config.dropout < 1.0:
            raise ValueError("dropout must be at least zero and less than one.")

        self.config = config
        self.encoder = EEGNet(
            n_chans=config.n_chans,
            n_outputs=config.embedding_dim,
            n_times=config.chunk_samples,
            drop_prob=config.dropout,
            final_layer_with_constraint=True,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(config.embedding_dim + config.n_chans),
            nn.Linear(config.embedding_dim + config.n_chans, 1),
        )

    def _encode_chunks(self, chunks: torch.Tensor) -> torch.Tensor:
        """Encode chunks in small groups to limit peak accelerator memory."""
        embeddings = [
            self.encoder(chunk_batch)
            for chunk_batch in chunks.split(self.config.encoder_chunk_batch_size)
        ]
        return torch.cat(embeddings, dim=0)

    def forward(
        self,
        signal: torch.Tensor,
        channel_availability: torch.Tensor,
    ) -> torch.Tensor:
        """Return one raw seizure-risk logit for each decision context."""
        expected_signal_shape = (
            self.config.sequence_chunks,
            self.config.n_chans,
            self.config.chunk_samples,
        )
        if signal.ndim != 4 or tuple(signal.shape[1:]) != expected_signal_shape:
            raise ValueError(
                "signal must have shape (batch, chunks, channels, samples) with "
                f"trailing dimensions {expected_signal_shape}; found "
                f"{tuple(signal.shape)}."
            )

        batch_size = signal.shape[0]
        if channel_availability.ndim == 1:
            channel_availability = channel_availability.unsqueeze(0)
        if channel_availability.shape != (batch_size, self.config.n_chans):
            raise ValueError(
                "channel_availability must have shape (batch, channels); "
                f"found {tuple(channel_availability.shape)}."
            )

        flattened_chunks = signal.reshape(
            batch_size * self.config.sequence_chunks,
            self.config.n_chans,
            self.config.chunk_samples,
        )
        chunk_embeddings = self._encode_chunks(flattened_chunks).reshape(
            batch_size,
            self.config.sequence_chunks,
            self.config.embedding_dim,
        )
        pooled_embedding = chunk_embeddings.mean(dim=1)
        features = torch.cat(
            [
                pooled_embedding,
                channel_availability.to(
                    dtype=pooled_embedding.dtype,
                    device=pooled_embedding.device,
                ),
            ],
            dim=1,
        )
        return self.classifier(features).squeeze(1)

    def predict_proba(
        self,
        signal: torch.Tensor,
        channel_availability: torch.Tensor,
    ) -> torch.Tensor:
        """Return probabilities corrected for negative subsampling."""
        logits = self(signal, channel_availability)
        corrected_logits = (
            logits + self.config.sampling_prior_logit_correction
        )
        return torch.sigmoid(corrected_logits)
