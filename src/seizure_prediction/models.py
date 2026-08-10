"""Baseline neural-network models for streaming seizure-risk prediction."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from braindecode.models import EEGNet
from torch import nn


@dataclass(frozen=True)
class EEGNetMeanPoolConfig:
    """Configuration for the 45-minute EEGNet mean-pooling baseline."""

    n_chans: int = 3
    chunk_samples: int = 5 * 256
    embedding_dim: int = 32
    encoder_chunk_batch_size: int = 64
    dropout: float = 0.25


class EEGNetMeanPoolRiskModel(nn.Module):
    """Encode five-second EEG chunks with EEGNet and mean-pool over 45 minutes.

    The input contains a full streaming decision context with shape
    ``(batch, chunks, channels, samples)``. Each chunk is encoded by the
    Braindecode implementation of EEGNet, then all chunk embeddings receive
    equal weight through mean pooling. A three-value electrode-availability
    mask is concatenated to the pooled EEG representation before the final
    binary risk classifier.

    This is intentionally a simple baseline. It establishes a fair starting
    point before replacing mean pooling with a recurrent or attention-based
    temporal aggregation layer.
    """

    def __init__(self, config: EEGNetMeanPoolConfig) -> None:
        super().__init__()
        if config.n_chans <= 0:
            raise ValueError("n_chans must be positive.")
        if config.chunk_samples <= 0:
            raise ValueError("chunk_samples must be positive.")
        if config.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive.")
        if config.encoder_chunk_batch_size <= 0:
            raise ValueError("encoder_chunk_batch_size must be positive.")

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
        """Encode chunks in micro-batches to keep 45-minute contexts tractable."""
        embeddings = []
        for chunk_batch in chunks.split(self.config.encoder_chunk_batch_size):
            embeddings.append(self.encoder(chunk_batch))
        return torch.cat(embeddings, dim=0)

    def forward(
        self,
        signal: torch.Tensor,
        channel_availability: torch.Tensor,
    ) -> torch.Tensor:
        """Return one uncalibrated seizure-risk logit per 45-minute context."""
        if signal.ndim != 4:
            raise ValueError(
                "signal must have shape (batch, chunks, channels, samples); "
                f"found {tuple(signal.shape)}."
            )

        batch_size, number_of_chunks, channels, samples = signal.shape
        if channels != self.config.n_chans or samples != self.config.chunk_samples:
            raise ValueError(
                "Unexpected EEG chunk shape: expected "
                f"{self.config.n_chans} channels and "
                f"{self.config.chunk_samples} samples, found "
                f"{channels} channels and {samples} samples."
            )

        if channel_availability.ndim == 1:
            channel_availability = channel_availability.unsqueeze(0)
        if channel_availability.shape != (batch_size, self.config.n_chans):
            raise ValueError(
                "channel_availability must have shape (batch, channels); "
                f"found {tuple(channel_availability.shape)}."
            )

        flattened_chunks = signal.reshape(
            batch_size * number_of_chunks,
            channels,
            samples,
        )
        chunk_embeddings = self._encode_chunks(flattened_chunks)
        chunk_embeddings = chunk_embeddings.reshape(
            batch_size,
            number_of_chunks,
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
        """Return the model's uncalibrated next-10-minute seizure probability."""
        return torch.sigmoid(self(signal, channel_availability))
