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


@dataclass(frozen=True)
class EEGNetAttentionConfig:
    """Configuration for the attention-pooling seizure-risk model.

    Differs from :class:`EEGNetMeanPoolConfig` in two ways that address the
    measured failure of the mean-pool baseline (test ROC AUC 0.5026):

    ``attention_dim``
        Enables learned, order-aware pooling over the 540 chunk embeddings.
        Mean pooling weights every chunk by 1/540, so a 30-second pre-ictal
        signature contributes ~1.1% of the pooled vector and receives 1/540 of
        the gradient. It is also permutation-invariant: the same chunks in any
        order produce an identical logit.

    ``global_pool_head``
        Replaces EEGNet's ``flatten(16,1,40) -> Linear(640, 32)`` final layer
        with average pooling over time followed by ``Linear(16, 32)``. That one
        layer held 20,480 of the baseline's 21,770 parameters (94%) and was the
        only place absolute position within a chunk was encoded -- the kind of
        nuisance detail that memorises patients rather than physiology.
    """

    n_chans: int = 3
    chunk_samples: int = 5 * 256
    embedding_dim: int = 32
    encoder_chunk_batch_size: int = 1024
    dropout: float = 0.25
    attention_dim: int = 64
    global_pool_head: bool = True
    n_chunks: int = 540
    positional_encoding: bool = True


def sinusoidal_position_encoding(n_positions: int, dimension: int) -> torch.Tensor:
    """Return the standard fixed sinusoidal position table, ``(n, d)``.

    Mean pooling discards chunk order entirely. Attention alone would too --
    the scores depend only on each embedding's content -- so the encoder needs
    an explicit signal for *when* in the 45 minutes a chunk sits.
    """
    if dimension % 2 != 0:
        raise ValueError("Position-encoding dimension must be even.")

    position = torch.arange(n_positions, dtype=torch.float32).unsqueeze(1)
    frequency = torch.exp(
        torch.arange(0, dimension, 2, dtype=torch.float32)
        * (-torch.log(torch.tensor(10000.0)) / dimension)
    )
    table = torch.zeros(n_positions, dimension, dtype=torch.float32)
    table[:, 0::2] = torch.sin(position * frequency)
    table[:, 1::2] = torch.cos(position * frequency)
    return table


class EEGNetAttentionRiskModel(nn.Module):
    """Encode 5-second chunks with EEGNet, then pool them by learned attention.

    Shapes, for one decision::

        (chunks=540, channels=3, samples=1280)
            -> EEGNet, shared weights            -> (540, embedding_dim)
            -> + sinusoidal position encoding    -> (540, embedding_dim)
            -> attention weights, softmax over m -> (540,)
            -> weighted sum                      -> (embedding_dim,)
            -> concat 3-value availability mask  -> (embedding_dim + 3,)
            -> LayerNorm + Linear                -> one logit
    """

    def __init__(self, config: EEGNetAttentionConfig) -> None:
        super().__init__()
        if config.n_chans <= 0 or config.chunk_samples <= 0:
            raise ValueError("n_chans and chunk_samples must be positive.")
        if config.embedding_dim <= 0 or config.attention_dim <= 0:
            raise ValueError("embedding_dim and attention_dim must be positive.")
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

        if config.global_pool_head:
            # conv_separable_point fixes the feature width entering the head.
            feature_channels = self.encoder.conv_separable_point.out_channels
            self.encoder.final_layer = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(feature_channels, config.embedding_dim),
            )

        if config.positional_encoding:
            self.register_buffer(
                "position_table",
                sinusoidal_position_encoding(config.n_chunks, config.embedding_dim),
                persistent=False,
            )
        else:
            self.position_table = None

        self.attention = nn.Sequential(
            nn.Linear(config.embedding_dim, config.attention_dim),
            nn.Tanh(),
            nn.Linear(config.attention_dim, 1),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(config.embedding_dim + config.n_chans),
            nn.Linear(config.embedding_dim + config.n_chans, 1),
        )

    def _encode_chunks(self, chunks: torch.Tensor) -> torch.Tensor:
        """Encode chunks in micro-batches to bound peak activation memory."""
        embeddings = []
        for chunk_batch in chunks.split(self.config.encoder_chunk_batch_size):
            embeddings.append(self.encoder(chunk_batch))
        return torch.cat(embeddings, dim=0)

    def forward(
        self,
        signal: torch.Tensor,
        channel_availability: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Return one seizure-risk logit per 45-minute context."""
        if signal.ndim != 4:
            raise ValueError(
                "signal must have shape (batch, chunks, channels, samples); "
                f"found {tuple(signal.shape)}."
            )
        batch_size, number_of_chunks, channels, samples = signal.shape
        if channels != self.config.n_chans or samples != self.config.chunk_samples:
            raise ValueError(
                "Unexpected EEG chunk shape: expected "
                f"{self.config.n_chans} channels and {self.config.chunk_samples} "
                f"samples, found {channels} and {samples}."
            )

        if channel_availability.ndim == 1:
            channel_availability = channel_availability.unsqueeze(0)
        if channel_availability.shape != (batch_size, self.config.n_chans):
            raise ValueError(
                "channel_availability must have shape (batch, channels); "
                f"found {tuple(channel_availability.shape)}."
            )

        chunk_embeddings = self._encode_chunks(
            signal.reshape(batch_size * number_of_chunks, channels, samples)
        ).reshape(batch_size, number_of_chunks, self.config.embedding_dim)

        if self.position_table is not None:
            if number_of_chunks > self.position_table.shape[0]:
                raise ValueError(
                    f"Model was configured for {self.position_table.shape[0]} "
                    f"chunks but received {number_of_chunks}."
                )
            chunk_embeddings = chunk_embeddings + self.position_table[
                :number_of_chunks
            ].unsqueeze(0)

        attention_logits = self.attention(chunk_embeddings).squeeze(-1)
        attention_weights = torch.softmax(attention_logits, dim=1)
        pooled_embedding = torch.einsum(
            "bm,bmd->bd", attention_weights, chunk_embeddings
        )

        features = torch.cat(
            [
                pooled_embedding,
                channel_availability.to(
                    dtype=pooled_embedding.dtype, device=pooled_embedding.device
                ),
            ],
            dim=1,
        )
        logits = self.classifier(features).squeeze(1)
        if return_attention:
            return logits, attention_weights
        return logits

    def predict_proba(
        self,
        signal: torch.Tensor,
        channel_availability: torch.Tensor,
    ) -> torch.Tensor:
        """Return the model's uncalibrated next-10-minute seizure probability."""
        return torch.sigmoid(self(signal, channel_availability))
