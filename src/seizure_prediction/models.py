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


@dataclass(frozen=True)
class SpectralAttentionConfig:
    """Configuration for the spectral-feature attention risk model."""

    n_features: int = 27
    n_chunks: int = 540
    embedding_dim: int = 32
    hidden_dim: int = 64
    attention_dim: int = 64
    temporal_kernel: int = 5
    dropout: float = 0.3
    n_chans: int = 3


class SpectralAttentionRiskModel(nn.Module):
    """Encode per-chunk spectral features, then pool them by learned attention.

    Replaces the from-scratch convolutional encoder with the representation the
    seizure literature already uses -- band power, Hjorth descriptors, line
    length -- computed once offline. The motivation is sample size: EEGNet had
    to learn a filter bank from 2,484 training positives and reached chance on
    held-out patients, while these features cost no parameters at all.

    Three deliberate pieces:

    ``encoder``
        A small per-chunk MLP. Shared across all 540 chunks.
    ``temporal``
        A depthwise convolution along the chunk axis, so the model can see local
        *dynamics* (a trend across ~25 s) rather than only per-chunk snapshots.
        Attention over independent embeddings cannot express that on its own.
    ``attention``
        Learned pooling with a fixed sinusoidal position encoding. Mean pooling
        weights each chunk 1/540 and is permutation-invariant; both properties
        were measured to be fatal for this task.
    """

    def __init__(self, config: SpectralAttentionConfig) -> None:
        super().__init__()
        if config.temporal_kernel % 2 != 1:
            raise ValueError("temporal_kernel must be odd so padding stays centred.")

        self.config = config

        self.encoder = nn.Sequential(
            nn.Linear(config.n_features, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.embedding_dim),
        )
        self.temporal = nn.Conv1d(
            config.embedding_dim,
            config.embedding_dim,
            kernel_size=config.temporal_kernel,
            padding=config.temporal_kernel // 2,
            groups=config.embedding_dim,
        )
        self.temporal_norm = nn.LayerNorm(config.embedding_dim)

        self.register_buffer(
            "position_table",
            sinusoidal_position_encoding(config.n_chunks, config.embedding_dim),
            persistent=False,
        )
        self.attention = nn.Sequential(
            nn.Linear(config.embedding_dim, config.attention_dim),
            nn.Tanh(),
            nn.Linear(config.attention_dim, 1),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(config.embedding_dim + config.n_chans),
            nn.Dropout(config.dropout),
            nn.Linear(config.embedding_dim + config.n_chans, 1),
        )

    def forward(
        self,
        features: torch.Tensor,
        availability_columns: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Return one seizure-risk logit per 45-minute context."""
        if features.ndim != 3:
            raise ValueError(
                "features must have shape (batch, chunks, n_features); "
                f"found {tuple(features.shape)}."
            )
        batch_size, number_of_chunks, n_features = features.shape
        if n_features != self.config.n_features:
            raise ValueError(
                f"expected {self.config.n_features} features, found {n_features}."
            )

        embeddings = self.encoder(features)

        residual = embeddings
        embeddings = self.temporal(embeddings.transpose(1, 2)).transpose(1, 2)
        embeddings = self.temporal_norm(embeddings + residual)

        if number_of_chunks > self.position_table.shape[0]:
            raise ValueError(
                f"model configured for {self.position_table.shape[0]} chunks, "
                f"received {number_of_chunks}."
            )
        embeddings = embeddings + self.position_table[:number_of_chunks].unsqueeze(0)

        attention_weights = torch.softmax(
            self.attention(embeddings).squeeze(-1), dim=1
        )
        pooled = torch.einsum("bm,bmd->bd", attention_weights, embeddings)

        # Collapse the per-column availability back to one flag per electrode.
        per_channel = availability_columns.shape[1] // self.config.n_chans
        channel_availability = availability_columns.reshape(
            batch_size, self.config.n_chans, per_channel
        )[:, :, 0]

        logits = self.classifier(
            torch.cat([pooled, channel_availability.to(pooled.dtype)], dim=1)
        ).squeeze(1)
        if return_attention:
            return logits, attention_weights
        return logits

    def predict_proba(
        self,
        features: torch.Tensor,
        availability_columns: torch.Tensor,
    ) -> torch.Tensor:
        """Return the model's uncalibrated next-10-minute seizure probability."""
        return torch.sigmoid(self(features, availability_columns))


class SpectralGRURiskModel(nn.Module):
    """Same per-chunk features, aggregated by a GRU instead of attention.

    A recurrent aggregator is the other obvious way to weight recent chunks
    above distant ones. Included so the temporal-aggregation axis is explored
    rather than assumed: attention and recurrence fail differently, and if both
    match a linear control then the aggregation strategy is not the limitation.
    """

    def __init__(self, config: SpectralAttentionConfig, bidirectional: bool = False) -> None:
        super().__init__()
        self.config = config
        self.encoder = nn.Sequential(
            nn.Linear(config.n_features, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.embedding_dim),
        )
        self.gru = nn.GRU(
            config.embedding_dim,
            config.embedding_dim,
            batch_first=True,
            bidirectional=bidirectional,
        )
        directions = 2 if bidirectional else 1
        self.classifier = nn.Sequential(
            nn.LayerNorm(config.embedding_dim * directions + config.n_chans),
            nn.Dropout(config.dropout),
            nn.Linear(config.embedding_dim * directions + config.n_chans, 1),
        )

    def forward(self, features: torch.Tensor, availability_columns: torch.Tensor) -> torch.Tensor:
        batch_size = features.shape[0]
        embeddings = self.encoder(features)
        output, _ = self.gru(embeddings)
        pooled = output[:, -1, :]  # state at the decision instant

        per_channel = availability_columns.shape[1] // self.config.n_chans
        flags = availability_columns.reshape(
            batch_size, self.config.n_chans, per_channel
        )[:, :, 0]
        return self.classifier(
            torch.cat([pooled, flags.to(pooled.dtype)], dim=1)
        ).squeeze(1)

    def predict_proba(self, features: torch.Tensor, availability_columns: torch.Tensor):
        return torch.sigmoid(self(features, availability_columns))


class SpectralMeanPoolRiskModel(nn.Module):
    """Per-chunk MLP encoder with plain mean pooling — the aggregation ablation."""

    def __init__(self, config: SpectralAttentionConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = nn.Sequential(
            nn.Linear(config.n_features, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.embedding_dim),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(config.embedding_dim + config.n_chans),
            nn.Dropout(config.dropout),
            nn.Linear(config.embedding_dim + config.n_chans, 1),
        )

    def forward(self, features: torch.Tensor, availability_columns: torch.Tensor) -> torch.Tensor:
        batch_size = features.shape[0]
        pooled = self.encoder(features).mean(dim=1)
        per_channel = availability_columns.shape[1] // self.config.n_chans
        flags = availability_columns.reshape(
            batch_size, self.config.n_chans, per_channel
        )[:, :, 0]
        return self.classifier(
            torch.cat([pooled, flags.to(pooled.dtype)], dim=1)
        ).squeeze(1)

    def predict_proba(self, features: torch.Tensor, availability_columns: torch.Tensor):
        return torch.sigmoid(self(features, availability_columns))


class SpectralContrastRiskModel(nn.Module):
    """Recency contrast pooling: recent chunks, and how they differ from the rest.

    Mean pooling over all 540 chunks dilutes any temporally localised marker by
    roughly the ratio of window to marker. Measured on a positive control where
    the signal is known to be present -- 45-minute windows containing 5 minutes
    of frank ictal EEG -- pooling operators separate ictal from interictal at:

        mean over 540 chunks              AUC 0.561
        max over 540                      AUC 0.536
        mean of the last 60 chunks        AUC 0.799
        [last-60 mean, last-60 - rest]    AUC 0.817

    So the aggregation, not the feature set, was the capacity bottleneck for
    anything time-localised. This module implements the last operator: the
    recent-window mean concatenated with its contrast against the earlier part
    of the same window, which is both recency-weighted and self-referencing.

    Note this fix recovers the ICTAL signal and, on the same machinery, recovers
    nothing pre-ictal (AUC 0.46-0.52). It is included because a known defect
    should be fixed regardless of whether it rescues the headline result.
    """

    def __init__(
        self,
        config: SpectralAttentionConfig,
        recent_chunks: int = 60,
    ) -> None:
        super().__init__()
        if recent_chunks < 1:
            raise ValueError("recent_chunks must be positive.")
        self.config = config
        self.recent_chunks = recent_chunks
        self.encoder = nn.Sequential(
            nn.Linear(config.n_features * 2, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.embedding_dim),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(config.embedding_dim + config.n_chans),
            nn.Dropout(config.dropout),
            nn.Linear(config.embedding_dim + config.n_chans, 1),
        )

    def forward(self, features: torch.Tensor, availability_columns: torch.Tensor) -> torch.Tensor:
        batch_size, n_chunks, _ = features.shape
        k = min(self.recent_chunks, n_chunks)
        recent = features[:, -k:, :].mean(dim=1)
        if n_chunks > k:
            rest = features[:, :-k, :].mean(dim=1)
        else:
            rest = torch.zeros_like(recent)
        pooled = self.encoder(torch.cat([recent, recent - rest], dim=1))

        per_channel = availability_columns.shape[1] // self.config.n_chans
        flags = availability_columns.reshape(
            batch_size, self.config.n_chans, per_channel
        )[:, :, 0]
        return self.classifier(
            torch.cat([pooled, flags.to(pooled.dtype)], dim=1)
        ).squeeze(1)

    def predict_proba(self, features: torch.Tensor, availability_columns: torch.Tensor):
        return torch.sigmoid(self(features, availability_columns))
