"""Baseline neural-network models for streaming seizure-risk prediction."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from braindecode.models import EEGNet
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class EEGNetMeanPoolConfig:
    """Configuration for the EEGNet mean-pooling baseline."""

    n_chans: int = 3
    chunk_samples: int = 5 * 256
    embedding_dim: int = 32
    encoder_chunk_batch_size: int = 64
    dropout: float = 0.25
    sampling_prior_logit_correction: float = 0.0


class EEGNetMeanPoolRiskModel(nn.Module):
    """Encode five-second EEG chunks with EEGNet and mean-pool over the input window.

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
        """Encode chunks in micro-batches to keep long contexts tractable."""
        embeddings = []
        for chunk_batch in chunks.split(self.config.encoder_chunk_batch_size):
            embeddings.append(self.encoder(chunk_batch))
        return torch.cat(embeddings, dim=0)

    def forward(
        self,
        signal: torch.Tensor,
        channel_availability: torch.Tensor,
    ) -> torch.Tensor:
        """Return one uncalibrated seizure-risk logit per decision context."""
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
        """Return a probability corrected from the sampled to natural prior."""
        logits = self(signal, channel_availability)
        return torch.sigmoid(
            logits + self.config.sampling_prior_logit_correction
        )


@dataclass(frozen=True)
class EEGNetTemporalTCNConfig:
    """Configuration for EEGNet followed by a causal temporal TCN."""

    n_chans: int = 3
    chunk_samples: int = 5 * 256
    # The default matches the widest swept input window; the training script
    # derives this from the label definition it was asked for.
    sequence_chunks: int = 30 * 60 // 5
    embedding_dim: int = 32
    tcn_channels: int = 16
    tcn_kernel_size: int = 3
    tcn_dilations: tuple[int, ...] = (1, 2, 4, 8, 15, 16, 32, 64, 128)
    encoder_chunk_batch_size: int = 64
    dropout: float = 0.25
    temporal_norm_groups: int = 4
    sampling_prior_logit_correction: float = 0.0

    @property
    def temporal_receptive_field_chunks(self) -> int:
        """Return the causal TCN receptive field measured in EEG chunks."""
        return 1 + (self.tcn_kernel_size - 1) * sum(self.tcn_dilations)


class CausalDepthwiseTCNBlock(nn.Module):
    """Microcontroller-oriented causal depthwise-separable residual block."""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
        norm_groups: int,
    ) -> None:
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=channels,
            bias=False,
        )
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.normalization = nn.GroupNorm(norm_groups, channels)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return a length-preserving causal residual transformation."""
        outputs = F.pad(inputs, (self.left_padding, 0))
        outputs = self.depthwise(outputs)
        outputs = self.pointwise(outputs)
        outputs = self.normalization(outputs)
        outputs = self.activation(outputs)
        outputs = self.dropout(outputs)
        return self.activation(inputs + outputs)


class EEGNetTemporalTCNRiskModel(nn.Module):
    """Combine local EEGNet features with ordered, configured-window TCN context.

    EEGNet independently encodes each five-second chunk. The embeddings retain
    their chronological order and are projected into a causal, dilated,
    depthwise-separable temporal convolutional network. The final temporal
    position summarizes the complete history and is combined with electrode
    availability before binary risk classification.
    """

    def __init__(self, config: EEGNetTemporalTCNConfig) -> None:
        super().__init__()
        if config.n_chans <= 0:
            raise ValueError("n_chans must be positive.")
        if config.chunk_samples <= 0 or config.sequence_chunks <= 0:
            raise ValueError("chunk_samples and sequence_chunks must be positive.")
        if config.embedding_dim <= 0 or config.tcn_channels <= 0:
            raise ValueError("embedding_dim and tcn_channels must be positive.")
        if config.tcn_kernel_size <= 1:
            raise ValueError("tcn_kernel_size must be greater than one.")
        if not config.tcn_dilations or any(
            dilation <= 0 for dilation in config.tcn_dilations
        ):
            raise ValueError("tcn_dilations must contain positive values.")
        if config.temporal_receptive_field_chunks < config.sequence_chunks:
            raise ValueError(
                "The temporal receptive field must cover the full EEG history."
            )
        if config.encoder_chunk_batch_size <= 0:
            raise ValueError("encoder_chunk_batch_size must be positive.")
        if not 0.0 <= config.dropout < 1.0:
            raise ValueError("dropout must be at least zero and less than one.")
        if config.temporal_norm_groups <= 0:
            raise ValueError("temporal_norm_groups must be positive.")
        if config.tcn_channels % config.temporal_norm_groups != 0:
            raise ValueError(
                "tcn_channels must be divisible by temporal_norm_groups."
            )

        self.config = config
        self.encoder = EEGNet(
            n_chans=config.n_chans,
            n_outputs=config.embedding_dim,
            n_times=config.chunk_samples,
            drop_prob=config.dropout,
            final_layer_with_constraint=True,
        )
        self.temporal_input_projection = nn.Sequential(
            nn.Conv1d(
                config.embedding_dim,
                config.tcn_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(config.temporal_norm_groups, config.tcn_channels),
            nn.ReLU(),
        )
        self.temporal_blocks = nn.Sequential(
            *[
                CausalDepthwiseTCNBlock(
                    channels=config.tcn_channels,
                    kernel_size=config.tcn_kernel_size,
                    dilation=dilation,
                    dropout=config.dropout,
                    norm_groups=config.temporal_norm_groups,
                )
                for dilation in config.tcn_dilations
            ]
        )
        self.classifier = nn.Linear(config.tcn_channels + config.n_chans, 1)

    def _encode_chunks(self, chunks: torch.Tensor) -> torch.Tensor:
        """Encode chunks in micro-batches to control peak activation memory."""
        embeddings = []
        for chunk_batch in chunks.split(self.config.encoder_chunk_batch_size):
            embeddings.append(self.encoder(chunk_batch))
        return torch.cat(embeddings, dim=0)

    def forward(
        self,
        signal: torch.Tensor,
        channel_availability: torch.Tensor,
    ) -> torch.Tensor:
        """Return one uncalibrated seizure-risk logit per decision context."""
        if signal.ndim != 4:
            raise ValueError(
                "signal must have shape (batch, chunks, channels, samples); "
                f"found {tuple(signal.shape)}."
            )

        batch_size, number_of_chunks, channels, samples = signal.shape
        expected_signal_shape = (
            self.config.sequence_chunks,
            self.config.n_chans,
            self.config.chunk_samples,
        )
        if (number_of_chunks, channels, samples) != expected_signal_shape:
            raise ValueError(
                "Unexpected EEG shape after the batch dimension: expected "
                f"{expected_signal_shape}, found "
                f"{(number_of_chunks, channels, samples)}."
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
        chunk_embeddings = self._encode_chunks(flattened_chunks).reshape(
            batch_size,
            number_of_chunks,
            self.config.embedding_dim,
        )
        temporal_features = self.temporal_input_projection(
            chunk_embeddings.transpose(1, 2)
        )
        temporal_features = self.temporal_blocks(temporal_features)
        history_embedding = temporal_features[:, :, -1]

        features = torch.cat(
            [
                history_embedding,
                channel_availability.to(
                    dtype=history_embedding.dtype,
                    device=history_embedding.device,
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
        """Return a probability corrected from the sampled to natural prior."""
        logits = self(signal, channel_availability)
        return torch.sigmoid(
            logits + self.config.sampling_prior_logit_correction
        )
