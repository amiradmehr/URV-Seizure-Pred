"""Active EEGNet models for streaming seizure-risk prediction."""

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
        return self.forward_from_chunk_embeddings(
            chunk_embeddings,
            channel_availability,
        )

    def forward_from_chunk_embeddings(
        self,
        chunk_embeddings: torch.Tensor,
        channel_availability: torch.Tensor,
    ) -> torch.Tensor:
        """Return baseline logits from cached chronological embeddings."""
        expected_embedding_shape = (
            self.config.sequence_chunks,
            self.config.embedding_dim,
        )
        if (
            chunk_embeddings.ndim != 3
            or tuple(chunk_embeddings.shape[1:]) != expected_embedding_shape
        ):
            raise ValueError(
                "chunk_embeddings must have shape (batch, chunks, features) "
                f"with trailing dimensions {expected_embedding_shape}; found "
                f"{tuple(chunk_embeddings.shape)}."
            )

        batch_size = chunk_embeddings.shape[0]
        if channel_availability.ndim == 1:
            channel_availability = channel_availability.unsqueeze(0)
        if channel_availability.shape != (batch_size, self.config.n_chans):
            raise ValueError(
                "channel_availability must have shape (batch, channels); "
                f"found {tuple(channel_availability.shape)}."
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


@dataclass(frozen=True)
class EEGNetMultiScaleTemporalConfig:
    """Configuration for the residual multi-scale temporal EEGNet.

    Five-second EEGNet embeddings are first averaged into minute-level
    embeddings.  Causal contrasts between nested recent-history windows add
    temporal information while the original 45-minute mean-pooling path is
    retained as an exact baseline bypass.
    """

    n_chans: int = 3
    chunk_samples: int = 5 * 256
    sequence_chunks: int = 45 * 60 // 5
    chunks_per_minute: int = 60 // 5
    embedding_dim: int = 32
    temporal_hidden_dim: int = 16
    temporal_windows_minutes: tuple[int, ...] = (1, 5, 15, 45)
    encoder_chunk_batch_size: int = 128
    encoder_dropout: float = 0.4
    temporal_dropout: float = 0.2
    sampling_prior_logit_correction: float = 0.0

    @property
    def history_minutes(self) -> int:
        """Return the complete input history in whole minutes."""
        return self.sequence_chunks // self.chunks_per_minute

    @property
    def temporal_feature_dim(self) -> int:
        """Return the width of the adjacent multi-scale contrast vector."""
        return (len(self.temporal_windows_minutes) - 1) * self.embedding_dim


class EEGNetMultiScaleTemporalRiskModel(nn.Module):
    """Add small causal temporal contrasts to the robust EEGNet baseline.

    The baseline branch is mathematically the same as :class:`BaselineEEGNet`:
    all five-second embeddings receive equal weight over 45 minutes.  A second
    branch compares nested 1-, 5-, 15-, and 45-minute means.  Its final layer
    is initialized to zero, so a model initialized from a baseline checkpoint
    produces the exact baseline logit before temporal-head training begins.

    The temporal branch is deliberately based on rolling means rather than a
    long recurrent state or a 540-step TCN.  At deployment it can be evaluated
    with causal ring-buffer sums over minute embeddings.
    """

    def __init__(self, config: EEGNetMultiScaleTemporalConfig) -> None:
        super().__init__()
        self._validate_config(config)
        self.config = config

        self.encoder = EEGNet(
            n_chans=config.n_chans,
            n_outputs=config.embedding_dim,
            n_times=config.chunk_samples,
            drop_prob=config.encoder_dropout,
            final_layer_with_constraint=True,
        )
        self.baseline_classifier = nn.Sequential(
            nn.LayerNorm(config.embedding_dim + config.n_chans),
            nn.Linear(config.embedding_dim + config.n_chans, 1),
        )
        self.temporal_head = nn.Sequential(
            nn.LayerNorm(config.temporal_feature_dim),
            nn.Linear(config.temporal_feature_dim, config.temporal_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.temporal_dropout),
            nn.Linear(config.temporal_hidden_dim, 1),
        )

        # The residual branch starts as an exact no-op.  Loading a baseline
        # checkpoint therefore cannot reduce its initial validation score.
        final_temporal_layer = self.temporal_head[-1]
        nn.init.zeros_(final_temporal_layer.weight)
        nn.init.zeros_(final_temporal_layer.bias)

    @staticmethod
    def _validate_config(config: EEGNetMultiScaleTemporalConfig) -> None:
        """Reject shapes that cannot form the configured temporal pyramid."""
        positive_values = {
            "n_chans": config.n_chans,
            "chunk_samples": config.chunk_samples,
            "sequence_chunks": config.sequence_chunks,
            "chunks_per_minute": config.chunks_per_minute,
            "embedding_dim": config.embedding_dim,
            "temporal_hidden_dim": config.temporal_hidden_dim,
            "encoder_chunk_batch_size": config.encoder_chunk_batch_size,
        }
        for name, value in positive_values.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive.")
        if config.sequence_chunks % config.chunks_per_minute != 0:
            raise ValueError(
                "sequence_chunks must divide into whole minute embeddings."
            )
        windows = config.temporal_windows_minutes
        if len(windows) < 2 or any(window <= 0 for window in windows):
            raise ValueError(
                "temporal_windows_minutes must contain at least two positive windows."
            )
        if tuple(sorted(set(windows))) != windows:
            raise ValueError(
                "temporal_windows_minutes must be strictly increasing."
            )
        if windows[-1] != config.history_minutes:
            raise ValueError(
                "The final temporal window must equal the complete input history."
            )
        for name, dropout in (
            ("encoder_dropout", config.encoder_dropout),
            ("temporal_dropout", config.temporal_dropout),
        ):
            if not 0.0 <= dropout < 1.0:
                raise ValueError(f"{name} must be at least zero and less than one.")

    def _encode_chunks(self, chunks: torch.Tensor) -> torch.Tensor:
        """Encode chunks in bounded groups to control peak accelerator memory."""
        embeddings = [
            self.encoder(chunk_batch)
            for chunk_batch in chunks.split(self.config.encoder_chunk_batch_size)
        ]
        return torch.cat(embeddings, dim=0)

    def _minute_embeddings(self, chunk_embeddings: torch.Tensor) -> torch.Tensor:
        """Average consecutive chunk embeddings into causal one-minute tokens."""
        batch_size = chunk_embeddings.shape[0]
        return chunk_embeddings.reshape(
            batch_size,
            self.config.history_minutes,
            self.config.chunks_per_minute,
            self.config.embedding_dim,
        ).mean(dim=2)

    def _multi_scale_features(
        self,
        minute_embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the global mean and adjacent causal window contrasts."""
        window_means = [
            minute_embeddings[:, -window_minutes:, :].mean(dim=1)
            for window_minutes in self.config.temporal_windows_minutes
        ]
        temporal_contrasts = torch.cat(
            [
                recent_mean - broader_mean
                for recent_mean, broader_mean in zip(
                    window_means[:-1],
                    window_means[1:],
                )
            ],
            dim=1,
        )
        return window_means[-1], temporal_contrasts

    def initialize_from_baseline_state_dict(
        self,
        baseline_state_dict: dict[str, torch.Tensor],
    ) -> None:
        """Load the shared encoder and classifier from ``BaselineEEGNet``."""
        encoder_prefix = "encoder."
        classifier_prefix = "classifier."
        encoder_state = {
            name.removeprefix(encoder_prefix): value
            for name, value in baseline_state_dict.items()
            if name.startswith(encoder_prefix)
        }
        classifier_state = {
            name.removeprefix(classifier_prefix): value
            for name, value in baseline_state_dict.items()
            if name.startswith(classifier_prefix)
        }
        if not encoder_state or not classifier_state:
            raise ValueError(
                "The checkpoint is not a BaselineEEGNet state dictionary."
            )
        self.encoder.load_state_dict(encoder_state, strict=True)
        self.baseline_classifier.load_state_dict(classifier_state, strict=True)

    def set_baseline_trainable(self, trainable: bool) -> None:
        """Freeze or unfreeze the pretrained encoder and baseline classifier."""
        for module in (self.encoder, self.baseline_classifier):
            for parameter in module.parameters():
                parameter.requires_grad_(trainable)

    def keep_frozen_baseline_in_eval_mode(self) -> None:
        """Prevent dropout and BatchNorm updates during head-only training."""
        if not any(parameter.requires_grad for parameter in self.encoder.parameters()):
            self.encoder.eval()
            self.baseline_classifier.eval()

    def forward(
        self,
        signal: torch.Tensor,
        channel_availability: torch.Tensor,
    ) -> torch.Tensor:
        """Return one raw seizure-risk logit per decision context."""
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
        return self.forward_from_chunk_embeddings(
            chunk_embeddings,
            channel_availability,
        )

    def forward_from_chunk_embeddings(
        self,
        chunk_embeddings: torch.Tensor,
        channel_availability: torch.Tensor,
    ) -> torch.Tensor:
        """Return logits from cached EEGNet embeddings.

        This path is equivalent to :meth:`forward` after EEGNet encoding.  It
        makes repeated temporal-head experiments inexpensive while keeping the
        deployed end-to-end model unchanged.
        """
        expected_embedding_shape = (
            self.config.sequence_chunks,
            self.config.embedding_dim,
        )
        if (
            chunk_embeddings.ndim != 3
            or tuple(chunk_embeddings.shape[1:]) != expected_embedding_shape
        ):
            raise ValueError(
                "chunk_embeddings must have shape (batch, chunks, features) "
                f"with trailing dimensions {expected_embedding_shape}; found "
                f"{tuple(chunk_embeddings.shape)}."
            )

        batch_size = chunk_embeddings.shape[0]
        if channel_availability.ndim == 1:
            channel_availability = channel_availability.unsqueeze(0)
        if channel_availability.shape != (batch_size, self.config.n_chans):
            raise ValueError(
                "channel_availability must have shape (batch, channels); "
                f"found {tuple(channel_availability.shape)}."
            )

        minute_embeddings = self._minute_embeddings(chunk_embeddings)
        global_mean, temporal_features = self._multi_scale_features(
            minute_embeddings
        )

        availability_features = channel_availability.to(
            dtype=global_mean.dtype,
            device=global_mean.device,
        )
        baseline_features = torch.cat(
            [global_mean, availability_features],
            dim=1,
        )
        baseline_logit = self.baseline_classifier(baseline_features).squeeze(1)
        temporal_delta = self.temporal_head(temporal_features).squeeze(1)
        return baseline_logit + temporal_delta

    def predict_proba(
        self,
        signal: torch.Tensor,
        channel_availability: torch.Tensor,
    ) -> torch.Tensor:
        """Return probabilities corrected for negative subsampling."""
        logits = self(signal, channel_availability)
        corrected_logits = logits + self.config.sampling_prior_logit_correction
        return torch.sigmoid(corrected_logits)


@dataclass(frozen=True)
class EEGNetRecencyWeightedConfig:
    """Configuration for constrained minute-level recency weighting."""

    n_chans: int = 3
    chunk_samples: int = 5 * 256
    sequence_chunks: int = 45 * 60 // 5
    chunks_per_minute: int = 60 // 5
    embedding_dim: int = 32
    encoder_chunk_batch_size: int = 128
    encoder_dropout: float = 0.4
    softmax_temperature: float = 1.0
    max_temporal_strength: float = 0.25
    sampling_prior_logit_correction: float = 0.0

    @property
    def history_minutes(self) -> int:
        """Return the number of minute embeddings in one decision history."""
        return self.sequence_chunks // self.chunks_per_minute


class EEGNetRecencyWeightedRiskModel(nn.Module):
    """Replace uniform mean pooling with a constrained causal weighted mean.

    Only one logit per minute is learned. Softmax produces nonnegative weights
    that sum to one, and ``max_temporal_strength`` mixes them with a fixed
    uniform distribution. The model therefore begins as the exact baseline and
    cannot completely discard any part of the 45-minute history.
    """

    def __init__(self, config: EEGNetRecencyWeightedConfig) -> None:
        super().__init__()
        self._validate_config(config)
        self.config = config
        self.encoder = EEGNet(
            n_chans=config.n_chans,
            n_outputs=config.embedding_dim,
            n_times=config.chunk_samples,
            drop_prob=config.encoder_dropout,
            final_layer_with_constraint=True,
        )
        self.baseline_classifier = nn.Sequential(
            nn.LayerNorm(config.embedding_dim + config.n_chans),
            nn.Linear(config.embedding_dim + config.n_chans, 1),
        )
        self.recency_logits = nn.Parameter(
            torch.zeros(config.history_minutes, dtype=torch.float32)
        )

    @staticmethod
    def _validate_config(config: EEGNetRecencyWeightedConfig) -> None:
        """Reject invalid pooling and encoder dimensions."""
        for name, value in (
            ("n_chans", config.n_chans),
            ("chunk_samples", config.chunk_samples),
            ("sequence_chunks", config.sequence_chunks),
            ("chunks_per_minute", config.chunks_per_minute),
            ("embedding_dim", config.embedding_dim),
            ("encoder_chunk_batch_size", config.encoder_chunk_batch_size),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive.")
        if config.sequence_chunks % config.chunks_per_minute != 0:
            raise ValueError(
                "sequence_chunks must divide into whole minute embeddings."
            )
        if not 0.0 <= config.encoder_dropout < 1.0:
            raise ValueError(
                "encoder_dropout must be at least zero and less than one."
            )
        if config.softmax_temperature <= 0.0:
            raise ValueError("softmax_temperature must be positive.")
        if not 0.0 < config.max_temporal_strength <= 1.0:
            raise ValueError(
                "max_temporal_strength must be greater than zero and at most one."
            )

    def _encode_chunks(self, chunks: torch.Tensor) -> torch.Tensor:
        """Encode chunks in bounded groups to control accelerator memory."""
        embeddings = [
            self.encoder(chunk_batch)
            for chunk_batch in chunks.split(self.config.encoder_chunk_batch_size)
        ]
        return torch.cat(embeddings, dim=0)

    def _minute_embeddings(self, chunk_embeddings: torch.Tensor) -> torch.Tensor:
        """Average consecutive five-second features into minute features."""
        batch_size = chunk_embeddings.shape[0]
        return chunk_embeddings.reshape(
            batch_size,
            self.config.history_minutes,
            self.config.chunks_per_minute,
            self.config.embedding_dim,
        ).mean(dim=2)

    def effective_recency_weights(self) -> torch.Tensor:
        """Return constrained chronological weights from oldest to newest."""
        learned_weights = torch.softmax(
            self.recency_logits / self.config.softmax_temperature,
            dim=0,
        )
        uniform_weights = torch.full_like(
            learned_weights,
            1.0 / self.config.history_minutes,
        )
        strength = self.config.max_temporal_strength
        return (1.0 - strength) * uniform_weights + strength * learned_weights

    def uniform_weight_kl(self) -> torch.Tensor:
        """Return KL(effective weights || uniform weights)."""
        weights = self.effective_recency_weights()
        uniform_log_weight = -torch.log(
            weights.new_tensor(float(self.config.history_minutes))
        )
        return torch.sum(weights * (torch.log(weights) - uniform_log_weight))

    def initialize_from_baseline_state_dict(
        self,
        baseline_state_dict: dict[str, torch.Tensor],
    ) -> None:
        """Load the encoder and classifier from ``BaselineEEGNet``."""
        encoder_state = {
            name.removeprefix("encoder."): value
            for name, value in baseline_state_dict.items()
            if name.startswith("encoder.")
        }
        classifier_state = {
            name.removeprefix("classifier."): value
            for name, value in baseline_state_dict.items()
            if name.startswith("classifier.")
        }
        if not encoder_state or not classifier_state:
            raise ValueError(
                "The checkpoint is not a BaselineEEGNet state dictionary."
            )
        self.encoder.load_state_dict(encoder_state, strict=True)
        self.baseline_classifier.load_state_dict(classifier_state, strict=True)

    def set_baseline_trainable(self, trainable: bool) -> None:
        """Freeze or unfreeze the selected baseline parameters."""
        for module in (self.encoder, self.baseline_classifier):
            for parameter in module.parameters():
                parameter.requires_grad_(trainable)

    def keep_frozen_baseline_in_eval_mode(self) -> None:
        """Prevent frozen dropout and BatchNorm state changes."""
        if not any(parameter.requires_grad for parameter in self.encoder.parameters()):
            self.encoder.eval()
            self.baseline_classifier.eval()

    def forward(
        self,
        signal: torch.Tensor,
        channel_availability: torch.Tensor,
    ) -> torch.Tensor:
        """Return logits from raw five-second EEG chunks."""
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
        return self.forward_from_chunk_embeddings(
            chunk_embeddings,
            channel_availability,
        )

    def forward_from_chunk_embeddings(
        self,
        chunk_embeddings: torch.Tensor,
        channel_availability: torch.Tensor,
    ) -> torch.Tensor:
        """Return logits from cached chronological EEGNet embeddings."""
        expected_embedding_shape = (
            self.config.sequence_chunks,
            self.config.embedding_dim,
        )
        if (
            chunk_embeddings.ndim != 3
            or tuple(chunk_embeddings.shape[1:]) != expected_embedding_shape
        ):
            raise ValueError(
                "chunk_embeddings must have shape (batch, chunks, features) "
                f"with trailing dimensions {expected_embedding_shape}; found "
                f"{tuple(chunk_embeddings.shape)}."
            )
        batch_size = chunk_embeddings.shape[0]
        if channel_availability.ndim == 1:
            channel_availability = channel_availability.unsqueeze(0)
        if channel_availability.shape != (batch_size, self.config.n_chans):
            raise ValueError(
                "channel_availability must have shape (batch, channels); "
                f"found {tuple(channel_availability.shape)}."
            )

        minute_embeddings = self._minute_embeddings(chunk_embeddings)
        recency_weights = self.effective_recency_weights().to(
            dtype=minute_embeddings.dtype,
            device=minute_embeddings.device,
        )
        weighted_embedding = torch.sum(
            minute_embeddings * recency_weights.reshape(1, -1, 1),
            dim=1,
        )
        availability_features = channel_availability.to(
            dtype=weighted_embedding.dtype,
            device=weighted_embedding.device,
        )
        features = torch.cat(
            [weighted_embedding, availability_features],
            dim=1,
        )
        return self.baseline_classifier(features).squeeze(1)

    def predict_proba(
        self,
        signal: torch.Tensor,
        channel_availability: torch.Tensor,
    ) -> torch.Tensor:
        """Return probabilities corrected for negative subsampling."""
        logits = self(signal, channel_availability)
        return torch.sigmoid(
            logits + self.config.sampling_prior_logit_correction
        )
