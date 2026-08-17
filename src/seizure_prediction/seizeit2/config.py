"""Central configuration for the SeizeIT2 preprocessing pipeline."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PreprocessingConfig:
    """
    Configuration values shared by all preprocessing scripts.

    The project root is inferred from:

        project_root/
            src/
                seizure_prediction/
                    seizeit2/
                        config.py
    """

    project_root: Path = Path(__file__).resolve().parents[3]

    # ------------------------------------------------------------------
    # BIDS dataset settings
    # ------------------------------------------------------------------

    bids_task: str = "szMonitoring"

    # ------------------------------------------------------------------
    # Patient-level evaluation split
    # ------------------------------------------------------------------
    #
    # Subjects are intentionally held out as whole patients.  No EEG
    # recording or window from a validation/test patient may contribute
    # to training, preprocessing fitting, or model selection.
    train_subjects: tuple[str, ...] = tuple(
        f"{subject:03d}" for subject in range(1, 101)
    )
    validation_subjects: tuple[str, ...] = tuple(
        f"{subject:03d}" for subject in range(101, 113)
    )
    test_subjects: tuple[str, ...] = tuple(
        f"{subject:03d}" for subject in range(113, 126)
    )

    # ------------------------------------------------------------------
    # Signal preprocessing
    # ------------------------------------------------------------------

    # Preserve the native SeizeIT2 sampling rate; no resampling is applied.
    target_sfreq: float = 256.0

    bandpass_low_hz: float = 0.5
    bandpass_high_hz: float = 40.0

    # European power-line frequency.
    notch_frequency_hz: float = 50.0

    # ------------------------------------------------------------------
    # Label definition
    # ------------------------------------------------------------------

    # At decision time t, the model receives the preceding 45 minutes of
    # EEG and estimates the probability of seizure onset in the next
    # 10 minutes. There is no minimum warning horizon in this risk task.
    prediction_horizon_minutes: float = 0.0
    seizure_occurrence_period_minutes: float = 10.0

    # A seizure is evaluated only when the preceding EEG period meets
    # this continuous, clear-data requirement.
    minimum_preseizure_clear_minutes: float = 60.0

    # Exclude this period following seizure termination from interictal
    # examples.
    postictal_exclusion_minutes: float = 60.0

    # ------------------------------------------------------------------
    # Model-window definition
    # ------------------------------------------------------------------

    input_window_seconds: float = 45.0 * 60.0
    input_stride_seconds: float = 60.0

    # The future model loader divides each 45-minute history into these
    # fixed local EEG chunks before temporal aggregation.
    chunk_window_seconds: float = 5.0

    # ------------------------------------------------------------------
    # Standardization
    # ------------------------------------------------------------------

    zscore_epsilon: float = 1e-8

    # ------------------------------------------------------------------
    # Data types
    # ------------------------------------------------------------------

    signal_dtype: str = "float32"
    label_dtype: str = "int64"

    # SeizeIT2 EDF recordings contain two of these three electrode
    # locations.  The preprocessing output always has this stable three
    # channel order; a per-recording availability mask identifies the two
    # channels that were physically recorded.
    canonical_channel_names: tuple[str, ...] = (
        "BTE_LEFT",
        "BTE_RIGHT",
        "CROSS_HEAD",
    )

    @property
    def subject_split_map(self) -> dict[str, str]:
        """Return the configured split for every included subject."""
        return {
            **{
                subject: "train"
                for subject in self.train_subjects
            },
            **{
                subject: "validation"
                for subject in self.validation_subjects
            },
            **{
                subject: "test"
                for subject in self.test_subjects
            },
        }

    @property
    def included_subjects(self) -> tuple[str, ...]:
        """Return every subject to load, in split order."""
        return (
            self.train_subjects
            + self.validation_subjects
            + self.test_subjects
        )

    @property
    def raw_data_dir(self) -> Path:
        """Root of the downloaded BIDS dataset."""
        return self.project_root / "data" / "seizeit2" / "raw"

    @property
    def interim_data_dir(self) -> Path:
        """Temporary preprocessing output."""
        return self.project_root / "data" / "seizeit2" / "interim"

    @property
    def unscaled_recordings_dir(self) -> Path:
        """Filtered continuous recordings before z-score standardization."""
        return self.interim_data_dir / "unscaled_recordings"

    @property
    def manifests_dir(self) -> Path:
        """Metadata tables generated during preprocessing."""
        return self.interim_data_dir / "manifests"

    @property
    def scaler_parameters_dir(self) -> Path:
        """Global channel z-score parameters fitted on training patients."""
        return self.interim_data_dir / "scaler_parameters"

    @property
    def processed_data_dir(self) -> Path:
        """Final standardized train, validation, and test shards."""
        return self.project_root / "data" / "seizeit2" / "processed"

    def validate(self) -> None:
        """Validate configuration values before preprocessing starts."""
        split_subjects = {
            "train": set(self.train_subjects),
            "validation": set(self.validation_subjects),
            "test": set(self.test_subjects),
        }

        if any(not subjects for subjects in split_subjects.values()):
            raise ValueError(
                "Every patient-level split must contain at least one subject."
            )

        total_subjects = sum(
            len(subjects) for subjects in split_subjects.values()
        )

        unique_subjects = set().union(*split_subjects.values())

        if len(unique_subjects) != total_subjects:
            raise ValueError(
                "Patient-level split subject lists must not overlap."
            )

        if any(
            not subject.isdigit() or len(subject) != 3
            for subject in unique_subjects
        ):
            raise ValueError(
                "Subject IDs must be zero-padded three-digit strings."
            )

        if len(self.canonical_channel_names) != 3:
            raise ValueError(
                "The SeizeIT2 layout requires three canonical electrode slots."
            )

        if self.target_sfreq <= 0:
            raise ValueError("target_sfreq must be positive.")

        nyquist = self.target_sfreq / 2.0

        if self.bandpass_high_hz >= nyquist:
            raise ValueError(
                "bandpass_high_hz must be below the target Nyquist "
                f"frequency of {nyquist} Hz."
            )

        if self.bandpass_low_hz <= 0:
            raise ValueError("bandpass_low_hz must be greater than zero.")

        if self.bandpass_low_hz >= self.bandpass_high_hz:
            raise ValueError(
                "bandpass_low_hz must be below bandpass_high_hz."
            )

        if self.prediction_horizon_minutes < 0:
            raise ValueError(
                "prediction_horizon_minutes cannot be negative."
            )

        if self.seizure_occurrence_period_minutes <= 0:
            raise ValueError(
                "seizure_occurrence_period_minutes must be positive."
            )

        if self.minimum_preseizure_clear_minutes <= 0:
            raise ValueError(
                "minimum_preseizure_clear_minutes must be positive."
            )

        minimum_required_minutes = (
            self.input_window_seconds / 60.0
            + self.prediction_horizon_minutes
            + self.seizure_occurrence_period_minutes
        )

        if self.minimum_preseizure_clear_minutes < minimum_required_minutes:
            raise ValueError(
                "minimum_preseizure_clear_minutes must cover the complete "
                "history and prediction interval; expected at least "
                f"{minimum_required_minutes} minutes."
            )

        if self.input_window_seconds <= 0:
            raise ValueError("input_window_seconds must be positive.")

        if self.input_stride_seconds <= 0:
            raise ValueError("input_stride_seconds must be positive.")

        if self.chunk_window_seconds <= 0:
            raise ValueError("chunk_window_seconds must be positive.")

        history_chunks = (
            self.input_window_seconds / self.chunk_window_seconds
        )
        stride_chunks = (
            self.input_stride_seconds / self.chunk_window_seconds
        )

        if not history_chunks.is_integer():
            raise ValueError(
                "input_window_seconds must be divisible by "
                "chunk_window_seconds."
            )

        if not stride_chunks.is_integer():
            raise ValueError(
                "input_stride_seconds must be divisible by "
                "chunk_window_seconds."
            )

    def create_directories(self) -> None:
        """Create all generated-data directories."""
        directories = [
            self.interim_data_dir,
            self.unscaled_recordings_dir,
            self.manifests_dir,
            self.scaler_parameters_dir,
            self.processed_data_dir,
            self.processed_data_dir / "train",
            self.processed_data_dir / "validation",
            self.processed_data_dir / "test",
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


CONFIG = PreprocessingConfig()
