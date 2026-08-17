"""Central configuration for the CHB-MIT preprocessing pipeline."""

import os
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
                    chb_mit/
                        config.py
    """

    project_root: Path = Path(__file__).resolve().parents[3]

    # ------------------------------------------------------------------
    # BIDS dataset settings
    # ------------------------------------------------------------------

    # The BIDS CHB-MIT release uses the same task name as SeizeIT2.
    bids_task: str = "szMonitoring"

    # ------------------------------------------------------------------
    # Patient-level evaluation split
    # ------------------------------------------------------------------
    #
    # Subjects are intentionally held out as whole patients.  No EEG
    # recording or window from a validation/test patient may contribute
    # to training, preprocessing fitting, or model selection.
    #
    # CHB-MIT contains 23 patients numbered sub-01 to sub-24; sub-21 does
    # not exist because the original chb21 recordings are a later session
    # of the same patient and were merged into sub-01/ses-02.
    #
    # The split deliberately places subjects 04, 06, 07, and 09 in train,
    # 10 in validation, and 23 in test: those are the only patients whose
    # recordings are long enough to expose a seizure with the required
    # 60 minutes of preceding clear EEG, so every split needs at least one
    # of them to contain positive decisions at all.
    train_subjects: tuple[str, ...] = (
        "01",
        "02",
        "03",
        "04",
        "05",
        "06",
        "07",
        "08",
        "09",
        "11",
        "12",
        "13",
        "14",
        "15",
        "16",
        "17",
    )
    validation_subjects: tuple[str, ...] = (
        "10",
        "18",
        "19",
        "20",
    )
    test_subjects: tuple[str, ...] = (
        "22",
        "23",
        "24",
    )

    # ------------------------------------------------------------------
    # Signal preprocessing
    # ------------------------------------------------------------------

    # Preserve the native CHB-MIT sampling rate; no resampling is applied.
    target_sfreq: float = 256.0

    bandpass_low_hz: float = 0.5
    bandpass_high_hz: float = 40.0

    # North American power-line frequency.  CHB-MIT was recorded at Boston
    # Children's Hospital and every sidecar reports PowerLineFrequency 60.
    notch_frequency_hz: float = 60.0

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

    # The BIDS CHB-MIT release stores exactly these eighteen double-banana
    # bipolar derivations, in this order, in every EDF.  The preprocessing
    # output always has this stable channel order; a per-recording
    # availability mask identifies the derivations that were physically
    # recorded, so a future release with a reduced montage still produces
    # aligned arrays.
    canonical_channel_names: tuple[str, ...] = (
        "FP1-F3",
        "F3-C3",
        "C3-P3",
        "P3-O1",
        "FP1-F7",
        "F7-T7",
        "T7-P7",
        "P7-O1",
        "FZ-CZ",
        "CZ-PZ",
        "FP2-F4",
        "F4-C4",
        "C4-P4",
        "P4-O2",
        "FP2-F8",
        "F8-T8",
        "T8-P8",
        "P8-O2",
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
        """
        Root of the downloaded BIDS dataset.

        ``src/dataset_downloads/chb_mit_download.py`` fetches the dataset
        into ``data/chb_mit/raw`` inside the project, so that location is
        the default.  Set ``CHB_MIT_BIDS_ROOT`` to read from elsewhere.
        """
        override = os.environ.get("CHB_MIT_BIDS_ROOT")

        if override:
            return Path(override).expanduser().resolve()

        return self.project_root / "data" / "chb_mit" / "raw" / "BIDS_CHB-MIT"

    @property
    def interim_data_dir(self) -> Path:
        """Temporary preprocessing output."""
        return self.project_root / "data" / "chb_mit" / "interim"

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
        return self.project_root / "data" / "chb_mit" / "processed"

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
            not subject.isdigit() or len(subject) != 2
            for subject in unique_subjects
        ):
            raise ValueError(
                "Subject IDs must be zero-padded two-digit strings."
            )

        if len(self.canonical_channel_names) != 18:
            raise ValueError(
                "The CHB-MIT layout requires eighteen bipolar derivations."
            )

        if len(set(self.canonical_channel_names)) != len(
            self.canonical_channel_names
        ):
            raise ValueError(
                "Canonical channel names must be unique."
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

        if not 0 < self.notch_frequency_hz < nyquist:
            raise ValueError(
                "notch_frequency_hz must lie between zero and the target "
                f"Nyquist frequency of {nyquist} Hz."
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
