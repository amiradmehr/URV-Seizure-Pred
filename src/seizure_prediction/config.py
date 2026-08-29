"""Central configuration for the SeizeIT2 preprocessing pipeline."""

import argparse
import dataclasses
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
                    config.py

    One instance describes one *label definition*: an input window paired with
    a seizure occurrence period.  `build_config` produces the instance for a
    given combination, tagging its generated directories so that the twelve
    combinations of the current sweep coexist on disk.  The filtered EEG
    itself is never tagged, because window and horizon change only which
    decision indices exist and how they are labeled -- see
    `unscaled_recordings_dir`.
    """

    project_root: Path = Path(__file__).resolve().parents[2]

    # ------------------------------------------------------------------
    # Label-definition identity
    # ------------------------------------------------------------------

    # Names the input-window/horizon combination this config describes, and
    # the subdirectory its generated manifests and shards live under.  `None`
    # means the untagged default layout.  Set by `build_config`.
    experiment_tag: str | None = None

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

    # At decision time t, the model receives the preceding
    # `input_window_seconds` of EEG and estimates the probability of seizure
    # onset in the next `seizure_occurrence_period_minutes`.  There is no
    # minimum warning horizon in this risk task.
    prediction_horizon_minutes: float = 0.0
    seizure_occurrence_period_minutes: float = 10.0

    # A seizure is evaluated only when the preceding EEG period meets this
    # continuous, clear-data requirement.  This is held fixed across the
    # window/horizon sweep rather than derived from either: a constant rule
    # makes every combination score the same cohort of eligible seizures, so
    # their metrics are directly comparable.  It is not a statement that a
    # 30-minute lead-in is intrinsically required -- a decision whose own
    # history is not clear is still discarded by
    # `create_labeled_prediction_decisions`.
    minimum_preseizure_clear_minutes: float = 30.0

    # Exclude this period following seizure termination from interictal
    # examples.
    postictal_exclusion_minutes: float = 60.0

    # ------------------------------------------------------------------
    # Model-window definition
    # ------------------------------------------------------------------

    # The longest window in `INPUT_WINDOW_MINUTE_CHOICES`, so the untagged
    # default is the widest history any swept combination reads.
    input_window_seconds: float = 30.0 * 60.0
    input_stride_seconds: float = 60.0

    # The model loader divides each history into these fixed local EEG chunks
    # before temporal aggregation.
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
    def dataset_data_dir(self) -> Path:
        """Root of every generated and downloaded file for this dataset."""
        return self.project_root / "data" / "seizeit2"

    @property
    def raw_data_dir(self) -> Path:
        """Root of the downloaded BIDS dataset."""
        return self.dataset_data_dir / "raw"

    @property
    def shared_interim_data_dir(self) -> Path:
        """
        Interim output that no label definition can change.

        Filtering and standardization statistics depend only on the EEG, not
        on the input window or the seizure occurrence period, so everything
        under here is written once and read by every tagged build.  Nothing
        in this tree may be cleared by a tagged build; see
        `clear_generated_directory`.
        """
        return self.dataset_data_dir / "interim" / "_shared"

    @property
    def interim_data_dir(self) -> Path:
        """Interim preprocessing output for this label definition."""
        base = self.dataset_data_dir / "interim"
        if self.experiment_tag is None:
            return base
        return base / self.experiment_tag

    @property
    def unscaled_recordings_dir(self) -> Path:
        """
        Filtered continuous recordings before z-score standardization.

        Shared across every label definition.  This is the only full-size
        copy of the EEG the pipeline keeps: standardization is a per-channel
        affine map applied when a decision is loaded, so no standardized
        duplicate is written.
        """
        return self.shared_interim_data_dir / "unscaled_recordings"

    @property
    def decision_checkpoints_dir(self) -> Path:
        """Per-recording decision tables, resumable within one label definition."""
        return self.interim_data_dir / "decision_checkpoints"

    @property
    def manifests_dir(self) -> Path:
        """Metadata tables generated during preprocessing."""
        return self.interim_data_dir / "manifests"

    @property
    def scaler_parameters_dir(self) -> Path:
        """Fitted per-channel standardization parameters, shared across tags."""
        return self.shared_interim_data_dir / "scaler_parameters"

    @property
    def processed_data_dir(self) -> Path:
        """Decision labels and metadata per train, validation, and test split."""
        base = self.dataset_data_dir / "processed"
        if self.experiment_tag is None:
            return base
        return base / self.experiment_tag

    @property
    def embedding_cache_dir(self) -> Path:
        """
        Cached per-chunk encoder embeddings, shared across label definitions.

        One embedding per five-second chunk of a whole recording, so the cache
        depends on the encoder and the standardization but not on the window
        or horizon that later aggregates those chunks.
        """
        return self.dataset_data_dir / "embedding_cache"

    @property
    def handcrafted_feature_cache_dir(self) -> Path:
        """Cached per-chunk handcrafted features, shared across label definitions."""
        return self.dataset_data_dir / "handcrafted_feature_cache"

    def scaler_document_path(self, mode: str, statistic: str) -> Path:
        """Return the shared scaler document for one normalization choice."""
        return self.scaler_parameters_dir / f"{mode}_{statistic}.json"

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

        # The clear-data rule decides which seizures may be prediction
        # targets, so it must cover at least the history the model reads at
        # the moment of onset.  It deliberately does not have to cover the
        # occurrence period as well: an earlier decision in that period whose
        # own history is not clear is discarded individually by
        # `create_labeled_prediction_decisions`, so requiring
        # window + horizon here would only discard whole seizures that still
        # have scoreable decisions.
        minimum_required_minutes = self.input_window_seconds / 60.0

        if self.minimum_preseizure_clear_minutes < minimum_required_minutes:
            raise ValueError(
                "minimum_preseizure_clear_minutes must cover at least the "
                "input window the model reads at seizure onset; expected at "
                f"least {minimum_required_minutes} minutes, found "
                f"{self.minimum_preseizure_clear_minutes}."
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

        if self.experiment_tag is not None:
            if not self.experiment_tag:
                raise ValueError("experiment_tag cannot be an empty string.")

            if Path(self.experiment_tag).name != self.experiment_tag:
                raise ValueError(
                    "experiment_tag must be a single path component; found "
                    f"{self.experiment_tag!r}."
                )

    def create_directories(self) -> None:
        """Create all generated-data directories."""
        directories = [
            self.shared_interim_data_dir,
            self.unscaled_recordings_dir,
            self.scaler_parameters_dir,
            self.interim_data_dir,
            self.decision_checkpoints_dir,
            self.manifests_dir,
            self.processed_data_dir,
            self.processed_data_dir / "train",
            self.processed_data_dir / "validation",
            self.processed_data_dir / "test",
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


CONFIG = PreprocessingConfig()


# ----------------------------------------------------------------------
# Label-definition sweep
# ----------------------------------------------------------------------

# The input windows and seizure occurrence periods this project sweeps.
# Their product is the set of label definitions `build_dataset.py` builds.
INPUT_WINDOW_MINUTE_CHOICES: tuple[float, ...] = (30.0, 15.0, 10.0, 5.0)

SEIZURE_OCCURRENCE_PERIOD_MINUTE_CHOICES: tuple[float, ...] = (2.0, 5.0, 10.0)


def experiment_tag_for(
    input_window_minutes: float,
    seizure_occurrence_period_minutes: float,
) -> str:
    """Return the directory name identifying one label definition."""
    return (
        f"w{input_window_minutes:g}"
        f"_h{seizure_occurrence_period_minutes:g}"
    )


def build_config(
    input_window_minutes: float | None = None,
    seizure_occurrence_period_minutes: float | None = None,
) -> PreprocessingConfig:
    """
    Return a validated config for one input-window/horizon combination.

    With both arguments omitted this returns the untagged `CONFIG` singleton
    unchanged.  Otherwise the returned config carries an `experiment_tag` of
    ``w{window}_h{horizon}``, which nests its manifests and processed shards
    so that combinations do not overwrite one another.  The filtered EEG and
    the fitted scalers stay shared; only the decision indices and their
    labels differ between combinations.
    """
    if (
        input_window_minutes is None
        and seizure_occurrence_period_minutes is None
    ):
        return CONFIG

    if input_window_minutes is None:
        input_window_minutes = CONFIG.input_window_seconds / 60.0

    if seizure_occurrence_period_minutes is None:
        seizure_occurrence_period_minutes = (
            CONFIG.seizure_occurrence_period_minutes
        )

    if input_window_minutes <= 0:
        raise ValueError(
            "input_window_minutes must be positive; found "
            f"{input_window_minutes}."
        )

    if seizure_occurrence_period_minutes <= 0:
        raise ValueError(
            "seizure_occurrence_period_minutes must be positive; found "
            f"{seizure_occurrence_period_minutes}."
        )

    config = dataclasses.replace(
        CONFIG,
        input_window_seconds=input_window_minutes * 60.0,
        seizure_occurrence_period_minutes=seizure_occurrence_period_minutes,
        experiment_tag=experiment_tag_for(
            input_window_minutes,
            seizure_occurrence_period_minutes,
        ),
    )
    config.validate()
    return config


def sweep_configurations(
    input_window_minutes: tuple[float, ...] | list[float] | None = None,
    seizure_occurrence_period_minutes: (
        tuple[float, ...] | list[float] | None
    ) = None,
) -> list[PreprocessingConfig]:
    """
    Return one validated config per requested window/horizon combination.

    Defaults to the full sweep, `INPUT_WINDOW_MINUTE_CHOICES` crossed with
    `SEIZURE_OCCURRENCE_PERIOD_MINUTE_CHOICES`.
    """
    windows = (
        INPUT_WINDOW_MINUTE_CHOICES
        if input_window_minutes is None
        else tuple(float(value) for value in input_window_minutes)
    )
    horizons = (
        SEIZURE_OCCURRENCE_PERIOD_MINUTE_CHOICES
        if seizure_occurrence_period_minutes is None
        else tuple(float(value) for value in seizure_occurrence_period_minutes)
    )

    if not windows:
        raise ValueError("At least one input window must be requested.")

    if not horizons:
        raise ValueError(
            "At least one seizure occurrence period must be requested."
        )

    return [
        build_config(
            input_window_minutes=window,
            seizure_occurrence_period_minutes=horizon,
        )
        for window in windows
        for horizon in horizons
    ]


def add_label_definition_arguments(
    parser: argparse.ArgumentParser,
) -> None:
    """Add the shared `--window-minutes`/`--horizon-minutes` flags."""
    parser.add_argument(
        "--window-minutes",
        type=float,
        default=None,
        help=(
            "Input window the model reads at each decision, in minutes "
            f"(default: {CONFIG.input_window_seconds / 60.0:g})."
        ),
    )
    parser.add_argument(
        "--horizon-minutes",
        type=float,
        default=None,
        help=(
            "Seizure occurrence period a decision predicts over, in minutes "
            f"(default: {CONFIG.seizure_occurrence_period_minutes:g})."
        ),
    )


def resolve_label_definition(
    arguments: argparse.Namespace,
) -> PreprocessingConfig:
    """Return the config named by `add_label_definition_arguments` flags."""
    return build_config(
        input_window_minutes=arguments.window_minutes,
        seizure_occurrence_period_minutes=arguments.horizon_minutes,
    )
