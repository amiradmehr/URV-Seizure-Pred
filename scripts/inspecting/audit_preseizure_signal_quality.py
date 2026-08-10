from pathlib import Path
import time

import mne
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

CONTAMINATION_PATH = (
    PROCESSED_DIR
    / "preseizure_60min_event_contamination.csv"
)

RUN_INVENTORY_PATH = (
    PROCESSED_DIR
    / "local_cross_run_inventory.csv"
)

WINDOW_OUTPUT_PATH = (
    PROCESSED_DIR
    / "preseizure_signal_quality_windows.csv"
)

SEIZURE_SUMMARY_OUTPUT_PATH = (
    PROCESSED_DIR
    / "preseizure_signal_quality_summary.csv"
)

FAILURE_OUTPUT_PATH = (
    PROCESSED_DIR
    / "preseizure_signal_quality_failures.csv"
)


HISTORY_SECONDS = 60 * 60
WINDOW_SECONDS = 10
EXPECTED_SFREQ = 256.0

EXPECTED_CONFIGURATIONS = {
    "BTEleft SD|CROSStop SD",
    "BTEright SD|CROSStop SD",
}


def load_csv(
    path: Path,
    required_columns: set[str],
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file does not exist: {path}"
        )

    dataframe = pd.read_csv(
        path,
        dtype={
            "subject": str,
            "session": str,
            "run": str,
        },
    )

    missing_columns = required_columns - set(
        dataframe.columns
    )

    if missing_columns:
        raise ValueError(
            f"{path.name} is missing columns: "
            f"{sorted(missing_columns)}"
        )

    return dataframe


def normalize_identifiers(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    result = dataframe.copy()

    result["subject"] = (
        result["subject"]
        .astype(str)
        .str.replace(
            r"\.0$",
            "",
            regex=True,
        )
        .str.zfill(3)
    )

    result["session"] = (
        result["session"]
        .astype(str)
        .str.replace(
            r"\.0$",
            "",
            regex=True,
        )
        .str.zfill(2)
    )

    result["run"] = (
        result["run"]
        .astype(str)
        .str.replace(
            r"\.0$",
            "",
            regex=True,
        )
        .str.zfill(2)
    )

    return result


def parse_boolean_series(
    values: pd.Series,
    column_name: str,
) -> pd.Series:
    parsed = (
        values.astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "true": True,
                "false": False,
                "1": True,
                "0": False,
            }
        )
    )

    if parsed.isna().any():
        invalid_values = (
            values[parsed.isna()]
            .drop_duplicates()
            .tolist()
        )

        raise ValueError(
            f"Could not parse Boolean values in "
            f"{column_name}: {invalid_values}"
        )

    return parsed.astype(bool)


def longest_constant_run(
    values: np.ndarray,
) -> int:
    """
    Return the longest number of consecutive identical samples.

    Examples:
        [1, 1, 1, 2] -> 3
        [1, 2, 3, 4] -> 1
    """
    if values.size == 0:
        return 0

    if values.size == 1:
        return 1

    equal_to_previous = (
        values[1:] == values[:-1]
    )

    if not equal_to_previous.any():
        return 1

    change_locations = (
        np.flatnonzero(
            ~equal_to_previous
        )
        + 1
    )

    boundaries = np.concatenate(
        [
            np.array([0]),
            change_locations,
            np.array([values.size]),
        ]
    )

    run_lengths = np.diff(boundaries)

    return int(run_lengths.max())


def calculate_channel_metrics(
    channel_volts: np.ndarray,
    sampling_frequency: float,
) -> dict[str, float | int | bool]:
    """
    Calculate raw-signal quality measurements for one channel
    in one 10-second window.
    """
    total_count = int(
        channel_volts.size
    )

    finite_mask = np.isfinite(
        channel_volts
    )

    finite_count = int(
        finite_mask.sum()
    )

    finite_percentage = (
        100.0
        * finite_count
        / total_count
        if total_count > 0
        else 0.0
    )

    if finite_count == 0:
        return {
            "all_finite": False,
            "finite_percentage": (
                finite_percentage
            ),
            "mean_uv": np.nan,
            "standard_deviation_uv": np.nan,
            "minimum_uv": np.nan,
            "maximum_uv": np.nan,
            "peak_to_peak_uv": np.nan,
            "maximum_absolute_uv": np.nan,
            "zero_percentage": np.nan,
            "identical_consecutive_percentage": (
                np.nan
            ),
            "longest_constant_run_samples": 0,
            "longest_constant_run_seconds": 0.0,
            "unique_value_count": 0,
            "minimum_value_percentage": np.nan,
            "maximum_value_percentage": np.nan,
        }

    finite_values_volts = (
        channel_volts[finite_mask]
    )

    finite_values_uv = (
        finite_values_volts * 1e6
    )

    zero_percentage = (
        100.0
        * np.count_nonzero(
            finite_values_volts == 0.0
        )
        / finite_values_volts.size
    )

    if finite_values_volts.size > 1:
        identical_consecutive_percentage = (
            100.0
            * np.count_nonzero(
                finite_values_volts[1:]
                == finite_values_volts[:-1]
            )
            / (
                finite_values_volts.size
                - 1
            )
        )
    else:
        identical_consecutive_percentage = 0.0

    longest_run_samples = (
        longest_constant_run(
            finite_values_volts
        )
    )

    minimum_value = float(
        np.min(finite_values_volts)
    )

    maximum_value = float(
        np.max(finite_values_volts)
    )

    minimum_value_percentage = (
        100.0
        * np.count_nonzero(
            finite_values_volts
            == minimum_value
        )
        / finite_values_volts.size
    )

    maximum_value_percentage = (
        100.0
        * np.count_nonzero(
            finite_values_volts
            == maximum_value
        )
        / finite_values_volts.size
    )

    return {
        "all_finite": bool(
            finite_mask.all()
        ),
        "finite_percentage": float(
            finite_percentage
        ),
        "mean_uv": float(
            np.mean(finite_values_uv)
        ),
        "standard_deviation_uv": float(
            np.std(
                finite_values_uv,
                ddof=0,
            )
        ),
        "minimum_uv": float(
            np.min(finite_values_uv)
        ),
        "maximum_uv": float(
            np.max(finite_values_uv)
        ),
        "peak_to_peak_uv": float(
            np.ptp(finite_values_uv)
        ),
        "maximum_absolute_uv": float(
            np.max(
                np.abs(
                    finite_values_uv
                )
            )
        ),
        "zero_percentage": float(
            zero_percentage
        ),
        "identical_consecutive_percentage": (
            float(
                identical_consecutive_percentage
            )
        ),
        "longest_constant_run_samples": (
            longest_run_samples
        ),
        "longest_constant_run_seconds": float(
            longest_run_samples
            / sampling_frequency
        ),
        "unique_value_count": int(
            np.unique(
                finite_values_volts
            ).size
        ),
        "minimum_value_percentage": float(
            minimum_value_percentage
        ),
        "maximum_value_percentage": float(
            maximum_value_percentage
        ),
    }


def standardize_channel_roles(
    raw: mne.io.BaseRaw,
) -> tuple[list[int], list[str]]:
    """
    Return channel indices and standardized functional roles.

    Every available physical electrode is retained in a stable location
    order.  A recording has two of ``BTE_LEFT``, ``BTE_RIGHT``, and
    ``CROSS_HEAD``; the audit intentionally does not synthesize the absent
    third channel.
    """
    channel_names = raw.ch_names

    raw_name_by_role = {
        "BTE_LEFT": "BTEleft SD",
        "BTE_RIGHT": "BTEright SD",
        "CROSS_HEAD": "CROSStop SD",
    }
    unexpected_channels = set(channel_names) - set(raw_name_by_role.values())

    if not unexpected_channels and len(channel_names) == 2:
        channel_indices = []
        channel_roles = []

        for role, raw_name in raw_name_by_role.items():
            if raw_name in channel_names:
                channel_indices.append(channel_names.index(raw_name))
                channel_roles.append(role)

        return channel_indices, channel_roles

    raise ValueError(
        "Unexpected channel configuration: "
        f"{channel_names}"
    )


def describe_numeric_column(
    dataframe: pd.DataFrame,
    column: str,
    percentiles: list[float],
) -> None:
    print(
        dataframe[column]
        .describe(
            percentiles=percentiles
        )
        .to_string()
    )


def main() -> None:
    script_start_time = time.time()

    contamination = load_csv(
        CONTAMINATION_PATH,
        required_columns={
            "seizure_id",
            "subject",
            "session",
            "run",
            "onset_seconds",
            "event_clean_strict",
        },
    )

    runs = load_csv(
        RUN_INVENTORY_PATH,
        required_columns={
            "subject",
            "session",
            "run",
            "edf_file",
            "channel_names",
        },
    )

    contamination = (
        normalize_identifiers(
            contamination
        )
    )

    runs = normalize_identifiers(
        runs
    )

    contamination[
        "event_clean_strict"
    ] = parse_boolean_series(
        contamination[
            "event_clean_strict"
        ],
        "event_clean_strict",
    )

    clean_seizures = contamination[
        contamination[
            "event_clean_strict"
        ]
    ].copy()

    merge_columns = [
        "subject",
        "session",
        "run",
    ]

    run_information = runs[
        merge_columns
        + [
            "edf_file",
            "channel_names",
        ]
    ].copy()

    duplicate_runs = (
        run_information.duplicated(
            merge_columns,
            keep=False,
        )
    )

    if duplicate_runs.any():
        duplicates = run_information[
            duplicate_runs
        ]

        raise ValueError(
            "Duplicate run identifiers found in "
            "the run inventory:\n"
            + duplicates[
                merge_columns
            ]
            .head(20)
            .to_string(index=False)
        )

    # The contamination CSV may already contain edf_file
    # and channel_names from earlier processing. Remove those
    # columns before merging to prevent pandas from creating:
    #
    #     edf_file_x
    #     edf_file_y
    #
    # We use the run inventory as the authoritative source.
    columns_to_drop_before_merge = [
        column
        for column in [
            "edf_file",
            "channel_names",
        ]
        if column in clean_seizures.columns
    ]

    if columns_to_drop_before_merge:
        clean_seizures = (
            clean_seizures.drop(
                columns=(
                    columns_to_drop_before_merge
                )
            )
        )

    clean_seizures = (
        clean_seizures.merge(
            run_information,
            on=merge_columns,
            how="left",
            validate="many_to_one",
        )
    )

    unmatched_edf_mask = (
        clean_seizures[
            "edf_file"
        ].isna()
    )

    if unmatched_edf_mask.any():
        unmatched = clean_seizures[
            unmatched_edf_mask
        ]

        raise ValueError(
            "Some seizures did not match an EDF run. "
            f"Unmatched seizures: "
            f"{len(unmatched)}\n"
            + unmatched[
                [
                    "seizure_id",
                    "subject",
                    "session",
                    "run",
                ]
            ]
            .head(20)
            .to_string(index=False)
        )

    unexpected_configuration_mask = ~(
        clean_seizures[
            "channel_names"
        ].isin(
            EXPECTED_CONFIGURATIONS
        )
    )

    if unexpected_configuration_mask.any():
        unexpected = (
            clean_seizures.loc[
                unexpected_configuration_mask,
                "channel_names",
            ]
            .value_counts(
                dropna=False
            )
        )

        raise ValueError(
            "Unexpected channel configurations:\n"
            + unexpected.to_string()
        )

    clean_seizures[
        "onset_seconds"
    ] = pd.to_numeric(
        clean_seizures[
            "onset_seconds"
        ],
        errors="coerce",
    )

    invalid_onset_mask = (
        clean_seizures[
            "onset_seconds"
        ].isna()
    )

    if invalid_onset_mask.any():
        raise ValueError(
            "Some clean seizures have invalid "
            "onset_seconds values."
        )

    clean_seizures = (
        clean_seizures.sort_values(
            by=[
                "subject",
                "session",
                "run",
                "onset_seconds",
            ]
        )
        .reset_index(drop=True)
    )

    expected_windows_per_seizure = (
        HISTORY_SECONDS
        // WINDOW_SECONDS
    )

    expected_channel_rows_per_seizure = (
        expected_windows_per_seizure
        * 2
    )

    print("=" * 80)
    print(
        "PRE-SEIZURE RAW SIGNAL QUALITY AUDIT"
    )
    print("=" * 80)

    print(
        "Strictly event-clean seizures to inspect: "
        f"{len(clean_seizures)}"
    )

    print(
        "History per seizure: "
        f"{HISTORY_SECONDS / 60:.0f} minutes"
    )

    print(
        "Quality-control window: "
        f"{WINDOW_SECONDS} seconds"
    )

    print(
        "Expected windows per seizure: "
        f"{expected_windows_per_seizure}"
    )

    print(
        "Expected channel-window rows per seizure: "
        f"{expected_channel_rows_per_seizure}"
    )

    window_rows: list[dict] = []
    failed_seizures: list[dict] = []

    total_seizures = len(
        clean_seizures
    )

    for seizure_number, seizure in enumerate(
        clean_seizures.itertuples(
            index=False
        ),
        start=1,
    ):
        edf_path = Path(
            seizure.edf_file
        )

        seizure_onset = float(
            seizure.onset_seconds
        )

        history_start_seconds = (
            seizure_onset
            - HISTORY_SECONDS
        )

        history_end_seconds = (
            seizure_onset
        )

        raw = None

        try:
            if not edf_path.exists():
                raise FileNotFoundError(
                    "EDF does not exist: "
                    f"{edf_path}"
                )

            raw = mne.io.read_raw_edf(
                edf_path,
                preload=False,
                verbose="ERROR",
            )

            sampling_frequency = float(
                raw.info["sfreq"]
            )

            if not np.isclose(
                sampling_frequency,
                EXPECTED_SFREQ,
            ):
                raise ValueError(
                    "Unexpected sampling rate: "
                    f"{sampling_frequency} Hz"
                )

            channel_types = (
                raw.get_channel_types()
            )

            if channel_types != [
                "eeg",
                "eeg",
            ]:
                raise ValueError(
                    "Expected exactly two EEG "
                    f"channels, found "
                    f"{channel_types}"
                )

            (
                channel_indices,
                channel_roles,
            ) = standardize_channel_roles(
                raw
            )

            start_sample = int(
                round(
                    history_start_seconds
                    * sampling_frequency
                )
            )

            end_sample = int(
                round(
                    history_end_seconds
                    * sampling_frequency
                )
            )

            if start_sample < 0:
                raise ValueError(
                    "Pre-seizure interval begins "
                    "before the start of the EDF."
                )

            if end_sample > raw.n_times:
                raise ValueError(
                    "Pre-seizure interval extends "
                    "past the end of the EDF."
                )

            expected_samples = int(
                round(
                    HISTORY_SECONDS
                    * sampling_frequency
                )
            )

            signal = raw.get_data(
                picks=channel_indices,
                start=start_sample,
                stop=end_sample,
            )

            expected_shape = (
                2,
                expected_samples,
            )

            if signal.shape != expected_shape:
                raise ValueError(
                    "Unexpected pre-seizure "
                    f"signal shape: {signal.shape}; "
                    f"expected {expected_shape}"
                )

            samples_per_window = int(
                round(
                    WINDOW_SECONDS
                    * sampling_frequency
                )
            )

            if (
                expected_samples
                % samples_per_window
                != 0
            ):
                raise ValueError(
                    "The one-hour sample count "
                    "cannot be divided evenly "
                    "into 10-second windows."
                )

            number_of_windows = (
                expected_samples
                // samples_per_window
            )

            windowed_signal = (
                signal.reshape(
                    2,
                    number_of_windows,
                    samples_per_window,
                )
            )

            for window_index in range(
                number_of_windows
            ):
                relative_window_start = (
                    window_index
                    * WINDOW_SECONDS
                )

                absolute_window_start = (
                    history_start_seconds
                    + relative_window_start
                )

                absolute_window_end = (
                    absolute_window_start
                    + WINDOW_SECONDS
                )

                seconds_until_seizure = (
                    seizure_onset
                    - absolute_window_start
                )

                for (
                    channel_position,
                    channel_role,
                ) in enumerate(
                    channel_roles
                ):
                    channel_values = (
                        windowed_signal[
                            channel_position,
                            window_index,
                            :,
                        ]
                    )

                    metrics = (
                        calculate_channel_metrics(
                            channel_values,
                            sampling_frequency,
                        )
                    )

                    original_channel_name = (
                        raw.ch_names[
                            channel_indices[
                                channel_position
                            ]
                        ]
                    )

                    window_rows.append(
                        {
                            "seizure_id": (
                                seizure.seizure_id
                            ),
                            "subject": (
                                seizure.subject
                            ),
                            "session": (
                                seizure.session
                            ),
                            "run": seizure.run,
                            "edf_file": str(
                                edf_path
                            ),
                            "channel_configuration": (
                                seizure.channel_names
                            ),
                            "original_channel_name": (
                                original_channel_name
                            ),
                            "channel_role": (
                                channel_role
                            ),
                            "sampling_frequency": (
                                sampling_frequency
                            ),
                            "window_index": (
                                window_index
                            ),
                            "window_start_seconds": (
                                absolute_window_start
                            ),
                            "window_end_seconds": (
                                absolute_window_end
                            ),
                            "seconds_until_seizure": (
                                seconds_until_seizure
                            ),
                            "minutes_until_seizure": (
                                seconds_until_seizure
                                / 60.0
                            ),
                            **metrics,
                        }
                    )

        except Exception as error:
            failed_seizures.append(
                {
                    "seizure_id": (
                        seizure.seizure_id
                    ),
                    "subject": (
                        seizure.subject
                    ),
                    "session": (
                        seizure.session
                    ),
                    "run": seizure.run,
                    "edf_file": str(
                        edf_path
                    ),
                    "error": (
                        f"{type(error).__name__}: "
                        f"{error}"
                    ),
                }
            )

        finally:
            if raw is not None:
                raw.close()

        if (
            seizure_number % 10 == 0
            or seizure_number
            == total_seizures
        ):
            elapsed_minutes = (
                time.time()
                - script_start_time
            ) / 60.0

            print(
                f"Processed "
                f"{seizure_number}/"
                f"{total_seizures} "
                f"seizures "
                f"({elapsed_minutes:.1f} "
                f"minutes elapsed)"
            )

    if not window_rows:
        raise RuntimeError(
            "No signal-quality rows "
            "were created."
        )

    windows = pd.DataFrame(
        window_rows
    )

    windows.to_csv(
        WINDOW_OUTPUT_PATH,
        index=False,
    )

    seizure_summary = (
        windows.groupby(
            [
                "seizure_id",
                "subject",
                "session",
                "run",
            ],
            as_index=False,
        )
        .agg(
            total_channel_windows=(
                "window_index",
                "size",
            ),
            unique_time_windows=(
                "window_index",
                "nunique",
            ),
            nonfinite_channel_windows=(
                "all_finite",
                lambda values: int(
                    (
                        ~values.astype(bool)
                    ).sum()
                ),
            ),
            minimum_standard_deviation_uv=(
                "standard_deviation_uv",
                "min",
            ),
            median_standard_deviation_uv=(
                "standard_deviation_uv",
                "median",
            ),
            maximum_standard_deviation_uv=(
                "standard_deviation_uv",
                "max",
            ),
            minimum_peak_to_peak_uv=(
                "peak_to_peak_uv",
                "min",
            ),
            median_peak_to_peak_uv=(
                "peak_to_peak_uv",
                "median",
            ),
            maximum_peak_to_peak_uv=(
                "peak_to_peak_uv",
                "max",
            ),
            maximum_zero_percentage=(
                "zero_percentage",
                "max",
            ),
            maximum_identical_percentage=(
                "identical_consecutive_percentage",
                "max",
            ),
            longest_constant_run_seconds=(
                "longest_constant_run_seconds",
                "max",
            ),
            maximum_absolute_uv=(
                "maximum_absolute_uv",
                "max",
            ),
            minimum_unique_value_count=(
                "unique_value_count",
                "min",
            ),
            maximum_minimum_value_percentage=(
                "minimum_value_percentage",
                "max",
            ),
            maximum_maximum_value_percentage=(
                "maximum_value_percentage",
                "max",
            ),
        )
    )

    seizure_summary.to_csv(
        SEIZURE_SUMMARY_OUTPUT_PATH,
        index=False,
    )

    if failed_seizures:
        failures = pd.DataFrame(
            failed_seizures
        )

        failures.to_csv(
            FAILURE_OUTPUT_PATH,
            index=False,
        )
    elif FAILURE_OUTPUT_PATH.exists():
        FAILURE_OUTPUT_PATH.unlink()

    print("\n" + "=" * 80)
    print(
        "SIGNAL-QUALITY AUDIT SUMMARY"
    )
    print("=" * 80)

    successful_seizure_count = (
        seizure_summary[
            "seizure_id"
        ].nunique()
    )

    print(
        "Seizures successfully inspected: "
        f"{successful_seizure_count}"
    )

    print(
        "Seizures that failed inspection: "
        f"{len(failed_seizures)}"
    )

    print(
        "Window-channel rows created: "
        f"{len(windows)}"
    )

    expected_rows = (
        len(clean_seizures)
        * expected_channel_rows_per_seizure
    )

    print(
        "Expected rows if all seizures "
        f"succeed: {expected_rows}"
    )

    print("\nFinite values:")

    nonfinite_window_count = int(
        (
            ~windows[
                "all_finite"
            ].astype(bool)
        ).sum()
    )

    print(
        "Channel-windows containing any "
        "nonfinite values: "
        f"{nonfinite_window_count}"
    )

    print(
        "\nStandard deviation "
        "in microvolts:"
    )

    describe_numeric_column(
        windows,
        "standard_deviation_uv",
        [
            0.001,
            0.01,
            0.05,
            0.50,
            0.95,
            0.99,
            0.999,
        ],
    )

    print(
        "\nPeak-to-peak amplitude "
        "in microvolts:"
    )

    describe_numeric_column(
        windows,
        "peak_to_peak_uv",
        [
            0.001,
            0.01,
            0.05,
            0.50,
            0.95,
            0.99,
            0.999,
        ],
    )

    print(
        "\nMaximum absolute amplitude "
        "in microvolts:"
    )

    describe_numeric_column(
        windows,
        "maximum_absolute_uv",
        [
            0.001,
            0.01,
            0.05,
            0.50,
            0.95,
            0.99,
            0.999,
        ],
    )

    print("\nZero percentage:")

    describe_numeric_column(
        windows,
        "zero_percentage",
        [
            0.50,
            0.95,
            0.99,
            0.999,
        ],
    )

    print(
        "\nIdentical consecutive-sample "
        "percentage:"
    )

    describe_numeric_column(
        windows,
        "identical_consecutive_percentage",
        [
            0.50,
            0.95,
            0.99,
            0.999,
        ],
    )

    print(
        "\nLongest constant run "
        "in seconds:"
    )

    describe_numeric_column(
        windows,
        "longest_constant_run_seconds",
        [
            0.50,
            0.95,
            0.99,
            0.999,
        ],
    )

    print(
        "\nUnique values per "
        "10-second channel-window:"
    )

    describe_numeric_column(
        windows,
        "unique_value_count",
        [
            0.001,
            0.01,
            0.05,
            0.50,
            0.95,
            0.99,
        ],
    )

    print(
        "\nDistributions by channel role:"
    )

    channel_role_summary = (
        windows.groupby(
            "channel_role"
        )
        .agg(
            channel_windows=(
                "window_index",
                "size",
            ),
            median_std_uv=(
                "standard_deviation_uv",
                "median",
            ),
            median_peak_to_peak_uv=(
                "peak_to_peak_uv",
                "median",
            ),
            maximum_zero_percentage=(
                "zero_percentage",
                "max",
            ),
            maximum_identical_percentage=(
                "identical_consecutive_percentage",
                "max",
            ),
            maximum_constant_run_seconds=(
                "longest_constant_run_seconds",
                "max",
            ),
            minimum_unique_value_count=(
                "unique_value_count",
                "min",
            ),
            maximum_absolute_uv=(
                "maximum_absolute_uv",
                "max",
            ),
        )
    )

    print(
        channel_role_summary.to_string()
    )

    if failed_seizures:
        failures = pd.DataFrame(
            failed_seizures
        )

        print("\nFailures:")

        print(
            failures.head(20)
            .to_string(index=False)
        )

        print(
            "\nSaved failures to:"
        )
        print(
            FAILURE_OUTPUT_PATH
        )

    elapsed_minutes = (
        time.time()
        - script_start_time
    ) / 60.0

    print("\nSaved:")
    print(WINDOW_OUTPUT_PATH)
    print(
        SEIZURE_SUMMARY_OUTPUT_PATH
    )

    print(
        "\nTotal runtime: "
        f"{elapsed_minutes:.1f} minutes"
    )


if __name__ == "__main__":
    main()
