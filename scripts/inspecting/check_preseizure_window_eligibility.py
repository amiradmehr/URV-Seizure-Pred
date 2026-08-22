from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

RUN_INVENTORY_PATH = (
    PROCESSED_DIR / "local_cross_run_inventory.csv"
)
SEIZURE_INVENTORY_PATH = (
    PROCESSED_DIR / "local_cross_seizure_inventory.csv"
)

OUTPUT_PATH = (
    PROCESSED_DIR / "preseizure_60min_eligibility.csv"
)

REQUIRED_HISTORY_SECONDS = 60 * 60


def load_csv(
    path: Path,
    required_columns: set[str],
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )

    dataframe = pd.read_csv(
        path,
        dtype={
            "subject": str,
            "session": str,
            "run": str,
        },
    )

    missing_columns = required_columns - set(dataframe.columns)

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
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(3)
    )

    result["session"] = (
        result["session"]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(2)
    )

    result["run"] = (
        result["run"]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(2)
    )

    return result


def main() -> None:
    runs = load_csv(
        RUN_INVENTORY_PATH,
        required_columns={
            "subject",
            "session",
            "run",
            "recording_duration_seconds",
            "recording_duration_hours",
            "channel_names",
            "edf_file",
        },
    )

    seizures = load_csv(
        SEIZURE_INVENTORY_PATH,
        required_columns={
            "subject",
            "session",
            "run",
            "onset_seconds",
            "duration_seconds",
            "event_type",
            "edf_file",
        },
    )

    runs = normalize_identifiers(runs)
    seizures = normalize_identifiers(seizures)

    merge_columns = [
        "subject",
        "session",
        "run",
    ]

    run_information = runs[
        merge_columns
        + [
            "recording_duration_seconds",
            "recording_duration_hours",
            "channel_names",
        ]
    ].copy()

    if run_information.duplicated(merge_columns).any():
        duplicates = run_information[
            run_information.duplicated(
                merge_columns,
                keep=False,
            )
        ]

        raise ValueError(
            "Duplicate run identifiers found:\n"
            + duplicates[
                merge_columns
            ].to_string(index=False)
        )

    eligibility = seizures.merge(
        run_information,
        on=merge_columns,
        how="left",
        validate="many_to_one",
    )

    missing_runs = eligibility[
        "recording_duration_seconds"
    ].isna()

    if missing_runs.any():
        raise ValueError(
            "Some seizures did not match a run inventory row. "
            f"Unmatched seizures: {int(missing_runs.sum())}"
        )

    eligibility["minutes_before_seizure"] = (
        eligibility["onset_seconds"] / 60
    )

    eligibility["preseizure_start_seconds"] = (
        eligibility["onset_seconds"]
        - REQUIRED_HISTORY_SECONDS
    )

    eligibility["eligible_60min_by_duration"] = (
        eligibility["onset_seconds"]
        >= REQUIRED_HISTORY_SECONDS
    )

    eligibility["seizure_end_seconds"] = (
        eligibility["onset_seconds"]
        + eligibility["duration_seconds"]
    )

    eligibility["seizure_fits_inside_run"] = (
        eligibility["seizure_end_seconds"]
        <= eligibility["recording_duration_seconds"]
    )

    eligibility["seconds_short_of_60min"] = (
        REQUIRED_HISTORY_SECONDS
        - eligibility["onset_seconds"]
    ).clip(lower=0)

    eligibility["minutes_short_of_60min"] = (
        eligibility["seconds_short_of_60min"] / 60
    )

    eligibility = eligibility.sort_values(
        by=[
            "subject",
            "session",
            "run",
            "onset_seconds",
        ]
    ).reset_index(drop=True)

    eligibility.insert(
        0,
        "seizure_id",
        [
            f"sub-{row.subject}_ses-{row.session}_"
            f"run-{row.run}_sz-{index + 1:03d}"
            for index, row in eligibility.iterrows()
        ],
    )

    eligibility.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    total_seizures = len(eligibility)

    eligible_count = int(
        eligibility[
            "eligible_60min_by_duration"
        ].sum()
    )

    ineligible_count = (
        total_seizures - eligible_count
    )

    eligible_percentage = (
        100 * eligible_count / total_seizures
        if total_seizures > 0
        else 0
    )

    eligible_subjects = eligibility.loc[
        eligibility[
            "eligible_60min_by_duration"
        ],
        "subject",
    ].nunique()

    all_subjects = eligibility[
        "subject"
    ].nunique()

    subjects_with_no_eligible_seizures = (
        eligibility.groupby("subject")[
            "eligible_60min_by_duration"
        ]
        .sum()
        .eq(0)
        .sum()
    )

    seizures_outside_run = int(
        (
            ~eligibility[
                "seizure_fits_inside_run"
            ]
        ).sum()
    )

    subject_summary = (
        eligibility.groupby("subject")
        .agg(
            total_seizures=(
                "seizure_id",
                "size",
            ),
            eligible_60min_seizures=(
                "eligible_60min_by_duration",
                "sum",
            ),
            earliest_seizure_minutes=(
                "minutes_before_seizure",
                "min",
            ),
            latest_seizure_minutes=(
                "minutes_before_seizure",
                "max",
            ),
        )
        .reset_index()
    )

    subject_summary["ineligible_seizures"] = (
        subject_summary["total_seizures"]
        - subject_summary[
            "eligible_60min_seizures"
        ]
    )

    subject_summary = subject_summary.sort_values(
        by=[
            "eligible_60min_seizures",
            "total_seizures",
            "subject",
        ],
        ascending=[
            False,
            False,
            True,
        ],
    )

    print("=" * 80)
    print("60-MINUTE PRE-SEIZURE WINDOW ELIGIBILITY")
    print("=" * 80)

    print(f"Required history: 60 minutes")
    print(f"Total seizures checked: {total_seizures}")

    print("\nEligibility:")
    print(
        f"Eligible seizures: {eligible_count}"
    )
    print(
        f"Ineligible seizures: {ineligible_count}"
    )
    print(
        "Eligible percentage: "
        f"{eligible_percentage:.2f}%"
    )

    print("\nSubjects:")
    print(
        f"Subjects with seizures: {all_subjects}"
    )
    print(
        "Subjects with at least one eligible seizure: "
        f"{eligible_subjects}"
    )
    print(
        "Subjects with no eligible seizures: "
        f"{subjects_with_no_eligible_seizures}"
    )

    print("\nConsistency checks:")
    print(
        "Seizures extending beyond run duration: "
        f"{seizures_outside_run}"
    )

    print("\nSeizures closest to the 60-minute threshold:")
    threshold_examples = eligibility[
        [
            "seizure_id",
            "subject",
            "session",
            "run",
            "onset_seconds",
            "minutes_before_seizure",
            "eligible_60min_by_duration",
            "minutes_short_of_60min",
        ]
    ].copy()

    threshold_examples[
        "distance_from_threshold"
    ] = (
        threshold_examples[
            "onset_seconds"
        ]
        - REQUIRED_HISTORY_SECONDS
    ).abs()

    print(
        threshold_examples.sort_values(
            "distance_from_threshold"
        )
        .head(20)
        .drop(
            columns="distance_from_threshold"
        )
        .to_string(index=False)
    )

    print("\nTop subjects by eligible seizure count:")
    print(
        subject_summary.head(25).to_string(
            index=False
        )
    )

    print("\nSaved:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()