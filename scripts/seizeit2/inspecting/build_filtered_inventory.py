from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DIR = PROJECT_ROOT / "data" / "seizeit2" / "processed"

RUN_INVENTORY_PATH = PROCESSED_DIR / "run_inventory.csv"
RECORDING_AUDIT_PATH = PROCESSED_DIR / "recording_audit.csv"
SEIZURE_INVENTORY_PATH = PROCESSED_DIR / "seizure_inventory.csv"

FILTERED_RUNS_PATH = (
    PROCESSED_DIR / "local_cross_run_inventory.csv"
)
FILTERED_SEIZURES_PATH = (
    PROCESSED_DIR / "local_cross_seizure_inventory.csv"
)
FILTERED_SUBJECTS_PATH = (
    PROCESSED_DIR / "local_cross_subject_summary.csv"
)
EXCLUDED_RUNS_PATH = (
    PROCESSED_DIR / "excluded_bilateral_run_inventory.csv"
)


LOCAL_CROSS_CONFIGURATIONS = {
    "BTEleft SD|CROSStop SD",
    "BTEright SD|CROSStop SD",
}

BILATERAL_CONFIGURATION = "BTEleft SD|BTEright SD"


def load_csv(
    path: Path,
    *,
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

    for column in ["subject", "session", "run"]:
        result[column] = (
            result[column]
            .astype(str)
            .str.replace(r"\.0$", "", regex=True)
            .str.zfill(3 if column == "subject" else 2)
        )

    return result


def main() -> None:
    runs = load_csv(
        RUN_INVENTORY_PATH,
        required_columns={
            "subject",
            "session",
            "run",
            "total_seizures",
            "contains_seizure",
            "recording_duration_hours",
            "edf_file",
        },
    )

    audit = load_csv(
        RECORDING_AUDIT_PATH,
        required_columns={
            "subject",
            "session",
            "run",
            "channel_names",
            "opened_successfully",
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
        },
    )

    runs = normalize_identifiers(runs)
    audit = normalize_identifiers(audit)
    seizures = normalize_identifiers(seizures)

    merge_columns = ["subject", "session", "run"]

    audit_subset = audit[
        merge_columns
        + [
            "channel_names",
            "channel_types",
            "sampling_frequency",
            "n_channels",
            "opened_successfully",
        ]
    ].copy()

    if audit_subset.duplicated(merge_columns).any():
        duplicates = audit_subset[
            audit_subset.duplicated(
                merge_columns,
                keep=False,
            )
        ]

        raise ValueError(
            "Recording audit contains duplicate run identifiers:\n"
            + duplicates[
                merge_columns
            ].head(20).to_string(index=False)
        )

    merged_runs = runs.merge(
        audit_subset,
        on=merge_columns,
        how="left",
        validate="one_to_one",
    )

    missing_audit = merged_runs[
        "channel_names"
    ].isna()

    if missing_audit.any():
        raise ValueError(
            "Some runs did not match a recording-audit row. "
            f"Unmatched runs: {int(missing_audit.sum())}"
        )

    merged_runs["is_local_cross"] = (
        merged_runs["channel_names"]
        .isin(LOCAL_CROSS_CONFIGURATIONS)
    )

    merged_runs["is_bilateral"] = (
        merged_runs["channel_names"]
        == BILATERAL_CONFIGURATION
    )

    unknown_configuration = ~(
        merged_runs["is_local_cross"]
        | merged_runs["is_bilateral"]
    )

    if unknown_configuration.any():
        print("\nWarning: unknown channel configurations found:")
        print(
            merged_runs.loc[
                unknown_configuration,
                "channel_names",
            ].value_counts()
        )

    filtered_runs = merged_runs[
        merged_runs["is_local_cross"]
    ].copy()

    excluded_runs = merged_runs[
        ~merged_runs["is_local_cross"]
    ].copy()

    filtered_run_keys = filtered_runs[
        merge_columns
    ].drop_duplicates()

    filtered_seizures = seizures.merge(
        filtered_run_keys,
        on=merge_columns,
        how="inner",
        validate="many_to_one",
    )

    subject_summary = (
        filtered_runs.groupby("subject")
        .agg(
            total_runs=("run", "size"),
            seizure_runs=("contains_seizure", "sum"),
            total_seizures=("total_seizures", "sum"),
            total_recording_hours=(
                "recording_duration_hours",
                "sum",
            ),
            left_cross_runs=(
                "channel_names",
                lambda values: int(
                    (
                        values
                        == "BTEleft SD|CROSStop SD"
                    ).sum()
                ),
            ),
            right_cross_runs=(
                "channel_names",
                lambda values: int(
                    (
                        values
                        == "BTEright SD|CROSStop SD"
                    ).sum()
                ),
            ),
        )
        .reset_index()
    )

    all_subjects = set(merged_runs["subject"])
    filtered_subjects = set(filtered_runs["subject"])
    excluded_subjects = sorted(
        all_subjects - filtered_subjects
    )

    original_total_runs = len(merged_runs)
    filtered_total_runs = len(filtered_runs)

    original_total_seizures = int(
        merged_runs["total_seizures"].sum()
    )
    filtered_total_seizures = int(
        filtered_runs["total_seizures"].sum()
    )

    original_total_hours = float(
        merged_runs[
            "recording_duration_hours"
        ].sum()
    )
    filtered_total_hours = float(
        filtered_runs[
            "recording_duration_hours"
        ].sum()
    )

    subject_summary = subject_summary.sort_values(
        by=["total_seizures", "subject"],
        ascending=[False, True],
    )

    filtered_runs.to_csv(
        FILTERED_RUNS_PATH,
        index=False,
    )
    filtered_seizures.to_csv(
        FILTERED_SEIZURES_PATH,
        index=False,
    )
    subject_summary.to_csv(
        FILTERED_SUBJECTS_PATH,
        index=False,
    )
    excluded_runs.to_csv(
        EXCLUDED_RUNS_PATH,
        index=False,
    )

    subjects_with_at_least_1 = int(
        (
            subject_summary["total_seizures"]
            >= 1
        ).sum()
    )
    subjects_with_at_least_2 = int(
        (
            subject_summary["total_seizures"]
            >= 2
        ).sum()
    )
    subjects_with_at_least_3 = int(
        (
            subject_summary["total_seizures"]
            >= 3
        ).sum()
    )
    subjects_with_at_least_5 = int(
        (
            subject_summary["total_seizures"]
            >= 5
        ).sum()
    )

    subjects_using_both_local_sides = int(
        (
            (subject_summary["left_cross_runs"] > 0)
            & (subject_summary["right_cross_runs"] > 0)
        ).sum()
    )

    print("=" * 80)
    print("SEIZEIT2 LOCAL-BTE + CROSS-HEAD INVENTORY")
    print("=" * 80)

    print("\nOriginal dataset:")
    print(f"Subjects: {len(all_subjects)}")
    print(f"Runs: {original_total_runs}")
    print(f"Seizures: {original_total_seizures}")
    print(
        "Recording hours: "
        f"{original_total_hours:.2f}"
    )

    print("\nFiltered local-plus-cross subset:")
    print(f"Subjects represented: {len(filtered_subjects)}")
    print(f"Runs retained: {filtered_total_runs}")
    print(
        "Runs excluded: "
        f"{original_total_runs - filtered_total_runs}"
    )
    print(f"Seizures retained: {filtered_total_seizures}")
    print(
        "Seizures excluded: "
        f"{original_total_seizures - filtered_total_seizures}"
    )
    print(
        "Recording hours retained: "
        f"{filtered_total_hours:.2f}"
    )

    print("\nRetention percentages:")
    print(
        "Runs retained: "
        f"{100 * filtered_total_runs / original_total_runs:.2f}%"
    )
    print(
        "Seizures retained: "
        f"{100 * filtered_total_seizures / original_total_seizures:.2f}%"
    )
    print(
        "Recording hours retained: "
        f"{100 * filtered_total_hours / original_total_hours:.2f}%"
    )
    print(
        "Subjects retained: "
        f"{100 * len(filtered_subjects) / len(all_subjects):.2f}%"
    )

    print("\nFiltered-subset subject eligibility:")
    print(
        "Subjects with at least 1 seizure: "
        f"{subjects_with_at_least_1}"
    )
    print(
        "Subjects with at least 2 seizures: "
        f"{subjects_with_at_least_2}"
    )
    print(
        "Subjects with at least 3 seizures: "
        f"{subjects_with_at_least_3}"
    )
    print(
        "Subjects with at least 5 seizures: "
        f"{subjects_with_at_least_5}"
    )

    print(
        "\nSubjects with both left-local+cross and "
        "right-local+cross runs: "
        f"{subjects_using_both_local_sides}"
    )

    print("\nSubjects completely removed by filtering:")
    print(len(excluded_subjects))

    if excluded_subjects:
        print(", ".join(excluded_subjects))

    print("\nRetained channel configurations:")
    print(
        filtered_runs[
            "channel_names"
        ].value_counts()
    )

    print("\nExcluded channel configurations:")
    print(
        excluded_runs[
            "channel_names"
        ].value_counts()
    )

    print("\nTop retained subjects by seizure count:")
    print(
        subject_summary[
            [
                "subject",
                "total_runs",
                "seizure_runs",
                "total_seizures",
                "total_recording_hours",
                "left_cross_runs",
                "right_cross_runs",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )

    print("\nSaved:")
    print(FILTERED_RUNS_PATH)
    print(FILTERED_SEIZURES_PATH)
    print(FILTERED_SUBJECTS_PATH)
    print(EXCLUDED_RUNS_PATH)


if __name__ == "__main__":
    main()