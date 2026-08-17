from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INVENTORY_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "local_cross_run_inventory.csv"
)


def main() -> None:
    runs = pd.read_csv(
        INVENTORY_PATH,
        dtype={
            "subject": str,
            "session": str,
            "run": str,
        },
    )

    side_summary = (
        runs.groupby("subject")["channel_names"]
        .nunique()
        .reset_index(name="n_configurations")
    )

    switching_subjects = side_summary.loc[
        side_summary["n_configurations"] > 1,
        "subject",
    ].tolist()

    print("=" * 80)
    print("SUBJECTS WITH LEFT/RIGHT LOCAL-CHANNEL SWITCHES")
    print("=" * 80)

    print(f"Number of switching subjects: {len(switching_subjects)}")
    print(f"Subjects: {switching_subjects}")

    if not switching_subjects:
        return

    switching_runs = runs[
        runs["subject"].isin(switching_subjects)
    ].copy()

    switching_runs = switching_runs.sort_values(
        ["subject", "session", "run"]
    )

    columns = [
        "subject",
        "session",
        "run",
        "channel_names",
        "contains_seizure",
        "total_seizures",
        "recording_duration_hours",
    ]

    print("\nRuns for switching subjects:")
    print(
        switching_runs[columns].to_string(index=False)
    )

    print("\nConfiguration counts by subject:")
    print(
        switching_runs.groupby(
            ["subject", "channel_names"]
        )
        .size()
        .rename("run_count")
        .reset_index()
        .to_string(index=False)
    )

    print("\nSeizures by subject and configuration:")
    print(
        switching_runs.groupby(
            ["subject", "channel_names"]
        )["total_seizures"]
        .sum()
        .rename("seizure_count")
        .reset_index()
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()