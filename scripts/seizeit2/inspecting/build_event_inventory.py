from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = PROJECT_ROOT / "data" / "seizeit2" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "data" / "seizeit2" / "processed"


def parse_entities(path: Path) -> tuple[str, str, str]:
    name = path.name

    subject = next(
        part.removeprefix("sub-")
        for part in name.split("_")
        if part.startswith("sub-")
    )

    session = next(
        part.removeprefix("ses-")
        for part in name.split("_")
        if part.startswith("ses-")
    )

    run = next(
        part.removeprefix("run-")
        for part in name.split("_")
        if part.startswith("run-")
    )

    return subject, session, run


def main() -> None:
    event_files = sorted(DATA_ROOT.rglob("*_events.tsv"))

    if not event_files:
        raise FileNotFoundError(
            f"No event files found under {DATA_ROOT}"
        )

    all_rows = []

    for event_file in event_files:
        events = pd.read_csv(event_file, sep="\t")

        subject, session, run = parse_entities(event_file)

        events["subject"] = subject
        events["session"] = session
        events["run"] = run
        events["event_file"] = str(event_file)

        events["is_seizure"] = (
            events["eventType"]
            .fillna("")
            .astype(str)
            .str.startswith("sz_")
        )

        all_rows.append(events)

    inventory = pd.concat(
        all_rows,
        ignore_index=True,
    )

    seizures = inventory[inventory["is_seizure"]].copy()

    summary = (
        inventory.groupby("subject")
        .agg(
            total_events=("eventType", "size"),
            total_runs=("run", "nunique"),
            total_recording_seconds=(
                "recordingDuration",
                "sum",
            ),
            total_seizures=("is_seizure", "sum"),
        )
        .reset_index()
    )

    summary["total_recording_hours"] = (
        summary["total_recording_seconds"] / 3600
    )

    summary = summary.sort_values(
        by=["total_seizures", "subject"],
        ascending=[False, True],
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    inventory_path = OUTPUT_DIR / "all_events.csv"
    seizures_path = OUTPUT_DIR / "seizures.csv"
    summary_path = OUTPUT_DIR / "subject_summary.csv"

    inventory.to_csv(
        inventory_path,
        index=False,
    )

    seizures.to_csv(
        seizures_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print("=" * 80)
    print("SEIZEIT2 EVENT INVENTORY")
    print("=" * 80)

    print(f"Event files found: {len(event_files)}")
    print(f"Subjects found: {summary['subject'].nunique()}")
    print(f"Total event rows: {len(inventory)}")
    print(f"Total seizure events: {len(seizures)}")

    print("\nSubjects by seizure count:")
    print(
        summary[
            [
                "subject",
                "total_runs",
                "total_seizures",
                "total_recording_hours",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )

    print("\nSubjects with no seizures:")
    print(
        int((summary["total_seizures"] == 0).sum())
    )

    print("\nSubjects with at least 2 seizures:")
    print(
        int((summary["total_seizures"] >= 2).sum())
    )

    print("\nSubjects with at least 3 seizures:")
    print(
        int((summary["total_seizures"] >= 3).sum())
    )

    print("\nSaved:")
    print(inventory_path)
    print(seizures_path)
    print(summary_path)


if __name__ == "__main__":
    main()