from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = PROJECT_ROOT / "data" / "seizeit2" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "data" / "seizeit2" / "processed"


def parse_entities(path: Path) -> tuple[str, str, str]:
    parts = path.stem.split("_")

    subject = next(
        part.removeprefix("sub-")
        for part in parts
        if part.startswith("sub-")
    )

    session = next(
        part.removeprefix("ses-")
        for part in parts
        if part.startswith("ses-")
    )

    run = next(
        part.removeprefix("run-")
        for part in parts
        if part.startswith("run-")
    )

    return subject, session, run


def find_matching_edf(event_file: Path) -> Path:
    edf_name = event_file.name.replace(
        "_events.tsv",
        "_eeg.edf",
    )

    return event_file.parent / edf_name


def main() -> None:
    event_files = sorted(DATA_ROOT.rglob("*_events.tsv"))

    if not event_files:
        raise FileNotFoundError(
            f"No event files found under {DATA_ROOT}"
        )

    run_rows = []
    seizure_rows = []

    for event_file in event_files:
        events = pd.read_csv(event_file, sep="\t")

        subject, session, run = parse_entities(event_file)

        edf_file = find_matching_edf(event_file)

        if "recordingDuration" not in events.columns:
            raise ValueError(
                f"recordingDuration missing from {event_file}"
            )

        durations = (
            events["recordingDuration"]
            .dropna()
            .astype(float)
            .unique()
        )

        if len(durations) == 0:
            recording_duration = float("nan")
        elif len(durations) == 1:
            recording_duration = float(durations[0])
        else:
            raise ValueError(
                "Multiple recording durations found in one run: "
                f"{event_file}"
            )

        event_types = (
            events["eventType"]
            .fillna("")
            .astype(str)
        )

        seizure_mask = event_types.str.startswith("sz_")
        seizure_events = events[seizure_mask].copy()

        run_rows.append(
            {
                "subject": subject,
                "session": session,
                "run": run,
                "recording_duration_seconds": recording_duration,
                "recording_duration_hours": (
                    recording_duration / 3600
                ),
                "total_events": len(events),
                "total_seizures": int(seizure_mask.sum()),
                "contains_seizure": bool(seizure_mask.any()),
                "event_file": str(event_file),
                "edf_file": str(edf_file),
                "edf_exists": edf_file.exists(),
            }
        )

        for _, seizure in seizure_events.iterrows():
            seizure_rows.append(
                {
                    "subject": subject,
                    "session": session,
                    "run": run,
                    "onset_seconds": float(seizure["onset"]),
                    "duration_seconds": float(seizure["duration"]),
                    "event_type": seizure["eventType"],
                    "lateralization": seizure.get(
                        "lateralization"
                    ),
                    "localization": seizure.get(
                        "localization"
                    ),
                    "vigilance": seizure.get(
                        "vigilance"
                    ),
                    "event_file": str(event_file),
                    "edf_file": str(edf_file),
                }
            )

    runs = pd.DataFrame(run_rows)
    seizures = pd.DataFrame(seizure_rows)

    subject_summary = (
        runs.groupby("subject")
        .agg(
            total_runs=("run", "size"),
            seizure_runs=("contains_seizure", "sum"),
            total_seizures=("total_seizures", "sum"),
            total_recording_hours=(
                "recording_duration_hours",
                "sum",
            ),
            missing_edf_files=(
                "edf_exists",
                lambda values: int((~values).sum()),
            ),
        )
        .reset_index()
        .sort_values(
            by=["total_seizures", "subject"],
            ascending=[False, True],
        )
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    runs_path = OUTPUT_DIR / "run_inventory.csv"
    seizures_path = OUTPUT_DIR / "seizure_inventory.csv"
    summary_path = OUTPUT_DIR / "subject_summary_corrected.csv"

    runs.to_csv(runs_path, index=False)
    seizures.to_csv(seizures_path, index=False)
    subject_summary.to_csv(summary_path, index=False)

    print("=" * 80)
    print("SEIZEIT2 RUN INVENTORY")
    print("=" * 80)

    print(f"Total runs: {len(runs)}")
    print(f"Total subjects: {runs['subject'].nunique()}")
    print(f"Total seizures: {len(seizures)}")
    print(
        "Runs containing seizures: "
        f"{int(runs['contains_seizure'].sum())}"
    )
    print(
        "Runs without seizures: "
        f"{int((~runs['contains_seizure']).sum())}"
    )
    print(
        "Missing EDF files: "
        f"{int((~runs['edf_exists']).sum())}"
    )
    print(
        "Total recording hours: "
        f"{runs['recording_duration_hours'].sum():.2f}"
    )

    print("\nTop subjects by seizure count:")
    print(
        subject_summary[
            [
                "subject",
                "total_runs",
                "seizure_runs",
                "total_seizures",
                "total_recording_hours",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )

    print("\nSaved:")
    print(runs_path)
    print(seizures_path)
    print(summary_path)


if __name__ == "__main__":
    main()