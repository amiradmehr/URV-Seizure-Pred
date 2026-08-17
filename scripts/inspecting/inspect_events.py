from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "raw"

SUBJECT_ID = "001"
SUBJECT_DIR = DATA_ROOT / f"sub-{SUBJECT_ID}"


def main() -> None:
    print("=" * 80)
    print("SEIZEIT2 EVENT FILE INSPECTION")
    print("=" * 80)

    print(f"Subject directory: {SUBJECT_DIR}")
    print(f"Subject directory exists: {SUBJECT_DIR.exists()}")

    if not SUBJECT_DIR.exists():
        raise FileNotFoundError(
            f"Subject directory does not exist: {SUBJECT_DIR}"
        )

    event_files = sorted(
        SUBJECT_DIR.rglob("*_events.tsv")
    )

    print(f"\nNumber of event files found: {len(event_files)}")

    if not event_files:
        raise FileNotFoundError(
            f"No _events.tsv files found under {SUBJECT_DIR}"
        )

    for index, event_file in enumerate(event_files, start=1):
        print("\n" + "-" * 80)
        print(f"EVENT FILE {index}")
        print("-" * 80)
        print(f"Path: {event_file}")

        events = pd.read_csv(
            event_file,
            sep="\t",
        )

        print(f"Rows: {len(events)}")
        print(f"Columns: {events.columns.tolist()}")

        if events.empty:
            print("This event file contains no rows.")
            continue

        print("\nFull event table:")
        print(events.to_string(index=False))

        print("\nUnique values by column:")

        for column in events.columns:
            unique_values = events[column].dropna().unique()

            print(f"\n{column}:")
            print(unique_values[:20])

            if len(unique_values) > 20:
                print(
                    f"... plus {len(unique_values) - 20} more values"
                )


if __name__ == "__main__":
    main()