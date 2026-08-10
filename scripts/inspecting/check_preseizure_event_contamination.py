from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

ELIGIBILITY_PATH = (
    PROCESSED_DIR / "preseizure_60min_eligibility.csv"
)

ALL_EVENTS_PATH = (
    PROCESSED_DIR / "all_events.csv"
)

OUTPUT_PATH = (
    PROCESSED_DIR
    / "preseizure_60min_event_contamination.csv"
)

SUMMARY_PATH = (
    PROCESSED_DIR
    / "preseizure_60min_event_contamination_summary.csv"
)

REQUIRED_HISTORY_SECONDS = 60 * 60


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

    missing_columns = (
        required_columns - set(dataframe.columns)
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


def event_overlaps_interval(
    event_onset: float,
    event_duration: float,
    interval_start: float,
    interval_end: float,
) -> bool:
    """
    Return True when an annotated event overlaps:

        [interval_start, interval_end)

    The interval includes its start but excludes the target
    seizure onset at interval_end.
    """
    event_end = event_onset + max(
        event_duration,
        0.0,
    )

    return (
        event_onset < interval_end
        and event_end > interval_start
    )


def format_event_description(
    event: pd.Series,
) -> str:
    onset = float(event["onset"])
    duration = float(event["duration"])
    event_type = str(event["eventType"])

    return (
        f"{event_type}"
        f"@{onset:.1f}s"
        f"+{duration:.1f}s"
    )


def main() -> None:
    eligibility = load_csv(
        ELIGIBILITY_PATH,
        required_columns={
            "seizure_id",
            "subject",
            "session",
            "run",
            "onset_seconds",
            "duration_seconds",
            "eligible_60min_by_duration",
        },
    )

    events = load_csv(
        ALL_EVENTS_PATH,
        required_columns={
            "subject",
            "session",
            "run",
            "onset",
            "duration",
            "eventType",
        },
    )

    eligibility = normalize_identifiers(
        eligibility
    )
    events = normalize_identifiers(events)

    eligibility["eligible_60min_by_duration"] = (
        eligibility[
            "eligible_60min_by_duration"
        ]
        .astype(str)
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

    if (
        eligibility[
            "eligible_60min_by_duration"
        ]
        .isna()
        .any()
    ):
        raise ValueError(
            "Could not interpret some "
            "eligible_60min_by_duration values."
        )

    eligible_seizures = eligibility[
        eligibility[
            "eligible_60min_by_duration"
        ]
    ].copy()

    events["eventType"] = (
        events["eventType"]
        .fillna("")
        .astype(str)
    )

    events["onset"] = pd.to_numeric(
        events["onset"],
        errors="coerce",
    )

    events["duration"] = pd.to_numeric(
        events["duration"],
        errors="coerce",
    ).fillna(0.0)

    invalid_event_times = (
        events["onset"].isna()
    )

    if invalid_event_times.any():
        raise ValueError(
            "Some event rows have invalid onset values: "
            f"{int(invalid_event_times.sum())}"
        )

    events_by_run = {
        key: group.sort_values(
            "onset"
        ).reset_index(drop=True)
        for key, group in events.groupby(
            ["subject", "session", "run"],
            sort=False,
        )
    }

    result_rows = []

    for _, seizure in eligible_seizures.iterrows():
        subject = seizure["subject"]
        session = seizure["session"]
        run = seizure["run"]

        seizure_onset = float(
            seizure["onset_seconds"]
        )

        window_start = (
            seizure_onset
            - REQUIRED_HISTORY_SECONDS
        )
        window_end = seizure_onset

        run_key = (
            subject,
            session,
            run,
        )

        run_events = events_by_run.get(
            run_key,
            pd.DataFrame(
                columns=events.columns
            ),
        )

        overlapping_events = []

        for _, event in run_events.iterrows():
            event_onset = float(
                event["onset"]
            )
            event_duration = float(
                event["duration"]
            )

            if event_overlaps_interval(
                event_onset=event_onset,
                event_duration=event_duration,
                interval_start=window_start,
                interval_end=window_end,
            ):
                overlapping_events.append(event)

        if overlapping_events:
            overlapping = pd.DataFrame(
                overlapping_events
            )
        else:
            overlapping = pd.DataFrame(
                columns=events.columns
            )

        if overlapping.empty:
            previous_seizures = overlapping
            impedance_events = overlapping
            other_events = overlapping

        else:
            event_types = (
                overlapping["eventType"]
                .fillna("")
                .astype(str)
            )

            previous_seizure_mask = (
                event_types.str.startswith("sz_")
            )

            impedance_mask = (
                event_types == "impd"
            )

            background_mask = (
                event_types == "bckg"
            )

            other_mask = ~(
                previous_seizure_mask
                | impedance_mask
                | background_mask
            )

            previous_seizures = overlapping[
                previous_seizure_mask
            ]

            impedance_events = overlapping[
                impedance_mask
            ]

            other_events = overlapping[
                other_mask
            ]

        previous_seizure_count = len(
            previous_seizures
        )

        impedance_event_count = len(
            impedance_events
        )

        other_event_count = len(
            other_events
        )

        has_previous_seizure = (
            previous_seizure_count > 0
        )

        has_impedance_event = (
            impedance_event_count > 0
        )

        has_other_event = (
            other_event_count > 0
        )

        event_clean_strict = not (
            has_previous_seizure
            or has_impedance_event
            or has_other_event
        )

        event_clean_no_prior_seizure = (
            not has_previous_seizure
        )

        result = seizure.to_dict()

        result.update(
            {
                "preseizure_start_seconds": (
                    window_start
                ),
                "preseizure_end_seconds": (
                    window_end
                ),
                "previous_seizure_count": (
                    previous_seizure_count
                ),
                "has_previous_seizure": (
                    has_previous_seizure
                ),
                "impedance_event_count": (
                    impedance_event_count
                ),
                "has_impedance_event": (
                    has_impedance_event
                ),
                "other_event_count": (
                    other_event_count
                ),
                "has_other_event": (
                    has_other_event
                ),
                "event_clean_no_prior_seizure": (
                    event_clean_no_prior_seizure
                ),
                "event_clean_strict": (
                    event_clean_strict
                ),
                "previous_seizure_details": (
                    "; ".join(
                        format_event_description(
                            event
                        )
                        for _, event
                        in previous_seizures.iterrows()
                    )
                ),
                "impedance_event_details": (
                    "; ".join(
                        format_event_description(
                            event
                        )
                        for _, event
                        in impedance_events.iterrows()
                    )
                ),
                "other_event_details": (
                    "; ".join(
                        format_event_description(
                            event
                        )
                        for _, event
                        in other_events.iterrows()
                    )
                ),
            }
        )

        result_rows.append(result)

    results = pd.DataFrame(result_rows)

    results = results.sort_values(
        by=[
            "subject",
            "session",
            "run",
            "onset_seconds",
        ]
    ).reset_index(drop=True)

    results.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    subject_summary = (
        results.groupby("subject")
        .agg(
            duration_eligible_seizures=(
                "seizure_id",
                "size",
            ),
            seizures_with_prior_seizure=(
                "has_previous_seizure",
                "sum",
            ),
            seizures_with_impedance_event=(
                "has_impedance_event",
                "sum",
            ),
            seizures_with_other_event=(
                "has_other_event",
                "sum",
            ),
            event_clean_no_prior_seizure=(
                "event_clean_no_prior_seizure",
                "sum",
            ),
            event_clean_strict=(
                "event_clean_strict",
                "sum",
            ),
        )
        .reset_index()
    )

    subject_summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    total = len(results)

    with_previous_seizure = int(
        results[
            "has_previous_seizure"
        ].sum()
    )

    with_impedance = int(
        results[
            "has_impedance_event"
        ].sum()
    )

    with_other = int(
        results[
            "has_other_event"
        ].sum()
    )

    clean_no_prior_seizure = int(
        results[
            "event_clean_no_prior_seizure"
        ].sum()
    )

    strictly_clean = int(
        results[
            "event_clean_strict"
        ].sum()
    )

    print("=" * 80)
    print("60-MINUTE PRE-SEIZURE EVENT CONTAMINATION")
    print("=" * 80)

    print(
        f"Duration-eligible seizures checked: {total}"
    )

    print("\nContamination counts:")
    print(
        "Preceding hour contains an earlier seizure: "
        f"{with_previous_seizure}"
    )
    print(
        "Preceding hour contains an impedance event: "
        f"{with_impedance}"
    )
    print(
        "Preceding hour contains another event type: "
        f"{with_other}"
    )

    print("\nClean eligibility:")
    print(
        "No earlier seizure in preceding hour: "
        f"{clean_no_prior_seizure}"
    )
    print(
        "No earlier seizure, impedance event, or "
        f"other annotation: {strictly_clean}"
    )

    if total > 0:
        print(
            "No-prior-seizure retention: "
            f"{100 * clean_no_prior_seizure / total:.2f}%"
        )
        print(
            "Strict event-clean retention: "
            f"{100 * strictly_clean / total:.2f}%"
        )

    subjects_with_strictly_clean_seizure = (
        subject_summary.loc[
            subject_summary[
                "event_clean_strict"
            ] > 0,
            "subject",
        ].nunique()
    )

    subjects_with_no_prior_clean_seizure = (
        subject_summary.loc[
            subject_summary[
                "event_clean_no_prior_seizure"
            ] > 0,
            "subject",
        ].nunique()
    )

    print("\nSubjects:")
    print(
        "Subjects with at least one seizure having "
        "no prior seizure in the hour: "
        f"{subjects_with_no_prior_clean_seizure}"
    )
    print(
        "Subjects with at least one strictly "
        "event-clean seizure: "
        f"{subjects_with_strictly_clean_seizure}"
    )

    contaminated = results[
        ~results["event_clean_strict"]
    ]

    if not contaminated.empty:
        print(
            "\nFirst 25 contaminated seizures:"
        )

        columns = [
            "seizure_id",
            "subject",
            "session",
            "run",
            "onset_seconds",
            "previous_seizure_count",
            "impedance_event_count",
            "other_event_count",
            "previous_seizure_details",
            "impedance_event_details",
            "other_event_details",
        ]

        print(
            contaminated[
                columns
            ]
            .head(25)
            .to_string(index=False)
        )

    print("\nTop subjects by strictly clean seizures:")
    print(
        subject_summary.sort_values(
            by=[
                "event_clean_strict",
                "duration_eligible_seizures",
                "subject",
            ],
            ascending=[
                False,
                False,
                True,
            ],
        )
        .head(25)
        .to_string(index=False)
    )

    print("\nSaved:")
    print(OUTPUT_PATH)
    print(SUMMARY_PATH)


if __name__ == "__main__":
    main()