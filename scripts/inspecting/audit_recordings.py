from pathlib import Path

import mne
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

RUN_INVENTORY_PATH = PROCESSED_DIR / "run_inventory.csv"
OUTPUT_PATH = PROCESSED_DIR / "recording_audit.csv"


def inspect_signal_sections(
    raw: mne.io.BaseRaw,
    section_seconds: float = 2.0,
) -> tuple[bool, float, float]:
    """
    Read short sections from the beginning, middle, and end.

    This checks that numerical data can be read without loading the
    entire recording into memory.
    """
    sfreq = float(raw.info["sfreq"])
    section_samples = max(1, int(section_seconds * sfreq))

    max_start = max(0, raw.n_times - section_samples)

    starts = [
        0,
        max_start // 2,
        max_start,
    ]

    sections = []

    for start in starts:
        stop = min(start + section_samples, raw.n_times)

        data = raw.get_data(
            start=start,
            stop=stop,
        )

        sections.append(data)

    combined = np.concatenate(sections, axis=1)

    all_finite = bool(np.isfinite(combined).all())
    minimum = float(np.nanmin(combined))
    maximum = float(np.nanmax(combined))

    return all_finite, minimum, maximum


def main() -> None:
    if not RUN_INVENTORY_PATH.exists():
        raise FileNotFoundError(
            f"Run inventory not found: {RUN_INVENTORY_PATH}"
        )

    runs = pd.read_csv(
        RUN_INVENTORY_PATH,
        dtype={
            "subject": str,
            "session": str,
            "run": str,
        },
    )

    audit_rows = []

    print("=" * 80)
    print("SEIZEIT2 RECORDING AUDIT")
    print("=" * 80)
    print(f"Recordings to inspect: {len(runs)}")

    for index, row in runs.iterrows():
        edf_path = Path(row["edf_file"])

        result = {
            "subject": row["subject"],
            "session": row["session"],
            "run": row["run"],
            "edf_file": str(edf_path),
            "opened_successfully": False,
            "error": "",
        }

        try:
            raw = mne.io.read_raw_edf(
                edf_path,
                preload=False,
                verbose="ERROR",
            )

            sfreq = float(raw.info["sfreq"])
            eeg_duration_seconds = (
                raw.n_times / sfreq
            )

            (
                all_finite,
                sampled_minimum,
                sampled_maximum,
            ) = inspect_signal_sections(raw)

            result.update(
                {
                    "opened_successfully": True,
                    "n_channels": len(raw.ch_names),
                    "channel_names": "|".join(raw.ch_names),
                    "channel_types": "|".join(
                        raw.get_channel_types()
                    ),
                    "sampling_frequency": sfreq,
                    "n_samples": int(raw.n_times),
                    "edf_duration_seconds": eeg_duration_seconds,
                    "event_duration_seconds": float(
                        row["recording_duration_seconds"]
                    ),
                    "duration_difference_seconds": (
                        eeg_duration_seconds
                        - float(
                            row["recording_duration_seconds"]
                        )
                    ),
                    "sampled_values_finite": all_finite,
                    "sampled_minimum_volts": sampled_minimum,
                    "sampled_maximum_volts": sampled_maximum,
                }
            )

            raw.close()

        except Exception as error:
            result["error"] = (
                f"{type(error).__name__}: {error}"
            )

        audit_rows.append(result)

        completed = index + 1

        if completed % 100 == 0 or completed == len(runs):
            print(
                f"Processed {completed}/{len(runs)} recordings"
            )

    audit = pd.DataFrame(audit_rows)

    audit.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    successful = audit[
        audit["opened_successfully"] == True
    ]

    print("\n" + "=" * 80)
    print("AUDIT SUMMARY")
    print("=" * 80)

    print(
        "Successfully opened: "
        f"{int(audit['opened_successfully'].sum())}"
    )
    print(
        "Failed to open: "
        f"{int((~audit['opened_successfully']).sum())}"
    )

    if not successful.empty:
        print("\nSampling frequencies:")
        print(
            successful[
                "sampling_frequency"
            ].value_counts().sort_index()
        )

        print("\nNumber of channels:")
        print(
            successful[
                "n_channels"
            ].value_counts().sort_index()
        )

        print("\nChannel configurations:")
        print(
            successful[
                "channel_names"
            ].value_counts().head(20)
        )

        print("\nChannel type configurations:")
        print(
            successful[
                "channel_types"
            ].value_counts()
        )

        print("\nRecordings with nonfinite sampled data:")
        print(
            int(
                (
                    successful[
                        "sampled_values_finite"
                    ] == False
                ).sum()
            )
        )

        duration_difference = successful[
            "duration_difference_seconds"
        ].abs()

        print("\nDuration agreement:")
        print(
            "Maximum absolute difference: "
            f"{duration_difference.max():.4f} seconds"
        )
        print(
            "Runs differing by more than 1 second: "
            f"{int((duration_difference > 1).sum())}"
        )

    failures = audit[
        audit["opened_successfully"] == False
    ]

    if not failures.empty:
        print("\nFailed recordings:")
        print(
            failures[
                [
                    "subject",
                    "session",
                    "run",
                    "error",
                ]
            ].to_string(index=False)
        )

    print(f"\nSaved audit to:\n{OUTPUT_PATH}")


if __name__ == "__main__":
    main()