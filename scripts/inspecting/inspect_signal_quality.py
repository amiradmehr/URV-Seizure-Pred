from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "signal_inspection"

SUBJECT = "001"
SESSION = "01"
RUN = "03"

# From subject 001, run 03 events.tsv
SEIZURE_ONSET_SECONDS = 57_975.0

DISPLAY_SECONDS = 60.0
PRE_SEIZURE_OFFSET_SECONDS = 120.0


def find_edf() -> Path:
    pattern = (
        f"sub-{SUBJECT}_ses-{SESSION}_"
        f"task-szMonitoring_run-{RUN}_eeg.edf"
    )

    matches = list(DATA_ROOT.rglob(pattern))

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one EDF match, found {len(matches)}: "
            f"{matches}"
        )

    return matches[0]


def load_segment(
    raw: mne.io.BaseRaw,
    start_seconds: float,
    duration_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    sfreq = float(raw.info["sfreq"])

    start_sample = max(
        0,
        int(round(start_seconds * sfreq)),
    )
    stop_sample = min(
        raw.n_times,
        start_sample + int(round(duration_seconds * sfreq)),
    )

    data = raw.get_data(
        start=start_sample,
        stop=stop_sample,
    )

    times = (
        np.arange(data.shape[1]) / sfreq
        + start_sample / sfreq
    )

    return data, times


def print_segment_statistics(
    name: str,
    data_volts: np.ndarray,
) -> None:
    data_uv = data_volts * 1e6

    print(f"\n{name}")

    for channel_index in range(data_uv.shape[0]):
        channel = data_uv[channel_index]

        print(
            f"Channel {channel_index}: "
            f"min={np.min(channel):.2f} µV, "
            f"max={np.max(channel):.2f} µV, "
            f"mean={np.mean(channel):.2f} µV, "
            f"std={np.std(channel):.2f} µV, "
            f"peak-to-peak={np.ptp(channel):.2f} µV"
        )


def plot_segment(
    raw: mne.io.BaseRaw,
    data_volts: np.ndarray,
    times: np.ndarray,
    title: str,
    filename: str,
) -> None:
    data_uv = data_volts * 1e6

    figure, axes = plt.subplots(
        nrows=data_uv.shape[0],
        ncols=1,
        figsize=(14, 7),
        sharex=True,
    )

    if data_uv.shape[0] == 1:
        axes = [axes]

    for channel_index, axis in enumerate(axes):
        axis.plot(
            times,
            data_uv[channel_index],
            linewidth=0.6,
        )
        axis.set_ylabel(
            f"{raw.ch_names[channel_index]}\nµV"
        )
        axis.grid(alpha=0.25)

    axes[-1].set_xlabel("Recording time (seconds)")
    figure.suptitle(title)
    figure.tight_layout()

    output_path = OUTPUT_DIR / filename
    figure.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(figure)

    print(f"Saved plot: {output_path}")


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    edf_path = find_edf()

    print("=" * 80)
    print("SEIZEIT2 RAW SIGNAL INSPECTION")
    print("=" * 80)

    print(f"EDF: {edf_path}")

    raw = mne.io.read_raw_edf(
        edf_path,
        preload=False,
        verbose="ERROR",
    )

    print(f"Channels: {raw.ch_names}")
    print(f"Channel types: {raw.get_channel_types()}")
    print(f"Sampling rate: {raw.info['sfreq']} Hz")
    print(f"Duration: {raw.times[-1] / 3600:.2f} hours")

    segments = {
        "recording_start": 0.0,
        "recording_middle": raw.times[-1] / 2,
        "two_minutes_before_seizure": (
            SEIZURE_ONSET_SECONDS
            - PRE_SEIZURE_OFFSET_SECONDS
        ),
        "seizure_onset": SEIZURE_ONSET_SECONDS,
    }

    for name, start_seconds in segments.items():
        data, times = load_segment(
            raw,
            start_seconds=start_seconds,
            duration_seconds=DISPLAY_SECONDS,
        )

        print_segment_statistics(
            name=name,
            data_volts=data,
        )

        plot_segment(
            raw=raw,
            data_volts=data,
            times=times,
            title=f"{name}: sub-{SUBJECT}, run-{RUN}",
            filename=f"{SUBJECT}_{RUN}_{name}.png",
        )

    print("\nComputing power spectrum from the recording...")

    spectrum = raw.compute_psd(
        fmin=0.1,
        fmax=100.0,
        picks="eeg",
        n_fft=4096,
        verbose="ERROR",
    )

    figure = spectrum.plot(
        show=False,
    )

    psd_path = OUTPUT_DIR / f"{SUBJECT}_{RUN}_psd.png"
    figure.savefig(
        psd_path,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(figure)

    print(f"Saved PSD plot: {psd_path}")

    raw.close()


if __name__ == "__main__":
    main()