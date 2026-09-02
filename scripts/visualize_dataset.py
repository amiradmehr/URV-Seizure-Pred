r"""Render an exploratory picture of the processed SeizeIT2 dataset.

The atlas is a set of PNGs in ``outputs/dataset_figures/`` plus a self-contained
``dataset_report.html`` that embeds them with a reading for each.  Most of the
figures show real EEG with the labelling geometry drawn on top: whole
recordings, seizures at five magnifications, the exact window one decision
reads, and the artefacts that dominate it.

Each figure lives in its own module in ``scripts/dataset_atlas/`` named
``fNN_slug.py`` and exposing ``NUMBER``, ``SLUG``, ``TITLE``, ``QUESTION``,
``READS``, ``TAKE`` and ``render(atlas)``.  Adding a figure means adding one
file; this script discovers them, runs them in order, and builds the report.

Units, stated on every axis:

    count      a number of things (EDF files, decisions, seizures, patients)
    h/min/s    wall-clock duration of EEG
    Hz         frequency
    z          dimensionless amplitude in units of the global per-channel
               standard deviation.  The stored shards are z-scored, so nothing
               downstream is in volts; the conversion back to microvolts is
               printed beside every channel name.

Everything is read from the manifests plus the stored shards; nothing is
recomputed from the raw EDFs except the annotation files.

    python scripts/visualize_dataset.py
    python scripts/visualize_dataset.py --only 05 06 --refresh-exemplars
"""

from __future__ import annotations

import argparse
import importlib
import json
import pkgutil
import sys
import time
import traceback
from pathlib import Path
from types import ModuleType

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import dataset_atlas  # noqa: E402
from dataset_atlas.common import CONFIG, load_manifests, microvolts_per_z  # noqa: E402
from dataset_atlas.context import Atlas  # noqa: E402
from dataset_atlas.exemplars import select_exemplars  # noqa: E402
from dataset_atlas.report import build_html_report  # noqa: E402


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "dataset_figures",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        metavar="NUMBER",
        help="Render only these figure numbers, e.g. --only 05 06.",
    )
    parser.add_argument(
        "--recording-sample",
        type=int,
        default=90,
        help="Recordings to open for the amplitude and spectral statistics.",
    )
    parser.add_argument(
        "--refresh-exemplars",
        action="store_true",
        help="Re-select the plotted examples instead of reusing the cached choice.",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Skip rebuilding summary.json (it walks every shard header).",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip the HTML report (useful while iterating on one figure).",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def discover_figures() -> list[ModuleType]:
    """Import every ``fNN_*`` module in the atlas package, in number order."""
    modules: list[ModuleType] = []
    for info in pkgutil.iter_modules(dataset_atlas.__path__):
        if not info.name.startswith("f") or "_" not in info.name:
            continue
        if not info.name[1:3].isdigit():
            continue
        try:
            module = importlib.import_module(f"dataset_atlas.{info.name}")
        except Exception:  # a broken module must not take the whole atlas down
            print(f"  cannot import {info.name}:\n{traceback.format_exc()}", flush=True)
            continue
        required = ("render", "NUMBER", "SLUG", "TITLE", "READS", "TAKE")
        missing = [name for name in required if not hasattr(module, name)]
        if missing:
            print(f"  skipping {info.name}: missing {', '.join(missing)}", flush=True)
            continue
        modules.append(module)
    modules.sort(key=lambda m: m.NUMBER)
    return modules


def build_summary(atlas: Atlas) -> dict:
    """The numbers the report and summary.json quote, all measured here."""
    decisions = atlas.manifests.decisions
    seizures = atlas.manifests.seizures
    shards = atlas.manifests.shards

    hours = 0.0
    for path in shards["X_path"]:
        hours += np.load(path, mmap_mode="r").shape[1] / CONFIG.target_sfreq / 3600.0

    summary = {
        "units": {
            "count": "number of things (EDF files, decisions, seizures, patients)",
            "h/min/s": "wall-clock duration of EEG",
            "Hz": "frequency",
            "z": "dimensionless amplitude, 1 z = 1 global channel sigma",
            "microvolts_per_z": microvolts_per_z(),
        },
        "patients_with_data": int(decisions["subject"].nunique()),
        "edf_files": int(shards["recording_id"].nunique()),
        "eeg_hours": float(hours),
        "decisions": int(len(decisions)),
        "positive_decisions": int((decisions["label"] == 1).sum()),
        "prevalence": float((decisions["label"] == 1).mean()),
        "seizures_annotated": int(len(seizures)),
        "seizures_eligible": int(seizures["eligible"].sum()),
        "seizures_targeted": int(
            decisions.loc[decisions["label"] == 1, "target_seizure_id"].nunique()
        ),
        "by_split": {
            split: {
                "patients": int(group["subject"].nunique()),
                "decisions": int(len(group)),
                "positive": int((group["label"] == 1).sum()),
            }
            for split, group in decisions.groupby("split")
        },
        "figure_facts": atlas.facts,
    }
    return summary


def main() -> None:
    arguments = parse_arguments()
    CONFIG.validate()
    output = arguments.output_dir
    output.mkdir(parents=True, exist_ok=True)

    print("Loading manifests...", flush=True)
    manifests = load_manifests()
    print(
        f"  decisions {len(manifests.decisions):,} | seizures {len(manifests.seizures)} "
        f"| shards {len(manifests.shards)}",
        flush=True,
    )

    exemplars = select_exemplars(
        manifests,
        seed=arguments.seed,
        cache_path=output / "exemplars.json",
        refresh=arguments.refresh_exemplars,
    )

    atlas = Atlas(
        manifests=manifests,
        exemplars=exemplars,
        output=output,
        rng=np.random.default_rng(arguments.seed),
        recording_sample=arguments.recording_sample,
    )

    figures = discover_figures()
    if arguments.only:
        wanted = {number.zfill(2) for number in arguments.only}
        figures = [module for module in figures if module.NUMBER in wanted]
    print(f"Rendering {len(figures)} figures into {output}", flush=True)

    rendered: list[ModuleType] = []
    failures: list[tuple[str, str]] = []
    for module in figures:
        started = time.time()
        print(f"[{module.NUMBER}] {module.TITLE}", flush=True)
        try:
            path = module.render(atlas)
        except Exception:  # keep going; one broken figure must not kill the atlas
            failures.append((module.NUMBER, traceback.format_exc()))
            print(f"  FAILED\n{traceback.format_exc()}", flush=True)
            continue
        if path is None:
            print("  skipped (no data)", flush=True)
            continue
        rendered.append(module)
        print(f"  {time.time() - started:.1f} s", flush=True)

    summary = None
    if not arguments.no_summary:
        summary = build_summary(atlas)
        (output / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        print("  wrote summary.json", flush=True)
    elif (output / "summary.json").exists():
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))

    if not arguments.no_report and summary is not None:
        report = build_html_report(output, summary, discover_figures())
        print(
            f"  wrote {report.name}  "
            f"({report.stat().st_size / 1e6:.1f} MB, self-contained)",
            flush=True,
        )

    print(f"\n{len(rendered)} figures in {output}", flush=True)
    if failures:
        print(f"{len(failures)} figures FAILED: {[n for n, _ in failures]}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
