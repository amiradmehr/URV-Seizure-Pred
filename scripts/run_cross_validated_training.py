r"""Run a training configuration across patient folds and seeds, then aggregate.

A single run of this pipeline reports a maximum over a noisy per-epoch series:
validation average precision swings five- to six-fold inside one run, so two
configurations cannot be separated by comparing their best epochs.  This driver
trains the same configuration over every patient fold and every seed, then
reports mean and spread.

Folds are cut inside the *training* patients only.  The real validation and
test splits are never touched here, so they remain clean for final reporting.

Any option this script does not recognize is forwarded to
``train_eegnet_baseline.py`` unchanged, so a configuration is described once::

    .\.venv-old\Scripts\python.exe scripts\run_cross_validated_training.py ^
        --folds 5 --seeds 42 43 44 ^
        --name patient_event_balanced ^
        -- --sampling-strategy patient-event-balanced --negative-to-positive-ratio 10
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIRECTORY = PROJECT_ROOT / "src"
if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

from seizure_prediction.cross_validation import aggregate_runs  # noqa: E402


SEPARATOR = "=" * 92


def parse_arguments() -> tuple[argparse.Namespace, list[str]]:
    """Parse the driver's options and collect pass-through training options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=(42,),
        help="Model seeds. Three or more is what makes the spread meaningful.",
    )
    parser.add_argument(
        "--cv-seed",
        type=int,
        default=0,
        help="Seed for the fold partition, held fixed so folds are comparable.",
    )
    parser.add_argument(
        "--name",
        required=True,
        help="Short label for this configuration; names the output directory.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Interpreter used for each training run.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "cross_validation",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the runs that would be executed, then stop.",
    )
    known, passthrough = parser.parse_known_args()
    # Strip an optional "--" separator used to start the pass-through block.
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]
    return known, passthrough


def run_directory(root: Path, name: str, fold: int, seed: int) -> Path:
    """Return the output directory for one fold/seed run."""
    return root / name / f"fold{fold}_seed{seed}"


def main() -> None:
    """Train every fold/seed combination and summarize the results."""
    arguments, passthrough = parse_arguments()
    if arguments.folds < 2:
        raise ValueError("folds must be at least two.")

    trainer = PROJECT_ROOT / "scripts" / "train_eegnet_baseline.py"
    if not trainer.exists():
        raise FileNotFoundError(f"Trainer not found: {trainer}")

    runs = [
        (fold, seed)
        for seed in arguments.seeds
        for fold in range(arguments.folds)
    ]
    print(SEPARATOR)
    print(f"CROSS-VALIDATED TRAINING: {arguments.name}".center(92))
    print(SEPARATOR)
    print(
        f"\n{arguments.folds} folds x {len(arguments.seeds)} seed(s) "
        f"= {len(runs)} training runs"
    )
    print(f"Pass-through options: {' '.join(passthrough) or '(none)'}\n")

    if arguments.dry_run:
        for fold, seed in runs:
            print(f"  fold {fold} seed {seed} -> "
                  f"{run_directory(arguments.output_root, arguments.name, fold, seed)}")
        print("\nDry run; nothing was trained.")
        return

    records: list[dict[str, object]] = []
    for position, (fold, seed) in enumerate(runs, start=1):
        directory = run_directory(arguments.output_root, arguments.name, fold, seed)
        directory.mkdir(parents=True, exist_ok=True)
        command = [
            arguments.python,
            str(trainer),
            "--cv-folds", str(arguments.folds),
            "--cv-fold", str(fold),
            "--cv-seed", str(arguments.cv_seed),
            "--seed", str(seed),
            "--output-dir", str(directory),
            *passthrough,
        ]
        print(f"[{position}/{len(runs)}] fold {fold}, seed {seed}")
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            raise RuntimeError(
                f"Training failed for fold {fold} seed {seed} "
                f"(exit code {completed.returncode})."
            )

        metrics_path = directory / "metrics.json"
        if not metrics_path.exists():
            raise FileNotFoundError(f"Run produced no metrics: {metrics_path}")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        control = metrics.get("positional_control", {})
        records.append(
            {
                "fold": fold,
                "seed": seed,
                "best_epoch": metrics.get("best_epoch"),
                "average_precision": metrics["best_validation_average_precision"],
                "positional_average_precision": control.get(
                    "best_positional_average_precision"
                ),
                "prevalence": control.get("prevalence"),
                "beats_positional_baseline": control.get(
                    "model_beats_positional_baseline"
                ),
                "output_dir": str(directory),
            }
        )

    frame = pd.DataFrame(records)
    summary_directory = arguments.output_root / arguments.name
    frame.to_csv(summary_directory / "runs.csv", index=False)

    average_precision = aggregate_runs(
        frame["average_precision"].astype(float).tolist()
    )
    lift = aggregate_runs(
        (
            frame["average_precision"].astype(float)
            / frame["prevalence"].astype(float)
        ).tolist()
    )
    positional_lift = aggregate_runs(
        (
            frame["positional_average_precision"].astype(float)
            / frame["prevalence"].astype(float)
        ).tolist()
    )
    summary = {
        "name": arguments.name,
        "folds": arguments.folds,
        "seeds": list(arguments.seeds),
        "runs": len(frame),
        "passthrough_options": passthrough,
        "held_out_test_used": False,
        "average_precision": average_precision,
        "lift_over_prevalence": lift,
        "positional_baseline_lift": positional_lift,
        "runs_beating_positional_baseline": int(
            frame["beats_positional_baseline"].fillna(False).sum()
        ),
    }
    (summary_directory / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n" + SEPARATOR)
    print("RESULT".center(92))
    print(SEPARATOR)
    print(
        f"\naverage precision  {average_precision['mean']:.4f} "
        f"+/- {average_precision['std']:.4f} "
        f"(min {average_precision['minimum']:.4f}, "
        f"max {average_precision['maximum']:.4f}, n={average_precision['runs']})"
    )
    print(
        f"lift over prevalence  {lift['mean']:.2f}x "
        f"+/- {lift['std']:.2f}"
    )
    print(
        f"positional baseline   {positional_lift['mean']:.2f}x "
        f"+/- {positional_lift['std']:.2f}"
    )
    print(
        f"\nruns clearing the positional baseline: "
        f"{summary['runs_beating_positional_baseline']}/{len(frame)}"
    )
    if lift["mean"] <= positional_lift["mean"]:
        print(
            "\nThis configuration does not beat a scorer that never reads EEG. "
            "Treat it as a null result."
        )
    print(f"\nOutputs: {summary_directory}")


if __name__ == "__main__":
    main()
