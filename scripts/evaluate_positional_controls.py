r"""Score every saved model against the within-recording positional confound.

Positive decisions in this task sit at the end of their recording's valid
decision sequence, because the 60-minute postictal exclusion truncates a
recording just after its seizure.  A one-line scorer that never reads EEG --
"how close is this decision to the end of its recording" -- therefore reaches
roughly a 5x lift over prevalence on the validation split, which is the same
range as the learned models.

This command makes that comparison routine.  For each saved prediction table it
reports the model beside the confound on the full split, and again on a
position-matched subset where negatives carry the positives' position
distribution so positional information is worth nothing.  Lift retained on the
matched subset is attributable to EEG; lift that disappears was positional.

It also prints the cost curve for the construction-side repair, so the
threshold for ``filter_by_following_valid_decisions`` can be chosen against real
numbers rather than guessed.

PowerShell example:

    .\.venv-old\Scripts\python.exe scripts\evaluate_positional_controls.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIRECTORY = PROJECT_ROOT / "src"
if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

from seizure_prediction.config import CONFIG  # noqa: E402
from seizure_prediction.positional_controls import (  # noqa: E402
    DEFAULT_POSITION_BINS,
    evaluate_against_positional_controls,
    positional_baseline_report,
    positional_cost_curve,
)


SEPARATOR = "=" * 96

# Saved prediction tables, discovered by convention when none are supplied.
DEFAULT_PREDICTION_GLOBS = (
    "outputs/evaluation/*/validation_predictions.csv",
    "outputs/models/*/validation_predictions.csv",
)


def parse_arguments() -> argparse.Namespace:
    """Parse the comparison's options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        nargs="*",
        default=(),
        help=(
            "Prediction CSVs to score. Defaults to every "
            "validation_predictions.csv under outputs/."
        ),
    )
    parser.add_argument(
        "--score-column",
        default="probability",
        help="Column holding each decision's model score.",
    )
    parser.add_argument(
        "--negatives-per-positive",
        type=float,
        default=20.0,
        help="Ratio held uniformly across position bins in the matched subset.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=DEFAULT_POSITION_BINS,
        help=(
            "Position bins used for matching. Coarser bins under-correct: the "
            "residual positional lift is 1.90 at 10 bins and 1.08 at 80."
        ),
    )
    parser.add_argument(
        "--cost-curve-thresholds",
        nargs="*",
        type=int,
        default=(0, 5, 10, 15, 20, 30, 45, 60),
        help="Minimum following-valid-decision counts to price.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "analysis" / "positional_controls",
    )
    return parser.parse_args()


def discover_prediction_tables(explicit: tuple[str, ...]) -> list[Path]:
    """Return the prediction tables to score."""
    if explicit:
        paths = [Path(value) for value in explicit]
        missing = [str(path) for path in paths if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Prediction tables not found: {missing}")
        return paths

    discovered: list[Path] = []
    for pattern in DEFAULT_PREDICTION_GLOBS:
        discovered.extend(sorted(PROJECT_ROOT.glob(pattern)))
    if not discovered:
        raise FileNotFoundError(
            "No validation_predictions.csv found under outputs/. Run an "
            "evaluation script first, or pass --predictions explicitly."
        )
    return discovered


def model_name_for(path: Path) -> str:
    """Name a run by its output directory."""
    return path.parent.name


def load_validation_decisions() -> pd.DataFrame:
    """Load the validation decisions straight from the manifest."""
    manifest_path = CONFIG.manifests_dir / "decision_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Decision manifest not found: {manifest_path}")
    decisions = pd.read_csv(manifest_path, dtype={"subject": str})
    validation = decisions[decisions["split"] == "validation"].copy()
    if validation.empty:
        raise ValueError("The manifest contains no validation decisions.")
    return validation


def save_comparison_figure(frame: pd.DataFrame, output_path: Path) -> None:
    """Plot each model's lift before and after position matching."""
    if frame.empty:
        return
    ordered = frame.sort_values("matched_model_lift")
    positions = np.arange(len(ordered))
    height = 0.36
    figure, axes = plt.subplots(figsize=(10, max(3.5, 0.9 * len(ordered))))

    axes.barh(
        positions + height / 2,
        ordered["full_model_lift"],
        height=height,
        color="#b7c9de",
        label="raw split",
    )
    axes.barh(
        positions - height / 2,
        ordered["matched_model_lift"],
        height=height,
        color="#2b7bba",
        label="position-matched",
    )
    axes.axvline(
        ordered["full_positional_lift"].iloc[0],
        color="#c44e52",
        linestyle="--",
        linewidth=1.2,
        label="positional baseline, raw split",
    )
    axes.axvline(
        ordered["matched_positional_lift"].iloc[0],
        color="#c44e52",
        linestyle=":",
        linewidth=1.2,
        label="positional baseline, matched",
    )
    axes.set_yticks(positions)
    axes.set_yticklabels(ordered["model"], fontsize=8)
    axes.set_xlabel("average precision / prevalence")
    axes.set_title("Model lift before and after neutralizing the positional confound")
    axes.legend(loc="lower right", fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def main() -> None:
    """Score every discovered model against the positional controls."""
    arguments = parse_arguments()
    CONFIG.validate()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)

    print(SEPARATOR)
    print("POSITIONAL CONTROL EVALUATION".center(96))
    print(SEPARATOR)

    validation = load_validation_decisions()
    baseline = positional_baseline_report(validation)
    print(
        f"\nValidation decisions {len(validation):,}  "
        f"positives {int(validation['label'].sum())}  "
        f"prevalence {baseline['prevalence']:.5f}"
    )
    print("\nTrivial scorers that never read EEG:")
    for name in ("position_within_recording", "closeness_to_recording_end"):
        print(
            f"  {name:32s} AP={baseline[f'{name}_average_precision']:.4f}  "
            f"lift={baseline[f'{name}_lift']:.2f}x"
        )

    tables = discover_prediction_tables(tuple(arguments.predictions))
    print(f"\nScoring {len(tables)} saved prediction table(s), "
          f"matching at {arguments.bins} bins, "
          f"{arguments.negatives_per_positive:g}:1.\n")

    rows: list[dict[str, object]] = []
    for path in tables:
        predictions = pd.read_csv(path, dtype={"subject": str})
        if arguments.score_column not in predictions.columns:
            print(f"  skipped {model_name_for(path)}: no "
                  f"{arguments.score_column!r} column")
            continue
        report = evaluate_against_positional_controls(
            predictions,
            score_column=arguments.score_column,
            negatives_per_positive=arguments.negatives_per_positive,
            bins=arguments.bins,
            seed=arguments.seed,
        )
        full, matched = report["full"], report["position_matched"]
        rows.append(
            {
                "model": model_name_for(path),
                "predictions_path": str(path),
                "full_decisions": full["decisions"],
                "full_prevalence": full["prevalence"],
                "full_model_average_precision": full["model_average_precision"],
                "full_model_lift": full["model_lift"],
                "full_positional_average_precision": full[
                    "best_positional_average_precision"
                ],
                "full_positional_lift": full["best_positional_lift"],
                "full_beats_positional": full["beats_positional_baseline"],
                "matched_decisions": matched["matched_decisions"],
                "matched_positives": matched["matched_positives"],
                "matched_positive_retention": matched["positive_retention"],
                "matched_prevalence": matched["matched_prevalence"],
                "matched_model_average_precision": matched[
                    "model_average_precision"
                ],
                "matched_model_lift": matched["model_lift"],
                "matched_positional_lift": matched["best_positional_lift"],
                "matched_beats_positional": matched["beats_positional_baseline"],
                "margin_over_residual": (
                    matched["model_lift"] / matched["best_positional_lift"]
                ),
            }
        )

    if not rows:
        raise RuntimeError("No prediction table could be scored.")
    frame = pd.DataFrame(rows).sort_values(
        "matched_model_lift",
        ascending=False,
    )
    frame.to_csv(arguments.output_dir / "model_comparison.csv", index=False)
    save_comparison_figure(frame, arguments.output_dir / "model_comparison.png")

    header = (
        f"{'model':44s} {'raw lift':>9s} {'beats?':>7s} | "
        f"{'matched':>8s} {'resid':>6s} {'margin':>7s}"
    )
    print(header)
    print("-" * len(header))
    for row in frame.itertuples(index=False):
        print(
            f"{row.model[:44]:44s} {row.full_model_lift:9.2f} "
            f"{str(row.full_beats_positional):>7s} | "
            f"{row.matched_model_lift:8.2f} {row.matched_positional_lift:6.2f} "
            f"{row.margin_over_residual:6.2f}x"
        )

    print("\n" + SEPARATOR)
    print("CONSTRUCTION REPAIR COST CURVE".center(96))
    print(SEPARATOR)
    print(
        "\nRequiring every decision to be followed by N further valid "
        "decisions removes\nthe terminal region where positives concentrate. "
        "Pick the smallest N that drives\nthe positional lift near 1.0 while "
        "retaining enough seizures to train on.\n"
    )
    curve = positional_cost_curve(
        validation,
        thresholds_following=tuple(arguments.cost_curve_thresholds),
    )
    curve.to_csv(arguments.output_dir / "cost_curve.csv", index=False)
    print(curve.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    summary = {
        "split": "validation",
        "held_out_test_used": False,
        "prevalence": baseline["prevalence"],
        "positional_baseline": {
            key: value for key, value in baseline.items() if key != "prevalence"
        },
        "matching": {
            "bins": arguments.bins,
            "negatives_per_positive": arguments.negatives_per_positive,
            "seed": arguments.seed,
        },
        "models_scored": len(frame),
        "models_beating_positional_baseline_raw": int(
            frame["full_beats_positional"].sum()
        ),
        "models_beating_positional_baseline_matched": int(
            frame["matched_beats_positional"].sum()
        ),
        "best_matched_model": str(frame.iloc[0]["model"]),
        "best_matched_lift": float(frame.iloc[0]["matched_model_lift"]),
        "residual_positional_lift_after_matching": float(
            frame.iloc[0]["matched_positional_lift"]
        ),
    }
    (arguments.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(f"\nOutputs: {arguments.output_dir}")


if __name__ == "__main__":
    main()
