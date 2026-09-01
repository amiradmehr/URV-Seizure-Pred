r"""Assemble every finished cross-validation run into one comparable table.

Reads each outputs/cv/<tag>/cv_metrics.json and writes:

    outputs/sweep/results.csv          one row per configuration
    outputs/sweep/results.md           the same, ranked, as a table
    outputs/sweep/sweep_overview.png   effect of each axis

Safe to run while the sweep is still going; it reports whatever has landed.
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
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK = "#52514e"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv-dir", type=Path, default=PROJECT_ROOT / "outputs" / "cv")
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_ROOT / "outputs" / "sweep")
    return parser.parse_args()


def style(axis: plt.Axes) -> None:
    axis.grid(alpha=0.22, linewidth=0.7)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(INK)
        axis.spines[side].set_linewidth(0.8)


def collect(cv_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(cv_dir.glob("*/cv_metrics.json")):
        try:
            metrics = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        budgets = {
            b["false_alarm_budget_per_hour"]: b for b in metrics.get("alarm_budgets", [])
        }
        one = budgets.get(1.0, {})
        row = {
            "tag": metrics.get("tag", path.parent.name),
            "architecture": metrics.get("architecture"),
            "parameters": metrics.get("model_parameters"),
            "history_min": metrics.get("history_minutes"),
            "sop_min": metrics.get("sop_minutes"),
            "chunks": metrics.get("chunks_used"),
            "seizures": metrics.get("target_seizures"),
            "decisions": metrics.get("decisions"),
            "prevalence": metrics.get("prevalence"),
            "ap": metrics.get("average_precision"),
            "ap_over_chance": metrics.get("ap_over_chance"),
            "auc": metrics.get("roc_auc"),
            "brier": metrics.get("brier_score"),
            "sens_at_1ph": one.get("seizure_sensitivity"),
            "sens_ci_low": one.get("ci_low"),
            "sens_ci_high": one.get("ci_high"),
            "detected_at_1ph": one.get("seizures_detected"),
        }
        folds = metrics.get("per_fold", [])
        if folds:
            row["auc_fold_std"] = float(np.std([f["roc_auc"] for f in folds]))
        for entry in metrics.get("vigilance_at_1_per_hour", []):
            row[f"sens_{entry['vigilance']}"] = entry["sensitivity"]
            row[f"n_{entry['vigilance']}"] = entry["seizures"]
        rows.append(row)
    if not rows:
        raise SystemExit(f"No finished runs found under {cv_dir}")
    return pd.DataFrame(rows).sort_values("ap_over_chance", ascending=False)


def plot_overview(results: pd.DataFrame, path: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(14, 9))

    axis = axes[0, 0]
    axis.axhline(1.0, color=ORANGE, linestyle="--", linewidth=1.5)
    axis.annotate("chance", xy=(0.02, 1.0), xycoords=("axes fraction", "data"),
                  xytext=(0, 4), textcoords="offset points", fontsize=9, color=ORANGE)
    axis.scatter(results["parameters"], results["ap_over_chance"],
                 s=34, color=BLUE, alpha=0.85)
    axis.set(title="Capacity buys nothing if the ratio stays at 1",
             xlabel="Model parameters (count)",
             ylabel="AP / chance (dimensionless)", xscale="log")
    style(axis)

    for index, (column, label) in enumerate(
        (("history_min", "History length (min)"), ("sop_min", "SOP (min)"))
    ):
        axis = axes[0, 1] if index == 0 else axes[1, 0]
        groups = results.groupby(column)["ap_over_chance"]
        positions = np.arange(len(groups))
        axis.bar(positions, groups.mean(), yerr=groups.std().fillna(0.0),
                 color=BLUE, width=0.55, capsize=4)
        axis.axhline(1.0, color=ORANGE, linestyle="--", linewidth=1.5)
        axis.set(title=f"Effect of {label.lower()}", xlabel=label,
                 ylabel="AP / chance (mean, sd over configs)",
                 xticks=positions,
                 xticklabels=[f"{v:g}" for v in groups.groups])
        style(axis)

    axis = axes[1, 1]
    order = results.groupby("architecture")["ap_over_chance"].mean().sort_values()
    axis.barh(np.arange(len(order)), order.values, color=AQUA, height=0.6)
    axis.axvline(1.0, color=ORANGE, linestyle="--", linewidth=1.5)
    axis.set(title="Effect of aggregation", xlabel="AP / chance (mean over configs)",
             yticks=np.arange(len(order)), yticklabels=list(order.index))
    style(axis)

    figure.suptitle(
        f"Exploration sweep — {len(results)} configurations, "
        f"patient-wise 5-fold CV on {int(results['seizures'].max())} seizures",
        fontsize=14,
    )
    figure.tight_layout()
    figure.savefig(path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    arguments = parse_arguments()
    output = arguments.output_dir
    output.mkdir(parents=True, exist_ok=True)

    results = collect(arguments.cv_dir)
    results.to_csv(output / "results.csv", index=False)

    columns = ["tag", "architecture", "parameters", "history_min", "sop_min",
               "seizures", "prevalence", "ap", "ap_over_chance", "auc", "brier",
               "sens_at_1ph", "sens_ci_low", "sens_ci_high"]
    table = results[[c for c in columns if c in results.columns]]
    (output / "results.md").write_text(
        f"# Exploration sweep — {len(results)} configurations\n\n"
        "Patient-wise 5-fold cross-validation, test patients excluded.\n"
        "Ranked by average precision relative to chance.\n\n"
        + table.to_markdown(index=False, floatfmt=".4f"),
        encoding="utf-8",
    )

    plot_overview(results, output / "sweep_overview.png")

    print(f"{len(results)} runs collected")
    print(table.head(12).to_string(index=False))
    print(f"\nBest AP/chance : {results['ap_over_chance'].max():.3f}")
    print(f"Best AUC       : {results['auc'].max():.4f}")
    print(f"Runs above chance (AP ratio > 1.0): "
          f"{int((results['ap_over_chance'] > 1.0).sum())}/{len(results)}")
    print(f"\nArtefacts: {output}")


if __name__ == "__main__":
    main()
