r"""Collect every cross-validation result into one table and one comparison chart.

Reads outputs/cv/*/cv_metrics.json and writes, into outputs/sweep/:

    sweep_results.csv     one row per configuration, sorted by the headline metric
    sweep_comparison.png  the grid as charts
    sweep_summary.json    the best rows plus the chance reference

    python scripts/collect_sweep.py
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

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK = "#52514e"


def style(axis: plt.Axes) -> None:
    axis.grid(alpha=0.22, linewidth=0.7)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(INK)
        axis.spines[side].set_linewidth(0.8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv-dir", type=Path, default=PROJECT_ROOT / "outputs" / "cv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "sweep")
    arguments = parser.parse_args()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for path in sorted(arguments.cv_dir.glob("*/cv_metrics.json")):
        m = json.loads(path.read_text(encoding="utf-8"))
        budgets = {b["false_alarm_budget_per_hour"]: b for b in m.get("alarm_budgets", [])}
        one = budgets.get(1.0, {})
        row = {
            "tag": m.get("tag"),
            "architecture": m.get("architecture"),
            "parameters": m.get("model_parameters"),
            "history_minutes": m.get("history_minutes"),
            "sop_minutes": m.get("sop_minutes"),
            "chunks_used": m.get("chunks_used"),
            "patients": m.get("patients"),
            "seizures": m.get("target_seizures"),
            "decisions": m.get("decisions"),
            "prevalence": m.get("prevalence"),
            "average_precision": m.get("average_precision"),
            "ap_over_chance": m.get("ap_over_chance"),
            "roc_auc": m.get("roc_auc"),
            "brier_score": m.get("brier_score"),
            "sz_sens_at_1ph": one.get("seizure_sensitivity"),
            "sz_ci_low": one.get("ci_low"),
            "sz_ci_high": one.get("ci_high"),
            "sz_detected": one.get("seizures_detected"),
            "sz_total": one.get("seizures_total"),
        }
        for entry in m.get("vigilance_at_1_per_hour", []):
            row[f"sens_{entry['vigilance']}"] = entry["sensitivity"]
            row[f"n_{entry['vigilance']}"] = entry["seizures"]
        rows.append(row)

    if not rows:
        raise SystemExit(f"No cv_metrics.json found under {arguments.cv_dir}")

    frame = pd.DataFrame(rows).sort_values("ap_over_chance", ascending=False)
    frame.to_csv(arguments.output_dir / "sweep_results.csv", index=False)
    print(f"{len(frame)} configurations collected")

    complete = frame.dropna(subset=["ap_over_chance", "roc_auc"])

    figure, axes = plt.subplots(2, 2, figsize=(15, 9))

    # 1. AP over chance, ranked
    axis = axes[0, 0]
    ordered = complete.sort_values("ap_over_chance")
    colors = [ORANGE if v > 1.0 else BLUE for v in ordered["ap_over_chance"]]
    axis.barh(np.arange(len(ordered)), ordered["ap_over_chance"], color=colors, height=0.7)
    axis.axvline(1.0, color=INK, linestyle="--", linewidth=1.5)
    axis.annotate("chance", xy=(1.0, 0.02), xycoords=("data", "axes fraction"),
                  xytext=(4, 0), textcoords="offset points", fontsize=9, color=INK)
    axis.set(title="Average precision relative to chance",
             xlabel="AP / prevalence  (1.0 = chance)",
             yticks=np.arange(len(ordered)),
             yticklabels=[t[:38] for t in ordered["tag"]])
    axis.tick_params(axis="y", labelsize=6.5)
    style(axis)

    # 2. AUC by architecture
    axis = axes[0, 1]
    architectures = sorted(complete["architecture"].unique())
    for index, architecture in enumerate(architectures):
        values = complete.loc[complete["architecture"] == architecture, "roc_auc"]
        axis.scatter(np.full(len(values), index) + np.random.uniform(-.12, .12, len(values)),
                     values, s=34, color=[BLUE, ORANGE, AQUA, INK][index % 4], alpha=0.85)
    axis.axhline(0.5, color=INK, linestyle="--", linewidth=1.5)
    axis.set(title="ROC AUC by aggregation strategy", ylabel="ROC AUC (pooled out-of-fold)",
             xticks=np.arange(len(architectures)),
             xticklabels=[a.replace("spectral-", "") for a in architectures])
    style(axis)

    # 3. effect of history length and SOP
    axis = axes[1, 0]
    for sop, color, marker in ((10.0, BLUE, "o"), (30.0, ORANGE, "s")):
        subset = complete[complete["sop_minutes"] == sop]
        grouped = subset.groupby("history_minutes")["roc_auc"]
        if len(grouped) == 0:
            continue
        axis.errorbar(grouped.mean().index, grouped.mean().values,
                      yerr=grouped.std().fillna(0).values, color=color, marker=marker,
                      markersize=8, linewidth=2, capsize=4, label=f"SOP {sop:g} min")
    axis.axhline(0.5, color=INK, linestyle="--", linewidth=1.5)
    axis.set(title="Context length and prediction window",
             xlabel="History used (min)", ylabel="ROC AUC (mean ± sd across configs)")
    axis.legend(frameon=False, fontsize=9)
    style(axis)

    # 4. capacity
    axis = axes[1, 1]
    axis.scatter(complete["parameters"], complete["roc_auc"], s=40, color=BLUE, alpha=0.8)
    axis.axhline(0.5, color=INK, linestyle="--", linewidth=1.5)
    axis.set(title="Model capacity", xlabel="Trainable parameters (count)",
             ylabel="ROC AUC", xscale="log")
    style(axis)

    figure.suptitle(
        f"Exploration sweep — {len(complete)} configurations, "
        f"patient-wise 5-fold CV, {int(complete['seizures'].max())} seizures",
        fontsize=14,
    )
    figure.tight_layout()
    figure.savefig(arguments.output_dir / "sweep_comparison.png", dpi=140,
                   bbox_inches="tight", facecolor="white")
    plt.close(figure)

    best = frame.iloc[0]
    summary = {
        "configurations": int(len(frame)),
        "best_by_ap_over_chance": {
            k: (None if pd.isna(v) else v)
            for k, v in best.to_dict().items()
        },
        "any_above_chance_ap": bool((frame["ap_over_chance"] > 1.0).any()),
        "max_roc_auc": float(frame["roc_auc"].max()),
        "max_seizure_sensitivity_at_1ph": float(frame["sz_sens_at_1ph"].max()),
        "note": "test patients (113-125) excluded from every configuration",
    }
    (arguments.output_dir / "sweep_summary.json").write_text(
        json.dumps(summary, indent=2, default=float), encoding="utf-8"
    )

    show = ["tag", "parameters", "history_minutes", "sop_minutes",
            "ap_over_chance", "roc_auc", "sz_sens_at_1ph"]
    print(frame[show].head(12).to_string(index=False))
    print(f"\nWrote {arguments.output_dir}")


if __name__ == "__main__":
    main()
