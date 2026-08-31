r"""Explain the model's input, output and metrics — as math and as charts.

Every chart is computed from a real prediction file, so the numbers on the page
are the numbers the pipeline actually produced. Writes a self-contained HTML
document with the figures embedded as data URIs:

    outputs/explainers/io_and_metrics.html

    python scripts/explain_io_metrics.py --predictions outputs/evaluation/validation/predictions.csv
"""

from __future__ import annotations

import argparse
import base64
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIRECTORY = PROJECT_ROOT / "src"
if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

from seizure_prediction.config import CONFIG  # noqa: E402

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK = "#52514e"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "evaluation" / "validation" / "predictions.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "explainers",
    )
    parser.add_argument("--alarm-budget", type=float, default=1.0)
    return parser.parse_args()


def log(message: str) -> None:
    print(message, flush=True)


def style(axis: plt.Axes) -> None:
    axis.grid(alpha=0.22, linewidth=0.7)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(INK)
        axis.spines[side].set_linewidth(0.8)


def save(figure: plt.Figure, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    log(f"  wrote {path.name}")


def threshold_for_budget(scores: np.ndarray, labels: np.ndarray, budget: float) -> float:
    """Lowest alarm threshold whose false-alarm rate stays within ``budget``/h."""
    negatives = np.sort(scores[labels == 0])[::-1]
    interictal_hours = (labels == 0).sum() * CONFIG.input_stride_seconds / 3600.0
    allowed = int(budget * interictal_hours)
    return float(negatives[min(allowed, len(negatives) - 1)])


# ----------------------------------------------------------------------
# Charts
# ----------------------------------------------------------------------


def chart_output(scores: np.ndarray, labels: np.ndarray, path: Path) -> None:
    """The model output: a sigmoid squashing one logit into one probability."""
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.4))

    axis = axes[0]
    z = np.linspace(-6, 6, 400)
    axis.plot(z, 1 / (1 + np.exp(-z)), color=BLUE, linewidth=2.2)
    axis.axhline(0.5, color=INK, linestyle=":", linewidth=1)
    axis.axvline(0.0, color=INK, linestyle=":", linewidth=1)
    axis.annotate("p = σ(ℓ) = 1/(1+e^−ℓ)", xy=(0.05, 0.86), xycoords="axes fraction",
                  fontsize=11, color=BLUE, fontweight="bold")
    axis.set(title="One logit ℓ becomes one probability p",
             xlabel="Logit ℓ (dimensionless)",
             ylabel="p = P(onset within next 10 min)", ylim=(0, 1))
    style(axis)

    axis = axes[1]
    bins = np.linspace(0, 1, 51)
    axis.hist(scores[labels == 0], bins=bins, color=BLUE, alpha=0.8,
              density=True, label=f"y = 0  (n = {(labels==0).sum():,})")
    axis.hist(scores[labels == 1], bins=bins, color=ORANGE, alpha=0.8,
              density=True, label=f"y = 1  (n = {(labels==1).sum():,})")
    axis.set(title="Observed output distribution, validation split",
             xlabel="p, model output (probability)", ylabel="Probability density")
    axis.legend(frameon=False, fontsize=9)
    style(axis)

    save(figure, path)


def chart_ranking_metrics(scores: np.ndarray, labels: np.ndarray, path: Path) -> None:
    """ROC and PR curves, with AUC and AP drawn as the areas they are."""
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    prevalence = labels.mean()

    fpr, tpr, _ = roc_curve(labels, scores)
    auc = roc_auc_score(labels, scores)
    axis = axes[0]
    axis.fill_between(fpr, tpr, alpha=0.16, color=BLUE)
    axis.plot(fpr, tpr, color=BLUE, linewidth=2)
    axis.plot([0, 1], [0, 1], color=INK, linestyle="--", linewidth=1)
    axis.annotate(f"AUC = {auc:.4f}\nshaded area", xy=(0.45, 0.22),
                  xycoords="axes fraction", fontsize=11, color=BLUE, fontweight="bold")
    axis.annotate("chance = 0.5", xy=(0.55, 0.47), xycoords="axes fraction",
                  fontsize=9, color=INK, rotation=32)
    axis.set(title="ROC — AUC is literally the shaded area",
             xlabel="False positive rate  FP/(FP+TN)",
             ylabel="True positive rate  TP/(TP+FN)", xlim=(0, 1), ylim=(0, 1))
    style(axis)

    precision, recall, _ = precision_recall_curve(labels, scores)
    ap = average_precision_score(labels, scores)
    axis = axes[1]
    axis.fill_between(recall, precision, alpha=0.16, color=ORANGE)
    axis.plot(recall, precision, color=ORANGE, linewidth=2)
    axis.axhline(prevalence, color=INK, linestyle="--", linewidth=1)
    axis.annotate(f"AP = {ap:.4f}\n= {ap/prevalence:.2f}x chance", xy=(0.45, 0.55),
                  xycoords="axes fraction", fontsize=11, color=ORANGE, fontweight="bold")
    axis.annotate(f"chance = prevalence = {prevalence:.4f}", xy=(0.02, prevalence),
                  xytext=(4, 8), textcoords="offset points", fontsize=9, color=INK)
    axis.set(title="Precision-recall — AP is the shaded area",
             xlabel="Recall  TP/(TP+FN)", ylabel="Precision  TP/(TP+FP)",
             xlim=(0, 1), ylim=(0, min(1.0, max(0.08, precision[:-1].max() * 1.15))))
    style(axis)

    save(figure, path)


def chart_confusion_and_far(
    scores: np.ndarray, labels: np.ndarray, budget: float, path: Path
) -> None:
    """Where the threshold lands, and how false positives become alarms per hour."""
    threshold = threshold_for_budget(scores, labels, budget)
    alarm = scores >= threshold
    tp = int((alarm & (labels == 1)).sum())
    fp = int((alarm & (labels == 0)).sum())
    fn = int((~alarm & (labels == 1)).sum())
    tn = int((~alarm & (labels == 0)).sum())
    interictal_hours = (labels == 0).sum() * CONFIG.input_stride_seconds / 3600.0

    figure, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    axis = axes[0]
    matrix = np.array([[tn, fp], [fn, tp]], dtype=float)
    axis.imshow(np.log10(matrix + 1), cmap="Blues", aspect="auto")
    for i in range(2):
        for j in range(2):
            axis.text(j, i, f"{int(matrix[i, j]):,}", ha="center", va="center",
                      fontsize=15, fontweight="bold",
                      color="white" if np.log10(matrix[i, j] + 1) > 3 else INK)
    axis.set(title=f"Confusion matrix at θ = {threshold:.3f}",
             xticks=[0, 1], xticklabels=["no alarm", "alarm"],
             yticks=[0, 1], yticklabels=["y = 0", "y = 1"],
             xlabel="Model decision", ylabel="Truth")
    for side in ("top", "right", "left", "bottom"):
        axis.spines[side].set_visible(False)

    axis = axes[1]
    parts = [tn, fp]
    axis.bar([0, 1], parts, color=[BLUE, ORANGE], width=0.6)
    for i, v in enumerate(parts):
        axis.annotate(f"{v:,}", (i, v), xytext=(0, 4), textcoords="offset points",
                      ha="center", fontsize=10, color=INK, fontweight="bold")
    axis.set(title=f"Negatives → {interictal_hours:,.0f} interictal hours",
             xticks=[0, 1], xticklabels=["correct silence", "false alarm"],
             ylabel="Decisions (count)", yscale="log")
    axis.annotate(
        f"each negative decision = {CONFIG.input_stride_seconds:.0f} s\n"
        f"FAR/h = {fp:,} / {interictal_hours:,.0f} h = {fp/interictal_hours:.2f}",
        xy=(0.03, 0.72), xycoords="axes fraction", fontsize=9.5, color=INK,
    )
    style(axis)

    axis = axes[2]
    thresholds = np.unique(scores)
    if len(thresholds) > 1200:
        thresholds = np.quantile(thresholds, np.linspace(0, 1, 1200))
    far, sens = [], []
    for t in thresholds:
        a = scores >= t
        far.append((a & (labels == 0)).sum() / interictal_hours)
        sens.append((a & (labels == 1)).sum() / max((labels == 1).sum(), 1))
    axis.plot(far, sens, color=BLUE, linewidth=2)
    axis.axvline(budget, color=ORANGE, linestyle="--", linewidth=1.5)
    axis.annotate(f"budget = {budget:g} alarm/h", xy=(budget, 0.9),
                  xycoords=("data", "axes fraction"), xytext=(5, 0),
                  textcoords="offset points", fontsize=9, color=ORANGE)
    axis.set(title="Decision-level sensitivity vs false-alarm rate",
             xlabel="False alarms per interictal hour", ylabel="Sensitivity",
             xscale="log", ylim=(0, 1.02))
    style(axis)

    save(figure, path)
    return {
        "threshold": threshold, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "interictal_hours": interictal_hours, "far": fp / interictal_hours,
    }


def chart_seizure_level(
    predictions: pd.DataFrame, scores: np.ndarray, labels: np.ndarray,
    budget: float, path: Path
) -> dict:
    """How 10 per-minute decisions collapse into one per-seizure verdict."""
    threshold = threshold_for_budget(scores, labels, budget)
    positives = predictions[predictions.label == 1].copy()
    positives["alarm"] = positives["probability"] >= threshold
    grouped = positives.groupby("target_seizure_id")
    caught = grouped["alarm"].any()

    figure, axes = plt.subplots(1, 2, figsize=(14, 4.8))

    # One seizure in detail
    example = grouped["probability"].max().idxmax()
    rows = positives[positives.target_seizure_id == example].sort_values(
        "decision_time_seconds"
    )
    minutes_before = (
        rows["decision_time_seconds"].max() - rows["decision_time_seconds"]
    ) / 60.0
    axis = axes[0]
    colors = [ORANGE if a else BLUE for a in rows["alarm"]]
    axis.bar(minutes_before, rows["probability"], color=colors, width=0.7)
    axis.axhline(threshold, color=INK, linestyle="--", linewidth=1.4)
    axis.annotate(f"θ = {threshold:.3f}", xy=(0.80, threshold),
                  xycoords=("axes fraction", "data"),
                  xytext=(0, 6), textcoords="offset points", fontsize=9, color=INK)
    # Sit low-left: the bars and the threshold line both occupy the upper band.
    axis.annotate("one bar clearing θ is enough —\nthe max decides the seizure",
                  xy=(0.03, 0.10), xycoords="axes fraction", fontsize=9.5, color=INK)
    axis.set(title="One seizure's 10 pre-onset decisions",
             xlabel="Minutes before the last decision", ylabel="p (probability)")
    axis.invert_xaxis()
    style(axis)

    # All seizures, max probability
    axis = axes[1]
    maxima = grouped["probability"].max().sort_values(ascending=False)
    bars = axis.bar(np.arange(len(maxima)), maxima.values,
                    color=[ORANGE if v >= threshold else BLUE for v in maxima.values],
                    width=0.8)
    axis.axhline(threshold, color=INK, linestyle="--", linewidth=1.4)
    axis.set(title=f"Every seizure's maximum p — {int(caught.sum())}/{len(caught)} above θ",
             xlabel="Seizures (rank, sorted by max p)",
             ylabel="max p over the pre-onset window")
    style(axis)

    save(figure, path)
    return {"caught": int(caught.sum()), "total": int(len(caught)), "threshold": threshold}


def chart_calibration(scores: np.ndarray, labels: np.ndarray, path: Path) -> None:
    """Brier score: what it measures and why it differs from ranking."""
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    axis = axes[0]
    quantiles = np.quantile(scores, np.linspace(0, 1, 11))
    quantiles = np.unique(quantiles)
    centres, observed = [], []
    for lo, hi in zip(quantiles[:-1], quantiles[1:], strict=True):
        sel = (scores >= lo) & (scores < hi)
        if sel.sum() < 20:
            continue
        centres.append(scores[sel].mean())
        observed.append(labels[sel].mean())
    top = max(max(centres, default=1), max(observed, default=1)) * 1.1
    axis.plot([0, top], [0, top], color=INK, linestyle="--", linewidth=1)
    axis.plot(centres, observed, color=ORANGE, linewidth=2, marker="o", markersize=8,
              markeredgecolor="white", markeredgewidth=1.4)
    axis.annotate("perfect calibration", xy=(0.55, 0.78), xycoords="axes fraction",
                  fontsize=9, color=INK, rotation=34)
    axis.set(title=f"Calibration — Brier = {brier_score_loss(labels, scores):.4f}",
             xlabel="Mean predicted p in bin", ylabel="Observed frequency of y = 1")
    style(axis)

    axis = axes[1]
    squared = (scores - labels) ** 2
    axis.hist(squared[labels == 0], bins=50, color=BLUE, alpha=0.8, density=True,
              label="y = 0 contributions")
    axis.hist(squared[labels == 1], bins=50, color=ORANGE, alpha=0.8, density=True,
              label="y = 1 contributions")
    axis.set(title="Per-decision squared error (p − y)²",
             xlabel="(p − y)²", ylabel="Probability density", yscale="log")
    axis.legend(frameon=False, fontsize=9)
    style(axis)

    save(figure, path)


# ----------------------------------------------------------------------
# HTML
# ----------------------------------------------------------------------


def build_html(output: Path, figures: list[str], facts: dict) -> Path:
    def embed(name: str) -> str:
        return "data:image/png;base64," + base64.b64encode(
            (output / name).read_bytes()
        ).decode("ascii")

    f = facts
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Input, Output, Metrics</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500&display=swap">
<style>
:root{{color-scheme:light;--paper:#F5F6F8;--surface:#FFF;--surface-2:#EEF1F5;
--ink:#10141A;--ink-2:#4C5568;--ink-3:#6E7788;--rule:#DCE0E6;--rule-firm:#C3CAD4;
--accent:#1E5FA8;--accent-bg:#E7EEF8;--hot:#C2542F;--hot-bg:#F8EAE6}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{color-scheme:dark;
--paper:#0F1216;--surface:#171B21;--surface-2:#1E242C;--ink:#EDF0F4;--ink-2:#A8B2C0;
--ink-3:#7E8797;--rule:#272D36;--rule-firm:#3A424E;--accent:#6BA5EC;--accent-bg:#16222F;
--hot:#E58468;--hot-bg:#2A1A16}}}}
:root[data-theme="dark"]{{color-scheme:dark;--paper:#0F1216;--surface:#171B21;
--surface-2:#1E242C;--ink:#EDF0F4;--ink-2:#A8B2C0;--ink-3:#7E8797;--rule:#272D36;
--rule-firm:#3A424E;--accent:#6BA5EC;--accent-bg:#16222F;--hot:#E58468;--hot-bg:#2A1A16}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--paper);color:var(--ink);
font-family:"IBM Plex Sans",ui-sans-serif,system-ui,sans-serif;font-size:15.5px;
line-height:1.62;-webkit-font-smoothing:antialiased}}
.wrap{{max-width:1120px;margin:0 auto;padding:56px 26px 96px;display:flex;
flex-direction:column;gap:46px}}
.eyebrow{{font-family:"IBM Plex Mono",monospace;font-size:11px;font-weight:500;
letter-spacing:.15em;text-transform:uppercase;color:var(--ink-3);margin:0 0 12px}}
h1{{font-family:Archivo,sans-serif;font-weight:700;font-size:clamp(32px,5vw,48px);
line-height:1.04;letter-spacing:-.022em;margin:0 0 16px;text-wrap:balance}}
.standfirst{{font-size:18px;color:var(--ink-2);max-width:64ch;margin:0}}
section{{display:flex;flex-direction:column;gap:18px}}
h2.sec{{font-family:Archivo,sans-serif;font-size:12.5px;font-weight:600;letter-spacing:.11em;
text-transform:uppercase;color:var(--ink-3);margin:0;padding-bottom:9px;
border-bottom:1px solid var(--rule-firm)}}
h3{{font-family:Archivo,sans-serif;font-size:17px;font-weight:600;margin:8px 0 0;
letter-spacing:-.008em}}
p{{margin:0;max-width:72ch;color:var(--ink-2)}}
.eq{{background:var(--surface);border:1px solid var(--rule);border-left:2px solid var(--accent);
border-radius:2px;padding:15px 18px;overflow-x:auto;font-family:"IBM Plex Mono",monospace;
font-size:13.5px;line-height:2.0;color:var(--ink);white-space:pre;
font-variant-numeric:tabular-nums}}
.eq b{{color:var(--accent);font-weight:600}}
.eq i{{color:var(--ink-3);font-style:normal}}
.eq u{{color:var(--hot);text-decoration:none;font-weight:600}}
figure{{margin:0;background:var(--surface);border:1px solid var(--rule);border-radius:3px;
padding:18px 20px 14px;display:flex;flex-direction:column;gap:12px}}
figure img{{width:100%;height:auto;display:block;border:1px solid var(--rule);
border-radius:2px;background:#fff}}
figure svg{{width:100%;height:auto;display:block;color:var(--ink)}}
figcaption{{font-size:13px;color:var(--ink-3);border-top:1px solid var(--rule);
padding-top:11px;max-width:82ch}}
.tw{{overflow-x:auto;border:1px solid var(--rule);border-radius:2px;background:var(--surface)}}
table{{border-collapse:collapse;width:100%;font-size:14px}}
th,td{{text-align:left;padding:10px 15px;border-bottom:1px solid var(--rule);white-space:nowrap}}
thead th{{font-family:"IBM Plex Mono",monospace;font-size:10px;font-weight:600;
letter-spacing:.1em;text-transform:uppercase;color:var(--ink-3);background:var(--surface-2)}}
tbody tr:last-child td{{border-bottom:none}}
td.m{{font-family:"IBM Plex Mono",monospace;font-variant-numeric:tabular-nums}}
.callout{{background:var(--hot-bg);border:1px solid var(--hot);border-radius:3px;
padding:18px 20px;display:flex;flex-direction:column;gap:9px}}
.callout h3{{margin:0;color:var(--ink);font-size:16px}}
.callout p{{color:var(--ink-2)}}
code{{font-family:"IBM Plex Mono",monospace;font-size:.9em;background:var(--surface-2);
padding:1px 5px;border-radius:2px}}
footer{{border-top:1px solid var(--rule-firm);padding-top:20px;
font-family:"IBM Plex Mono",monospace;font-size:11.5px;color:var(--ink-3);line-height:1.9}}
</style></head><body><div class="wrap">

<header>
<p class="eyebrow">SeizeIT2 · behind-the-ear EEG · input, output, metrics</p>
<h1>What goes in, what comes out, and how it is scored</h1>
<p class="standfirst">The three things you need to read any result from this pipeline,
in math and in charts. Every chart is computed from {f['n']:,} real validation
decisions, so the numbers here are the numbers the model actually produced.</p>
</header>

<section>
<h2 class="sec">1 · Input</h2>
<p>One example is 45 minutes of EEG ending at the decision instant, plus a
three-value mask saying which electrodes physically existed. Nothing else: no
patient ID, no time of day, no prior prediction.</p>

<figure>
<svg viewBox="0 0 960 300" role="img" aria-label="Input shape cascade: a 3-by-691200 continuous slice is reshaped into 540 chunks of 3 by 1280, folded into the batch, encoded to 540 by 32, pooled to 32, concatenated with a 3-value mask to 35.">
<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
<polygon points="0,0 10,5 0,10" fill="currentColor"/></marker></defs>
<g font-family="IBM Plex Mono, monospace" font-size="11.5">
<rect x="16" y="60" width="182" height="72" fill="var(--accent-bg)" stroke="var(--accent)" stroke-width="1.5"/>
<text x="107" y="84" text-anchor="middle" fill="var(--accent)" font-weight="600">X ∈ ℝ^(3 × 691200)</text>
<text x="107" y="101" text-anchor="middle" fill="var(--accent)">45 min × 256 Hz</text>
<text x="107" y="118" text-anchor="middle" fill="var(--accent)">2.6 MB float32</text>
<line x1="198" y1="96" x2="240" y2="96" stroke="currentColor" marker-end="url(#a)"/>
<text x="219" y="88" text-anchor="middle" fill="var(--ink-3)" font-size="10">reshape</text>

<rect x="242" y="60" width="182" height="72" fill="var(--surface-2)" stroke="currentColor"/>
<text x="333" y="84" text-anchor="middle" fill="currentColor" font-weight="600">U ∈ ℝ^(540 × 3 × 1280)</text>
<text x="333" y="101" text-anchor="middle" fill="var(--ink-3)">540 chunks of 5 s</text>
<text x="333" y="118" text-anchor="middle" fill="var(--ink-3)">same bytes, no copy</text>
<line x1="424" y1="96" x2="466" y2="96" stroke="currentColor" marker-end="url(#a)"/>
<text x="445" y="88" text-anchor="middle" fill="var(--ink-3)" font-size="10">φ ×540</text>

<rect x="468" y="60" width="170" height="72" fill="var(--surface-2)" stroke="currentColor"/>
<text x="553" y="84" text-anchor="middle" fill="currentColor" font-weight="600">E ∈ ℝ^(540 × 32)</text>
<text x="553" y="101" text-anchor="middle" fill="var(--ink-3)">EEGNet embeddings</text>
<text x="553" y="118" text-anchor="middle" fill="var(--ink-3)">shared weights</text>
<line x1="638" y1="96" x2="680" y2="96" stroke="currentColor" marker-end="url(#a)"/>
<text x="659" y="88" text-anchor="middle" fill="var(--ink-3)" font-size="10">pool</text>

<rect x="682" y="60" width="120" height="72" fill="var(--surface-2)" stroke="currentColor"/>
<text x="742" y="90" text-anchor="middle" fill="currentColor" font-weight="600">ē ∈ ℝ³²</text>
<text x="742" y="108" text-anchor="middle" fill="var(--ink-3)">one vector</text>
<line x1="802" y1="96" x2="844" y2="96" stroke="currentColor" marker-end="url(#a)"/>

<rect x="846" y="60" width="100" height="72" fill="var(--hot-bg)" stroke="var(--hot)" stroke-width="1.5"/>
<text x="896" y="90" text-anchor="middle" fill="var(--hot)" font-weight="600">p ∈ (0,1)</text>
<text x="896" y="108" text-anchor="middle" fill="var(--hot)">one number</text>

<rect x="682" y="180" width="120" height="52" fill="var(--surface-2)" stroke="currentColor" stroke-dasharray="4 3"/>
<text x="742" y="202" text-anchor="middle" fill="currentColor" font-weight="600">a ∈ {{0,1}}³</text>
<text x="742" y="219" text-anchor="middle" fill="var(--ink-3)">Σa = 2</text>
<line x1="802" y1="206" x2="880" y2="206" stroke="currentColor" stroke-dasharray="4 3"/>
<line x1="880" y1="206" x2="880" y2="136" stroke="currentColor" marker-end="url(#a)" stroke-dasharray="4 3"/>
<text x="742" y="256" text-anchor="middle" fill="var(--ink-3)" font-size="10">electrode availability mask</text>
</g></svg>
<figcaption>The full input path. 2.6 MB of float32 collapses to a single scalar; the
mask is concatenated just before the classifier, never seen by the encoder.</figcaption>
</figure>

<div class="eq"><i>the slice, for decision k in recording r</i>
n<sub>k</sub> = h + k·S·f<sub>s</sub>        h = 691,200 samples,  S = 60 s,  f<sub>s</sub> = 256 Hz
<b>X<sub>k</sub> = Z<sup>(r)</sup>[ : , n<sub>k</sub>−h : n<sub>k</sub> ]  ∈ ℝ<sup>3×691200</sup></b>

<i>reshaped into the encoder's unit of work</i>
U[m] = X<sub>k</sub>[ : , 1280m : 1280(m+1) ]     m = 0 … 539

<i>plus the mask — exactly two of three electrodes exist in any recording</i>
a = (a<sub>BTE_L</sub>, a<sub>BTE_R</sub>, a<sub>CROSS</sub>) ∈ {{0,1}}³ ,   Σ<sub>c</sub> a<sub>c</sub> = 2</div>
</section>

<section>
<h2 class="sec">2 · Output</h2>
<p>One scalar per decision. It is <em>not</em> a seizure detection and it carries no
time-to-onset: it answers only "will an onset begin in the next 10 minutes?"</p>

<div class="eq"><i>the model emits a logit; the sigmoid maps it to a probability</i>
ℓ = wᵀ·LayerNorm([ ē ; a ]) + b        w ∈ ℝ³⁵
<b>p = σ(ℓ) = 1/(1 + e<sup>−ℓ</sup>) ∈ (0,1)</b>

<i>read as</i>   p = P( ∃ onset τ ∈ (t , t + 600 s] | preceding 45 min )

<i>a decision fires an alarm only after comparison with a threshold</i>
alarm = [ p ≥ θ ]      θ is chosen afterwards, from the false-alarm budget</div>

<figure><img src="{embed(figures[0])}" alt="Sigmoid and observed output distribution">
<figcaption>Left, the squashing function. Right, the real distribution of p on the
validation split — the two classes overlap almost completely, which is what an
AUC near {f['auc']:.2f} looks like from the inside.</figcaption></figure>
</section>

<section>
<h2 class="sec">3 · Metrics that need no threshold</h2>
<p>Both summarise the ranking over all thresholds at once, so neither depends on a
choice of θ. They differ in what they treat as the baseline.</p>

<div class="eq"><i>ROC AUC — probability a random positive outranks a random negative</i>
AUC = P( p<sub>i</sub> &gt; p<sub>j</sub> | y<sub>i</sub>=1, y<sub>j</sub>=0 )        chance = 0.5   <b>measured {f['auc']:.4f}</b>

<i>average precision — area under precision-recall</i>
AP = Σ<sub>n</sub> (R<sub>n</sub> − R<sub>n−1</sub>)·P<sub>n</sub>          chance = prevalence = {f['prev']:.5f}
<b>measured {f['ap']:.4f} = {f['ap_ratio']:.2f}× chance</b>

<i>why AP is the honest one here</i>
at {f['prev']*100:.2f}% prevalence a useless model still gets AUC 0.5 but AP {f['prev']:.5f};
AUC is inflated by the {f['n_neg']:,} negatives that are easy to rank low.</div>

<figure><img src="{embed(figures[1])}" alt="ROC and precision-recall curves with shaded areas">
<figcaption>AUC and AP drawn as the areas they are. The PR panel's y-axis is clipped
near the top of the observed precision — at this prevalence the curve never leaves
the floor, which the ROC panel visually hides.</figcaption></figure>
</section>

<section>
<h2 class="sec">4 · Metrics that need a threshold</h2>
<p>Everything clinical starts by fixing how many false alarms per hour are tolerable,
then reading sensitivity off at that point. The threshold is an output of the budget,
not a free parameter.</p>

<div class="eq"><i>each negative decision stands for S seconds of interictal time</i>
interictal hours = N⁻ · S / 3600 = {f['n_neg']:,} · 60 / 3600 = <b>{f['hours']:,.0f} h</b>
FAR<sub>/h</sub>(θ) = FP(θ) / {f['hours']:,.0f}

<i>at the {f['budget']:g} alarm/h budget used throughout this project</i>
θ = {f['threshold']:.4f}    TP = {f['tp']}   FP = {f['fp']:,}   FN = {f['fn']}   TN = {f['tn']:,}
achieved FAR = {f['fp']:,} / {f['hours']:,.0f} = <b>{f['far']:.2f} alarms/h</b>

<i>decision level — every one-minute decision counts separately</i>
Sens<sub>dec</sub> = TP / (TP + FN) = {f['tp']} / {f['tp']+f['fn']} = <b>{f['sens_dec']:.3f}</b></div>

<figure><img src="{embed(figures[2])}" alt="Confusion matrix, negative-to-hours conversion, sensitivity versus false alarm rate">
<figcaption>Left: the confusion matrix at the budgeted threshold, log-shaded because
TN dwarfs everything else. Middle: how a count of false positives becomes a rate per
hour. Right: the full trade-off curve, with the budget marked.</figcaption></figure>
</section>

<section>
<h2 class="sec">5 · The metric that actually matters</h2>
<p>A patient needs one warning per seizure, not ten. Seizure-level sensitivity
collapses each seizure's whole pre-onset window into a single verdict by taking the
maximum — so nine misses and one hit still counts as caught.</p>

<div class="eq"><i>K<sub>s</sub> = the decisions targeting seizure s</i>
<b>Sens<sub>sz</sub>(θ) = |{{ s : max<sub>k∈K<sub>s</sub></sub> p<sub>k</sub> ≥ θ }}| / |𝒮*|</b>

<i>measured at {f['budget']:g} alarm/h</i>
{f['caught']}/{f['total']} seizures = <b>{f['caught']/f['total']:.3f}</b>

<i>why it differs so much from the decision-level number</i>
each seizure contributes SOP/S = 600/60 = 10 chances; one suffices.</div>

<figure><img src="{embed(figures[3])}" alt="One seizure's ten decisions, and the maximum probability for every seizure">
<figcaption>Left: a single seizure's ten pre-onset decisions against the threshold.
Right: every seizure reduced to its maximum p — the count above the dashed line is
the seizure-level numerator.</figcaption></figure>
</section>

<section>
<h2 class="sec">6 · Calibration</h2>
<p>AUC and AP only care about order. Brier asks whether p means what it says — whether
decisions assigned p = 0.3 are followed by a seizure 30 % of the time.</p>

<div class="eq"><i>mean squared error between probability and outcome</i>
<b>Brier = (1/N) Σ<sub>i</sub> (p<sub>i</sub> − y<sub>i</sub>)²</b>        measured {f['brier']:.4f}

<i>the trap at low prevalence</i>
always predicting p = {f['prev']:.5f} gives Brier ≈ {f['prev']*(1-f['prev']):.5f} while being useless,
so Brier alone can look excellent for a model that never fires.</div>

<figure><img src="{embed(figures[4])}" alt="Calibration curve and per-decision squared error">
<figcaption>Left: observed frequency against predicted probability, quantile-binned so
each point rests on the same number of decisions. A model on the dashed line is
calibrated. Right: where the Brier score's mass comes from.</figcaption></figure>
</section>

<section class="callout">
<h3>Reading these together</h3>
<p>The four numbers answer different questions and can disagree. A model can improve
AUC while losing seizures, which is exactly what the attention-pooling variant did:
AUC rose to 0.6027 while seizure-level sensitivity fell to 1/33. Ranking improved in
the bulk of the distribution and got worse at the high-confidence end where alarms
are actually triggered.</p>
<p>Report order for this project: <strong>seizure-level sensitivity at a stated alarm
budget</strong> first, AP second (with its chance ratio, never bare), AUC third, Brier
only alongside a claim about probabilities. And always with the denominator — {f['total']}
seizures means a two-seizure change is noise.</p>
</section>

<footer>
Generated by scripts/explain_io_metrics.py from {f['source']}<br>
{f['n']:,} decisions · {f['n_pos']} positive · {f['total']} target seizures · {f['hours']:,.0f} interictal hours<br>
Figures embedded as data URIs; this file is self-contained and opens from disk
</footer>

</div></body></html>"""
    path = output / "io_and_metrics.html"
    path.write_text(html, encoding="utf-8")
    return path


def main() -> None:
    arguments = parse_arguments()
    CONFIG.validate()
    output = arguments.output_dir
    output.mkdir(parents=True, exist_ok=True)

    log(f"Loading {arguments.predictions}...")
    predictions = pd.read_csv(arguments.predictions, dtype={"subject": str})
    scores = predictions["probability"].to_numpy(dtype=np.float64)
    labels = predictions["label"].to_numpy(dtype=np.int64)
    log(f"  {len(predictions):,} decisions, {int(labels.sum())} positive")

    log("Charts...")
    names = ["m1_output.png", "m2_ranking.png", "m3_threshold.png",
             "m4_seizure_level.png", "m5_calibration.png"]
    chart_output(scores, labels, output / names[0])
    chart_ranking_metrics(scores, labels, output / names[1])
    conf = chart_confusion_and_far(scores, labels, arguments.alarm_budget, output / names[2])
    seiz = chart_seizure_level(predictions, scores, labels, arguments.alarm_budget,
                              output / names[3])
    chart_calibration(scores, labels, output / names[4])

    prevalence = float(labels.mean())
    ap = float(average_precision_score(labels, scores))
    facts = {
        "n": len(predictions), "n_pos": int(labels.sum()), "n_neg": int((labels == 0).sum()),
        "prev": prevalence, "ap": ap, "ap_ratio": ap / prevalence,
        "auc": float(roc_auc_score(labels, scores)),
        "brier": float(brier_score_loss(labels, scores)),
        "budget": arguments.alarm_budget,
        "sens_dec": conf["tp"] / max(conf["tp"] + conf["fn"], 1),
        "hours": conf["interictal_hours"],
        "source": str(arguments.predictions.relative_to(PROJECT_ROOT)),
        **conf, **seiz,
    }
    path = build_html(output, names, facts)
    log(f"  wrote {path.name}  ({path.stat().st_size/1e6:.1f} MB, self-contained)")
    log(f"\nDone: {path}")


if __name__ == "__main__":
    main()
