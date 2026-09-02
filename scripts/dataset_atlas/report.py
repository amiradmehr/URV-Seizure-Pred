"""Build the self-contained HTML report from whatever figures were rendered."""

from __future__ import annotations

import base64
import html
from pathlib import Path
from types import ModuleType

from .common import CONFIG


def _embed(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def _escape(text: str) -> str:
    return html.escape(str(text))


STYLE = """
:root {
  color-scheme: light;
  --paper:#F5F6F8; --surface:#FFFFFF; --surface-2:#EEF1F5;
  --ink:#10141A; --ink-2:#4C5568; --ink-3:#6E7788;
  --rule:#DCE0E6; --rule-firm:#C3CAD4;
  --accent:#1E5FA8; --accent-bg:#E7EEF8; --hot:#4a3aa7;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --paper:#0F1216; --surface:#171B21; --surface-2:#1E242C;
    --ink:#EDF0F4; --ink-2:#A8B2C0; --ink-3:#7E8797;
    --rule:#272D36; --rule-firm:#3A424E;
    --accent:#6BA5EC; --accent-bg:#16222F; --hot:#9085e9;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --paper:#0F1216; --surface:#171B21; --surface-2:#1E242C;
  --ink:#EDF0F4; --ink-2:#A8B2C0; --ink-3:#7E8797;
  --rule:#272D36; --rule-firm:#3A424E;
  --accent:#6BA5EC; --accent-bg:#16222F; --hot:#9085e9;
}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);
  font-family:system-ui,-apple-system,"Segoe UI",sans-serif;
  font-size:15.5px;line-height:1.62;-webkit-font-smoothing:antialiased}
.wrap{max-width:1280px;margin:0 auto;padding:56px 26px 96px;
  display:flex;flex-direction:column;gap:44px}
.eyebrow{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11px;font-weight:500;
  letter-spacing:.15em;text-transform:uppercase;color:var(--ink-3);margin:0 0 12px}
h1{font-weight:700;font-size:clamp(32px,5vw,50px);
  line-height:1.04;letter-spacing:-.022em;margin:0 0 16px;text-wrap:balance}
.standfirst{font-size:18px;color:var(--ink-2);max-width:66ch;margin:0}
.units{margin-top:22px;padding:16px 18px;background:var(--surface);border:1px solid var(--rule);
  border-left:2px solid var(--accent);border-radius:2px;white-space:pre-wrap;
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12.5px;line-height:1.95;
  color:var(--ink-2);overflow-x:auto}
.units b{color:var(--accent);font-weight:600}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px}
.stat{background:var(--surface);border:1px solid var(--rule);border-radius:2px;
  padding:15px 17px;display:flex;flex-direction:column;gap:3px}
.stat-label{font-size:12.5px;color:var(--ink-3)}
.stat-value{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:25px;font-weight:600;
  letter-spacing:-.01em;font-variant-numeric:tabular-nums}
.stat-note{font-size:12px;color:var(--ink-3)}
h2.sec{font-size:12.5px;font-weight:600;letter-spacing:.11em;
  text-transform:uppercase;color:var(--ink-3);margin:0 0 14px;padding-bottom:9px;
  border-bottom:1px solid var(--rule-firm)}
.tw{overflow-x:auto;border:1px solid var(--rule);border-radius:2px;background:var(--surface)}
table{border-collapse:collapse;width:100%;font-size:14px}
th,td{text-align:left;padding:10px 15px;border-bottom:1px solid var(--rule);white-space:nowrap}
thead th{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:10px;font-weight:600;
  letter-spacing:.1em;text-transform:uppercase;color:var(--ink-3);background:var(--surface-2)}
tbody tr:last-child td{border-bottom:none}
td.m{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-variant-numeric:tabular-nums}
.toc{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:8px}
.toc a{display:flex;gap:10px;align-items:baseline;text-decoration:none;color:var(--ink-2);
  background:var(--surface);border:1px solid var(--rule);border-radius:2px;padding:10px 13px;
  font-size:13.5px}
.toc a:hover{border-color:var(--accent);color:var(--ink)}
.toc .n{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11px;font-weight:600;
  color:var(--accent)}
.fig{background:var(--surface);border:1px solid var(--rule);border-radius:3px;
  padding:22px 24px 18px;display:flex;flex-direction:column;gap:16px;scroll-margin-top:20px}
.fig-head{display:flex;align-items:baseline;gap:13px;flex-wrap:wrap;
  border-bottom:1px solid var(--rule);padding-bottom:12px}
.fig-num{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;font-weight:600;
  color:#fff;background:var(--accent);border-radius:2px;padding:3px 8px}
.fig h2{font-size:21px;font-weight:600;margin:0;letter-spacing:-.012em}
.fig .q{font-size:14px;color:var(--ink-3);font-style:italic;margin:0}
.fig img{display:block;width:100%;height:auto;border:1px solid var(--rule);border-radius:2px;
  background:#fff}
.fig-body{display:grid;grid-template-columns:1fr 1fr;gap:26px}
@media (max-width:900px){.fig-body{grid-template-columns:1fr;gap:16px}}
.fig-col h3{font-size:11px;font-weight:600;letter-spacing:.1em;
  text-transform:uppercase;color:var(--ink-3);margin:0 0 7px}
.fig-col p{margin:0;color:var(--ink-2);font-size:14.5px}
.fig-file{margin:0;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11.5px;
  color:var(--ink-3);border-top:1px solid var(--rule);padding-top:11px}
.closing{background:var(--accent-bg);border:1px solid var(--accent);border-radius:3px;
  padding:24px 26px;display:flex;flex-direction:column;gap:12px}
.closing h2{font-size:20px;font-weight:600;margin:0;letter-spacing:-.012em}
.closing p{margin:0;color:var(--ink-2);max-width:78ch}
.closing ol{margin:4px 0 0;padding-left:20px;color:var(--ink-2)}
.closing li{margin-bottom:7px}
footer{border-top:1px solid var(--rule-firm);padding-top:20px;
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11.5px;color:var(--ink-3);
  line-height:1.9}
code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.9em;
  background:var(--surface-2);padding:1px 5px;border-radius:2px}
"""


def build_html_report(
    output: Path, summary: dict, modules: list[ModuleType]
) -> Path:
    """Write ``dataset_report.html`` with every rendered figure embedded."""
    present = [
        module
        for module in modules
        if (output / f"{module.NUMBER}_{module.SLUG}.png").exists()
    ]

    micro = summary["units"]["microvolts_per_z"]
    by_split = summary["by_split"]
    facts = summary.get("figure_facts", {})

    stat_cards = [
        ("Patients with usable data", f"{summary['patients_with_data']}", "of 125 configured"),
        ("EDF files", f"{summary['edf_files']:,}", "behind-the-ear recordings"),
        ("EEG duration", f"{summary['eeg_hours']:,.0f} h", "continuous, stored unwindowed"),
        ("Decisions", f"{summary['decisions']:,}", "one per 60 s of usable EEG"),
        (
            "Positive decisions",
            f"{summary['positive_decisions']:,}",
            f"{summary['prevalence'] * 100:.2f} % prevalence",
        ),
        ("Seizures annotated", f"{summary['seizures_annotated']}", "in the event files"),
        (
            "Seizures usable",
            f"{summary['seizures_targeted']}",
            f"{summary['seizures_targeted'] / max(summary['seizures_annotated'], 1) * 100:.0f} % "
            f"have 60 min of clean history",
        ),
        (
            "Input per decision",
            f"{3 * int(CONFIG.input_window_seconds * CONFIG.target_sfreq):,}",
            "numbers -> one probability",
        ),
    ]

    cards_html = "\n".join(
        f'      <div class="stat"><span class="stat-label">{_escape(label)}</span>'
        f'<span class="stat-value">{_escape(value)}</span>'
        f'<span class="stat-note">{_escape(note)}</span></div>'
        for label, value, note in stat_cards
    )

    splits_html = "\n".join(
        f"          <tr><td>{name}</td><td class='m'>{d['patients']}</td>"
        f"<td class='m'>{d['decisions']:,}</td><td class='m'>{d['positive']:,}</td>"
        f"<td class='m'>{d['positive'] / d['decisions'] * 100:.2f} %</td></tr>"
        for name, d in (
            ("train", by_split["train"]),
            ("validation", by_split["validation"]),
            ("test", by_split["test"]),
        )
    )

    toc_html = "\n".join(
        f'      <a href="#fig{module.NUMBER}"><span class="n">{module.NUMBER}</span>'
        f"<span>{_escape(module.TITLE)}</span></a>"
        for module in present
    )

    figures_html = "\n".join(
        f'''  <section class="fig" id="fig{module.NUMBER}">
    <div class="fig-head">
      <span class="fig-num">{module.NUMBER}</span>
      <h2>{_escape(module.TITLE)}</h2>
      <p class="q">{_escape(getattr(module, "QUESTION", ""))}</p>
    </div>
    <img src="{_embed(output / f"{module.NUMBER}_{module.SLUG}.png")}"
         alt="{_escape(module.TITLE)}">
    <div class="fig-body">
      <div class="fig-col"><h3>What it shows</h3><p>{_escape(module.READS)}</p></div>
      <div class="fig-col"><h3>What to take from it</h3><p>{_escape(module.TAKE)}</p></div>
    </div>
    <p class="fig-file">{module.NUMBER}_{module.SLUG}.png</p>
  </section>'''
        for module in present
    )

    sigma_rows = "\n        ".join(
        f"{name:<11s} 1 z = {value:6.0f} uV" for name, value in micro.items()
    )

    hero = facts.get("hero_recording", "a 21-hour recording")
    prevalence = summary["prevalence"] * 100

    html_text = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SeizeIT2 Dataset Atlas</title>
<style>{STYLE}</style>
</head>
<body>
<div class="wrap">

  <header>
    <p class="eyebrow">SeizeIT2 &middot; behind-the-ear EEG &middot; dataset atlas</p>
    <h1>What the seizure-prediction data actually looks like</h1>
    <p class="standfirst">{len(present)} figures over {summary['eeg_hours']:,.0f} hours of EEG
    from {summary['patients_with_data']} patients, generated by
    <code>scripts/visualize_dataset.py</code>. Most of them show real EEG with the labelling
    geometry drawn on top &mdash; whole recordings, seizures at five magnifications, the exact
    window one decision reads, and the artefacts that dominate it. Each is paired with what it
    shows and what it implies for modelling.</p>
    <div class="units"><b>Units used on every axis</b>
count       a number of things &mdash; EDF files, decisions, seizures, patients
h / min / s wall-clock duration of EEG
Hz          frequency
z           dimensionless amplitude; 1 z = one global channel sigma
        {sigma_rows}</div>
  </header>

  <section>
    <h2 class="sec">At a glance</h2>
    <div class="stats">
{cards_html}
    </div>
  </section>

  <section>
    <h2 class="sec">Splits &mdash; whole patients, never mixed</h2>
    <div class="tw"><table>
      <thead><tr><th>Split</th><th>Patients</th><th>Decisions</th><th>Positive</th>
      <th>Prevalence</th></tr></thead>
      <tbody>
{splits_html}
      </tbody>
    </table></div>
  </section>

  <section>
    <h2 class="sec">The figures</h2>
    <div class="toc">
{toc_html}
    </div>
  </section>

{figures_html}

  <section class="closing">
    <h2>What the data says about the modelling problem</h2>
    <p>Four constraints are visible in the figures and none of them are fixable by training
    longer.</p>
    <ol>
      <li><strong>The positive class is set by arithmetic.</strong> The occurrence period
      divided by the stride gives ten positive decisions per seizure, and only
      {summary['seizures_targeted']} seizures qualify, so
      {summary['positive_decisions']:,} positives is the ceiling until the stride or the
      eligibility rule changes. Prevalence is {prevalence:.2f} %.</li>
      <li><strong>The model never sees the seizure.</strong> Figure 09 puts the evidence and
      the question on one axis: the onset always falls after the decision instant, inside the
      occurrence period, and the history that precedes it carries no visible landmark.</li>
      <li><strong>Between-patient amplitude dwarfs any pre-ictal signal.</strong> One global
      scaler leaves the typical recording far from sigma = 1 and the patient-to-patient spread
      intact. Separating patients is easy; separating pre-ictal states is not.</li>
      <li><strong>Artefact is larger than physiology.</strong> In {_escape(str(hero))} the
      biggest excursions of the night are electrode and movement transients, not the three
      seizures.</li>
    </ol>
    <p>Together these are the argument for normalising each window on its own robust
    statistics rather than on one global constant, for densifying positives near onset, and
    for a temporal aggregator that does not discard chunk order.</p>
  </section>

  <footer>
    Generated by scripts/visualize_dataset.py &middot; figures embedded, file is
    self-contained<br>
    Source PNGs, exemplars.json and summary.json sit beside this file in
    outputs/dataset_figures/<br>
    Amplitude is z throughout; 1 z equals one global per-channel sigma, converted to
    microvolts beside every channel name.
  </footer>

</div>
</body>
</html>"""

    path = output / "dataset_report.html"
    path.write_text(html_text, encoding="utf-8")
    return path
