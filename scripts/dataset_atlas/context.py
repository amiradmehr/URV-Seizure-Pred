"""The object every figure module receives.

A figure module is a file named ``fNN_slug.py`` exposing module-level metadata
plus ``render(atlas) -> Path | None``.  The orchestrator in
``scripts/visualize_dataset.py`` imports each one, calls ``render``, and uses the
metadata to build the HTML report, so a new figure is added by dropping one file
into this package and nothing else.

Required module attributes::

    NUMBER    "05"                two-digit ordering prefix
    SLUG      "recording_timeline"
    TITLE     "One recording, end to end"
    QUESTION  "What does a single night of behind-the-ear EEG look like?"
    READS     what the figure shows, for the report
    TAKE      what to take from it, for the report

``render`` may also stash measured numbers in ``atlas.facts`` so the report and
``summary.json`` can quote them instead of hard-coding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .common import Manifests
from .exemplars import Exemplars


@dataclass
class Atlas:
    """Shared state for one atlas build."""

    manifests: Manifests
    exemplars: Exemplars
    output: Path
    rng: np.random.Generator
    recording_sample: int = 60
    facts: dict[str, object] = field(default_factory=dict)

    def path(self, number: str, slug: str) -> Path:
        return self.output / f"{number}_{slug}.png"

    def fact(self, key: str, value: object) -> object:
        """Record a measured number for the report, and return it unchanged."""
        self.facts[key] = value
        return value
