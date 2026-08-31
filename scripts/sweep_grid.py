r"""Define the exploration grid and emit one line of CLI flags per configuration.

Kept as a script rather than a shell array so the grid is version-controlled,
inspectable, and reproducible:

    python scripts/sweep_grid.py --list           # human-readable table
    python scripts/sweep_grid.py --flags 7        # flags for one config
    python scripts/sweep_grid.py --count          # grid size, for --array
"""

from __future__ import annotations

import argparse
import itertools

# Axes chosen from measured evidence, not convenience.
#
# architecture  five-fold CV showed attention matching a linear control, so the
#               aggregation strategy is swept explicitly rather than assumed.
# capacity      the failure could be under- or over-fitting; span 20x parameters.
# history       45 min may simply be too much context for a 10-minute question.
# sop           the literature usually allows 30-60 min; 10 min may be too tight
#               for scalp-adjacent electrodes. Relabelling is free.
ARCHITECTURES = ("logistic-mean", "spectral-meanpool", "spectral-attention", "spectral-gru")
CAPACITIES = {"small": (16, 32), "large": (64, 128)}
HISTORY_MINUTES = (5.0, 15.0, 45.0)
SOP_MINUTES = (10.0, 30.0)


def build_grid() -> list[dict]:
    configurations: list[dict] = []
    for architecture, (capacity, (embedding, hidden)), history, sop in itertools.product(
        ARCHITECTURES, CAPACITIES.items(), HISTORY_MINUTES, SOP_MINUTES
    ):
        # The linear control has no embedding, so only emit it once per capacity.
        if architecture == "logistic-mean" and capacity != "small":
            continue
        configurations.append({
            "architecture": architecture,
            "capacity": capacity,
            "embedding_dim": embedding,
            "hidden_dim": hidden,
            "history_minutes": history,
            "sop_minutes": sop,
        })
    return configurations


def tag_for(configuration: dict) -> str:
    return (
        f"{configuration['architecture']}__{configuration['capacity']}"
        f"__h{configuration['history_minutes']:g}__sop{configuration['sop_minutes']:g}"
    )


def flags_for(configuration: dict) -> str:
    return " ".join([
        f"--architecture {configuration['architecture']}",
        f"--embedding-dim {configuration['embedding_dim']}",
        f"--hidden-dim {configuration['hidden_dim']}",
        f"--history-minutes {configuration['history_minutes']:g}",
        f"--sop-minutes {configuration['sop_minutes']:g}",
        f"--tag {tag_for(configuration)}",
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--count", action="store_true")
    parser.add_argument("--flags", type=int, default=None)
    arguments = parser.parse_args()

    grid = build_grid()
    if arguments.count:
        print(len(grid))
    elif arguments.flags is not None:
        print(flags_for(grid[arguments.flags]))
    else:
        print(f"{len(grid)} configurations\n")
        print(f"{'#':>3}  {'architecture':<19}{'cap':<7}{'hist':>6}{'sop':>6}  tag")
        for index, configuration in enumerate(grid):
            print(f"{index:>3}  {configuration['architecture']:<19}"
                  f"{configuration['capacity']:<7}"
                  f"{configuration['history_minutes']:>5.0f}m"
                  f"{configuration['sop_minutes']:>5.0f}m  {tag_for(configuration)}")


if __name__ == "__main__":
    main()
