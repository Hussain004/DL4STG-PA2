"""Structural reach of the proposed architecture on the river graph.

This is the river analogue of the companion notebook's receptive-field widget:
it answers "which (place, time) inputs can possibly influence the forecast at
S4?" using only the architecture's structure - no training, no gradients.

Structure of the proposed model
-------------------------------
* temporal encoder: 4 dilated convolutions (dilations 1, 2, 4, 8) -> a radius
  of 1+2+4+8 = 15 steps around the read-out index, at the SAME place;
* graph space-time mixing: 4 rounds along the directed river graph; every edge
  carries a learned kernel over W = 12 lags (0 ... 110 min), so a message may
  be taken from any lag in that window;
* read-out at the forecast origin t = 0, one state per forecast station.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import river_data as rd          # noqa: E402

ROW_OF = {r: i for i, r in enumerate(rd.ROWS)}

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
RES = ROOT / "results"

EDGES = [("Dam", "S1"), ("S1", "S2"), ("T1", "S2"), ("S2", "S3"), ("S3", "S4")]
UP = {}
for _a, _b in EDGES:
    UP.setdefault(_b, []).append(_a)


def reach(target="S4", rounds=4, lag_window=12, encoder_radius=15, horizon=6):
    """Set of (place, time) pairs that can influence the forecast at `target`."""
    reach_set = {(target, 0)}
    # undo the graph rounds (reverse order)
    for _ in range(rounds):
        new = set(reach_set)
        for (place, t) in list(reach_set):
            for up in UP.get(place, []):
                for tau in range(lag_window):
                    new.add((up, t - tau))
        reach_set = new
    # undo the temporal encoder (symmetric kernel)
    new = set(reach_set)
    for (place, t) in reach_set:
        for dt in range(-encoder_radius, encoder_radius + 1):
            new.add((place, t + dt))
    reach_set = new
    return reach_set


def summarise(rounds=4, lag_window=12, encoder_radius=15):
    rows = []
    for target in ["S1", "S2", "S3", "S4"]:
        R = reach(target, rounds, lag_window, encoder_radius)
        info = {"target": target}
        for place in rd.ROWS:
            times = sorted(t for (p, t) in R if p == place and t < 0)
            info[place] = {"n_steps_back": len(times),
                           "furthest_back": -min(times) if times else 0}
        rows.append(info)
    return rows


def figure(rounds=4, lag_window=12, encoder_radius=15, lookback=24, span=60):
    R = reach("S4", rounds, lag_window, encoder_radius)
    grid = np.zeros((len(rd.ROWS), span + 1))
    for (place, t) in R:
        if -span <= t <= 0:
            grid[ROW_OF[place], t + span] = 1.0

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.2), layout="constrained",
                             width_ratios=[1.25, 1])
    im = axes[0].imshow(grid, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    axes[0].set_yticks(range(len(rd.ROWS)))
    axes[0].set_yticklabels(rd.ROWS)
    axes[0].set_xlabel("steps before the forecast origin")
    axes[0].set_title("Structural reach of the S4 forecast\n"
                      "(blue = an input that can reach it; the architecture reaches\n"
                      "further back than the 24-step window we actually supply)", fontsize=10)
    for i in range(len(rd.ROWS)):
        for j in range(lookback + 1):
            if grid[i, j] > 0:
                axes[0].text(j, i, "", va="center")
    axes[0].axvline(span, color="#c44963", lw=2)
    axes[0].text(span + 0.8, 0, "origin", color="#c44963", fontsize=8, va="center")
    axes[0].axvline(span - lookback, color="#333333", lw=2, ls="--")
    axes[0].text(span - lookback - 1, 5, "window L=24", color="#333333", fontsize=8,
                 va="center", ha="right")
    fig.colorbar(im, ax=axes[0], fraction=0.03, label="can reach the forecast")

    # how far back each place is still relevant
    for i, place in enumerate(rd.ROWS):
        times = [t for (p, t) in R if p == place and t < 0]
        if times:
            axes[1].barh(i, -min(times), color="#4774b5")
            axes[1].text(-min(times) + 0.4, i, f"{-min(times)} steps "
                                             f"({-min(times)*10} min)", va="center", fontsize=8)
    axes[1].set_yticks(range(len(rd.ROWS)))
    axes[1].set_yticklabels(rd.ROWS)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("furthest-back step that can still reach the S4 forecast")
    axes[1].set_title("Reach per place", fontsize=11)
    axes[1].axvline(lookback, color="#c44963", lw=2, ls="--")
    axes[1].text(lookback + 0.5, 4.2, "window L=24", color="#c44963", fontsize=8)
    axes[1].grid(axis="x", alpha=0.25)
    save(fig, "B3_reach_on_the_river.png")


def save(fig, name):
    p = FIG / name
    fig.savefig(p, dpi=150, bbox_inches="tight")
    try:
        fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight")
    except Exception:
        pass
    plt.close(fig)
    print("wrote", p.name)


if __name__ == "__main__":
    rows = summarise()
    print(json.dumps(rows, indent=1))
    figure()
    (RES / "structural_reach.json").write_text(json.dumps(rows, indent=2))
