"""Data panels of report Figure 1 (the river graph itself is drawn in TikZ).

Writes figures/fig_river_series.pdf and figures/fig_river_grid.pdf.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

sys.path.insert(0, str(Path(__file__).resolve().parent))
import river_data as rd   # noqa: E402
import river_sim as rs    # noqa: E402

FIG = Path(__file__).resolve().parents[2] / "figures"
if any("Google Sans" in f.name for f in font_manager.fontManager.ttflist):
    plt.rcParams["font.family"] = "Google Sans"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42})

spec = rd.GridSpec(lookback=24, horizon=6)
cfg = rs.RiverConfig()
sim = rs.simulate(cfg, 4000, np.random.default_rng(3), burn_in=1000)
lv = sim["levels"]

# --- simulated stretch: levels on top, rain + release below ---------------------
t = slice(900, 1500)
x = np.arange(t.stop - t.start)
fig, (a, b) = plt.subplots(2, 1, figsize=(4.0, 2.3), sharex=True,
                           gridspec_kw={"height_ratios": [2.2, 1], "hspace": 0.12})
for col, c, lab in ((2, "#2b6ca3", "S2"), (3, "#ce791c", "S3"), (4, "#228b73", "S4")):
    a.plot(x, lv[t, col], color=c, lw=1.2, label=lab)
a.set_ylabel("water level (m)")
a.legend(loc="upper right", ncol=3, frameon=False, fontsize=8, handlelength=1.2)
a.grid(alpha=0.25)
b.fill_between(x, sim["release"][t], color="#444444", alpha=0.25, step="mid", lw=0)
b.plot(x, sim["release"][t], color="#444444", lw=1.1, label="Dam release (m$^3$/s)")
b2 = b.twinx()
b2.spines["right"].set_visible(True)
b2.plot(x, sim["rain1"][t], color="#4774b5", lw=0.7, alpha=0.8, label="R1")
b2.plot(x, sim["rain2"][t], color="#8d5aad", lw=0.7, alpha=0.8, label="R2")
b2.set_ylabel("rain (mm/10 min)", color="#4774b5")
b.set_ylabel("release", color="#444444")
b.set_xlabel("step (10 min)")
b.grid(alpha=0.25)
fig.savefig(FIG / "fig_river_series.pdf", bbox_inches="tight")
plt.close(fig)

# --- the grid: only the cells that actually exist are coloured ---------------------
X, _, _ = rd.make_windows(rd.build_arrays(sim, cfg, np.random.default_rng(0)), spec)
g = X[0]                                    # [30, 6, 4] = time, place, channel
L = spec.L
seen = (g[:, :, 1] > 0) & (np.arange(30)[:, None] < L) & (np.arange(6)[None, :] != rd.R_DAM)
mats = [
    (np.where(seen, g[:, :, 0], np.nan), "level (m)", "RdBu_r"),
    (np.where(np.arange(6)[None, :] == rd.R_DAM, g[:, :, 3], np.nan), "release (m$^3$/s)", "viridis"),
    (np.where(np.isin(np.arange(6), [rd.R_S2, rd.R_S3])[None, :] & (np.arange(30)[:, None] < L),
              g[:, :, 2], np.nan), "rain (mm/10 min)", "Blues"),
]
fig, axes = plt.subplots(1, 3, figsize=(3.3, 2.3), sharey=True,
                         gridspec_kw={"wspace": 0.08})
for k, (ax, (m, name, cm)) in enumerate(zip(axes, mats)):
    cmap = plt.get_cmap(cm).copy()
    cmap.set_bad("#e3e3e3")
    ax.imshow(m.T, aspect="auto", cmap=cmap, interpolation="nearest")
    ax.axvline(L - 0.5, color="#c44963", lw=1.6)
    ax.set_title(name, fontsize=7.5)
    ax.set_xticks([0, L - 1])
    ax.set_xticklabels(["0", "23"], fontsize=7.5)
    ax.set_xlabel("column", fontsize=7.5, labelpad=1)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
axes[0].set_yticks(range(6))
axes[0].set_yticklabels(rd.ROWS, fontsize=8.5)
fig.savefig(FIG / "fig_river_grid.pdf", bbox_inches="tight")
plt.close(fig)
print("wrote fig_river_series.pdf, fig_river_grid.pdf")
