"""Analysis and figures for the Q1 design study.

Produces
  B1_river_system.png        the river graph, the grid layout, a simulated event
  B2_travel_times.png        measured calm vs flood travel times
  B3_reach.png               the receptive field of the chosen model on the river
  B4_learned_delays.png      learned per-edge delay kernels vs the hydrology
  B5_architecture.png        the proposed architecture diagram
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import river_data as rd          # noqa: E402
import river_models as rm        # noqa: E402
import river_sim as rs           # noqa: E402
import river_train as rt         # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
RES = ROOT / "results"
FIG.mkdir(parents=True, exist_ok=True)
RES.mkdir(parents=True, exist_ok=True)

PLOT = {
    "T": "#4774b5", "S-first": "#ce791c", "T-first": "#8d5aad",
    "ST": "#228b73", "A": "#c44963", "graph": "#4774b5",
    "graphlag": "#228b73", "graphattn": "#c44963", "temporal": "#ce791c",
    "jointST": "#8d5aad", "globalattn": "#242424", "linear": "#7a7a7a",
    "persistence": "#b0b0b0",
}


def save(fig, name):
    p = FIG / name
    fig.savefig(p, dpi=150, bbox_inches="tight")
    try:
        fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight")
    except Exception:
        pass
    plt.close(fig)
    print("wrote", p.name, flush=True)


# ----------------------------------------------------------------- B1 --------
def fig_river_system(spec, cfg):
    rng = np.random.default_rng(3)
    sim = rs.simulate(cfg, 4000, rng, burn_in=1000)
    lv = sim["levels"]

    fig = plt.figure(figsize=(13.5, 8.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.12, 1], hspace=0.42, wspace=0.16,
                          top=0.90, bottom=0.08, left=0.06, right=0.98)

    # --- the river graph ---------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    pos = {"Dam": (0.06, 0.62), "S1": (0.26, 0.62), "T1": (0.43, 0.90),
           "S2": (0.48, 0.58), "S3": (0.70, 0.58), "S4": (0.93, 0.58),
           "U": (0.70, 0.90), "R1": (0.43, 0.20), "R2": (0.74, 0.20)}
    edges = [("Dam", "S1"), ("S1", "S2"), ("T1", "S2"), ("S2", "S3"), ("S3", "S4"),
             ("U", "S3"), ("R1", "S2"), ("R2", "U")]
    for a, b in edges:
        x0, y0 = pos[a]; x1, y1 = pos[b]
        kw = dict(color="#444444", lw=1.8, arrowstyle="-|>", mutation_scale=14)
        if a == "U":
            kw.update(color="#c44963", ls="--")
        if a in ("R1", "R2"):
            kw.update(color="#4774b5", ls=":", lw=1.4)
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=kw)
    for name, (x, y) in pos.items():
        observed = name not in ("U", "R1", "R2")
        ax.scatter([x], [y], s=700, zorder=5,
                   facecolor="#dbe9f7" if observed else "#fbe3e8",
                   edgecolor="#2b6ca3" if observed else "#c44963", linewidth=2)
        ax.text(x, y, name, ha="center", va="center", fontsize=13,
                fontweight="bold" if observed else "normal",
                color="#12365c" if observed else "#8c2337")
    ax.text(0.02, 0.44, "reservoir", fontsize=12, color="#555")
    ax.text(0.43, 1.02, "monitored\ntributary", fontsize=11, color="#555", ha="center")
    ax.text(0.70, 1.02, "UNMONITORED\ntributary", fontsize=11, color="#c44963", ha="center")
    ax.text(0.60, 0.04, "rain gauges", fontsize=12, color="#4774b5", ha="center")
    ax.set_title("The river as a directed graph (arrows point downstream)",
                 fontsize=14, pad=14)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(0.0, 1.12); ax.axis("off")

    # --- a simulated event -------------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    t = slice(900, 1500)
    ax.plot(lv[t, 2], label="S2", color="#4774b5")
    ax.plot(lv[t, 3], label="S3", color="#ce791c")
    ax.plot(lv[t, 4], label="S4", color="#228b73")
    ax.plot(sim["release"][t] / 6.0 + 1.0, label="dam release (scaled)", color="#242424",
            ls="--", lw=1.2)
    ax2 = ax.twinx()
    ax2.plot(sim["rain1"][t] * 4, color="#9fc3e8", lw=1.0)
    ax2.plot(sim["rain2"][t] * 4, color="#f0c39a", lw=1.0)
    ax.set_xlabel("step (10 min)", fontsize=12); ax.set_ylabel("water level (m)", fontsize=12)
    ax2.set_ylabel("rain (mm/10 min)", color="#888888", fontsize=12)
    ax.set_title("A simulated stretch of river: a release change arrives downstream later",
                 fontsize=13)
    ax.legend(loc="upper left", fontsize=10, frameon=False, ncol=2)

    # --- the grid ----------------------------------------------------------
    ax = fig.add_subplot(gs[:, 1])
    X, Y, yn = rd.make_windows(rd.build_arrays(sim, cfg, np.random.default_rng(0)), spec)
    grid = X[0]
    panels = [(grid[:, :, 0].T, "level (m)", "RdBu_r"),
              (grid[:, :, 3].T, "release (m$^3$/s)", "viridis"),
              (grid[:, :, 2].T, "rain (mm/10 min)", "Blues")]
    for k, (mat, name, cm) in enumerate(panels):
        sub = ax.inset_axes([0.06 + 0.32 * k, 0.17, 0.25, 0.66])
        im = sub.imshow(mat, aspect="auto", cmap=cm)
        sub.axvline(spec.L - 0.5, color="#c44963", lw=2)
        sub.set_yticks(range(6)); sub.set_yticklabels(rd.ROWS, fontsize=10)
        sub.set_xlabel("column", fontsize=11, labelpad=1)
        sub.set_title(name, fontsize=12)
        sub.tick_params(labelsize=9)
        if k == 0:
            sub.set_ylabel("grid row", fontsize=10)
    ax.set_axis_off()
    ax.text(0.5, 0.95, "The grid the model sees:  24 observed columns + 6 forecast columns",
            ha="center", fontsize=14, transform=ax.transAxes)
    ax.text(0.5, 0.06, "red line = the forecast origin.  To the right of it the only\n"
                       "non-empty content is the announced plan in the Dam row.",
            ha="center", fontsize=11, color="#555", transform=ax.transAxes)
    save(fig, "B1_river_system.png")


# ----------------------------------------------------------------- B2 --------
def fig_travel_times():
    path = RES / "travel_times.json"
    if not path.exists():
        return
    data = json.loads(path.read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), layout="constrained")
    for ax, tag in zip(axes, ["calm", "flood"]):
        if tag not in data:
            continue
        lags = data[tag]["edge_lag_min"]
        names = [k for k in lags if lags[k] is not None]
        vals = [lags[k] for k in names]
        ax.bar(range(len(names)), vals, color="#4774b5" if tag == "calm" else "#c44963")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n.replace(">", " → ") for n in names], rotation=20)
        ax.set_ylabel("first-response travel time (min)")
        ax.set_title(f"{tag} conditions")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Travel time shortens when the river is in flood\n"
                 "(the S2→S3 reach: 70 min calm against 60 min in flood; the transport\n"
                 "parameter itself halves, but the storage response does not)",
                 fontsize=10)
    save(fig, "B2_travel_times.png")


# ----------------------------------------------------------------- B3/B4 -----
def train_probe_model(spec, ds, kind="graphlag", steps=3000, seed=0, device="cpu"):
    out = rt.train(kind, ds, spec, steps=steps, seed=seed, device=device,
                   loss_kind="mse", eval_every=250, patience=8)
    return out["model"]


def fig_reach_and_delays(spec, ds):
    """Gradient-based receptive field on the river + the learned delay kernels."""
    model = train_probe_model(spec, ds)
    model.eval()
    scales, means = ds.scales()
    Xte, Yte, yte = rt.to_tensors(ds, "test")
    x = Xte[:64].clone().requires_grad_(True)
    q = torch.zeros(4, dtype=torch.long)
    q[3] = 5                                    # S4
    out = model(x, yte[:64])
    sel = out[:, 3, 5]                          # S4 at horizon +60 min
    g = torch.autograd.grad(sel.sum(), x)[0].detach().numpy()   # [B, T, N, C]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4), layout="constrained",
                             width_ratios=[1.2, 1.2, 1])
    imp = np.abs(g).mean(0)                     # [T, N, C]
    lev = imp[:, :, 0]
    im = axes[0].imshow(lev, aspect="auto", cmap="magma",
                        vmin=0, vmax=max(1e-9, lev.max()))
    axes[0].set_yticks(range(6)); axes[0].set_yticklabels(rd.ROWS)
    axes[0].axvline(spec.L - 0.5, color="cyan", lw=2)
    axes[0].set_xlabel("time column"); axes[0].set_title(
        "Which input cells can move the S4 forecast?\nmean |gradient| over 64 test windows",
        fontsize=10)
    fig.colorbar(im, ax=axes[0], fraction=0.04)

    # per-source-lag profile for the Dam row and for S2/S3
    ax = axes[1]
    for row, lab, col in ((rd.R_DAM, "Dam release", "#242424"),
                          (rd.R_S2, "S2 level", "#4774b5"),
                          (rd.R_S3, "S3 level", "#ce791c")):
        prof = np.abs(g)[:, :, row, :].mean(axis=(0, 2))
        prof = prof[:spec.L]
        ax.plot(np.arange(spec.L)[::-1], prof, label=lab, color=col)
    ax.set_xlabel("steps before the forecast origin (0 = now)")
    ax.set_ylabel("mean |gradient|")
    ax.set_title("How far back does the S4 forecast look?", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)

    # learned delay kernels
    ax = axes[2]
    if hasattr(model, "lag_logits"):
        alpha = torch.softmax(model.lag_logits, dim=-1).detach().numpy()
        for e, (a, b) in enumerate(model.edges):
            ax.plot(np.arange(alpha.shape[1]) * 10, alpha[e], marker="o", ms=3,
                    label=f"{a} → {b}")
        ax.set_xlabel("learned lag (min)")
        ax.set_ylabel("mixing weight")
        ax.set_title("Learned transport delay per edge\n(the model's own idea of travel time)",
                     fontsize=10)
        ax.legend(fontsize=7, frameon=False)
        ax.grid(alpha=0.25)
    fig.suptitle("Reach of the chosen architecture on the river graph", fontsize=12)
    save(fig, "B3_reach_and_delays.png")
    return model


# ----------------------------------------------------------------- B5 --------
def fig_architecture(spec):
    fig, ax = plt.subplots(figsize=(14, 8.2))
    ax.set_xlim(0, 14); ax.set_ylim(0, 8.2); ax.axis("off")
    from matplotlib.patches import FancyBboxPatch

    def style(fc, ec):
        return dict(boxstyle="round,pad=0.18", fc=fc, ec=ec, lw=1.6)

    box = style("#eef3fa", "#3b6ea5")
    boxg = style("#e7f5ef", "#228b73")
    boxr = style("#fdf0e3", "#ce791c")
    boxy = style("#fbe9ee", "#c44963")

    def draw(x, y, w, h, title, sub, st, fs=9):
        ax.add_patch(FancyBboxPatch((x, y), w, h, **st))
        ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center",
                fontsize=fs, fontweight="bold")
        ax.text(x + w / 2, y + h * 0.30, sub, ha="center", va="center",
                fontsize=fs - 1.3, color="#334")

    draw(0.15, 5.75, 2.35, 1.4, "S1–S4 levels", "10-min, noisy,\nsometimes missing", box)
    draw(0.15, 4.25, 2.35, 1.2, "T1 level", "monitored tributary", box)
    draw(0.15, 2.75, 2.35, 1.2, "Dam release", "past: recorded\nfuture: announced plan", box)
    draw(0.15, 1.25, 2.35, 1.2, "Rain R1, R2", "R1 → S2, R2 → S3\n(tutorial assumption)", box)

    draw(3.25, 3.35, 2.5, 1.7, "GRID", "24 observed + 6 future columns\n"
                                       "6 rows × 4 channels\nB × 30 × 6 × 4", boxg)
    draw(6.45, 5.65, 3.0, 1.45, "Temporal encoder", "4 dilated convs (1,2,4,8)\n"
                                                "per place, shared weights\nB × 30 × 6 × d", boxr)
    draw(6.45, 3.45, 3.0, 1.7, "Graph space-time mixing", "4 rounds along the river graph\n"
                                                         "learned lag kernel per edge\n"
                                                         "B × 30 × 6 × d", boxg)
    draw(6.45, 1.45, 3.0, 1.45, "Read-out at t = now", "state of S1…S4\nB × 4 × d", box)
    draw(10.2, 3.45, 3.5, 1.6, "Forecast head", "ŷ = y_now + g(h)\n"
                                                "residual on the last level\nB × 4 × 6", boxy)
    draw(10.2, 1.45, 3.5, 1.45, "Objective", "weighted MSE in metres\n"
                                             "high water counts more\n"
                                             "per-station balanced", boxy)

    arrows = [((2.5, 6.4), (3.25, 4.6)), ((2.5, 4.8), (3.25, 4.3)),
              ((2.5, 3.3), (3.25, 4.0)), ((2.5, 1.8), (3.25, 3.7)),
              ((5.75, 4.6), (6.45, 6.3)), ((5.75, 4.2), (6.45, 4.4)),
              ((7.95, 6.3), (7.95, 5.2)), ((7.95, 4.3), (7.95, 3.3)),
              ((7.95, 2.3), (7.95, 2.0)), ((9.45, 4.3), (10.2, 4.3)),
              ((9.45, 2.2), (10.2, 2.2))]
    for a, b in arrows:
        ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", color="#555",
                                                        lw=1.5))
    ax.text(7.0, 7.55, "Proposed architecture:  per-place temporal encoding  →  graph "
                      "space-time mixing  →  residual read-out",
            ha="center", fontsize=12.5, fontweight="bold")
    ax.text(0.15, 0.55, "Everything entering the grid is known at issuance.  The six "
                        "horizon columns contain only the announced release plan; "
                        "no future measurement ever enters the model.",
            fontsize=8.8, color="#555")
    ax.text(10.2, 5.35, "output:\n6 forecast steps × 4 stations", fontsize=8.8, color="#333")
    save(fig, "B5_architecture.png")


if __name__ == "__main__":
    spec = rd.GridSpec(lookback=24, horizon=6)
    cfg = rs.RiverConfig()
    fig_river_system(spec, cfg)
    fig_travel_times()
    fig_architecture(spec)
    ds = rt.get_dataset(spec, sizes=(90000, 30000, 30000), tag="main")
    fig_reach_and_delays(spec, ds)
    print("done")
