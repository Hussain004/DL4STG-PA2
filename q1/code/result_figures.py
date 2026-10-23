"""Figures from the experiment results (reads results/*.json)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

PLOT = {
    "persistence": "#b0b0b0", "linear": "#7a7a7a", "temporal": "#ce791c",
    "graph": "#4774b5", "graphlag": "#228b73", "graphattn": "#c44963",
    "jointST": "#8d5aad", "globalattn": "#242424",
}
SHORT = {
    "persistence": "persistence", "linear": "ridge", "temporal": "temporal only",
    "graph": "graph MP", "graphlag": "graph + delay", "graphattn": "graph + attn",
    "jointST": "joint ST", "globalattn": "global attn",
}
NICE = {
    "persistence": "Persistence\n(no learning)",
    "linear": "Per-station ridge\n(univariate)",
    "temporal": "Temporal encoder only\n(no spatial path)",
    "graph": "Graph message passing\n(no delay learning)",
    "graphlag": "Graph + learned delay\n(proposed)",
    "graphattn": "Graph + delay attention",
    "jointST": "Joint ST convolution\n(on the row index)",
    "globalattn": "Global space-time\nattention",
}


def load(name):
    p = RES / f"{name}_summary.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())["summary"]


def save(fig, name):
    p = FIG / name
    fig.savefig(p, dpi=150, bbox_inches="tight")
    try:
        fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight")
    except Exception:
        pass
    plt.close(fig)
    print("wrote", p.name)


def fig_arch():
    s = load("arch")
    if not s:
        return
    s = sorted(s, key=lambda r: r.get("test_mae", 9e9))
    kinds = [r["kind"] for r in s]
    mae = [r.get("test_mae", np.nan) for r in s]
    sd = [r.get("test_mae_sd", 0.0) for r in s]
    npar = [r.get("n_params", 0) for r in s]
    flood = [r.get("test_mae_flood", np.nan) for r in s]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), layout="constrained",
                             gridspec_kw={"width_ratios": [1.5, 1]})
    x = np.arange(len(s))
    colors = [PLOT.get(k, "#888888") for k in kinds]
    axes[0].bar(x, mae, yerr=sd, capsize=3, color=colors)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([SHORT.get(k, k) for k in kinds], fontsize=9,
                            rotation=20, ha="right")
    axes[0].set_ylabel("test MAE (m), 1-hour horizon")
    axes[0].set_title("Forecasting accuracy by architecture\n(mean ± sd over 3 seeds)", fontsize=11)
    for i, v in enumerate(mae):
        axes[0].text(i, v + 0.002, f"{v:.4f}", ha="center", fontsize=7.5)
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].scatter(npar, mae, c=colors, s=60)
    for r in s:
        axes[1].annotate(SHORT.get(r["kind"], r["kind"]),
                         (r.get("n_params", 0), r.get("test_mae", 0)),
                         textcoords="offset points", xytext=(6, 4), fontsize=7.5)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("trainable parameters")
    axes[1].set_ylabel("test MAE (m)")
    axes[1].set_title("Accuracy vs capacity", fontsize=11)
    axes[1].grid(alpha=0.25)
    save(fig, "C1_architecture_comparison.png")


def fig_lookback():
    s = load("lookback")
    if not s:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), layout="constrained")
    Ls = sorted({r["config"]["lookback"] for r in s})
    for ax, metric, lab in ((axes[0], "test_mae", "test MAE (m)"),
                            (axes[1], "test_mae_flood", "test MAE on flood windows (m)")):
        vals, sds = [], []
        for L in Ls:
            rows = [r for r in s if r["config"]["lookback"] == L]
            vals.append(np.mean([r.get(metric, np.nan) for r in rows]))
            sds.append(np.std([r.get(metric, np.nan) for r in rows]))
        ax.errorbar(Ls, vals, yerr=sds, marker="o", capsize=3, color="#228b73")
        ax.set_xlabel("history window L (steps of 10 min)")
        ax.set_ylabel(lab)
        ax.grid(alpha=0.25)
    axes[0].axvline(24, color="#c44963", ls=":", lw=1.5)
    axes[0].text(24.6, axes[0].get_ylim()[1], " chosen L=24\n (physical: slowest transit + horizon)",
                 color="#c44963", fontsize=8.5, va="top")
    lo, hi = min(vals), max(vals)
    axes[0].set_title(f"Total spread over a factor of eight in L: "
                      f"{100*(hi-lo)/lo:.1f} % - the model is insensitive", fontsize=10)
    fig.suptitle("How far back must the model look?", fontsize=11)
    save(fig, "C2_lookback_ablation.png")


def fig_rounds():
    s = load("rounds")
    if not s:
        return
    Rs = sorted({r["config"]["rounds"] for r in s})
    fig, ax = plt.subplots(figsize=(6.4, 3.8), layout="constrained")
    vals = [np.mean([r["test_mae"] for r in s if r["config"]["rounds"] == R]) for R in Rs]
    sds = [np.std([r["test_mae"] for r in s if r["config"]["rounds"] == R]) for R in Rs]
    ax.errorbar(Rs, vals, yerr=sds, marker="o", capsize=3, color="#4774b5")
    ax.set_xlabel("number of spatial message-passing rounds")
    ax.set_ylabel("test MAE (m)")
    ax.set_title("Reach: how many rounds are needed?\n(the dam is 4 hops from S4)", fontsize=11)
    ax.grid(alpha=0.25)
    ax.axvline(4, color="#c44963", ls=":", lw=1.5)
    save(fig, "C3_rounds_ablation.png")


def fig_inputs():
    s = load("inputs")
    if not s:
        return
    names = []
    for r in s:
        g = r.get("group", r["kind"])
        if g not in names:
            names.append(g)
    vals = [np.mean([r["test_mae"] for r in s if r.get("group") == g]) for g in names]
    sds = [np.std([r["test_mae"] for r in s if r.get("group") == g]) for g in names]
    base = vals[0]
    fig, ax = plt.subplots(figsize=(8.4, 4.2), layout="constrained")
    y = np.arange(len(names))
    ax.barh(y, vals, xerr=sds, color=["#228b73"] + ["#c44963"] * (len(names) - 1),
            capsize=3)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("test MAE (m)")
    ax.set_title("Which information actually matters?\n"
                 "(removing any source costs accuracy; the proposed design uses them all)",
                 fontsize=11)
    for i, v in enumerate(vals):
        ax.text(v + 0.0006, i, f"{v:.4f}  ({100*(v-base)/base:+.1f} %)", va="center",
                fontsize=8.5)
    ax.grid(axis="x", alpha=0.25)
    save(fig, "C4_inputs_ablation.png")


def fig_robust():
    s = load("robust")
    if not s:
        return
    names = []
    for r in s:
        g = r.get("group", r["kind"])
        if g not in names:
            names.append(g)
    vals = [np.mean([r["test_mae"] for r in s if r.get("group") == g]) for g in names]
    sds = [np.std([r["test_mae"] for r in s if r.get("group") == g]) for g in names]
    fig, ax = plt.subplots(figsize=(8.8, 4.6), layout="constrained")
    y = np.arange(len(names))
    cols = []
    for g in names:
        if g.startswith("missing"):
            cols.append("#4774b5")
        elif g.startswith("late"):
            cols.append("#8d5aad")
        elif g.startswith("plan"):
            cols.append("#ce791c")
        else:
            cols.append("#c44963")
    ax.barh(y, vals, xerr=sds, color=cols, capsize=3)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("test MAE (m)")
    ax.set_title("Robustness: missing readings, late readings, plan error, unmonitored inflow",
                 fontsize=11)
    for i, v in enumerate(vals):
        ax.text(v + 0.0004, i, f"{v:.4f}", va="center", fontsize=8.5)
    ax.grid(axis="x", alpha=0.25)
    save(fig, "C5_robustness.png")


def fig_loss():
    s = load("loss")
    if not s:
        return
    names = []
    for r in s:
        g = r.get("group", r["kind"])
        if g not in names:
            names.append(g)
    fig, ax = plt.subplots(figsize=(7, 3.8), layout="constrained")
    x = np.arange(len(names))
    mae = [np.mean([r["test_mae"] for r in s if r.get("group") == g]) for g in names]
    fl = [np.mean([r.get("test_mae_flood", np.nan) for r in s if r.get("group") == g])
          for g in names]
    ax.bar(x - 0.2, mae, 0.4, label="overall MAE", color="#4774b5")
    ax.bar(x + 0.2, fl, 0.4, label="MAE on flood windows", color="#c44963")
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("test MAE (m)")
    ax.set_title("Training objective", fontsize=11)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    save(fig, "C6_loss_ablation.png")


def fig_horizon():
    s = load("arch")
    if not s:
        return
    kinds = ["persistence", "temporal", "graph", "graphlag", "graphattn"]
    fig, ax = plt.subplots(figsize=(6.6, 3.8), layout="constrained")
    for k in kinds:
        rows = [r for r in s if r["kind"] == k]
        if not rows:
            continue
        raw = load_raw("arch")
        rs = [r for r in raw if r["kind"] == k]
        hs = [np.mean([r["test"][f"mae_h{h}"] for r in rs]) for h in range(1, 7)]
        ax.plot(range(1, 7), hs, marker="o", label=SHORT.get(k, k),
                color=PLOT.get(k))
    ax.set_xlabel("forecast horizon (steps of 10 min)")
    ax.set_ylabel("test MAE (m)")
    ax.set_title("Error grows with the horizon", fontsize=11)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25)
    save(fig, "C7_error_by_horizon.png")


def fig_flood():
    s = load("flood")
    if not s:
        return
    groups = []
    for r in s:
        g = r.get("group", "")
        if g and g not in groups:
            groups.append(g)
    scales = []
    for g in groups:
        sc = g.split("(")[1].split(")")[0].split("x")[-1]
        if sc not in scales:
            scales.append(sc)
    kinds = ["temporal", "graph", "graphlag"]
    fig, ax = plt.subplots(figsize=(8.2, 4.0), layout="constrained")
    w = 0.26
    for i, k in enumerate(kinds):
        vals, sds = [], []
        for sc in scales:
            rows = [r for r in s if k in r.get("group", "") and sc in r.get("group", "")]
            vals.append(np.mean([r["test_mae"] for r in rows]) if rows else np.nan)
            sds.append(np.std([r["test_mae"] for r in rows]) if rows else 0.0)
        ax.bar(np.arange(len(scales)) + (i - 1) * w, vals, w, yerr=sds, capsize=3,
               color=PLOT.get(k), label=NICE.get(k, k).split("\n")[0])
    ax.set_xticks(np.arange(len(scales)))
    ax.set_xticklabels([f"travel ×{sc}" for sc in scales])
    ax.set_xlabel("river condition (travel-time scale)")
    ax.set_ylabel("test MAE (m)")
    ax.set_title("The spatial path pays more when the water moves faster\n"
                 "(shorter travel times bring the upstream influence inside the horizon)",
                 fontsize=11)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    save(fig, "C8_flood_river.png")


def fig_intervention():
    p = RES / "upstream_intervention.json"
    if not p.exists():
        return
    data = json.loads(p.read_text())
    kinds = [k for k in ("temporal", "graph", "graphlag", "graphattn") if k in data]
    fig, ax = plt.subplots(figsize=(7.4, 3.8), layout="constrained")
    x = np.arange(len(kinds))
    vals = [data[k]["mean_abs_forecast_change_m"] * 1000 for k in kinds]
    ax.bar(x, vals, color=[PLOT.get(k, "#888") for k in kinds])
    ax.set_xticks(x)
    ax.set_xticklabels([NICE.get(k, k).split("\n")[0] for k in kinds], fontsize=8.5)
    ax.set_ylabel("change in the S4 forecast (mm)")
    ax.set_title("Does the forecast actually use the upstream?\n"
                 "test-time intervention: every row except S4 is replaced by another window",
                 fontsize=11)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.06, f"{v:.2f} mm", ha="center", fontsize=8.5)
    ax.grid(axis="y", alpha=0.25)
    save(fig, "C9_upstream_intervention.png")


def fig_probe():
    p = RES / "diagnosis.json"
    if not p.exists():
        return
    data = json.loads(p.read_text())
    if "probe_normal" not in data:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), layout="constrained",
                             sharey=True)
    for ax, tag, title in ((axes[0], "probe_normal", "normal river"),
                            (axes[1], "probe_flood", "flood river")):
        if tag not in data:
            continue
        pr = data[tag]
        st = ["S1", "S2", "S3", "S4"]
        x = np.arange(4)
        ax.bar(x - 0.2, [pr[s]["own_mae"] * 1000 for s in st], 0.4,
               label="own history only", color="#b0b0b0")
        ax.bar(x + 0.2, [pr[s]["up_mae"] * 1000 for s in st], 0.4,
               label="+ upstream now", color="#228b73")
        ax.set_xticks(x); ax.set_xticklabels(st)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("1-hour-ahead MAE (mm)")
    axes[0].legend(frameon=False, fontsize=8.5)
    fig.suptitle("How much is the upstream worth? (ridge probe)", fontsize=11)
    save(fig, "C10_information_probe.png")


def load_raw(name):
    p = RES / f"{name}_raw.json"
    if not p.exists():
        return []
    return json.loads(p.read_text())


if __name__ == "__main__":
    fig_arch()
    fig_lookback()
    fig_rounds()
    fig_inputs()
    fig_robust()
    fig_loss()
    fig_horizon()
    fig_flood()
    fig_intervention()
    fig_probe()
    print("done")
