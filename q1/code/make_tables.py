"""Emit the result tables (markdown) from the experiment JSONs."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"

NICE = {
    "persistence": "persistence (no learning)",
    "linear": "per-station ridge (univariate)",
    "temporal": "temporal encoder only (no spatial path)",
    "graph": "graph message passing (same-time)",
    "graphlag": "graph + learned delay kernel",
    "graphattn": "graph + delay attention",
    "jointST": "joint ST convolution (row index)",
    "globalattn": "global space-time attention",
}


def load(name):
    p = RES / f"{name}_summary.json"
    return json.loads(p.read_text())["summary"] if p.exists() else None


def table_arch():
    s = load("arch")
    if not s:
        return ""
    rows = sorted(s, key=lambda r: r.get("test_mae", 9e9))
    out = ["| architecture | test MAE (m) | S1 | S2 | S3 | S4 | flood-window MAE | params |",
           "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append(
            f"| {NICE.get(r['kind'], r['kind'])} | **{r.get('test_mae', 0):.4f}** ± "
            f"{r.get('test_mae_sd', 0):.4f} | {r.get('test_mae_S1', 0):.4f} | "
            f"{r.get('test_mae_S2', 0):.4f} | {r.get('test_mae_S3', 0):.4f} | "
            f"{r.get('test_mae_S4', 0):.4f} | {r.get('test_mae_flood', 0):.4f} | "
            f"{r.get('n_params', 0):,} |")
    return "\n".join(out)


def table_generic(name, key=None, label=""):
    s = load(name)
    if not s:
        return ""
    groups = []
    for r in s:
        g = r.get("group", r["kind"])
        if g not in groups:
            groups.append(g)
    out = [f"| {label or 'setting'} | test MAE (m) | S1 | S2 | S3 | S4 |",
           "|---|---|---|---|---|---|"]
    for g in groups:
        rs = [r for r in s if r.get("group", r["kind"]) == g]
        def m(k):
            vals = []
            for r in rs:
                if k == "":
                    v = r.get("test_mae")
                else:
                    v = r.get("test_" + k)
                if isinstance(v, (int, float)):
                    vals.append(v)
            return sum(vals) / len(vals) if vals else float("nan")

        out.append(f"| {g} | **{m(''):.4f}** | {m('mae_S1'):.4f} | {m('mae_S2'):.4f} | "
                   f"{m('mae_S3'):.4f} | {m('mae_S4'):.4f} |")
    return "\n".join(out)


def table_flood():
    s = load("flood")
    if not s:
        return ""
    groups = []
    for r in s:
        g = r.get("group", "")
        if g and g not in groups:
            groups.append(g)
    scales = []
    for g in groups:
        sc = g.split("(")[1].split(")")[0]
        if sc not in scales:
            scales.append(sc)
    out = ["| river condition | temporal only | graph | graph + learned delay |",
           "|---|---|---|---|"]
    for sc in scales:
        vals = []
        for k in ("temporal", "graph", "graphlag"):
            rows = [r for r in s if k in r.get("group", "") and sc in r.get("group", "")]
            vals.append(sum(r.get("test_mae", float("nan")) for r in rows) / len(rows) if rows else float("nan"))
        out.append(f"| travel ×{sc} | {vals[0]:.4f} | {vals[1]:.4f} | {vals[2]:.4f} |")
    return "\n".join(out)


def table_intervention():
    p = RES / "upstream_intervention.json"
    if not p.exists():
        return ""
    d = json.loads(p.read_text())
    out = ["| model | |Δ forecast| at S4 (mm) | MAE vs the true S4 (mm) | MAE vs the swapped truth (mm) |",
           "|---|---|---|---|"]
    for k, v in d.items():
        if k == "reverse":
            continue
        out.append(f"| {NICE.get(k, k)} | {v['mean_abs_forecast_change_m']*1000:.2f} | "
                   f"{v['mae_vs_original_truth_m']*1000:.2f} | "
                   f"{v['mae_vs_swapped_truth_m']*1000:.2f} |")
    return "\n".join(out)


def table_reverse():
    p = RES / "upstream_intervention.json"
    if not p.exists():
        return ""
    d = json.loads(p.read_text())
    if "reverse" not in d:
        return ""
    out = ["| model | change in the **S1** forecast when the downstream rows are replaced (mm) |",
           "|---|---|"]
    for k, v in d["reverse"].items():
        out.append(f"| {NICE.get(k, k)} | {v['S1_change_m']*1000:.2f} |")
    return "\n".join(out)


def table_probe():
    p = RES / "diagnosis.json"
    if not p.exists():
        return ""
    d = json.loads(p.read_text())
    if "probe_normal" not in d:
        return ""
    out = ["| station | own history only (mm) | + upstream now (mm) | gain |",
           "|---|---|---|---|"]
    for s, v in d["probe_normal"].items():
        out.append(f"| {s} | {v['own_mae']*1000:.2f} | {v['up_mae']*1000:.2f} | "
                   f"**{100*v['improvement']:.1f} %** |")
    return "\n".join(out)


if __name__ == "__main__":
    print("### architecture\n" + table_arch())
    print("\n### lookback\n" + table_generic("lookback", label="history window L (steps)"))
    print("\n### rounds\n" + table_generic("rounds", label="spatial rounds"))
    print("\n### inputs\n" + table_generic("inputs", label="input variant"))
    print("\n### loss\n" + table_generic("loss", label="objective"))
    print("\n### robust\n" + table_generic("robust", label="degradation"))
    print("\n### flood\n" + table_flood())
    print("\n### intervention\n" + table_intervention())
    print("\n### reverse\n" + table_reverse())
    print("\n### probe\n" + table_probe())
