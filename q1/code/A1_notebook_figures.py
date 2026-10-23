"""A1: Reproduce the companion-notebook experiments and save every figure.

We do not train anything here: the bundled checkpoints are used as-is (CPU).
Purpose: turn the notebook's qualitative claims into numbers and pictures that
can be cited in the Q1 report.

Bundled checkpoints available (3 seeds each):
    moving : T, S-first-p3
    relay  : T, ST, A
Anything else is skipped and recorded as "not bundled".
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from assignment2 import visuals                                   # noqa: E402
from assignment2.evaluation import (load_model, mae, predict,     # noqa: E402
                                     results, score_summary)
from assignment2.models import receptive_field                    # noqa: E402
from assignment2.worlds import (generate, moving_pair, relay_pair)  # noqa: E402

FIG = ROOT / "investigation" / "figures"
RES = ROOT / "investigation" / "results"
FIG.mkdir(parents=True, exist_ok=True)
RES.mkdir(parents=True, exist_ok=True)

torch.set_num_threads(4)
summary = {}


def save(fig, name):
    path = FIG / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path.name)
    return path


def kinds_of(arch, depth=3):
    if arch in ("S-first", "T-first"):
        return (["S", "T"] if arch == "S-first" else ["T", "S"]) * depth
    return [arch] * depth


def available(world, arch):
    """Return the 3 seeds of an architecture for a world, or None if not bundled."""
    key = f"{arch}-p3" if arch in ("S-first", "T-first") else arch
    try:
        return [load_model(world, key, seed=s) for s in (0, 1, 2)]
    except FileNotFoundError:
        return None


# NOTE: `load_model` is lru_cached, and the bundled notebook helpers call it from
# inside `torch.inference_mode()`, which would freeze the cached parameters as
# inference tensors and break later autograd checks. Pre-load every model we
# need outside inference mode, before any helper runs.
for _world, _arch in [("moving", "T"), ("moving", "S-first"), ("relay", "ST"), ("relay", "A")]:
    available(_world, _arch)


# ---------------------------------------------------------------- Part 1 ----
first, second = moving_pair()
fig = visuals.identical_histories(first, second)
save(fig, "A1_identical_histories.png")

summary["part1_targets"] = {
    "target_example_1": float(first.y[0, 8]),
    "target_example_2": float(second.y[0, 8]),
    "sensor8_histories_identical": bool(torch.equal(first.x[0, :, 8], second.x[0, :, 8])),
    "sensor7_latest": [float(first.x[0, -1, 7, 0]), float(second.x[0, -1, 7, 0])],
}

paired = {}
for arch in ("T", "S-first", "ST", "A"):
    models = available("moving", arch)
    if models is None:
        paired[arch] = "not bundled"
        continue
    shifts = []
    for model in models:
        with torch.inference_mode():
            a = float(model(first.x)[0, 8])
            b = float(model(second.x)[0, 8])
        shifts.append(b - a)
    with torch.inference_mode():
        base = float(models[0](first.x)[0, 8])
    paired[arch] = {"shift_mean": float(np.mean(shifts)), "shift_sd": float(np.std(shifts)),
                    "example1_forecast": base}
summary["part1_forecast_shift"] = paired
summary["part1_required_shift"] = float(second.y[0, 8] - first.y[0, 8])

fig = visuals.moving_pair_predictions(first, second)
save(fig, "A2_moving_pair_predictions.png")

# ---------------------------------------------------------------- Part 2 ----
manifest_scores = {}
for world in ("moving", "relay"):
    for name in results()["results"][world]:
        manifest_scores[f"{world}/{name}"] = list(score_summary(world, name))
summary["bundled_scores"] = manifest_scores
summary["bundled_baselines"] = results()["baselines"]

reach = {}
for arch in ("T", "S-first", "T-first", "ST", "A"):
    for depth in (1, 2, 3, 4, 5, 6):
        kinds = kinds_of(arch, depth)
        mask = receptive_field(kinds, 7, 8)
        sensors = np.flatnonzero(mask.any(axis=0))
        times = np.flatnonzero(mask.any(axis=1))
        reach[f"{arch}-d{depth}"] = {
            "n_sensors_reachable": int(len(sensors)),
            "sensor_span": [int(sensors[0]), int(sensors[-1])] if len(sensors) else [],
            "n_times_reachable": int(len(times)),
            "time_span_back": [int(times[0]), int(times[-1])] if len(times) else [],
            "area": int(mask.sum()),
            "kinds": "".join(kinds),
        }
summary["reach"] = reach

fig = visuals.field_view("ST", 3, 8)
save(fig, "A3_receptive_field_ST_d3.png")
fig = visuals.field_view("S-first", 3, 8)
save(fig, "A3_receptive_field_Sfirst_d3.png")
fig = visuals.field_view("T", 4, 8)
save(fig, "A3_receptive_field_T_d4.png")

# ---------------------------------------------------------------- Part 3 ----
fig = visuals.relay_scores()
save(fig, "A4_relay_scores.png")

relay = {}
for arch in ("T", "S-first", "ST", "A"):
    try:
        relay[arch] = {"near": list(score_summary("relay", arch, case="near")),
                       "far": list(score_summary("relay", arch, case="far"))}
    except KeyError:
        relay[arch] = "not in manifest"
summary["relay_near_far"] = relay

# Independent re-implementation of the near/far split, checked against the manifest.
def relay_split_mae(arch, pairs=3):
    """Reproduce the manifest's near/far MAE for a bundled relay model."""
    from assignment2.worlds import generate
    ev = results()["evaluation"]
    b = generate("relay", ev["test_size"], ev["test_seed"])
    src = torch.as_tensor(b.meta["source"])
    qry = torch.as_tensor(b.meta["query"])
    dist = (qry - src).abs()
    near = torch.zeros_like(b.mask)
    far = torch.zeros_like(b.mask)
    rows = torch.arange(b.x.shape[0])
    near[rows, qry] = ((dist >= 1) & (dist <= 3)).float()
    far[rows, qry] = (dist >= 4).float()
    out = []
    for seed in (0, 1, 2):
        try:
            model = load_model("relay", arch, pairs, seed)
        except FileNotFoundError:
            return "not bundled"
        with torch.inference_mode():
            p = model(b.x)
        out.append((float(mae(p, b.y, near)), float(mae(p, b.y, far))))
    return {"near_mean": float(np.mean([o[0] for o in out])),
            "near_sd": float(np.std([o[0] for o in out])),
            "far_mean": float(np.mean([o[1] for o in out])),
            "far_sd": float(np.std([o[1] for o in out]))}


relay_recheck = {a: relay_split_mae(a) for a in ("T", "ST", "A")}
summary["relay_near_far_recheck"] = relay_recheck

rf, rs = relay_pair(distance=12)
fig = visuals.relay_pair_predictions(rf, rs, int(rf.meta["query"][0]))
save(fig, "A5_relay_pair_distance12.png")

# Gradient support: prove ST literally cannot see beyond its reach, attention can.
grad_report = {}
for arch in ("T", "S-first", "ST", "A"):
    models = available("relay", arch)
    if models is None:
        continue
    model = models[0]
    x = rf.x.clone().requires_grad_(True)
    q = int(rf.meta["query"][0])
    g = torch.autograd.grad(model(x)[0, q], x)[0][0, :, :, 0].detach().numpy()
    support = receptive_field(model.kinds, 7, q)
    outside = np.abs(g[~support]).max() if (~support).any() else 0.0
    grad_report[arch] = {
        "n_gradient_nonzero": int((np.abs(g) > 1e-12).sum()),
        "max_grad_outside_structural_reach": float(outside),
        "total_abs_gradient": float(np.abs(g).sum()),
    }
summary["gradient_support"] = grad_report

# Attention weights: which block/head locks onto the source?
att = {}
for seed in (0, 1, 2):
    grid = visuals.source_attention(seed)
    att[f"seed{seed}"] = grid.tolist()
    fig = visuals.attention_view(generate("relay", 1, 23, distance=8), 0, 0, seed)
    save(fig, f"A6_attention_view_seed{seed}.png")
summary["source_attention_grid"] = att

# ------------------------------------------------------------------ costs ---
def costs(L, d):
    return {"attention_2Ld2": 2 * L * L * d, "st_9Ld2": 9 * L * d * d,
            "ratio_attention_over_st": (2 * L * L * d) / (9 * L * d * d)}


cost_table = {}
for L, d, tag in ((8 * 16, 32, "notebook toy: L=H*N=8*16=128, d=32"),
                   (24 * 6, 64, "river grid (6 rows): L=H*N=24*6=144, d=64"),
                   (24 * 8, 64, "river grid (8 rows, +R1,R2): L=24*8=192, d=64"),
                   (24 * 12, 64, "larger grid (12 rows): L=24*12=288, d=64")):
    cost_table[tag] = costs(L, d)
summary["cost_table"] = cost_table

fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), layout="constrained")
Ls = np.arange(16, 400, 8)
for d, color in ((32, "#4774b5"), (64, "#ce791c"), (128, "#228b73")):
    axes[0].plot(Ls, 2 * Ls**2 * d, color=color, label=f"attention, d={d}")
    axes[1].plot(Ls, 9 * Ls * d**2, color=color, label=f"joint ST, d={d}")
axes[0].set(xlabel="tokens L = H x N", ylabel="mixing cost", title="Attention: 2 L^2 d")
axes[1].set(xlabel="tokens L = H x N", ylabel="mixing cost", title="Joint ST: 9 L d^2")
for ax in axes:
    ax.set_yscale("log"); ax.grid(alpha=.2); ax.legend(frameon=False)
fig.suptitle("Cost of a global path (notebook cell 22 formulas)", fontsize=11)
save(fig, "A7_cost_attention_vs_st.png")

summary["evaluation_protocol"] = results()["evaluation"]
summary["training_protocol"] = results()["training"]
summary["world_definition"] = results()["world_definition"]

(RES / "A1_notebook_findings.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2)[:4000])
print("done")
