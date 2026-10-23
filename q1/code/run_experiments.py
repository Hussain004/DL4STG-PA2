"""Experiment runner for the Q1 design study (runs on the Lab PC / GPU).

Usage:  python3 run_experiments.py <experiment> [--device cuda] [--quick]

Experiments
-----------
arch        architecture comparison (the headline table)
lookback    history-window length
rounds      number of spatial message-passing rounds
inputs      which information sources actually matter
robust      missing / late readings, plan-vs-actual error, unmonitored inflow
loss        training objective
final       the chosen configuration, several seeds, for the final table
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import river_data as rd          # noqa: E402
import river_models as rm        # noqa: E402
import river_sim as rs           # noqa: E402
import river_train as rt         # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

MAIN_KINDS = ["persistence", "linear", "temporal", "graph", "graphlag",
              "graphattn", "jointST", "globalattn"]


# --------------------------------------------------------------- utilities ---
def modify_grid(X, drop=(), no_plan=False, no_rain=False, L=24):
    """Ablate information sources from the grid (in place on a copy)."""
    X = X.copy()
    if no_rain:
        X[:, :, :, 2] = 0.0
    if "dam" in drop:
        X[:, :, rd.R_DAM, :] = 0.0
    if "T1" in drop:
        X[:, :, rd.R_T1, :] = 0.0
    if "plan" in drop or no_plan:
        X[:, L:, rd.R_DAM, 3] = 0.0
    return X


def make_dataset(spec, tag, cfg=None, sizes=(90000, 30000, 30000),
                 missing_prob=None, delay_prob=None, rebuild=False):
    cfg = cfg or rs.RiverConfig()
    return rt.get_dataset(spec, cfg=cfg, sizes=sizes, missing_prob=missing_prob,
                          delay_prob=delay_prob, tag=tag, rebuild=rebuild)


def apply_mods(ds, **mods):
    """Return a shallow dataset copy whose grids are modified."""
    if not mods:
        return ds
    new = object.__new__(rd.RiverDataset)
    new.cfg, new.spec = ds.cfg, ds.spec
    new.stats, new.release_stats, new.rain_stats = ds.stats, ds.release_stats, ds.rain_stats
    new.splits = {}
    for k, (X, Y, yn) in ds.splits.items():
        new.splits[k] = (modify_grid(X, L=new.spec.L, **mods), Y, yn)
    return new


def summarise(results):
    """Mean and sd over seeds for each configuration."""
    groups = {}
    for r in results:
        cfg = {k: v for k, v in (r.get("config") or {}).items()
               if k not in ("seed",)}
        key = json.dumps({"kind": r.get("kind"), "tag": r.get("tag"),
                          "group": r.get("group"), "config": cfg},
                         sort_keys=True, default=str)
        groups.setdefault(key, []).append(r)
    out = []
    for key, rs_ in groups.items():
        base = rs_[0]
        row = {"kind": base.get("kind"), "tag": base.get("tag"),
               "group": base.get("group"), "config": base.get("config"), "n": len(rs_),
               "n_params": base.get("n_params")}
        for m in ("mae", "rmse", "mae_flood", "rmse_flood"):
            vals = [r["test"][m] for r in rs_ if m in r["test"]]
            if vals:
                row[f"test_{m}"] = float(np.mean(vals))
                row[f"test_{m}_sd"] = float(np.std(vals))
        for s in ("S1", "S2", "S3", "S4"):
            vals = [r["test"][f"mae_{s}"] for r in rs_]
            row[f"test_mae_{s}"] = float(np.mean(vals))
        vals = [r["test"][f"mae_h{h}"] for r in rs_ for h in range(1, 7)]
        row["test_mae_h"] = float(np.mean(vals))
        vals = [r["val"]["mae"] for r in rs_]
        row["val_mae"] = float(np.mean(vals))
        vals = [r.get("seconds", 0.0) for r in rs_]
        row["seconds"] = float(np.mean(vals))
        out.append(row)
    return out


def save(name, payload):
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, default=str))
    print("wrote", path, flush=True)
    return path


# ------------------------------------------------------------- experiments ---
def exp_arch(args, spec, ds):
    res = []
    for kind in MAIN_KINDS:
        for seed in range(args.seeds):
            t0 = time.time()
            if kind == "persistence":
                r, _ = rt.run(kind, ds, spec)
            elif kind == "linear":
                r = rt.ridge_baseline(ds, spec)
            else:
                r, _ = rt.run(kind, ds, spec, seed=seed, steps=args.steps,
                              width=args.width, depth=args.depth, rounds=4,
                              lag_window=args.lag_window, loss_kind="mse",
                              device=args.device, eval_every=250, patience=6)
            r["group"] = kind
            res.append(r)
            print(f"  {kind:12s} seed {seed}  test MAE {r['test']['mae']:.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    return res


def exp_lookback(args, spec, ds):
    res = []
    for L in args.lookbacks:
        sp = rd.GridSpec(lookback=L, horizon=spec.H)
        d = make_dataset(sp, f"main_L{L}", sizes=(60000, 20000, 20000))
        for seed in range(args.seeds):
            r, _ = rt.run("graphlag", d, sp, seed=seed, steps=args.steps,
                          width=args.width, depth=args.depth, rounds=4,
                          lag_window=args.lag_window, loss_kind="mse",
                          device=args.device, eval_every=250, patience=6)
            r["group"] = f"L={L}"
            r["config"] = {"lookback": L}
            res.append(r)
            print(f"  L={L:3d} seed {seed}  test MAE {r['test']['mae']:.4f}", flush=True)
    return res


def exp_rounds(args, spec, ds):
    res = []
    for R in args.rounds_list:
        for seed in range(args.seeds):
            r, _ = rt.run("graphlag", ds, spec, seed=seed, steps=args.steps,
                          width=args.width, depth=args.depth, rounds=R,
                          lag_window=args.lag_window, loss_kind="mse",
                          device=args.device, eval_every=250, patience=6)
            r["group"] = f"rounds={R}"
            r["config"] = {"rounds": R}
            res.append(r)
            print(f"  rounds={R} seed {seed}  test MAE {r['test']['mae']:.4f}", flush=True)
    return res


def exp_inputs(args, spec, ds):
    variants = {
        "full": {},
        "no rain": {"no_rain": True},
        "no dam row": {"drop": ("dam",)},
        "no T1 row": {"drop": ("T1",)},
        "no announced plan": {"no_plan": True},
        "no dam, no plan": {"drop": ("dam",), "no_plan": True},
        "levels only (no rain/dam/plan)": {"no_rain": True, "drop": ("dam",),
                                           "no_plan": True},
    }
    res = []
    for name, mods in variants.items():
        d = apply_mods(ds, **mods)
        for seed in range(args.seeds):
            r, _ = rt.run("graphlag", d, spec, seed=seed, steps=args.steps,
                          width=args.width, depth=args.depth, rounds=4,
                          lag_window=args.lag_window, loss_kind="mse",
                          device=args.device, eval_every=250, patience=6)
            r["group"] = name
            r["config"] = {"variant": name}
            res.append(r)
            print(f"  {name:32s} seed {seed}  test MAE {r['test']['mae']:.4f}", flush=True)
    return res


def exp_robust(args, spec, ds):
    res = []
    cases = []
    for mp in (0.0, 0.05, 0.15):
        cases.append((f"missing={mp}", {"missing_prob": mp, "delay_prob": None}))
    for dp in (0.05, 0.12):
        cases.append((f"late={dp}", {"missing_prob": None, "delay_prob": dp}))
    for pe in (0.0, 3.5, 7.0):
        cases.append((f"plan_error={pe}", {"plan_error": pe}))
    for uf in (0.0, 0.45, 0.85):
        cases.append((f"unmonitored={uf}", {"unmonitored": uf}))

    for name, kw in cases:
        cfg = rs.RiverConfig()
        if "plan_error" in kw:
            cfg.dam_plan_error = kw["plan_error"]
        if "unmonitored" in kw:
            cfg.u_hidden_frac = kw["unmonitored"]
        tag = "robust_" + name.replace("=", "").replace(" ", "_")
        d = make_dataset(spec, tag, cfg=cfg, sizes=(60000, 20000, 20000),
                         missing_prob=kw.get("missing_prob"),
                         delay_prob=kw.get("delay_prob"))
        for seed in range(args.seeds):
            r, _ = rt.run("graphlag", d, spec, seed=seed, steps=args.steps,
                          width=args.width, depth=args.depth, rounds=4,
                          lag_window=args.lag_window, loss_kind="mse",
                          device=args.device, eval_every=250, patience=6)
            r["group"] = name
            r["config"] = {k: v for k, v in kw.items()}
            res.append(r)
            print(f"  {name:22s} seed {seed}  test MAE {r['test']['mae']:.4f}", flush=True)
    return res


def exp_loss(args, spec, ds):
    res = []
    for loss in ("mse", "mae", "huber", "wmse"):
        for seed in range(args.seeds):
            r, _ = rt.run("graphlag", ds, spec, seed=seed, steps=args.steps,
                          width=args.width, depth=args.depth, rounds=4,
                          lag_window=args.lag_window, loss_kind=loss,
                          device=args.device, eval_every=250, patience=6)
            r["group"] = loss
            r["config"] = {"loss": loss}
            res.append(r)
            print(f"  loss={loss:6s} seed {seed}  test MAE {r['test']['mae']:.4f}", flush=True)
    return res


def exp_flood(args, spec, ds):
    """Flood river: travel times compressed, so upstream influence arrives
    INSIDE the one-hour horizon.  Does the spatial path start to pay?"""
    res = []
    for scale, name in ((1.0, "normal river (travel x1.0)"),
                        (0.6, "rising river (travel x0.6)"),
                        (0.4, "flood river (travel x0.4)")):
        cfg = rs.RiverConfig(travel_scale=scale)
        d = make_dataset(spec, f"flood_{int(scale*100)}", cfg=cfg,
                         sizes=(60000, 20000, 20000))
        for kind in ("temporal", "graph", "graphlag"):
            for seed in range(args.seeds):
                r, _ = rt.run(kind, d, spec, seed=seed, steps=args.steps,
                              width=args.width, depth=args.depth, rounds=4,
                              lag_window=args.lag_window, loss_kind="mse",
                              device=args.device, eval_every=250, patience=6)
                r["group"] = f"{name} | {kind}"
                r["config"] = {"travel_scale": scale, "kind": kind}
                res.append(r)
                print(f"  {name:28s} {kind:10s} seed {seed}  test MAE "
                      f"{r['test']['mae']:.4f}", flush=True)
    return res


def exp_final(args, spec, ds):
    """The chosen architecture, several seeds, plus a persistence reference."""
    res = []
    for kind in ("graphlag", "graphattn", "graph"):
        for seed in range(args.seeds):
            r, _ = rt.run(kind, ds, spec, seed=seed, steps=args.steps,
                          width=args.width, depth=args.depth, rounds=4,
                          lag_window=args.lag_window, loss_kind="mse",
                          device=args.device, eval_every=250, patience=6)
            r["group"] = kind
            res.append(r)
            print(f"  {kind:12s} seed {seed}  test MAE {r['test']['mae']:.4f}", flush=True)
    return res


EXPERIMENTS = {
    "arch": exp_arch, "lookback": exp_lookback, "rounds": exp_rounds,
    "inputs": exp_inputs, "robust": exp_robust, "loss": exp_loss,
    "flood": exp_flood, "final": exp_final,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment", choices=list(EXPERIMENTS))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--lag-window", type=int, default=12)
    ap.add_argument("--lookback", type=int, default=24)
    ap.add_argument("--lookbacks", type=int, nargs="+", default=[6, 12, 18, 24, 36, 48])
    ap.add_argument("--rounds-list", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6])
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.steps = max(200, args.steps // 6)
        args.seeds = 1
        args.lookbacks = [6, 24, 48]
        args.rounds_list = [0, 2, 4, 6]
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"
    torch.set_num_threads(8)

    spec = rd.GridSpec(lookback=args.lookback, horizon=6)
    t0 = time.time()
    ds = make_dataset(spec, "main")
    print(f"dataset ready in {time.time()-t0:.0f}s  "
          f"train={ds.splits['train'][0].shape}", flush=True)

    fn = EXPERIMENTS[args.experiment]
    res = fn(args, spec, ds)
    summary = summarise(res)
    save(f"{args.experiment}_summary", {"args": vars(args), "summary": summary})
    save(f"{args.experiment}_raw", res)
    print(json.dumps(summary, indent=1, default=str)[:4000])


if __name__ == "__main__":
    main()
