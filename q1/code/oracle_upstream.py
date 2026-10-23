"""How much could upstream information possibly buy?

We give the model ORACLE access to the true future levels of ONE upstream
station during the forecast hour and re-train, then read the error of the
station that sits downstream of it.  The gap to the honest model is the value
of that upstream path - an upper bound on what any architecture could extract
from it.

Run on the Lab PC:  python3 scripts/oracle_upstream.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import river_data as rd          # noqa: E402
import river_train as rt         # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
STATIONS = ["S1", "S2", "S3", "S4"]


def oracle_copy(ds, spec, upstream_idx):
    """Copy of the dataset where the horizon columns of station `upstream_idx`
    carry its TRUE future levels (and are flagged as seen)."""
    new = object.__new__(rd.RiverDataset)
    new.cfg, new.spec = ds.cfg, ds.spec
    new.stats, new.release_stats, new.rain_stats = ds.stats, ds.release_stats, ds.rain_stats
    new.splits = {}
    row = rd.TARGET_ROWS[upstream_idx]
    for name, (X, Y, yn) in ds.splits.items():
        Xo = X.copy()
        Xo[:, spec.L:, row, 0] = Y[:, upstream_idx, :]
        Xo[:, spec.L:, row, 1] = 1.0
        new.splits[name] = (Xo, Y, yn)
    return new


def main():
    spec = rd.GridSpec(lookback=24, horizon=6)
    ds = rt.get_dataset(spec, sizes=(60000, 20000, 20000), tag="main")
    out = {}

    print("baseline (no oracle)", flush=True)
    r, _ = rt.run("graphlag", ds, spec, steps=3000, seed=0, device="cuda",
                  loss_kind="mse", eval_every=250, patience=6)
    base = {s: r["test"][f"mae_{s}"] for s in STATIONS}
    out["baseline"] = base
    print("   " + "  ".join(f"{s} {base[s]:.4f}" for s in STATIONS), flush=True)

    for up in range(3):                     # oracle on S1 -> S2, S2 -> S3, S3 -> S4
        d = oracle_copy(ds, spec, up)
        r, _ = rt.run("graphlag", d, spec, steps=3000, seed=0, device="cuda",
                      loss_kind="mse", eval_every=250, patience=6)
        tgt = up + 1
        out[f"oracle_{STATIONS[up]}_for_{STATIONS[tgt]}"] = {
            "mae_target": r["test"][f"mae_{STATIONS[tgt]}"],
            "mae_all": {s: r["test"][f"mae_{s}"] for s in STATIONS},
        }
        print(f"oracle {STATIONS[up]} revealed  ->  {STATIONS[tgt]} MAE "
              f"{r['test'][f'mae_{STATIONS[tgt]}']:.4f}  (baseline "
              f"{base[STATIONS[tgt]]:.4f})", flush=True)

    RES.mkdir(parents=True, exist_ok=True)
    (RES / "upstream_value.json").write_text(json.dumps(out, indent=2))
    print("wrote results/upstream_value.json")


if __name__ == "__main__":
    main()
