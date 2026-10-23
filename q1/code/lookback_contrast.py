"""Contrast: how much does the history window matter WITHOUT a spatial path?

The graph model is insensitive to the lookback because the upstream's *current*
reading is the useful signal.  Without a spatial path the station's own past is
the only signal, so the window should matter much more.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import river_data as rd          # noqa: E402
import river_train as rt         # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"


def main():
    out = {}
    for kind in ("temporal", "graphlag"):
        for L in (6, 12, 24, 48):
            spec = rd.GridSpec(lookback=L, horizon=6)
            ds = rt.get_dataset(spec, sizes=(60000, 20000, 20000),
                                 tag=f"main_L{L}")
            maes = []
            for seed in (0, 1):
                r, _ = rt.run(kind, ds, spec, steps=2500, seed=seed,
                              device="cuda", loss_kind="mse", eval_every=250,
                              patience=6, rounds=4, lag_window=12)
                maes.append(r["test"]["mae"])
            out[f"{kind}|L={L}"] = {"mae": float(np.mean(maes)),
                                    "sd": float(np.std(maes)),
                                    "per_station": {s: float(np.mean(
                                        [r["test"][f"mae_{s}"] for r in [r]]))
                                        for s in ["S1", "S2", "S3", "S4"]}}
            print(f"{kind:10s} L={L:3d}  test MAE {np.mean(maes):.4f} "
                  f"± {np.std(maes):.4f}", flush=True)
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "lookback_contrast.json").write_text(json.dumps(out, indent=2))
    print("wrote results/lookback_contrast.json")


if __name__ == "__main__":
    main()
