"""The river version of the notebook's Part 1: does the forecast actually USE
the upstream stations?

Test-time intervention.  We keep the target station's own history (and its
target) and replace every OTHER row of the grid with the rows of a different
window.  A model that only looks at its own station cannot react at all; a
model with an upstream path must change its forecast, and should move it
towards the truth of the window it was actually given.
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


def main():
    spec = rd.GridSpec(lookback=24, horizon=6)
    ds = rt.get_dataset(spec, sizes=(60000, 20000, 20000), tag="main")
    scales, means = ds.scales()
    Xte, Yte, yte = rt.to_tensors(ds, "test")
    n = min(4000, len(Xte))
    X = Xte[:n].clone()
    Y = Yte[:n].numpy()
    yn = yte[:n].clone()

    # a derangement of the batch: row j takes the upstream rows of window j+1
    perm = torch.roll(torch.arange(n), 1)
    Xs = X.clone()
    for row in (rd.R_DAM, rd.R_S1, rd.R_T1, rd.R_S2, rd.R_S3):
        Xs[:, :, row] = X[perm][:, :, row]
    # NOTE: the forecast station S4 keeps its own history and its own target
    # the target station keeps its own history and its own "now" level
    Ys = Y[perm.numpy()]

    out = {}
    for kind in ("temporal", "graph", "graphlag", "graphattn", "jointST",
                 "globalattn"):
        res, model = rt.run(kind, ds, spec, steps=3000, seed=0, device="cuda",
                            loss_kind="mse", eval_every=250, patience=6)
        model.eval()
        with torch.no_grad():
            p0 = model(X.cuda(), yn.cuda()).cpu().numpy()
            p1 = model(Xs.cuda(), yn.cuda()).cpu().numpy()
        sc = np.asarray(scales).reshape(1, 4, 1)
        mu = np.asarray(means).reshape(1, 4, 1)
        p0m, p1m = p0 * sc + mu, p1 * sc + mu
        Ym = Y * sc + mu
        Ysm = Ys * sc + mu
        # S4 only (index 3) is the station whose upstream we swapped
        out[kind] = {
            "mean_abs_forecast_change_m": float(np.abs(p1m[:, 3] - p0m[:, 3]).mean()),
            "max_abs_forecast_change_m": float(np.abs(p1m[:, 3] - p0m[:, 3]).max()),
            "mae_vs_original_truth_m": float(np.abs(p0m[:, 3] - Ym[:, 3]).mean()),
            "mae_vs_swapped_truth_m": float(np.abs(p1m[:, 3] - Ysm[:, 3]).mean()),
            "mae_allstations_change_m": float(np.abs(p1m - p0m).mean()),
        }
        print(f"{kind:12s} |Δ forecast| at S4 = {out[kind]['mean_abs_forecast_change_m']*1000:7.2f} mm"
              f"   MAE vs original {out[kind]['mae_vs_original_truth_m']*1000:6.2f} mm"
              f"   MAE vs swapped {out[kind]['mae_vs_swapped_truth_m']*1000:6.2f} mm", flush=True)

    # --- the reverse test: does an UPSTREAM station use the DOWNSTREAM ones?
    # Water does not flow upstream, so a physically honest model must not.
    Xd = X.clone()
    for row in (rd.R_S2, rd.R_S3, rd.R_S4):
        Xd[:, :, row] = X[perm][:, :, row]
    print("\nreverse test: the DOWNSTREAM rows are replaced; the S1 forecast "
          "must not move", flush=True)
    out["reverse"] = {}
    for kind in ("temporal", "graph", "graphlag", "graphattn", "jointST",
                 "globalattn"):
        res, model = rt.run(kind, ds, spec, steps=3000, seed=0, device="cuda",
                            loss_kind="mse", eval_every=250, patience=6)
        model.eval()
        with torch.no_grad():
            p0 = model(X.cuda(), yn.cuda()).cpu().numpy()
            p1 = model(Xd.cuda(), yn.cuda()).cpu().numpy()
        sc = np.asarray(scales).reshape(1, 4, 1)
        mu = np.asarray(means).reshape(1, 4, 1)
        d = np.abs((p1 - p0) * sc).mean(axis=(0, 2))
        out["reverse"][kind] = {"per_station_change_m": [float(v) for v in d],
                                "S1_change_m": float(d[0])}
        print(f"  {kind:12s} |Δ forecast| per station (mm): "
              + "  ".join(f"{v*1000:6.2f}" for v in d), flush=True)

    RES.mkdir(parents=True, exist_ok=True)
    (RES / "upstream_intervention.json").write_text(json.dumps(out, indent=2))
    print("wrote results/upstream_intervention.json")


if __name__ == "__main__":
    main()
