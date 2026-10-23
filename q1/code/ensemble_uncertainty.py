"""Uncertainty: can the model's own disagreement stand in for its error?

Train the same architecture with three seeds, take the ensemble mean as the
forecast and the ensemble spread as the uncertainty, and ask whether the
spread predicts the error.  (Run on the Lab PC.)"""
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
    ds = rt.get_dataset(spec, sizes=(90000, 30000, 30000), tag="main")
    scales, means = ds.scales()
    Xte, Yte, yte = rt.to_tensors(ds, "test")
    thr = float(np.percentile(ds.splits["train"][1], 90))

    preds = []
    for seed in (0, 1, 2):
        res, model = rt.run("graphlag", ds, spec, steps=3000, seed=seed,
                            device="cuda", loss_kind="mse", eval_every=250,
                            patience=6)
        m, p = rt.evaluate(model, Xte, Yte, yte, scales, means, device="cuda",
                            high_threshold=thr)
        preds.append(p)
        print(f"seed {seed}: test MAE {m['mae']*1000:.2f} mm", flush=True)
    P = np.stack(preds)                     # [3, N, 4, 6] in metres
    mean = P.mean(0)
    spread = P.std(0)
    true = Yte.numpy() * np.asarray(scales).reshape(1, -1, 1) + \
        np.asarray(means).reshape(1, -1, 1)
    err = np.abs(mean - true)
    out = {
        "ensemble_mae_m": float(err.mean()),
        "single_model_mae_m": float(np.abs(P[0] - true).mean()),
        "corr_spread_error": float(np.corrcoef(spread.ravel(), err.ravel())[0, 1]),
        "mean_spread_m": float(spread.mean()),
        "mean_error_m": float(err.mean()),
        "coverage_2sigma": float((err <= 2 * spread + 1e-9).mean()),
        "coverage_3sigma": float((err <= 3 * spread + 1e-9).mean()),
    }
    # in flood windows
    flood = true >= thr
    out["flood_ensemble_mae_m"] = float(err[flood].mean())
    out["flood_mean_spread_m"] = float(spread[flood].mean())
    print(json.dumps(out, indent=2), flush=True)
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "ensemble_uncertainty.json").write_text(json.dumps(out, indent=2))
    print("wrote results/ensemble_uncertainty.json")


if __name__ == "__main__":
    main()
