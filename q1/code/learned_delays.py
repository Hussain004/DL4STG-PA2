"""Learned-delay analysis: what delay did the model actually learn, and does it
match the hydrology?  (Runs on the Lab PC: it trains one model.)"""
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
    out = {}
    for seed in (0, 1, 2):
        res, model = rt.run("graphlag", ds, spec, steps=3000, seed=seed,
                            device="cuda", loss_kind="mse", eval_every=250,
                            patience=6)
        alpha = torch.softmax(model.lag_logits, dim=-1).detach().cpu().numpy()
        out[f"seed{seed}"] = {"edges": [list(e) for e in model.edges],
                              "alpha": alpha.tolist(),
                              "mean_lag_min": [float((a * np.arange(a.shape[0]) * 10).sum())
                                               for a in alpha],
                              "test_mae": res["test"]["mae"]}
        print(f"seed {seed}: test MAE {res['test']['mae']:.4f}", flush=True)
        for e, a in zip(model.edges, alpha):
            w = np.arange(a.shape[0]) * 10
            print(f"   {e[0]:>3s} → {e[1]:<3s}  mean lag {float((a*w).sum()):5.1f} min"
                  f"   mode {int(w[np.argmax(a)]):4d} min"
                  f"   mass at 0-2 steps {float(a[:3].sum()):.2f}", flush=True)
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "learned_delays.json").write_text(json.dumps(out, indent=2))
    print("wrote results/learned_delays.json")


if __name__ == "__main__":
    main()
