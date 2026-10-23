"""Why does the spatial path not pay?  Three hypotheses, one experiment each.

H1  The shared MSE in metres lets a noisy station (S2) dominate the gradient,
    so the quiet station (S4) - where the upstream lead matters most - is
    under-fitted.                      -> test: per-station normalised loss.
H2  The models are simply under-trained. -> test: more steps.
H3  The upstream is genuinely useless at a one-hour horizon.
    -> test: the linear probe (own history vs own history + upstream now).
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
ST = ["S1", "S2", "S3", "S4"]


def linear_probe(travel_scale=1.0):
    """Ridge probe: how much does the upstream's CURRENT level add?"""
    import river_sim as rs
    cfg = rs.RiverConfig(travel_scale=travel_scale)
    rng = np.random.default_rng(0)
    o = rs.simulate(cfg, 60000, rng, burn_in=300)
    lv = o["levels"]
    HYD = {"S1": 0, "T1": 1, "S2": 2, "S3": 3, "S4": 4}
    UP = {"S1": [], "S2": ["S1"], "S3": ["S2", "T1"], "S4": ["S3"]}

    def fit(target, extra, L=24, H=6):
        idx = np.arange(L, len(lv) - H)
        A = np.stack([lv[t - np.arange(L), HYD[target]] for t in idx])
        for e in extra:
            A = np.concatenate([A, lv[idx, HYD[e]][:, None]], 1)
        A = np.concatenate([A, np.ones((len(A), 1))], 1)
        Y = np.stack([lv[t + H - 1, HYD[target]] - lv[t - 1, HYD[target]] for t in idx])
        W = np.linalg.solve(A.T @ A + 1e-3 * np.eye(A.shape[1]), A.T @ Y)
        pred = A @ W
        return float(np.abs(pred - Y).mean()), float(1 - np.var(Y - pred) / np.var(Y))

    out = {}
    for s in ST:
        own = fit(s, [])
        up = fit(s, UP[s])
        out[s] = {"own_mae": own[0], "own_r2": own[1],
                  "up_mae": up[0], "up_r2": up[1],
                  "improvement": 1 - up[0] / own[0]}
    return out


def main():
    spec = rd.GridSpec(lookback=24, horizon=6)
    ds = rt.get_dataset(spec, sizes=(90000, 30000, 30000), tag="main")
    out = {}

    print("H3: linear probe (ridge) - own history vs own history + upstream now",
          flush=True)
    out["probe_normal"] = linear_probe(1.0)
    out["probe_flood"] = linear_probe(0.4)
    for tag in ("normal", "flood"):
        print(f"  {tag}:")
        for s, v in out[f"probe_{tag}"].items():
            print(f"    {s}: own {v['own_mae']*1000:6.2f} mm (R2 {v['own_r2']:.3f})"
                  f"   +upstream {v['up_mae']*1000:6.2f} mm (R2 {v['up_r2']:.3f})"
                  f"   -> {100*v['improvement']:5.1f} % better", flush=True)

    print("\nH1/H2: neural models with different losses and budgets", flush=True)
    for kind in ("temporal", "graphlag"):
        for loss in ("mse", "mse_norm"):
            for steps in (3000, 8000):
                r, _ = rt.run(kind, ds, spec, seed=0, device="cuda", steps=steps,
                              loss_kind=loss, eval_every=250, patience=12)
                key = f"{kind}|{loss}|{steps}"
                out[key] = {k: r["test"][k] for k in r["test"] if isinstance(r["test"][k], float)}
                print(f"  {key:26s} mae {r['test']['mae']:.4f}  "
                      + "  ".join(f"{s} {r['test'][f'mae_{s}']:.4f}" for s in ST)
                      + f"   (best step {r['best_step']})", flush=True)

    RES.mkdir(parents=True, exist_ok=True)
    (RES / "diagnosis.json").write_text(json.dumps(out, indent=2))
    print("wrote results/diagnosis.json")


if __name__ == "__main__":
    main()
