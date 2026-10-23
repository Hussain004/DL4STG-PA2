"""Diagnostic: why is the neural forecaster not beating persistence?"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import river_data as rd      # noqa: E402
import river_train as rt     # noqa: E402
import river_models as rm    # noqa: E402

torch.set_num_threads(8)
spec = rd.GridSpec(24, 6)
ds = rt.get_dataset(spec, sizes=(90000, 30000, 30000), tag="main")
scales, means = ds.scales()
Xtr, Ytr, ytr = rt.to_tensors(ds, "train")
Xva, Yva, yva = rt.to_tensors(ds, "val")


def persistence_val():
    p = yva.unsqueeze(-1).expand(-1, -1, spec.H).numpy()
    t = Yva.numpy()
    sc = scales.reshape(1, -1, 1); mu = means.reshape(1, -1, 1)
    return float(np.abs((p - t) * sc).mean())


print("persistence val MAE (m):", round(persistence_val(), 4), flush=True)

for kind, loss in [("temporal", "mse"), ("temporal", "wmse"),
                   ("graphlag", "mse"), ("graphlag", "wmse")]:
    for lr in (1e-3, 3e-3):
        out = rt.train(kind, ds, spec, steps=3000, lr=lr, loss_kind=loss,
                       eval_every=250, patience=99, seed=0, device="cuda")
        h = out["history"]
        best = min(x["val_mae"] for x in h)
        print(f"{kind:10s} loss={loss:5s} lr={lr:g}  best val MAE {best:.4f}  "
              f"final {h[-1]['val_mae']:.4f}  train loss {h[-1]['train_loss']:.5f}  "
              f"step {out['best_step']}", flush=True)
        print("   curve:", [round(x["val_mae"], 4) for x in h], flush=True)
