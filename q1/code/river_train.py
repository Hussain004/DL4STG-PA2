"""Training and evaluation harness for the river forecaster."""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import river_data as rd          # noqa: E402
import river_models as rm        # noqa: E402
import river_sim as rs           # noqa: E402

CACHE = Path(__file__).resolve().parents[1] / "results" / "cache"


# ------------------------------------------------------------------ dataset --
def get_dataset(spec, cfg=None, sizes=(90000, 30000, 30000), missing_prob=None,
                delay_prob=None, tag="main", rebuild=False):
    """Simulate (and cache) the three splits."""
    CACHE.mkdir(parents=True, exist_ok=True)
    cfg = cfg or rs.RiverConfig()
    path = CACHE / f"{tag}_L{spec.L}_H{spec.H}_v{rs.SIM_VERSION}.npz"
    if path.exists() and not rebuild:
        z = np.load(path, allow_pickle=True)
        ds = object.__new__(rd.RiverDataset)
        ds.cfg = cfg
        ds.spec = spec
        ds.splits = {k: (z[f"{k}_X"], z[f"{k}_Y"], z[f"{k}_yn"]) for k in ("train", "val", "test")}
        ds.stats = {int(k): tuple(v) for k, v in json.loads(str(z["stats"])).items()}
        ds.release_stats = tuple(z["release_stats"])
        ds.rain_stats = tuple(z["rain_stats"])
        return ds
    ds = rd.RiverDataset(cfg, spec, sizes=sizes, missing_prob=missing_prob,
                         delay_prob=delay_prob)
    payload = {"stats": json.dumps(ds.stats), "release_stats": ds.release_stats,
               "rain_stats": ds.rain_stats}
    for k, (X, Y, yn) in ds.splits.items():
        payload[f"{k}_X"] = X
        payload[f"{k}_Y"] = Y
        payload[f"{k}_yn"] = yn
    np.savez_compressed(path, **payload)
    return ds


def to_tensors(ds, name, device="cpu"):
    X, Y, yn = ds.normalised(name)
    return (torch.tensor(X, device=device), torch.tensor(Y, device=device),
            torch.tensor(yn, device=device))


# -------------------------------------------------------------------- loss ---
def make_loss(kind, scales, means, high_q=90.0):
    """Loss computed in METRES so the weights are physically meaningful.

    "wmse" weights the samples whose CURRENT level is already high (flood
    risk).  The weight is a function of the *input* (the level now), not of
    the target, so the objective stays smooth and the optimisation stable.
    """
    def fn(pred_norm, target_norm, y_now_norm):
        dev = pred_norm.device
        sc = torch.as_tensor(scales, dtype=torch.float32, device=dev).view(1, -1, 1)
        mu = torch.as_tensor(means, dtype=torch.float32, device=dev).view(1, -1, 1)
        p = pred_norm * sc + mu
        t = target_norm * sc + mu
        if kind == "mse":
            return ((p - t) ** 2).mean()
        if kind == "mse_norm":
            # MSE in per-station-normalised units: every station contributes
            # equally, so a noisy station cannot drown out a quiet one whose
            # forecast depends on the upstream lead.
            return ((pred_norm - target_norm) ** 2).mean()
        if kind == "mae":
            return (p - t).abs().mean()
        if kind == "huber":
            return nn.functional.huber_loss(p, t, delta=0.2)
        if kind == "wmse":
            now = y_now_norm * sc.view(1, -1) + mu.view(1, -1)
            thr = float(np.percentile(now.detach().cpu().numpy(), high_q))
            w = 1.0 + 3.0 * (now > thr).float()
            return (w.unsqueeze(-1) * (p - t) ** 2).mean() / w.unsqueeze(-1).mean()
        if kind == "wmse_target":
            thr = float(np.percentile(t.detach().cpu().numpy(), high_q))
            w = 1.0 + 3.0 * torch.clamp((t - thr) / max(float(sc.mean()), 1e-6), 0, 1)
            return (w * (p - t) ** 2).mean() / w.mean()
        raise ValueError(kind)

    return fn


# ----------------------------------------------------------------- metrics ---
def metrics(pred_m, true_m, scales=None, high_threshold=None):
    """pred_m, true_m: numpy arrays [N, 4, H] in metres."""
    err = pred_m - true_m
    out = {
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "bias": float(err.mean()),
    }
    for i, nm in enumerate(["S1", "S2", "S3", "S4"]):
        out[f"mae_{nm}"] = float(np.abs(err[:, i]).mean())
        out[f"rmse_{nm}"] = float(np.sqrt((err[:, i] ** 2).mean()))
    for h in range(err.shape[2]):
        out[f"mae_h{h+1}"] = float(np.abs(err[:, :, h]).mean())
    if high_threshold is not None:
        m = true_m >= high_threshold
        if m.any():
            out["mae_flood"] = float(np.abs(err[m]).mean())
            out["rmse_flood"] = float(np.sqrt((err[m] ** 2).mean()))
            out["n_flood"] = int(m.sum())
    return out


# ---------------------------------------------------------------- training ---
def evaluate(model, X, Y, yn, scales, means, device="cpu", batch=1024,
             high_threshold=None):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            p = model(X[i:i + batch].to(device), yn[i:i + batch].to(device))
            preds.append(p.cpu().numpy())
    pred_n = np.concatenate(preds)
    sc = np.asarray(scales).reshape(1, -1, 1); mu = np.asarray(means).reshape(1, -1, 1)
    pred_m = pred_n * sc + mu
    true_m = Y.detach().cpu().numpy() * sc + mu
    return metrics(pred_m, true_m, high_threshold=high_threshold), pred_m


def train(kind, ds, spec, *, width=64, depth=4, rounds=4, lag_window=12, heads=4,
          loss_kind="mse", steps=2000, batch=256, lr=2e-3, seed=0, device="cpu",
          patience=8, eval_every=250, verbose=False, warmup=100):
    torch.manual_seed(seed)
    np.random.seed(seed)
    scales, means = ds.scales()
    model = rm.build(kind, spec, width=width, depth=depth, rounds=rounds,
                     lag_window=lag_window, heads=heads).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    if n_params == 0:                       # e.g. the persistence baseline
        return {"model": model, "n_params": 0, "best_step": 0,
                "val_mae": float("nan"), "history": [], "seconds": 0.0}

    Xtr, Ytr, ytr = to_tensors(ds, "train", device)
    Xva, Yva, yva = to_tensors(ds, "val", device)

    fn = make_loss(loss_kind, scales, means)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=steps, pct_start=0.15, final_div_factor=50)

    best = (np.inf, None, -1)
    bad = 0
    n = len(Xtr)
    hist = []
    t0 = time.time()
    for step in range(steps):
        model.train()
        idx = torch.randint(0, n, (batch,), device=device)
        pred = model(Xtr[idx], ytr[idx])
        loss = fn(pred, Ytr[idx], ytr[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if (step + 1) % eval_every == 0 or step == steps - 1:
            m, _ = evaluate(model, Xva, Yva, yva, scales, means, device)
            hist.append({"step": step + 1, "train_loss": float(loss.item()),
                         "val_mae": m["mae"]})
            if m["mae"] < best[0] - 1e-5:
                best = (m["mae"], {k: v.detach().clone() for k, v in model.state_dict().items()},
                        step + 1)
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    break
            if verbose:
                print(f"  step {step+1:5d} loss {loss.item():.5f} val MAE {m['mae']:.4f}")
    if best[1] is not None:
        model.load_state_dict(best[1])
    return {"model": model, "n_params": n_params, "best_step": best[2],
            "val_mae": best[0], "history": hist, "seconds": time.time() - t0}


def run(kind, ds, spec, tag="main", device="cpu", **kw):
    """Train + evaluate; returns a results dict."""
    scales, means = ds.scales()
    Xva, Yva, yva = to_tensors(ds, "val")
    Xte, Yte, yte = to_tensors(ds, "test")
    # flood threshold: 90th percentile of the training targets, in metres
    thr = float(np.percentile(ds.splits["train"][1], 90))

    out = train(kind, ds, spec, device=device, **kw)
    model = out["model"]
    vm, _ = evaluate(model, Xva, Yva, yva, scales, means, device=device,
                     high_threshold=thr)
    tm, _ = evaluate(model, Xte, Yte, yte, scales, means, device=device,
                     high_threshold=thr)

    # baselines on the same test split
    sc_b = np.asarray(scales).reshape(1, -1, 1)
    mu_b = np.asarray(means).reshape(1, -1, 1)
    persist = (yte.numpy() * sc_b.reshape(1, -1) + mu_b.reshape(1, -1))
    true_m = Yte.numpy() * sc_b + mu_b
    persist_m = np.repeat(persist[:, :, None], spec.H, axis=2)
    pers = metrics(persist_m, true_m, high_threshold=thr)

    res = {"kind": kind, "tag": tag, "n_params": out["n_params"],
           "best_step": out["best_step"], "val_mae": out["val_mae"],
           "seconds": out["seconds"], "val": vm, "test": tm,
           "persistence_test": pers, "flood_threshold": thr,
           "config": {k: v for k, v in kw.items()}}
    return res, model


def ridge_baseline(ds, spec, lam=1.0):
    """Closed-form per-station linear map from its own history to its future."""
    scales, means = ds.scales()
    Xtr, Ytr, ytr = ds.normalised("train")
    Xva, Yva, yna = ds.normalised("val")
    Xte, Yte, ynte = ds.normalised("test")
    thr = float(np.percentile(ds.splits["train"][1], 90))
    res = {"kind": "ridge", "tag": "main"}
    preds_te = np.zeros_like(Yte)
    preds_va = np.zeros_like(Yva)
    for i, row in enumerate(rd.TARGET_ROWS):
        A = Xtr[:, :spec.L, row, 0]
        A = np.concatenate([A, np.ones((len(A), 1))], axis=1)
        B = Ytr[:, i] - ytr[:, i][:, None]      # predict the CHANGE, not the level
        W = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ B)
        Av = np.concatenate([Xva[:, :spec.L, row, 0], np.ones((len(Xva), 1))], axis=1)
        At = np.concatenate([Xte[:, :spec.L, row, 0], np.ones((len(Xte), 1))], axis=1)
        preds_va[:, i] = Av @ W + yna[:, i][:, None]
        preds_te[:, i] = At @ W + ynte[:, i][:, None]
    sc_b = np.asarray(scales).reshape(1, -1, 1)
    mu_b = np.asarray(means).reshape(1, -1, 1)
    res["val"] = metrics(preds_va * sc_b + mu_b, Yva * sc_b + mu_b, high_threshold=thr)
    res["test"] = metrics(preds_te * sc_b + mu_b, Yte * sc_b + mu_b, high_threshold=thr)
    res["n_params"] = 4 * (spec.L + 1) * spec.H
    res["config"] = {"lam": lam}
    return res
