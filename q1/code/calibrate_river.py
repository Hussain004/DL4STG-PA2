"""Calibrate the synthetic river: travel times, level ranges, and the
calm-vs-flood contrast the tutorial describes."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from river_sim import (RiverConfig, S1, T1, S2, S3, S4, simulate,  # noqa: E402
                       travel_time)


def highpass(x, win=41):
    """Remove the slow trend so that a transport delay is measurable."""
    k = np.ones(win) / win
    trend = np.convolve(x, k, mode="same")
    return x - trend


def measured_lag(a, b, max_lag=30, win=41):
    """Lag (steps) at which b's fluctuations follow a's."""
    a = highpass(a, win)
    b = highpass(b, win)
    best, best_lag = -np.inf, 0
    for lag in range(0, max_lag + 1):
        c = float(np.corrcoef(a[:-lag or None], b[lag:])[0, 1])
        if c > best:
            best, best_lag = c, lag
    return best_lag, best


def regime_lags(cfg, seed=0, n=20000):
    """Measured lags at low flow (calm) and at high flow (flood)."""
    rng = np.random.default_rng(seed)
    out = simulate(cfg, n, rng)
    lv = out["levels"]
    calm = lv[:, 4] < np.quantile(lv[:, 4], 0.5)      # lower half of S4
    flood = lv[:, 4] > np.quantile(lv[:, 4], 0.9)     # upper decile of S4
    res = {}
    for tag, mask in (("calm", calm), ("flood", flood)):
        if mask.sum() < 400:
            continue
        idx = np.flatnonzero(mask)
        for (i, j, nm) in ((S2, S3, "S2->S3"), (S3, S4, "S3->S4"),
                           (S1, S2, "S1->S2"), (T1, S2, "T1->S2")):
            a, b = lv[idx, i], lv[idx, j]
            res[f"{nm}@{tag}"] = measured_lag(a, b)[0]
    res["levels_mean"] = np.round(lv.mean(0), 3).tolist()
    res["levels_p99"] = np.round(np.quantile(lv, 0.99, axis=0), 3).tolist()
    res["levels_p01"] = np.round(np.quantile(lv, 0.01, axis=0), 3).tolist()
    res["release"] = [float(out["release"].min()), float(out["release"].max())]
    return res


if __name__ == "__main__":
    cfg = RiverConfig()
    res = regime_lags(cfg, seed=0, n=12000)
    for k, v in res.items():
        print(f"{k:16s} {v}")
