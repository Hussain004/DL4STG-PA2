"""Turn a simulated river into the model-ready tensors described in the
Q1 tutorial.

The grid
--------
rows (places) : Dam, S1, T1, S2, S3, S4          (N = 6)
columns (time) : L past steps + H future steps   (T = L + H)
channels       : [level, observed?, rain, release]

  * the Dam row carries recorded releases in the past columns and the
    ANNOUNCED schedule in the future columns;
  * R1's rain is folded into the S2 row and R2's rain into the S3 row
    (tutorial assumption);
  * future columns are empty and flagged "not seen", except for the Dam row
    where the announced schedule is known at issuance.

Nothing that is unknown at issuance ever enters the grid.
"""
from __future__ import annotations

import numpy as np

import river_sim as rs

ROWS = ["Dam", "S1", "T1", "S2", "S3", "S4"]
R_DAM, R_S1, R_T1, R_S2, R_S3, R_S4 = range(6)
TARGET_ROWS = [R_S1, R_S2, R_S3, R_S4]          # grid rows that are forecast
STATION_HYD = [rs.S1, rs.S2, rs.S3, rs.S4]     # their indices in `levels`
# grid row for each forecast station, in the same order as STATION_HYD
TARGET_GRID_ROWS = [R_S1, R_S2, R_S3, R_S4]
N_ROWS = 6
N_CHANNELS = 4                                  # level, observed?, rain, release


class GridSpec:
    """Static description of the grid + the river graph."""

    def __init__(self, lookback=24, horizon=6):
        self.L = lookback
        self.H = horizon
        self.T = lookback + horizon
        self.rows = ROWS
        self.edges = [("S1", "S2"), ("T1", "S2"), ("S2", "S3"), ("S3", "S4"),
                      ("Dam", "S1")]
        self.upstream = {r: [] for r in ROWS}
        for a, b in self.edges:
            self.upstream[b].append(a)
        self.row_of = {r: i for i, r in enumerate(ROWS)}

    def upstream_indices(self, row):
        return [self.row_of[u] for u in self.upstream[row]]


def add_observations(sim, cfg, rng, missing_prob=None, delay_prob=None):
    """Apply noise, dropouts and stale (late) readings to a simulated path."""
    mp = cfg.missing_prob if missing_prob is None else missing_prob
    dp = cfg.delay_prob if delay_prob is None else delay_prob
    levels = sim["levels"]
    n, k = levels.shape
    obs = levels + rng.normal(0, cfg.level_noise, levels.shape)
    flag = np.ones((n, k))
    # dropouts
    drop = rng.random((n, k)) < mp
    obs[drop] = 0.0
    flag[drop] = 0.0
    # late (stale) readings: the value reported at t was measured at t - d
    late = rng.random((n, k)) < dp
    d = rng.integers(1, cfg.delay_max + 1, size=(n, k))
    idx = np.arange(n)[:, None]
    src = np.clip(idx - d, 0, n - 1)
    stale = np.where(late, obs[src, np.arange(k)[None, :]], obs)
    obs = stale
    return obs, flag


def build_arrays(sim, cfg, rng, missing_prob=None, delay_prob=None):
    """Whole-trajectory grids, so that windows are pure slicing.

    Returns a dict with level/flag/rain/release grids indexed by time.
    """
    obs, flag = add_observations(sim, cfg, rng, missing_prob, delay_prob)
    n = sim["levels"].shape[0]
    level = np.zeros((n, N_ROWS), dtype=np.float32)
    seen = np.zeros((n, N_ROWS), dtype=np.float32)
    rain = np.zeros((n, N_ROWS), dtype=np.float32)
    rel = np.zeros((n, N_ROWS), dtype=np.float32)

    for row, hid in zip([R_S1, R_T1, R_S2, R_S3, R_S4],
                        [rs.S1, rs.T1, rs.S2, rs.S3, rs.S4]):
        level[:, row] = obs[:, hid]
        seen[:, row] = flag[:, hid]
    rain[:, R_S2] = sim["rain1"]
    rain[:, R_S3] = sim["rain2"]
    rel[:, R_DAM] = sim["release"]
    return {"level": level, "seen": seen, "rain": rain, "release": rel,
            "plan": sim["plan"], "levels": sim["levels"]}


def make_windows(arrays, spec):
    """Slice every issuance window (vectorised).

    Window w has forecast origin o = L-1+w; its columns are
    [o-L+1 .. o] (past) and [o+1 .. o+H] (horizon).
    """
    n = arrays["level"].shape[0]
    L, H = spec.L, spec.H
    origins = np.arange(L - 1, n - H)
    W = len(origins)
    sw = np.lib.stride_tricks.sliding_window_view

    X = np.zeros((W, L + H, N_ROWS, N_CHANNELS), dtype=np.float32)
    for ch, key in enumerate(("level", "seen", "rain", "release")):
        v = arrays[key]                                   # [n, N_ROWS]
        win = sw(v, L, axis=0)                            # [n-L+1, N_ROWS, L]
        X[:, :L, :, ch] = np.transpose(win[origins - L + 1], (0, 2, 1))
    # future columns: only the announced schedule is known
    plan_win = sw(arrays["plan"], H, axis=0)              # [n-H+1, H]
    X[:, L:, R_DAM, 3] = plan_win[origins]
    X[:, L:, R_DAM, 1] = 1.0

    Y = np.zeros((W, 4, H), dtype=np.float32)
    yn = np.zeros((W, 4), dtype=np.float32)
    lev = arrays["levels"]                               # [n, 5]
    for i, hid in enumerate(STATION_HYD):
        win = sw(lev[:, hid], H, axis=0)                  # [n-H+1, H]
        Y[:, i] = win[origins]
    # last observed level at or before the origin (carry-forward through gaps)
    for i, row in enumerate(TARGET_GRID_ROWS):
        col = arrays["level"][:, row]
        seen = arrays["seen"][:, row]
        last_idx = np.maximum.accumulate(np.where(seen > 0, np.arange(n), -1))
        yn_col = np.zeros(W, dtype=np.float32)
        good = last_idx[origins]
        yn_col = col[good]
        yn_col[good < 0] = col[0]
        yn[:, i] = yn_col
    return X, Y, yn


def normaliser(X, Y, yn, L=24):
    """Per-station level scale and mean from the training split."""
    lvl = X[:, :L, :, 0].reshape(-1, N_ROWS)
    stats = {}
    for row in TARGET_ROWS:
        col = lvl[:, row]
        stats[row] = (float(col.mean()), float(col.std() + 1e-6))
    return stats


class RiverDataset:
    """Holds the three splits and the normalisation constants."""

    def __init__(self, cfg, spec, seeds=(11, 12, 13, 14), sizes=(90000, 30000, 30000),
                 burn_in=3000, missing_prob=None, delay_prob=None):
        self.cfg = cfg
        self.spec = spec
        self.splits = {}
        names = ("train", "val", "test")
        for name, seed, size in zip(names, seeds, sizes):
            rng = np.random.default_rng(seed)
            sim = rs.simulate(cfg, size, rng, burn_in=burn_in)
            arrays = build_arrays(sim, cfg, rng, missing_prob, delay_prob)
            X, Y, yn = make_windows(arrays, spec)
            self.splits[name] = (X, Y, yn)
            if name == "train":
                self.stats = normaliser(X, Y, yn, spec.L)
                self.release_stats = (float(sim["release"].mean()),
                                      float(sim["release"].std() + 1e-6))
                self.rain_stats = (float(sim["rain1"].mean()), float(sim["rain1"].std() + 1e-6),
                                   float(sim["rain2"].mean()), float(sim["rain2"].std() + 1e-6))

    def normalised(self, name):
        X, Y, yn = self.splits[name]
        Xn = X.copy()
        yn = yn.copy()          # never mutate the stored split
        for row in TARGET_ROWS:
            mu, sd = self.stats[row]
            Xn[:, :, row, 0] = (X[:, :, row, 0] - mu) / sd
        # the Dam row carries release; normalise it with the release stats
        rm, rs_ = self.release_stats
        Xn[:, :, R_DAM, 3] = (X[:, :, R_DAM, 3] - rm) / rs_
        # rain
        r1m, r1s, r2m, r2s = self.rain_stats
        Xn[:, :, R_S2, 2] = (X[:, :, R_S2, 2] - r1m) / r1s
        Xn[:, :, R_S3, 2] = (X[:, :, R_S3, 2] - r2m) / r2s
        Yn = np.zeros_like(Y)
        for i, row in enumerate(TARGET_ROWS):
            mu, sd = self.stats[row]
            Yn[:, i] = (Y[:, i] - mu) / sd
            yn[:, i] = (yn[:, i] - mu) / sd
        return Xn, Yn, yn

    def scales(self):
        return np.array([self.stats[r][1] for r in TARGET_ROWS]), \
               np.array([self.stats[r][0] for r in TARGET_ROWS])
