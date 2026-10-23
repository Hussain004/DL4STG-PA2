"""Travel-time probe for the synthetic river.

A step change in the reservoir release is the cleanest way to measure travel
time: run the river to equilibrium, step the release, and record when each
station starts to respond.

We report two numbers per station:
  * t10  -- "time of first rise": steps until the level has moved 10 % of its
            eventual change.  This is the transport delay a forecaster must
            anticipate.
  * t50  -- steps until the level has moved 50 % of its eventual change.
            This includes the storage response and the dispersion.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import river_sim as rs  # noqa: E402


def step_response(cfg, base_release, jump=1.6, n_eq=2500, n_step=500,
                  storm=False, u_step=False, pulse=None):
    """Equilibrate at ``base_release``, then step it up by ``jump``."""
    rng = np.random.default_rng(7)
    n = n_eq + n_step

    rain1 = np.zeros(n)
    rain2 = np.zeros(n)
    if storm:
        rain1 = rs.two_state_rain(n, cfg.rain_r1_mean * 1.6, cfg.rain_wet_prob,
                                   cfg.rain_burst_len, rng)
        rain2 = rs.two_state_rain(n, cfg.rain_r2_mean * 1.6, cfg.rain_wet_prob,
                                   cfg.rain_burst_len, rng)
    r1_rain = rs._runoff(rain1, cfg.t1_infiltration, cfg.t1_tau, n)
    u_rain = rs._runoff(rain2, cfg.u_infiltration, cfg.u_tau, n)
    u_hidden = np.zeros(n)
    e = rng.normal(0, 1, n)
    for t in range(1, n):
        u_hidden[t] = cfg.u_extra_ar * u_hidden[t - 1] + e[t]
    if u_hidden.std() > 0:
        u_hidden *= cfg.u_hidden_frac * (u_rain.std() + 1e-9) / u_hidden.std()
    u_inflow = (1.0 - cfg.u_hidden_frac) * u_rain + u_hidden

    release = np.empty(n)
    release[:n_eq] = base_release
    release[n_eq:] = base_release * jump
    if pulse is not None:
        release[n_eq:] = base_release
        release[n_eq] = base_release + pulse

    levels = np.zeros((n, 5))
    h = np.array([2.60, 1.55, 2.70, 2.80, 2.85])
    out_hist = {nm: [0.0] * 8 for nm in rs.HYD_NODES}
    w0, w1, w2 = cfg.dispersion

    for t in range(n):
        inflow = [0.0] * 5
        inflow[rs.S1] = release[t] + cfg.base_inflow[rs.S1]
        inflow[rs.T1] = cfg.base_inflow[rs.T1]
        u_in = cfg.u_gain * u_inflow[t]
        if u_step:
            u_in = u_in if t < n_eq else u_in * jump
        out = [cfg.node_rating[n_] * max(h[n_], 0.0) ** (5.0 / 3.0) for n_ in range(5)]
        for (src, dst, tau0) in rs.EDGES:
            if src == "Dam":
                continue
            if src == "U":
                inflow[rs.S3] += u_in
                continue
            j = rs.NODE_INDEX[src]
            i = rs.NODE_INDEX[dst]
            q = max(out[j], 1e-6)
            tau = (tau0 * cfg.travel_scale) * (rs.EDGE_QREF[f"{src}>{dst}"] / q) ** cfg.travel_alpha
            tau = min(max(tau, 0.35 * tau0 * cfg.travel_scale),
                      1.5 * tau0 * cfg.travel_scale)
            lag = tau / rs.DT
            d = int(np.floor(lag))
            f = lag - d
            hh = out_hist[src]
            v0 = hh[-1 - d] if d + 1 <= len(hh) else 0.0
            v1 = hh[-2 - d] if d + 2 <= len(hh) else 0.0
            inflow[i] += (1.0 - f) * v0 + f * v1
        inflow[rs.S2] += cfg.t1_gain * r1_rain[t]
        for n_ in range(5):
            kk = cfg.node_rating[n_]
            do = (5.0 / 3.0) * kk * max(h[n_], 0.0) ** (2.0 / 3.0)
            h[n_] = max(h[n_] + (inflow[n_] - out[n_]) / (cfg.node_area[n_] + do), 0.0)
            prev = out_hist[rs.HYD_NODES[n_]]
            prev.append(w0 * out[n_] + w1 * prev[-1] + w2 * prev[-2])
        levels[t] = h
    return levels, release


def pulse_response(cfg, base_release, amplitude=40.0, n_eq=2500, n_pulse=200):
    """Equilibrate, then inject a ONE-step pulse into the reservoir release.

    The centroid of each station's level response is a continuous, well
    defined effective travel time (the standard hydraulic measure).
    """
    lv, _ = step_response(cfg, base_release, jump=1.0, n_eq=n_eq, n_step=n_pulse,
                           pulse=amplitude)
    return lv


def centroids(levels, n_eq, n_pulse):
    """Centroid (in steps) of each station's response to a one-step pulse."""
    out = {}
    for n_ in range(5):
        before = levels[n_eq - 30:n_eq, n_].mean()
        seg = levels[n_eq:n_eq + n_pulse, n_] - before
        w = np.clip(seg, 0, None)          # only the positive part
        if w.sum() <= 0:
            out[rs.HYD_NODES[n_]] = np.nan
            continue
        idx = np.arange(n_pulse)
        out[rs.HYD_NODES[n_]] = float((idx * w).sum() / w.sum())
    return out


def response_times(levels, n_eq, fracs=(0.1, 0.5)):
    out = {f: {} for f in fracs}
    for n_ in range(5):
        before = levels[n_eq - 30:n_eq, n_].mean()
        after = levels[-30:, n_].mean()
        change = after - before
        seg = levels[n_eq:, n_] - before
        for f in fracs:
            if abs(change) < 1e-9:
                out[f][rs.HYD_NODES[n_]] = np.nan
                continue
            idx = np.flatnonzero(np.abs(seg) >= f * abs(change))
            out[f][rs.HYD_NODES[n_]] = int(idx[0]) if len(idx) else len(seg)
    return out


def edge_lags(levels, n_eq, src, dst, frac=0.1):
    rt = response_times(levels, n_eq, fracs=(frac,))[frac]
    if src == "Dam":
        s = 0
    elif src == "U":
        s = 0
    else:
        s = rt[src]
    d = rt[dst]
    if np.isnan(s) or np.isnan(d):
        return np.nan
    return int(d - s)


if __name__ == "__main__":
    cfg = rs.RiverConfig()
    n_eq = 2500
    report = {}
    for tag, base, storm in (("calm", 30.0, False), ("flood", 37.0, False),
                             ("storm-on-calm", 30.0, True)):
        lv, _ = step_response(cfg, base, n_eq=n_eq, storm=storm)
        rt = response_times(lv, n_eq)
        print(f"--- {tag} (base release {base} m3/s) ---")
        for nm in rs.HYD_NODES:
            print(f"  {nm}: t10 = {rt[0.1][nm]:6.1f} steps ({rt[0.1][nm]*rs.DT:6.1f} min)   "
                  f"t50 = {rt[0.5][nm]:6.1f} steps ({rt[0.5][nm]*rs.DT:6.1f} min)")
        lags = {}
        for (s_, d_) in (("S1", "S2"), ("S2", "S3"), ("S3", "S4"), ("Dam", "S1")):
            lag = edge_lags(lv, n_eq, s_, d_)
            lags[f"{s_}>{d_}"] = None if np.isnan(lag) else lag * rs.DT
            print(f"  {s_}->{d_}: {lags[f'{s_}>{d_}']} min")
        report[tag] = {"response": {k: {n: float(v) for n, v in d.items()}
                                    for k, d in rt.items()},
                       "edge_lag_min": lags}
    # --- impulse (pulse) probe: continuous effective travel times -----------
    print()
    for tag, base in (("calm", 30.0), ("flood", 37.0)):
        lv = pulse_response(cfg, base)
        cen = centroids(lv, 2500, 200)
        prev = 0.0
        print(f"--- pulse centroid, {tag} (release {base}) ---")
        for nm in rs.HYD_NODES:
            if np.isnan(cen[nm]):
                print(f"  {nm}: n/a")
                continue
            print(f"  {nm}: centroid {cen[nm]:6.2f} steps = {cen[nm]*rs.DT:6.1f} min"
                  f"   (from upstream: {cen[nm]-prev:5.2f} steps = {(cen[nm]-prev)*rs.DT:5.1f} min)")
            if nm != "T1":
                prev = cen[nm]
        report[f"{tag}_centroid"] = {k: float(v) for k, v in cen.items()}

    out = Path(__file__).resolve().parents[1] / "results" / "travel_times.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print("wrote", out)
