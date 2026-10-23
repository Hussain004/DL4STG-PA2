"""Synthetic river system for the PA2 Q1 design study.

The simulator is a *design laboratory*, not a claim about the real river. It is
built so that every property the Q1 handout emphasises is present and tunable:

  * a directed downstream topology  Dam -> S1 -> S2 -> S3 -> S4
    with a monitored tributary  T1 -> S2  and an UNMONITORED tributary  U -> S3;
  * travel times that SHORTEN when the river is in flood (calm ~50 min for the
    S2->S3 reach, flood ~25 min), as described in the tutorial;
  * rainfall at two gauges R1, R2, whose catchments differ, R2 sitting above
    the unmonitored stream;
  * reservoir releases recorded in the past and ANNOUNCED for the future, with
    the announcement not always equal to what actually happens;
  * missing and late sensor readings, noisy readings;
  * a level target at S1..S4 one hour ahead.

Hydrology
---------
Each station is a storage with a power-law rating curve  O = k h^{5/3}.  Water
leaves a station and is transported to the next one through a *travel delay*
line whose length depends on the current discharge in that reach:

    tau(t) = tau0 * (Q_ref / Q(t))**alpha        (clamped to a safe band)

so a rising, faster river shortens the delay -- the mechanism the handout
calls "travel times can change with river conditions".  A short dispersion
kernel broadens the pulse.  The scheme is explicit and mass-conserving.
"""
from __future__ import annotations

import numpy as np

DT = 10  # minutes per step
SIM_VERSION = 4   # bump to invalidate cached datasets

S1, T1, S2, S3, S4 = 0, 1, 2, 3, 4
HYD_NODES = ["S1", "T1", "S2", "S3", "S4"]
NODE_INDEX = {n: i for i, n in enumerate(HYD_NODES)}
GRID_ROWS = ["Dam", "S1", "T1", "S2", "S3", "S4"]

# (source node, destination node, base travel time in minutes)
EDGES = [
    ("Dam", "S1", 4.0),
    ("S1", "S2", 8.0),
    ("T1", "S2", 6.0),
    ("S2", "S3", 10.0),   # the reach quoted in the tutorial (50 min calm)
    ("S3", "S4", 16.0),
    ("U", "S3", 9.0),
]
# reference discharge per edge for the flow-dependent travel time
# reference discharge per edge = the calm-condition discharge in that reach
EDGE_QREF = {"Dam>S1": 34.0, "S1>S2": 34.0, "T1>S2": 5.0, "S2>S3": 39.0, "S3>S4": 39.0,
             "U>S3": 4.0}


class RiverConfig:
    def __init__(self, **kw):
        # travel_scale shrinks every travel time: 1.0 = the calibrated calm
        # river, 0.4 = a flood river whose wave runs much faster
        self.travel_scale = 1.0
        self.node_area = {S1: 2.0, T1: 1.5, S2: 2.5, S3: 2.5, S4: 3.0}
        self.node_rating = {S1: 6.5, T1: 5.0, S2: 6.8, S3: 7.0, S4: 7.2}
        self.base_inflow = {S1: 6.0, T1: 5.0, S2: 0.0, S3: 0.0, S4: 0.0}
        self.rain_r1_mean = 0.60       # mm per 10 min, long-run average
        self.rain_r2_mean = 0.75
        self.rain_wet_prob = 0.055
        self.rain_burst_len = (3, 15)
        self.t1_infiltration = 0.30    # mm/10min
        self.t1_tau = 8.0              # steps
        self.t1_gain = 1.3
        self.u_infiltration = 0.45
        self.u_tau = 10.0
        self.u_gain = 1.3
        self.u_hidden_frac = 0.45      # share of U's inflow NOT explained by R2
        self.u_extra_ar = 0.97
        self.dam_plan_error = 7.0     # sd of a plan deviation, m3/s
        self.plan_change_prob = 0.18   # chance the operator deviates at all
        self.level_noise = 0.012       # m
        self.rain_noise = 0.02
        self.missing_prob = 0.012
        self.delay_prob = 0.010
        self.delay_max = 3
        self.travel_alpha = 1.55
        self.t1_base_ar = 0.985       # exponent of the flow-dependent delay
        self.dispersion = (0.10, 0.80, 0.10)
        self.__dict__.update(kw)

    def copy_with(self, **kw):
        cfg = RiverConfig()
        cfg.node_area = dict(self.node_area)
        cfg.node_rating = dict(self.node_rating)
        cfg.base_inflow = dict(self.base_inflow)
        cfg.__dict__.update(kw)
        return cfg


# --------------------------------------------------------------- utilities ---
def two_state_rain(n, mean_rate, wet_prob, burst, rng):
    """Rainfall with dry/wet spells (a simple Markov storm process)."""
    rain = np.zeros(n)
    i = 0
    while i < n:
        if rng.random() < wet_prob:
            length = int(rng.integers(burst[0], burst[1] + 1))
            target = mean_rate / max(wet_prob, 1e-6)
            sigma = 0.85
            mu = np.log(target) - sigma ** 2 / 2
            for k in range(length):
                if i >= n:
                    break
                rain[i] = max(0.0, rng.lognormal(mu, sigma))
                i += 1
        else:
            i += 1
    return rain


def _runoff(rain, infiltration, tau, n):
    """Excess rainfall routed through a linear reservoir."""
    excess = np.maximum(0.0, rain - infiltration)
    out = np.zeros(n)
    store = 0.0
    for t in range(n):
        store += (excess[t] - store) / tau
        out[t] = store
    return out


def simulate(cfg: RiverConfig, n_steps: int, rng: np.random.Generator,
             burn_in: int = 3000):
    """Simulate a river trajectory; returns a dict of length-n_steps arrays.

    ``burn_in`` steps are simulated and discarded so that the trajectory
    starts in equilibrium.
    """
    n_total = n_steps + burn_in
    n_steps = n_total
    rain1 = two_state_rain(n_steps, cfg.rain_r1_mean, cfg.rain_wet_prob,
                            cfg.rain_burst_len, rng)
    rain2 = two_state_rain(n_steps, cfg.rain_r2_mean, cfg.rain_wet_prob,
                            cfg.rain_burst_len, rng)

    # catchments, following the tutorial's stated assumption:
    #   R1 -> S2 catchment (main stem, above the T1 confluence is irrelevant)
    #   R2 -> the UNMONITORED stream, which joins between S2 and S3
    r1_rain = _runoff(rain1, cfg.t1_infiltration, cfg.t1_tau, n_steps)
    u_rain = _runoff(rain2, cfg.u_infiltration, cfg.u_tau, n_steps)

    # an AR(1) component of U's inflow that R2 does NOT explain
    u_hidden = np.zeros(n_steps)
    e = rng.normal(0, 1, n_steps)
    for t in range(1, n_steps):
        u_hidden[t] = cfg.u_extra_ar * u_hidden[t - 1] + e[t]
    s = u_hidden.std()
    if s > 0:
        u_hidden *= cfg.u_hidden_frac * (u_rain.std() + 1e-9) / s
    u_inflow = (1.0 - cfg.u_hidden_frac) * u_rain + u_hidden

    # the monitored tributary has its own (unobserved) catchment: a slow baseflow
    t1_base = np.zeros(n_steps)
    e = rng.normal(0, 1, n_steps)
    b = cfg.base_inflow[T1]
    for t in range(n_steps):
        b = cfg.t1_base_ar * b + (1 - cfg.t1_base_ar) * cfg.base_inflow[T1] + 0.9 * e[t]
        t1_base[t] = max(1.2, b)

    # ---- reservoir release: piecewise-constant blocks -----------------------
    release = np.zeros(n_steps)
    true_target = np.zeros(n_steps)
    bands = ((25.0, 28.5), (27.5, 33.0), (33.5, 39.0))  # drought, normal, flood
    t = 0
    while t < n_steps:
        length = int(rng.integers(20, 81))
        # the hydrological regime changes every few weeks, so every simulated
        # trajectory contains drought, normal and flood periods
        if (t // 2000) % 3 == 0:
            regime = 0
        elif (t // 2000) % 3 == 1:
            regime = 1
        else:
            regime = 2
        target = rng.uniform(*bands[regime])
        for k in range(length):
            if t + k >= n_steps:
                break
            true_target[t + k] = target
        t += length
    r = true_target[0]
    for t in range(n_steps):
        r += 0.12 * (true_target[t] - r) + rng.normal(0, 0.25)
        release[t] = max(0.0, r)

    # the announced plan for the coming hour
    plan = np.empty(n_steps)
    for t in range(n_steps):
        ahead = release[min(t + 6, n_steps - 1)]
        if rng.random() < cfg.plan_change_prob:
            ahead = max(0.0, ahead + rng.normal(0, cfg.dam_plan_error))
        plan[t] = ahead
    plan = np.maximum(0.0, plan)

    # ---- levels -------------------------------------------------------------
    levels = np.zeros((n_steps, 5))
    h = np.array([2.60, 1.55, 2.70, 2.80, 2.85])
    out_hist = {n: [0.0] * 8 for n in HYD_NODES}
    inflow = [0.0] * 5
    w0, w1, w2 = cfg.dispersion

    for t in range(n_steps):
        inflow = [0.0] * 5
        inflow[S1] = release[t] + cfg.base_inflow[S1]
        inflow[T1] = t1_base[t]
        # unmonitored tributary joins between S2 and S3
        u_in = cfg.u_gain * u_inflow[t]

        out = [cfg.node_rating[n] * max(h[n], 0.0) ** (5.0 / 3.0) for n in range(5)]

        # transport along the edges with a flow-dependent travel time
        for (src, dst, tau0) in EDGES:
            if src == "Dam":
                continue
            if src == "U":
                inflow[S3] += u_in
                continue
            j = NODE_INDEX[src]
            i = NODE_INDEX[dst]
            q = max(out[j], 1e-6)
            tau = (tau0 * cfg.travel_scale) * (EDGE_QREF[f"{src}>{dst}"] / q) ** cfg.travel_alpha
            tau = min(max(tau, 0.35 * tau0 * cfg.travel_scale),
                      1.5 * tau0 * cfg.travel_scale)
            lag = tau / DT
            d = int(np.floor(lag))
            f = lag - d
            hh = out_hist[src]
            v0 = hh[-1 - d] if d + 1 <= len(hh) else 0.0
            v1 = hh[-2 - d] if d + 2 <= len(hh) else 0.0
            inflow[i] += (1.0 - f) * v0 + f * v1

        # local catchment rain: R1 drains into the main stem above S2
        inflow[S2] += cfg.t1_gain * r1_rain[t]

        # integrate the storages with a linearly-implicit (unconditionally
        # stable) update:  h+ = h + (I - o) / (A + do/dh)
        for n in range(5):
            kk = cfg.node_rating[n]
            do = (5.0 / 3.0) * kk * max(h[n], 0.0) ** (2.0 / 3.0)
            h[n] = max(h[n] + (inflow[n] - out[n]) / (cfg.node_area[n] + do), 0.0)
            # dispersed outflow history for the delay lines
            prev = out_hist[HYD_NODES[n]]
            prev.append(w0 * out[n] + w1 * prev[-1] + w2 * prev[-2])
        levels[t] = h

    sl = slice(burn_in, n_total)
    return {"levels": levels[sl], "rain1": rain1[sl], "rain2": rain2[sl],
            "release": release[sl], "plan": plan[sl], "u_inflow": u_inflow[sl],
            "r1_rain": r1_rain[sl], "t1_base": t1_base[sl]}


def travel_time(levels, i, j, max_lag=30, win=21):
    """Lag (steps) at which j's fluctuations follow i's (high-pass filtered)."""
    a = levels[:, i] - levels[:, i].mean()
    b = levels[:, j] - levels[:, j].mean()
    k = np.ones(win) / win
    a = a - np.convolve(a, k, mode="same")
    b = b - np.convolve(b, k, mode="same")
    best, best_lag = -np.inf, 0
    for lag in range(0, max_lag + 1):
        c = float(np.corrcoef(a[:-lag or None], b[lag:])[0, 1])
        if c > best:
            best, best_lag = c, lag
    return best_lag, best


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = RiverConfig()
    rng = np.random.default_rng(0)
    out = simulate(cfg, 8000, rng)
    lv = out["levels"]
    print("level mean", np.round(lv.mean(0), 3))
    print("level p01 ", np.round(np.quantile(lv, 0.01, 0), 3))
    print("level p99 ", np.round(np.quantile(lv, 0.99, 0), 3))
    print("release range", out["release"].min(), out["release"].max())
    for (i, j, nm) in ((S1, S2, "S1->S2"), (S2, S3, "S2->S3"), (S3, S4, "S3->S4"),
                       (T1, S2, "T1->S2")):
        print(nm, "lag(steps)=", travel_time(lv, i, j)[0])

    fig, axes = plt.subplots(5, 1, figsize=(12, 11), sharex=True)
    axes[0].plot(lv[:, 1], label="T1")
    axes[0].plot(lv[:, 2], label="S2")
    axes[0].plot(lv[:, 3], label="S3")
    axes[0].plot(lv[:, 4], label="S4")
    axes[0].legend(ncol=4); axes[0].set_ylabel("level (m)")
    axes[1].plot(out["rain1"], label="R1")
    axes[1].plot(out["rain2"], label="R2")
    axes[1].legend(ncol=2); axes[1].set_ylabel("rain (mm/10min)")
    axes[2].plot(out["release"], label="actual release")
    axes[2].plot(out["plan"], label="announced plan")
    axes[2].legend(ncol=2); axes[2].set_ylabel("m3/s")
    axes[3].plot(out["u_inflow"], label="U inflow (latent truth)")
    axes[3].legend(); axes[3].set_ylabel("m3/s")
    axes[4].plot(lv[:, 2], label="S2")
    axes[4].plot(lv[:, 3], label="S3")
    axes[4].plot(lv[:, 4], label="S4")
    axes[4].legend(ncol=3); axes[4].set_ylabel("level (m)"); axes[4].set_xlabel("step (10 min)")
    fig.tight_layout()
    fig.savefig("/tmp/opencode/river_sanity.png", dpi=110)
    print("saved /tmp/opencode/river_sanity.png")
