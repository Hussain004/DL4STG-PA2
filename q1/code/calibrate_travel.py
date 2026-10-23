"""Auto-calibrate the synthetic river so that the OBSERVED calm travel times
match the tutorial's sketch (Dam->S1 30 min, S1->S2 40 min, S2->S3 50 min,
S3->S4 60 min), and report the flood-case contrast."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import river_sim as rs          # noqa: E402
from travel_probe import arrival_times, step_response  # noqa: E402

TARGETS = {("Dam", "S1"): 30.0, ("S1", "S2"): 40.0, ("T1", "S2"): 20.0,
           ("S2", "S3"): 50.0, ("S3", "S4"): 60.0, ("U", "S3"): 25.0}
NEQ = 2500


def observed_lags(cfg, base_release=6.0, n_eq=NEQ):
    lv, _ = step_response(cfg, base_release, n_eq=n_eq)
    arr = arrival_times(lv, n_eq)
    return arr


def calibrate(iters=6, base_release=6.0, alpha=0.85, verbose=True):
    cfg = rs.RiverConfig(travel_alpha=alpha)
    tau0 = {f"{s}>{d}": t for (s, d, t) in rs.EDGES}
    for it in range(iters):
        arr = observed_lags(cfg, base_release)
        err = {}
        for (s, d), target in TARGETS.items():
            if s == "Dam":
                src_arr = 0.0
            elif s == "U":
                continue
            else:
                src_arr = arr[s]
            meas = (arr[d] - src_arr) * rs.DT
            err[f"{s}>{d}"] = meas
            tau0[f"{s}>{d}"] = max(4.0, tau0[f"{s}>{d}"] - (meas - target))
        # rebuild the edge list with the new tau0
        rs.EDGES = [(s, d, tau0[f"{s}>{d}"]) for (s, d, _) in
                    [tuple(e) for e in
                     [("Dam", "S1", 0), ("S1", "S2", 0), ("T1", "S2", 0),
                      ("S2", "S3", 0), ("S3", "S4", 0), ("U", "S3", 0)]]]
        if verbose:
            print(f"iter {it}: " + "  ".join(f"{k}={v:.0f}" for k, v in err.items()))
    return cfg, tau0, err


if __name__ == "__main__":
    cfg, tau0, err = calibrate()
    print("\ncalibrated tau0 (minutes):", {k: round(v, 1) for k, v in tau0.items()})
    print("calm observed lags (min):", {k: round(v) for k, v in err.items()})

    for tag, base in (("calm", 6.0), ("flood", 13.0), ("flood+", 16.0)):
        arr = observed_lags(cfg, base)
        lags = {}
        for (s, d) in TARGETS:
            if s == "Dam":
                src = 0.0
            elif s == "U":
                continue
            else:
                src = arr[s]
            lags[f"{s}>{d}"] = round((arr[d] - src) * rs.DT)
        print(f"{tag:8s} release={base:5.1f}  lags(min) = {lags}")

    (Path(__file__).resolve().parents[1] / "results").mkdir(parents=True, exist_ok=True)
    (Path(__file__).resolve().parents[1] / "results" / "calibration.json").write_text(
        json.dumps({"tau0": tau0, "alpha": cfg.travel_alpha,
                    "node_area": cfg.node_area, "node_rating": cfg.node_rating}, indent=2))
    print("\nwrote results/calibration.json")
