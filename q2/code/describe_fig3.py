"""Per-horizon centroid diagnostics for the Figure 3 videos (seed-0 model A).

For the four Figure 3 videos this records, for every horizon h = 1..5, the
intensity-weighted centroid (x, y) of

    (1) f(X)       forecast from the original input
    (2) T(Y)       true future after the transformation
    (3) f(T(X))    forecast from the transformed input
    (4) T(f(X))    transformed copy of the original forecast

plus the per-horizon squared difference between rows 3 and 4.  The mean of
these per-horizon values over the five horizons equals the per-video C_T
stored in q2/results/transform_results_seed0.json, so the numbers can be
read next to the figure directly.
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
import torchvision

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conv_lstm import ConvLSTMBaseline

VIDEOS = [("mirror", "M", 9691), ("mirror", "M", 9163),
          ("rot90", "R", 9497), ("rot90", "R", 9964)]


def apply_transform(frames, name):
    if name == "mirror":
        return frames.flip(-1)
    if name == "rot90":
        return torch.rot90(frames, k=1, dims=[-2, -1])
    raise ValueError(name)


def centroid(frame):
    """Intensity-weighted centroid (x, y) of one 64x64 frame."""
    w = frame.flatten()
    ys, xs = torch.meshgrid(
        torch.arange(frame.shape[-2], dtype=torch.float32),
        torch.arange(frame.shape[-1], dtype=torch.float32),
        indexing="ij",
    )
    total = w.sum()
    if total <= 0:
        return float("nan"), float("nan")
    cx = float((w * xs.flatten()).sum() / total)
    cy = float((w * ys.flatten()).sum() / total)
    return cx, cy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--data-root", type=str, default="~/data/moving_mnist")
    parser.add_argument("--out-dir", type=str, default="./runs/fig3_diagnostics")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    args.data_root = os.path.expanduser(args.data_root)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = ConvLSTMBaseline().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    ds = torchvision.datasets.MovingMNIST(
        root=args.data_root, split=None, download=False, transform=None
    )

    results = {"seed": args.seed, "videos": {}}
    for tag, symbol, vid in VIDEOS:
        video = ds[vid].float() / 255.0
        x = video[:5].unsqueeze(0).to(device)
        y = video[5:10]
        with torch.no_grad():
            pred = model(x).squeeze(0).cpu()
            pred_t = model(apply_transform(x, tag)).squeeze(0).cpu()
        y_t = apply_transform(y, tag)
        pred_orig_t = apply_transform(pred, tag)

        entry = {"symbol": symbol, "horizons": []}
        diff_total = 0.0
        for h in range(5):
            diff = float(((pred_t[h] - pred_orig_t[h]) ** 2).mean())
            diff_total += diff
            row = {"h": h + 1, "mse_rows34": diff}
            for key, frame in [
                ("c_fX", pred[h]), ("c_TY", y_t[h]),
                ("c_fTX", pred_t[h]), ("c_TfX", pred_orig_t[h]),
            ]:
                cx, cy = centroid(frame)
                row[key] = [round(cx, 2), round(cy, 2)]
            entry["horizons"].append(row)
        entry["C_total"] = diff_total / 5
        results["videos"][f"{tag}_{vid}"] = entry

        print(f"{tag} video {vid}:  C (rows 3-4 mean) = {entry['C_total']:.6f}")
        print("  h   fTX (x,y)        TfX (x,y)        dy")
        for row in entry["horizons"]:
            fx, fy = row["c_fTX"]
            gx, gy = row["c_TfX"]
            print(f"  {row['h']}   ({fx:5.1f},{fy:5.1f})   ({gx:5.1f},{gy:5.1f})"
                  f"   {fy - gy:+6.2f}")

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "fig3_diagnostics.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
