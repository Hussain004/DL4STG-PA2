"""Evaluate a trained (optionally symmetrised) forecaster: Table 5 / Table 6 columns.

Per model: E^(5) (1x, t=1), consistency gap C and transformed error E for mirror and rotation,
E^(2) at 1x, 2x, 3x (and E^(5) at 2x), and inference time for the 1,000 original test videos
(1x, t=1) at batch size 32.  --split val evaluates videos 8,000-8,999 instead (design choices).
"""
import os
import sys
import json
import time
import argparse
import torch
import torchvision

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conv_lstm_reach import build_model, count_parameters
from symmetry import Symmetrised

M = lambda x: x.flip(-1)
R = lambda x: torch.rot90(x, k=1, dims=(-2, -1))
SPEED_H = {1: 5, 2: 5, 3: 2}


def pv(a, b):
    """Per-video MSE averaged over frames, channels and pixels: (B, h, 1, 64, 64) -> (B,)."""
    return ((a - b) ** 2).mean(dim=(1, 2, 3, 4))


def evaluate(fn, videos, device, bs=32):
    n = videos.size(0)
    s = dict(E5=0.0, C_M=0.0, E_M=0.0, C_R=0.0, E_R=0.0, E2_1=0.0, E2_2=0.0, E2_3=0.0, E5_2=0.0, peak_h1=0.0)
    with torch.no_grad():
        for i in range(0, n, bs):
            v = videos[i:i + bs].to(device)
            x, y = v[:, :5], v[:, 5:10]
            p = fn(x)
            s["E5"] += pv(p, y).sum().item()
            s["peak_h1"] += p[:, 0].amax(dim=(1, 2, 3)).sum().item()    # brightest pixel of the first forecast
            for tag, T in (("M", M), ("R", R)):
                pt = fn(T(x))
                s[f"C_{tag}"] += pv(pt, T(p)).sum().item()
                s[f"E_{tag}"] += pv(pt, T(y)).sum().item()
            for k in (1, 2, 3):
                h = SPEED_H[k]
                xk, yk = v[:, ::k][:, :5], v[:, 5 * k::k][:, :h]
                pk = fn(xk)
                per_h = torch.stack([pv(pk[:, j:j + 1], yk[:, j:j + 1]) for j in range(h)], dim=1)  # (B, h)
                s[f"E2_{k}"] += per_h[:, :2].mean(1).sum().item()
                if k == 2:
                    s["E5_2"] += per_h.mean(1).sum().item()
    return {k: val / n for k, val in s.items()}


def time_inference(fn, videos, device, bs=32):
    """Wall-clock time to predict all videos at 1x, t=1; timed after each batch is on the device."""
    with torch.no_grad():
        fn(videos[:bs].to(device)[:, :5])           # warm-up (cuDNN autotune), not timed
        total = 0.0
        for i in range(0, videos.size(0), bs):
            x = videos[i:i + bs, :5].to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            fn(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            total += time.time() - t0
    return total


def load_fn(ckpt, arch, patch, dilations, wrap, device, hidden=64, kernels=None):
    model = build_model(arch, patch, dilations, hidden, kernels).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    return (Symmetrised(model).eval() if wrap else model), count_parameters(model)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arch", default="baseline", choices=["baseline", "reach"])
    ap.add_argument("--patch", type=int, default=4)
    ap.add_argument("--dilations", default="1,1")
    ap.add_argument("--kernels", default=None)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--wrap", action="store_true")
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--data-root", default="~/data/moving_mnist")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    fn, n_params = load_fn(a.ckpt, a.arch, a.patch, tuple(int(d) for d in a.dilations.split(",")),
                           a.wrap, device, a.hidden,
                           tuple(int(k) for k in a.kernels.split(",")) if a.kernels else None)
    ds = torchvision.datasets.MovingMNIST(root=os.path.expanduser(a.data_root), split=None,
                                          download=False, transform=None)
    lo = 9000 if a.split == "test" else 8000
    videos = torch.stack([ds[i] for i in range(lo, lo + 1000)]).float() / 255.0
    res = evaluate(fn, videos, device)
    res.update(params=n_params, split=a.split, wrap=a.wrap, ckpt=os.path.basename(a.ckpt))
    if a.split == "test":
        res["inference_time"] = time_inference(fn, videos, device)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
