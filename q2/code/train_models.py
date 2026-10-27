"""Train models B, C, D (and D+B) on Moving MNIST, same protocol as model A.

Adam lr 1e-3, batch 32, 30 epochs, final-epoch weights, one random window per video per epoch.
  --speed-aug : k ~ U{1,2,3}, then t ~ U{allowed starts for k} (t<=14, 8, 2 at 1x, 2x, 3x)
  --sym-aug   : g ~ U(G), applied to the input and the target
Without --speed-aug the window is a normal-speed one with t ~ U{1..11}.
Loss: squared error averaged over pixels and the available target frames of each video,
then over the videos in the batch.  All randomness (init, shuffling, k, t, g) is seeded by --seed.
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conv_lstm_reach import build_model, count_parameters
from symmetry import apply_g

MAX_T = {1: 14, 2: 8, 3: 2}


class Windows:
    """One random window per video.  torch.randint is reseeded by DataLoader per worker and epoch."""

    def __init__(self, ds, indices, speed_aug, sym_aug):
        self.ds, self.indices = ds, indices
        self.speed_aug, self.sym_aug = speed_aug, sym_aug

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, pos):
        video = self.ds[self.indices[pos]].float() / 255.0              # (20, 1, 64, 64)
        if self.speed_aug:
            k = int(torch.randint(1, 4, (1,)))
            t = int(torch.randint(1, MAX_T[k] + 1, (1,)))
        else:
            k, t = 1, int(torch.randint(1, 12, (1,)))
        inp = (t - 1) + k * torch.arange(5)
        tgt = (t - 1) + k * (5 + torch.arange(5))
        avail = tgt < 20
        x = video[inp]
        y = torch.zeros(5, 1, 64, 64)
        y[avail] = video[tgt[avail]]
        if self.sym_aug:
            g = int(torch.randint(0, 8, (1,)))
            x, y = apply_g(x, g), apply_g(y, g)                          # unavailable frames stay masked
        return x, y, avail.float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--arch", default="baseline", choices=["baseline", "reach"])
    ap.add_argument("--patch", type=int, default=4)
    ap.add_argument("--dilations", default="1,1")
    ap.add_argument("--kernels", default=None)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--speed-aug", action="store_true")
    ap.add_argument("--sym-aug", action="store_true")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--data-root", default="~/data/moving_mnist")
    ap.add_argument("--out-dir", default="./runs/models")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    model = build_model(a.arch, a.patch, tuple(int(d) for d in a.dilations.split(",")), a.hidden,
                        tuple(int(k) for k in a.kernels.split(",")) if a.kernels else None).to(device)
    n_params = count_parameters(model)
    print(f"{a.name} seed {a.seed}: {n_params:,} trainable parameters, device {device}", flush=True)

    ds = torchvision.datasets.MovingMNIST(root=os.path.expanduser(a.data_root), split=None,
                                          download=False, transform=None)
    loader = DataLoader(Windows(ds, list(range(8000)), a.speed_aug, a.sym_aug),
                        batch_size=32, shuffle=True, num_workers=4, pin_memory=True,
                        generator=torch.Generator().manual_seed(a.seed))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    times, losses = [], []
    for ep in range(a.epochs):
        t0, tot, n = time.time(), 0.0, 0
        for x, y, mask in loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            opt.zero_grad()
            per_frame = ((model(x) - y) ** 2).mean(dim=(2, 3, 4))       # (B, 5)
            loss = ((per_frame * mask).sum(1) / mask.sum(1)).mean()     # available frames only
            loss.backward()
            opt.step()
            tot += loss.item() * x.size(0)
            n += x.size(0)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.time() - t0)
        losses.append(tot / n)
        print(f"  epoch {ep + 1}/{a.epochs}: loss={losses[-1]:.6f} time={times[-1]:.1f}s", flush=True)

    os.makedirs(a.out_dir, exist_ok=True)
    tag = f"{a.name}_seed{a.seed}"
    torch.save(model.state_dict(), os.path.join(a.out_dir, tag + ".pt"))
    meta = dict(name=a.name, seed=a.seed, arch=a.arch, patch=a.patch, dilations=a.dilations, kernels=a.kernels, hidden=a.hidden,
                speed_aug=a.speed_aug, sym_aug=a.sym_aug, params=n_params,
                epoch_times=times, mean_epoch_time=float(np.mean(times)), losses=losses,
                gpu=torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu")
    with open(os.path.join(a.out_dir, tag + "_train.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"{tag}: mean epoch time {meta['mean_epoch_time']:.2f}s, saved", flush=True)


if __name__ == "__main__":
    main()
