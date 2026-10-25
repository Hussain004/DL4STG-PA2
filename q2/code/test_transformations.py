"""Run model A under mirror, rotation, and speed-up transformations on the test set."""
import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conv_lstm import ConvLSTMBaseline, count_parameters


class TransformDataset:
    def __init__(self, dataset, indices, transform, k=1, t=1):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
        self.k = k
        self.t = t

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        idx = self.indices[position]
        video = self.dataset[idx]  # (20, 1, 64, 64) uint8
        video = video.float() / 255.0  # (20, 1, 64, 64)
        frames = video  # (T=20, C=1, H=64, W=64)

        if self.transform == 'mirror':
            frames = frames.flip(-1)
        elif self.transform == 'rot90':
            frames = torch.rot90(frames, k=1, dims=[-2, -1])

        # Window in original frame numbers (1-based). At speed k, the input is
        # t, t+k, ..., t+4k and the target is t+5k, ..., t+9k. Slicing the
        # original 20-frame video (rather than a pre-subsampled copy) keeps t
        # as the original frame number for every k.
        start = self.t - 1
        x = frames[start:start + 5 * self.k:self.k]              # (5, C, H, W)
        y = frames[start + 5 * self.k:start + 10 * self.k:self.k]  # (<=5, C, H, W)
        return x, y


def collate_fn(batch):
    xs = torch.stack([b[0] for b in batch])
    ys = torch.stack([b[1] for b in batch])
    return xs, ys


def run_eval(model, dataset, device):
    model.eval()
    loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    mse_per_horizon = np.zeros(5)
    total_videos = 0
    n_h = 0
    with torch.no_grad():
        for xs, ys in loader:
            xs = xs.to(device)
            ys = ys.to(device)
            preds = model(xs)
            n_h = min(5, preds.size(1), ys.size(1))
            for h in range(n_h):
                mse_h = F.mse_loss(preds[:, h], ys[:, h], reduction='none')
                mse_h = mse_h.mean(dim=(1, 2, 3))
                mse_per_horizon[h] += mse_h.sum().item()
            total_videos += xs.size(0)
    mse_per_horizon /= total_videos
    e2 = float(mse_per_horizon[:2].mean())
    # E^(5) is only defined when all five target frames exist.
    e5 = float(mse_per_horizon[:n_h].mean()) if n_h == 5 else float('nan')
    return mse_per_horizon, e2, e5, total_videos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--data-root', type=str, default='~/data/moving_mnist')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--transform', type=str, default=None, choices=['mirror', 'rot90', 'speed'])
    parser.add_argument('--k', type=int, default=1)
    parser.add_argument('--t', type=int, default=1)
    args = parser.parse_args()
    args.data_root = os.path.expanduser(args.data_root)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = ConvLSTMBaseline().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    print(f"Loaded model from {args.model_path}, transform={args.transform}, k={args.k}, t={args.t}")

    ds = torchvision.datasets.MovingMNIST(root=args.data_root, split=None, download=False, transform=None)
    test_indices = list(range(9000, 10000))
    dataset = TransformDataset(ds, test_indices, transform=args.transform, k=args.k, t=args.t)
    mse, e2, e5, n = run_eval(model, dataset, device)
    print(f"\nModel A (seed {args.seed}) transform={args.transform} k={args.k} t={args.t}:")
    for h in range(5):
        print(f"  MSE{h+1} = {mse[h]:.6f}")
    print(f"  E(2) = {e2:.6f}")
    if np.isfinite(e5):
        print(f"  E(5) = {e5:.6f}")
    else:
        print("  E(5) = n/a (fewer than five target frames)")


if __name__ == '__main__':
    main()
