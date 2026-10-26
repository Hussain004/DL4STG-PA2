"""Evaluate model A under mirror and rotation on the 1,000 test videos.

For T in {M, R} this computes, at 1x with t=1,

    C_T = MSE(f(T(X)), T(f(X)))          consistency gap
    E_T = MSE(f(T(X)), T(Y))             transformed error

with the squared error averaged over the pixels and the five forecast
horizons of each video, then over the 1,000 test videos.  Per-video values
are saved as well, for Figure 3 (top two videos) and Figure 4 (histograms).

The Blank and Copy-last references are exactly equivariant, so their C_T is
zero; their E_T is computed here as a check that it equals their E^(5).
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conv_lstm import ConvLSTMBaseline, count_parameters


class TestVideos:
    """First ten frames of each test video, as float tensors in [0, 1]."""

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        idx = self.indices[position]
        video = self.dataset[idx].float() / 255.0
        return video[:10]


def collate_fn(batch):
    return torch.stack(batch)


def apply_transform(frames, name):
    """Apply M or R to the last two (spatial) dimensions."""
    if name == "mirror":
        return frames.flip(-1)
    if name == "rot90":
        return torch.rot90(frames, k=1, dims=[-2, -1])
    raise ValueError(name)


def per_video_mse(a, b):
    """Mean squared error per video, averaged over frames, channels, pixels."""
    return ((a - b) ** 2).mean(dim=(1, 2, 3, 4))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--data-root", type=str, default="~/data/moving_mnist")
    parser.add_argument("--out-dir", type=str, default="./runs/transform_eval")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    args.data_root = os.path.expanduser(args.data_root)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ConvLSTMBaseline().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    print(f"Loaded model from {args.model_path}")
    print(f"Trainable parameters: {count_parameters(model):,}")

    ds = torchvision.datasets.MovingMNIST(
        root=args.data_root, split=None, download=False, transform=None
    )
    test_indices = list(range(9000, 10000))
    dataset = TestVideos(ds, test_indices)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    names = ["E5", "C_mirror", "E_mirror", "C_rot90", "E_rot90",
             "E5_blank", "E_mirror_blank", "E_rot90_blank",
             "E5_copy", "E_mirror_copy", "E_rot90_copy"]
    per_video = {name: [] for name in names}

    with torch.no_grad():
        for videos in loader:
            videos = videos.to(device, non_blocking=True)
            x = videos[:, :5]
            y = videos[:, 5:10]

            pred = model(x)
            per_video["E5"].append(per_video_mse(pred, y).cpu())

            # Blank and Copy-last references.  Orthogonal transforms preserve
            # the sum of squares, so E_T equals E^(5) for both.
            blank = torch.zeros_like(y)
            copy = x[:, -1:].expand_as(y)
            per_video["E5_blank"].append(per_video_mse(blank, y).cpu())
            per_video["E5_copy"].append(per_video_mse(copy, y).cpu())

            for tag, name in [("mirror", "mirror"), ("rot90", "rot90")]:
                x_t = apply_transform(x, name)
                y_t = apply_transform(y, name)
                pred_t = model(x_t)
                pred_orig_t = apply_transform(pred, name)

                per_video[f"C_{tag}"].append(
                    per_video_mse(pred_t, pred_orig_t).cpu()
                )
                per_video[f"E_{tag}"].append(per_video_mse(pred_t, y_t).cpu())

                per_video[f"E_{tag}_blank"].append(
                    per_video_mse(blank, y_t).cpu()
                )
                # Copy-last commutes with T: f(T(X)) = T(f(X)).
                copy_t = apply_transform(copy, name)
                per_video[f"E_{tag}_copy"].append(
                    per_video_mse(copy_t, y_t).cpu()
                )

    per_video = {
        name: torch.cat(values).numpy().astype(np.float64)
        for name, values in per_video.items()
    }
    n_videos = len(per_video["E5"])

    results = {
        "seed": args.seed,
        "params": count_parameters(model),
        "n_videos": n_videos,
        "test_indices": test_indices,
    }
    for name, values in per_video.items():
        results[name] = float(values.mean())
    results["per_video"] = {name: values.tolist() for name, values in per_video.items()}

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"transform_results_seed{args.seed}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nModel A (seed {args.seed}), {n_videos} test videos:")
    print(f"  E(5)     = {results['E5']:.6f}")
    print(f"  C_M      = {results['C_mirror']:.6f}   E_M = {results['E_mirror']:.6f}")
    print(f"  C_R      = {results['C_rot90']:.6f}   E_R = {results['E_rot90']:.6f}")
    print("\nBlank:")
    print(f"  E(5) = {results['E5_blank']:.6f}  E_M = {results['E_mirror_blank']:.6f}"
          f"  E_R = {results['E_rot90_blank']:.6f}")
    print("Copy-last:")
    print(f"  E(5) = {results['E5_copy']:.6f}  E_M = {results['E_mirror_copy']:.6f}"
          f"  E_R = {results['E_rot90_copy']:.6f}")

    for tag in ["mirror", "rot90"]:
        order = np.argsort(per_video[f"C_{tag}"])[::-1]
        print(f"\nTop C_{tag}:")
        for rank, pos in enumerate(order[:5], 1):
            print(f"  {rank}: video {test_indices[pos]}  C = {per_video[f'C_{tag}'][pos]:.6f}"
                  f"  E = {per_video[f'E_{tag}'][pos]:.6f}")

    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
