"""Evaluate Model A on Moving MNIST test set."""
import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conv_lstm import ConvLSTMBaseline, count_parameters


class MovingMNISTTest:
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        idx = self.indices[position]
        video = self.dataset[idx]
        video = video.float() / 255.0
        x = video[0:5]
        y = video[5:10]
        return x, y


def collate_fn(batch):
    xs = torch.stack([b[0] for b in batch])
    ys = torch.stack([b[1] for b in batch])
    return xs, ys


def evaluate(model, test_loader, device):
    model.eval()
    mse_per_horizon = np.zeros(5)
    total_videos = 0
    total_time = 0.0

    with torch.no_grad():
        for xs, ys in test_loader:
            xs = xs.to(device)
            ys = ys.to(device)

            start = time.time()
            preds = model(xs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.time() - start
            total_time += elapsed

            for h in range(5):
                mse_h = F.mse_loss(preds[:, h], ys[:, h], reduction='none')
                mse_h = mse_h.mean(dim=(1, 2, 3))
                mse_per_horizon[h] += mse_h.sum().item()
            total_videos += xs.size(0)

    mse_per_horizon /= total_videos
    e5 = float(mse_per_horizon.mean())
    return mse_per_horizon, e5, total_time, total_videos


def compute_baselines(test_loader, device):
    mse_blank = np.zeros(5)
    mse_copylast = np.zeros(5)
    total_videos = 0

    with torch.no_grad():
        for xs, ys in test_loader:
            xs = xs.to(device)
            ys = ys.to(device)
            batch_size = xs.size(0)

            blank_preds = torch.zeros_like(ys)
            for h in range(5):
                mse_h = F.mse_loss(blank_preds[:, h], ys[:, h], reduction='none')
                mse_blank[h] += mse_h.mean(dim=(1, 2, 3)).sum().item()

            last_frame = xs[:, -1]
            for h in range(5):
                copy_preds = last_frame
                mse_h = F.mse_loss(copy_preds, ys[:, h], reduction='none')
                mse_copylast[h] += mse_h.mean(dim=(1, 2, 3)).sum().item()

            total_videos += batch_size

    mse_blank /= total_videos
    mse_copylast /= total_videos
    e5_blank = float(mse_blank.mean())
    e5_copylast = float(mse_copylast.mean())

    return mse_blank, e5_blank, mse_copylast, e5_copylast


def _has_overlap(video, n_frames=10, threshold=0.3, min_area=20):
    """Return True if any displayed frame contains overlapping/touching digits.

    Two separate digits normally give two large connected components.  When
    they overlap they merge into one, so fewer than two large components is a
    reliable overlap signal for Moving MNIST.
    """
    import cv2

    for f in range(min(n_frames, video.shape[0])):
        img = (video[f, 0].numpy() > threshold).astype(np.uint8)
        _, _, stats, _ = cv2.connectedComponentsWithStats(img, 8)
        large = [int(a) for a in stats[1:, 4] if a > min_area]
        if len(large) < 2:
            return True
    return False


def save_figure_2(model, test_dataset, device, out_dir, n_videos=3):
    model.eval()
    os.makedirs(out_dir, exist_ok=True)

    # Pick videos that actually contain overlapping digits in the displayed
    # input/target frames; fill any remaining slots with the first videos.
    candidates = list(range(len(test_dataset)))
    with_overlap = [
        i for i in candidates
        if _has_overlap(test_dataset.dataset[test_dataset.indices[i]])
    ]
    selected_indices = with_overlap[:n_videos]
    for i in candidates:
        if len(selected_indices) >= n_videos:
            break
        if i not in selected_indices:
            selected_indices.append(i)
    selected_indices.sort()
    print("Figure 2 test positions:", selected_indices,
          "global indices:", [test_dataset.indices[i] for i in selected_indices])

    fig, axes = plt.subplots(n_videos, 3, figsize=(9, 3 * n_videos))
    if n_videos == 1:
        axes = axes.reshape(1, 3)

    titles = ["Input (frames 1-5)", "Prediction (frames 6-10)", "Ground Truth (frames 6-10)"]

    for row, vid_idx in enumerate(selected_indices):
        x, y = test_dataset[vid_idx]
        x = x.unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(x)
        pred = pred.squeeze(0).cpu()

        def make_strip(frames):
            strip = torch.cat([f.squeeze(0) for f in frames], dim=1)
            return strip.cpu().numpy()

        input_strip = make_strip(x.squeeze(0))
        pred_strip = make_strip(pred)
        target_strip = make_strip(y)

        for col, (strip, title) in enumerate(zip([input_strip, pred_strip, target_strip], titles)):
            ax = axes[row, col]
            ax.imshow(strip, cmap="gray", vmin=0, vmax=1, aspect="auto")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(title, fontsize=9)

    plt.tight_layout()
    fig_path = os.path.join(out_dir, "fig_model_a_predictions.pdf")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure 2 saved to {fig_path}")
    return fig_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--data-root', type=str, default='~/data/moving_mnist')
    parser.add_argument('--out-dir', type=str, default='./runs/model_a_eval')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()

    args.data_root = os.path.expanduser(args.data_root)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
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
    test_dataset = MovingMNISTTest(ds, test_indices)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, collate_fn=collate_fn, pin_memory=True
    )

    mse_a, e5_a, inf_time, n_videos = evaluate(model, test_loader, device)
    print(f"\nModel A (seed {args.seed}):")
    for h in range(5):
        print(f"  MSE{h+1} = {mse_a[h]:.6f}")
    print(f"  E(5) = {e5_a:.6f}")
    print(f"  Inference time: {inf_time:.2f}s for {n_videos} videos")

    mse_blank, e5_blank, mse_copy, e5_copy = compute_baselines(test_loader, device)
    print(f"\nBlank:")
    for h in range(5):
        print(f"  MSE{h+1} = {mse_blank[h]:.6f}")
    print(f"  E(5) = {e5_blank:.6f}")
    print(f"\nCopy-last:")
    for h in range(5):
        print(f"  MSE{h+1} = {mse_copy[h]:.6f}")
    print(f"  E(5) = {e5_copy:.6f}")

    os.makedirs(args.out_dir, exist_ok=True)
    import json
    results = {
        "seed": args.seed,
        "params": count_parameters(model),
        "mse_model_a": mse_a.tolist(),
        "e5_model_a": e5_a,
        "mse_blank": mse_blank.tolist(),
        "e5_blank": e5_blank,
        "mse_copy_last": mse_copy.tolist(),
        "e5_copy_last": e5_copy,
        "inference_time": inf_time,
        "n_videos": n_videos,
    }
    with open(os.path.join(args.out_dir, f"results_seed{args.seed}.json"), "w") as f:
        json.dump(results, f, indent=2)

    if args.seed == 0:
        save_figure_2(model, test_dataset, device, args.out_dir)


if __name__ == '__main__':
    main()