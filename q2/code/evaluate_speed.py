"""Evaluate model A on sped-up test videos (Table 4 and Figure 5 data).

For speed k in {1, 2, 3} and t = 1, the model sees the five input frames

    X = frames 1, 1+k, 1+2k, 1+3k, 1+4k

and is scored against the targets

    Y = frames 1+5k, 1+6k, ...   (stride k)

The video has 20 frames, so at k = 1 and 2 there are five targets and at
k = 3 only two (frames 16 and 19).  Following the handout,

    E(2)@k = MSE over the first two horizons of speed k
    E(5)@k = MSE over all five horizons of speed k        (k = 1, 2)

The squared error is averaged over the pixels and the horizons of each
video, then over the 1,000 test videos.  Copy-last is computed the same
way at each speed (it copies the last input frame).

Timing follows evaluate_model_a.py: a torch.cuda.synchronize around each
model call, batches are timed after they reach the device.  Timing is per
model call; Copy-last is a plain slice and is not timed.
"""
import os
import sys
import json
import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
import torchvision

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conv_lstm import ConvLSTMBaseline, count_parameters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--data-root", type=str, default="~/data/moving_mnist")
    parser.add_argument("--out-dir", type=str, default="./runs/speed_eval")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    args.data_root = os.path.expanduser(args.data_root)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ConvLSTMBaseline().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    ds = torchvision.datasets.MovingMNIST(
        root=args.data_root, split=None, download=False, transform=None
    )
    test_indices = list(range(9000, 10000))
    # All test videos as one float tensor: (N, 20, 1, 64, 64).
    stack = torch.stack([ds[i] for i in test_indices]).float() / 255.0
    n = stack.size(0)

    n_h = {1: 5, 2: 5, 3: 2}
    results = {
        "seed": args.seed,
        "params": count_parameters(model),
        "n_videos": n,
        "speeds": {},
    }
    for k in (1, 2, 3):
        h = n_h[k]
        mse_sum_model = np.zeros(h)
        mse_sum_copy = np.zeros(h)
        total = 0
        total_time = 0.0
        with torch.no_grad():
            for start in range(0, n, args.batch_size):
                batch = stack[start:start + args.batch_size].to(device)
                x = batch[:, ::k][:, :5]            # frames 1, 1+k, ..., 1+4k
                y = batch[:, 5 * k::k][:, :h]       # frames 1+5k, 1+6k, ...
                total += batch.size(0)

                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.time()
                pred = model(x)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                total_time += time.time() - t0

                for i in range(h):
                    mse_i = F.mse_loss(pred[:, i], y[:, i], reduction="none")
                    mse_i = mse_i.mean(dim=(1, 2, 3))
                    mse_sum_model[i] += mse_i.sum().item()

                copy = x[:, -1:].expand_as(y)
                for i in range(h):
                    mse_i = F.mse_loss(copy[:, i], y[:, i], reduction="none")
                    mse_i = mse_i.mean(dim=(1, 2, 3))
                    mse_sum_copy[i] += mse_i.sum().item()

        mse_model = (mse_sum_model / total).tolist()
        mse_copy = (mse_sum_copy / total).tolist()
        entry = {
            "n_horizons": h,
            "mse_model": mse_model,
            "mse_copy": mse_copy,
            "e2_model": float(np.mean(mse_model[:2])),
            "e2_copy": float(np.mean(mse_copy[:2])),
            "ms_per_video_model": 1000.0 * total_time / total,
        }
        if h == 5:
            entry["e5_model"] = float(np.mean(mse_model))
            entry["e5_copy"] = float(np.mean(mse_copy))
        results["speeds"][str(k)] = entry
        print(f"speed {k}x (t=1), {h} horizons, {total} videos:")
        print(f"  E(2) model = {entry['e2_model']:.6f}   copy = {entry['e2_copy']:.6f}")
        if h == 5:
            print(f"  E(5) model = {entry['e5_model']:.6f}   copy = {entry['e5_copy']:.6f}")
        print(f"  per-horizon model = {[f'{v:.6f}' for v in mse_model]}")
        print(f"  {entry['ms_per_video_model']:.3f} ms per video")

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"results_speed_seed{args.seed}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
