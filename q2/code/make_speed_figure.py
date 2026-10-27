"""Render Figure 5: model A on one test video at 1x, 2x and 3x (seed 0).

Each row shows the five input frames, the first two predicted frames and
the first two true future frames of a speed-up test video:

    1x: input frames 1,2,3,4,5       forecast frames 6,7
    2x: input frames 1,3,5,7,9       forecast frames 11,13
    3x: input frames 1,4,7,10,13     forecast frames 16,19

The video is chosen automatically, following the handout rule that its
digits must not overlap in any of the six displayed true frames.  Among
the videos that satisfy this the script keeps the one with the largest
digit area, so the figure stays readable.
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
import torchvision
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conv_lstm import ConvLSTMBaseline

MAX_H = {1: 5, 2: 5, 3: 2}
TRUE_FRAMES = {1: "6-7", 2: "11, 13", 3: "16, 19"}


def component_areas(mask):
    """Areas of the connected components of a boolean 64x64 frame."""
    mask = np.asarray(mask, dtype=bool)
    try:
        from scipy import ndimage

        labels, n = ndimage.label(mask, structure=np.ones((3, 3)))
        if n == 0:
            return []
        areas = ndimage.sum_labels(mask, labels, index=np.arange(1, n + 1))
        return [float(a) for a in areas]
    except ImportError:
        visited = np.zeros_like(mask, dtype=bool)
        areas = []
        h, w = mask.shape
        for y0, x0 in np.argwhere(mask):
            if visited[y0, x0]:
                continue
            stack = [(y0, x0)]
            visited[y0, x0] = True
            area = 0
            while stack:
                y, x = stack.pop()
                area += 1
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dy == 0 and dx == 0:
                            continue
                        yy, xx = y + dy, x + dx
                        if (0 <= yy < h and 0 <= xx < w
                                and mask[yy, xx] and not visited[yy, xx]):
                            visited[yy, xx] = True
                            stack.append((yy, xx))
            areas.append(float(area))
        return areas


def displayed_summary(video, threshold, min_area):
    """(ok, total digit area) for the six displayed true frames."""
    ok, area = True, 0.0
    for k in (1, 2, 3):
        for h in (0, 1):
            frame = video[k * (5 + h), 0]
            large = [a for a in component_areas(frame > threshold) if a >= min_area]
            if len(large) < 2:
                ok = False
                area = 0.0
                break
            area += sum(large)
        if not ok:
            break
    return ok, area


def window(frames, k):
    """Input, targets and horizon count for speed k at t = 1 (0-based)."""
    h = MAX_H[k]
    x = frames[::k][:5]
    y = frames[5 * k::k][:h]
    return x, y, h


def mse_video(pred, y):
    return float(((pred - y) ** 2).mean(dim=(1, 2, 3)).mean())


def draw_speed_figure(rows, paths):
    """rows: [(input, pred, true)] for k = 1, 2, 3; saves the figure to every path in `paths`."""
    h, w = 64, 64
    gap = 2
    wide = 14
    col_start = [0]
    for c in range(1, 9):
        col_start.append(col_start[-1] + w + (wide if c == 7 else gap))
    canvas_w = col_start[8] + w
    canvas_h = 3 * h + 2 * gap
    canvas = np.ones((canvas_h, canvas_w), dtype=np.float32)
    for r, (inp, pred, true) in enumerate(rows):
        strip = torch.cat([inp, pred, true], dim=0).squeeze(1).cpu().numpy()
        for c in range(9):
            xoff = col_start[c]
            yoff = r * (h + gap)
            canvas[yoff:yoff + h, xoff:xoff + w] = strip[c]

    fig, ax = plt.subplots(figsize=(8.6, 2.7))
    ax.imshow(canvas, cmap="gray", vmin=0, vmax=1, aspect="equal",
              interpolation="nearest")
    ax.set_yticks([r * (h + gap) + h / 2 for r in range(3)])
    ax.set_yticklabels(
        [f"$1\\times$\ntrue {TRUE_FRAMES[1]}",
         f"$2\\times$\ntrue {TRUE_FRAMES[2]}",
         f"$3\\times$\ntrue {TRUE_FRAMES[3]}"],
        fontsize=7,
    )
    ax.set_xticks([
        (col_start[0] + col_start[4] + w) / 2,
        (col_start[5] + col_start[6] + w) / 2,
        (col_start[7] + col_start[8] + w) / 2,
    ])
    ax.set_xticklabels(["input", "pred", "true"], fontsize=8.5)
    ax.tick_params(length=0, pad=2)
    ax.axvline(col_start[4] + w + gap / 2, color="#d62728", lw=1.0)
    ax.axvline(col_start[6] + w + wide / 2, color="#1f77b4", lw=1.0, ls=":")
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout()
    for path in paths:
        plt.savefig(path, dpi=200 if path.endswith(".pdf") else 150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--data-root", type=str, default="~/data/moving_mnist")
    parser.add_argument("--out-dir", type=str, default="./runs/speed_figure")
    parser.add_argument("--preview-dir", type=str, default="./runs/preview")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--min-area", type=int, default=20)
    args = parser.parse_args()
    args.data_root = os.path.expanduser(args.data_root)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ds = torchvision.datasets.MovingMNIST(
        root=args.data_root, split=None, download=False, transform=None
    )

    # Choose the video to display.
    candidates = []
    for idx in range(9000, 10000):
        video = ds[idx].float() / 255.0
        ok, area = displayed_summary(video, args.threshold, args.min_area)
        if ok:
            candidates.append({"video": idx, "area": area})
            if len(candidates) >= 200:
                break
    print(f"Test videos with two separate digits in all six displayed "
          f"true frames: {len(candidates)}")
    for c in candidates[:8]:
        print(f"  video {c['video']}  digit area {c['area']:.0f}")
    if not candidates:
        raise SystemExit("No test video satisfies the non-overlap condition.")

    chosen = max(candidates, key=lambda c: (c["area"], -c["video"]))
    print(f"Chosen video: {chosen['video']}")

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.preview_dir, exist_ok=True)

    model = ConvLSTMBaseline().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    video = ds[chosen["video"]].float() / 255.0
    rows, e2 = [], {}
    for k in (1, 2, 3):
        x, y, h = window(video, k)
        with torch.no_grad():
            pred = model(x.unsqueeze(0).to(device)).squeeze(0).cpu()[:2]
        true = y[:2]
        rows.append((x, pred, true))
        e2[str(k)] = mse_video(pred, true)
        print(f"  {k}x: per-video E(2) = {e2[str(k)]:.6f}")

    fig_path = os.path.join(args.out_dir, "fig_speed_panels.pdf")
    png_path = os.path.join(args.preview_dir, "fig_speed_panels_preview.png")
    draw_speed_figure(rows, [fig_path, png_path])
    with open(os.path.join(args.out_dir, "chosen.json"), "w") as f:
        json.dump({"video": chosen["video"], "e2": e2,
                   "threshold": args.threshold, "min_area": args.min_area,
                   "candidates": candidates[:50]}, f, indent=2)
    print(f"Saved {fig_path}")
    print(f"Saved {png_path}")


if __name__ == "__main__":
    main()
