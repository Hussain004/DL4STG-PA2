"""Build Figure 3 (transformation panels + GIFs) and Figure 4 (histograms).

Figure 3 uses the seed-0 model A and the two test videos with the largest
per-video C_M, plus the two with the largest per-video C_R.  Each panel has
four rows and ten columns: the five input frames and five future frames.

The GIF for each panel steps through forecast horizons 1-5, revealing one
future column at a time while keeping the same four rows.
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
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conv_lstm import ConvLSTMBaseline


def apply_transform(frames, name):
    if name == "mirror":
        return frames.flip(-1)
    if name == "rot90":
        return torch.rot90(frames, k=1, dims=[-2, -1])
    raise ValueError(name)


def fmt(x):
    """Three significant figures, scientific notation below 1e-3."""
    if x == 0:
        return "0"
    if abs(x) < 1e-3:
        return f"{x:.2e}"
    return f"{x:.3g}"


def compose_panel(rows, reveal=5, gap=2):
    """Stack four rows of ten frames into one grayscale image.

    rows: list of (input_frames, future_frames), each (5,1,64,64).
    reveal: number of future columns to show (1..5).
    """
    h, w = rows[0][0].shape[-2:]
    n_rows, n_cols = len(rows), 10
    canvas = np.ones(
        (n_rows * h + (n_rows - 1) * gap, n_cols * w + (n_cols - 1) * gap),
        dtype=np.float32,
    )
    for r, (inp, fut) in enumerate(rows):
        strip = torch.cat([inp, fut], dim=0).squeeze(1).cpu().numpy()
        for c in range(n_cols):
            if c >= 5 and c - 5 >= reveal:
                continue
            y = r * (h + gap)
            x = c * (w + gap)
            canvas[y:y + h, x:x + w] = strip[c]
    return canvas


def draw_panel(ax, rows, reveal, title, show_ylabels=True):
    ax.clear()
    img = compose_panel(rows, reveal=reveal)
    h, w = rows[0][0].shape[-2:]
    gap = 2
    ax.imshow(img, cmap="gray", vmin=0, vmax=1, aspect="equal",
              interpolation="nearest")
    ax.set_xticks([c * (w + gap) + w / 2 for c in range(10)])
    ax.set_xticklabels([str(c + 1) for c in range(10)], fontsize=5.5)
    ax.tick_params(axis="x", length=0, pad=1)
    ax.set_yticks([r * (h + gap) + h / 2 for r in range(4)])
    if show_ylabels:
        ax.set_yticklabels(
            [r"(1) $X$ | $f(X)$",
             r"(2) $T(X)$ | $T(Y)$",
             r"(3) $T(X)$ | $f(T(X))$",
             r"(4) $T(X)$ | $T(f(X))$"],
            fontsize=6.5,
        )
    else:
        ax.set_yticklabels([])
    ax.tick_params(axis="y", length=0, pad=2)
    # Red line between the five input frames and the five future frames.
    ax.axvline(5 * w + 4.5 * gap, color="#d62728", lw=1.0)
    ax.set_title(title, fontsize=8, pad=4)
    for spine in ax.spines.values():
        spine.set_visible(False)


def save_gif(rows, title, gif_path):
    """One GIF per panel: the four rows stay fixed, one forecast column is revealed per frame."""
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    fig.subplots_adjust(left=0.2, right=0.99, top=0.9, bottom=0.08)
    # Render each frame ourselves: FuncAnimation/PillowWriter truncates a
    # 9.2 in x 100 dpi figure to 919 px but writes a 920 px buffer, which
    # shears every frame.
    frames = []
    for h in range(1, 6):
        draw_panel(ax, rows, reveal=h, title=title + f"   -   horizon {h}",
                   show_ylabels=True)
        fig.canvas.draw()
        frames.append(Image.fromarray(
            np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()))
    frames[0].save(gif_path, save_all=True, append_images=frames[1:],
                   duration=700, loop=0)
    plt.close(fig)


def plot_histograms(per_video, selections, path):
    """Figure 4: per-video C_M and C_R with the Figure 3 videos marked.

    The two marked videos can have nearly equal C, so they are labelled in a
    legend rather than with annotations on the lines (which overlapped).
    """
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8))
    colors = {1: "#d62728", 2: "#1f77b4"}
    for ax, tag, label, symbol in [
        (axes[0], "mirror", "Mirror", "M"),
        (axes[1], "rot90", "Rotation", "R"),
    ]:
        values = np.asarray(per_video[f"C_{tag}"])
        ax.hist(values, bins=50, color="#7f7f7f", edgecolor="white", linewidth=0.3)
        ax.set_yscale("log")
        ax.set_xlabel(f"per-video $C_{{{symbol}}}$")
        ax.set_ylabel("number of test videos")
        ax.set_title(f"{label} consistency gap, seed-0 model A", fontsize=9)
        for sel in [s for s in selections if s["tag"] == tag]:
            ax.axvline(sel["C"], color=colors[sel["rank"]],
                       ls="--" if sel["rank"] == 1 else "-.", lw=1.5,
                       label=f"video {sel['video']} (rank {sel['rank']})")
        ax.legend(loc="upper right", bbox_to_anchor=(0.88, 1.0), fontsize=7.5,
                  frameon=False)
        ax.grid(alpha=0.2, axis="y")
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--data-root", type=str, default="~/data/moving_mnist")
    parser.add_argument("--results", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="./figures")
    parser.add_argument("--gif-dir", type=str, default="./q2/figures")
    parser.add_argument("--preview-dir", type=str, default="./runs/preview")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    args.data_root = os.path.expanduser(args.data_root)
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.gif_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    with open(args.results) as f:
        results = json.load(f)
    per_video = results["per_video"]
    test_indices = results["test_indices"]

    model = ConvLSTMBaseline().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    ds = torchvision.datasets.MovingMNIST(
        root=args.data_root, split=None, download=False, transform=None
    )

    # Top two videos for each transformation.  A video may appear twice if it
    # is in both lists; the handout asks for it once per transformation.
    selections = []
    for tag, label, symbol in [("mirror", "Mirror", "M"), ("rot90", "Rotation", "R")]:
        values = np.asarray(per_video[f"C_{tag}"])
        order = np.argsort(values)[::-1]
        for rank in range(2):
            pos = int(order[rank])
            selections.append({
                "tag": tag,
                "label": label,
                "symbol": symbol,
                "rank": rank + 1,
                "pos": pos,
                "video": test_indices[pos],
                "C": float(values[pos]),
            })

    panels = []
    for sel in selections:
        video = ds[sel["video"]].float() / 255.0
        frames = video[:10]
        x = frames[:5].unsqueeze(0).to(device)
        y = frames[5:10]
        with torch.no_grad():
            pred = model(x).squeeze(0).cpu()
        x_t = apply_transform(x, sel["tag"])
        with torch.no_grad():
            pred_t = model(x_t).squeeze(0).cpu()
        y_t = apply_transform(y, sel["tag"])
        pred_orig_t = apply_transform(pred, sel["tag"])

        rows = [
            (x.squeeze(0).cpu(), pred),
            (x_t.squeeze(0).cpu(), y_t),
            (x_t.squeeze(0).cpu(), pred_t),
            (x_t.squeeze(0).cpu(), pred_orig_t),
        ]
        panels.append((sel, rows))

    # ---- static Figure 3: 2 x 2 panels --------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 5.8))
    for ax, (sel, rows) in zip(axes.flat, panels):
        title = (f"{sel['label']}: video {sel['video']} "
                 f"({'largest' if sel['rank'] == 1 else '2nd largest'} "
                 f"$C_{{{sel['symbol']}}}$ = {fmt(sel['C'])})")
        draw_panel(ax, rows, reveal=5, title=title, show_ylabels=True)
    fig.text(0.5, 0.015,
             "columns 1-5: input frames      columns 6-10: forecast horizons 1-5 "
             "(red line separates input from forecast)",
             ha="center", fontsize=7)
    plt.tight_layout(rect=(0, 0.035, 1, 1))
    os.makedirs(args.preview_dir, exist_ok=True)
    fig_path = os.path.join(args.out_dir, "fig_transformation_panels.pdf")
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fig_path}")

    # ---- one GIF per panel -------------------------------------------
    for sel, rows in panels:
        title = (f"{sel['label']}: video {sel['video']} "
                 f"({'largest' if sel['rank'] == 1 else '2nd largest'} "
                 f"$C_{{{sel['symbol']}}}$ = {fmt(sel['C'])})")
        tag_name = "mirror" if sel["tag"] == "mirror" else "rotation"
        gif_path = os.path.join(
            args.gif_dir,
            f"fig3_{tag_name}_rank{sel['rank']}_video{sel['video']}.gif",
        )
        save_gif(rows, title, gif_path)
        print(f"Saved {gif_path}")

    # ---- Figure 4: per-video histograms ------------------------------
    plot_histograms(per_video, selections, os.path.join(
        args.out_dir, "fig_consistency_histograms.pdf"))

    # Also save a PNG preview of Figure 3 for quick inspection.
    preview_path = os.path.join(args.preview_dir, "fig_transformation_panels_preview.png")
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 5.8))
    for ax, (sel, rows) in zip(axes.flat, panels):
        title = (f"{sel['label']}: video {sel['video']} "
                 f"({'largest' if sel['rank'] == 1 else '2nd largest'} "
                 f"$C_{{{sel['symbol']}}}$ = {fmt(sel['C'])})")
        draw_panel(ax, rows, reveal=5, title=title, show_ylabels=True)
    plt.tight_layout()
    plt.savefig(preview_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {preview_path}")


if __name__ == "__main__":
    main()
