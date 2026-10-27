"""Figure 6: Figures 3 and 5 repeated with the final (symmetrised) model on the same videos.

(a) the four Figure 3 panels (test videos 9691 and 9163 for mirror, 9497 and 9964 for rotation),
    saved as a 2 x 2 PDF and one GIF per panel;
(b) the Figure 5 video (test video 9633) at 1x, 2x and 3x.
"""
import os
import sys
import argparse
import numpy as np
import torch
import torchvision
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conv_lstm_reach import ConvLSTMReach, MODEL_D
from symmetry import Symmetrised
from make_transformation_figures import apply_transform, draw_panel, fmt, save_gif
from make_speed_figure import draw_speed_figure, window

PANELS = [("mirror", "Mirror", "M", 1, 9691), ("mirror", "Mirror", "M", 2, 9163),
          ("rot90", "Rotation", "R", 1, 9497), ("rot90", "Rotation", "R", 2, 9964)]
SPEED_VIDEO = 9633


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--no-wrap", action="store_true")
    ap.add_argument("--data-root", default="~/data/moving_mnist")
    ap.add_argument("--out-dir", default="./figures")
    ap.add_argument("--gif-dir", default="./q2/figures")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    os.makedirs(a.out_dir, exist_ok=True)
    os.makedirs(a.gif_dir, exist_ok=True)

    model = ConvLSTMReach(**MODEL_D).to(device)
    model.load_state_dict(torch.load(a.ckpt, map_location=device))
    model.eval()
    f = model if a.no_wrap else Symmetrised(model).eval()
    ds = torchvision.datasets.MovingMNIST(root=os.path.expanduser(a.data_root), split=None,
                                          download=False, transform=None)

    def predict(x):                                   # x: (5, 1, 64, 64) -> (5, 1, 64, 64)
        with torch.no_grad():
            return f(x.unsqueeze(0).to(device)).squeeze(0).cpu()

    # ---- (a) Figure 3 with the final model -----------------------------------
    panels = []
    for tag, label, sym, rank, vid in PANELS:
        frames = ds[vid].float()[:10] / 255.0
        x, y = frames[:5], frames[5:10]
        pred = predict(x)
        pred_t = predict(apply_transform(x, tag))
        rows = [(x, pred), (apply_transform(x, tag), apply_transform(y, tag)),
                (apply_transform(x, tag), pred_t), (apply_transform(x, tag), apply_transform(pred, tag))]
        c = float(((pred_t - apply_transform(pred, tag)) ** 2).mean())
        e = float(((pred_t - apply_transform(y, tag)) ** 2).mean())
        title = (f"{label}: video {vid} ($C_{{{sym}}}$ = {fmt(c)}, $E_{{{sym}}}$ = {fmt(e)})")
        panels.append((tag, label, rank, vid, rows, title))
        print(f"{tag} video {vid}: C = {c:.3e}  E = {e:.4f}")

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 5.8))
    for ax, (_, _, _, _, rows, title) in zip(axes.flat, panels):
        draw_panel(ax, rows, reveal=5, title=title, show_ylabels=True)
    fig.text(0.5, 0.015, "columns 1-5: input frames      columns 6-10: forecast horizons 1-5 "
             "(red line separates input from forecast)", ha="center", fontsize=7)
    plt.tight_layout(rect=(0, 0.035, 1, 1))
    path = os.path.join(a.out_dir, "fig_final_panels.pdf")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved", path)
    for tag, label, rank, vid, rows, title in panels:
        name = "mirror" if tag == "mirror" else "rotation"
        gif = os.path.join(a.gif_dir, f"fig6_{name}_rank{rank}_video{vid}.gif")
        save_gif(rows, title, gif)
        print("Saved", gif)

    # ---- (b) Figure 5 with the final model -------------------------------------
    video = ds[SPEED_VIDEO].float() / 255.0
    rows = []
    for k in (1, 2, 3):
        x, y, h = window(video, k)
        pred = predict(x)[:2]
        rows.append((x, pred, y[:2]))
        print(f"{k}x: per-video E(2) = {float(((pred - y[:2]) ** 2).mean()):.4f}")
    path = os.path.join(a.out_dir, "fig_final_speed.pdf")
    draw_speed_figure(rows, [path])
    print("Saved", path)


if __name__ == "__main__":
    main()
