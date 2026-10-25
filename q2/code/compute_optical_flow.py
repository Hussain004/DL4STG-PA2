#!/usr/bin/env python3
"""Compute dense optical flow on Moving MNIST training videos for Figure 1."""
import os
import numpy as np
import torch
import torchvision
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTDIR = os.path.join(REPO_ROOT, "figures")
os.makedirs(OUTDIR, exist_ok=True)

ds = torchvision.datasets.MovingMNIST(
    root=os.path.expanduser("~/data/moving_mnist"),
    split=None, download=True,
    transform=None,
)
print(f"Dataset size: {len(ds)}")

videos = []
for i in range(50):
    video = ds[i]
    videos.append(video.numpy().astype(np.float32) / 255.0)
videos = np.stack(videos)
print(f"Videos shape: {videos.shape}")

flow_dxs = []
flow_dys = []
kept_count = 0

farneback_params = dict(
    flow=None, pyr_scale=0.5, levels=3, winsize=15,
    iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
)

for v in range(50):
    for s in range(19):
        prev = (videos[v, s, 0] * 255).astype(np.uint8)
        nxt = (videos[v, s + 1, 0] * 255).astype(np.uint8)
        flow = cv2.calcOpticalFlowFarneback(prev, nxt, **farneback_params)
        dx = flow[..., 0].astype(np.float32)
        dy = flow[..., 1].astype(np.float32)

        prev_i = videos[v, s, 0]
        next_i = videos[v, s + 1, 0]
        intensity_mask = (prev_i > 0.2) & (next_i > 0.2)
        mag = np.sqrt(dx ** 2 + dy ** 2)
        mag_mask = mag > 0.5
        keep = intensity_mask & mag_mask
        kept_count += int(keep.sum())
        flow_dxs.append(dx[keep])
        flow_dys.append(dy[keep])

flow_dx = np.concatenate(flow_dxs)
flow_dy = np.concatenate(flow_dys)
print(f"Total kept pixels: {kept_count:,}")

theta = np.degrees(np.arctan2(-flow_dy, flow_dx)) % 360.0
direction_hist, direction_bins = np.histogram(
    theta, bins=18, range=(0.0, 360.0)
)
direction_frac = direction_hist / direction_hist.sum()

speed = np.sqrt(flow_dx ** 2 + flow_dy ** 2)
median_speed = float(np.median(speed))
bin_centers = (direction_bins[:-1] + direction_bins[1:]) / 2
fig, ax = plt.subplots(figsize=(6.5, 3.2))
ax.bar(bin_centers, direction_frac, width=20.0, align="center",
       edgecolor="black", linewidth=0.4, color="#1d6cbd", alpha=0.85)
ax.axhline(1 / 18, color="red", linestyle="--", linewidth=1.2,
           label=f"uniform = {1/18:.3f}")
ax.set_xlabel("Flow direction θ = atan2(−dy, dx) (degrees)")
ax.set_ylabel("Fraction of kept pixels")
ax.set_title("Figure 1(a): Flow direction over Moving MNIST training videos")
ax.set_xlim(0, 360)
ax.set_xticks(np.arange(0, 361, 20))
ax.legend(loc="upper right", fontsize=8)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "fig_flow_direction.pdf"), dpi=300)
plt.close()

speed_max = np.percentile(speed, 99)
speed_bins = np.linspace(0, speed_max, 31)
speed_hist, _ = np.histogram(speed, bins=speed_bins)
speed_frac = speed_hist / speed_hist.sum()
speed_centers = (speed_bins[:-1] + speed_bins[1:]) / 2

fig, ax = plt.subplots(figsize=(6.5, 3.2))
ax.bar(speed_centers, speed_frac, width=speed_bins[1] - speed_bins[0], align="center",
       edgecolor="black", linewidth=0.4, color="#1d6cbd", alpha=0.85)
ax.axvline(median_speed, color="red", linestyle="--", linewidth=1.4,
           label=f"median = {median_speed:.2f} px/frame")
ax.set_xlabel("Flow speed √(dx² + dy²) (pixels per frame)")
ax.set_ylabel("Fraction of kept pixels")
ax.set_title("Figure 1(b): Flow speed over Moving MNIST training videos")
ax.legend(loc="upper right", fontsize=8)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "fig_flow_speed.pdf"), dpi=300)
plt.close()

print(f"Figures saved to {OUTDIR}/")
print(f"Median speed: {median_speed:.3f} px/frame")
print(f"Kept pixels: {kept_count:,}")
print(f"Median speed: {median_speed:.3f} pixels/frame")