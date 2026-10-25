#!/usr/bin/env python3
"""Reproduce Table 1 of Question 2.

The checks use the same TransformDataset/window code that the experiments use,
so a mismatch here means the experimental windows are wrong.
"""
import sys
import os

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_transformations import TransformDataset


class _FakeDataset:
    """One-video dataset wrapper used only for the frame-number checks."""

    def __init__(self, video):
        self.video = video

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return self.video


def mirror(x):
    return x.flip(-1)


def rotate(x):
    return torch.rot90(x, k=1, dims=(-2, -1))


def bright_pixel(x):
    coords = (x > 0.5).nonzero()
    if coords.numel() == 0:
        return None
    frame, _channel, row, col = coords[0].tolist()
    return (frame, row, col)


def window_frames(k, t=1):
    video = torch.zeros(20, 1, 64, 64, dtype=torch.uint8)
    for f in range(20):
        video[f].fill_((f + 1) * 10)
    dataset = TransformDataset(_FakeDataset(video), [0], transform="speed", k=k, t=t)
    x, y = dataset[0]
    # Undo the /255 conversion and the factor 10 used to encode frame numbers.
    to_frame = lambda frames: [
        int(round(float(v) * 255.0 / 10.0)) for v in frames[:, 0, 0, 0]
    ]
    return to_frame(x), to_frame(y)


def main():
    torch.manual_seed(0)
    x = torch.rand(20, 1, 64, 64)
    p = torch.zeros(20, 1, 64, 64)
    p[7, 0, 10, 20] = 1

    print("Table 1 checks")
    print("-" * 72)
    print(f"(a) max|M(M(x))-x|          = {(mirror(mirror(x)) - x).abs().max().item():g}"
          f"  torch.equal={torch.equal(mirror(mirror(x)), x)}")
    print(f"(b) max|R4(x)-x|            = {(rotate(rotate(rotate(rotate(x)))) - x).abs().max().item():g}"
          f"  torch.equal={torch.equal(rotate(rotate(rotate(rotate(x)))), x)}")
    r_inv = rotate(rotate(rotate(x)))
    print(f"(c) max|M R M(x)-R^-1(x)|   = {(mirror(rotate(mirror(x))) - r_inv).abs().max().item():g}"
          f"  torch.equal={torch.equal(mirror(rotate(mirror(x))), r_inv)}")
    print(f"(d) bright pixel of M(p)    = {bright_pixel(mirror(p))}")
    print(f"(e) bright pixel of R(p)    = {bright_pixel(rotate(p))}")
    print(f"(f) 1x input / target       = {window_frames(1)[0]} / {window_frames(1)[1]}")
    print(f"(g) 2x input / target       = {window_frames(2)[0]} / {window_frames(2)[1]}")
    print(f"(h) 3x input / target       = {window_frames(3)[0]} / {window_frames(3)[1]}")
    print("-" * 72)

    expected = {
        "a": 0.0,
        "b": 0.0,
        "c": 0.0,
        "d": (7, 10, 43),
        "e": (7, 43, 10),
        "f": ([1, 2, 3, 4, 5], [6, 7, 8, 9, 10]),
        "g": ([1, 3, 5, 7, 9], [11, 13, 15, 17, 19]),
        "h": ([1, 4, 7, 10, 13], [16, 19]),
    }
    actual = {
        "a": (mirror(mirror(x)) - x).abs().max().item(),
        "b": (rotate(rotate(rotate(rotate(x)))) - x).abs().max().item(),
        "c": (mirror(rotate(mirror(x))) - r_inv).abs().max().item(),
        "d": bright_pixel(mirror(p)),
        "e": bright_pixel(rotate(p)),
        "f": window_frames(1),
        "g": window_frames(2),
        "h": window_frames(3),
    }
    ok = True
    for key in expected:
        if actual[key] != expected[key]:
            ok = False
            print(f"MISMATCH ({key}): expected {expected[key]}, got {actual[key]}")
    print("ALL CHECKS PASS" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
