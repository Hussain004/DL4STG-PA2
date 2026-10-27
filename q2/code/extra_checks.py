"""Extra numbers quoted in Q2.16 and Q2.17 (seed-0 models unless stated).

1. Time-reversed test videos: X = original frames 20..16, Y = original frames 15..11.
   Mean E^(5) of model A and the final model (D symmetrised), both seeds, and of Copy-last / Blank.
2. Centre and peak intensity of the bright mass for test video 9497 under rotation: true T(Y),
   model A f(T(X)) and the final model f(T(X)), per horizon (the quantity used in Q2.7 and Q2.16).
3. Per-video E^(2) of test video 9633 at 1x, 2x, 3x for model A and the final model (Figures 5 and 6).
"""
import os
import sys
import json
import argparse
import torch
import torchvision

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conv_lstm import ConvLSTMBaseline
from conv_lstm_reach import ConvLSTMReach, MODEL_D
from symmetry import Symmetrised

R = lambda x: torch.rot90(x, 1, (-2, -1))


def load_a(path):
    m = ConvLSTMBaseline()
    m.load_state_dict(torch.load(path, map_location="cpu"))
    return m.eval()


def load_final(path):
    m = ConvLSTMReach(**MODEL_D)
    m.load_state_dict(torch.load(path, map_location="cpu"))
    return Symmetrised(m.eval()).eval()


def run(m, x, bs=50):
    with torch.no_grad():
        return torch.cat([m(x[i:i + bs]) for i in range(0, len(x), bs)])


def centre(frame):
    ys, xs = torch.meshgrid(torch.arange(64.0), torch.arange(64.0), indexing="ij")
    w = frame.flatten()
    return dict(x=float((w * xs.flatten()).sum() / w.sum()), y=float((w * ys.flatten()).sum() / w.sum()),
                peak=float(w.max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a-dir", required=True, help="folder with model_a_seed{0,1}.pt")
    ap.add_argument("--d-dir", required=True, help="folder with D_seed{0,1}.pt")
    ap.add_argument("--data-root", default="~/data/moving_mnist")
    ap.add_argument("--out", default="q2/results/final/extra_checks.json")
    a = ap.parse_args()
    ds = torchvision.datasets.MovingMNIST(root=os.path.expanduser(a.data_root), split=None,
                                          download=False, transform=None)
    V = torch.stack([ds[i] for i in range(9000, 10000)]).float() / 255
    mse = lambda p, y: ((p - y) ** 2).mean(dim=(1, 2, 3, 4))
    out = {}

    rev = V.flip(1)
    X, Y, Xr, Yr = V[:, :5], V[:, 5:10], rev[:, :5], rev[:, 5:10]
    out["time_reversed_E5"] = {"copy_last": float(mse(Xr[:, -1:].expand_as(Yr), Yr).mean()),
                               "blank": float((Yr ** 2).mean()),
                               "copy_last_forward": float(mse(X[:, -1:].expand_as(Y), Y).mean())}
    for name, loader, pat, dirn in [("A", load_a, "model_a_seed{}.pt", a.a_dir),
                                    ("final", load_final, "D_seed{}.pt", a.d_dir)]:
        fw, bw = [], []
        for s in (0, 1):
            m = loader(os.path.join(dirn, pat.format(s)))
            fw.append(float(mse(run(m, X), Y).mean()))
            bw.append(float(mse(run(m, Xr), Yr).mean()))
        out["time_reversed_E5"][name] = dict(forward=fw, reversed=bw)

    A0 = load_a(os.path.join(a.a_dir, "model_a_seed0.pt"))
    F0 = load_final(os.path.join(a.d_dir, "D_seed0.pt"))
    v = ds[9497].float() / 255
    ty = R(v[5:10])
    with torch.no_grad():
        pa, pf = A0(R(v[:5])[None])[0], F0(R(v[:5])[None])[0]
    out["video9497_rotation"] = {
        "E_R_model_A": float(((pa - ty) ** 2).mean()), "E_R_final": float(((pf - ty) ** 2).mean()),
        "horizons": [dict(h=h + 1, true=centre(ty[h, 0]), model_A=centre(pa[h, 0]), final=centre(pf[h, 0]))
                     for h in range(5)]}

    v = ds[9633].float() / 255
    out["video9633_E2"] = {}
    for k in (1, 2, 3):
        x, y = v[::k][:5], v[5 * k::k][:2]
        with torch.no_grad():
            pa, pf = A0(x[None])[0][:2], F0(x[None])[0][:2]
        out["video9633_E2"][f"{k}x"] = dict(model_A=float(((pa - y) ** 2).mean()),
                                            final=float(((pf - y) ** 2).mean()),
                                            final_peak=float(pf.max()), model_A_peak=float(pa.max()))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print(json.dumps(out, indent=1)[:1800])


if __name__ == "__main__":
    main()
