"""Unit tests for the river models.

The most important one is LOCALITY: a model that claims to have no spatial
path must be *provably* unable to see another place's inputs.  The companion
notebook's whole argument is about information boundaries, so a silent leak
would invalidate every conclusion.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import river_data as rd          # noqa: E402
import river_models as rm        # noqa: E402

FAILED = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")
    if not ok:
        FAILED.append(name)


def test_encoder_locality():
    torch.manual_seed(0)
    spec = rd.GridSpec(24, 6)
    m = rm.build("temporal", spec).eval()
    x = torch.randn(3, 30, 6, 4)
    x2 = x.clone()
    x2[:, :, 3] = torch.randn_like(x2[:, :, 3])          # randomise the S2 row
    with torch.no_grad():
        z = m.encoder(x, m.place_emb)
        z2 = m.encoder(x2, m.place_emb)
    diff = (z2 - z).abs().amax(dim=(0, 1, 3))
    check("encoder is place-local", float(diff[5]) == 0.0 and float(diff[0]) == 0.0,
          f"max change per place = {[round(float(v), 8) for v in diff]}")


def test_temporal_has_no_spatial_path():
    torch.manual_seed(0)
    spec = rd.GridSpec(24, 6)
    m = rm.build("temporal", spec).eval()
    x = torch.randn(3, 30, 6, 4)
    yn = torch.randn(3, 4)
    x2 = x.clone()
    for row in (rd.R_DAM, rd.R_S1, rd.R_T1, rd.R_S2, rd.R_S3):
        x2[:, :, row] = torch.randn_like(x2[:, :, row])
    with torch.no_grad():
        p0 = m(x, yn)
        p1 = m(x2, yn)
    # only the S4 forecast (output index 3) must be unaffected: its own row
    # was not touched
    d = (p1[:, 3] - p0[:, 3]).abs().amax().item()
    check("temporal S4 forecast cannot see other places", d == 0.0, f"max change {d}")
    # the other stations must react (their own rows changed)
    d2 = (p1[:, 0] - p0[:, 0]).abs().amax().item()
    check("temporal S1 forecast does react to its own row", d2 > 0, f"max change {d2:.4f}")


def test_graph_model_can_see_upstream():
    torch.manual_seed(0)
    spec = rd.GridSpec(24, 6)
    for kind in ("graph", "graphlag", "graphattn"):
        m = rm.build(kind, spec).eval()
        x = torch.randn(3, 30, 6, 4)
        yn = torch.randn(3, 4)
        x2 = x.clone()
        x2[:, :, rd.R_S3] = torch.randn_like(x2[:, :, rd.R_S3])
        with torch.no_grad():
            p0 = m(x, yn)
            p1 = m(x2, yn)
        d = (p1[:, 3] - p0[:, 3]).abs().amax().item()
        check(f"{kind} can see S3 when forecasting S4", d > 0, f"max change {d:.4f}")


def test_no_upstream_leakage():
    """Water does not flow upstream: the forecast at S1 must not depend on
    S2, S3 or S4.  A lattice convolution or a global attention block breaks
    this; a directed graph does not."""
    torch.manual_seed(0)
    spec = rd.GridSpec(24, 6)
    x = torch.randn(3, 30, 6, 4)
    yn = torch.randn(3, 4)
    perm = torch.roll(torch.arange(3), 1)
    for kind, must_be_zero in (("temporal", True), ("graph", True),
                                ("graphlag", True), ("graphattn", True),
                                ("jointST", False), ("globalattn", False)):
        m = rm.build(kind, spec).eval()
        xd = x.clone()
        for row in (rd.R_S2, rd.R_S3, rd.R_S4):
            xd[:, :, row] = x[perm][:, :, row]
        with torch.no_grad():
            p0 = m(x, yn)
            p1 = m(xd, yn)
        d = float((p1[:, 0] - p0[:, 0]).abs().amax())
        if must_be_zero:
            check(f"{kind}: S1 forecast ignores the downstream stations",
                  d == 0.0, f"max change {d}")
        else:
            check(f"{kind}: S1 forecast DOES depend on downstream stations "
                  f"(physically wrong)", d > 0, f"max change {d:.4f}")


def test_lag_stack():
    from river_models import RiverModel
    v = torch.arange(1., 6.).view(1, 5, 1)
    st = RiverModel._lag_stack(v, 3)[0, :, 0, :]
    exp = torch.tensor([[1., 0., 0.], [2., 1., 0.], [3., 2., 1.],
                        [4., 3., 2.], [5., 4., 3.]])
    check("lag stack is the right shift", torch.equal(st, exp))


def test_shapes():
    spec = rd.GridSpec(24, 6)
    x = torch.randn(5, 30, 6, 4)
    yn = torch.randn(5, 4)
    for kind in ("persistence", "linear", "temporal", "graph", "graphlag",
                 "graphattn", "jointST", "globalattn"):
        m = rm.build(kind, spec).eval()
        with torch.no_grad():
            o = m(x, yn)
        check(f"{kind} output shape", tuple(o.shape) == (5, 4, 6), str(tuple(o.shape)))


if __name__ == "__main__":
    test_encoder_locality()
    test_temporal_has_no_spatial_path()
    test_graph_model_can_see_upstream()
    test_no_upstream_leakage()
    test_lag_stack()
    test_shapes()
    print()
    print("FAILED:", FAILED if FAILED else "none")
    sys.exit(1 if FAILED else 0)
