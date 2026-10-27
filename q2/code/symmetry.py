"""The 8 symmetries of the square, G = {R^a, M R^a : a = 0..3}, and the symmetrised forecaster.

Element g = 4*m + a means M^m(R^a(x)): rotate first, then mirror (as in the handout).
All maps act on the last two (spatial) dimensions and only rearrange pixels.
"""
import torch
import torch.nn as nn


def apply_g(x, g):
    out = torch.rot90(x, k=g % 4, dims=(-2, -1))
    return out.flip(-1) if g >= 4 else out


def apply_g_inv(x, g):
    # (M R^a)^-1 = R^-a M, and (R^a)^-1 = R^-a
    out = x.flip(-1) if g >= 4 else x
    return torch.rot90(out, k=-(g % 4), dims=(-2, -1))


class Symmetrised(nn.Module):
    """f_bar(X) = 1/8 sum_g g^-1(f(g(X))); X is (B, 5, 1, 64, 64)."""

    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, x):
        b = x.size(0)
        xs = torch.cat([apply_g(x, g) for g in range(8)], dim=0)  # (8B, ...)
        ps = self.f(xs).view(8, b, *x.shape[1:2], *x.shape[2:])    # (8, B, 5, 1, 64, 64)
        return sum(apply_g_inv(ps[g], g) for g in range(8)) / 8


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.rand(3, 5, 1, 64, 64)
    for g in range(8):
        assert torch.equal(apply_g_inv(apply_g(x, g), g), x), g
        assert torch.equal(apply_g(apply_g_inv(x, g), g), x), g
    assert torch.equal(apply_g(x, 0), x)
    assert torch.equal(apply_g(x, 4), x.flip(-1))                      # M
    assert torch.equal(apply_g(x, 1), torch.rot90(x, 1, (-2, -1)))     # R
    # the 8 images of a generic frame are distinct
    imgs = {apply_g(x[0, 0], g).flatten().numpy().tobytes() for g in range(8)}
    assert len(imgs) == 8
    print("symmetry checks pass")
