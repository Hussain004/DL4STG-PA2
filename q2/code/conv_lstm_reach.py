"""ConvLSTM with a configurable patch size and dilations (model D candidates).

With patch=4, kernels (3, 3) and dilations (1, 1) this is exactly model A
(same parameter names, shapes and initialisation order), so A's checkpoints load.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DilatedConvLSTMCell(nn.Module):
    def __init__(self, in_ch, hidden, kernel=3, dilation=1):
        super().__init__()
        self.hidden = hidden
        # one convolution produces the four gates i, f, o, g
        self.conv = nn.Conv2d(in_ch + hidden, 4 * hidden, kernel, stride=1,
                              padding=dilation * (kernel - 1) // 2,
                              dilation=dilation, bias=True)

    def forward(self, x, h, c):
        i, f, o, g = torch.split(self.conv(torch.cat([x, h], dim=1)), self.hidden, dim=1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, c


class ConvLSTMReach(nn.Module):
    """ConvLSTM stack with a configurable patch size, hidden size, kernels and dilations (one entry per layer)."""

    def __init__(self, patch=4, hidden=64, kernels=(3, 3), dilations=(1, 1)):
        super().__init__()
        self.patch, self.hidden = patch, hidden
        self.kernels, self.dilations = tuple(kernels), tuple(dilations)
        self.grid = 64 // patch
        self.n_layers = len(self.kernels)
        for i, (k, d) in enumerate(zip(self.kernels, self.dilations), start=1):
            setattr(self, f"convlstm{i}",
                    DilatedConvLSTMCell(patch * patch if i == 1 else hidden, hidden, k, d))
        self.conv_out = nn.Conv2d(hidden, patch * patch, kernel_size=1, bias=True)

    def reach(self):
        """One-step reach in original pixels, with the conventions of How far can information travel?

        A convolution moves information by dilation*(kernel-1)/2 grid cells; one cell is `patch` pixels.
        The longest path goes from the new input (or from h^(1) of the previous step) through every layer.
        """
        return self.patch * sum(d * (k - 1) // 2 for k, d in zip(self.kernels, self.dilations))

    def forward(self, frames):
        b, dev = frames.size(0), frames.device
        cells = [getattr(self, f"convlstm{i}") for i in range(1, self.n_layers + 1)]
        hs = [torch.zeros(b, self.hidden, self.grid, self.grid, device=dev) for _ in cells]
        cs = [torch.zeros_like(h) for h in hs]
        preds, x = [], frames[:, 0]
        for tau in range(1, 10):
            u = F.pixel_unshuffle(x, self.patch)
            for l, cell in enumerate(cells):
                hs[l], cs[l] = cell(u, hs[l], cs[l])
                u = hs[l]
            x_pred = torch.sigmoid(F.pixel_shuffle(self.conv_out(u), self.patch))
            if tau >= 5:
                preds.append(x_pred)
            x = frames[:, tau] if tau < 5 else x_pred
        return torch.stack(preds, dim=1)


def build_model(arch="baseline", patch=4, dilations=(1, 1), hidden=64, kernels=None):
    if arch == "baseline":
        from conv_lstm import ConvLSTMBaseline
        return ConvLSTMBaseline()
    return ConvLSTMReach(patch=patch, hidden=hidden, kernels=kernels or (3,) * len(dilations),
                         dilations=dilations)


# Model D: 5x5 kernels in both ConvLSTM layers, 40 hidden channels (544,976 parameters, reach 16 pixels).
MODEL_D = dict(patch=4, hidden=40, kernels=(5, 5), dilations=(1, 1))


def count_parameters(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


if __name__ == "__main__":
    from conv_lstm import ConvLSTMBaseline
    torch.manual_seed(0)
    a, r = ConvLSTMBaseline(), ConvLSTMReach()
    assert count_parameters(a) == count_parameters(r) == 480784
    r.load_state_dict(a.state_dict())
    x = torch.rand(2, 5, 1, 64, 64)
    assert torch.allclose(a(x), r(x), atol=1e-6)
    assert r.reach() == 8
    for kw in [dict(patch=4, dilations=(1, 2)), dict(patch=4, dilations=(1, 3)), dict(patch=8, dilations=(1, 1)),
               dict(patch=4, hidden=56, kernels=(5, 3), dilations=(1, 1)),
               dict(patch=4, hidden=40, kernels=(5, 5), dilations=(1, 1)),
               dict(patch=4, hidden=56, kernels=(3, 3, 3), dilations=(1, 1, 1))]:
        m = ConvLSTMReach(**kw)
        assert m(x).shape == (2, 5, 1, 64, 64)
        print(kw, "params", count_parameters(m), "reach", m.reach())
    print("reach model checks pass")
