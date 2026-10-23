"""Candidate architectures for the river forecaster.

Every model consumes the same grid

        X  [B, T, N, 4]      T = L + H columns, N = 6 rows, 4 channels

and produces the same output

        y_hat  [B, 4, H]     water levels at S1..S4 for the next H steps

through the same head (a residual on the last observed level), so that the
comparison isolates the SPATIO-TEMPORAL mechanism.

The families
------------
temporal      : per-station temporal encoder, no spatial path at all
               (reproduces the companion notebook's information boundary)
graph         : temporal encoder + R rounds of message passing on the river
               graph, same-time mixing
graphlag      : graph message passing where every edge carries a LEARNED
               delay kernel (the transport time is learned, not fixed)
graphattn     : graph message passing where every edge pools the upstream
               station's recent history with content-dependent weights
               (the delay is inferred from the current state)
jointST       : joint space-time convolution over the ROW INDEX (the notebook's
               "ST" block applied to the grid, treating neighbours as indices)
globalattn    : joint space-time attention over every (time, row) token
linear        : per-station linear map from its OWN history (ridge baseline)
"""
from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

import river_data as rd

ROW_OF = {r: i for i, r in enumerate(rd.ROWS)}
EDGES = [("Dam", "S1"), ("S1", "S2"), ("T1", "S2"), ("S2", "S3"), ("S3", "S4")]


class TemporalEncoder(nn.Module):
    """Per-place temporal encoder: a stack of dilated 1-D convolutions.

    Weights are shared across places; a learned place embedding lets the same
    kernel behave differently at the dam and at a gauged station.  The encoder
    is *not* causal: the columns that cover the forecast horizon contain only
    information known at issuance (the announced release schedule), so looking
    forward into them leaks nothing.
    """

    def __init__(self, n_channels, width, n_blocks=4, dilations=(1, 2, 4, 8)):
        super().__init__()
        self.stem = nn.Linear(n_channels, width)
        self.dilations = list(dilations[:n_blocks])
        self.blocks = nn.ModuleList()
        for dil in self.dilations[:n_blocks]:
            self.blocks.append(nn.ModuleDict({
                "conv": nn.Conv1d(width, width, 3, dilation=dil),
                "norm": nn.LayerNorm(width),
            }))
        self.out = nn.LayerNorm(width)

    def forward(self, x, place_emb):
        # x [B, T, N, C] -> [B, T, N, d]
        #
        # NOTE ON RESHAPES: reshaping a *permuted* (non-contiguous) tensor can
        # silently produce a view with the wrong element mapping.  Every
        # reshape below is therefore either a split of the leading dimension of
        # a contiguous tensor (always a valid view) or is preceded by an
        # explicit .contiguous() copy.
        B, T, N, C = x.shape
        d = place_emb.shape[-1]
        z = self.stem(x) + place_emb                      # [B, T, N, d]
        z = z.permute(0, 2, 3, 1).contiguous().view(B * N, d, T)
        for blk, dil in zip(self.blocks, self.dilations):
            h = blk["conv"](F.pad(z, (dil, dil)))         # [B*N, d, T]
            h = h.view(B, N, d, T).permute(0, 3, 1, 2)    # [B, T, N, d]
            h = blk["norm"](h)                            # LayerNorm over d
            h = h.permute(0, 2, 3, 1).contiguous().view(B * N, d, T)
            z = z + F.gelu(h)
        z = self.out(z.view(B, N, d, T).permute(0, 3, 1, 2))   # [B, T, N, d]
        return z.contiguous()


class Head(nn.Module):
    """Residual head: y_hat = y_now + g(h)."""

    def __init__(self, width, horizon):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(width, width), nn.GELU(), nn.Linear(width, horizon))
        self.scale = nn.Parameter(torch.ones(4))

    def forward(self, h, y_now):
        r = self.net(h)                                   # [B, 4, H]
        return y_now.unsqueeze(-1) + r * self.scale.view(1, 4, 1)


class RiverModel(nn.Module):
    """Common scaffolding: encoder, spatial stage, head."""

    def __init__(self, spec, width=64, depth=4, n_channels=4, n_places=6,
                 spatial="graphlag", rounds=4, lag_window=12, heads=4):
        super().__init__()
        self.spec = spec
        self.width = width
        self.spatial = spatial
        self.rounds = rounds
        self.lag_window = lag_window
        self.encoder = TemporalEncoder(n_channels, width, n_blocks=depth)
        self.place_emb = nn.Parameter(torch.randn(n_places, width) * 0.02)

        self.edges = EDGES
        self.edge_emb = nn.Parameter(torch.randn(len(EDGES), width) * 0.02)

        if spatial in ("graphlag", "graphattn", "graph"):
            self.msg = nn.ModuleList([nn.Linear(width, width, bias=False)
                                      for _ in range(rounds)])
        if spatial == "graphlag":
            self.lag_logits = nn.Parameter(torch.randn(len(EDGES), lag_window) * 0.5)
        if spatial == "graphattn":
            self.q = nn.ModuleList([nn.Linear(width, width) for _ in range(rounds)])
            self.k = nn.ModuleList([nn.Linear(width, width) for _ in range(rounds)])
            self.v = nn.ModuleList([nn.Linear(width, width) for _ in range(rounds)])
        if spatial == "jointST":
            self.st_blocks = nn.ModuleList()
            for _ in range(rounds):
                self.st_blocks.append(nn.ModuleDict({
                    "conv": nn.Conv2d(width, width, (3, 3)),
                    "norm": nn.LayerNorm(width)}))
        if spatial == "globalattn":
            self.attn_blocks = nn.ModuleList()
            for _ in range(rounds):
                self.attn_blocks.append(nn.ModuleDict({
                    "n1": nn.LayerNorm(width),
                    "attn": nn.MultiheadAttention(width, heads, batch_first=True),
                    "n2": nn.LayerNorm(width),
                    "ff": nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(),
                                        nn.Linear(2 * width, width))}))

        self.head = Head(width, spec.H)

    @staticmethod
    def _lag_stack(v, W):
        """v [B, T, d] -> [B, T, d, W] with [..., tau] = v[t - tau] (0 if t < tau)."""
        pad = F.pad(v, (0, 0, W - 1, 0))          # pad the time axis on the left
        st = pad.unfold(1, W, 1)                  # [B, T, d, W]
        return st.flip(-1)                        # [..., tau] = v[t - tau]

    def forward(self, x, y_now):
        B, T, N, C = x.shape
        z = self.encoder(x, self.place_emb)               # [B, T, N, d]

        if self.spatial == "temporal":
            pass
        elif self.spatial == "graph":
            for r in range(self.rounds):
                upd = torch.zeros_like(z)
                for e, (a, b) in enumerate(self.edges):
                    ja, jb = ROW_OF[a], ROW_OF[b]
                    upd[:, :, jb] = upd[:, :, jb] + \
                        self.msg[r](z[:, :, ja]) + self.edge_emb[e]
                z = z + upd
        elif self.spatial == "graphlag":
            for r in range(self.rounds):
                upd = torch.zeros_like(z)
                for e, (a, b) in enumerate(self.edges):
                    ja, jb = ROW_OF[a], ROW_OF[b]
                    alpha = torch.softmax(self.lag_logits[e], dim=-1)   # [W]
                    zz = self.msg[r](z[:, :, ja])                       # [B, T, d]
                    st = self._lag_stack(zz, self.lag_window)           # [B,T,d,W]
                    m = (st * alpha.view(1, 1, 1, -1)).sum(-1)
                    upd[:, :, jb] = upd[:, :, jb] + m + self.edge_emb[e]
                z = z + upd
        elif self.spatial == "graphattn":
            for r in range(self.rounds):
                upd = torch.zeros_like(z)
                for e, (a, b) in enumerate(self.edges):
                    ja, jb = ROW_OF[a], ROW_OF[b]
                    q = self.q[r](z[:, :, jb])                          # [B, T, d]
                    k = self.k[r](z[:, :, ja])
                    v = self.v[r](z[:, :, ja])
                    K = self._lag_stack(k, self.lag_window)              # [B,T,d,W]
                    V = self._lag_stack(v, self.lag_window)
                    score = torch.einsum("btd,btdw->btw", q, K) / math.sqrt(self.width)
                    alpha = torch.softmax(score, dim=-1)
                    m = torch.einsum("btw,btdw->btd", alpha, V)
                    upd[:, :, jb] = upd[:, :, jb] + m + self.edge_emb[e]
                z = z + upd
        elif self.spatial == "jointST":
            for blk in self.st_blocks:
                u = z.permute(0, 3, 1, 2)                    # [B, d, T, N]
                h = blk["conv"](F.pad(u, (1, 1, 2, 0)))
                h = h.permute(0, 2, 3, 1)
                h = blk["norm"](h)
                z = z + F.gelu(h)
        elif self.spatial == "globalattn":
            tok = z.reshape(B, T * N, self.width)
            for blk in self.attn_blocks:
                h = blk["n1"](tok)
                a, _ = blk["attn"](h, h, h, need_weights=False)
                tok = tok + a
                tok = tok + blk["ff"](blk["n2"](tok))
            z = tok.reshape(B, T, N, self.width)

        now = self.spec.L - 1
        h = z[:, now][:, rd.TARGET_ROWS]                     # [B, 4, d]
        return self.head(h, y_now)


class Persistence(nn.Module):
    """y_hat = last observed level for every horizon."""

    def __init__(self, spec):
        super().__init__()
        self.spec = spec

    def forward(self, x, y_now):
        return y_now.unsqueeze(-1).expand(-1, -1, self.spec.H)


class LinearUni(nn.Module):
    """Per-station linear map from its OWN history to its own future
    (a DLinear / ridge-style baseline: no spatial path at all)."""

    def __init__(self, spec):
        super().__init__()
        self.spec = spec
        self.lin = nn.Linear(spec.L, spec.H)

    def forward(self, x, y_now):
        hist = x[:, :self.spec.L, :, 0]
        outs = []
        for row in rd.TARGET_ROWS:
            outs.append(self.lin(hist[:, :, row]))      # [B, L] -> [B, H]
        pred = torch.stack(outs, dim=1)
        return y_now.unsqueeze(-1) + pred


def build(kind, spec, **kw):
    if kind == "persistence":
        return Persistence(spec)
    if kind == "linear":
        return LinearUni(spec)
    return RiverModel(spec, spatial=kind, **kw)
