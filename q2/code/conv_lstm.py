"""Model A: ConvLSTM baseline for Moving MNIST (PA2 Question 2).

Architecture follows Shi et al. (2015) ConvLSTM without peephole terms:
1. pixel_unshuffle(4): 1x64x64 -> 16x16x16
2. Two ConvLSTM layers, 64 hidden channels, 3x3 kernels, stride 1, padding 1
3. 1x1 conv 64->16, pixel_shuffle(4), sigmoid -> 1x64x64

Rollout: 9 steps. Steps 1-5 use input frames; steps 6-9 use own predictions.
Outputs of steps 5-9 form Y_hat (5 predicted frames). No teacher forcing.

Expected trainable parameters: 480,784.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLSTMCell(nn.Module):
    """Single ConvLSTM layer cell without peephole connections."""

    def __init__(self, input_channels, hidden_channels, kernel_size=3, padding=1):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = padding
        # One convolution produces all four gates: i, f, o, g
        self.conv = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size,
            stride=1,
            padding=padding,
            bias=True,
        )

    def forward(self, x, h, c):
        """x: (B, C_in, H, W), h: (B, C_h, H, W), c: (B, C_h, H, W)"""
        combined = torch.cat([x, h], dim=1)  # (B, C_in + C_h, H, W)
        gates = self.conv(combined)  # (B, 4*C_h, H, W)
        i, f, o, g = torch.split(gates, self.hidden_channels, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, height, width, device):
        return (
            torch.zeros(batch_size, self.hidden_channels, height, width, device=device),
            torch.zeros(batch_size, self.hidden_channels, height, width, device=device),
        )


class ConvLSTMBaseline(nn.Module):
    """Model A: two-layer ConvLSTM forecaster for Moving MNIST."""

    def __init__(self):
        super().__init__()
        # After pixel_unshuffle(4): 1x64x64 -> 16x16x16
        # ConvLSTM layer 1: input 16 channels, hidden 64
        self.convlstm1 = ConvLSTMCell(16, 64, kernel_size=3, padding=1)
        # ConvLSTM layer 2: input 64 (from layer 1 hidden), hidden 64
        self.convlstm2 = ConvLSTMCell(64, 64, kernel_size=3, padding=1)
        # Output head: 1x1 conv 64->16, then pixel_shuffle(4) -> 1x64x64
        self.conv_out = nn.Conv2d(64, 16, kernel_size=1, stride=1, padding=0, bias=True)

    def fold(self, x):
        """pixel_unshuffle: (B, 1, 64, 64) -> (B, 16, 16, 16)"""
        return F.pixel_unshuffle(x, 4)

    def unfold(self, x):
        """pixel_shuffle: (B, 16, 16, 16) -> (B, 1, 64, 64)"""
        return F.pixel_shuffle(x, 4)

    def forward(self, frames):
        """frames: (B, T_in, 1, 64, 64). Returns predictions (B, 5, 1, 64, 64).

        9-step rollout:
        - Steps 1-5: input frames
        - Steps 6-9: own previous predictions
        - Outputs of steps 5-9 form Y_hat.
        """
        batch_size = frames.size(0)
        device = frames.device

        # Initialize hidden and cell states for both layers (zero states)
        h1, c1 = self.convlstm1.init_hidden(batch_size, 16, 16, device)
        h2, c2 = self.convlstm2.init_hidden(batch_size, 16, 16, device)

        predictions = []
        # Step input: frames[0..4] for steps 1-5, then own predictions
        x = frames[:, 0]  # (B, 1, 64, 64) - first input frame

        for tau in range(1, 10):  # tau = 1..9
            # Fold input frame
            u = self.fold(x)  # (B, 16, 16, 16)

            # ConvLSTM layer 1
            h1, c1 = self.convlstm1(u, h1, c1)

            # ConvLSTM layer 2 takes layer 1's new hidden state
            h2, c2 = self.convlstm2(h1, h2, c2)

            # Output head
            out = self.conv_out(h2)  # (B, 16, 16, 16)
            out = self.unfold(out)   # (B, 1, 64, 64)
            x_pred = torch.sigmoid(out)  # (B, 1, 64, 64)

            # Collect outputs from steps 5-9
            if tau >= 5:
                predictions.append(x_pred)

            # Next step input: input frames for tau=1..5, own predictions for tau=6..9
            if tau < 5:
                x = frames[:, tau]  # next input frame
            else:
                x = x_pred  # own prediction fed back

        # Stack: (5, B, 1, 64, 64) -> (B, 5, 1, 64, 64)
        predictions = torch.stack(predictions, dim=1)
        return predictions

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)