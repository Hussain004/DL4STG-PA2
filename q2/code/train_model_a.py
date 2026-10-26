"""Train Model A (ConvLSTM baseline) on Moving MNIST.

Protocol (from PA2 handout, Training and evaluation):
- Adam, lr=1e-3, no schedule, batch size 32, 30 epochs
- One epoch = one pass over 8,000 training videos, one random window per video
- Loss: MSE averaged over pixels and available target frames, then over batch
- Seeds: 0 and 1
- Keep final-epoch weights
"""
import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conv_lstm import ConvLSTMBaseline, count_parameters


class MovingMNISTWindow:
    """Wrapper that samples one random window per video per epoch.

    For normal-speed training (models A and B): t in {1, ..., 11},
    so all five target frames exist.
    Window: X = (x_t, ..., x_{t+4}), Y = (x_{t+5}, ..., x_{t+9}).
    """

    def __init__(self, dataset, indices, seed, speed=1):
        self.dataset = dataset
        self.indices = indices
        self.speed = speed

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        idx = self.indices[position]
        video = self.dataset[idx]  # (20, 1, 64, 64) uint8 [0,255]
        video = video.float() / 255.0  # (20, 1, 64, 64) in [0,1]

        if self.speed == 1:
            # Normal speed: t in {1, ..., 11}
            # torch.randint, not a dataset-held RNG: DataLoader reseeds torch per worker
            # and per epoch, so t differs across workers and epochs.
            t = int(torch.randint(1, 12, (1,)))  # 1-based, frames 1..20
            x = video[t - 1:t + 4]  # frames t..t+4 (0-indexed: t-1..t+3)
            y = video[t + 4:t + 9]  # frames t+5..t+9 (0-indexed: t+4..t+8)
        else:
            raise ValueError(f"Speed {self.speed} not supported in this script")

        return x, y  # each (5, 1, 64, 64)


def collate_fn(batch):
    xs = torch.stack([b[0] for b in batch])  # (B, 5, 1, 64, 64)
    ys = torch.stack([b[1] for b in batch])  # (B, 5, 1, 64, 64)
    return xs, ys


def train_model(seed, data_root, out_dir, device, epochs=30, batch_size=32, lr=1e-3):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load full dataset
    ds = torchvision.datasets.MovingMNIST(
        root=data_root, split=None, download=False, transform=None
    )
    # Split: 0-7999 train, 8000-8999 val, 9000-9999 test
    train_indices = list(range(0, 8000))
    train_dataset = MovingMNISTWindow(ds, train_indices, seed, speed=1)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, collate_fn=collate_fn, pin_memory=True
    )

    model = ConvLSTMBaseline().to(device)
    print(f"Seed {seed}: trainable parameters = {count_parameters(model):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss(reduction='none')

    model.train()
    epoch_times = []
    for epoch in range(epochs):
        epoch_start = time.time()
        total_loss = 0.0
        total_frames = 0
        for xs, ys in train_loader:
            xs = xs.to(device)   # (B, 5, 1, 64, 64)
            ys = ys.to(device)   # (B, 5, 1, 64, 64)

            optimizer.zero_grad()
            preds = model(xs)    # (B, 5, 1, 64, 64)

            # MSE averaged over pixels and the available target frames of each
            # video, then over the videos in the batch. Slicing preds to the
            # number of available targets also makes this correct for the
            # speed-augmented models C and D.
            n_target = ys.size(1)
            loss_per_video = criterion(preds[:, :n_target], ys).mean(dim=(2, 3, 4))  # (B, n_target)
            loss = loss_per_video.mean()

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xs.size(0)
            total_frames += xs.size(0)

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        avg_loss = total_loss / total_frames
        print(f"  Seed {seed}, Epoch {epoch + 1}/{epochs}: "
              f"loss={avg_loss:.6f}, time={epoch_time:.1f}s")

    # Save final-epoch weights
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, f"model_a_seed{seed}.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Seed {seed}: saved model to {model_path}")

    # Training time per epoch (mean over all 30 epochs)
    mean_epoch_time = np.mean(epoch_times)
    print(f"Seed {seed}: mean epoch time = {mean_epoch_time:.2f}s")

    return model, mean_epoch_time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True, help='Random seed (0 or 1)')
    parser.add_argument('--data-root', type=str, default='~/data/moving_mnist')
    parser.add_argument('--out-dir', type=str, default='./runs/model_a')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()

    args.data_root = os.path.expanduser(args.data_root)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model, epoch_time = train_model(
        seed=args.seed,
        data_root=args.data_root,
        out_dir=args.out_dir,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )

    print(f"\nDone. Seed {args.seed}, params={count_parameters(model):,}, "
          f"epoch_time={epoch_time:.2f}s")


if __name__ == '__main__':
    main()