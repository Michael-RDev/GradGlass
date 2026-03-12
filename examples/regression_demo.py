import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent))
from gradglass import gg

EPOCHS = 12
BATCH_SIZE = 128
LR = 0.001
CHECKPOINT_EVERY_N_EPOCHS = 2
PROBE_EVERY_N_BATCHES = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Regressor(nn.Module):
    def __init__(self, in_features=12):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_features, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def make_synthetic_regression(seed=42, n_train=4096, n_val=1024, n_features=12, noise=0.20):
    g = torch.Generator().manual_seed(seed)
    x_train = torch.randn(n_train, n_features, generator=g)
    x_val = torch.randn(n_val, n_features, generator=g)

    weights = torch.linspace(1.25, -1.75, n_features)

    def build_targets(x):
        linear = x @ weights
        nonlinear = 0.8 * torch.sin(x[:, 0]) + 0.5 * (x[:, 1] ** 2) - 0.3 * x[:, 2] * x[:, 3]
        return linear + nonlinear

    y_train = build_targets(x_train) + noise * torch.randn(n_train, generator=g)
    y_val = build_targets(x_val) + noise * torch.randn(n_val, generator=g)

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader


def train_epoch(model, loader, optimizer, run, epoch):
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    total_count = 0

    for batch_idx, (x, y) in enumerate(loader):
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()
        y_pred = model(x)
        loss = F.mse_loss(y_pred, y)
        mae = F.l1_loss(y_pred, y)
        loss.backward()
        optimizer.step()

        bs = y.size(0)
        total_loss += loss.item() * bs
        total_mae += mae.item() * bs
        total_count += bs

        run.log(loss=loss.item(), mae=mae.item(), rmse=loss.sqrt().item(), epoch=epoch, split="train")
        if batch_idx % PROBE_EVERY_N_BATCHES == 0:
            run.log_batch(x=x, y=y, y_pred=y_pred, loss=loss)

    return total_loss / total_count, total_mae / total_count


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total_mse = 0.0
    total_mae = 0.0
    total_count = 0

    for x, y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        y_pred = model(x)
        mse = F.mse_loss(y_pred, y)
        mae = F.l1_loss(y_pred, y)

        bs = y.size(0)
        total_mse += mse.item() * bs
        total_mae += mae.item() * bs
        total_count += bs

    val_mse = total_mse / total_count
    val_mae = total_mae / total_count
    val_rmse = val_mse**0.5
    return val_mse, val_mae, val_rmse


def main():
    print(f"GradGlass Regression Demo  |  device={DEVICE}  epochs={EPOCHS}")
    train_loader, val_loader = make_synthetic_regression()

    model = Regressor(in_features=12).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    run = gg.run("synthetic-regression", lr=LR, epochs=EPOCHS, batch_size=BATCH_SIZE, task="regression")
    run.watch(model, optimizer, activations="auto", gradients="summary", every=5, sample_batches=2, monitor=False)

    print(f"\nModel: {sum(p.numel() for p in model.parameters()):,} parameters\n")

    for epoch in range(EPOCHS):
        train_mse, train_mae = train_epoch(model, train_loader, optimizer, run, epoch)
        val_mse, val_mae, val_rmse = evaluate(model, val_loader)

        run.log(loss=val_mse, val_loss=val_mse, val_mae=val_mae, val_rmse=val_rmse, epoch=epoch, split="val")

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS}  "
            f"train_mse={train_mse:.4f}  train_mae={train_mae:.4f}  "
            f"val_mse={val_mse:.4f}  val_rmse={val_rmse:.4f}  val_mae={val_mae:.4f}"
        )

        if (epoch + 1) % CHECKPOINT_EVERY_N_EPOCHS == 0:
            run.checkpoint(tag=f"epoch_{epoch + 1}")

    run.finish(open=True, analyze=True, print_summary=True)
    print("\nDone")


if __name__ == "__main__":
    main()
