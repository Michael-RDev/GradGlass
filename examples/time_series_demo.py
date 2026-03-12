import sys
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent))
from gradglass import gg

SEED = 0
N_SERIES = 5  # number of input features
LOOKBACK = 48  # input window (time-steps)
HORIZON = 12  # steps to forecast
N_TRAIN = 8_000
N_VAL = 2_000
HIDDEN = 128
NUM_LAYERS = 3
EPOCHS = 10
BATCH_SIZE = 256
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(SEED)
np.random.seed(SEED)


class LSTMForecaster(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(N_SERIES, HIDDEN, num_layers=NUM_LAYERS, batch_first=True, dropout=0.2)
        self.head = nn.Sequential(nn.Linear(HIDDEN, 64), nn.ReLU(), nn.Linear(64, HORIZON))

    def forward(self, x):  # x: (B, T, F)
        out, _ = self.lstm(x)  # (B, T, H)
        return self.head(out[:, -1, :])  # (B, horizon)


def make_series(n: int, rng: np.random.Generator):
    t = np.linspace(0, 4 * np.pi, n)
    freqs = [1.0, 1.7, 2.3, 3.1, 0.5]
    phases = rng.uniform(0, np.pi, N_SERIES)
    noise = rng.normal(0, 0.05, (n, N_SERIES))
    data = np.stack([np.sin(f * t + phases[i]) + noise[:, i] for i, f in enumerate(freqs)], axis=1).astype(np.float32)
    return data


def sliding_windows(data: np.ndarray):
    X, y = [], []
    target_col = 0  # forecast the first channel
    for i in range(len(data) - LOOKBACK - HORIZON + 1):
        X.append(data[i : i + LOOKBACK])
        y.append(data[i + LOOKBACK : i + LOOKBACK + HORIZON, target_col])
    return np.array(X), np.array(y)


def make_dataset():
    rng = np.random.default_rng(SEED)
    total = N_TRAIN + N_VAL + LOOKBACK + HORIZON + 200
    series = make_series(total, rng)

    mu, sigma = series.mean(0), series.std(0) + 1e-8
    series = (series - mu) / sigma

    X, y = sliding_windows(series)
    return (
        torch.from_numpy(X[:N_TRAIN]),
        torch.from_numpy(y[:N_TRAIN]),
        torch.from_numpy(X[N_TRAIN : N_TRAIN + N_VAL]),
        torch.from_numpy(y[N_TRAIN : N_TRAIN + N_VAL]),
    )


def mape(pred: torch.Tensor, true: torch.Tensor, eps: float = 1e-8):
    return float(((pred - true).abs() / (true.abs() + eps)).mean().item()) * 100


def train_epoch(model, loader, optimizer, run, epoch):
    model.train()
    total_mse, total_mae, total_count = 0.0, 0.0, 0

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        pred = model(x)
        mse = nn.functional.mse_loss(pred, y)
        mae = nn.functional.l1_loss(pred, y)
        mse.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        bs = y.size(0)
        total_mse += mse.item() * bs
        total_mae += mae.item() * bs
        total_count += bs

        run.log(loss=mse.item(), mae=mae.item(), rmse=math.sqrt(mse.item()), epoch=epoch)

    return total_mse / total_count, total_mae / total_count


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total_mse, total_mae, total_mape, total_count = 0.0, 0.0, 0.0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        pred = model(x)
        bs = y.size(0)
        total_mse += nn.functional.mse_loss(pred, y).item() * bs
        total_mae += nn.functional.l1_loss(pred, y).item() * bs
        total_mape += mape(pred, y) * bs
        total_count += bs
    n = total_count
    return total_mse / n, total_mae / n, total_mape / n


def main():
    print(f"GradGlass Time-Series Demo  |  device={DEVICE}  epochs={EPOCHS}")
    print(f"  Input:  {N_SERIES} features, lookback={LOOKBACK}")
    print(f"  Target: horizon={HORIZON} steps ahead\n")

    X_tr, y_tr, X_val, y_val = make_dataset()
    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE)

    model = LSTMForecaster().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR, steps_per_epoch=len(train_loader), epochs=EPOCHS
    )

    params = sum(p.numel() for p in model.parameters())
    print(f"Model: {params:,} parameters")

    run = gg.run(
        "lstm-forecaster",
        task="time-series/forecasting",
        n_series=N_SERIES,
        lookback=LOOKBACK,
        horizon=HORIZON,
        hidden=HIDDEN,
        num_layers=NUM_LAYERS,
        lr=LR,
        epochs=EPOCHS,
    )
    run.watch(model, optimizer, activations="auto", gradients="summary", every=40, sample_batches=2)
    run.checkpoint_every(len(train_loader))

    for epoch in range(1, EPOCHS + 1):
        tr_mse, tr_mae = train_epoch(model, train_loader, optimizer, run, epoch)
        val_mse, val_mae, val_mape = evaluate(model, val_loader)
        scheduler.step()

        run.log(val_mse=val_mse, val_rmse=math.sqrt(val_mse), val_mae=val_mae, val_mape=val_mape, epoch_end=epoch)
        run.checkpoint(tag=f"epoch_{epoch}")

        print(
            f"  Epoch {epoch:2d}/{EPOCHS}  "
            f"tr_rmse={math.sqrt(tr_mse):.4f}  "
            f"val_rmse={math.sqrt(val_mse):.4f}  "
            f"val_mae={val_mae:.4f}  "
            f"val_mape={val_mape:.2f}%"
        )

    report = run.finish(open=False)
    print(f"\n✅  Time-series demo complete — run id: {run.run_id}")


if __name__ == "__main__":
    main()
