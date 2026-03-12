import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent))
from gradglass import gg

EPOCHS = 15
N_TRAIN = 20
N_VAL = 500
N_FEATURES = 20
N_CLASSES = 4
LR = 5e-3
BATCH = 32
DEVICE = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")


class OverfitNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(N_FEATURES, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, N_CLASSES)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


def make_data(n):
    torch.manual_seed(0)
    X = torch.randn(n, N_FEATURES)
    y = (X[:, 0] + X[:, 1]).long() % N_CLASSES
    return TensorDataset(X, y)


def main():
    print("GradGlass Overfitting Demo")
    train_ds = make_data(N_TRAIN)
    val_ds = make_data(N_VAL)
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128)

    model = OverfitNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0)

    run = gg.run("overfit-demo", lr=LR, epochs=EPOCHS, note="Intentional overfitting demo")
    run.watch(model, optimizer, gradients="summary", activations="auto", every=1, sample_batches=3)

    print(f"Training with {N_TRAIN} examples — expect overfitting after epoch ~5")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = F.cross_entropy(out, y)
            loss.backward()
            optimizer.step()
            pred = out.argmax(1)
            run.log(loss=loss.item(), acc=(pred == y).float().mean().item(), epoch=epoch)
            run.log_batch(x=x, y=y, y_pred=pred, loss=loss)

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = model(x)
                val_loss += F.cross_entropy(out, y, reduction="sum").item()
                val_correct += (out.argmax(1) == y).sum().item()
                val_total += y.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total
        run.log(val_loss=val_loss, val_acc=val_acc, epoch=epoch)
        print(f"Epoch {epoch:2d}  val_loss={val_loss:.4f}  val_acc={val_acc:.3f}")

        if epoch in (5, 10, EPOCHS):
            run.checkpoint(tag=f"epoch_{epoch}")

    run.finish()
    run.analyze(print_summary=True)
    print("\nLaunching dashboard — press Ctrl+C to stop.")
    run.open()


if __name__ == "__main__":
    main()
