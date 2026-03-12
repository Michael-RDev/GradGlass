import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent))
from gradglass import gg

EPOCHS = 6
N_TRAIN = 1200
N_VAL = 400
N_FEATURES = 24
N_CLASSES = 6
LR = 1e-3
BATCH = 64
DEVICE = "cpu"


class InterpretNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(N_FEATURES, 64)
        self.h1 = nn.Linear(64, 128)
        self.h2 = nn.Linear(128, 64)
        self.h3 = nn.Linear(64, 32)
        self.out = nn.Linear(32, N_CLASSES)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = F.relu(self.embed(x))
        x = F.relu(self.h1(x))
        x = self.dropout(x)
        x = F.relu(self.h2(x))
        x = F.relu(self.h3(x))
        return self.out(x)


def make_ds(n, seed=0):
    torch.manual_seed(seed)
    X = torch.randn(n, N_FEATURES)
    y = X[:, :N_CLASSES].argmax(1)
    return TensorDataset(X, y)


def main():
    print("GradGlass Interpretability & Attribution Demo")
    train_ds = make_ds(N_TRAIN, seed=0)
    val_ds = make_ds(N_VAL, seed=99)
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128)

    model = InterpretNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    run = gg.run("interpretability-demo", lr=LR, epochs=EPOCHS, n_classes=N_CLASSES, note="Interpretability & attribution tests demo")
    run.watch(model, optimizer, gradients="summary", activations="auto", every=1, sample_batches=4)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            lr_now = optimizer.param_groups[0]["lr"]
            acc = (logits.argmax(1) == y).float().mean().item()
            conf = logits.softmax(-1).max(1).values.mean().item()
            run.log(loss=loss.item(), acc=acc, lr=lr_now, epoch=epoch)
            run.log_batch(x=x, y=y, y_pred=logits.argmax(1), loss=loss)

        scheduler.step()

        model.eval()
        val_loss, val_correct, val_n = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                out = model(x)
                val_loss += F.cross_entropy(out, y, reduction="sum").item()
                val_correct += (out.argmax(1) == y).sum().item()
                val_n += y.size(0)
        val_loss /= val_n
        run.log(val_loss=val_loss, val_acc=val_correct / val_n, epoch=epoch)
        print(f"Epoch {epoch:2d}  val_loss={val_loss:.4f}  val_acc={val_correct/val_n:.3f}")

        if epoch % 2 == 0:
            run.checkpoint(tag=f"epoch_{epoch}")

    run.finish()
    print("\nRunning analysis")
    run.analyze(print_summary=True)
    print("\nDone:) Open the dashboard → Analysis to see interpretability tests.")


if __name__ == "__main__":
    main()
