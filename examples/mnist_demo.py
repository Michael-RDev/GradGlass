import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from torchvision.datasets import MNIST
import warnings


warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))
from gradglass import gg

EPOCHS = 3

BATCH_SIZE = 256
LR = 0.001
CHECKPOINT_EVERY_N_EPOCHS = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = Path.home() / ".cache" / "gradglass_data"


class MnistCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout1(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        return self.fc2(x)


def get_loaders():
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_ds = MNIST(DATA_DIR, train=True, download=True, transform=tf)  # type: ignore
    test_ds = MNIST(DATA_DIR, train=False, download=True, transform=tf)  # type: ignore
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return (train_loader, test_loader)


def train_epoch(model, loader, optimizer, run, epoch):
    model.train()
    (total_loss, correct, total) = (0.0, 0, 0)
    for x, y in loader:
        (x, y) = (x.to(DEVICE), y.to(DEVICE))
        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        batch_loss = loss.item()
        preds = logits.argmax(dim=1)
        batch_correct = (preds == y).sum().item()
        batch_total = y.size(0)
        total_loss += batch_loss * batch_total
        correct += batch_correct
        total += batch_total
        run.log(loss=batch_loss, acc=batch_correct / batch_total, epoch=epoch)
        run.log_batch(x=x, y=y, y_pred=logits, loss=loss)
    return (total_loss / total, correct / total)


def evaluate(model, loader):
    model.eval()
    (correct, total) = (0, 0)
    with torch.no_grad():
        for x, y in loader:
            (x, y) = (x.to(DEVICE), y.to(DEVICE))
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total


def main():
    print(f"GradGlass MNIST Demo  |  device={DEVICE}  epochs={EPOCHS}")
    print(f"Downloading / loading MNIST …")
    (train_loader, test_loader) = get_loaders()
    model = MnistCNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    run = gg.run("mnist-cnn", lr=LR, epochs=EPOCHS, batch_size=BATCH_SIZE)
    run.check_leakage_from_loaders(train_loader=train_loader, test_loader=test_loader)
    run.watch(model, optimizer, activations="auto", gradients="summary", every=1, sample_batches=3, monitor=True)
    print(f"\nModel: {sum((p.numel() for p in model.parameters())):,} parameters\n")
    for epoch in range(EPOCHS):
        (train_loss, train_acc) = train_epoch(model, train_loader, optimizer, run, epoch)
        test_acc = evaluate(model, test_loader)
        print(
            f"Epoch {epoch + 1}/{EPOCHS}  train_loss={train_loss:.4f}  train_acc={train_acc * 100:.1f}%  test_acc={test_acc * 100:.1f}%"
        )
        if (epoch + 1) % CHECKPOINT_EVERY_N_EPOCHS == 0:
            run.checkpoint(tag=f"epoch_{epoch}")
    run.finish(open=True, analyze=True, print_summary=True)
    print("\nDone.")


if __name__ == "__main__":
    main()
