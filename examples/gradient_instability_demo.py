import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent))
from gradglass import gg

EPOCHS = 5
N_TRAIN = 800
N_FEATURES = 16
N_CLASSES = 3
LR = 1.0  # aggressively high — causes instability
BATCH = 32
DEPTH = 8  # very deep — prone to vanishing gradients
DEVICE = "cpu"


class DeepNet(nn.Module):
    def __init__(self, depth=8):
        super().__init__()
        layers = [nn.Linear(N_FEATURES, 64), nn.ReLU()]
        for _ in range(depth - 2):
            layers += [nn.Linear(64, 64), nn.ReLU()]
        layers.append(nn.Linear(64, N_CLASSES))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def make_loader(n, batch):
    torch.manual_seed(0)
    X = torch.randn(n, N_FEATURES)
    y = (X[:, 0] > 0).long() + (X[:, 1] > 0).long()
    return DataLoader(TensorDataset(X, y), batch_size=batch, shuffle=True)


def main():
    print("GradGlass Gradient Instability Demo")
    print(f"  LR={LR}  depth={DEPTH}  (expect exploding/vanishing gradients)")
    loader = make_loader(N_TRAIN, BATCH)

    model = DeepNet(depth=DEPTH).to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.0)

    run = gg.run(
        "gradient-instability", lr=LR, depth=DEPTH, epochs=EPOCHS, note="High LR + deep net — instability demo"
    )
    run.watch(model, optimizer, gradients="summary", activations="auto", every=1, sample_batches=5)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for x, y in loader:
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            acc = (logits.argmax(1) == y).float().mean().item()
            run.log(loss=loss.item(), acc=acc, epoch=epoch)
            run.log_batch(x=x, y=y, y_pred=logits.argmax(1), loss=loss)

        if epoch in (2, EPOCHS):
            run.checkpoint(tag=f"epoch_{epoch}")

    run.finish(open=True)
    print("\nDone! Check GRAD_EXPLODING, GRAD_VANISHING, and LOSS_SPIKE_DETECTION.")


if __name__ == "__main__":
    main()
