import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent))
from gradglass import gg

N_TRAIN = 1000
N_FEATURES = 32
N_CLASSES = 5
PHASE1_EPOCHS = 4  # head-only
PHASE2_EPOCHS = 4  # full fine-tune
BATCH = 64
LR_HEAD = 1e-3
LR_FINETUNE = 2e-4
DEVICE = "cpu"


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(N_FEATURES, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 32)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return F.relu(self.layer3(x))


class TaskHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32, 32)
        self.fc2 = nn.Linear(32, N_CLASSES)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class TransferModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Backbone()
        self.head = TaskHead()

    def forward(self, x):
        return self.head(self.backbone(x))


def make_loader(n, batch):
    torch.manual_seed(42)
    X = torch.randn(n, N_FEATURES)
    y = X[:, :N_CLASSES].argmax(1)
    return DataLoader(TensorDataset(X, y), batch_size=batch, shuffle=True)


def freeze_backbone(model):
    for p in model.backbone.parameters():
        p.requires_grad_(False)


def unfreeze_backbone(model):
    for p in model.backbone.parameters():
        p.requires_grad_(True)


def train_epoch(model, loader, optimizer, run, epoch):
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


def main():
    print("GradGlass Transfer Learning Demo")
    loader = make_loader(N_TRAIN, BATCH)
    model = TransferModel().to(DEVICE)

    print("  [init] Simulating pretrained backbone weights…")
    torch.manual_seed(7)
    with torch.no_grad():
        for p in model.backbone.parameters():
            nn.init.orthogonal_(p) if p.dim() >= 2 else nn.init.normal_(p)

    run = gg.run(
        "transfer-learning",
        lr_head=LR_HEAD,
        lr_finetune=LR_FINETUNE,
        phase1_epochs=PHASE1_EPOCHS,
        phase2_epochs=PHASE2_EPOCHS,
        note="Two-phase: frozen backbone → full fine-tune",
    )
    run.watch(model, gradients="summary", activations="auto", every=1, sample_batches=3)

    print(f"\nPhase 1: Training head only ({PHASE1_EPOCHS} epochs, backbone frozen)")
    freeze_backbone(model)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_HEAD)

    for epoch in range(1, PHASE1_EPOCHS + 1):
        train_epoch(model, loader, optimizer, run, epoch)
        print(f"  [Phase 1] epoch {epoch}/{PHASE1_EPOCHS}")

    run.checkpoint(tag="phase1_end")

    print(f"\nPhase 2: Fine-tuning all layers ({PHASE2_EPOCHS} epochs)")
    unfreeze_backbone(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_FINETUNE)

    for epoch in range(PHASE1_EPOCHS + 1, PHASE1_EPOCHS + PHASE2_EPOCHS + 1):
        train_epoch(model, loader, optimizer, run, epoch)
        print(f"  [Phase 2] epoch {epoch - PHASE1_EPOCHS}/{PHASE2_EPOCHS}")

    run.checkpoint(tag="phase2_end")
    run.finish()

    print("\nRunning analysis…")
    run.analyze(print_summary=True)
    print("\nDone! Check TRAINABLE_FROZEN_CONSISTENCY and FREEZE_RECOMMENDATION in the dashboard.")


if __name__ == "__main__":
    main()
