import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from gradglass import gg

from _example_output import print_dashboard_next_steps, repo_workspace_root


def make_pattern(label, size=16):
    image = torch.zeros(size, size)
    if label == 0:
        image[:, 4:6] = 1.0
    elif label == 1:
        image[8:10, :] = 1.0
    else:
        diagonal = torch.arange(size)
        image[diagonal, diagonal] = 1.0
        image[diagonal[:-1], diagonal[1:]] = 0.8
    image += 0.08 * torch.randn_like(image)
    return image.clamp(0.0, 1.0)


def build_datasets(train_per_class=48, val_per_class=18):
    torch.manual_seed(13)
    train_images = []
    train_labels = []
    val_images = []
    val_labels = []

    for label in range(3):
        for _ in range(train_per_class):
            train_images.append(make_pattern(label))
            train_labels.append(label)
        for _ in range(val_per_class):
            val_images.append(make_pattern(label))
            val_labels.append(label)

    train_x = torch.stack(train_images).unsqueeze(1)
    train_y = torch.tensor(train_labels, dtype=torch.long)
    val_x = torch.stack(val_images).unsqueeze(1)
    val_y = torch.tensor(val_labels, dtype=torch.long)
    return train_x, train_y, val_x, val_y


class TinyVisionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def evaluate(model, x, y, criterion):
    model.eval()
    with torch.no_grad():
        logits = model(x)
        loss = criterion(logits, y).item()
        predictions = logits.argmax(dim=1)
        accuracy = (predictions == y).float().mean().item()
    return logits, loss, accuracy


def main():
    gg.configure(root=str(repo_workspace_root()), auto_open=False)

    train_x, train_y, val_x, val_y = build_datasets()
    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=24, shuffle=True)

    model = TinyVisionNet()
    optimizer = optim.Adam(model.parameters(), lr=5e-3)
    criterion = nn.CrossEntropyLoss()

    run = gg.run(name="pytorch_full_observability", task="vision classification", monitor=True)
    run.checkpoint_every(1)
    run.watch(
        model,
        optimizer=optimizer,
        activations="auto",
        gradients="summary",
        saliency="auto",
        every=1,
        probe_examples=8,
        monitor=True,
        monitor_open_browser=False,
    )

    print(f"Started full observability run: {run.run_id}")

    for epoch in range(6):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_examples = 0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)
            total_correct += int((logits.argmax(dim=1) == batch_y).sum().item())
            total_examples += batch_x.size(0)

        val_logits, val_loss, val_acc = evaluate(model, val_x, val_y, criterion)
        train_loss = total_loss / max(total_examples, 1)
        train_acc = total_correct / max(total_examples, 1)

        run.log(loss=train_loss, acc=train_acc, val_loss=val_loss, val_acc=val_acc)
        # Use raw logits so prediction confidence, evaluation, and hard-example views stay rich.
        run.log_batch(x=val_x[:8], y=val_y[:8], y_pred=val_logits[:8], loss=val_loss)

        print(
            f"Epoch {epoch + 1}: train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
        )

    print("Running the full GradGlass analysis suite...")
    run.analyze(print_summary=True)
    run.finish(open=False, analyze=False)
    print_dashboard_next_steps(gg.store.root, live_monitor=True)


if __name__ == "__main__":
    main()
