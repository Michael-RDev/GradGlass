import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from gradglass import gg

from _example_output import print_dashboard_next_steps, repo_workspace_root


def make_toy_loader(batch_size=32):
    torch.manual_seed(7)
    features = torch.randn(256, 4)
    decision = (
        1.8 * features[:, 0]
        - 1.2 * features[:, 1]
        + 0.7 * features[:, 2]
        + 0.5 * features[:, 0] * features[:, 3]
    )
    labels = (decision > 0).long()
    dataset = TensorDataset(features, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
        )

    def forward(self, x):
        return self.layers(x)


def main():
    gg.configure(root=str(repo_workspace_root()), auto_open=False)

    loader = make_toy_loader()
    model = TinyClassifier()
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.CrossEntropyLoss()

    run = gg.run(name="pytorch_core_tracking_minimal", task="classification")
    run.checkpoint_every(4)
    run.watch(
        model,
        optimizer=optimizer,
        activations="auto",
        gradients="summary",
        saliency="auto",
        every=4,
        probe_examples=16,
    )

    print(f"Started run: {run.run_id}")
    model.train()

    global_step = 0
    for epoch in range(3):
        for features, labels in loader:
            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            predictions = logits.argmax(dim=1)
            accuracy = (predictions == labels).float().mean().item()
            run.log(loss=loss.item(), acc=accuracy)
            global_step += 1

            if global_step % 4 == 0:
                # Pass raw class scores so GradGlass can derive confidences.
                run.log_batch(x=features, y=labels, y_pred=logits.detach(), loss=loss.item())

        print(f"Epoch {epoch + 1}: loss={loss.item():.4f} acc={accuracy:.3f}")

    print("Running GradGlass analysis...")
    run.analyze(print_summary=True)
    run.finish(open=False, analyze=False)
    print_dashboard_next_steps(gg.store.root)


if __name__ == "__main__":
    main()
