import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from gradglass import gg
from gradglass.analysis.registry import test, TestCategory, TestContext, TestResult, TestSeverity, TestStatus

from _example_output import print_dashboard_next_steps, repo_workspace_root


FEATURE_NAMES = [
    "dominant_signal",
    "support_signal",
    "counter_signal",
    "interaction_signal",
    "noise_left",
    "noise_right",
]


@test(
    id="custom-shap-dominance",
    title="No single feature dominates SHAP mass",
    category=TestCategory.MODEL,
    severity=TestSeverity.MEDIUM,
    description="Warn when one feature accounts for most of the global SHAP attribution mass.",
)
def validate_shap_dominance(ctx: TestContext) -> TestResult:
    shap_data = ctx.store.get_shap(ctx.run_id)
    if not shap_data or not shap_data.get("summary_plot"):
        return TestResult(
            id="custom-shap-dominance",
            title="No single feature dominates SHAP mass",
            status=TestStatus.SKIP,
            severity=TestSeverity.MEDIUM,
            category=TestCategory.MODEL,
            details={"reason": "No SHAP summary data found in artifacts."},
        )

    summary_plot = shap_data["summary_plot"]
    total_mass = sum(float(item.get("mean_shap", 0.0)) for item in summary_plot)
    top_feature = summary_plot[0]
    dominance_ratio = float(top_feature.get("mean_shap", 0.0)) / max(total_mass, 1e-12)

    if dominance_ratio > 0.55:
        return TestResult(
            id="custom-shap-dominance",
            title="No single feature dominates SHAP mass",
            status=TestStatus.WARN,
            severity=TestSeverity.MEDIUM,
            category=TestCategory.MODEL,
            details={
                "top_feature": top_feature.get("feature"),
                "mean_shap": top_feature.get("mean_shap"),
                "dominance_ratio": round(dominance_ratio, 4),
            },
            recommendation="One feature dominates the global attribution map. Check for shortcuts, leakage, or spurious correlations.",
        )

    return TestResult(
        id="custom-shap-dominance",
        title="No single feature dominates SHAP mass",
        status=TestStatus.PASS,
        severity=TestSeverity.MEDIUM,
        category=TestCategory.MODEL,
        details={
            "top_feature": top_feature.get("feature"),
            "mean_shap": top_feature.get("mean_shap"),
            "dominance_ratio": round(dominance_ratio, 4),
        },
    )


class TabularMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(len(FEATURE_NAMES), 24),
            nn.ReLU(),
            nn.Linear(24, 12),
            nn.ReLU(),
            nn.Linear(12, 2),
        )

    def forward(self, x):
        return self.network(x)


def make_dataset():
    torch.manual_seed(21)
    features = torch.randn(320, len(FEATURE_NAMES))
    signal = (
        4.5 * features[:, 0]
        + 1.2 * features[:, 1]
        - 0.8 * features[:, 2]
        + 0.7 * features[:, 0] * features[:, 3]
        + 0.1 * torch.randn(features.size(0))
    )
    labels = (signal > 0).long()
    train_x, val_x = features[:256], features[256:]
    train_y, val_y = labels[:256], labels[256:]
    return train_x, train_y, val_x, val_y


def evaluate(model, x, y, criterion):
    model.eval()
    with torch.no_grad():
        logits = model(x)
        loss = criterion(logits, y).item()
        predictions = logits.argmax(dim=1)
        accuracy = (predictions == y).float().mean().item()
    return logits, loss, accuracy


def main():
    try:
        import shap
    except ImportError as exc:
        raise RuntimeError("Install explainability dependencies with `pip install -e .[torch,explainability]`.") from exc

    gg.configure(root=str(repo_workspace_root()), auto_open=False)

    train_x, train_y, val_x, val_y = make_dataset()
    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=32, shuffle=True)

    model = TabularMLP()
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.CrossEntropyLoss()

    run = gg.run(name="tabular_shap_explainability", task="classification")
    run.checkpoint_every(1)
    run.watch(
        model,
        optimizer=optimizer,
        activations="auto",
        gradients="summary",
        saliency="auto",
        every=1,
        probe_examples=16,
    )

    print(f"Started SHAP example run: {run.run_id}")

    for epoch in range(6):
        model.train()
        running_loss = 0.0
        running_correct = 0
        seen = 0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_x.size(0)
            running_correct += int((logits.argmax(dim=1) == batch_y).sum().item())
            seen += batch_x.size(0)

        val_logits, val_loss, val_acc = evaluate(model, val_x, val_y, criterion)
        train_loss = running_loss / max(seen, 1)
        train_acc = running_correct / max(seen, 1)

        run.log(loss=train_loss, acc=train_acc, val_loss=val_loss, val_acc=val_acc)
        run.log_batch(x=val_x[:16], y=val_y[:16], y_pred=val_logits[:16], loss=val_loss)

        print(
            f"Epoch {epoch + 1}: train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
        )

    def predict_positive_class(batch):
        model.eval()
        with torch.no_grad():
            tensor = torch.tensor(batch, dtype=torch.float32)
            probs = torch.softmax(model(tensor), dim=1)
        return probs[:, 1].cpu().numpy()

    background = train_x[:24].numpy()
    eval_slice = val_x[:16].numpy()
    explainer = shap.KernelExplainer(predict_positive_class, background)
    shap_values = explainer.shap_values(eval_slice, nsamples=80)
    shap_payload = run.log_shap(
        FEATURE_NAMES,
        shap_values,
        message="Kernel SHAP values on a held-out validation slice.",
        top_k=len(FEATURE_NAMES),
    )

    print("Top SHAP features:")
    for item in shap_payload["summary_plot"][:3]:
        print(f"  {item['feature']}: {item['mean_shap']:.4f}")

    print("Running GradGlass analysis with built-ins + custom SHAP dominance test...")
    run.analyze(print_summary=True)
    run.finish(open=False, analyze=False)
    print_dashboard_next_steps(gg.store.root)


if __name__ == "__main__":
    main()
