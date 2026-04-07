from __future__ import annotations

import json
import socket
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from gradglass import gg
from gradglass.analysis.data_monitor import PipelineStage
from gradglass.artifacts import ArtifactStore
from gradglass.server import find_free_port


FEATURE_NAMES = ["signal_a", "signal_b", "signal_c", "interaction", "noise_a", "noise_b"]


class CoverageMLP(nn.Module):
    def __init__(self, width: int = 24):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(len(FEATURE_NAMES), width),
            nn.ReLU(),
            nn.Linear(width, width // 2),
            nn.ReLU(),
            nn.Linear(width // 2, 2),
        )

    def forward(self, x):
        return self.network(x)


def make_dataset(seed: int = 17) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    features = torch.randn(320, len(FEATURE_NAMES))
    decision = (
        3.8 * features[:, 0]
        - 1.4 * features[:, 1]
        + 0.8 * features[:, 2]
        + 1.1 * features[:, 0] * features[:, 3]
        + 0.1 * torch.randn(features.size(0))
    )
    labels = (decision > 0).long()
    train_x, val_x = features[:256], features[256:]
    train_y, val_y = labels[:256], labels[256:]
    return train_x, train_y, val_x, val_y


def build_lime_samples(feature_names: list[str], values: np.ndarray) -> list[dict[str, Any]]:
    scores = np.asarray(values, dtype=np.float64)
    samples = []
    for index, row in enumerate(scores[:4]):
        ranked = np.argsort(np.abs(row))[::-1][:3]
        samples.append(
            {
                "index": index,
                "prediction": "positive" if np.sum(row) >= 0 else "negative",
                "probability": float(min(0.99, max(0.51, 0.65 + 0.05 * index))),
                "explanation": [
                    {"feature": feature_names[int(idx)], "weight": float(round(row[int(idx)], 6))} for idx in ranked
                ],
            }
        )
    return samples


def evaluate_model(
    model: nn.Module, x: torch.Tensor, y: torch.Tensor, criterion: nn.Module
) -> tuple[torch.Tensor, float, float]:
    model.eval()
    with torch.no_grad():
        logits = model(x)
        loss = float(criterion(logits, y).item())
        accuracy = float((logits.argmax(dim=1) == y).float().mean().item())
    return logits, loss, accuracy


def _predict_positive_probabilities(model: nn.Module, batch: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        tensor = torch.tensor(batch, dtype=torch.float32)
        probs = torch.softmax(model(tensor), dim=1)
    return probs[:, 1].cpu().numpy()


def build_tabular_run(
    store: ArtifactStore,
    *,
    name: str,
    width: int,
    seed: int,
    include_extra_artifacts: bool = True,
    finish_run: bool = True,
    epochs: int = 5,
) -> dict[str, Any]:
    train_x, train_y, val_x, val_y = make_dataset(seed=seed)
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=32, shuffle=True)

    model = CoverageMLP(width=width)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.CrossEntropyLoss()

    run = gg.run(name=name, task="classification", monitor=False)
    run.checkpoint_every(1)
    run.watch(
        model, optimizer=optimizer, activations="auto", gradients="summary", saliency="auto", every=1, probe_examples=16
    )

    for _epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_seen = 0

        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * batch_x.size(0)
            total_correct += int((logits.argmax(dim=1) == batch_y).sum().item())
            total_seen += batch_x.size(0)

        val_logits, val_loss, val_acc = evaluate_model(model, val_x, val_y, criterion)
        train_loss = total_loss / max(total_seen, 1)
        train_acc = total_correct / max(total_seen, 1)

        run.log(loss=train_loss, acc=train_acc, val_loss=val_loss, val_acc=val_acc)
        run.log_batch(x=val_x[:16], y=val_y[:16], y_pred=val_logits[:16], loss=val_loss)

    if include_extra_artifacts:
        background = train_x[:24].numpy()
        eval_slice = val_x[:16].numpy()
        try:
            import shap

            explainer = shap.KernelExplainer(lambda batch: _predict_positive_probabilities(model, batch), background)
            shap_values = explainer.shap_values(eval_slice, nsamples=64)
        except Exception:
            probs = _predict_positive_probabilities(model, eval_slice)
            centered_inputs = eval_slice - np.mean(background, axis=0, keepdims=True)
            shap_values = centered_inputs * probs.reshape(-1, 1)

        run.log_shap(FEATURE_NAMES, shap_values, message=f"{name} SHAP summary.", top_k=len(FEATURE_NAMES))

        shap_scores = run.store.get_shap(run.run_id)["summary_plot"]
        lime_weights = np.array([item["mean_shap"] for item in shap_scores], dtype=np.float64)
        lime_rows = np.tile(lime_weights, (4, 1))
        run.log_lime(
            build_lime_samples([item["feature"] for item in shap_scores], lime_rows),
            message=f"{name} LIME-style local explanations.",
        )

        monitor = run.monitor_dataset("classification", dataset_name=f"{name}_dataset", task_hint="tabular")
        monitor.record_stage(
            PipelineStage.SPLITTING,
            split="train",
            data=train_x.numpy(),
            labels=train_y.numpy(),
            metadata={"feature_names": FEATURE_NAMES},
        )
        monitor.record_stage(
            PipelineStage.SPLITTING,
            split="validation",
            data=val_x.numpy(),
            labels=val_y.numpy(),
            metadata={"feature_names": FEATURE_NAMES},
        )
        monitor.finalize(save=True)

        leak_test_x = val_x.clone()
        leak_test_y = val_y.clone()
        leak_test_x[:12] = train_x[:12]
        leak_test_y[:12] = train_y[:12]
        run.check_leakage(train_x, train_y, leak_test_x, leak_test_y, max_samples=128, print_summary=False)

    run.flush()
    report = run.analyze(print_summary=False)
    if finish_run:
        run.finish(open=False, analyze=False)
    return {"run": run, "model": model, "report": report, "val_x": val_x, "val_y": val_y}


def create_distributed_artifacts(store: ArtifactStore, run_id: str, *, healthy_nodes: int = 2) -> dict[str, Any]:
    run_dir = store.get_run_dir(run_id)
    payload = {
        "run_id": run_id,
        "world_size": 2,
        "total_nodes": 2,
        "active_nodes": healthy_nodes,
        "healthy_nodes": healthy_nodes,
        "backend": "synthetic",
    }
    with open(run_dir / "distributed_index.json", "w") as f:
        json.dump(payload, f, indent=2)
    for rank in range(2):
        rank_dir = run_dir / f"rank_{rank}"
        rank_dir.mkdir(parents=True, exist_ok=True)
        with open(rank_dir / "heartbeat.json", "w") as f:
            json.dump({"rank": rank, "heartbeat_ts": time.time()}, f)
    return payload


def degrade_run_for_showcase(run, *, val_x: torch.Tensor, val_y: torch.Tensor, failure_message: str) -> None:
    bad_targets = 1 - val_y[:16]
    bad_logits = torch.full((16, 2), -4.0)
    bad_logits[torch.arange(16), bad_targets] = 6.5

    for offset, (loss, acc, val_loss, val_acc) in enumerate(
        [(0.92, 0.69, 1.08, 0.58), (1.37, 0.54, 1.82, 0.42), (1.81, 0.48, 2.24, 0.31)], start=1
    ):
        run.log(loss=loss, acc=acc, val_loss=val_loss, val_acc=val_acc, error_rate=1.0 - val_acc, lr=0.02 * offset)
        run.log_batch(x=val_x[:16], y=val_y[:16], y_pred=bad_logits, loss=val_loss)

    run.flush()
    run.analyze(print_summary=False)
    run.fail(failure_message, open=False, analyze=False)


def configure_store(root: Path) -> ArtifactStore:
    gg.configure(root=str(root), auto_open=False)
    store = ArtifactStore(root=root)
    gg.store = store
    return store


def port_is_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def resolve_server_port(requested_port: int) -> tuple[int, str | None]:
    if requested_port == 0 or port_is_available(requested_port):
        return requested_port, None
    fallback_port = find_free_port()
    return fallback_port, f"Port {requested_port} is already in use; using free port {fallback_port} instead."
