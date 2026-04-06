from __future__ import annotations

import argparse
import asyncio
import json
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import websockets
from torch.utils.data import DataLoader, TensorDataset

from gradglass import gg
from gradglass.analysis.data_monitor import PipelineStage
from gradglass.artifacts import ArtifactStore, resolve_default_root
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
                    {"feature": feature_names[int(idx)], "weight": float(round(row[int(idx)], 6))}
                    for idx in ranked
                ],
            }
        )
    return samples


def evaluate_model(model: nn.Module, x: torch.Tensor, y: torch.Tensor, criterion: nn.Module) -> tuple[torch.Tensor, float, float]:
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
    include_extra_artifacts: bool,
    finish_run: bool,
) -> dict[str, Any]:
    train_x, train_y, val_x, val_y = make_dataset(seed=seed)
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=32, shuffle=True)

    model = CoverageMLP(width=width)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.CrossEntropyLoss()

    run = gg.run(name=name, task="classification", monitor=False)
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

    for _epoch in range(5):
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

        run.log_shap(
            FEATURE_NAMES,
            shap_values,
            message="Coverage harness SHAP summary.",
            top_k=len(FEATURE_NAMES),
        )

        shap_scores = run.store.get_shap(run.run_id)["summary_plot"]
        lime_weights = np.array([item["mean_shap"] for item in shap_scores], dtype=np.float64)
        lime_rows = np.tile(lime_weights, (4, 1))
        run.log_lime(
            build_lime_samples([item["feature"] for item in shap_scores], lime_rows),
            message="Coverage harness LIME-style local explanations.",
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


def create_distributed_artifacts(store: ArtifactStore, run_id: str) -> dict[str, Any]:
    run_dir = store.get_run_dir(run_id)
    payload = {
        "run_id": run_id,
        "world_size": 2,
        "total_nodes": 2,
        "active_nodes": 2,
        "healthy_nodes": 2,
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


def create_runs_for_coverage(root: Path) -> dict[str, Any]:
    gg.configure(root=str(root), auto_open=False)
    store = ArtifactStore(root=root)
    gg.store = store

    baseline = build_tabular_run(
        store,
        name="api_baseline_run",
        width=16,
        seed=11,
        include_extra_artifacts=False,
        finish_run=True,
    )
    coverage = build_tabular_run(
        store,
        name="api_coverage_run",
        width=24,
        seed=17,
        include_extra_artifacts=True,
        finish_run=False,
    )
    distributed = create_distributed_artifacts(store, coverage["run"].run_id)

    return {
        "store": store,
        "baseline_run": baseline["run"],
        "coverage_run": coverage["run"],
        "coverage_model": coverage["model"],
        "coverage_val_x": coverage["val_x"],
        "coverage_val_y": coverage["val_y"],
        "distributed": distributed,
    }


def _read_json(url: str, *, method: str = "GET", payload: dict[str, Any] | None = None) -> Any:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"{method} {url} failed with HTTP {exc.code}: {detail}") from exc


def _port_is_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def resolve_server_port(requested_port: int) -> tuple[int, str | None]:
    if requested_port == 0 or _port_is_available(requested_port):
        return requested_port, None
    fallback_port = find_free_port()
    return fallback_port, f"Port {requested_port} is already in use; using free port {fallback_port} instead."


async def collect_stream_events(
    websocket_url: str,
    trigger_fn,
    *,
    expected_types: set[str] | None = None,
    attempts: int = 8,
) -> list[dict[str, Any]]:
    expected = expected_types or {"metrics_update", "overview_update", "alerts_update"}
    events: list[dict[str, Any]] = []
    async with websockets.connect(websocket_url) as websocket:
        trigger_fn()
        seen = set()
        for _ in range(attempts):
            message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
            payload = json.loads(message)
            events.append(payload)
            seen.add(payload.get("type"))
            if expected.issubset(seen):
                break
    return events


def exercise_api(
    base_url: str,
    *,
    baseline_run,
    coverage_run,
    coverage_model: nn.Module,
    coverage_val_x: torch.Tensor,
    coverage_val_y: torch.Tensor,
) -> list[dict[str, Any]]:
    encoded_coverage = urllib.parse.quote(coverage_run.run_id, safe="")
    encoded_baseline = urllib.parse.quote(baseline_run.run_id, safe="")

    results = []

    def check(name: str, condition: bool, detail: str) -> None:
        results.append({"name": name, "ok": bool(condition), "detail": detail})

    runs_payload = _read_json(f"{base_url}/api/runs")
    check("GET /api/runs", runs_payload.get("total", 0) >= 2, f"runs={runs_payload.get('total', 0)}")

    compare_payload = _read_json(
        f"{base_url}/api/compare?run_ids={encoded_baseline},{encoded_coverage}"
    )
    check(
        "GET /api/compare",
        baseline_run.run_id in compare_payload and coverage_run.run_id in compare_payload,
        f"keys={sorted(compare_payload.keys())}",
    )

    run_payload = _read_json(f"{base_url}/api/runs/{encoded_coverage}")
    check("GET /api/runs/{run_id}", run_payload.get("run_id") == coverage_run.run_id, f"status={run_payload.get('status')}")

    metrics_payload = _read_json(f"{base_url}/api/runs/{encoded_coverage}/metrics")
    check("GET /metrics", metrics_payload.get("total", 0) > 0, f"metrics={metrics_payload.get('total', 0)}")

    overview_payload = _read_json(f"{base_url}/api/runs/{encoded_coverage}/overview")
    check("GET /overview", overview_payload.get("run_id") == coverage_run.run_id, f"health={overview_payload.get('health_state')}")

    alerts_payload = _read_json(f"{base_url}/api/runs/{encoded_coverage}/alerts")
    check("GET /alerts", "summary" in alerts_payload, f"alerts={alerts_payload.get('summary', {}).get('total')}")

    checkpoints_payload = _read_json(f"{base_url}/api/runs/{encoded_coverage}/checkpoints")
    checkpoint_steps = [entry["step"] for entry in checkpoints_payload.get("checkpoints", [])]
    check("GET /checkpoints", len(checkpoint_steps) >= 2, f"steps={checkpoint_steps}")

    if len(checkpoint_steps) >= 2:
        diff_payload = _read_json(
            f"{base_url}/api/runs/{encoded_coverage}/diff?a={checkpoint_steps[0]}&b={checkpoint_steps[-1]}&include_deltas=true"
        )
        check("GET /diff", bool(diff_payload.get("layers")), f"layers={len(diff_payload.get('layers', []))}")
    else:
        check("GET /diff", False, "Need >=2 checkpoints")

    gradients_payload = _read_json(f"{base_url}/api/runs/{encoded_coverage}/gradients")
    check("GET /gradients", bool(gradients_payload.get("summaries")), f"summaries={len(gradients_payload.get('summaries', []))}")

    activations_payload = _read_json(f"{base_url}/api/runs/{encoded_coverage}/activations")
    check("GET /activations", bool(activations_payload.get("activations")), f"activations={len(activations_payload.get('activations', []))}")

    distributions_payload = _read_json(f"{base_url}/api/runs/{encoded_coverage}/distributions")
    check(
        "GET /distributions",
        distributions_payload.get("weights", {}).get("available") or distributions_payload.get("activations", {}).get("available"),
        f"default_mode={distributions_payload.get('default_mode')}",
    )

    saliency_payload = _read_json(f"{base_url}/api/runs/{encoded_coverage}/saliency")
    check("GET /saliency", saliency_payload.get("available") is True, f"modality={saliency_payload.get('modality')}")

    embeddings_payload = _read_json(f"{base_url}/api/runs/{encoded_coverage}/embeddings")
    check("GET /embeddings", embeddings_payload.get("available") is True, f"layers={len(embeddings_payload.get('layers', []))}")

    shap_payload = _read_json(f"{base_url}/api/runs/{encoded_coverage}/shap")
    check("GET /shap", bool(shap_payload.get("summary_plot")), f"features={len(shap_payload.get('summary_plot', []))}")

    lime_payload = _read_json(f"{base_url}/api/runs/{encoded_coverage}/lime")
    check("GET /lime", bool(lime_payload.get("samples")), f"samples={len(lime_payload.get('samples', []))}")

    predictions_payload = _read_json(f"{base_url}/api/runs/{encoded_coverage}/predictions")
    check(
        "GET /predictions",
        bool(predictions_payload.get("predictions")) and bool(predictions_payload.get("diffs")),
        f"predictions={len(predictions_payload.get('predictions', []))}",
    )

    architecture_payload = _read_json(f"{base_url}/api/runs/{encoded_coverage}/architecture")
    non_input_layers = [layer for layer in architecture_payload.get("layers", []) if "." in layer.get("id", "") or layer.get("param_count", 0) > 0]
    check("GET /architecture", bool(architecture_payload.get("layers")), f"layers={len(architecture_payload.get('layers', []))}")

    analysis_payload = _read_json(f"{base_url}/api/runs/{encoded_coverage}/analysis")
    check("GET /analysis", "tests" in analysis_payload, f"tests={analysis_payload.get('tests', {}).get('total')}")

    freeze_payload = _read_json(f"{base_url}/api/runs/{encoded_coverage}/freeze_code")
    check("GET /freeze_code", "pytorch_code" in freeze_payload, f"candidates={freeze_payload.get('total_candidates')}")

    architecture_diff_payload = _read_json(
        f"{base_url}/api/runs/{encoded_coverage}/architecture/diff?compare_run_id={encoded_baseline}"
    )
    check(
        "GET /architecture/diff",
        "added_layers" in architecture_diff_payload and "removed_layers" in architecture_diff_payload,
        f"identical={architecture_diff_payload.get('is_identical')}",
    )

    leakage_payload = _read_json(f"{base_url}/api/runs/{encoded_coverage}/leakage")
    check("GET /leakage", bool(leakage_payload.get("results")), f"checks={len(leakage_payload.get('results', []))}")

    data_monitor_payload = _read_json(f"{base_url}/api/runs/{encoded_coverage}/data-monitor")
    check("GET /data-monitor", "metadata" in data_monitor_payload, f"dataset={data_monitor_payload.get('metadata', {}).get('dataset_name')}")

    if non_input_layers:
        target_layer = non_input_layers[0]["id"]
        mutate_payload = _read_json(
            f"{base_url}/api/runs/{encoded_coverage}/architecture/mutate",
            method="POST",
            payload={"operation": "freeze", "target_layer": target_layer, "params": {}},
        )
        check("POST /architecture/mutate", mutate_payload.get("valid") is True, f"target={target_layer}")
    else:
        check("POST /architecture/mutate", False, "No mutable layer found")

    distributed_payload = _read_json(f"{base_url}/api/runs/{encoded_coverage}/distributed")
    check(
        "GET /distributed",
        bool(distributed_payload.get("distributed_index")) and len(distributed_payload.get("ranks", [])) >= 2,
        f"ranks={distributed_payload.get('ranks', [])}",
    )

    infrastructure_payload = _read_json(f"{base_url}/api/runs/{encoded_coverage}/infrastructure")
    check("GET /infrastructure", "telemetry_v2" in infrastructure_payload, f"mode={infrastructure_payload.get('mode')}")

    eval_payload = _read_json(f"{base_url}/api/runs/{encoded_coverage}/eval")
    check("GET /eval", bool(eval_payload.get("report", {}).get("evaluations")), f"evals={len(eval_payload.get('report', {}).get('evaluations', []))}")

    criterion = nn.CrossEntropyLoss()

    def trigger_new_events():
        logits, val_loss, val_acc = evaluate_model(coverage_model, coverage_val_x, coverage_val_y, criterion)
        coverage_run.log(loss=val_loss, acc=val_acc, val_loss=val_loss, val_acc=val_acc)
        coverage_run.log_batch(x=coverage_val_x[:16], y=coverage_val_y[:16], y_pred=logits[:16], loss=val_loss)

    events = asyncio.run(
        collect_stream_events(
            f"{base_url.replace('http', 'ws')}/api/runs/{encoded_coverage}/stream",
            trigger_new_events,
        )
    )
    event_types = {event.get("type") for event in events}
    check("WS /stream", {"metrics_update", "overview_update", "alerts_update"}.issubset(event_types), f"events={sorted(event_types)}")

    return results


def print_coverage_table(results: list[dict[str, Any]]) -> None:
    print("\nGradGlass API Feature Coverage")
    print("=" * 72)
    for row in results:
        status = "PASS" if row["ok"] else "FAIL"
        print(f"{status:<5} {row['name']:<32} {row['detail']}")
    passed = sum(1 for row in results if row["ok"])
    print("-" * 72)
    print(f"Passed {passed}/{len(results)} coverage checks")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic GradGlass runs and exercise every current /api feature.")
    parser.add_argument(
        "--root",
        default=None,
        help="Workspace root for generated artifacts. Defaults to ./gg_workspace beside this example.",
    )
    parser.add_argument("--port", type=int, default=8432, help="Port for the GradGlass server.")
    parser.add_argument("--open-browser", action="store_true", help="Open the dashboard in a browser.")
    args = parser.parse_args()

    root = Path(args.root).resolve() if args.root else resolve_default_root(entrypoint=__file__)
    bundle = create_runs_for_coverage(root)
    chosen_port, port_note = resolve_server_port(args.port)
    if port_note:
        print(port_note)
    actual_port = bundle["coverage_run"].serve(port=chosen_port, open_browser=args.open_browser)
    base_url = f"http://127.0.0.1:{actual_port}"
    results = exercise_api(
        base_url,
        baseline_run=bundle["baseline_run"],
        coverage_run=bundle["coverage_run"],
        coverage_model=bundle["coverage_model"],
        coverage_val_x=bundle["coverage_val_x"],
        coverage_val_y=bundle["coverage_val_y"],
    )
    bundle["coverage_run"].finish(open=False, analyze=False)
    print_coverage_table(results)
    print(f"\nWorkspace: {root}")
    print(f"Dashboard: {base_url}")


if __name__ == "__main__":
    main()
