from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import pytest

pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from gradglass.artifacts import ArtifactStore
from gradglass.server import create_app
import gradglass.telemetry as telemetry


@pytest.fixture
def tmp_store():
    tmpdir = tempfile.mkdtemp()
    store = ArtifactStore(root=tmpdir)
    yield store
    shutil.rmtree(tmpdir)


def _seed_run(store: ArtifactStore, run_id: str) -> Path:
    run_dir = store.ensure_run_dir(run_id)
    meta = {
        "name": "overview-run",
        "run_id": run_id,
        "framework": "pytorch",
        "status": "running",
        "start_time_epoch": 100.0,
        "config": {"total_steps": 10, "monitor": True},
    }
    (run_dir / "metadata.json").write_text(json.dumps(meta))

    with open(run_dir / "metrics.jsonl", "w") as f:
        f.write(json.dumps({"step": 1, "timestamp": 101.0, "loss": 1.0, "val_loss": 1.2, "lr": 0.01}) + "\n")
        f.write(json.dumps({"step": 2, "timestamp": 102.0, "loss": 0.8, "val_loss": 1.0, "lr": 0.005}) + "\n")

    runtime = {
        "status": "running",
        "heartbeat_ts": 102.1,
        "current_step": 2,
        "monitor_enabled": True,
        "resource_tracking_available": True,
    }
    (run_dir / "runtime_state.json").write_text(json.dumps(runtime))
    return run_dir


def test_overview_endpoint_returns_normalized_snapshot(tmp_store):
    run_id = "overview-api-run"
    _seed_run(tmp_store, run_id)
    app = create_app(tmp_store)

    client = TestClient(app)
    response = client.get(f"/api/runs/{run_id}/overview")

    assert response.status_code == 200
    payload = response.json()
    assert payload["run_id"] == run_id
    assert payload["health_state"] in {"HEALTHY", "WARNING", "STALLED", "FAILED"}
    assert payload["current_step"] == 2
    assert payload["total_steps_source"] in {
        "runtime",
        "config",
        "epoch_inference",
        "diagnostics",
        "completion_fallback",
        "unknown",
    }
    assert isinstance(payload["loss_history"], list)
    assert isinstance(payload["lr_history"], list)


def test_stream_emits_metrics_and_overview_updates(tmp_store):
    run_id = "overview-ws-run"
    _seed_run(tmp_store, run_id)
    app = create_app(tmp_store)

    client = TestClient(app)
    with client.websocket_connect(f"/api/runs/{run_id}/stream") as ws:
        seen = set()
        for _ in range(4):
            msg = ws.receive_json()
            seen.add(msg.get("type"))
            if "metrics_update" in seen and "overview_update" in seen:
                if msg.get("type") == "overview_update":
                    assert "total_steps_source" in msg.get("data", {})
                break

        assert "metrics_update" in seen
        assert "overview_update" in seen


def test_infrastructure_endpoint_returns_metric_payload(tmp_store):
    run_id = "infra-api-run"
    _seed_run(tmp_store, run_id)
    app = create_app(tmp_store)

    client = TestClient(app)
    response = client.get(f"/api/runs/{run_id}/infrastructure")

    assert response.status_code == 200
    payload = response.json()
    assert payload["run_id"] == run_id
    assert payload["mode"] in {"standalone", "distributed"}
    assert "collected_at" in payload
    assert "metrics" in payload
    assert "live_guard" in payload
    assert set(payload["live_guard"].keys()) == {
        "ok",
        "reasons",
        "collected_at",
        "server_pid",
        "hostname",
        "run_id",
    }
    assert isinstance(payload["live_guard"]["ok"], bool)
    assert isinstance(payload["live_guard"]["reasons"], list)
    assert payload["live_guard"]["run_id"] == run_id

    expected_metrics = {
        "cluster_nodes",
        "system_cpu",
        "system_ram",
        "power_draw",
        "multi_gpu_compute_utilization",
        "gpu_memory_fragmentation",
    }
    assert set(payload["metrics"].keys()) == expected_metrics
    for metric in payload["metrics"].values():
        assert telemetry.REQUIRED_METRIC_KEYS.issubset(metric.keys())

    assert "telemetry_v2" in payload
    v2 = payload["telemetry_v2"]
    assert "external_usage" in v2
    assert "graph_hints" in v2
    assert v2["graph_hints"]["preferred_layout"] in {"host_process_first", "accelerator_first"}
    assert isinstance(v2["graph_hints"]["run_terminal"], bool)
    assert isinstance(v2["graph_hints"]["throughput_warmup"], bool)


def test_infrastructure_endpoint_survives_partial_probe_failure(tmp_store, monkeypatch):
    run_id = "infra-partial-failure-run"
    _seed_run(tmp_store, run_id)
    app = create_app(tmp_store)

    def _boom():
        raise RuntimeError("forced cpu failure")

    monkeypatch.setattr(telemetry, "query_system_cpu", _boom)

    client = TestClient(app)
    response = client.get(f"/api/runs/{run_id}/infrastructure")

    assert response.status_code == 200
    payload = response.json()
    assert payload["metrics"]["system_cpu"]["status"] == "error"
    assert "forced cpu failure" in (payload["metrics"]["system_cpu"]["error"] or "")


def test_infrastructure_endpoint_returns_404_for_unknown_run(tmp_store):
    app = create_app(tmp_store)
    client = TestClient(app)

    response = client.get("/api/runs/missing-run/infrastructure")
    assert response.status_code == 404
