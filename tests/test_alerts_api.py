from __future__ import annotations

import json
import shutil
import tempfile
import time
from pathlib import Path

import pytest

pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from gradglass.artifacts import ArtifactStore
from gradglass.server import create_app


@pytest.fixture
def tmp_store():
    tmpdir = tempfile.mkdtemp()
    store = ArtifactStore(root=tmpdir)
    yield store
    shutil.rmtree(tmpdir)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _write_metrics(path: Path, rows: list[dict]) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _seed_completed_alert_run(store: ArtifactStore, run_id: str) -> None:
    run_dir = store.ensure_run_dir(run_id)
    _write_json(
        run_dir / "metadata.json",
        {
            "name": "transfer-learning",
            "run_id": run_id,
            "framework": "pytorch",
            "status": "complete",
            "start_time_epoch": 100.0,
            "config": {"monitor": True},
        },
    )
    _write_json(
        run_dir / "runtime_state.json",
        {
            "status": "complete",
            "heartbeat_ts": 120.0,
            "current_step": 256,
            "monitor_enabled": True,
            "resource_tracking_available": True,
        },
    )
    _write_metrics(
        run_dir / "metrics.jsonl",
        [
            {"step": 1, "timestamp": 101.0, "loss": 1.65, "acc": 0.15},
            {"step": 2, "timestamp": 102.0, "loss": 1.58, "acc": 0.32},
        ],
    )
    _write_json(
        run_dir / "gradients" / "summaries_step_256.json",
        {
            "head.fc1.weight": {"mean": 1e-08, "var": 0.04, "max": 0.15, "norm": 0.104173, "kl_div": 0.66},
            "head.fc1.bias": {"mean": 2e-08, "var": 0.02, "max": 0.08, "norm": 0.020949, "kl_div": 0.72},
            "head.fc2.weight": {"mean": 1e-08, "var": 0.09, "max": 0.1, "norm": 0.135705, "kl_div": 0.1},
        },
    )
    _write_json(
        run_dir / "analysis" / "report.json",
        {
            "run_id": run_id,
            "generated_at": "2026-03-24 00:00:00",
            "checkpoint_diff_summary": {},
            "gradient_flow_analysis": {},
            "training_metrics_summary": {},
            "artifact_store_summary": {},
            "tests": {
                "passed": 43,
                "warned": 4,
                "failed": 0,
                "skipped": 15,
                "total": 62,
                "results": [
                    {
                        "id": "LEARNING_RATE_LOGGED",
                        "title": "Learning rate is being tracked",
                        "status": "warn",
                        "severity": "LOW",
                        "category": "Training Metrics",
                        "details": {},
                        "recommendation": "Pass optimizer to run.watch() to enable LR tracking.",
                    },
                    {
                        "id": "WEIGHT_DIFF_SEVERITY_COUNTS",
                        "title": "Weight diff severity distribution",
                        "status": "warn",
                        "severity": "MEDIUM",
                        "category": "Checkpoint Diff",
                        "details": {
                            "severity_counts": {"low": 0, "medium": 0, "high": 2, "critical": 8},
                            "total_layers": 10,
                            "critical_ratio": 0.8,
                        },
                        "recommendation": "Over 50% of layers have CRITICAL severity changes. This is expected for full training.",
                    },
                    {
                        "id": "LABEL_FLIP_RATE",
                        "title": "Label flip rate is reasonable",
                        "status": "warn",
                        "severity": "MEDIUM",
                        "category": "Predictions",
                        "details": {"flips": 6, "total": 10, "flip_rate": 0.6, "step_a": 1, "step_b": 256},
                        "recommendation": "High label flip rate may indicate unstable training.",
                    },
                    {
                        "id": "SEED_LOGGED",
                        "title": "Random seed is logged",
                        "status": "warn",
                        "severity": "LOW",
                        "category": "Reproducibility",
                        "details": {},
                        "recommendation": "Pass seed= in run config for reproducibility tracking.",
                    },
                ],
            },
        },
    )


def _seed_overfitting_and_gradient_run(store: ArtifactStore, run_id: str) -> None:
    run_dir = store.ensure_run_dir(run_id)
    now = time.time()
    _write_json(
        run_dir / "metadata.json",
        {
            "name": "unstable-live-run",
            "run_id": run_id,
            "framework": "pytorch",
            "status": "running",
            "start_time_epoch": now - 60,
            "config": {"monitor": True},
        },
    )
    _write_json(
        run_dir / "runtime_state.json",
        {
            "status": "running",
            "heartbeat_ts": now,
            "current_step": 10,
            "monitor_enabled": True,
            "resource_tracking_available": True,
        },
    )
    rows = []
    for step, (loss, val_loss) in enumerate(
        [
            (1.2, 0.8),
            (1.0, 0.82),
            (0.85, 0.88),
            (0.75, 0.96),
            (0.65, 1.08),
            (0.55, 1.18),
            (0.48, 1.28),
            (0.4, 1.38),
            (0.34, 1.5),
            (0.28, 1.62),
        ],
        start=1,
    ):
        rows.append({"step": step, "timestamp": now - 40 + step, "loss": loss, "val_loss": val_loss})
    _write_metrics(run_dir / "metrics.jsonl", rows)
    _write_json(
        run_dir / "gradients" / "summaries_step_10.json",
        {
            "encoder.weight": {"mean": 50.0, "var": 10.0, "max": 200.0, "norm": 500.0, "kl_div": 0.2},
            "head.weight": {"mean": 1e-10, "var": 1e-14, "max": 1e-09, "norm": 5e-09, "kl_div": 0.1},
        },
    )


def _seed_stalled_nan_run(store: ArtifactStore, run_id: str) -> None:
    run_dir = store.ensure_run_dir(run_id)
    _write_json(
        run_dir / "metadata.json",
        {
            "name": "stalled-run",
            "run_id": run_id,
            "framework": "pytorch",
            "status": "running",
            "start_time_epoch": 10.0,
            "config": {"monitor": True},
        },
    )
    _write_json(
        run_dir / "runtime_state.json",
        {
            "status": "running",
            "heartbeat_ts": 1.0,
            "current_step": 3,
            "monitor_enabled": True,
            "resource_tracking_available": True,
        },
    )
    _write_metrics(
        run_dir / "metrics.jsonl",
        [
            {"step": 1, "timestamp": 1.0, "loss": 1.0},
            {"step": 2, "timestamp": 2.0, "loss": 0.9},
            {"step": 3, "timestamp": 3.0, "loss": float("nan")},
        ],
    )


def _seed_dedup_run(store: ArtifactStore, run_id: str) -> None:
    run_dir = store.ensure_run_dir(run_id)
    now = time.time()
    _write_json(
        run_dir / "metadata.json",
        {
            "name": "dedupe-run",
            "run_id": run_id,
            "framework": "pytorch",
            "status": "complete",
            "start_time_epoch": now - 100,
            "config": {"monitor": True},
        },
    )
    _write_json(
        run_dir / "runtime_state.json",
        {
            "status": "complete",
            "heartbeat_ts": now,
            "current_step": 8,
            "monitor_enabled": True,
            "resource_tracking_available": True,
        },
    )
    rows = []
    for step, (loss, val_loss) in enumerate(
        [
            (1.1, 0.8),
            (0.9, 0.86),
            (0.75, 0.98),
            (0.63, 1.08),
            (0.52, 1.16),
            (0.44, 1.26),
            (0.38, 1.36),
            (0.32, 1.46),
        ],
        start=1,
    ):
        rows.append({"step": step, "timestamp": now - 20 + step, "loss": loss, "val_loss": val_loss})
    _write_metrics(run_dir / "metrics.jsonl", rows)
    _write_json(
        run_dir / "analysis" / "report.json",
        {
            "run_id": run_id,
            "generated_at": "2026-03-24 00:00:00",
            "checkpoint_diff_summary": {},
            "gradient_flow_analysis": {},
            "training_metrics_summary": {},
            "artifact_store_summary": {},
            "tests": {
                "passed": 0,
                "warned": 0,
                "failed": 2,
                "skipped": 0,
                "total": 2,
                "results": [
                    {
                        "id": "OVERFITTING_HEURISTIC",
                        "title": "No overfitting detected",
                        "status": "fail",
                        "severity": "HIGH",
                        "category": "Training Metrics",
                        "details": {"val_loss_increase_ratio": 0.55},
                        "recommendation": "Overfitting confirmed: validation loss is rising while training loss is falling. Use early stopping, dropout, weight decay, or more data.",
                    },
                    {
                        "id": "VAL_LOSS_DIVERGENCE",
                        "title": "Validation loss not diverging",
                        "status": "fail",
                        "severity": "HIGH",
                        "category": "Training Metrics",
                        "details": {"val_loss_start": 1.08, "val_loss_end": 1.46, "rise_rate": 1.0},
                        "recommendation": "Validation loss is diverging (rising continuously). Stop training earlier, add early stopping, or reduce model complexity.",
                    },
                ],
            },
        },
    )


def _seed_stream_run(store: ArtifactStore, run_id: str) -> None:
    run_dir = store.ensure_run_dir(run_id)
    now = time.time()
    _write_json(
        run_dir / "metadata.json",
        {
            "name": "stream-run",
            "run_id": run_id,
            "framework": "pytorch",
            "status": "running",
            "start_time_epoch": now - 30,
            "config": {"monitor": True, "total_steps": 10},
        },
    )
    _write_json(
        run_dir / "runtime_state.json",
        {
            "status": "running",
            "heartbeat_ts": now,
            "current_step": 2,
            "monitor_enabled": True,
            "resource_tracking_available": True,
        },
    )
    _write_metrics(
        run_dir / "metrics.jsonl",
        [
            {"step": 1, "timestamp": now - 2, "loss": 1.0, "val_loss": 1.1, "lr": 0.01},
            {"step": 2, "timestamp": now - 1, "loss": 0.9, "val_loss": 1.0, "lr": 0.008},
        ],
    )


def test_alerts_endpoint_merges_report_and_gradient_flags(tmp_store):
    run_id = "alerts-completed"
    _seed_completed_alert_run(tmp_store, run_id)
    app = create_app(tmp_store)
    client = TestClient(app)

    response = client.get(f"/api/runs/{run_id}/alerts")
    assert response.status_code == 200

    payload = response.json()
    ids = {alert["id"] for alert in payload["alerts"]}
    assert payload["health_state"] == "HEALTHY"
    assert payload["summary"]["total"] >= 6
    assert payload["summary"]["high_severity"] == 0
    assert payload["summary"]["warnings"] >= 6
    assert {
        "LEARNING_RATE_LOGGED",
        "WEIGHT_DIFF_SEVERITY_COUNTS",
        "LABEL_FLIP_RATE",
        "SEED_LOGGED",
        "GRAD_FLAG_NOISY",
        "GRAD_FLAG_DISTRIBUTION_SHIFT",
    }.issubset(ids)


def test_alerts_endpoint_surfaces_unhealthy_metrics_and_gradients_with_fixes(tmp_store):
    run_id = "alerts-unhealthy"
    _seed_overfitting_and_gradient_run(tmp_store, run_id)
    app = create_app(tmp_store)
    client = TestClient(app)

    response = client.get(f"/api/runs/{run_id}/alerts")
    assert response.status_code == 200

    payload = response.json()
    by_id = {alert["id"]: alert for alert in payload["alerts"]}
    expected_ids = {"OVERFITTING_HEURISTIC", "VAL_LOSS_DIVERGENCE", "GRAD_EXPLODING", "GRAD_VANISHING"}
    assert expected_ids.issubset(by_id.keys())
    for alert_id in expected_ids:
        assert by_id[alert_id]["recommendation"]


def test_alerts_endpoint_surfaces_stalled_runtime_and_non_finite_loss(tmp_store):
    run_id = "alerts-stalled-nan"
    _seed_stalled_nan_run(tmp_store, run_id)
    app = create_app(tmp_store)
    client = TestClient(app)

    response = client.get(f"/api/runs/{run_id}/alerts")
    assert response.status_code == 200

    payload = response.json()
    by_id = {alert["id"]: alert for alert in payload["alerts"]}
    assert payload["health_state"] == "STALLED"
    assert {"RUNTIME_STALLED", "LOSS_FINITE"}.issubset(by_id.keys())
    assert by_id["RUNTIME_STALLED"]["recommendation"]
    assert by_id["LOSS_FINITE"]["recommendation"]


def test_alerts_endpoint_deduplicates_report_and_live_heuristics(tmp_store):
    run_id = "alerts-dedupe"
    _seed_dedup_run(tmp_store, run_id)
    app = create_app(tmp_store)
    client = TestClient(app)

    response = client.get(f"/api/runs/{run_id}/alerts")
    assert response.status_code == 200

    payload = response.json()
    ids = [alert["id"] for alert in payload["alerts"]]
    assert ids.count("OVERFITTING_HEURISTIC") == 1


def test_stream_emits_alert_updates(tmp_store):
    run_id = "alerts-stream"
    _seed_stream_run(tmp_store, run_id)
    app = create_app(tmp_store)

    client = TestClient(app)
    with client.websocket_connect(f"/api/runs/{run_id}/stream") as ws:
        seen = set()
        for _ in range(6):
            msg = ws.receive_json()
            seen.add(msg.get("type"))
            if {"metrics_update", "overview_update", "alerts_update"}.issubset(seen):
                if msg.get("type") == "alerts_update":
                    assert "summary" in msg.get("data", {})
                break

        assert {"metrics_update", "overview_update", "alerts_update"}.issubset(seen)
