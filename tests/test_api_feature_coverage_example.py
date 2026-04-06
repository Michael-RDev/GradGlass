from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest
from gradglass.server import create_app, start_server


EXPECTED_CHECKS = {
    "GET /api/runs",
    "GET /api/compare",
    "GET /api/runs/{run_id}",
    "GET /metrics",
    "GET /overview",
    "GET /alerts",
    "GET /checkpoints",
    "GET /diff",
    "GET /gradients",
    "GET /activations",
    "GET /distributions",
    "GET /saliency",
    "GET /embeddings",
    "GET /shap",
    "GET /lime",
    "GET /predictions",
    "GET /architecture",
    "GET /analysis",
    "GET /freeze_code",
    "GET /architecture/diff",
    "GET /leakage",
    "GET /data-monitor",
    "POST /architecture/mutate",
    "GET /distributed",
    "GET /infrastructure",
    "GET /eval",
    "WS /stream",
}


def _load_example_module():
    root = Path(__file__).resolve().parents[1]
    example_path = root / "examples" / "08_api_feature_coverage.py"
    spec = importlib.util.spec_from_file_location("api_feature_coverage_example", example_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_api_feature_coverage_example_exercises_every_endpoint(tmp_path):
    probe = subprocess.run(
        [sys.executable, "-c", "import torch"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if probe.returncode != 0:
        pytest.skip("Torch is not importable in this environment; skipping API coverage example test.")

    example = _load_example_module()
    workspace = tmp_path / "api-coverage-workspace"

    bundle = example.create_runs_for_coverage(workspace)
    app = create_app(bundle["store"])
    port = start_server(app, port=0)
    base_url = f"http://127.0.0.1:{port}"

    try:
        results = example.exercise_api(
            base_url,
            baseline_run=bundle["baseline_run"],
            coverage_run=bundle["coverage_run"],
            coverage_model=bundle["coverage_model"],
            coverage_val_x=bundle["coverage_val_x"],
            coverage_val_y=bundle["coverage_val_y"],
        )
    finally:
        bundle["coverage_run"].finish(open=False, analyze=False)

    names = {row["name"] for row in results}
    assert names == EXPECTED_CHECKS
    assert all(row["ok"] for row in results), results
    assert len(list((workspace / "runs").iterdir())) >= 2
