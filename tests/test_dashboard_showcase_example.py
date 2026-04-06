from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys
import urllib.parse
import urllib.request

import pytest

from gradglass.server import create_app, start_server


def _load_example_module():
    root = Path(__file__).resolve().parents[1]
    example_path = root / "examples" / "09_dashboard_showcase.py"
    spec = importlib.util.spec_from_file_location("dashboard_showcase_example", example_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _read_json(url: str):
    with urllib.request.urlopen(url, timeout=10) as response:
        import json

        return json.loads(response.read().decode("utf-8"))


def test_dashboard_showcase_example_populates_dashboard_endpoints(tmp_path):
    probe = subprocess.run(
        [sys.executable, "-c", "import torch"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if probe.returncode != 0:
        pytest.skip("Torch is not importable in this environment; skipping dashboard showcase example test.")

    example = _load_example_module()
    workspace = tmp_path / "dashboard-showcase-workspace"
    bundle = example.create_showcase_runs(workspace)
    app = create_app(bundle["store"])
    port = start_server(app, port=0)
    base_url = f"http://127.0.0.1:{port}"

    runs_payload = _read_json(f"{base_url}/api/runs")
    assert runs_payload["total"] >= 3

    for run in bundle["runs"].values():
        run_id = urllib.parse.quote(run.run_id, safe="")
        overview = _read_json(f"{base_url}/api/runs/{run_id}/overview")
        metrics = _read_json(f"{base_url}/api/runs/{run_id}/metrics")
        checkpoints = _read_json(f"{base_url}/api/runs/{run_id}/checkpoints")
        evaluation = _read_json(f"{base_url}/api/runs/{run_id}/eval")
        alerts = _read_json(f"{base_url}/api/runs/{run_id}/alerts")
        analysis = _read_json(f"{base_url}/api/runs/{run_id}/analysis")
        architecture = _read_json(f"{base_url}/api/runs/{run_id}/architecture")
        infrastructure = _read_json(f"{base_url}/api/runs/{run_id}/infrastructure")
        data_monitor = _read_json(f"{base_url}/api/runs/{run_id}/data-monitor")
        saliency = _read_json(f"{base_url}/api/runs/{run_id}/saliency")
        shap = _read_json(f"{base_url}/api/runs/{run_id}/shap")
        lime = _read_json(f"{base_url}/api/runs/{run_id}/lime")
        predictions = _read_json(f"{base_url}/api/runs/{run_id}/predictions")

        assert overview["run_id"] == run.run_id
        assert metrics["total"] > 0
        assert len(checkpoints["checkpoints"]) >= 2
        assert evaluation["report"]["evaluations"]
        assert analysis["tests"]["total"] > 0
        assert architecture["layers"]
        assert "telemetry_v2" in infrastructure
        assert data_monitor["metadata"]["dataset_name"]
        assert saliency["available"] is True
        assert shap["summary_plot"]
        assert lime["samples"]
        assert predictions["predictions"]
        assert "summary" in alerts

    baseline_id = bundle["runs"]["baseline"].run_id
    primary_id = bundle["runs"]["primary"].run_id
    compare = _read_json(f"{base_url}/api/compare?run_ids={baseline_id},{primary_id}")
    assert baseline_id in compare
    assert primary_id in compare


def test_dashboard_showcase_main_prints_workspace_matching_server_root(tmp_path, monkeypatch, capsys):
    probe = subprocess.run(
        [sys.executable, "-c", "import torch"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if probe.returncode != 0:
        pytest.skip("Torch is not importable in this environment; skipping dashboard showcase example test.")

    example = _load_example_module()
    workspace = tmp_path / "dashboard-showcase-main"
    popen_calls = []

    class DummyProcess:
        pid = 424242

        def poll(self):
            return None

    def fake_popen(cmd, **kwargs):
        popen_calls.append((cmd, kwargs))
        return DummyProcess()

    monkeypatch.setattr("subprocess.Popen", fake_popen)
    monkeypatch.setattr("gradglass.server._wait_for_server", lambda host, port, timeout=10.0: True)

    example.main(["--root", str(workspace), "--port", "8459", "--no-browser"])
    captured = capsys.readouterr().out

    assert f"Workspace: {workspace}" in captured
    assert "Dashboard: http://127.0.0.1:8459" in captured
    assert popen_calls
    cmd = next(call[0] for call in popen_calls if "--root" in call[0])
    assert "--root" in cmd
    assert cmd[cmd.index("--root") + 1] == str(workspace)
