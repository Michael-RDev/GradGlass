from __future__ import annotations

import json
import sys
import threading
import time

import pytest

import gradglass.cli as cli
from gradglass.artifacts import ArtifactStore
from gradglass.core import gg
from gradglass.monitor_control import (
    MonitorStopResult,
    MonitorTarget,
    list_standalone_gradglass_targets,
    stop_gradglass_monitor,
    stop_monitor_targets,
)
from gradglass.run import Run
from gradglass.server import create_app, start_server_blocking
from examples._example_output import repo_workspace_root, serve_command_for_workspace


def _write_runtime_state(store: ArtifactStore, run_id: str, payload: dict) -> None:
    run_dir = store.ensure_run_dir(run_id)
    with open(run_dir / "runtime_state.json", "w") as handle:
        json.dump(payload, handle)


def test_stop_gradglass_monitor_by_run_id_updates_runtime_state(tmp_path, monkeypatch):
    store = ArtifactStore(root=tmp_path)
    _write_runtime_state(
        store,
        "run-1",
        {
            "monitor_enabled": True,
            "monitor_port": 8432,
            "monitor_pid": 3210,
        },
    )

    terminated: list[int] = []
    monkeypatch.setattr("gradglass.monitor_control._is_pid_alive", lambda pid: True)
    monkeypatch.setattr("gradglass.monitor_control._terminate_pid", lambda pid: terminated.append(pid) or True)

    results = stop_gradglass_monitor(store, run_id="run-1")

    assert terminated == [3210]
    assert len(results) == 1
    assert results[0].status == "stopped"
    assert results[0].stopped is True
    state = store.get_runtime_state("run-1")
    assert state["monitor_enabled"] is False
    assert state["monitor_pid"] is None
    assert state["monitor_port"] == 8432
    assert state["monitor_stop_reason"] == "cli_stop"
    assert isinstance(state["monitor_stopped_at"], float)


def test_stop_gradglass_monitor_marks_stale_pid_without_failing(tmp_path, monkeypatch):
    store = ArtifactStore(root=tmp_path)
    _write_runtime_state(
        store,
        "run-stale",
        {
            "monitor_enabled": True,
            "monitor_port": 8432,
            "monitor_pid": 9999,
        },
    )

    monkeypatch.setattr("gradglass.monitor_control._is_pid_alive", lambda pid: False)

    results = stop_gradglass_monitor(store, run_id="run-stale")

    assert len(results) == 1
    assert results[0].status == "stale"
    assert results[0].stale is True
    state = store.get_runtime_state("run-stale")
    assert state["monitor_enabled"] is False
    assert state["monitor_pid"] is None
    assert state["monitor_port"] == 8432
    assert state["monitor_stop_reason"] == "stale_pid"
    assert isinstance(state["monitor_stopped_at"], float)


def test_stop_gradglass_monitor_all_stops_multiple_runs(tmp_path, monkeypatch):
    store = ArtifactStore(root=tmp_path)
    _write_runtime_state(store, "run-a", {"monitor_enabled": True, "monitor_port": 8432, "monitor_pid": 1001})
    _write_runtime_state(store, "run-b", {"monitor_enabled": True, "monitor_port": 9432, "monitor_pid": 1002})

    terminated: list[int] = []
    monkeypatch.setattr("gradglass.monitor_control._is_pid_alive", lambda pid: True)
    monkeypatch.setattr("gradglass.monitor_control._terminate_pid", lambda pid: terminated.append(pid) or True)

    results = stop_gradglass_monitor(store, stop_all=True)

    assert len(results) == 2
    assert {result.status for result in results} == {"stopped"}
    assert terminated == [1001, 1002]
    assert store.get_runtime_state("run-a")["monitor_enabled"] is False
    assert store.get_runtime_state("run-b")["monitor_enabled"] is False


def test_stop_gradglass_monitor_all_stops_standalone_verified_server(tmp_path, monkeypatch):
    store = ArtifactStore(root=tmp_path)

    monkeypatch.setattr(
        "gradglass.monitor_control.list_standalone_gradglass_targets",
        lambda store_arg: [
            MonitorTarget(
                run_id=None,
                runtime_state_path=None,
                port=8432,
                pid=5555,
                source="process_scan",
                metadata={"command": "python -m gradglass.server --root /tmp/demo --port 8432"},
            )
        ],
    )
    monkeypatch.setattr("gradglass.monitor_control._is_pid_alive", lambda pid: True)
    terminated: list[int] = []
    monkeypatch.setattr("gradglass.monitor_control._terminate_pid", lambda pid: terminated.append(pid) or True)

    results = stop_gradglass_monitor(store, stop_all=True)

    assert len(results) == 1
    assert results[0].status == "stopped"
    assert results[0].message == "Stopped GradGlass monitor on port 8432 (pid 5555)."
    assert terminated == [5555]


def test_stop_monitor_targets_refuses_multiple_without_all():
    results = stop_monitor_targets(
        [
            MonitorTarget(run_id="run-a", runtime_state_path=None, port=8432, pid=1001, source="runtime_state"),
            MonitorTarget(run_id="run-b", runtime_state_path=None, port=9432, pid=1002, source="runtime_state"),
        ],
        allow_multiple=False,
    )

    assert len(results) == 1
    assert results[0].status == "refused"


def test_stop_monitor_targets_dedupes_shared_pid_and_port(monkeypatch):
    terminated: list[int] = []
    monkeypatch.setattr("gradglass.monitor_control._is_pid_alive", lambda pid: True)
    monkeypatch.setattr("gradglass.monitor_control._terminate_pid", lambda pid: terminated.append(pid) or True)

    results = stop_monitor_targets(
        [
            MonitorTarget(run_id="run-a", runtime_state_path=None, port=8432, pid=5555, source="runtime_state"),
            MonitorTarget(run_id="run-b", runtime_state_path=None, port=8432, pid=5555, source="runtime_state"),
        ],
        allow_multiple=True,
    )

    assert len(results) == 1
    assert results[0].status == "stopped"
    assert terminated == [5555]


def test_stop_gradglass_monitor_by_port_uses_verified_gradglass_process(tmp_path, monkeypatch):
    store = ArtifactStore(root=tmp_path)
    monkeypatch.setattr(
        "gradglass.monitor_control.find_process_by_port",
        lambda port: (5555, "python -m gradglass.server --root /tmp/demo --port 8432"),
    )
    monkeypatch.setattr("gradglass.monitor_control._is_pid_alive", lambda pid: True)
    monkeypatch.setattr("gradglass.monitor_control._terminate_pid", lambda pid: True)

    results = stop_gradglass_monitor(store, port=8432)

    assert len(results) == 1
    assert results[0].status == "stopped"
    assert results[0].message == "Stopped GradGlass monitor on port 8432 (pid 5555)."


def test_stop_gradglass_monitor_by_port_refuses_non_gradglass_process(tmp_path, monkeypatch):
    store = ArtifactStore(root=tmp_path)
    monkeypatch.setattr(
        "gradglass.monitor_control.find_process_by_port",
        lambda port: (5555, "python -m http.server 8432"),
    )

    results = stop_gradglass_monitor(store, port=8432)

    assert len(results) == 1
    assert results[0].status == "refused"


def test_stop_gradglass_monitor_by_port_accepts_repo_cwd_fallback(tmp_path, monkeypatch):
    store = ArtifactStore(root=tmp_path)
    monkeypatch.setattr("gradglass.monitor_control.find_process_by_port", lambda port: (5555, ""))
    monkeypatch.setattr("gradglass.monitor_control._get_process_cwd", lambda pid: str(tmp_path))
    (tmp_path / "gradglass").mkdir()
    (tmp_path / "gradglass" / "server.py").write_text("# marker\n")
    monkeypatch.setattr("gradglass.monitor_control._is_pid_alive", lambda pid: True)
    monkeypatch.setattr("gradglass.monitor_control._terminate_pid", lambda pid: True)

    results = stop_gradglass_monitor(store, port=8432)

    assert len(results) == 1
    assert results[0].status == "stopped"
    assert results[0].message == "Stopped GradGlass monitor on port 8432 (pid 5555)."


def test_list_standalone_gradglass_targets_filters_non_gradglass(monkeypatch):
    monkeypatch.setattr(
        "gradglass.monitor_control._list_listening_processes_with_psutil",
        lambda: [
            (5555, 8432, "python -m gradglass.server --root /tmp/demo --port 8432"),
            (6666, 9000, "python -m http.server 9000"),
        ],
    )

    targets = list_standalone_gradglass_targets()

    assert len(targets) == 1
    assert targets[0].pid == 5555
    assert targets[0].port == 8432
    assert targets[0].source == "process_scan"


def test_cli_stop_requires_a_selector(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["gradglass", "stop"])

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    captured = capsys.readouterr()
    assert exc_info.value.code == 2
    assert "Specify a run_id, --port, or --all." in captured.err


def test_cli_stop_by_run_id_prints_result(monkeypatch, tmp_path, capsys):
    store = ArtifactStore(root=tmp_path)
    monkeypatch.setattr(gg, "store", store)

    recorded: dict[str, object] = {}

    def fake_stop(store_arg, *, run_id=None, port=None, stop_all=False):
        recorded["store"] = store_arg
        recorded["run_id"] = run_id
        recorded["port"] = port
        recorded["stop_all"] = stop_all
        return [MonitorStopResult(status="stopped", message="Stopped GradGlass monitor for run 'run-123' on port 8432 (pid 4321).")]

    monkeypatch.setattr("gradglass.monitor_control.stop_gradglass_monitor", fake_stop)
    monkeypatch.setattr(sys, "argv", ["gradglass", "stop", "run-123"])

    cli.main()

    captured = capsys.readouterr()
    assert recorded == {"store": store, "run_id": "run-123", "port": None, "stop_all": False}
    assert "Stopped GradGlass monitor for run 'run-123' on port 8432 (pid 4321)." in captured.out


def test_run_monitor_persists_monitor_pid_and_started_at(tmp_path, monkeypatch):
    store = ArtifactStore(root=tmp_path)
    run = Run(name="monitor-stop-test", store=store, auto_open=False)

    class FakeProcess:
        pid = 43210

        def poll(self):
            return None

        def terminate(self):
            return None

        def wait(self, timeout=None):
            return 0

        def kill(self):
            return None

    monkeypatch.setattr("gradglass.server.ensure_dashboard_build_available", lambda workspace_root=None: tmp_path)
    monkeypatch.setattr("gradglass.server._wait_for_server", lambda host, port, timeout=10.0: True)
    monkeypatch.setattr("subprocess.Popen", lambda *args, **kwargs: FakeProcess())

    port = run.monitor(port=8432, open_browser=False)

    state = store.get_runtime_state(run.run_id)
    assert port == 8432
    assert state["monitor_enabled"] is True
    assert state["monitor_port"] == 8432
    assert state["monitor_pid"] == 43210
    assert isinstance(state["monitor_started_at"], float)
    assert state["monitor_stopped_at"] is None
    assert state["monitor_stop_reason"] is None


def test_start_server_blocking_fails_cleanly_when_port_is_in_use(tmp_path, monkeypatch, capsys):
    app = create_app(ArtifactStore(root=tmp_path))
    port = 8432

    class FakeSocket:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def setsockopt(self, *args, **kwargs):
            return None

        def settimeout(self, timeout):
            return None

        def connect_ex(self, address):
            return 0

    monkeypatch.setattr("gradglass.server.ensure_dashboard_build_available", lambda workspace_root=None: tmp_path)
    monkeypatch.setattr("gradglass.server.socket.socket", lambda *args, **kwargs: FakeSocket())
    monkeypatch.setattr(
        "gradglass.server.get_port_conflict",
        lambda host, bound_port: (23910, "python -m gradglass.server --root /tmp/demo --port 8432"),
    )

    with pytest.raises(RuntimeError, match=rf"Port {port} is already in use by pid 23910"):
        start_server_blocking(app, port=port, open_browser=False)

    captured = capsys.readouterr()
    assert "GradGlass server running" not in captured.out


def test_start_server_blocking_prints_success_after_server_starts(tmp_path, monkeypatch, capsys):
    app = create_app(ArtifactStore(root=tmp_path))
    browser_calls: list[tuple[str, float, bool]] = []

    class FakeServer:
        def __init__(self, config):
            self.config = config
            self.started = False
            self.should_exit = False

        def run(self):
            time.sleep(0.1)
            self.started = True
            time.sleep(0.1)
            self.should_exit = True

    monkeypatch.setattr("gradglass.server.ensure_dashboard_build_available", lambda workspace_root=None: tmp_path)
    monkeypatch.setattr("gradglass.server.ensure_port_available", lambda host, port: None)
    monkeypatch.setattr("gradglass.server.uvicorn.Server", FakeServer)
    monkeypatch.setattr(
        "gradglass.server.schedule_url_open_detached",
        lambda url, delay_s=0.0, force_reload=False: browser_calls.append((url, delay_s, force_reload)) or threading.Thread(),
    )

    start_server_blocking(app, port=8432, open_browser=True)

    captured = capsys.readouterr()
    assert f"Workspace: {tmp_path}" in captured.out
    assert "GradGlass server running at http://localhost:8432" in captured.out
    assert browser_calls == [("http://localhost:8432", 0.0, True)]


def test_cli_serve_reports_missing_dashboard_build(monkeypatch, tmp_path, capsys):
    store = ArtifactStore(root=tmp_path)
    monkeypatch.setattr(gg, "store", store)
    monkeypatch.setattr(
        "gradglass.server.ensure_dashboard_build_available",
        lambda workspace_root=None: (_ for _ in ()).throw(RuntimeError("Dashboard build missing for tests.")),
    )
    monkeypatch.setattr(sys, "argv", ["gradglass", "serve", "--no-browser"])

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    captured = capsys.readouterr()
    assert exc_info.value.code == 1
    assert "Dashboard build missing for tests." in captured.err


def test_example_output_uses_plain_serve_for_repo_workspace():
    assert serve_command_for_workspace(repo_workspace_root(), port=8432) == "gradglass serve --port 8432"
