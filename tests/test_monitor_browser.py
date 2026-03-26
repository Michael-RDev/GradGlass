from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import gradglass.browser as browser
import gradglass.run as run_module
from gradglass.artifacts import ArtifactStore
from gradglass.run import Run


@pytest.fixture
def tmp_store():
    tmpdir = tempfile.mkdtemp()
    store = ArtifactStore(root=tmpdir)
    yield store
    shutil.rmtree(tmpdir, ignore_errors=True)


def _install_fake_capture(monkeypatch):
    class FakeCaptureEngine:
        def __init__(self, model, optimizer, framework, run_dir, config):
            self.run_dir = Path(run_dir)

        def attach_hooks(self):
            return None

        def extract_architecture(self):
            (self.run_dir / "model_structure.json").write_text(json.dumps({"layers": [], "edges": []}))

        def flush_writes(self):
            return None

        def cleanup(self):
            return None

    fake_capture = types.ModuleType("gradglass.capture")
    fake_capture.CaptureEngine = FakeCaptureEngine
    monkeypatch.setitem(sys.modules, "gradglass.capture", fake_capture)


def _install_fake_server(monkeypatch, port: int = 8765):
    fake_server = types.ModuleType("gradglass.server")
    fake_server.create_app = lambda store: object()
    fake_server.start_server = lambda app, port=0: port or 8765
    fake_server.find_free_port = lambda: port
    monkeypatch.setitem(sys.modules, "gradglass.server", fake_server)


def _patch_write_metadata(monkeypatch):
    def fake_write_metadata(self, status="running"):
        payload = {
            "name": self.name,
            "run_id": self.run_id,
            "framework": self.framework,
            "status": status,
            "start_time_epoch": self.start_time,
            "config": self.options,
            "capture_config": self.capture_config,
            "environment": {},
            "git_commit": None,
        }
        with open(self.run_dir / "metadata.json", "w") as f:
            json.dump(payload, f)

    monkeypatch.setattr(Run, "write_metadata", fake_write_metadata)


@pytest.mark.parametrize(
    ("platform_name", "expected_command"),
    [
        ("darwin", ["open", "http://localhost:9999"]),
        ("linux", ["xdg-open", "http://localhost:9999"]),
        ("win32", ["cmd", "/c", "start", "", "http://localhost:9999"]),
        ("freebsd", [sys.executable, "-m", "webbrowser", "http://localhost:9999"]),
    ],
)
def test_open_url_detached_uses_platform_specific_command(monkeypatch, platform_name, expected_command):
    captured = {}

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(browser.sys, "platform", platform_name)
    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    assert browser.open_url_detached("http://localhost:9999") is True
    assert captured["command"] == expected_command
    assert captured["kwargs"]["start_new_session"] is True


def test_open_url_detached_warns_instead_of_raising(monkeypatch):
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("boom")))

    with pytest.warns(RuntimeWarning, match="could not open the browser automatically"):
        assert browser.open_url_detached("http://localhost:9999") is False


def test_resolve_open_browser_preference_honors_env(monkeypatch):
    monkeypatch.setenv("GRADGLASS_OPEN_BROWSER", "0")
    assert browser.resolve_open_browser_preference(None, None) is False

    monkeypatch.setenv("GRADGLASS_OPEN_BROWSER", "1")
    assert browser.resolve_open_browser_preference(None, None) is True


def test_watch_respects_monitor_open_browser_override(tmp_store, monkeypatch):
    _install_fake_capture(monkeypatch)
    _patch_write_metadata(monkeypatch)
    monkeypatch.setattr(Run, "detectframework", lambda self, model: "pytorch")

    run = Run(name="monitor-opt-out", store=tmp_store, monitor=True, port=9876)
    called = {}

    def fake_monitor(port=0, open_browser=True):
        called["port"] = port
        called["open_browser"] = open_browser
        return 9876

    monkeypatch.setattr(run, "monitor", fake_monitor)

    run.watch(object(), optimizer=None, monitor_open_browser=False)

    assert called["port"] == 9876
    assert called["open_browser"] is False
    run.finish(open=False, analyze=False)


def test_watch_uses_run_option_when_monitor_open_browser_not_passed(tmp_store, monkeypatch):
    _install_fake_capture(monkeypatch)
    _patch_write_metadata(monkeypatch)
    monkeypatch.setattr(Run, "detectframework", lambda self, model: "pytorch")

    run = Run(name="monitor-opt-run-config", store=tmp_store, monitor=True, port=4321, monitor_open_browser=False)
    called = {}

    def fake_monitor(port=0, open_browser=True):
        called["port"] = port
        called["open_browser"] = open_browser
        return 4321

    monkeypatch.setattr(run, "monitor", fake_monitor)

    run.watch(object(), optimizer=None)

    assert called["port"] == 4321
    assert called["open_browser"] is False
    run.finish(open=False, analyze=False)


def test_monitor_uses_detached_browser_launcher(tmp_store, monkeypatch):
    _install_fake_server(monkeypatch, port=8123)
    _patch_write_metadata(monkeypatch)

    called = {}

    def fake_open(url):
        called["url"] = url
        return True

    monkeypatch.setattr(run_module, "open_url_detached", fake_open)

    run = Run(name="live-monitor", store=tmp_store)
    port = run.monitor(port=8123, open_browser=True)

    assert port == 8123
    assert called["url"] == f"http://localhost:{port}/?run={run.run_id}"
