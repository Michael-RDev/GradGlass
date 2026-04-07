from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

from gradglass.artifacts import DEFAULT_WORKSPACE_DIRNAME, resolve_default_root


ROOT = Path(__file__).resolve().parents[1]


def _pythonpath_with_repo() -> str:
    existing = os.environ.get("PYTHONPATH")
    if existing:
        return f"{ROOT}{os.pathsep}{existing}"
    return str(ROOT)


def test_default_workspace_tracks_launched_script_directory(tmp_path):
    script_dir = tmp_path / "examples"
    script_dir.mkdir()
    script_path = script_dir / "train.py"
    script_path.write_text("from gradglass import gg\nprint(gg.store.root.resolve())\n", encoding="utf-8")

    launch_dir = tmp_path / "launch_here"
    launch_dir.mkdir()
    env = os.environ.copy()
    env["PYTHONPATH"] = _pythonpath_with_repo()

    result = subprocess.run(
        [sys.executable, str(script_path)], cwd=launch_dir, env=env, capture_output=True, text=True, check=True
    )

    assert result.stdout.strip() == str((script_dir / DEFAULT_WORKSPACE_DIRNAME).resolve())


def test_internal_gradglass_entrypoints_fall_back_to_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    internal_entrypoint = ROOT / "gradglass" / "server.py"

    assert resolve_default_root(entrypoint=internal_entrypoint) == tmp_path / DEFAULT_WORKSPACE_DIRNAME


def test_cli_launchers_fall_back_to_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    launcher_path = tmp_path / "bin" / "gradglass"

    assert resolve_default_root(entrypoint=launcher_path) == tmp_path / DEFAULT_WORKSPACE_DIRNAME
