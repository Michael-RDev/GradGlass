from __future__ import annotations

import json
from pathlib import Path

from gradglass import __version__
from gradglass.artifacts import ArtifactStore
from gradglass.server import create_app


ROOT = Path(__file__).resolve().parents[1]


def test_release_versions_are_aligned(tmp_path):
    package_json = json.loads((ROOT / "gradglass" / "dashboard" / "package.json").read_text())
    readme = (ROOT / "README.md").read_text()
    pyproject = (ROOT / "pyproject.toml").read_text()

    app = create_app(ArtifactStore(root=tmp_path / "workspace"))

    assert __version__ == "1.1.0"
    assert package_json["version"] == __version__
    assert app.version == __version__
    assert f"version-{__version__}-" in readme
    assert "https://github.com/Michael-RDev/GradGlass" in pyproject
