from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py
from setuptools.command.sdist import sdist as _sdist

try:
    from distutils.errors import DistutilsSetupError
except ImportError:  # pragma: no cover
    from setuptools.errors import SetupError as DistutilsSetupError


ROOT = Path(__file__).parent.resolve()
DASHBOARD_DIR = ROOT / "gradglass" / "dashboard"
DASHBOARD_DIST_DIR = DASHBOARD_DIR / "dist"
DASHBOARD_ENTRYPOINT = DASHBOARD_DIST_DIR / "index.html"


def ensure_dashboard_bundle() -> None:
    if DASHBOARD_ENTRYPOINT.exists():
        return

    npm = shutil.which("npm")
    if npm and (DASHBOARD_DIR / "package.json").exists() and (DASHBOARD_DIR / "node_modules").exists():
        subprocess.run([npm, "--prefix", str(DASHBOARD_DIR), "run", "build"], check=True)

    if DASHBOARD_ENTRYPOINT.exists():
        return

    raise DistutilsSetupError(
        "GradGlass requires a built dashboard bundle at gradglass/dashboard/dist. "
        "Either build from a checkout that already contains the bundle, or run "
        "`npm --prefix gradglass/dashboard install && npm --prefix gradglass/dashboard run build` "
        "before packaging."
    )


class build_py(_build_py):
    def run(self):
        ensure_dashboard_bundle()
        super().run()


class sdist(_sdist):
    def run(self):
        ensure_dashboard_bundle()
        super().run()


setup(cmdclass={"build_py": build_py, "sdist": sdist})
