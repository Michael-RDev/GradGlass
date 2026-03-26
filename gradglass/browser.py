from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
import warnings
from typing import Any, Optional


def _coerce_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return None


def resolve_open_browser_preference(
    explicit: Optional[bool],
    option_value: Any = None,
    *,
    env_var: str = "GRADGLASS_OPEN_BROWSER",
    default: bool = True,
) -> bool:
    explicit_value = _coerce_bool(explicit)
    if explicit_value is not None:
        return explicit_value

    option_bool = _coerce_bool(option_value)
    if option_bool is not None:
        return option_bool

    env_bool = _coerce_bool(os.environ.get(env_var))
    if env_bool is not None:
        return env_bool

    return default


def _browser_command(url: str) -> list[str]:
    if sys.platform == "darwin":
        return ["open", url]
    if sys.platform.startswith("linux"):
        return ["xdg-open", url]
    if sys.platform.startswith("win"):
        return ["cmd", "/c", "start", "", url]
    return [sys.executable, "-m", "webbrowser", url]


def open_url_detached(url: str) -> bool:
    command = _browser_command(url)
    kwargs: dict[str, Any] = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "start_new_session": True,
    }
    if sys.platform.startswith("win"):
        kwargs["creationflags"] = (
            getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(subprocess, "DETACHED_PROCESS", 0)
        )

    try:
        subprocess.Popen(command, **kwargs)
        return True
    except Exception as exc:
        warnings.warn(f"GradGlass could not open the browser automatically for {url}: {exc}", RuntimeWarning)
        return False


def schedule_url_open_detached(url: str, *, delay_s: float = 0.0) -> threading.Thread:
    def _runner():
        if delay_s > 0:
            time.sleep(delay_s)
        open_url_detached(url)

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    return thread
