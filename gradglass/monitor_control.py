from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_UNSET = object()


@dataclass
class MonitorTarget:
    run_id: Optional[str]
    runtime_state_path: Optional[Path]
    port: Optional[int]
    pid: Optional[int]
    source: str
    metadata: dict = field(default_factory=dict)


@dataclass
class MonitorStopResult:
    status: str
    message: str
    target: Optional[MonitorTarget] = None
    stopped: bool = False
    stale: bool = False


def _read_json(path: Path) -> dict:
    try:
        with open(path) as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    try:
        with open(tmp_path, "w") as handle:
            json.dump(payload, handle, indent=2, default=str)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass


def _safe_int(value) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _runtime_state_path(store, run_id: str) -> Path:
    return store.get_run_dir(run_id) / "runtime_state.json"


def _iter_runtime_state_targets(store) -> list[MonitorTarget]:
    runs_dir = store.root / "runs"
    if not runs_dir.exists():
        return []
    targets: list[MonitorTarget] = []
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        state_path = run_dir / "runtime_state.json"
        if not state_path.exists():
            continue
        state = _read_json(state_path)
        monitor_enabled = bool(state.get("monitor_enabled"))
        port = _safe_int(state.get("monitor_port"))
        pid = _safe_int(state.get("monitor_pid"))
        if not monitor_enabled and port is None and pid is None:
            continue
        targets.append(
            MonitorTarget(
                run_id=run_dir.name,
                runtime_state_path=state_path,
                port=port,
                pid=pid,
                source="runtime_state",
                metadata=state,
            )
        )
    return targets


def list_run_monitor_targets(store) -> list[MonitorTarget]:
    return _iter_runtime_state_targets(store)


def find_run_monitor_target(store, run_id: str) -> Optional[MonitorTarget]:
    state_path = _runtime_state_path(store, run_id)
    if not state_path.exists():
        return None
    state = _read_json(state_path)
    port = _safe_int(state.get("monitor_port"))
    pid = _safe_int(state.get("monitor_pid"))
    if not bool(state.get("monitor_enabled")) and port is None and pid is None:
        return None
    return MonitorTarget(
        run_id=run_id,
        runtime_state_path=state_path,
        port=port,
        pid=pid,
        source="runtime_state",
        metadata=state,
    )


def _update_runtime_state(
    target: MonitorTarget,
    *,
    monitor_enabled=_UNSET,
    monitor_pid=_UNSET,
    monitor_port=_UNSET,
    monitor_stopped_at=_UNSET,
    monitor_stop_reason=_UNSET,
) -> None:
    if target.runtime_state_path is None:
        return
    payload = _read_json(target.runtime_state_path)
    if monitor_enabled is not _UNSET:
        payload["monitor_enabled"] = bool(monitor_enabled)
    if monitor_pid is not _UNSET:
        payload["monitor_pid"] = monitor_pid
    if monitor_port is not _UNSET:
        payload["monitor_port"] = monitor_port
    if monitor_stopped_at is not _UNSET:
        payload["monitor_stopped_at"] = None if monitor_stopped_at is None else float(monitor_stopped_at)
    if monitor_stop_reason is not _UNSET:
        payload["monitor_stop_reason"] = None if monitor_stop_reason is None else str(monitor_stop_reason)
    _atomic_write_json(target.runtime_state_path, payload)


def _is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _wait_for_pid_exit(pid: int, timeout_s: float = 2.0, interval_s: float = 0.05) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not _is_pid_alive(pid):
            return True
        time.sleep(interval_s)
    return not _is_pid_alive(pid)


def _terminate_pid(pid: int) -> bool:
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    if _wait_for_pid_exit(pid, timeout_s=2.0):
        return True
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return _wait_for_pid_exit(pid, timeout_s=1.0)


def _looks_like_gradglass_command(command: str) -> bool:
    lowered = (command or "").lower()
    return "gradglass.server" in lowered or "gradglass serve" in lowered or "gradglass monitor" in lowered


def _find_process_by_port_with_psutil(port: int) -> tuple[Optional[int], Optional[str]]:
    try:
        import psutil  # type: ignore
    except ImportError:
        return None, None

    try:
        connections = psutil.net_connections(kind="tcp")
    except Exception:
        return None, None

    for conn in connections:
        if conn.status != getattr(psutil, "CONN_LISTEN", "LISTEN"):
            continue
        if not conn.laddr:
            continue
        if getattr(conn.laddr, "port", None) != port:
            continue
        pid = conn.pid
        if pid is None:
            continue
        try:
            process = psutil.Process(pid)
            command = " ".join(process.cmdline())
        except Exception:
            command = ""
        return pid, command
    return None, None


def _find_process_by_port_with_lsof(port: int) -> tuple[Optional[int], Optional[str]]:
    try:
        result = subprocess.run(
            ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-Fp"],
            capture_output=True,
            text=True,
            timeout=3,
        )
    except Exception:
        return None, None
    if result.returncode != 0:
        return None, None
    pid = None
    for line in result.stdout.splitlines():
        if line.startswith("p"):
            pid = _safe_int(line[1:])
            break
    if pid is None:
        return None, None
    try:
        ps_result = subprocess.run(
            ["ps", "-o", "command=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=3,
        )
        command = ps_result.stdout.strip() if ps_result.returncode == 0 else ""
    except Exception:
        command = ""
    return pid, command


def find_process_by_port(port: int) -> tuple[Optional[int], Optional[str]]:
    pid, command = _find_process_by_port_with_psutil(port)
    if pid is not None:
        return pid, command
    return _find_process_by_port_with_lsof(port)


def _dedupe_targets(targets: list[MonitorTarget]) -> list[MonitorTarget]:
    deduped: list[MonitorTarget] = []
    seen: set[tuple[Optional[int], Optional[int], Optional[str]]] = set()
    for target in targets:
        key = (target.pid, target.port, None)
        if key == (None, None, None):
            key = (None, None, target.run_id)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(target)
    return deduped


def _target_label(target: MonitorTarget) -> str:
    if target.run_id:
        return f"run '{target.run_id}'"
    if target.port is not None:
        return f"port {target.port}"
    return "monitor target"


def _stop_success_message(target: MonitorTarget, pid: int, port: int | None) -> str:
    if target.run_id:
        return f"Stopped GradGlass monitor for run '{target.run_id}' on port {port} (pid {pid})."
    if port is not None:
        return f"Stopped GradGlass monitor on port {port} (pid {pid})."
    return f"Stopped GradGlass monitor (pid {pid})."


def _stop_target(target: MonitorTarget, *, stop_reason: str = "cli_stop") -> MonitorStopResult:
    pid = target.pid
    port = target.port
    timestamp = time.time()
    label = _target_label(target)

    if pid is not None:
        if not _is_pid_alive(pid):
            _update_runtime_state(
                target,
                monitor_enabled=False,
                monitor_pid=None,
                monitor_stopped_at=timestamp,
                monitor_stop_reason="stale_pid",
            )
            return MonitorStopResult(
                status="stale",
                message=f"Monitor for {label} is already dead (stale pid {pid}).",
                target=target,
                stale=True,
            )
        if _terminate_pid(pid):
            _update_runtime_state(
                target,
                monitor_enabled=False,
                monitor_pid=None,
                monitor_stopped_at=timestamp,
                monitor_stop_reason=stop_reason,
            )
            return MonitorStopResult(
                status="stopped",
                message=_stop_success_message(target, pid, port),
                target=target,
                stopped=True,
            )
        return MonitorStopResult(
            status="error",
            message=f"Failed to stop GradGlass monitor for {label} (pid {pid}).",
            target=target,
        )

    if port is not None:
        found_pid, command = find_process_by_port(port)
        if found_pid is None:
            _update_runtime_state(
                target,
                monitor_enabled=False,
                monitor_pid=None,
                monitor_stopped_at=timestamp,
                monitor_stop_reason="stale_port",
            )
            return MonitorStopResult(
                status="stale",
                message=f"Monitor for {label} is already gone on port {port}.",
                target=target,
                stale=True,
            )
        if not _looks_like_gradglass_command(command or ""):
            return MonitorStopResult(
                status="refused",
                message=f"Port {port} is not owned by a verified GradGlass server.",
                target=target,
            )
        target.pid = found_pid
        if _terminate_pid(found_pid):
            _update_runtime_state(
                target,
                monitor_enabled=False,
                monitor_pid=None,
                monitor_stopped_at=timestamp,
                monitor_stop_reason=stop_reason,
            )
            return MonitorStopResult(
                status="stopped",
                message=_stop_success_message(target, found_pid, port),
                target=target,
                stopped=True,
            )
        return MonitorStopResult(
            status="error",
            message=f"Failed to stop GradGlass monitor on port {port} (pid {found_pid}).",
            target=target,
        )

    return MonitorStopResult(
        status="not_found",
        message=f"No monitor target found for run '{target.run_id}'.",
        target=target,
    )


def stop_monitor_targets(targets: list[MonitorTarget], *, allow_multiple: bool = False, stop_reason: str = "cli_stop") -> list[MonitorStopResult]:
    deduped = _dedupe_targets(targets)
    if not deduped:
        return [MonitorStopResult(status="not_found", message="No GradGlass monitor targets found.")]
    if len(deduped) > 1 and not allow_multiple:
        return [
            MonitorStopResult(
                status="refused",
                message="Multiple GradGlass monitor targets matched. Re-run with --all or provide a specific run_id/--port.",
            )
        ]
    return [_stop_target(target, stop_reason=stop_reason) for target in deduped]


def stop_gradglass_monitor(store, *, run_id: str | None = None, port: int | None = None, stop_all: bool = False) -> list[MonitorStopResult]:
    if stop_all:
        return stop_monitor_targets(list_run_monitor_targets(store), allow_multiple=True)

    if port is not None:
        matching_targets = [target for target in list_run_monitor_targets(store) if target.port == int(port)]
        if matching_targets:
            return stop_monitor_targets(matching_targets, allow_multiple=False)
        found_pid, command = find_process_by_port(int(port))
        if found_pid is None:
            return [MonitorStopResult(status="not_found", message=f"No listening server found on port {port}.")]
        if not _looks_like_gradglass_command(command or ""):
            return [MonitorStopResult(status="refused", message=f"Port {port} is not owned by a verified GradGlass server.")]
        target = MonitorTarget(run_id=None, runtime_state_path=None, port=int(port), pid=found_pid, source="explicit_port")
        return stop_monitor_targets([target], allow_multiple=False)

    if run_id is not None:
        target = find_run_monitor_target(store, run_id)
        if target is None:
            return [MonitorStopResult(status="not_found", message=f"No run monitor metadata found for run '{run_id}'.")]
        return stop_monitor_targets([target], allow_multiple=False)

    return [
        MonitorStopResult(
            status="usage_error",
            message="Specify a run_id, --port, or --all to stop a GradGlass monitor.",
        )
    ]
