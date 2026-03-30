from __future__ import annotations

import os
import socket
import time
from pathlib import Path
from typing import Any, Optional

from gradglass.artifacts import ArtifactStore
from gradglass.experiment_tracking import build_overview_snapshot

REQUIRED_METRIC_KEYS = {"status", "source", "function_name", "command", "timestamp", "label", "value", "error"}

RUNTIME_HEARTBEAT_STALE_SECONDS = 120.0
RUNTIME_FILE_STALE_SECONDS = 300.0
DISTRIBUTED_FILE_STALE_SECONDS = 300.0
RANK_ARTIFACT_STALE_SECONDS = 300.0

SEMANTIC_STATUS_LABELS = {
    "active": "Active",
    "not_detected": "Not detected",
    "not_supported": "Not supported on this platform",
    "disabled_local_mode": "Disabled in local mode",
    "requires_cluster_connection": "Requires cluster connection",
    "dependency_missing": "Dependency missing",
    "interrupted_training_stopped": "Interrupted because training stopped",
    "error": "Error",
}

_COUNTER_CACHE: dict[str, dict[str, float]] = {}
_PROCESS_CPU_SAMPLE_CACHE: dict[int, dict[str, Any]] = {}


def _metric(
    *,
    status: str,
    source: str,
    function_name: str,
    command: str,
    label: str,
    value: Any,
    error: Optional[str] = None,
    timestamp: Optional[float] = None,
):
    return {
        "status": status,
        "source": source,
        "function_name": function_name,
        "command": command,
        "timestamp": float(timestamp if timestamp is not None else time.time()),
        "label": label,
        "value": value,
        "error": error,
    }


def _normalize_metric(metric: Any, *, function_name: str, command: str) -> dict[str, Any]:
    base = metric if isinstance(metric, dict) else {}
    normalized = dict(base)
    normalized.setdefault("status", "unavailable")
    normalized.setdefault("source", "local_host")
    normalized.setdefault("function_name", function_name)
    normalized.setdefault("command", command)
    raw_timestamp = normalized.get("timestamp", time.time())
    try:
        normalized["timestamp"] = float(time.time() if raw_timestamp is None else raw_timestamp)
    except (TypeError, ValueError):
        normalized["timestamp"] = float(time.time())
    normalized.setdefault("label", "Unavailable")
    normalized.setdefault("value", None)
    normalized.setdefault("error", None)
    return normalized


def _unavailable(*, function_name: str, command: str, reason: str):
    return _metric(
        status="unavailable",
        source="local_host",
        function_name=function_name,
        command=command,
        label="Unavailable",
        value=None,
        error=reason,
    )


def _decode_nvml_text(raw: Any):
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="ignore")
    if raw is None:
        return "Unknown accelerator"
    return str(raw)


def _safe_float(raw: Any) -> Optional[float]:
    try:
        if raw is None:
            return None
        value = float(raw)
        if value != value:
            return None
        return value
    except (TypeError, ValueError):
        return None


def _safe_positive_int(raw: Any) -> Optional[int]:
    try:
        if raw is None:
            return None
        value = int(raw)
        return value if value > 0 else None
    except (TypeError, ValueError):
        return None


def _safe_non_negative_float(raw: Any) -> Optional[float]:
    value = _safe_float(raw)
    if value is None:
        return None
    return max(0.0, value)


def _drop_process_cpu_cache(pid: Optional[int]) -> None:
    if pid is None:
        return
    _PROCESS_CPU_SAMPLE_CACHE.pop(pid, None)


def _sample_training_process_cpu_percent(
    *, pid: int, process: Any, process_start_time: Optional[float]
) -> tuple[Optional[float], bool]:
    cached = _PROCESS_CPU_SAMPLE_CACHE.get(pid)
    if cached is not None:
        cached_start = _safe_float(cached.get("start_time"))
        if cached_start is not None and process_start_time is not None and abs(cached_start - process_start_time) > 2.0:
            cached = None
            _drop_process_cpu_cache(pid)

    if cached is None:
        cached = {"process": process, "primed": False, "start_time": process_start_time}
        _PROCESS_CPU_SAMPLE_CACHE[pid] = cached

    sampler = cached.get("process") or process
    if not cached.get("primed"):
        sampler.cpu_percent(interval=None)
        cached["process"] = sampler
        cached["primed"] = True
        cached["start_time"] = process_start_time
        return None, True

    value = float(sampler.cpu_percent(interval=None))
    cached["process"] = sampler
    cached["start_time"] = process_start_time
    return value, False


def _file_mtime(path: Path) -> Optional[float]:
    try:
        return float(path.stat().st_mtime)
    except (FileNotFoundError, OSError, ValueError):
        return None


def _is_fresh_timestamp(ts: Optional[float], now: float, max_age_s: float) -> bool:
    if ts is None:
        return False
    age = now - ts
    return 0 <= age <= max_age_s


def _latest_rank_artifact_mtime(run_dir: Path, ranks: list[str]) -> Optional[float]:
    latest: Optional[float] = None
    for rank in ranks:
        rank_path = run_dir / rank
        rank_mtime = _file_mtime(rank_path)
        if rank_mtime is None:
            continue
        if latest is None or rank_mtime > latest:
            latest = rank_mtime
    return latest


def _resolve_distributed_world_size(dist_info: Any) -> int:
    if not isinstance(dist_info, dict):
        return 0
    for key in ("world_size", "total_nodes", "num_nodes"):
        value = _safe_positive_int(dist_info.get(key))
        if value is not None:
            return value
    return 0


def _build_live_guard(store: ArtifactStore, run_id: str, collected_at: float):
    run_dir = store.get_run_dir(run_id)
    runtime_state = store.get_runtime_state(run_id) or {}
    dist_info = store.get_distributed_info(run_id)
    ranks = store.list_ranks(run_id)

    runtime_state_path = run_dir / "runtime_state.json"
    distributed_path = run_dir / "distributed_index.json"
    runtime_state_mtime = _file_mtime(runtime_state_path)
    distributed_mtime = _file_mtime(distributed_path)
    rank_mtime = _latest_rank_artifact_mtime(run_dir, ranks)

    heartbeat_ts = _safe_float(runtime_state.get("heartbeat_ts")) if isinstance(runtime_state, dict) else None
    heartbeat_fresh = _is_fresh_timestamp(heartbeat_ts, collected_at, RUNTIME_HEARTBEAT_STALE_SECONDS)
    runtime_file_fresh = _is_fresh_timestamp(runtime_state_mtime, collected_at, RUNTIME_FILE_STALE_SECONDS)
    runtime_fresh = heartbeat_fresh or runtime_file_fresh

    reasons: list[str] = []
    if not isinstance(runtime_state, dict) or not runtime_state:
        reasons.append("runtime_state_unavailable")
    elif not runtime_fresh:
        reasons.append("stale_runtime_state")

    world_size = _resolve_distributed_world_size(dist_info)
    distributed_candidate = world_size > 1 or len(ranks) > 1
    distributed_reasons: list[str] = []
    if distributed_candidate:
        if not isinstance(dist_info, dict) or not dist_info:
            distributed_reasons.append("distributed_index_missing")
        if not _is_fresh_timestamp(distributed_mtime, collected_at, DISTRIBUTED_FILE_STALE_SECONDS):
            distributed_reasons.append("stale_distributed_artifacts_ignored")
        if world_size > 1 and len(ranks) < min(world_size, 2):
            distributed_reasons.append("missing_rank_artifacts")
        if len(ranks) > 0 and not _is_fresh_timestamp(rank_mtime, collected_at, RANK_ARTIFACT_STALE_SECONDS):
            distributed_reasons.append("stale_rank_artifacts")
        if not runtime_fresh:
            distributed_reasons.append("runtime_not_fresh_for_distributed")

    reasons.extend(distributed_reasons)
    distributed_verified = distributed_candidate and len(distributed_reasons) == 0

    live_guard = {
        "ok": len(reasons) == 0,
        "reasons": reasons,
        "collected_at": float(collected_at),
        "server_pid": int(os.getpid()),
        "hostname": socket.gethostname(),
        "run_id": run_id,
    }
    return live_guard, distributed_verified, distributed_candidate


def _force_cluster_standalone(metric: dict[str, Any], reasons: list[str]):
    guarded = dict(metric)
    old_value = metric.get("value")
    suppressed_total = None
    if isinstance(old_value, dict):
        suppressed_total = _safe_positive_int(old_value.get("total_nodes"))

    guarded["status"] = "estimated"
    guarded["label"] = "1 / 1 (guarded)"
    guarded["value"] = {
        "active_nodes": 1,
        "total_nodes": 1,
        "mode": "standalone",
        "guarded": True,
        "suppressed_total_nodes": suppressed_total,
    }
    reason_text = ", ".join(reasons) if reasons else "live_guard_active"
    guarded["error"] = f"Distributed claim suppressed by live guard: {reason_text}"
    guarded["source"] = metric.get("source", "run_artifacts")
    guarded["function_name"] = metric.get("function_name", "query_cluster_nodes")
    guarded["command"] = metric.get("command", "cluster guard")
    guarded["timestamp"] = _safe_float(metric.get("timestamp")) or time.time()
    return _normalize_metric(guarded, function_name=str(guarded["function_name"]), command=str(guarded["command"]))


def _import_psutil():
    try:
        import psutil  # type: ignore
    except ImportError:
        return None
    return psutil


def _get_nvml_module():
    try:
        import pynvml as nvml  # type: ignore
    except ImportError:
        return None, "pynvml is not installed"

    try:
        nvml.nvmlInit()
    except Exception as exc:  # pragma: no cover - depends on host NVML/runtime
        return None, f"NVML initialization failed: {exc}"
    return nvml, None


def _shutdown_nvml(nvml):
    try:
        nvml.nvmlShutdown()
    except Exception:
        pass


def query_cluster_nodes(store: ArtifactStore, run_id: str):
    function_name = "query_cluster_nodes"
    command = "ArtifactStore.get_distributed_info + ArtifactStore.list_ranks"

    dist_info = store.get_distributed_info(run_id) or {}
    ranks = store.list_ranks(run_id)

    total_nodes = 1
    active_nodes = 1

    if isinstance(dist_info, dict):
        for key in ("total_nodes", "world_size", "num_nodes"):
            candidate = dist_info.get(key)
            if isinstance(candidate, int) and candidate > 0:
                total_nodes = candidate
                break
        for key in ("active_nodes", "healthy_nodes"):
            candidate = dist_info.get(key)
            if isinstance(candidate, int) and candidate >= 0:
                active_nodes = candidate
                break

    if ranks:
        total_nodes = max(total_nodes, len(ranks))
        active_nodes = max(active_nodes, min(len(ranks), total_nodes))

    total_nodes = max(int(total_nodes), 1)
    active_nodes = max(0, min(int(active_nodes), total_nodes))
    mode = "distributed" if total_nodes > 1 else "standalone"

    return _metric(
        status="live",
        source="run_artifacts",
        function_name=function_name,
        command=command,
        label=f"{active_nodes} / {total_nodes}",
        value={
            "active_nodes": active_nodes,
            "total_nodes": total_nodes,
            "mode": mode,
            "ranks": ranks,
            "distributed_index_present": bool(dist_info),
        },
        error=None,
    )


def query_system_cpu():
    function_name = "query_system_cpu"
    command = "psutil.cpu_percent + psutil.Process.cpu_percent"
    psutil = _import_psutil()
    if psutil is None:
        return _unavailable(function_name=function_name, command=command, reason="psutil is not installed")

    host_percent = float(psutil.cpu_percent(interval=None))
    process_percent = float(psutil.Process().cpu_percent(interval=None))
    logical_cores = int(psutil.cpu_count(logical=True) or 0)
    physical_raw = psutil.cpu_count(logical=False)
    physical_cores = int(physical_raw if physical_raw is not None else logical_cores)

    return _metric(
        status="live",
        source="local_host",
        function_name=function_name,
        command=command,
        label=f"{host_percent:.1f}% host (proc {process_percent:.1f}%)",
        value={
            "host_percent": host_percent,
            "process_percent": process_percent,
            "logical_cores": logical_cores,
            "physical_cores": physical_cores,
        },
        error=None,
    )


def query_system_ram():
    function_name = "query_system_ram"
    command = "psutil.virtual_memory"
    psutil = _import_psutil()
    if psutil is None:
        return _unavailable(function_name=function_name, command=command, reason="psutil is not installed")

    vm = psutil.virtual_memory()
    used_bytes = int(vm.used)
    total_bytes = int(vm.total)
    used_gb = used_bytes / float(1024**3)
    total_gb = total_bytes / float(1024**3)

    return _metric(
        status="live",
        source="local_host",
        function_name=function_name,
        command=command,
        label=f"{used_gb:.0f} / {total_gb:.0f} GB",
        value={"used_bytes": used_bytes, "total_bytes": total_bytes, "used_percent": float(vm.percent)},
        error=None,
    )


def query_power_draw():
    function_name = "query_power_draw"
    command = "pynvml.nvmlDeviceGetPowerUsage"
    nvml, reason = _get_nvml_module()
    if nvml is None:
        return _unavailable(function_name=function_name, command=command, reason=reason or "NVML unavailable")

    try:
        gpu_count = int(nvml.nvmlDeviceGetCount())
        if gpu_count < 1:
            return _unavailable(function_name=function_name, command=command, reason="No NVIDIA GPUs detected")

        per_gpu = []
        total_watts = 0.0
        for index in range(gpu_count):
            handle = nvml.nvmlDeviceGetHandleByIndex(index)
            name = _decode_nvml_text(nvml.nvmlDeviceGetName(handle))
            watts = float(nvml.nvmlDeviceGetPowerUsage(handle)) / 1000.0
            total_watts += watts
            per_gpu.append({"index": index, "name": name, "power_watts": round(watts, 2)})

        return _metric(
            status="live",
            source="nvml",
            function_name=function_name,
            command=command,
            label=f"{total_watts:.0f} W",
            value={"total_watts": round(total_watts, 2), "per_gpu": per_gpu},
            error=None,
        )
    finally:
        _shutdown_nvml(nvml)


def query_multi_gpu_compute_utilization():
    function_name = "query_multi_gpu_compute_utilization"
    command = "pynvml.nvmlDeviceGetUtilizationRates"
    nvml, reason = _get_nvml_module()
    if nvml is None:
        return _unavailable(function_name=function_name, command=command, reason=reason or "NVML unavailable")

    try:
        gpu_count = int(nvml.nvmlDeviceGetCount())
        if gpu_count < 1:
            return _unavailable(function_name=function_name, command=command, reason="No NVIDIA GPUs detected")

        per_gpu = []
        for index in range(gpu_count):
            handle = nvml.nvmlDeviceGetHandleByIndex(index)
            name = _decode_nvml_text(nvml.nvmlDeviceGetName(handle))
            util = nvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_utilization = float(getattr(util, "gpu", 0.0))
            per_gpu.append({"index": index, "name": name, "utilization_percent": gpu_utilization})

        avg_util = sum(g["utilization_percent"] for g in per_gpu) / float(gpu_count)
        label = f"{avg_util:.1f}% avg across {gpu_count} GPU{'s' if gpu_count != 1 else ''}"

        return _metric(
            status="live",
            source="nvml",
            function_name=function_name,
            command=command,
            label=label,
            value={"gpu_count": gpu_count, "average_percent": round(avg_util, 2), "per_gpu": per_gpu},
            error=None,
        )
    finally:
        _shutdown_nvml(nvml)


def query_gpu_memory_fragmentation() -> dict[str, Any]:
    function_name = "query_gpu_memory_fragmentation"
    command = "pynvml.nvmlDeviceGetMemoryInfo"
    nvml, reason = _get_nvml_module()
    if nvml is None:
        return _unavailable(function_name=function_name, command=command, reason=reason or "NVML unavailable")

    try:
        gpu_count = int(nvml.nvmlDeviceGetCount())
        if gpu_count < 1:
            return _unavailable(function_name=function_name, command=command, reason="No NVIDIA GPUs detected")

        per_gpu = []
        for index in range(gpu_count):
            handle = nvml.nvmlDeviceGetHandleByIndex(index)
            name = _decode_nvml_text(nvml.nvmlDeviceGetName(handle))
            memory_info = nvml.nvmlDeviceGetMemoryInfo(handle)
            used_bytes = int(memory_info.used)
            total_bytes = int(memory_info.total)
            free_bytes = int(memory_info.free)
            if total_bytes > 0:
                fragmentation_percent = (used_bytes / float(total_bytes)) * 100.0
            else:
                fragmentation_percent = 0.0
            per_gpu.append(
                {
                    "index": index,
                    "name": name,
                    "fragmentation_percent": round(fragmentation_percent, 2),
                    "used_bytes": used_bytes,
                    "free_bytes": free_bytes,
                    "total_bytes": total_bytes,
                }
            )

        avg_fragmentation = sum(g["fragmentation_percent"] for g in per_gpu) / float(gpu_count)
        return _metric(
            status="estimated",
            source="nvml",
            function_name=function_name,
            command=command,
            label=f"Estimated from memory pressure ({avg_fragmentation:.1f}% avg)",
            value={
                "gpu_count": gpu_count,
                "average_percent": round(avg_fragmentation, 2),
                "heuristic": "used_memory_ratio",
                "per_gpu": per_gpu,
            },
            error=(
                "Allocator-level fragmentation counters are not exposed by NVML; showing estimated memory pressure."
            ),
        )
    finally:
        _shutdown_nvml(nvml)


def _collect_gpu_devices(metrics: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    devices: dict[int, dict[str, Any]] = {}
    for metric_key in ("multi_gpu_compute_utilization", "gpu_memory_fragmentation", "power_draw"):
        per_gpu = metrics.get(metric_key, {}).get("value", {})
        entries = per_gpu.get("per_gpu") if isinstance(per_gpu, dict) else None
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            index = entry.get("index")
            if not isinstance(index, int):
                continue
            existing = devices.get(index, {"index": index})
            name = entry.get("name")
            if isinstance(name, str) and name:
                existing["name"] = name
            devices[index] = existing
    return [devices[idx] for idx in sorted(devices.keys())]


def _semantic_metric(
    *,
    key: str,
    name: str,
    value: Any,
    status: str,
    source: str,
    probe: str,
    unit: str = "",
    display: Optional[str] = None,
    error: Optional[str] = None,
    details: Optional[dict[str, Any]] = None,
    timestamp: Optional[float] = None,
) -> dict[str, Any]:
    safe_status = status if status in SEMANTIC_STATUS_LABELS else "error"
    ts = float(timestamp if timestamp is not None else time.time())
    if display is None:
        if value is None:
            display = "—"
        elif isinstance(value, float):
            display = f"{value:.2f}{unit}"
        else:
            display = f"{value}{unit}"
    return {
        "key": key,
        "name": name,
        "value": value,
        "unit": unit,
        "display": display,
        "status": safe_status,
        "status_label": SEMANTIC_STATUS_LABELS.get(safe_status, SEMANTIC_STATUS_LABELS["error"]),
        "source": source,
        "probe": probe,
        "timestamp": ts,
        "error": error,
        "details": details or {},
    }


def _detect_mps_active() -> tuple[bool, Optional[str]]:
    import importlib.util
    import sys

    # Avoid importing torch inside telemetry probes because binary import
    # failures can abort the process on some local setups.
    torch = sys.modules.get("torch")
    if torch is None:
        if importlib.util.find_spec("torch") is None:
            return False, "torch is not installed"
        return False, "torch is installed but not loaded; skipping direct MPS probe for process safety"

    try:
        backend = getattr(getattr(torch, "backends", None), "mps", None)
        if backend is None:
            return False, None
        if bool(getattr(backend, "is_available", lambda: False)()) and bool(
            getattr(backend, "is_built", lambda: True)()
        ):
            return True, None
        return False, None
    except Exception as exc:
        return False, str(exc)


def _safe_nvml_call(func, *args):
    try:
        return func(*args), None
    except Exception as exc:  # pragma: no cover - host-dependent runtime
        return None, str(exc)


def _collect_cuda_accelerators(collected_at: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accelerators: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    nvml, reason = _get_nvml_module()
    if nvml is None:
        diagnostics.append(
            {
                "scope": "cuda",
                "status": "dependency_missing",
                "message": reason or "NVML unavailable",
                "source": "pynvml",
                "probe": "pynvml.nvmlInit",
            }
        )
        return accelerators, diagnostics

    try:
        gpu_count = int(nvml.nvmlDeviceGetCount())
        if gpu_count < 1:
            diagnostics.append(
                {
                    "scope": "cuda",
                    "status": "not_detected",
                    "message": "No NVIDIA accelerators detected",
                    "source": "nvml",
                    "probe": "nvmlDeviceGetCount",
                }
            )
            return accelerators, diagnostics

        for index in range(gpu_count):
            handle = nvml.nvmlDeviceGetHandleByIndex(index)
            name = _decode_nvml_text(nvml.nvmlDeviceGetName(handle))
            util, util_err = _safe_nvml_call(nvml.nvmlDeviceGetUtilizationRates, handle)
            mem_info, mem_err = _safe_nvml_call(nvml.nvmlDeviceGetMemoryInfo, handle)
            power, power_err = _safe_nvml_call(nvml.nvmlDeviceGetPowerUsage, handle)
            temp, temp_err = _safe_nvml_call(
                nvml.nvmlDeviceGetTemperature, handle, getattr(nvml, "NVML_TEMPERATURE_GPU", 0)
            )
            fan, fan_err = _safe_nvml_call(nvml.nvmlDeviceGetFanSpeed, handle)

            rx_bytes = None
            tx_bytes = None
            rx_err = None
            tx_err = None
            rx_tag = getattr(nvml, "NVML_PCIE_UTIL_RX_BYTES", None)
            tx_tag = getattr(nvml, "NVML_PCIE_UTIL_TX_BYTES", None)
            if rx_tag is not None and tx_tag is not None:
                rx_bytes, rx_err = _safe_nvml_call(nvml.nvmlDeviceGetPcieThroughput, handle, rx_tag)
                tx_bytes, tx_err = _safe_nvml_call(nvml.nvmlDeviceGetPcieThroughput, handle, tx_tag)
            else:
                rx_err = "PCIe throughput counters unavailable in this NVML build"
                tx_err = rx_err

            util_percent = _safe_non_negative_float(getattr(util, "gpu", None)) if util is not None else None
            mem_used = int(getattr(mem_info, "used", 0)) if mem_info is not None else None
            mem_total = int(getattr(mem_info, "total", 0)) if mem_info is not None else None
            mem_pressure = None
            if mem_used is not None and mem_total is not None and mem_total > 0:
                mem_pressure = round((mem_used / float(mem_total)) * 100.0, 2)

            metrics = {
                "utilization_percent": _semantic_metric(
                    key="utilization_percent",
                    name="Utilization",
                    value=util_percent,
                    status="active" if util_percent is not None else "error",
                    source="nvml",
                    probe="nvmlDeviceGetUtilizationRates",
                    unit="%",
                    error=util_err,
                    timestamp=collected_at,
                ),
                "memory_pressure_percent": _semantic_metric(
                    key="memory_pressure_percent",
                    name="Memory pressure",
                    value=mem_pressure,
                    status="active" if mem_pressure is not None else "error",
                    source="nvml",
                    probe="nvmlDeviceGetMemoryInfo",
                    unit="%",
                    error=mem_err,
                    timestamp=collected_at,
                    details={"used_bytes": mem_used, "total_bytes": mem_total},
                ),
                "memory_fragmentation_percent": _semantic_metric(
                    key="memory_fragmentation_percent",
                    name="Memory fragmentation estimate",
                    value=mem_pressure,
                    status="active" if mem_pressure is not None else "not_supported",
                    source="nvml",
                    probe="nvmlDeviceGetMemoryInfo",
                    unit="%",
                    error=(
                        "Allocator-level fragmentation counters are not exposed by NVML; showing estimate from memory pressure"
                        if mem_pressure is not None
                        else mem_err
                    ),
                    timestamp=collected_at,
                    details={"estimated": True, "used_bytes": mem_used, "total_bytes": mem_total},
                ),
                "power_watts": _semantic_metric(
                    key="power_watts",
                    name="Power draw",
                    value=(round(float(power) / 1000.0, 2) if power is not None else None),
                    status="active" if power is not None else "not_supported",
                    source="nvml",
                    probe="nvmlDeviceGetPowerUsage",
                    unit="W",
                    error=power_err,
                    timestamp=collected_at,
                ),
                "temperature_c": _semantic_metric(
                    key="temperature_c",
                    name="Temperature",
                    value=(float(temp) if temp is not None else None),
                    status="active" if temp is not None else "not_supported",
                    source="nvml",
                    probe="nvmlDeviceGetTemperature",
                    unit="°C",
                    error=temp_err,
                    timestamp=collected_at,
                ),
                "fan_speed_percent": _semantic_metric(
                    key="fan_speed_percent",
                    name="Fan speed",
                    value=(float(fan) if fan is not None else None),
                    status="active" if fan is not None else "not_supported",
                    source="nvml",
                    probe="nvmlDeviceGetFanSpeed",
                    unit="%",
                    error=fan_err,
                    timestamp=collected_at,
                ),
                "interconnect_rx_mb_s": _semantic_metric(
                    key="interconnect_rx_mb_s",
                    name="Interconnect RX",
                    value=(round(float(rx_bytes) / 1024.0, 2) if rx_bytes is not None else None),
                    status="active" if rx_bytes is not None else "not_supported",
                    source="nvml",
                    probe="nvmlDeviceGetPcieThroughput(RX)",
                    unit="MB/s",
                    error=rx_err,
                    timestamp=collected_at,
                ),
                "interconnect_tx_mb_s": _semantic_metric(
                    key="interconnect_tx_mb_s",
                    name="Interconnect TX",
                    value=(round(float(tx_bytes) / 1024.0, 2) if tx_bytes is not None else None),
                    status="active" if tx_bytes is not None else "not_supported",
                    source="nvml",
                    probe="nvmlDeviceGetPcieThroughput(TX)",
                    unit="MB/s",
                    error=tx_err,
                    timestamp=collected_at,
                ),
            }

            accelerators.append(
                {
                    "id": f"cuda:{index}",
                    "index": index,
                    "backend": "cuda",
                    "vendor": "nvidia",
                    "name": name,
                    "status": "active",
                    "metrics": metrics,
                }
            )
    finally:
        _shutdown_nvml(nvml)

    return accelerators, diagnostics


def _collect_mps_accelerators(collected_at: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    mps_active, reason = _detect_mps_active()
    diagnostics: list[dict[str, Any]] = []
    if not mps_active:
        if reason:
            reason_lower = str(reason).lower()
            if "not loaded" in reason_lower or "skipping direct mps probe" in reason_lower:
                status = "not_supported"
            elif "not installed" in reason_lower:
                status = "dependency_missing"
            else:
                status = "dependency_missing"
            diagnostics.append(
                {
                    "scope": "mps",
                    "status": status,
                    "message": reason,
                    "source": "torch",
                    "probe": "torch.backends.mps.is_available",
                }
            )
        else:
            diagnostics.append(
                {
                    "scope": "mps",
                    "status": "not_detected",
                    "message": "MPS accelerator not detected",
                    "source": "torch",
                    "probe": "torch.backends.mps.is_available",
                }
            )
        return [], diagnostics

    metrics = {
        "utilization_percent": _semantic_metric(
            key="utilization_percent",
            name="Utilization",
            value=None,
            status="not_supported",
            source="torch",
            probe="torch.backends.mps",
            unit="%",
            error="Low-level MPS utilization counters are not exposed by the platform",
            timestamp=collected_at,
        ),
        "memory_pressure_percent": _semantic_metric(
            key="memory_pressure_percent",
            name="Memory pressure estimate",
            value=None,
            status="not_supported",
            source="torch",
            probe="torch.backends.mps",
            unit="%",
            error="Allocator-level MPS memory pressure is not exposed directly",
            timestamp=collected_at,
        ),
        "memory_fragmentation_percent": _semantic_metric(
            key="memory_fragmentation_percent",
            name="Memory fragmentation estimate",
            value=None,
            status="not_supported",
            source="torch",
            probe="torch.backends.mps",
            unit="%",
            error="Allocator-level MPS memory fragmentation telemetry is not exposed by stable APIs",
            timestamp=collected_at,
        ),
        "power_watts": _semantic_metric(
            key="power_watts",
            name="Power draw",
            value=None,
            status="not_supported",
            source="mps",
            probe="platform API",
            unit="W",
            error="MPS power telemetry is not available through stable APIs",
            timestamp=collected_at,
        ),
        "temperature_c": _semantic_metric(
            key="temperature_c",
            name="Temperature",
            value=None,
            status="not_supported",
            source="mps",
            probe="platform API",
            unit="°C",
            error="MPS temperature telemetry is not available through stable APIs",
            timestamp=collected_at,
        ),
        "fan_speed_percent": _semantic_metric(
            key="fan_speed_percent",
            name="Fan speed",
            value=None,
            status="not_supported",
            source="mps",
            probe="platform API",
            unit="%",
            error="Fan telemetry is not exposed for MPS through this probe",
            timestamp=collected_at,
        ),
        "interconnect_rx_mb_s": _semantic_metric(
            key="interconnect_rx_mb_s",
            name="Interconnect RX",
            value=None,
            status="not_supported",
            source="mps",
            probe="platform API",
            unit="MB/s",
            error="Interconnect counters are unavailable for MPS",
            timestamp=collected_at,
        ),
        "interconnect_tx_mb_s": _semantic_metric(
            key="interconnect_tx_mb_s",
            name="Interconnect TX",
            value=None,
            status="not_supported",
            source="mps",
            probe="platform API",
            unit="MB/s",
            error="Interconnect counters are unavailable for MPS",
            timestamp=collected_at,
        ),
    }

    accelerators = [
        {
            "id": "mps:0",
            "index": 0,
            "backend": "mps",
            "vendor": "apple",
            "name": "Apple Silicon MPS",
            "status": "active",
            "metrics": metrics,
        }
    ]
    return accelerators, diagnostics


def _aggregate_accelerator_metrics(accelerators: list[dict[str, Any]], collected_at: float) -> dict[str, Any]:
    def _extract_metric_values(metric_key: str) -> list[float]:
        values: list[float] = []
        for accelerator in accelerators:
            metric = accelerator.get("metrics", {}).get(metric_key, {})
            value = _safe_non_negative_float(metric.get("value")) if isinstance(metric, dict) else None
            if value is not None:
                values.append(value)
        return values

    util_values = _extract_metric_values("utilization_percent")
    mem_values = _extract_metric_values("memory_pressure_percent")
    frag_values = _extract_metric_values("memory_fragmentation_percent")
    power_values = _extract_metric_values("power_watts")
    temp_values = _extract_metric_values("temperature_c")
    fan_values = _extract_metric_values("fan_speed_percent")
    interconnect_rx_values = _extract_metric_values("interconnect_rx_mb_s")
    interconnect_tx_values = _extract_metric_values("interconnect_tx_mb_s")

    def _avg(values: list[float]) -> Optional[float]:
        if not values:
            return None
        return round(sum(values) / float(len(values)), 2)

    return {
        "device_count": len(accelerators),
        "metrics": {
            "utilization_percent": _semantic_metric(
                key="utilization_percent",
                name="Aggregate utilization",
                value=_avg(util_values),
                status="active" if util_values else "not_detected",
                source="aggregator",
                probe="accelerator.utilization",
                unit="%",
                timestamp=collected_at,
            ),
            "memory_pressure_percent": _semantic_metric(
                key="memory_pressure_percent",
                name="Aggregate memory pressure",
                value=_avg(mem_values),
                status="active" if mem_values else "not_detected",
                source="aggregator",
                probe="accelerator.memory",
                unit="%",
                timestamp=collected_at,
            ),
            "memory_fragmentation_percent": _semantic_metric(
                key="memory_fragmentation_percent",
                name="Aggregate memory fragmentation estimate",
                value=_avg(frag_values),
                status="active" if frag_values else "not_supported",
                source="aggregator",
                probe="accelerator.memory.fragmentation",
                unit="%",
                timestamp=collected_at,
            ),
            "power_watts": _semantic_metric(
                key="power_watts",
                name="Aggregate power draw",
                value=(round(sum(power_values), 2) if power_values else None),
                status="active" if power_values else "not_supported",
                source="aggregator",
                probe="accelerator.power",
                unit="W",
                timestamp=collected_at,
            ),
            "temperature_c": _semantic_metric(
                key="temperature_c",
                name="Aggregate temperature",
                value=_avg(temp_values),
                status="active" if temp_values else "not_supported",
                source="aggregator",
                probe="accelerator.temperature",
                unit="°C",
                timestamp=collected_at,
            ),
            "fan_speed_percent": _semantic_metric(
                key="fan_speed_percent",
                name="Aggregate fan speed",
                value=_avg(fan_values),
                status="active" if fan_values else "not_supported",
                source="aggregator",
                probe="accelerator.fan",
                unit="%",
                timestamp=collected_at,
            ),
            "interconnect_rx_mb_s": _semantic_metric(
                key="interconnect_rx_mb_s",
                name="Aggregate interconnect RX",
                value=_avg(interconnect_rx_values),
                status="active" if interconnect_rx_values else "not_supported",
                source="aggregator",
                probe="accelerator.interconnect.rx",
                unit="MB/s",
                timestamp=collected_at,
            ),
            "interconnect_tx_mb_s": _semantic_metric(
                key="interconnect_tx_mb_s",
                name="Aggregate interconnect TX",
                value=_avg(interconnect_tx_values),
                status="active" if interconnect_tx_values else "not_supported",
                source="aggregator",
                probe="accelerator.interconnect.tx",
                unit="MB/s",
                timestamp=collected_at,
            ),
        },
    }


def _get_counter_rate(
    cache_key: str, values: dict[str, float], now_ts: float
) -> tuple[dict[str, Optional[float]], bool]:
    previous = _COUNTER_CACHE.get(cache_key)
    _COUNTER_CACHE[cache_key] = {**values, "ts": now_ts}
    if not previous:
        return ({k: None for k in values.keys()}, True)

    prev_ts = _safe_float(previous.get("ts"))
    if prev_ts is None or now_ts <= prev_ts:
        return ({k: None for k in values.keys()}, False)

    dt = now_ts - prev_ts
    out: dict[str, Optional[float]] = {}
    for key, value in values.items():
        prev_value = _safe_float(previous.get(key))
        if prev_value is None or value < prev_value:
            out[key] = None
        else:
            out[key] = (value - prev_value) / dt
    return out, False


def _collect_host_metrics(run_id: str, collected_at: float) -> dict[str, dict[str, Any]]:
    psutil = _import_psutil()
    if psutil is None:
        reason = "psutil is not installed"
        return {
            "system_cpu_percent": _semantic_metric(
                key="system_cpu_percent",
                name="System CPU",
                value=None,
                status="dependency_missing",
                source="local_host",
                probe="psutil.cpu_percent",
                unit="%",
                error=reason,
                timestamp=collected_at,
            ),
            "system_ram_percent": _semantic_metric(
                key="system_ram_percent",
                name="System RAM",
                value=None,
                status="dependency_missing",
                source="local_host",
                probe="psutil.virtual_memory",
                unit="%",
                error=reason,
                timestamp=collected_at,
            ),
            "disk_read_mb_s": _semantic_metric(
                key="disk_read_mb_s",
                name="Disk read throughput",
                value=None,
                status="dependency_missing",
                source="local_host",
                probe="psutil.disk_io_counters",
                unit="MB/s",
                error=reason,
                timestamp=collected_at,
            ),
            "disk_write_mb_s": _semantic_metric(
                key="disk_write_mb_s",
                name="Disk write throughput",
                value=None,
                status="dependency_missing",
                source="local_host",
                probe="psutil.disk_io_counters",
                unit="MB/s",
                error=reason,
                timestamp=collected_at,
            ),
            "network_rx_mb_s": _semantic_metric(
                key="network_rx_mb_s",
                name="Network RX throughput",
                value=None,
                status="dependency_missing",
                source="local_host",
                probe="psutil.net_io_counters",
                unit="MB/s",
                error=reason,
                timestamp=collected_at,
            ),
            "network_tx_mb_s": _semantic_metric(
                key="network_tx_mb_s",
                name="Network TX throughput",
                value=None,
                status="dependency_missing",
                source="local_host",
                probe="psutil.net_io_counters",
                unit="MB/s",
                error=reason,
                timestamp=collected_at,
            ),
        }

    cpu = float(psutil.cpu_percent(interval=None))
    vm = psutil.virtual_memory()

    disk_read: Optional[float] = None
    disk_write: Optional[float] = None
    net_rx: Optional[float] = None
    net_tx: Optional[float] = None
    disk_probe_error: Optional[str] = None
    net_probe_error: Optional[str] = None
    disk_warmup = False
    net_warmup = False

    disk_probe = getattr(psutil, "disk_io_counters", None)
    if callable(disk_probe):
        try:
            disk_counters = disk_probe()
            if disk_counters is not None:
                disk_rates, disk_warmup = _get_counter_rate(
                    f"{run_id}:disk",
                    {
                        "read_bytes": float(getattr(disk_counters, "read_bytes", 0.0)),
                        "write_bytes": float(getattr(disk_counters, "write_bytes", 0.0)),
                    },
                    collected_at,
                )
            else:
                disk_rates = {"read_bytes": None, "write_bytes": None}
        except Exception as exc:
            disk_rates = {"read_bytes": None, "write_bytes": None}
            disk_probe_error = str(exc)
    else:
        disk_rates = {"read_bytes": None, "write_bytes": None}
        disk_probe_error = "psutil.disk_io_counters is unavailable on this platform"

    net_probe = getattr(psutil, "net_io_counters", None)
    if callable(net_probe):
        try:
            net_counters = net_probe()
            if net_counters is not None:
                net_rates, net_warmup = _get_counter_rate(
                    f"{run_id}:net",
                    {
                        "bytes_recv": float(getattr(net_counters, "bytes_recv", 0.0)),
                        "bytes_sent": float(getattr(net_counters, "bytes_sent", 0.0)),
                    },
                    collected_at,
                )
            else:
                net_rates = {"bytes_recv": None, "bytes_sent": None}
        except Exception as exc:
            net_rates = {"bytes_recv": None, "bytes_sent": None}
            net_probe_error = str(exc)
    else:
        net_rates = {"bytes_recv": None, "bytes_sent": None}
        net_probe_error = "psutil.net_io_counters is unavailable on this platform"

    def _to_mb_s(value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        return round(value / float(1024**2), 3)

    disk_read = _to_mb_s(disk_rates.get("read_bytes"))
    disk_write = _to_mb_s(disk_rates.get("write_bytes"))
    net_rx = _to_mb_s(net_rates.get("bytes_recv"))
    net_tx = _to_mb_s(net_rates.get("bytes_sent"))

    if disk_warmup:
        if disk_read is None:
            disk_read = 0.0
        if disk_write is None:
            disk_write = 0.0
    if net_warmup:
        if net_rx is None:
            net_rx = 0.0
        if net_tx is None:
            net_tx = 0.0

    return {
        "system_cpu_percent": _semantic_metric(
            key="system_cpu_percent",
            name="System CPU",
            value=round(cpu, 2),
            status="active",
            source="local_host",
            probe="psutil.cpu_percent",
            unit="%",
            timestamp=collected_at,
        ),
        "system_ram_percent": _semantic_metric(
            key="system_ram_percent",
            name="System RAM",
            value=round(float(vm.percent), 2),
            status="active",
            source="local_host",
            probe="psutil.virtual_memory",
            unit="%",
            timestamp=collected_at,
            details={"used_bytes": int(vm.used), "total_bytes": int(vm.total)},
        ),
        "disk_read_mb_s": _semantic_metric(
            key="disk_read_mb_s",
            name="Disk read throughput",
            value=disk_read,
            status=(
                "active"
                if disk_read is not None
                else ("not_supported" if disk_probe_error is not None else "not_detected")
            ),
            source="local_host",
            probe="psutil.disk_io_counters",
            unit="MB/s",
            error=disk_probe_error,
            details={"warmup": bool(disk_warmup), "counter_delta": True},
            display=("0.00MB/s (warmup)" if disk_warmup else None),
            timestamp=collected_at,
        ),
        "disk_write_mb_s": _semantic_metric(
            key="disk_write_mb_s",
            name="Disk write throughput",
            value=disk_write,
            status=(
                "active"
                if disk_write is not None
                else ("not_supported" if disk_probe_error is not None else "not_detected")
            ),
            source="local_host",
            probe="psutil.disk_io_counters",
            unit="MB/s",
            error=disk_probe_error,
            details={"warmup": bool(disk_warmup), "counter_delta": True},
            display=("0.00MB/s (warmup)" if disk_warmup else None),
            timestamp=collected_at,
        ),
        "network_rx_mb_s": _semantic_metric(
            key="network_rx_mb_s",
            name="Network RX throughput",
            value=net_rx,
            status=(
                "active" if net_rx is not None else ("not_supported" if net_probe_error is not None else "not_detected")
            ),
            source="local_host",
            probe="psutil.net_io_counters",
            unit="MB/s",
            error=net_probe_error,
            details={"warmup": bool(net_warmup), "counter_delta": True},
            display=("0.00MB/s (warmup)" if net_warmup else None),
            timestamp=collected_at,
        ),
        "network_tx_mb_s": _semantic_metric(
            key="network_tx_mb_s",
            name="Network TX throughput",
            value=net_tx,
            status=(
                "active" if net_tx is not None else ("not_supported" if net_probe_error is not None else "not_detected")
            ),
            source="local_host",
            probe="psutil.net_io_counters",
            unit="MB/s",
            error=net_probe_error,
            details={"warmup": bool(net_warmup), "counter_delta": True},
            display=("0.00MB/s (warmup)" if net_warmup else None),
            timestamp=collected_at,
        ),
    }


def _collect_training_process_metrics(
    runtime_state: dict[str, Any], collected_at: float, host_cpu_metric: Optional[dict[str, Any]] = None
) -> dict[str, dict[str, Any]]:
    pid = _safe_positive_int(runtime_state.get("training_pid"))
    expected_start = _safe_float(runtime_state.get("training_process_start_time"))

    psutil = _import_psutil()
    if pid is None:
        return {
            "pid": _semantic_metric(
                key="pid",
                name="Training PID",
                value=None,
                status="not_detected",
                source="runtime_state",
                probe="runtime_state.training_pid",
                timestamp=collected_at,
            ),
            "process_cpu_percent": _semantic_metric(
                key="process_cpu_percent",
                name="Training process CPU",
                value=None,
                status="not_detected",
                source="runtime_state",
                probe="psutil.Process.cpu_percent",
                unit="%",
                timestamp=collected_at,
            ),
            "process_ram_percent": _semantic_metric(
                key="process_ram_percent",
                name="Training process RAM",
                value=None,
                status="not_detected",
                source="runtime_state",
                probe="psutil.Process.memory_info",
                unit="%",
                timestamp=collected_at,
            ),
            "process_rss_mb": _semantic_metric(
                key="process_rss_mb",
                name="Training process RSS",
                value=None,
                status="not_detected",
                source="runtime_state",
                probe="psutil.Process.memory_info",
                unit="MB",
                timestamp=collected_at,
            ),
            "liveness": _semantic_metric(
                key="liveness",
                name="Training process liveness",
                value=None,
                status="not_detected",
                source="runtime_state",
                probe="runtime_state.training_pid",
                timestamp=collected_at,
            ),
        }

    if psutil is None:
        return {
            "pid": _semantic_metric(
                key="pid",
                name="Training PID",
                value=pid,
                status="active",
                source="runtime_state",
                probe="runtime_state.training_pid",
                timestamp=collected_at,
            ),
            "process_cpu_percent": _semantic_metric(
                key="process_cpu_percent",
                name="Training process CPU",
                value=_safe_non_negative_float(runtime_state.get("resource", {}).get("cpu_percent")),
                status="dependency_missing",
                source="runtime_state",
                probe="runtime_state.resource.cpu_percent",
                unit="%",
                error="psutil is not installed",
                timestamp=collected_at,
            ),
            "process_ram_percent": _semantic_metric(
                key="process_ram_percent",
                name="Training process RAM",
                value=None,
                status="dependency_missing",
                source="runtime_state",
                probe="runtime_state.resource",
                unit="%",
                error="psutil is not installed",
                timestamp=collected_at,
            ),
            "process_rss_mb": _semantic_metric(
                key="process_rss_mb",
                name="Training process RSS",
                value=(
                    round(float(runtime_state.get("resource", {}).get("rss_bytes", 0.0)) / float(1024**2), 2)
                    if _safe_float(runtime_state.get("resource", {}).get("rss_bytes")) is not None
                    else None
                ),
                status="dependency_missing",
                source="runtime_state",
                probe="runtime_state.resource.rss_bytes",
                unit="MB",
                error="psutil is not installed",
                timestamp=collected_at,
            ),
            "liveness": _semantic_metric(
                key="liveness",
                name="Training process liveness",
                value=None,
                status="dependency_missing",
                source="runtime_state",
                probe="os.kill",
                error="psutil is not installed",
                timestamp=collected_at,
            ),
        }

    try:
        proc = psutil.Process(pid)
        try:
            create_time = _safe_float(proc.create_time())
        except Exception:
            create_time = None
        if expected_start is not None and create_time is not None and abs(create_time - expected_start) > 2.0:
            _drop_process_cpu_cache(pid)
            raise RuntimeError("PID reused by a different process")
        alive = bool(proc.is_running()) and proc.status() != getattr(psutil, "STATUS_ZOMBIE", "zombie")
        if not alive:
            _drop_process_cpu_cache(pid)
            raise RuntimeError("training process is not running")
        proc_cpu, cpu_warmup = _sample_training_process_cpu_percent(
            pid=pid, process=proc, process_start_time=create_time
        )
        mem_info = proc.memory_info()
        rss_bytes = int(getattr(mem_info, "rss", 0))
        process_ram_percent = float(proc.memory_percent())
        host_cpu = _safe_float((host_cpu_metric or {}).get("value"))
        cpu_share = None
        if host_cpu is not None and host_cpu > 0 and proc_cpu is not None:
            cpu_share = round(min(100.0, max(0.0, (proc_cpu / host_cpu) * 100.0)), 2)

        cpu_details: dict[str, Any] = {"warmup": bool(cpu_warmup)}
        if cpu_share is not None:
            cpu_details["cpu_share_of_host_percent"] = cpu_share

        return {
            "pid": _semantic_metric(
                key="pid",
                name="Training PID",
                value=pid,
                status="active",
                source="psutil",
                probe="psutil.Process.pid",
                timestamp=collected_at,
            ),
            "process_cpu_percent": _semantic_metric(
                key="process_cpu_percent",
                name="Training process CPU",
                value=round(proc_cpu, 2) if proc_cpu is not None else None,
                status="active" if proc_cpu is not None else "not_detected",
                source="psutil",
                probe="psutil.Process.cpu_percent",
                unit="%",
                timestamp=collected_at,
                details=cpu_details,
            ),
            "process_ram_percent": _semantic_metric(
                key="process_ram_percent",
                name="Training process RAM",
                value=round(process_ram_percent, 2),
                status="active",
                source="psutil",
                probe="psutil.Process.memory_percent",
                unit="%",
                timestamp=collected_at,
            ),
            "process_rss_mb": _semantic_metric(
                key="process_rss_mb",
                name="Training process RSS",
                value=round(rss_bytes / float(1024**2), 2),
                status="active",
                source="psutil",
                probe="psutil.Process.memory_info",
                unit="MB",
                timestamp=collected_at,
                details={"rss_bytes": rss_bytes},
            ),
            "liveness": _semantic_metric(
                key="liveness",
                name="Training process liveness",
                value=True,
                status="active",
                source="psutil",
                probe="psutil.Process.is_running",
                timestamp=collected_at,
            ),
        }
    except Exception as exc:
        _drop_process_cpu_cache(pid)
        return {
            "pid": _semantic_metric(
                key="pid",
                name="Training PID",
                value=pid,
                status="active",
                source="runtime_state",
                probe="runtime_state.training_pid",
                timestamp=collected_at,
            ),
            "process_cpu_percent": _semantic_metric(
                key="process_cpu_percent",
                name="Training process CPU",
                value=None,
                status="interrupted_training_stopped",
                source="psutil",
                probe="psutil.Process.cpu_percent",
                unit="%",
                error=str(exc),
                timestamp=collected_at,
            ),
            "process_ram_percent": _semantic_metric(
                key="process_ram_percent",
                name="Training process RAM",
                value=None,
                status="interrupted_training_stopped",
                source="psutil",
                probe="psutil.Process.memory_percent",
                unit="%",
                error=str(exc),
                timestamp=collected_at,
            ),
            "process_rss_mb": _semantic_metric(
                key="process_rss_mb",
                name="Training process RSS",
                value=None,
                status="interrupted_training_stopped",
                source="psutil",
                probe="psutil.Process.memory_info",
                unit="MB",
                error=str(exc),
                timestamp=collected_at,
            ),
            "liveness": _semantic_metric(
                key="liveness",
                name="Training process liveness",
                value=False,
                status="interrupted_training_stopped",
                source="psutil",
                probe="psutil.Process.is_running",
                error=str(exc),
                timestamp=collected_at,
            ),
        }


def _latest_numeric_metric(metrics: list[dict[str, Any]], keys: list[str]) -> Optional[float]:
    for row in reversed(metrics):
        if not isinstance(row, dict):
            continue
        for key in keys:
            value = _safe_float(row.get(key))
            if value is not None:
                return value
    return None


def _build_local_performance_insights(
    *,
    metrics_rows: list[dict[str, Any]],
    host_metrics: dict[str, dict[str, Any]],
    process_metrics: dict[str, dict[str, Any]],
    aggregate_accelerator: dict[str, Any],
    collected_at: float,
) -> dict[str, dict[str, Any]]:
    samples_per_sec = _latest_numeric_metric(metrics_rows, ["samples_per_sec", "tokens_per_sec", "throughput"])
    dataloader_wait_s = _latest_numeric_metric(
        metrics_rows, ["dataloader_wait_time_s", "dataloader_wait_s", "data_wait_s", "data_time"]
    )
    h2d_transfer_s = _latest_numeric_metric(
        metrics_rows, ["host_to_device_transfer_time_s", "host_to_device_time_s", "h2d_time_s", "transfer_time_s"]
    )

    host_cpu = _safe_float(host_metrics.get("system_cpu_percent", {}).get("value"))
    process_cpu = _safe_float(process_metrics.get("process_cpu_percent", {}).get("value"))
    ram_pressure = _safe_float(host_metrics.get("system_ram_percent", {}).get("value"))

    accel_util = _safe_float(
        aggregate_accelerator.get("metrics", {}).get("utilization_percent", {}).get("value")
        if isinstance(aggregate_accelerator, dict)
        else None
    )

    disk_read = _safe_float(host_metrics.get("disk_read_mb_s", {}).get("value")) or 0.0
    disk_write = _safe_float(host_metrics.get("disk_write_mb_s", {}).get("value")) or 0.0
    disk_total = disk_read + disk_write

    cpu_bottleneck_score = None
    if host_cpu is not None and process_cpu is not None:
        cpu_bottleneck_score = round(min(100.0, max(0.0, (0.55 * host_cpu) + (0.45 * process_cpu))), 2)

    memory_pressure_score = None
    if ram_pressure is not None:
        memory_pressure_score = round(min(100.0, max(0.0, ram_pressure)), 2)

    disk_pressure_score = round(min(100.0, disk_total * 2.0), 2) if disk_total > 0 else None

    accelerator_starvation = None
    if accel_util is not None and process_cpu is not None:
        starvation = max(0.0, min(100.0, (100.0 - accel_util) * (process_cpu / 100.0)))
        accelerator_starvation = round(starvation, 2)

    scaling_readiness = None
    if any(
        v is not None
        for v in (cpu_bottleneck_score, memory_pressure_score, disk_pressure_score, accelerator_starvation)
    ):
        penalties = [
            v
            for v in (cpu_bottleneck_score, memory_pressure_score, disk_pressure_score, accelerator_starvation)
            if v is not None
        ]
        scaling_readiness = round(max(0.0, 100.0 - (sum(penalties) / float(len(penalties)))), 2)

    return {
        "batch_loading_speed": _semantic_metric(
            key="batch_loading_speed",
            name="Batch loading speed",
            value=samples_per_sec,
            status="active" if samples_per_sec is not None else "not_detected",
            source="training_metrics",
            probe="metrics.jsonl(samples_per_sec|throughput)",
            unit=" samples/s",
            timestamp=collected_at,
        ),
        "samples_per_sec": _semantic_metric(
            key="samples_per_sec",
            name="Samples per second",
            value=samples_per_sec,
            status="active" if samples_per_sec is not None else "not_detected",
            source="training_metrics",
            probe="metrics.jsonl(samples_per_sec|tokens_per_sec|throughput)",
            unit=" samples/s",
            timestamp=collected_at,
        ),
        "dataloader_wait_time_s": _semantic_metric(
            key="dataloader_wait_time_s",
            name="Dataloader wait time",
            value=dataloader_wait_s,
            status="active" if dataloader_wait_s is not None else "not_detected",
            source="training_metrics",
            probe="metrics.jsonl(data_wait)",
            unit="s",
            timestamp=collected_at,
        ),
        "host_to_device_transfer_time_s": _semantic_metric(
            key="host_to_device_transfer_time_s",
            name="Host-to-device transfer time",
            value=h2d_transfer_s,
            status="active" if h2d_transfer_s is not None else "not_detected",
            source="training_metrics",
            probe="metrics.jsonl(host_to_device)",
            unit="s",
            timestamp=collected_at,
        ),
        "cpu_bottleneck_score": _semantic_metric(
            key="cpu_bottleneck_score",
            name="CPU bottleneck score",
            value=cpu_bottleneck_score,
            status="active" if cpu_bottleneck_score is not None else "not_detected",
            source="heuristic",
            probe="host_cpu + process_cpu",
            unit="%",
            timestamp=collected_at,
        ),
        "memory_pressure": _semantic_metric(
            key="memory_pressure",
            name="Memory pressure",
            value=memory_pressure_score,
            status="active" if memory_pressure_score is not None else "not_detected",
            source="heuristic",
            probe="system_ram_percent",
            unit="%",
            timestamp=collected_at,
        ),
        "disk_io_pressure": _semantic_metric(
            key="disk_io_pressure",
            name="Disk I/O pressure",
            value=disk_pressure_score,
            status="active" if disk_pressure_score is not None else "not_detected",
            source="heuristic",
            probe="disk_read_mb_s + disk_write_mb_s",
            unit="%",
            timestamp=collected_at,
        ),
        "accelerator_starvation_idle": _semantic_metric(
            key="accelerator_starvation_idle",
            name="Accelerator starvation / idle",
            value=accelerator_starvation,
            status="active" if accelerator_starvation is not None else "not_detected",
            source="heuristic",
            probe="aggregate_accelerator.utilization + process_cpu",
            unit="%",
            timestamp=collected_at,
        ),
        "scaling_readiness": _semantic_metric(
            key="scaling_readiness",
            name="Estimated scaling readiness",
            value=scaling_readiness,
            status="active" if scaling_readiness is not None else "not_detected",
            source="heuristic",
            probe="combined local bottleneck heuristics",
            unit="%",
            timestamp=collected_at,
        ),
    }


def _build_cluster_metrics(
    *,
    mode: str,
    cluster_metric: dict[str, Any],
    aggregate_accelerator: dict[str, Any],
    host_metrics: dict[str, dict[str, Any]],
    collected_at: float,
) -> Optional[dict[str, dict[str, Any]]]:
    if mode != "distributed":
        return None

    value = cluster_metric.get("value", {}) if isinstance(cluster_metric, dict) else {}
    active_nodes = _safe_positive_int(value.get("active_nodes")) or 1
    total_nodes = _safe_positive_int(value.get("total_nodes")) or max(active_nodes, 1)

    interconnect_rx = _safe_float(
        aggregate_accelerator.get("metrics", {}).get("interconnect_rx_mb_s", {}).get("value")
        if isinstance(aggregate_accelerator, dict)
        else None
    )
    interconnect_tx = _safe_float(
        aggregate_accelerator.get("metrics", {}).get("interconnect_tx_mb_s", {}).get("value")
        if isinstance(aggregate_accelerator, dict)
        else None
    )

    if interconnect_rx is None:
        interconnect_rx = _safe_float(host_metrics.get("network_rx_mb_s", {}).get("value"))
    if interconnect_tx is None:
        interconnect_tx = _safe_float(host_metrics.get("network_tx_mb_s", {}).get("value"))

    return {
        "cluster_nodes": _semantic_metric(
            key="cluster_nodes",
            name="Cluster nodes",
            value={"active_nodes": active_nodes, "total_nodes": total_nodes},
            status="active",
            source="run_artifacts",
            probe="distributed_index.json",
            display=f"{active_nodes} / {total_nodes}",
            timestamp=collected_at,
        ),
        "interconnect_rx_mb_s": _semantic_metric(
            key="interconnect_rx_mb_s",
            name="Interconnect RX",
            value=interconnect_rx,
            status="active" if interconnect_rx is not None else "not_supported",
            source="nvml|host_network",
            probe="pcie_throughput|net_io",
            unit="MB/s",
            timestamp=collected_at,
        ),
        "interconnect_tx_mb_s": _semantic_metric(
            key="interconnect_tx_mb_s",
            name="Interconnect TX",
            value=interconnect_tx,
            status="active" if interconnect_tx is not None else "not_supported",
            source="nvml|host_network",
            probe="pcie_throughput|net_io",
            unit="MB/s",
            timestamp=collected_at,
        ),
    }


def _build_external_usage(
    host_metrics: dict[str, dict[str, Any]], process_metrics: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    return {
        "host_cpu_percent": host_metrics.get(
            "system_cpu_percent",
            _semantic_metric(
                key="host_cpu_percent",
                name="Host CPU",
                value=None,
                status="not_detected",
                source="local_host",
                probe="host_metrics.system_cpu_percent",
            ),
        ),
        "host_ram_percent": host_metrics.get(
            "system_ram_percent",
            _semantic_metric(
                key="host_ram_percent",
                name="Host RAM",
                value=None,
                status="not_detected",
                source="local_host",
                probe="host_metrics.system_ram_percent",
            ),
        ),
        "process_cpu_percent": process_metrics.get(
            "process_cpu_percent",
            _semantic_metric(
                key="process_cpu_percent",
                name="Training process CPU",
                value=None,
                status="not_detected",
                source="training_process",
                probe="training_process_metrics.process_cpu_percent",
                unit="%",
            ),
        ),
        "process_ram_percent": process_metrics.get(
            "process_ram_percent",
            _semantic_metric(
                key="process_ram_percent",
                name="Training process RAM",
                value=None,
                status="not_detected",
                source="training_process",
                probe="training_process_metrics.process_ram_percent",
                unit="%",
            ),
        ),
        "disk_read_mb_s": host_metrics.get(
            "disk_read_mb_s",
            _semantic_metric(
                key="disk_read_mb_s",
                name="Disk read throughput",
                value=None,
                status="not_detected",
                source="local_host",
                probe="host_metrics.disk_read_mb_s",
                unit="MB/s",
            ),
        ),
        "disk_write_mb_s": host_metrics.get(
            "disk_write_mb_s",
            _semantic_metric(
                key="disk_write_mb_s",
                name="Disk write throughput",
                value=None,
                status="not_detected",
                source="local_host",
                probe="host_metrics.disk_write_mb_s",
                unit="MB/s",
            ),
        ),
        "net_rx_mb_s": host_metrics.get(
            "network_rx_mb_s",
            _semantic_metric(
                key="net_rx_mb_s",
                name="Network RX throughput",
                value=None,
                status="not_detected",
                source="local_host",
                probe="host_metrics.network_rx_mb_s",
                unit="MB/s",
            ),
        ),
        "net_tx_mb_s": host_metrics.get(
            "network_tx_mb_s",
            _semantic_metric(
                key="net_tx_mb_s",
                name="Network TX throughput",
                value=None,
                status="not_detected",
                source="local_host",
                probe="host_metrics.network_tx_mb_s",
                unit="MB/s",
            ),
        ),
    }


def _is_metric_warmup(metric: Any) -> bool:
    if not isinstance(metric, dict):
        return False
    details = metric.get("details")
    if not isinstance(details, dict):
        return False
    return bool(details.get("warmup"))


def _has_active_numeric_metric(metric: Any) -> bool:
    if not isinstance(metric, dict):
        return False
    if metric.get("status") != "active":
        return False
    return _safe_float(metric.get("value")) is not None


def _build_telemetry_v2(
    *,
    store: ArtifactStore,
    run_id: str,
    collected_at: float,
    mode: str,
    metrics_legacy: dict[str, dict[str, Any]],
    live_guard: dict[str, Any],
) -> dict[str, Any]:
    runtime_state = store.get_runtime_state(run_id) or {}
    metadata = store.get_run_metadata(run_id) or {}
    metrics_rows = store.get_metrics(run_id)

    try:
        overview = build_overview_snapshot(
            run_id=run_id, metadata=metadata, metrics=metrics_rows, runtime_state=runtime_state, now_ts=collected_at
        )
    except Exception:
        overview = {
            "status": str(runtime_state.get("status") or metadata.get("status") or "idle").strip().lower() or "idle",
            "status_reason": None,
            "heartbeat_ts": _safe_float(runtime_state.get("heartbeat_ts")),
            "process_alive": None,
            "health_state": "WARNING",
        }

    host_metrics = _collect_host_metrics(run_id, collected_at)
    process_metrics = _collect_training_process_metrics(
        runtime_state, collected_at, host_cpu_metric=host_metrics.get("system_cpu_percent")
    )

    accelerators_cuda, cuda_diag = _collect_cuda_accelerators(collected_at)
    accelerators_mps, mps_diag = _collect_mps_accelerators(collected_at)

    diagnostics: list[dict[str, Any]] = []
    diagnostics.extend(cuda_diag)
    diagnostics.extend(mps_diag)

    accelerators = accelerators_cuda + accelerators_mps

    if accelerators:
        backends = {acc.get("backend") for acc in accelerators if isinstance(acc, dict)}
        if "cuda" in backends and "mps" in backends:
            accelerator_mode = "heterogeneous_active"
        elif "cuda" in backends:
            accelerator_mode = "cuda_active"
        elif "mps" in backends:
            accelerator_mode = "mps_active"
        else:
            accelerator_mode = "accelerator_unavailable"
    else:
        accelerator_mode = "cpu_only"

    aggregate_accelerator = _aggregate_accelerator_metrics(accelerators, collected_at)
    external_usage = _build_external_usage(host_metrics, process_metrics)
    local_performance_insights = _build_local_performance_insights(
        metrics_rows=metrics_rows,
        host_metrics=host_metrics,
        process_metrics=process_metrics,
        aggregate_accelerator=aggregate_accelerator,
        collected_at=collected_at,
    )

    cluster_metrics = _build_cluster_metrics(
        mode=mode,
        cluster_metric=metrics_legacy.get("cluster_nodes", {}),
        aggregate_accelerator=aggregate_accelerator,
        host_metrics=host_metrics,
        collected_at=collected_at,
    )

    if mode != "distributed":
        cluster_metrics = None
        diagnostics.append(
            {
                "scope": "cluster",
                "status": "requires_cluster_connection",
                "message": "Cluster telemetry is available only when distributed state is verified",
                "source": "run_artifacts",
                "probe": "distributed_index.json",
            }
        )

    run_status = str(overview.get("status") or "").strip().lower() or "idle"
    run_terminal = run_status in {"completed", "failed", "cancelled", "interrupted"}
    throughput_warmup = any(
        _is_metric_warmup(external_usage.get(key))
        for key in ("disk_read_mb_s", "disk_write_mb_s", "net_rx_mb_s", "net_tx_mb_s")
    )
    accelerator_series_active = any(
        _has_active_numeric_metric((aggregate_accelerator.get("metrics") or {}).get(key))
        for key in (
            "utilization_percent",
            "memory_pressure_percent",
            "memory_fragmentation_percent",
            "power_watts",
            "temperature_c",
        )
    )
    preferred_layout = "accelerator_first" if accelerator_series_active else "host_process_first"

    return {
        "accelerator_mode": accelerator_mode,
        "panel_mode": "cluster" if mode == "distributed" else "local_insights",
        "accelerators": accelerators,
        "aggregate_accelerator": aggregate_accelerator,
        "host_metrics": host_metrics,
        "training_process_metrics": process_metrics,
        "external_usage": external_usage,
        "graph_hints": {
            "preferred_layout": preferred_layout,
            "run_terminal": run_terminal,
            "throughput_warmup": throughput_warmup,
        },
        "local_performance_insights": local_performance_insights,
        "cluster_metrics": cluster_metrics,
        "run_state": {
            "status": run_status,
            "status_reason": overview.get("status_reason"),
            "health_state": overview.get("health_state"),
            "heartbeat_ts": overview.get("heartbeat_ts"),
            "process_alive": overview.get("process_alive"),
            "last_event": str(runtime_state.get("last_event", "") or "").strip().lower() or None,
        },
        "diagnostics": diagnostics,
        "metric_status_legend": SEMANTIC_STATUS_LABELS,
        "source": {
            "hostname": socket.gethostname(),
            "server_pid": int(os.getpid()),
            "collected_at": float(collected_at),
            "live_guard": live_guard,
        },
    }


def collect_infrastructure_telemetry(store: ArtifactStore, run_id: str) -> dict[str, Any]:
    collected_at = time.time()

    probes = {
        "cluster_nodes": lambda: query_cluster_nodes(store, run_id),
        "system_cpu": query_system_cpu,
        "system_ram": query_system_ram,
        "power_draw": query_power_draw,
        "multi_gpu_compute_utilization": query_multi_gpu_compute_utilization,
        "gpu_memory_fragmentation": query_gpu_memory_fragmentation,
    }

    metrics: dict[str, dict[str, Any]] = {}
    for metric_name, probe_fn in probes.items():
        command = f"{metric_name} probe"
        try:
            raw_metric = probe_fn()
        except Exception as exc:
            raw_metric = _metric(
                status="error",
                source="local_probe",
                function_name=getattr(probe_fn, "__name__", metric_name),
                command=command,
                label="Probe failed",
                value=None,
                error=str(exc),
                timestamp=collected_at,
            )
        metrics[metric_name] = _normalize_metric(
            raw_metric, function_name=getattr(probe_fn, "__name__", metric_name), command=command
        )

    live_guard, distributed_verified, distributed_candidate = _build_live_guard(store, run_id, collected_at)
    if distributed_candidate and not distributed_verified:
        metrics["cluster_nodes"] = _force_cluster_standalone(
            metrics.get("cluster_nodes", {}),
            [r for r in live_guard.get("reasons", []) if "distributed" in r or "rank" in r or "runtime" in r],
        )

    mode = "distributed" if distributed_verified else "standalone"
    telemetry_v2 = _build_telemetry_v2(
        store=store, run_id=run_id, collected_at=collected_at, mode=mode, metrics_legacy=metrics, live_guard=live_guard
    )

    return {
        "run_id": run_id,
        "mode": mode,
        "metrics": metrics,
        "gpu_devices": _collect_gpu_devices(metrics),
        "collected_at": float(collected_at),
        "live_guard": live_guard,
        "telemetry_v2": telemetry_v2,
    }
