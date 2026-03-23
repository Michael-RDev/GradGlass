from __future__ import annotations

import os
import statistics
import time
from typing import Any, Optional, Union

Number = Union[float, int]
Series = list[list[float]]
TERMINAL_STATUSES = {"completed", "failed", "cancelled", "interrupted"}
ACTIVE_STATUSES = {"idle", "starting", "running", "paused"}

RUN_STATUS_ALIASES = {
    "": "idle",
    "unknown": "idle",
    "idle": "idle",
    "starting": "starting",
    "initializing": "starting",
    "running": "running",
    "run": "running",
    "paused": "paused",
    "pause": "paused",
    "complete": "completed",
    "completed": "completed",
    "finished": "completed",
    "failed": "failed",
    "error": "failed",
    "cancelled": "cancelled",
    "canceled": "cancelled",
    "stopped": "cancelled",
    "stop": "cancelled",
    "interrupted": "interrupted",
    "abort": "interrupted",
    "aborted": "interrupted",
    "terminated": "interrupted",
}


class BaseExperimentAdapter:
    train_loss_candidates: tuple[str, ...] = (
        "loss",
        "train_loss",
        "training_loss",
        "train_rmse",
        "train_mae",
        "train_mse",
    )
    val_loss_candidates: tuple[str, ...] = (
        "val_loss",
        "validation_loss",
        "valid_loss",
        "test_loss",
        "val_rmse",
        "val_mae",
        "val_mse",
    )
    lr_candidates: tuple[str, ...] = ("lr", "learning_rate")

    def __init__(
        self,
        *,
        run_id: str,
        framework: str,
        metadata: dict[str, Any],
        metrics: list[dict[str, Any]],
        runtime_state: Optional[dict[str, Any]],
        now_ts: Optional[float] = None,
    ):
        self.run_id = run_id
        self.framework = framework
        self.metadata = metadata or {}
        self.metrics = metrics or []
        self.runtime_state = runtime_state or {}
        self.now_ts = float(now_ts) if now_ts is not None else time.time()

    def build_snapshot(self) -> dict[str, Any]:
        status, status_reason, process_alive = self._resolve_status()

        current_step = self._resolve_current_step()
        total_steps, total_steps_source = self._resolve_total_steps(current_step)

        train_loss = self._resolve_loss_history()
        val_loss = self._resolve_val_loss_history()
        lr_history = self._resolve_lr_history(current_step)

        latest_loss = train_loss[-1][1] if train_loss else None
        latest_val_loss = val_loss[-1][1] if val_loss else None
        current_lr = lr_history[-1][1] if lr_history else None

        elapsed_time_s = self._resolve_elapsed_time()
        eta_s, eta_reason = self._estimate_eta(status=status, current_step=current_step, total_steps=total_steps)

        heartbeat_ts = self._resolve_heartbeat()
        monitor_enabled = bool(
            self.runtime_state.get("monitor_enabled", self.metadata.get("config", {}).get("monitor", False))
        )
        resource_tracking_available = bool(self.runtime_state.get("resource_tracking_available", False))

        health_state = self._compute_health(
            status=status,
            heartbeat_ts=heartbeat_ts,
            current_step=current_step,
            resource_tracking_available=resource_tracking_available,
        )

        return {
            "run_id": self.run_id,
            "framework": self.framework,
            "status": status,
            "status_reason": status_reason,
            "health_state": health_state,
            "current_step": current_step,
            "total_steps": total_steps,
            "total_steps_source": total_steps_source,
            "elapsed_time_s": elapsed_time_s,
            "eta_s": eta_s,
            "eta_reason": eta_reason,
            "eta_is_live": bool(eta_s is not None and status == "running" and monitor_enabled),
            "loss_history": train_loss,
            "val_loss_history": val_loss,
            "lr_history": lr_history,
            "heartbeat_ts": heartbeat_ts,
            "process_alive": process_alive,
            "last_event": str(self.runtime_state.get("last_event", "") or "").strip().lower() or None,
            "resource_tracking_available": resource_tracking_available,
            "monitor_enabled": monitor_enabled,
            "latest_loss": latest_loss,
            "latest_val_loss": latest_val_loss,
            "current_lr": current_lr,
            "updated_at": self.now_ts,
        }

    def _resolve_status(self) -> tuple[str, Optional[str], Optional[bool]]:
        runtime_status = _normalize_run_status(self.runtime_state.get("status"))
        metadata_status = _normalize_run_status(self.metadata.get("status"))
        status = runtime_status or metadata_status or "idle"

        if self.runtime_state.get("fatal_exception"):
            return "failed", "fatal exception recorded in runtime state", _is_pid_alive(self.runtime_state)

        if status in TERMINAL_STATUSES:
            return status, None, _is_pid_alive(self.runtime_state)

        last_event = str(self.runtime_state.get("last_event", "") or "").strip().lower()
        if last_event in {"cancel", "cancelled", "manual_stop", "keyboard_interrupt", "ctrl_c"}:
            return "cancelled", "explicit stop event recorded", _is_pid_alive(self.runtime_state)
        if last_event in {"interrupt", "interrupted", "abort", "aborted", "terminated"}:
            return "interrupted", "interrupt event recorded", _is_pid_alive(self.runtime_state)
        if last_event in {"fail", "exception", "crash"}:
            return "failed", "failure event recorded", _is_pid_alive(self.runtime_state)

        process_alive = _is_pid_alive(self.runtime_state)
        heartbeat_ts = self._resolve_heartbeat()
        cadence = _metric_cadence_seconds(self.metrics)
        if cadence is None:
            cadence = 2.0 if self._resolve_current_step() > 0 else 5.0
        stale_after = max(30.0, cadence * 8.0)
        heartbeat_stale = heartbeat_ts is None or (self.now_ts - heartbeat_ts) >= stale_after

        if status in ACTIVE_STATUSES:
            if process_alive is False:
                return "interrupted", "training process is no longer alive", process_alive
            if heartbeat_stale and process_alive in {False, None}:
                return "interrupted", "heartbeat is stale and process is unreachable", process_alive

        return status, None, process_alive

    def _resolve_current_step(self) -> int:
        runtime_step = _safe_int(self.runtime_state.get("current_step"))
        if runtime_step is not None:
            return max(runtime_step, 0)

        metric_steps = [_safe_int(m.get("step")) for m in self.metrics]
        metric_steps = [s for s in metric_steps if s is not None]
        if metric_steps:
            return max(metric_steps)

        return 0

    def _resolve_total_steps(self, current_step: int) -> tuple[Optional[int], str]:
        runtime_total = _safe_int(self.runtime_state.get("total_steps"))
        if runtime_total and runtime_total > 0:
            return runtime_total, "runtime"

        config = self.metadata.get("config") or {}
        explicit_total = infer_total_steps_from_config(config)
        if explicit_total and explicit_total > 0:
            return explicit_total, "config"

        epoch_total = infer_total_steps_from_epoch_progress(self.metrics, config)
        if epoch_total and epoch_total > 0:
            return epoch_total, "epoch_inference"

        # If this run is already complete and we do not have a better estimate,
        # current step is the only truthful total we can report.
        status, _, _ = self._resolve_status()
        if status == "completed" and current_step > 0:
            return current_step, "completion_fallback"
        return None, "unknown"

    def _resolve_elapsed_time(self) -> float:
        start_ts = _safe_float(self.runtime_state.get("start_time_epoch"))
        if start_ts is None:
            start_ts = _safe_float(self.metadata.get("start_time_epoch"))
        if start_ts is None:
            first_metric_ts = _first_metric_timestamp(self.metrics)
            if first_metric_ts is not None:
                start_ts = first_metric_ts

        if start_ts is None:
            return 0.0
        return max(0.0, self.now_ts - start_ts)

    def _resolve_heartbeat(self) -> Optional[float]:
        heartbeat = _safe_float(self.runtime_state.get("heartbeat_ts"))
        if heartbeat is not None:
            return heartbeat
        latest_metric_ts = _latest_metric_timestamp(self.metrics)
        if latest_metric_ts is not None:
            return latest_metric_ts
        return _safe_float(self.metadata.get("start_time_epoch"))

    def _resolve_loss_history(self) -> Series:
        for key in self.train_loss_candidates:
            series = _series_from_metrics(self.metrics, key)
            if series:
                return series

        # Generic fallback: any non-val key containing "loss"
        for key in _discover_metric_keys(self.metrics):
            low = key.lower()
            if "loss" in low and "val" not in low and "valid" not in low and "test" not in low:
                series = _series_from_metrics(self.metrics, key)
                if series:
                    return series
        return []

    def _resolve_val_loss_history(self) -> Series:
        for key in self.val_loss_candidates:
            series = _series_from_metrics(self.metrics, key)
            if series:
                return series

        # Generic fallback: val/test keys containing "loss"
        for key in _discover_metric_keys(self.metrics):
            low = key.lower()
            if ("val" in low or "valid" in low or "test" in low) and "loss" in low:
                series = _series_from_metrics(self.metrics, key)
                if series:
                    return series
        return []

    def _resolve_lr_history(self, current_step: int) -> Series:
        for key in self.lr_candidates:
            series = _series_from_metrics(self.metrics, key)
            if series:
                return series

        # Config fallback for frameworks that do not expose a dynamic LR schedule.
        config = self.metadata.get("config") or {}
        static_lr = _safe_float(config.get("lr"))
        if static_lr is None:
            static_lr = _safe_float(config.get("learning_rate"))
        if static_lr is None:
            static_lr = _safe_float(config.get("eta"))

        if static_lr is None or current_step <= 0:
            return []
        return [[1.0, static_lr], [float(current_step), static_lr]]

    def _estimate_eta(
        self, *, status: str, current_step: int, total_steps: Optional[int]
    ) -> tuple[Optional[float], Optional[str]]:
        if status == "completed":
            return 0.0, None
        if status == "failed":
            return None, "ETA unavailable (run failed)"
        if status == "cancelled":
            return None, "ETA unavailable (run cancelled)"
        if status == "interrupted":
            return None, "ETA unavailable (run interrupted)"
        if total_steps is None:
            return None, "ETA unavailable (total steps unknown)"
        if current_step <= 0:
            return None, "ETA unavailable (insufficient progress)"
        if current_step >= total_steps:
            return None, "ETA recalibrating (progress reached estimated total)"

        step_seconds = _smoothed_step_time(self.metrics)
        if step_seconds is None:
            return None, "ETA unavailable (not enough timing samples)"

        remaining_steps = max(total_steps - current_step, 0)
        return max(0.0, remaining_steps * step_seconds), None

    def _compute_health(
        self, *, status: str, heartbeat_ts: Optional[float], current_step: int, resource_tracking_available: bool
    ) -> str:
        if self.runtime_state.get("fatal_exception"):
            return "FAILED"

        if status == "failed":
            return "FAILED"

        if status == "completed":
            return "HEALTHY"
        if status == "cancelled":
            return "WARNING"
        if status == "interrupted":
            return "STALLED"

        if heartbeat_ts is None:
            return "WARNING"

        heartbeat_age = max(0.0, self.now_ts - heartbeat_ts)

        cadence = _metric_cadence_seconds(self.metrics)
        if cadence is None:
            cadence = 2.0 if current_step > 0 else 5.0

        warning_after = max(10.0, cadence * 3.0)
        stalled_after = max(30.0, cadence * 8.0)

        if heartbeat_age >= stalled_after:
            return "STALLED"
        if heartbeat_age >= warning_after:
            return "WARNING"

        # Resource telemetry is optional, but if explicitly required and unavailable
        # we mark WARNING while the run is active.
        if bool(self.runtime_state.get("resource_tracking_required")) and not resource_tracking_available:
            return "WARNING"

        return "HEALTHY"


class PyTorchExperimentAdapter(BaseExperimentAdapter):
    pass


class TensorFlowExperimentAdapter(BaseExperimentAdapter):
    lr_candidates = ("lr", "learning_rate")


_ADAPTER_REGISTRY: dict[str, type[BaseExperimentAdapter]] = {}


def register_experiment_adapter(
    frameworks: Union[str, list[str], tuple[str, ...]], adapter_cls: type[BaseExperimentAdapter]
) -> None:
    names = [frameworks] if isinstance(frameworks, str) else list(frameworks)
    for name in names:
        _ADAPTER_REGISTRY[name.lower()] = adapter_cls


def get_experiment_adapter(framework: str) -> type[BaseExperimentAdapter]:
    return _ADAPTER_REGISTRY.get((framework or "unknown").lower(), BaseExperimentAdapter)


def infer_framework_for_tracking(metadata: dict[str, Any], metrics: list[dict[str, Any]]) -> str:
    fw = str((metadata or {}).get("framework") or "").strip().lower()
    if fw == "keras":
        return "tensorflow"
    return fw or "unknown"


def infer_total_steps_from_config(config: dict[str, Any]) -> Optional[int]:
    if not isinstance(config, dict):
        return None

    direct_keys = (
        "total_steps",
        "max_steps",
        "num_steps",
        "steps",
        "max_iter",
        "num_train_steps",
        "train_steps",
        "total_train_steps",
        "max_train_steps",
    )
    for key in direct_keys:
        value = _safe_int(config.get(key))
        if value and value > 0:
            return value

    epochs = infer_total_epochs_from_config(config)
    steps_per_epoch = _safe_int(config.get("steps_per_epoch"))
    if epochs and epochs > 0 and steps_per_epoch and steps_per_epoch > 0:
        return epochs * steps_per_epoch

    return None


def infer_total_epochs_from_config(config: dict[str, Any]) -> Optional[int]:
    if not isinstance(config, dict):
        return None

    epoch_keys = ("epochs", "num_epochs", "n_epochs", "max_epochs")
    for key in epoch_keys:
        value = _safe_int(config.get(key))
        if value and value > 0:
            return value

    phase1 = _safe_int(config.get("phase1_epochs")) or 0
    phase2 = _safe_int(config.get("phase2_epochs")) or 0
    phase_total = phase1 + phase2
    if phase_total > 0:
        return phase_total

    return None


def infer_total_steps_from_epoch_progress(metrics: list[dict[str, Any]], config: dict[str, Any]) -> Optional[int]:
    if not metrics or not isinstance(config, dict):
        return None

    total_epochs = infer_total_epochs_from_config(config)
    if total_epochs is None or total_epochs <= 0:
        return None

    max_step_by_epoch: dict[int, int] = {}
    for metric in metrics:
        step = _safe_int(metric.get("step"))
        epoch = _metric_epoch_value(metric)
        if step is None or epoch is None:
            continue
        max_step_by_epoch[epoch] = max(step, max_step_by_epoch.get(epoch, step))

    if len(max_step_by_epoch) < 2:
        return None

    ordered_epoch_max = sorted(max_step_by_epoch.items(), key=lambda pair: pair[0])
    boundary_deltas: list[int] = []
    for idx in range(1, len(ordered_epoch_max)):
        prev_step = ordered_epoch_max[idx - 1][1]
        curr_step = ordered_epoch_max[idx][1]
        delta = curr_step - prev_step
        if delta > 0:
            boundary_deltas.append(delta)

    if not boundary_deltas:
        return None

    estimated_steps_per_epoch = int(round(statistics.median(boundary_deltas)))
    if estimated_steps_per_epoch <= 0:
        return None

    inferred_total = estimated_steps_per_epoch * total_epochs
    return inferred_total if inferred_total > 0 else None


def build_overview_snapshot(
    *,
    run_id: str,
    metadata: dict[str, Any],
    metrics: list[dict[str, Any]],
    runtime_state: Optional[dict[str, Any]],
    now_ts: Optional[float] = None,
) -> dict[str, Any]:
    framework = infer_framework_for_tracking(metadata, metrics)
    adapter_cls = get_experiment_adapter(framework)
    adapter = adapter_cls(
        run_id=run_id,
        framework=framework,
        metadata=metadata,
        metrics=metrics,
        runtime_state=runtime_state,
        now_ts=now_ts,
    )
    return adapter.build_snapshot()


def normalize_run_status(status: Any) -> str:
    """Public status normalizer used by APIs/UI-facing payloads."""
    return _normalize_run_status(status)


def _normalize_run_status(status: Any) -> str:
    raw = str(status if status is not None else "").strip().lower()
    if raw in RUN_STATUS_ALIASES:
        return RUN_STATUS_ALIASES[raw]
    return raw or "idle"


def _is_pid_alive(runtime_state: dict[str, Any]) -> Optional[bool]:
    pid = _safe_int(runtime_state.get("training_pid"))
    if pid is None or pid <= 0:
        return None

    expected_start = _safe_float(runtime_state.get("training_process_start_time"))

    try:
        import psutil  # type: ignore

        proc = psutil.Process(pid)
        if expected_start is not None:
            try:
                if abs(float(proc.create_time()) - expected_start) > 2.0:
                    return False
            except Exception:
                pass
        return bool(proc.is_running()) and proc.status() != getattr(psutil, "STATUS_ZOMBIE", "zombie")
    except ImportError:
        pass
    except Exception:
        return False

    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return None


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f != f:
        return None
    return f


def _safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _series_from_metrics(metrics: list[dict[str, Any]], key: str) -> Series:
    out: Series = []
    for m in metrics:
        step = _safe_int(m.get("step"))
        value = _safe_float(m.get(key))
        if step is None or value is None:
            continue
        out.append([float(step), value])
    return out


def _discover_metric_keys(metrics: list[dict[str, Any]]) -> list[str]:
    keys: set[str] = set()
    for m in metrics:
        keys.update((k for k in m.keys() if k not in {"step", "timestamp"}))
    return sorted(keys)


def _latest_metric_timestamp(metrics: list[dict[str, Any]]) -> Optional[float]:
    for m in reversed(metrics):
        ts = _safe_float(m.get("timestamp"))
        if ts is not None:
            return ts
    return None


def _first_metric_timestamp(metrics: list[dict[str, Any]]) -> Optional[float]:
    for m in metrics:
        ts = _safe_float(m.get("timestamp"))
        if ts is not None:
            return ts
    return None


def _metric_cadence_seconds(metrics: list[dict[str, Any]]) -> Optional[float]:
    times = [_safe_float(m.get("timestamp")) for m in metrics]
    times = [t for t in times if t is not None]
    if len(times) < 3:
        return None

    deltas = []
    for idx in range(1, len(times)):
        dt = times[idx] - times[idx - 1]
        if dt > 0:
            deltas.append(dt)

    if len(deltas) < 2:
        return None

    recent = deltas[-20:]
    return float(statistics.median(recent)) if recent else None


def _smoothed_step_time(metrics: list[dict[str, Any]], alpha: float = 0.25, window: int = 80) -> Optional[float]:
    if len(metrics) < 2:
        return None

    # Robust to uneven logging: use seconds-per-step from each positive delta.
    points: list[tuple[int, float]] = []
    for m in metrics:
        step = _safe_int(m.get("step"))
        ts = _safe_float(m.get("timestamp"))
        if step is None or ts is None:
            continue
        points.append((step, ts))

    if len(points) < 2:
        return None

    samples: list[float] = []
    for idx in range(1, len(points)):
        prev_step, prev_ts = points[idx - 1]
        step, ts = points[idx]
        d_step = step - prev_step
        d_ts = ts - prev_ts
        if d_step <= 0 or d_ts <= 0:
            continue
        samples.append(d_ts / d_step)

    if len(samples) < 2:
        return None

    recent_samples = samples[-window:] if window > 0 else samples
    filtered_samples = _filter_outlier_samples(recent_samples)

    ewma: Optional[float] = None
    for sample in filtered_samples:
        if ewma is None:
            ewma = sample
        else:
            ewma = alpha * sample + (1.0 - alpha) * ewma

    return ewma


def _filter_outlier_samples(samples: list[float]) -> list[float]:
    if len(samples) < 5:
        return samples

    median = statistics.median(samples)
    abs_dev = [abs(sample - median) for sample in samples]
    mad = statistics.median(abs_dev)

    if mad > 0:
        # Modified z-score style filtering with a wide threshold to keep
        # realistic bursts while dropping major timing spikes.
        scale = 1.4826 * mad
        filtered = [sample for sample in samples if abs(sample - median) <= 6.0 * scale]
    else:
        lower = max(0.0, median * 0.25)
        upper = median * 4.0 if median > 0 else float("inf")
        filtered = [sample for sample in samples if lower <= sample <= upper]

    return filtered if len(filtered) >= 2 else samples


def _metric_epoch_value(metric: dict[str, Any]) -> Optional[int]:
    for key in ("epoch", "epoch_idx", "epoch_end"):
        value = _safe_int(metric.get(key))
        if value is not None:
            return value
    return None


register_experiment_adapter("pytorch", PyTorchExperimentAdapter)
register_experiment_adapter("tensorflow", TensorFlowExperimentAdapter)
register_experiment_adapter("keras", TensorFlowExperimentAdapter)
