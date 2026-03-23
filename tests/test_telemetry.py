from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from typing import Optional

import pytest

from gradglass.artifacts import ArtifactStore
import gradglass.telemetry as telemetry


@pytest.fixture
def tmp_store():
    tmpdir = tempfile.mkdtemp()
    store = ArtifactStore(root=tmpdir)
    yield store
    shutil.rmtree(tmpdir)


def _seed_run(
    store: ArtifactStore,
    run_id: str,
    *,
    heartbeat_ts: Optional[float] = None,
    status: str = "running",
) -> None:
    run_dir = store.ensure_run_dir(run_id)
    metadata = {
        "name": "telemetry-run",
        "run_id": run_id,
        "framework": "pytorch",
        "status": status,
        "config": {"monitor": True},
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata))
    runtime_state = {"status": status, "heartbeat_ts": float(time.time() if heartbeat_ts is None else heartbeat_ts)}
    (run_dir / "runtime_state.json").write_text(json.dumps(runtime_state))


class _FakeProcess:
    @staticmethod
    def memory_percent():
        return 18.0

    @staticmethod
    def memory_info():
        class _Mem:
            rss = 512 * 1024 * 1024

        return _Mem()

    @staticmethod
    def cpu_percent(interval=None):
        return 12.5


class _FakePsutil:
    _disk_read_bytes = 4 * 1024 * 1024
    _disk_write_bytes = 2 * 1024 * 1024
    _net_recv_bytes = 3 * 1024 * 1024
    _net_sent_bytes = 1 * 1024 * 1024

    @staticmethod
    def cpu_percent(interval=None):
        return 37.5

    @staticmethod
    def Process():
        return _FakeProcess()

    @staticmethod
    def cpu_count(logical=True):
        return 12 if logical else 6

    @staticmethod
    def virtual_memory():
        gib = 1024**3

        class _VM:
            used = 8 * gib
            total = 16 * gib
            percent = 50.0

        return _VM()

    @classmethod
    def disk_io_counters(cls):
        class _Disk:
            read_bytes = cls._disk_read_bytes
            write_bytes = cls._disk_write_bytes

        cls._disk_read_bytes += 512 * 1024
        cls._disk_write_bytes += 256 * 1024
        return _Disk()

    @classmethod
    def net_io_counters(cls):
        class _Net:
            bytes_recv = cls._net_recv_bytes
            bytes_sent = cls._net_sent_bytes

        cls._net_recv_bytes += 768 * 1024
        cls._net_sent_bytes += 128 * 1024
        return _Net()


class _FakeTrainingProcess:
    def __init__(self, create_time: float = 1234.0):
        self._create_time = create_time
        self._cpu_calls = 0

    def create_time(self):
        return self._create_time

    def is_running(self):
        return True

    def status(self):
        return "running"

    def cpu_percent(self, interval=None):
        self._cpu_calls += 1
        return 0.0 if self._cpu_calls == 1 else 41.75

    def memory_percent(self):
        return 17.5

    def memory_info(self):
        class _Mem:
            rss = 256 * 1024 * 1024

        return _Mem()


class _FakeTrainingPsutil:
    STATUS_ZOMBIE = "zombie"
    _processes: dict[int, _FakeTrainingProcess] = {}

    @classmethod
    def Process(cls, pid):
        if pid not in cls._processes:
            cls._processes[pid] = _FakeTrainingProcess()
        return cls._processes[pid]


def test_collect_infrastructure_cpu_ram_live_without_gpu(tmp_store, monkeypatch):
    run_id = "cpu-only"
    _seed_run(tmp_store, run_id)
    monkeypatch.setattr(telemetry, "_import_psutil", lambda: _FakePsutil())
    monkeypatch.setattr(telemetry, "_get_nvml_module", lambda: (None, "pynvml is not installed"))

    payload = telemetry.collect_infrastructure_telemetry(tmp_store, run_id)

    assert payload["run_id"] == run_id
    assert payload["mode"] == "standalone"
    assert payload["live_guard"]["ok"] is True
    assert isinstance(payload["collected_at"], float)
    assert payload["gpu_devices"] == []
    for metric in payload["metrics"].values():
        assert telemetry.REQUIRED_METRIC_KEYS.issubset(metric.keys())

    cpu = payload["metrics"]["system_cpu"]
    assert cpu["status"] == "live"
    assert cpu["value"]["host_percent"] == 37.5
    assert cpu["value"]["process_percent"] == 12.5

    ram = payload["metrics"]["system_ram"]
    assert ram["status"] == "live"
    assert ram["value"]["used_percent"] == 50.0

    for key in ("power_draw", "multi_gpu_compute_utilization", "gpu_memory_fragmentation"):
        assert payload["metrics"][key]["status"] == "unavailable"
        assert "pynvml" in (payload["metrics"][key]["error"] or "")

    v2 = payload["telemetry_v2"]
    aggregate = v2["aggregate_accelerator"]["metrics"]
    external = v2["external_usage"]

    assert aggregate["memory_pressure_percent"]["status"] == "not_detected"
    assert aggregate["memory_pressure_percent"]["value"] is None
    assert aggregate["power_watts"]["status"] == "not_supported"
    assert aggregate["power_watts"]["value"] is None
    assert aggregate["temperature_c"]["status"] == "not_supported"
    assert aggregate["temperature_c"]["value"] is None
    assert external["process_cpu_percent"]["value"] is None
    assert external["process_cpu_percent"]["status"] == "not_detected"


def test_cpu_and_ram_probes_are_unavailable_without_psutil(monkeypatch):
    monkeypatch.setattr(telemetry, "_import_psutil", lambda: None)

    cpu = telemetry.query_system_cpu()
    ram = telemetry.query_system_ram()

    assert cpu["status"] == "unavailable"
    assert "psutil" in (cpu["error"] or "")
    assert ram["status"] == "unavailable"
    assert "psutil" in (ram["error"] or "")


def test_cluster_nodes_mode_comes_from_distributed_index(tmp_store, monkeypatch):
    run_id = "distributed-run"
    _seed_run(tmp_store, run_id)
    run_dir = tmp_store.get_run_dir(run_id)
    (run_dir / "distributed_index.json").write_text(json.dumps({"world_size": 4, "active_nodes": 3}))

    monkeypatch.setattr(telemetry, "_import_psutil", lambda: _FakePsutil())
    monkeypatch.setattr(telemetry, "_get_nvml_module", lambda: (None, "pynvml is not installed"))

    cluster = telemetry.query_cluster_nodes(tmp_store, run_id)
    assert cluster["status"] == "live"
    assert cluster["value"]["total_nodes"] == 4
    assert cluster["value"]["active_nodes"] == 3
    assert cluster["value"]["mode"] == "distributed"
    assert cluster["label"] == "3 / 4"

    payload = telemetry.collect_infrastructure_telemetry(tmp_store, run_id)
    assert payload["mode"] == "standalone"
    assert payload["live_guard"]["ok"] is False
    assert "missing_rank_artifacts" in payload["live_guard"]["reasons"]


def test_live_guard_forces_standalone_when_distributed_artifacts_are_stale(tmp_store, monkeypatch):
    now = 2_000_000.0
    run_id = "stale-distributed-run"
    _seed_run(tmp_store, run_id, heartbeat_ts=now - 1_000.0)
    run_dir = tmp_store.get_run_dir(run_id)

    distributed_index = run_dir / "distributed_index.json"
    distributed_index.write_text(json.dumps({"world_size": 4, "active_nodes": 4}))
    os.utime(distributed_index, (now - 2_000.0, now - 2_000.0))

    (run_dir / "rank_0").mkdir(exist_ok=True)
    (run_dir / "rank_1").mkdir(exist_ok=True)
    os.utime(run_dir / "rank_0", (now - 2_000.0, now - 2_000.0))
    os.utime(run_dir / "rank_1", (now - 2_000.0, now - 2_000.0))
    os.utime(run_dir / "runtime_state.json", (now - 2_000.0, now - 2_000.0))

    monkeypatch.setattr(telemetry.time, "time", lambda: now)
    monkeypatch.setattr(telemetry, "_import_psutil", lambda: _FakePsutil())
    monkeypatch.setattr(telemetry, "_get_nvml_module", lambda: (None, "pynvml is not installed"))

    payload = telemetry.collect_infrastructure_telemetry(tmp_store, run_id)

    assert payload["mode"] == "standalone"
    assert payload["live_guard"]["ok"] is False
    assert "stale_distributed_artifacts_ignored" in payload["live_guard"]["reasons"]
    assert "stale_runtime_state" in payload["live_guard"]["reasons"]
    assert payload["metrics"]["cluster_nodes"]["value"]["mode"] == "standalone"
    assert payload["metrics"]["cluster_nodes"]["value"]["guarded"] is True


def test_live_guard_keeps_distributed_when_evidence_is_fresh(tmp_store, monkeypatch):
    now = 2_000_000.0
    run_id = "fresh-distributed-run"
    _seed_run(tmp_store, run_id, heartbeat_ts=now - 10.0)
    run_dir = tmp_store.get_run_dir(run_id)

    distributed_index = run_dir / "distributed_index.json"
    distributed_index.write_text(json.dumps({"world_size": 4, "active_nodes": 3}))
    os.utime(distributed_index, (now - 5.0, now - 5.0))

    (run_dir / "rank_0").mkdir(exist_ok=True)
    (run_dir / "rank_1").mkdir(exist_ok=True)
    os.utime(run_dir / "rank_0", (now - 5.0, now - 5.0))
    os.utime(run_dir / "rank_1", (now - 5.0, now - 5.0))
    os.utime(run_dir / "runtime_state.json", (now - 5.0, now - 5.0))

    monkeypatch.setattr(telemetry.time, "time", lambda: now)
    monkeypatch.setattr(telemetry, "_import_psutil", lambda: _FakePsutil())
    monkeypatch.setattr(telemetry, "_get_nvml_module", lambda: (None, "pynvml is not installed"))

    payload = telemetry.collect_infrastructure_telemetry(tmp_store, run_id)

    assert payload["mode"] == "distributed"
    assert payload["live_guard"]["ok"] is True
    assert payload["live_guard"]["reasons"] == []


def test_telemetry_v2_coexists_with_legacy_payload(tmp_store, monkeypatch):
    run_id = "telemetry-v2-run"
    _seed_run(tmp_store, run_id)
    monkeypatch.setattr(telemetry, "_import_psutil", lambda: _FakePsutil())
    monkeypatch.setattr(telemetry, "_get_nvml_module", lambda: (None, "pynvml is not installed"))

    payload = telemetry.collect_infrastructure_telemetry(tmp_store, run_id)

    legacy_keys = {
        "cluster_nodes",
        "system_cpu",
        "system_ram",
        "power_draw",
        "multi_gpu_compute_utilization",
        "gpu_memory_fragmentation",
    }
    assert set(payload["metrics"].keys()) == legacy_keys
    assert "telemetry_v2" in payload

    v2 = payload["telemetry_v2"]
    assert v2["panel_mode"] == "local_insights"
    assert v2["accelerator_mode"] in {"cpu_only", "accelerator_unavailable"}
    assert isinstance(v2["host_metrics"], dict)
    assert isinstance(v2["training_process_metrics"], dict)
    assert isinstance(v2["external_usage"], dict)
    assert isinstance(v2["graph_hints"], dict)
    assert isinstance(v2["local_performance_insights"], dict)
    assert isinstance(v2["diagnostics"], list)
    assert "memory_fragmentation_percent" in v2["aggregate_accelerator"]["metrics"]
    assert "metric_status_legend" in v2
    assert "cluster_metrics" in v2
    assert v2["cluster_metrics"] is None
    assert any(diag.get("status") == "requires_cluster_connection" for diag in v2["diagnostics"])
    assert v2["graph_hints"]["preferred_layout"] in {"host_process_first", "accelerator_first"}
    assert v2["graph_hints"]["run_terminal"] is False
    assert v2["graph_hints"]["throughput_warmup"] is True

    ext = v2["external_usage"]
    assert ext["host_cpu_percent"]["value"] == v2["host_metrics"]["system_cpu_percent"]["value"]
    assert ext["host_ram_percent"]["value"] == v2["host_metrics"]["system_ram_percent"]["value"]
    assert ext["disk_read_mb_s"]["value"] == 0.0
    assert ext["net_rx_mb_s"]["value"] == 0.0
    assert ext["disk_read_mb_s"]["details"]["warmup"] is True
    assert ext["net_rx_mb_s"]["details"]["warmup"] is True


def test_telemetry_v2_run_status_is_normalized(tmp_store, monkeypatch):
    run_id = "telemetry-v2-complete"
    _seed_run(tmp_store, run_id, status="complete")
    monkeypatch.setattr(telemetry, "_import_psutil", lambda: _FakePsutil())
    monkeypatch.setattr(telemetry, "_get_nvml_module", lambda: (None, "pynvml is not installed"))

    payload = telemetry.collect_infrastructure_telemetry(tmp_store, run_id)
    assert payload["telemetry_v2"]["run_state"]["status"] == "completed"
    assert payload["telemetry_v2"]["graph_hints"]["run_terminal"] is True


def test_training_process_cpu_warmup_then_active(monkeypatch):
    telemetry._PROCESS_CPU_SAMPLE_CACHE.clear()
    _FakeTrainingPsutil._processes = {}
    monkeypatch.setattr(telemetry, "_import_psutil", lambda: _FakeTrainingPsutil)

    runtime_state = {"training_pid": 4321, "training_process_start_time": 1234.0}

    first = telemetry._collect_training_process_metrics(
        runtime_state=runtime_state,
        collected_at=1.0,
        host_cpu_metric={"value": 80.0},
    )
    first_cpu = first["process_cpu_percent"]
    assert first_cpu["status"] == "not_detected"
    assert first_cpu["value"] is None
    assert first_cpu["details"]["warmup"] is True

    second = telemetry._collect_training_process_metrics(
        runtime_state=runtime_state,
        collected_at=3.0,
        host_cpu_metric={"value": 80.0},
    )
    second_cpu = second["process_cpu_percent"]
    assert second_cpu["status"] == "active"
    assert second_cpu["value"] == 41.75
    assert second_cpu["details"]["warmup"] is False
    assert second_cpu["details"]["cpu_share_of_host_percent"] == 52.19


def test_local_insights_samples_per_sec_uses_non_zero_training_metric(tmp_store, monkeypatch):
    run_id = "samples-per-sec-run"
    _seed_run(tmp_store, run_id)
    run_dir = tmp_store.get_run_dir(run_id)
    with open(run_dir / "metrics.jsonl", "w") as f:
        f.write(json.dumps({"step": 1, "timestamp": 100.0, "samples_per_sec": 256.5}) + "\n")

    monkeypatch.setattr(telemetry, "_import_psutil", lambda: _FakePsutil())
    monkeypatch.setattr(telemetry, "_get_nvml_module", lambda: (None, "pynvml is not installed"))

    payload = telemetry.collect_infrastructure_telemetry(tmp_store, run_id)
    local = payload["telemetry_v2"]["local_performance_insights"]
    samples_metric = local["samples_per_sec"]
    loading_metric = local["batch_loading_speed"]

    assert samples_metric["status"] == "active"
    assert samples_metric["value"] == 256.5
    assert loading_metric["status"] == "active"
    assert loading_metric["value"] == 256.5


def test_collect_infrastructure_survives_malformed_runtime_state(tmp_store, monkeypatch):
    run_id = "malformed-runtime-state"
    _seed_run(tmp_store, run_id)
    run_dir = tmp_store.get_run_dir(run_id)
    (run_dir / "runtime_state.json").write_text("")

    monkeypatch.setattr(telemetry, "_import_psutil", lambda: _FakePsutil())
    monkeypatch.setattr(telemetry, "_get_nvml_module", lambda: (None, "pynvml is not installed"))

    payload = telemetry.collect_infrastructure_telemetry(tmp_store, run_id)
    assert payload["run_id"] == run_id
    assert payload["mode"] in {"standalone", "distributed"}
    assert "telemetry_v2" in payload
    assert payload["telemetry_v2"]["run_state"]["status"] in {
        "running",
        "starting",
        "paused",
        "idle",
        "completed",
        "failed",
        "interrupted",
        "cancelled",
    }
