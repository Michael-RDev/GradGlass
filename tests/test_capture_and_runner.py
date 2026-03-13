from __future__ import annotations

import json
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from gradglass.artifacts import ArtifactStore
from gradglass.capture import CaptureEngine
from gradglass.run import Run, json_default
from gradglass.core import GradGlass
from gradglass.analysis.runner import AnalysisRunner
from gradglass.analysis.report import PostRunReport
from gradglass.analysis.registry import (
    TestCategory,
    TestContext,
    TestRegistry,
    TestResult,
    TestSeverity,
    TestStatus,
)
from gradglass.diff import (
    full_diff,
    gradient_flow_analysis,
    weight_diff,
)


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class TinyConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(4 * 4 * 4, 10)  # assuming 8x8 input

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        return self.fc1(x)


class ModelWithFrozenLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.frozen = nn.Linear(4, 4)
        self.trainable = nn.Linear(4, 2)
        # Freeze the first layer
        for p in self.frozen.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.trainable(F.relu(self.frozen(x)))


def _load_runtime_state(run: Run) -> dict:
    path = run.run_dir / "runtime_state.json"
    assert path.exists()
    return json.loads(path.read_text())


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def tmp_store(tmp_dir):
    return ArtifactStore(root=tmp_dir)


@pytest.fixture
def model():
    return TinyMLP()


@pytest.fixture
def optimizer(model):
    return torch.optim.SGD(model.parameters(), lr=0.01)


@pytest.fixture
def run_dir(tmp_store):
    return tmp_store.ensure_run_dir("test-run-001")


@pytest.fixture
def default_config():
    return {
        "activations": "auto",
        "gradients": "summary",
        "layers": "trainable",
        "sample_batches": 2,
        "every": 5,
    }


@pytest.fixture
def engine(model, optimizer, run_dir, default_config):
    eng = CaptureEngine(
        model=model,
        optimizer=optimizer,
        framework="pytorch",
        run_dir=run_dir,
        config=default_config,
    )
    yield eng
    eng.cleanup()


@pytest.fixture
def populated_run(tmp_store, model, optimizer):
    run = Run(name="integ", store=tmp_store, auto_open=False)
    run.watch(model, optimizer, every=2, sample_batches=1)

    x = torch.randn(8, 4)
    y = torch.randint(0, 2, (8,))
    for epoch in range(3):
        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        run.log(loss=loss.item(), acc=0.5 + epoch * 0.1, epoch=epoch)
        run.log_batch(x=x, y=y, y_pred=logits, loss=loss)

    run.checkpoint(step=run.step, tag="mid")
    # another round
    for _ in range(3):
        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        run.log(loss=loss.item(), acc=0.75)

    run.checkpoint(step=run.step, tag="end")
    run.finish(open=False, analyze=False, print_summary=False)
    return run, tmp_store


class TestCaptureEngineInit:
    def test_directories_created(self, run_dir, model, optimizer, default_config):
        eng = CaptureEngine(model, optimizer, "pytorch", run_dir, default_config)
        for sub in ("checkpoints", "gradients", "activations", "predictions"):
            assert (run_dir / sub).is_dir()
        eng.cleanup()

    def test_writer_thread_starts(self, engine):
        assert engine.writer_thread.is_alive()

    def test_buffers_initially_empty(self, engine):
        assert engine.activation_buffer == {}
        assert engine.gradient_buffer == {}
        assert engine.prediction_buffer == []


class TestArchitectureExtraction:
    def test_extracts_layers_and_edges(self, engine):
        structure = engine.extract_architecture()
        layer_ids = [l["id"] for l in structure["layers"]]
        assert "fc1" in layer_ids
        assert "fc2" in layer_ids
        assert len(structure["edges"]) == len(structure["layers"]) - 1

    def test_saves_model_structure_json(self, engine):
        engine.extract_architecture()
        path = engine.run_dir / "model_structure.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert "layers" in data and "edges" in data

    def test_param_counts_correct(self, engine):
        structure = engine.extract_architecture()
        fc1 = next(l for l in structure["layers"] if l["id"] == "fc1")
        # fc1: 4*8 weight + 8 bias = 40
        assert fc1["param_count"] == 40
        assert fc1["trainable"] is True

    def test_convnet_architecture(self, run_dir, default_config):
        m = TinyConvNet()
        eng = CaptureEngine(m, None, "pytorch", run_dir, default_config)
        structure = eng.extract_architecture()
        types = {l["id"]: l["type"] for l in structure["layers"]}
        assert types["conv1"] == "Conv2d"
        assert types["pool"] == "MaxPool2d"
        assert types["fc1"] == "Linear"
        eng.cleanup()

    def test_unknown_framework_returns_empty(self, run_dir, default_config, model):
        eng = CaptureEngine(model, None, "jax", run_dir, default_config)
        structure = eng.extract_architecture()
        assert structure == {"layers": [], "edges": []}
        eng.cleanup()


class TestHookAttachment:
    def test_hooks_registered(self, engine):
        engine.extract_architecture()
        engine.attach_hooks()
        # fc1 + fc2 have trainable params → forward hooks + grad hooks
        assert len(engine.hooks) > 0

    def test_forward_hook_captures_activations(self, engine):
        engine.extract_architecture()
        engine.attach_hooks()

        x = torch.randn(4, 4)
        _ = engine.model(x)

        # At least one layer should have activations buffered
        assert len(engine.activation_buffer) > 0
        for key, acts in engine.activation_buffer.items():
            assert len(acts) > 0
            assert isinstance(acts[0], np.ndarray)

    def test_grad_hook_captures_gradients(self, engine, optimizer):
        engine.extract_architecture()
        engine.attach_hooks()

        x = torch.randn(4, 4)
        y = torch.randint(0, 2, (4,))
        logits = engine.model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()

        assert len(engine.gradient_buffer) > 0
        for name, grads in engine.gradient_buffer.items():
            assert len(grads) > 0
            g = grads[0]
            assert "mean" in g and "var" in g and "norm" in g

    def test_sample_batches_limit(self, run_dir, model, optimizer):
        config = {
            "activations": "auto",
            "gradients": "summary",
            "layers": "trainable",
            "sample_batches": 2,
            "every": 5,
        }
        eng = CaptureEngine(model, optimizer, "pytorch", run_dir, config)
        eng.extract_architecture()
        eng.attach_hooks()

        for _ in range(5):
            _ = model(torch.randn(4, 4))

        # Should cap at sample_batches=2
        for acts in eng.activation_buffer.values():
            assert len(acts) <= 2
        eng.cleanup()

    def test_layers_config_list(self, run_dir, model, optimizer):
        config = {
            "activations": "auto",
            "gradients": "summary",
            "layers": ["fc1"],
            "sample_batches": 2,
            "every": 5,
        }
        eng = CaptureEngine(model, optimizer, "pytorch", run_dir, config)
        eng.extract_architecture()
        eng.attach_hooks()

        x = torch.randn(4, 4)
        _ = model(x)

        # Only fc1 should appear in activation buffer
        for key in eng.activation_buffer:
            assert "fc1" in key
        eng.cleanup()

    def test_frozen_layer_no_grad_hook(self, run_dir):
        m = ModelWithFrozenLayer()
        opt = torch.optim.SGD(m.parameters(), lr=0.01)
        config = {
            "activations": "auto",
            "gradients": "summary",
            "layers": "trainable",
            "sample_batches": 2,
            "every": 5,
        }
        eng = CaptureEngine(m, opt, "pytorch", run_dir, config)
        eng.extract_architecture()
        eng.attach_hooks()

        x = torch.randn(4, 4)
        y = torch.randint(0, 2, (4,))
        logits = m(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()

        # gradient buffer should NOT contain frozen layer params
        for name in eng.gradient_buffer:
            assert "frozen" not in name
        eng.cleanup()


class TestCheckpointSaving:
    def test_checkpoint_creates_files(self, engine):
        engine.extract_architecture()

        engine.save_checkpoint(step=10, tag="test")
        engine.flush_writes()

        ckpt_dir = engine.run_dir / "checkpoints"
        assert (ckpt_dir / "step_10.npz").exists()
        assert (ckpt_dir / "step_10_meta.json").exists()

    def test_checkpoint_meta_contents(self, engine):
        engine.save_checkpoint(step=42, tag="epoch5")
        engine.flush_writes()

        meta_path = engine.run_dir / "checkpoints" / "step_42_meta.json"
        meta = json.loads(meta_path.read_text())
        assert meta["step"] == 42
        assert meta["tag"] == "epoch5"
        assert "timestamp" in meta
        assert "num_params" in meta

    def test_checkpoint_weights_loadable(self, engine, tmp_store):
        engine.save_checkpoint(step=1)
        engine.flush_writes()

        loaded = tmp_store.load_checkpoint("test-run-001", 1)
        # TinyMLP has fc1.weight, fc1.bias, fc2.weight, fc2.bias
        assert "fc1.weight" in loaded
        assert loaded["fc1.weight"].shape == (8, 4)

    def test_multiple_checkpoints(self, engine):
        engine.save_checkpoint(step=1)
        engine.save_checkpoint(step=2)
        engine.save_checkpoint(step=3)
        engine.flush_writes()

        ckpt_dir = engine.run_dir / "checkpoints"
        assert len(list(ckpt_dir.glob("step_*.npz"))) == 3


class TestGradientCapture:
    def test_capture_gradients_writes_file(self, engine, optimizer):
        engine.extract_architecture()
        engine.attach_hooks()

        x = torch.randn(4, 4)
        y = torch.randint(0, 2, (4,))
        logits = engine.model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()

        engine.capture_gradients(step=100)

        grad_path = engine.run_dir / "gradients" / "summaries_step_100.json"
        assert grad_path.exists()
        data = json.loads(grad_path.read_text())
        assert len(data) > 0
        for param, summary in data.items():
            assert "mean" in summary
            assert "var" in summary
            assert "norm" in summary

    def test_empty_buffer_does_nothing(self, engine):
        engine.gradient_buffer.clear()
        engine.capture_gradients(step=999)
        grad_path = engine.run_dir / "gradients" / "summaries_step_999.json"
        assert not grad_path.exists()

    def test_kl_divergence_computed(self, engine, optimizer):
        engine.extract_architecture()
        engine.attach_hooks()

        # First backward
        x = torch.randn(4, 4)
        y = torch.randint(0, 2, (4,))
        logits = engine.model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        engine.capture_gradients(step=50)

        # Second backward (different data so distributions differ)
        optimizer.zero_grad()
        x2 = torch.randn(4, 4) * 5
        logits2 = engine.model(x2)
        loss2 = F.cross_entropy(logits2, y)
        loss2.backward()
        engine.capture_gradients(step=100)

        data = json.loads((engine.run_dir / "gradients" / "summaries_step_100.json").read_text())
        # At least some params should have kl_div
        has_kl = any("kl_div" in v for v in data.values())
        assert has_kl


class TestActivationFlushing:
    def test_activations_saved_as_npy(self, engine):
        engine.extract_architecture()
        engine.attach_hooks()

        _ = engine.model(torch.randn(4, 4))
        engine.flush_activations(step=10)

        act_dir = engine.run_dir / "activations"
        files = list(act_dir.glob("*_step_10*"))
        assert len(files) > 0

    def test_buffer_cleared_after_flush(self, engine):
        engine.extract_architecture()
        engine.attach_hooks()
        _ = engine.model(torch.randn(4, 4))

        engine.flush_activations(step=1)
        assert engine.activation_buffer == {}


class TestBatchPredictions:
    def test_predictions_saved(self, engine):
        x = torch.randn(8, 4)
        y = torch.randint(0, 2, (8,))
        logits = engine.model(x)

        engine.log_batch_predictions(step=5, x=x, y=y, y_pred=logits, loss=0.5)

        path = engine.run_dir / "predictions" / "probe_step_5.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["step"] == 5
        assert "y_true" in data
        assert "y_pred" in data

    def test_handles_none_y(self, engine):
        x = torch.randn(4, 4)
        logits = engine.model(x)
        engine.log_batch_predictions(step=1, x=x, y=None, y_pred=logits, loss=None)
        path = engine.run_dir / "predictions" / "probe_step_1.json"
        assert path.exists()


class TestCleanup:
    def test_cleanup_removes_hooks(self, model, optimizer, run_dir, default_config):
        eng = CaptureEngine(model, optimizer, "pytorch", run_dir, default_config)
        eng.extract_architecture()
        eng.attach_hooks()
        assert len(eng.hooks) > 0

        eng.cleanup()
        assert len(eng.hooks) == 0
        assert not eng.running

    def test_writer_thread_stops(self, model, optimizer, run_dir, default_config):
        eng = CaptureEngine(model, optimizer, "pytorch", run_dir, default_config)
        eng.cleanup()
        time.sleep(0.5)
        assert not eng.writer_thread.is_alive()


class TestRunInit:
    def test_run_id_format(self, tmp_store):
        run = Run(name="demo", store=tmp_store)
        assert run.run_id.startswith("demo-")
        assert run.step == 0

    def test_metadata_written(self, tmp_store):
        run = Run(name="meta-test", store=tmp_store)
        meta_path = run.run_dir / "metadata.json"
        assert meta_path.exists()
        data = json.loads(meta_path.read_text())
        assert data["name"] == "meta-test"
        assert data["status"] == "running"

    def test_subdirs_created(self, tmp_store):
        run = Run(name="dir-test", store=tmp_store)
        for sub in ("checkpoints", "gradients", "activations", "predictions", "slices"):
            assert (run.run_dir / sub).is_dir()


class TestRunWatch:
    def test_framework_detected(self, tmp_store, model, optimizer):
        run = Run(name="fw", store=tmp_store)
        run.watch(model, optimizer)
        assert run.framework == "pytorch"
        run.finish(open=False, analyze=False)

    def test_architecture_file_created(self, tmp_store, model, optimizer):
        run = Run(name="arch", store=tmp_store)
        run.watch(model, optimizer)
        assert (run.run_dir / "model_structure.json").exists()
        run.finish(open=False, analyze=False)

    def test_watch_returns_self(self, tmp_store, model, optimizer):
        run = Run(name="chain", store=tmp_store)
        ret = run.watch(model, optimizer)
        assert ret is run
        run.finish(open=False, analyze=False)

    def test_watch_honors_monitor_option_from_run_config(self, tmp_store, model, optimizer, monkeypatch):
        run = Run(name="monitor-opt", store=tmp_store, monitor=True, port=9876)
        called = {}

        def fake_monitor(port=0, open_browser=True):
            called["port"] = port
            called["open_browser"] = open_browser
            return 9876

        monkeypatch.setattr(run, "monitor", fake_monitor)
        run.watch(model, optimizer)
        assert called["port"] == 9876
        assert called["open_browser"] is True
        assert _load_runtime_state(run)["monitor_enabled"] is True
        run.finish(open=False, analyze=False)

    def test_watch_explicit_monitor_override(self, tmp_store, model, optimizer, monkeypatch):
        run = Run(name="monitor-off", store=tmp_store, monitor=True, port=9876)
        called = {"count": 0}

        def fake_monitor(port=0, open_browser=True):
            called["count"] += 1
            return 9876

        monkeypatch.setattr(run, "monitor", fake_monitor)
        run.watch(model, optimizer, monitor=False)
        assert called["count"] == 0
        assert _load_runtime_state(run)["monitor_enabled"] is False
        run.finish(open=False, analyze=False)

    def test_watch_infers_total_steps_from_model_attributes(self, tmp_store, model, optimizer):
        model.n_estimators = 123
        run = Run(name="watch-model-steps", store=tmp_store)
        run.watch(model, optimizer, monitor=False)
        state = _load_runtime_state(run)
        assert state["total_steps"] == 123
        run.finish(open=False, analyze=False)

    def test_watch_prefers_config_total_steps_over_model_attributes(self, tmp_store, model, optimizer):
        model.n_estimators = 999
        run = Run(name="watch-config-steps", store=tmp_store, total_steps=42)
        run.watch(model, optimizer, monitor=False)
        state = _load_runtime_state(run)
        assert state["total_steps"] == 42
        run.finish(open=False, analyze=False)


class TestRunLogging:
    def test_log_increments_step(self, tmp_store, model, optimizer):
        run = Run(name="log", store=tmp_store)
        run.watch(model, optimizer, every=100)
        run.log(loss=1.0)
        assert run.step == 1
        run.log(loss=0.5)
        assert run.step == 2
        run.finish(open=False, analyze=False)

    def test_metrics_file_written(self, tmp_store, model, optimizer):
        run = Run(name="metrics", store=tmp_store)
        run.watch(model, optimizer, every=100)
        run.log(loss=2.5, acc=0.3)
        run.log(loss=1.2, acc=0.6)
        run.finish(open=False, analyze=False)

        metrics = tmp_store.get_metrics(run.run_id)
        assert len(metrics) == 2
        assert metrics[0]["loss"] == 2.5
        assert metrics[1]["acc"] == 0.6

    def test_lr_captured(self, tmp_store, model, optimizer):
        run = Run(name="lr", store=tmp_store)
        run.watch(model, optimizer, every=100)
        run.log(loss=1.0)
        run.finish(open=False, analyze=False)

        metrics = tmp_store.get_metrics(run.run_id)
        assert "lr" in metrics[0]
        assert metrics[0]["lr"] == pytest.approx(0.01)

    def test_auto_checkpoint(self, tmp_store, model, optimizer):
        run = Run(name="auto-ckpt", store=tmp_store)
        run.watch(model, optimizer, every=100)
        run.checkpoint_every(2)

        # First forward+backward so checkpoint has something to save
        x = torch.randn(4, 4)
        y = torch.randint(0, 2, (4,))
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()

        run.log(loss=1.0)  # step 1 — no checkpoint
        run.log(loss=0.5)  # step 2 — should checkpoint
        run.flush()

        ckpts = tmp_store.list_checkpoints(run.run_id)
        assert len(ckpts) >= 1
        run.finish(open=False, analyze=False)

    def test_runtime_heartbeat_updates_on_log(self, tmp_store, model, optimizer):
        run = Run(name="heartbeat-log", store=tmp_store)
        run.watch(model, optimizer, every=100)
        before = _load_runtime_state(run)["heartbeat_ts"]
        time.sleep(0.01)
        run.log(loss=1.0)
        state = _load_runtime_state(run)
        assert state["heartbeat_ts"] >= before
        assert state["current_step"] == 1
        assert state["last_event"] == "log"
        run.finish(open=False, analyze=False)


class TestRunLogBatch:
    def test_log_batch_increments_step(self, tmp_store, model, optimizer):
        run = Run(name="batch", store=tmp_store)
        run.watch(model, optimizer, every=100)
        x = torch.randn(4, 4)
        logits = model(x)
        run.log_batch(x=x, y_pred=logits)
        assert run.step == 1
        state = _load_runtime_state(run)
        assert state["current_step"] == 1
        assert state["last_event"] == "log_batch"
        run.finish(open=False, analyze=False)


class TestKerasCallbackRuntime:
    def test_keras_train_begin_sets_total_steps(self, tmp_store):
        run = Run(name="keras-runtime", store=tmp_store)
        cb = run.keras_callback()
        cb.params = {"epochs": 3, "steps": 4}
        cb.on_train_begin()

        state = _load_runtime_state(run)
        assert state["total_steps"] == 12
        assert state["last_event"] == "keras_train_begin"

    def test_keras_batch_end_updates_runtime_state(self, tmp_store):
        run = Run(name="keras-batch-runtime", store=tmp_store)
        cb = run.keras_callback()
        cb.params = {"epochs": 2, "steps": 2}
        cb.on_train_begin()
        before = _load_runtime_state(run)["heartbeat_ts"]
        time.sleep(0.01)
        cb.on_batch_end(batch=0, logs={"loss": 1.23})

        state = _load_runtime_state(run)
        assert state["heartbeat_ts"] >= before
        assert state["current_step"] == 1
        assert state["last_event"] == "keras_batch_end"


class TestRunCheckpoint:

    def test_checkpoint_before_watch_raises(self, tmp_store):
        run = Run(name="err", store=tmp_store)
        with pytest.raises(RuntimeError, match="watch"):
            run.checkpoint()

    def test_checkpoint_with_tag(self, tmp_store, model, optimizer):
        run = Run(name="tag", store=tmp_store)
        run.watch(model, optimizer, every=100)
        run.checkpoint(step=1, tag="init")
        run.flush()

        ckpts = tmp_store.list_checkpoints(run.run_id)
        assert len(ckpts) == 1
        assert ckpts[0]["tag"] == "init"
        run.finish(open=False, analyze=False)


class TestRunFinish:

    def test_status_set_to_complete(self, tmp_store, model, optimizer):
        run = Run(name="fin", store=tmp_store)
        run.watch(model, optimizer, every=100)
        run.finish(open=False, analyze=False)

        meta = json.loads((run.run_dir / "metadata.json").read_text())
        assert meta["status"] == "complete"
        runtime = _load_runtime_state(run)
        assert runtime["status"] == "complete"

    def test_fail_sets_status(self, tmp_store, model, optimizer):
        run = Run(name="failed-run", store=tmp_store)
        run.watch(model, optimizer, every=100)
        run.fail("boom", open=False, analyze=False)

        meta = json.loads((run.run_dir / "metadata.json").read_text())
        runtime = _load_runtime_state(run)
        assert meta["status"] == "failed"
        assert runtime["status"] == "failed"
        assert "boom" in runtime["fatal_exception"]

    def test_finish_with_analyze(self, populated_run):
        run, store = populated_run
        # Re-create from existing and analyze
        run2 = Run.from_existing(run.run_id, store)
        report = run2.analyze(print_summary=False)
        assert report is not None
        assert report.run_id == run.run_id


class TestRunFromExisting:

    def test_loads_metadata(self, populated_run):
        run, store = populated_run
        loaded = Run.from_existing(run.run_id, store)
        assert loaded.name == "integ"
        assert loaded.framework == "pytorch"

    def test_handles_missing_metadata(self, tmp_store):
        run_dir = tmp_store.ensure_run_dir("ghost")
        loaded = Run.from_existing("ghost", tmp_store)
        assert loaded.name == "ghost"
        assert loaded.framework is None


class TestFrameworkDetection:

    def test_detects_pytorch(self, tmp_store, model):
        run = Run(name="fw", store=tmp_store)
        assert run.detectframework(model) == "pytorch"

    def test_unknown_framework(self, tmp_store):
        run = Run(name="fw", store=tmp_store)

        class PlainModel:
            pass

        assert run.detectframework(PlainModel()) == "unknown"


class TestJsonDefault:
    def test_numpy_int(self):
        assert json_default(np.int64(42)) == 42

    def test_numpy_float(self):
        assert isinstance(json_default(np.float32(3.14)), float)

    def test_numpy_array(self):
        arr = np.array([1, 2, 3])
        assert json_default(arr) == [1, 2, 3]

    def test_fallback_str(self):
        assert json_default(object()) is not None  # returns str(...)



class TestGradGlassCore:
    def test_configure(self, tmp_dir):
        gg = GradGlass(root=tmp_dir)
        ret = gg.configure(auto_open=True, root=tmp_dir)
        assert ret is gg

    def test_run_returns_run(self, tmp_dir):
        gg = GradGlass(root=tmp_dir)
        run = gg.run("test-run")
        assert isinstance(run, Run)
        assert run.name == "test-run"

    def test_list_runs_empty(self, tmp_dir):
        gg = GradGlass(root=tmp_dir)
        assert gg.list_runs() == []

    def test_list_runs_populated(self, tmp_dir, model, optimizer):
        gg = GradGlass(root=tmp_dir)
        run = gg.run("listed")
        run.watch(model, optimizer, every=100)
        run.finish(open=False, analyze=False)

        runs = gg.list_runs()
        assert len(runs) == 1
        assert runs[0]["name"] == "listed"

    def test_get_run(self, tmp_dir, model, optimizer):
        gg = GradGlass(root=tmp_dir)
        run = gg.run("fetch")
        run.watch(model, optimizer, every=100)
        run.finish(open=False, analyze=False)

        fetched = gg.get_run(run.run_id)
        assert fetched.run_id == run.run_id


class TestAnalysisRunner:
    def test_build_context(self, populated_run):
        run, store = populated_run
        runner = AnalysisRunner(run.run_id, store, run.run_dir)
        ctx = runner.build_context()

        assert ctx.run_id == run.run_id
        assert ctx.has_metrics
        assert ctx.has_checkpoints
        assert ctx.metadata is not None

    def test_run_all_returns_results(self, populated_run):
        run, store = populated_run
        runner = AnalysisRunner(run.run_id, store, run.run_dir)
        results = runner.run_all()
        assert len(results) > 0
        assert all(isinstance(r, TestResult) for r in results)

    def test_run_specific_tests(self, populated_run):
        run, store = populated_run
        runner = AnalysisRunner(run.run_id, store, run.run_dir)
        results = runner.run_all(tests=["STORE_LAYOUT_VALID", "METADATA_VALID_JSON"])
        assert len(results) == 2

    def test_unknown_test_skips(self, populated_run):
        run, store = populated_run
        runner = AnalysisRunner(run.run_id, store, run.run_dir)
        results = runner.run_all(tests=["NONEXISTENT_TEST_XYZ"])
        assert len(results) == 1
        assert results[0].status == TestStatus.SKIP

    def test_generate_summary_sections(self, populated_run):
        run, store = populated_run
        runner = AnalysisRunner(run.run_id, store, run.run_dir)
        sections = runner.generate_summary_sections()

        assert "checkpoint_diff_summary" in sections
        assert "gradient_flow_analysis" in sections
        assert "training_metrics_summary" in sections
        assert "artifact_store_summary" in sections
        assert sections["training_metrics_summary"]["has_data"] is True

    def test_render_text(self, populated_run):
        run, store = populated_run
        runner = AnalysisRunner(run.run_id, store, run.run_dir)
        ctx = runner.build_context()
        sections = runner.generate_summary_sections(ctx)
        results = runner.run_all()
        text = runner.render_text(sections, results)
        assert "Checkpoint Diff Summary" in text
        assert "Gradient Flow Analysis" in text
        assert "Test Suite Results" in text

    def test_store_layout_test_passes(self, populated_run):
        run, store = populated_run
        runner = AnalysisRunner(run.run_id, store, run.run_dir)
        results = runner.run_all(tests=["STORE_LAYOUT_VALID"])
        # populated_run creates all required dirs
        assert results[0].status == TestStatus.PASS

    def test_metadata_test_passes(self, populated_run):
        run, store = populated_run
        runner = AnalysisRunner(run.run_id, store, run.run_dir)
        results = runner.run_all(tests=["METADATA_VALID_JSON"])
        assert results[0].status == TestStatus.PASS

    def test_checkpoint_readable_test(self, populated_run):
        run, store = populated_run
        runner = AnalysisRunner(run.run_id, store, run.run_dir)
        results = runner.run_all(tests=["CHECKPOINT_READABLE"])
        assert results[0].status == TestStatus.PASS

    def test_checkpoint_shape_consistency(self, populated_run):
        run, store = populated_run
        runner = AnalysisRunner(run.run_id, store, run.run_dir)
        results = runner.run_all(tests=["CHECKPOINT_SHAPE_CONSISTENCY"])
        assert results[0].status == TestStatus.PASS



class TestPostRunReport:
  
    def test_generate_report(self, populated_run):
        run, store = populated_run
        report = PostRunReport.generate(
            run_id=run.run_id,
            store=store,
            run_dir=run.run_dir,
            save=True,
            print_summary=False,
        )
        assert report.run_id == run.run_id
        assert report.tests["total"] > 0

    def test_report_saved_to_disk(self, populated_run):
        run, store = populated_run
        PostRunReport.generate(
            run_id=run.run_id,
            store=store,
            run_dir=run.run_dir,
            save=True,
            print_summary=False,
        )
        assert (run.run_dir / "analysis" / "report.json").exists()
        assert (run.run_dir / "analysis" / "summary.txt").exists()
        assert (run.run_dir / "analysis" / "tests.jsonl").exists()

    def test_report_round_trip(self, populated_run):
        run, store = populated_run
        original = PostRunReport.generate(
            run_id=run.run_id,
            store=store,
            run_dir=run.run_dir,
            save=True,
            print_summary=False,
        )
        loaded = PostRunReport.from_file(run.run_dir)
        assert loaded is not None
        assert loaded.run_id == original.run_id
        assert loaded.tests["total"] == original.tests["total"]

    def test_from_file_missing_returns_none(self, tmp_dir):
        result = PostRunReport.from_file(tmp_dir / "nonexistent")
        assert result is None


class TestTestRegistry:
    def test_all_tests_populated(self):
        all_tests = TestRegistry.all_tests()
        assert len(all_tests) > 0

    def test_builtin_tests_registered(self):
        ids = TestRegistry.ids()
        assert "STORE_LAYOUT_VALID" in ids
        assert "METADATA_VALID_JSON" in ids
        assert "CHECKPOINT_READABLE" in ids

    def test_get_existing(self):
        t = TestRegistry.get("STORE_LAYOUT_VALID")
        assert t is not None
        assert t.id == "STORE_LAYOUT_VALID"
        assert t.category == TestCategory.ARTIFACT

    def test_get_nonexistent(self):
        assert TestRegistry.get("NO_SUCH_TEST") is None

    def test_by_category(self):
        artifact_tests = TestRegistry.by_category(TestCategory.ARTIFACT)
        assert len(artifact_tests) > 0
        assert all(t.category == TestCategory.ARTIFACT for t in artifact_tests)

    def test_custom_test_registration(self):
        @TestRegistry.register(
            id="CUSTOM_DEBUG_TEST",
            title="Custom test for debugging",
            category=TestCategory.METRICS,
            severity=TestSeverity.LOW,
        )
        def custom_test(ctx):
            return TestResult(
                id="CUSTOM_DEBUG_TEST",
                title="Custom test for debugging",
                status=TestStatus.PASS,
                severity=TestSeverity.LOW,
                category=TestCategory.METRICS,
            )

        assert "CUSTOM_DEBUG_TEST" in TestRegistry.ids()
        registered = TestRegistry.get("CUSTOM_DEBUG_TEST")
        assert registered.fn is custom_test


class TestTestContext:
    def test_has_properties_empty(self, tmp_dir, tmp_store):
        ctx = TestContext(run_id="x", run_dir=tmp_dir, store=tmp_store)
        assert not ctx.has_checkpoints
        assert not ctx.has_metrics
        assert not ctx.has_grad_summaries
        assert not ctx.has_activations
        assert not ctx.has_predictions
        assert not ctx.has_architecture
        assert not ctx.is_distributed

    def test_has_properties_set(self, tmp_dir, tmp_store):
        ctx = TestContext(
            run_id="x",
            run_dir=tmp_dir,
            store=tmp_store,
            metrics=[{"step": 1, "loss": 0.5}],
            checkpoints_meta=[{"step": 1}],
            architecture={"layers": [], "edges": []},
            gradient_summaries=[{"step": 1, "layers": {}}],
        )
        assert ctx.has_metrics
        assert ctx.has_checkpoints
        assert ctx.has_architecture
        assert ctx.has_grad_summaries

    def test_checkpoint_steps(self, tmp_dir, tmp_store):
        ctx = TestContext(
            run_id="x",
            run_dir=tmp_dir,
            store=tmp_store,
            checkpoints_meta=[{"step": 30}, {"step": 10}, {"step": 20}],
        )
        assert ctx.checkpoint_steps() == [10, 20, 30]


class TestTestResult: 
    def test_to_dict(self):
        r = TestResult(
            id="T1",
            title="Test One",
            status=TestStatus.PASS,
            severity=TestSeverity.LOW,
            category=TestCategory.METRICS,
            details={"key": "val"},
            recommendation="all good",
            duration_ms=12.345,
        )
        d = r.to_dict()
        assert d["id"] == "T1"
        assert d["status"] == "pass"
        assert d["severity"] == "LOW"
        assert d["category"] == "Training Metrics"
        assert d["duration_ms"] == 12.35


class TestEndToEnd:
    def test_full_pipeline(self, tmp_dir):
        gg = GradGlass(root=tmp_dir)
        model = TinyMLP()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        run = gg.run("e2e-test")
        run.watch(model, optimizer, every=2, sample_batches=2)
        run.checkpoint_every(3)

        x = torch.randn(16, 4)
        y = torch.randint(0, 2, (16,))

        for epoch in range(5):
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            run.log(loss=loss.item(), acc=0.5 + epoch * 0.1, epoch=epoch)
            run.log_batch(x=x, y=y, y_pred=logits, loss=loss)

        run.checkpoint(step=run.step, tag="final")
        report = run.finish(open=False, analyze=True, print_summary=False)

        assert (run.run_dir / "model_structure.json").exists()
        assert (run.run_dir / "metadata.json").exists()

        metrics = gg.store.get_metrics(run.run_id)
        assert len(metrics) == 5

        ckpts = gg.store.list_checkpoints(run.run_id)
        assert len(ckpts) >= 1

        assert report is not None
        assert report.tests["total"] > 0
        assert report.tests["passed"] > 0

        runs = gg.list_runs()
        assert len(runs) == 1
        assert runs[0]["status"] == "complete"

    def test_gradient_flow_end_to_end(self, tmp_dir):
        gg = GradGlass(root=tmp_dir)
        model = TinyMLP()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        run = gg.run("grad-flow")
        run.watch(model, optimizer, every=2, sample_batches=1)

        x = torch.randn(8, 4)
        y = torch.randint(0, 2, (8,))

        for i in range(6):
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            run.log(loss=loss.item())

        run.checkpoint(step=run.step, tag="grad")
        run.finish(open=False, analyze=False, print_summary=False)

        # Gradient summaries should have been captured
        grad_summaries = gg.store.get_gradient_summaries(run.run_id)
        assert len(grad_summaries) > 0

        # Analyze the gradient flow
        analysis = gradient_flow_analysis(grad_summaries)
        assert len(analysis) > 0
        for entry in analysis:
            assert "layer" in entry
            assert "flags" in entry

    def test_multiple_runs_isolation(self, tmp_dir):
        gg = GradGlass(root=tmp_dir)
        model1 = TinyMLP()
        model2 = TinyMLP()
        opt1 = torch.optim.SGD(model1.parameters(), lr=0.01)
        opt2 = torch.optim.SGD(model2.parameters(), lr=0.01)

        run1 = gg.run("run-a")
        run1.watch(model1, opt1, every=100)
        run1.log(loss=1.0)
        run1.finish(open=False, analyze=False)

        run2 = gg.run("run-b")
        run2.watch(model2, opt2, every=100)
        run2.log(loss=2.0)
        run2.finish(open=False, analyze=False)

        runs = gg.list_runs()
        assert len(runs) == 2
        names = {r["name"] for r in runs}
        assert names == {"run-a", "run-b"}

        m1 = gg.store.get_metrics(run1.run_id)
        m2 = gg.store.get_metrics(run2.run_id)
        assert m1[0]["loss"] == 1.0
        assert m2[0]["loss"] == 2.0
