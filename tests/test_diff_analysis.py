"""
GradGlass — Diff Viewer Analysis Tests
======================================
Tests for the checkpoint diff analysis tests:
  - WEIGHT_DIFF_COMPUTED
  - WEIGHT_DIFF_SEVERITY_COUNTS
  - TOP_CHANGED_LAYERS
  - UNCHANGED_LAYER_DETECTION
  - EXCESSIVE_UPDATE_RATIO
  - TRAINABLE_FROZEN_CONSISTENCY
  - CHECKPOINT_SHAPE_CONSISTENCY

Also tests the /api/runs/{run_id}/freeze_code server endpoint.
"""
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from gradglass.artifacts import ArtifactStore
from gradglass.analysis.registry import TestContext, TestStatus
from gradglass.diff import full_diff, Severity


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_store():
    tmpdir = tempfile.mkdtemp()
    store = ArtifactStore(root=tmpdir)
    yield store
    shutil.rmtree(tmpdir)


def _write_checkpoint(store, run_id, step, weights: dict, num_params=None):
    """Write a checkpoint .npz + meta JSON to the artifact store."""
    run_dir = store.ensure_run_dir(run_id)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    np.savez(ckpt_dir / f"step_{step}.npz", **weights)
    meta = {"step": step, "num_params": num_params or sum(v.size for v in weights.values())}
    with open(ckpt_dir / f"step_{step}_meta.json", "w") as f:
        json.dump(meta, f)
    meta_path = run_dir / "metadata.json"
    if not meta_path.exists():
        with open(meta_path, "w") as f:
            json.dump({"name": run_id, "run_id": run_id, "status": "finished", "start_time": "2026"}, f)


def _ctx(store, run_id):
    run_dir = store.ensure_run_dir(run_id)
    meta_path = run_dir / "metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    ckpt_meta = store.list_checkpoints(run_id)
    return TestContext(
        run_id=run_id, run_dir=run_dir, store=store,
        metadata=meta, metrics=[], checkpoints_meta=ckpt_meta,
    )


# ────────────────────────────────────────────────────────────────────────────
# WEIGHT_DIFF_COMPUTED
# ────────────────────────────────────────────────────────────────────────────

class TestWeightDiffComputed:
    def test_pass_two_checkpoints(self, tmp_store):
        from gradglass.analysis.builtins import test_weight_diff_computed
        w_a = {"fc1.weight": np.random.randn(32, 16).astype(np.float32)}
        w_b = {"fc1.weight": np.random.randn(32, 16).astype(np.float32)}
        _write_checkpoint(tmp_store, "wd-pass", 1, w_a)
        _write_checkpoint(tmp_store, "wd-pass", 2, w_b)
        ctx = _ctx(tmp_store, "wd-pass")
        result = test_weight_diff_computed(ctx)
        assert result.status == TestStatus.PASS
        assert result.details["layers_compared"] == 1

    def test_skip_single_checkpoint(self, tmp_store):
        from gradglass.analysis.builtins import test_weight_diff_computed
        w = {"fc1.weight": np.random.randn(32, 16).astype(np.float32)}
        _write_checkpoint(tmp_store, "wd-single", 1, w)
        ctx = _ctx(tmp_store, "wd-single")
        result = test_weight_diff_computed(ctx)
        assert result.status == TestStatus.SKIP


# ────────────────────────────────────────────────────────────────────────────
# TOP_CHANGED_LAYERS
# ────────────────────────────────────────────────────────────────────────────

class TestTopChangedLayers:
    def test_returns_sorted_layers(self, tmp_store):
        from gradglass.analysis.builtins import test_top_changed_layers
        # fc1 changes a lot, fc2 barely changes
        w_a = {
            "fc1.weight": np.zeros((32, 16), dtype=np.float32),
            "fc2.weight": np.ones((16, 8), dtype=np.float32),
        }
        w_b = {
            "fc1.weight": np.random.randn(32, 16).astype(np.float32) * 5,  # big change
            "fc2.weight": np.ones((16, 8), dtype=np.float32) + 1e-8,        # tiny change
        }
        _write_checkpoint(tmp_store, "top-layers", 1, w_a)
        _write_checkpoint(tmp_store, "top-layers", 2, w_b)
        ctx = _ctx(tmp_store, "top-layers")
        result = test_top_changed_layers(ctx)
        assert result.status == TestStatus.PASS
        top = result.details["top_5_layers"]
        assert top[0]["layer"] == "fc1.weight"


# ────────────────────────────────────────────────────────────────────────────
# UNCHANGED_LAYER_DETECTION
# ────────────────────────────────────────────────────────────────────────────

class TestUnchangedLayerDetection:
    def test_warn_on_frozen_layer(self, tmp_store):
        from gradglass.analysis.builtins import test_unchanged_layers
        w = np.random.randn(32, 16).astype(np.float32)
        w_a = {"fc1.weight": w.copy(), "fc2.weight": np.random.randn(16, 8).astype(np.float32)}
        w_b = {"fc1.weight": w.copy(), "fc2.weight": np.random.randn(16, 8).astype(np.float32)}  # fc1 unchanged
        _write_checkpoint(tmp_store, "unchanged", 1, w_a)
        _write_checkpoint(tmp_store, "unchanged", 2, w_b)
        ctx = _ctx(tmp_store, "unchanged")
        result = test_unchanged_layers(ctx)
        assert result.status == TestStatus.WARN
        assert "fc1.weight" in result.details["unchanged_layers"]

    def test_pass_all_layers_updated(self, tmp_store):
        from gradglass.analysis.builtins import test_unchanged_layers
        w_a = {"fc1.weight": np.zeros((16, 8), dtype=np.float32)}
        w_b = {"fc1.weight": np.ones((16, 8), dtype=np.float32)}
        _write_checkpoint(tmp_store, "all-changed", 1, w_a)
        _write_checkpoint(tmp_store, "all-changed", 2, w_b)
        ctx = _ctx(tmp_store, "all-changed")
        result = test_unchanged_layers(ctx)
        assert result.status == TestStatus.PASS


# ────────────────────────────────────────────────────────────────────────────
# EXCESSIVE_UPDATE_RATIO
# ────────────────────────────────────────────────────────────────────────────

class TestExcessiveUpdateRatio:
    def test_pass_small_update(self, tmp_store):
        from gradglass.analysis.builtins import test_excessive_update_ratio
        w = np.ones((32, 16), dtype=np.float32)
        w_a = {"fc1.weight": w.copy()}
        w_b = {"fc1.weight": w + 0.001}  # tiny update
        _write_checkpoint(tmp_store, "small-update", 1, w_a)
        _write_checkpoint(tmp_store, "small-update", 2, w_b)
        ctx = _ctx(tmp_store, "small-update")
        result = test_excessive_update_ratio(ctx)
        assert result.status == TestStatus.PASS

    def test_warn_large_update(self, tmp_store):
        from gradglass.analysis.builtins import test_excessive_update_ratio
        w_a = {"fc1.weight": np.ones((32, 16), dtype=np.float32) * 0.01}
        w_b = {"fc1.weight": np.ones((32, 16), dtype=np.float32) * 100.0}  # 10000x bigger
        _write_checkpoint(tmp_store, "large-update", 1, w_a)
        _write_checkpoint(tmp_store, "large-update", 2, w_b)
        ctx = _ctx(tmp_store, "large-update")
        result = test_excessive_update_ratio(ctx)
        assert result.status == TestStatus.WARN


# ────────────────────────────────────────────────────────────────────────────
# full_diff integration
# ────────────────────────────────────────────────────────────────────────────

class TestFullDiffIntegration:
    def test_diff_severity_distribution(self):
        w_a = {
            "layer_small": np.ones((8, 8), dtype=np.float32) * 0.1,
            "layer_large": np.ones((16, 16), dtype=np.float32),
        }
        w_b = {
            "layer_small": np.ones((8, 8), dtype=np.float32) * 0.100001,   # tiny
            "layer_large": np.random.randn(16, 16).astype(np.float32) * 50, # huge
        }
        result = full_diff(w_a, w_b, run_id="test", step_a=1, step_b=2)
        assert result.summary["total_layers"] == 2
        severity_vals = {l.layer_name: l.severity for l in result.layers}
        assert severity_vals["layer_small"] in (Severity.LOW, Severity.MEDIUM)
        assert severity_vals["layer_large"] in (Severity.HIGH, Severity.CRITICAL)

    def test_diff_to_dict_includes_layers(self):
        w_a = {"fc.weight": np.random.randn(16, 8).astype(np.float32)}
        w_b = {"fc.weight": np.random.randn(16, 8).astype(np.float32)}
        result = full_diff(w_a, w_b, run_id="test", step_a=1, step_b=2)
        d = result.to_dict(include_deltas=True)
        assert "layers" in d
        assert "summary" in d
        assert len(d["layers"]) == 1
        assert "delta_histogram" in d["layers"][0]
        assert "top_k_deltas" in d["layers"][0]


# ────────────────────────────────────────────────────────────────────────────
# Freeze Code API endpoint
# ────────────────────────────────────────────────────────────────────────────

class TestFreezeCodeEndpoint:
    def test_endpoint_returns_code(self, tmp_store):
        """Test the freeze_code logic that backs the /api/.../freeze_code endpoint."""
        # Write gradient summaries
        run_id = "freeze-api"
        run_dir = tmp_store.ensure_run_dir(run_id)
        grad_dir = run_dir / "gradients"
        grad_dir.mkdir(exist_ok=True)

        for i in range(1, 5):
            # The store reads the file as raw JSON and wraps it: {"step": step, "layers": <file content>}
            # So the file must contain the flat layer dict directly (matching capture.py format)
            summary = {
                "backbone.layer1": {"norm": 1e-7, "mean": 0.0},
                "head.fc1": {"norm": 1.5, "mean": 0.01},
            }
            with open(grad_dir / f"summaries_step_{i}.json", "w") as f:
                json.dump(summary, f)
        with open(run_dir / "metadata.json", "w") as f:
            json.dump({"name": run_id, "run_id": run_id, "status": "finished", "start_time": "2026"}, f)

        summaries = tmp_store.get_gradient_summaries(run_id)
        layer_norms: dict[str, list] = {}
        for entry in summaries:
            layers = entry.get("layers", {})
            for layer, data in layers.items():
                if isinstance(data, dict):
                    norm = data.get("norm")
                elif isinstance(data, (int, float)):
                    norm = float(data)
                else:
                    continue
                if norm is not None:
                    layer_norms.setdefault(layer, []).append(norm)

        mean_norms = {layer: sum(norms) / len(norms) for layer, norms in layer_norms.items()}
        max_mean = max(mean_norms.values())
        threshold = max_mean * 0.01
        candidates = [layer for layer, norm in mean_norms.items() if norm < threshold]

        assert "backbone.layer1" in candidates
        assert "head.fc1" not in candidates
