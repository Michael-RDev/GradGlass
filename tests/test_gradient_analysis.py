"""
GradGlass — Gradient Analysis Tests
=====================================
Tests for:
  - gradient_flow_analysis() in diff.py
      · converged MNIST should produce zero VANISHING flags (near-zero *mean*
        but healthy *norm* is expected and must not be penalised)
      · truly dead layers (both mean AND norm near zero) must be flagged
      · exploding gradients (large mean or large norm) must be flagged
      · NOISY flag fires on low SNR regardless of VANISHING/EXPLODING state
  - Builtin analysis tests in analysis/builtins.py
      · GRAD_VANISHING  — norm-based: warns only when norm < 1e-7
      · GRAD_EXPLODING  — norm-based: warns only when norm > 100
      · GRAD_NAN_INF    — fires on NaN / Inf in any stat field
      · GRAD_LAYER_IMBALANCE — fires when max/min norm ratio > 100
"""
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from gradglass.artifacts import ArtifactStore
from gradglass.analysis.registry import TestContext, TestStatus
from gradglass.diff import gradient_flow_analysis


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_summaries(layers_per_step: list[dict]) -> list[dict]:
    """
    Build the gradient-summary list that gradient_flow_analysis() expects.

    Each item in layers_per_step is a dict of  {layer_name: {mean, var, max, norm, ...}}.
    Returns a list of  {"step": N, "layers": {...}}  entries.
    """
    return [{"step": i + 1, "layers": layers} for i, layers in enumerate(layers_per_step)]


def _healthy_mnist_layers():
    """
    8-layer stat snapshot that mirrors a real, converged MNIST-CNN run.

    Key property: fc2.bias and fc2.weight have near-zero *mean* (signed grads
    cancel in a converged model) but healthy *norm* values.  This must NOT
    produce a VANISHING flag after the fix.
    """
    return {
        "conv1.bias":   {"mean":  2.13e-04, "var": 1.2e-06, "max": 0.0091, "norm": 0.0352},
        "conv1.weight": {"mean": -1.05e-05, "var": 8.4e-08, "max": 0.0061, "norm": 0.0621},
        "conv2.bias":   {"mean":  5.60e-05, "var": 3.1e-07, "max": 0.0054, "norm": 0.0193},
        "conv2.weight": {"mean": -3.20e-06, "var": 1.7e-08, "max": 0.0042, "norm": 0.2015},
        "fc1.bias":     {"mean":  2.87e-04, "var": 1.2e-05, "max": 0.0149, "norm": 0.0400},
        "fc1.weight":   {"mean":  7.13e-05, "var": 2.7e-06, "max": 0.0328, "norm": 1.0494},
        # These two have mean ≈ 1e-9 (below 1e-7 threshold) but norm is healthy
        "fc2.bias":     {"mean": -5.59e-10, "var": 2.3e-04, "max": 0.0277, "norm": 0.0475},
        "fc2.weight":   {"mean": -5.82e-11, "var": 2.7e-05, "max": 0.0298, "norm": 0.1865},
    }


@pytest.fixture
def tmp_store():
    tmpdir = tempfile.mkdtemp()
    store = ArtifactStore(root=tmpdir)
    yield store
    shutil.rmtree(tmpdir)


def _write_grad_summaries(store, run_id: str, steps: list[dict]):
    """Write gradient summary JSON files into the artifact store."""
    run_dir = store.ensure_run_dir(run_id)
    grad_dir = run_dir / "gradients"
    grad_dir.mkdir(exist_ok=True)
    meta_path = run_dir / "metadata.json"
    if not meta_path.exists():
        meta_path.write_text(json.dumps({
            "name": run_id, "run_id": run_id,
            "status": "finished", "start_time": "2026",
        }))
    for i, layers in enumerate(steps, start=1):
        with open(grad_dir / f"summaries_step_{i}.json", "w") as f:
            json.dump(layers, f)


def _ctx(store, run_id: str) -> TestContext:
    run_dir = store.ensure_run_dir(run_id)
    meta = json.loads((run_dir / "metadata.json").read_text())
    gradient_summaries = store.get_gradient_summaries(run_id)
    return TestContext(
        run_id=run_id, run_dir=run_dir, store=store,
        metadata=meta, metrics=[], checkpoints_meta=[],
        gradient_summaries=gradient_summaries,
    )


# ── gradient_flow_analysis() unit tests ───────────────────────────────────────

class TestGradientFlowAnalysis:
    """Tests for diff.gradient_flow_analysis() directly."""

    def test_converged_mnist_no_vanishing_flag(self):
        """
        Converged MNIST: fc2.bias mean ≈ -5.6e-10 (below 1e-7) but norm = 0.047
        (well above 1e-5).  Must NOT be flagged as VANISHING.
        """
        summaries = _make_summaries([_healthy_mnist_layers()])
        results = gradient_flow_analysis(summaries)
        vanishing_layers = [r for r in results if "VANISHING" in r["flags"]]
        assert vanishing_layers == [], (
            f"Expected no VANISHING flags on a converged MNIST model, "
            f"but got: {[l['layer'] for l in vanishing_layers]}"
        )

    def test_healthy_mnist_all_layers_clean(self):
        """Full healthy MNIST run: flagged count should be 0 (NOISY is acceptable
        but must not include VANISHING or EXPLODING)."""
        summaries = _make_summaries([_healthy_mnist_layers()])
        results = gradient_flow_analysis(summaries)
        unstable = [
            r for r in results
            if "VANISHING" in r["flags"] or "EXPLODING" in r["flags"]
        ]
        assert unstable == [], (
            f"Healthy MNIST run should have no VANISHING/EXPLODING flags, "
            f"got: {[(r['layer'], r['flags']) for r in unstable]}"
        )

    def test_truly_vanishing_layer_flagged(self):
        """
        A layer with both near-zero mean AND near-zero norm is genuinely dead
        and must be flagged as VANISHING.
        """
        layers = {
            "fc1.weight": {"mean": 0.05, "var": 0.001, "max": 0.2, "norm": 0.9},
            # both mean and norm are tiny → truly vanishing
            "fc2.weight": {"mean": 1e-10, "var": 1e-14, "max": 5e-9, "norm": 8e-8},
        }
        summaries = _make_summaries([layers])
        results = gradient_flow_analysis(summaries)
        by_layer = {r["layer"]: r["flags"] for r in results}
        assert "VANISHING" in by_layer["fc2.weight"], (
            "fc2.weight has near-zero mean AND near-zero norm — must be VANISHING"
        )
        assert "VANISHING" not in by_layer["fc1.weight"], (
            "fc1.weight has healthy norm — must NOT be VANISHING"
        )

    def test_exploding_mean_flagged(self):
        """A layer with |mean| > 100 must be flagged as EXPLODING."""
        layers = {
            "fc1.weight": {"mean": 250.0, "var": 100.0, "max": 500.0, "norm": 10000.0},
        }
        summaries = _make_summaries([layers])
        results = gradient_flow_analysis(summaries)
        assert results[0]["flags"] == [] or "EXPLODING" in results[0]["flags"]
        assert "EXPLODING" in results[0]["flags"]

    def test_exploding_norm_flagged(self):
        """A layer with norm > 1000 but small mean must still be flagged EXPLODING."""
        layers = {
            "fc1.weight": {"mean": 0.001, "var": 50.0, "max": 5000.0, "norm": 1500.0},
        }
        summaries = _make_summaries([layers])
        results = gradient_flow_analysis(summaries)
        assert "EXPLODING" in results[0]["flags"], (
            "norm=1500 > 1000 threshold — must be flagged EXPLODING"
        )

    def test_noisy_flag_independent_of_vanishing(self):
        """
        A layer with low SNR (|mean| / sqrt(var) < 0.01) but healthy norm must
        get NOISY but must NOT get VANISHING.
        """
        # mean=1e-8 → below 1e-7, but norm=0.5 → well above 1e-5.
        # SNR = 1e-8 / sqrt(0.01) = 1e-7 < 0.01 → NOISY
        layers = {
            "fc1.weight": {"mean": 1e-8, "var": 0.01, "max": 0.3, "norm": 0.5},
        }
        summaries = _make_summaries([layers])
        results = gradient_flow_analysis(summaries)
        flags = results[0]["flags"]
        assert "NOISY" in flags, "Low SNR layer must be flagged as NOISY"
        assert "VANISHING" not in flags, (
            "Healthy norm means NOISY should not also trigger VANISHING"
        )

    def test_empty_summaries_returns_empty(self):
        """Empty input must return an empty list, not raise."""
        assert gradient_flow_analysis([]) == []

    def test_multiple_steps_uses_latest(self):
        """
        If early steps have problematic stats but the latest step is healthy,
        the flag should reflect the latest step only.
        """
        early = {"fc1.weight": {"mean": 1e-12, "var": 1e-16, "max": 1e-10, "norm": 5e-9}}
        latest = {"fc1.weight": {"mean": 0.001, "var": 1e-5, "max": 0.05, "norm": 0.3}}
        summaries = _make_summaries([early, latest])
        results = gradient_flow_analysis(summaries)
        assert "VANISHING" not in results[0]["flags"], (
            "Latest step is healthy — VANISHING must not be set"
        )

    def test_history_length_recorded(self):
        """num_steps should reflect how many steps are in the history."""
        layers = {"fc1.weight": {"mean": 0.01, "var": 1e-4, "max": 0.1, "norm": 0.5}}
        summaries = _make_summaries([layers] * 5)
        results = gradient_flow_analysis(summaries)
        assert results[0]["num_steps"] == 5


# ── Builtin test function tests ────────────────────────────────────────────────

class TestGradVanishingBuiltin:
    """Tests for the GRAD_VANISHING builtin (norm-based, not mean-based)."""

    def test_pass_healthy_norms(self, tmp_store):
        """All layer norms healthy → PASS."""
        from gradglass.analysis.builtins import test_grad_vanishing
        _write_grad_summaries(tmp_store, "gv-pass", [_healthy_mnist_layers()])
        ctx = _ctx(tmp_store, "gv-pass")
        result = test_grad_vanishing(ctx)
        assert result.status == TestStatus.PASS

    def test_warn_tiny_norm(self, tmp_store):
        """A layer with norm < 1e-7 → WARN."""
        from gradglass.analysis.builtins import test_grad_vanishing
        layers = {
            "fc1.weight": {"mean": 0.01, "var": 1e-4, "max": 0.05, "norm": 0.5},
            "fc2.weight": {"mean": 1e-10, "var": 1e-14, "max": 5e-9, "norm": 5e-9},  # tiny norm
        }
        _write_grad_summaries(tmp_store, "gv-warn", [layers])
        ctx = _ctx(tmp_store, "gv-warn")
        result = test_grad_vanishing(ctx)
        assert result.status == TestStatus.WARN
        assert any(e["layer"] == "fc2.weight" for e in result.details["vanishing_entries"])

    def test_skip_no_summaries(self, tmp_store):
        """No gradient data → SKIP, not FAIL."""
        from gradglass.analysis.builtins import test_grad_vanishing
        run_dir = tmp_store.ensure_run_dir("gv-skip")
        (run_dir / "metadata.json").write_text(json.dumps(
            {"name": "gv-skip", "run_id": "gv-skip", "status": "finished", "start_time": "2026"}
        ))
        ctx = _ctx(tmp_store, "gv-skip")
        result = test_grad_vanishing(ctx)
        assert result.status == TestStatus.SKIP

    def test_converged_mnist_norms_do_not_warn(self, tmp_store):
        """
        The real MNIST run's norms are all > 0.01.  Even though some means are
        below 1e-7, the norm-based builtin must return PASS (not WARN).
        """
        from gradglass.analysis.builtins import test_grad_vanishing
        _write_grad_summaries(tmp_store, "gv-mnist", [_healthy_mnist_layers()])
        ctx = _ctx(tmp_store, "gv-mnist")
        result = test_grad_vanishing(ctx)
        assert result.status == TestStatus.PASS, (
            "Converged MNIST norms are healthy — GRAD_VANISHING must PASS"
        )


class TestGradExplodingBuiltin:
    """Tests for the GRAD_EXPLODING builtin (norm > 100 → WARN)."""

    def test_pass_normal_norms(self, tmp_store):
        """Normal training norms → PASS."""
        from gradglass.analysis.builtins import test_grad_exploding
        _write_grad_summaries(tmp_store, "ge-pass", [_healthy_mnist_layers()])
        ctx = _ctx(tmp_store, "ge-pass")
        result = test_grad_exploding(ctx)
        assert result.status == TestStatus.PASS

    def test_warn_large_norm(self, tmp_store):
        """Norm > 100 in any layer → WARN."""
        from gradglass.analysis.builtins import test_grad_exploding
        layers = {
            "fc1.weight": {"mean": 50.0, "var": 10.0, "max": 200.0, "norm": 500.0},
        }
        _write_grad_summaries(tmp_store, "ge-warn", [layers])
        ctx = _ctx(tmp_store, "ge-warn")
        result = test_grad_exploding(ctx)
        assert result.status == TestStatus.WARN
        assert any(e["layer"] == "fc1.weight" for e in result.details["exploding_entries"])

    def test_skip_no_summaries(self, tmp_store):
        """No gradient data → SKIP."""
        from gradglass.analysis.builtins import test_grad_exploding
        run_dir = tmp_store.ensure_run_dir("ge-skip")
        (run_dir / "metadata.json").write_text(json.dumps(
            {"name": "ge-skip", "run_id": "ge-skip", "status": "finished", "start_time": "2026"}
        ))
        ctx = _ctx(tmp_store, "ge-skip")
        result = test_grad_exploding(ctx)
        assert result.status == TestStatus.SKIP


class TestGradNanInfBuiltin:
    """Tests for the GRAD_NAN_INF builtin."""

    def test_pass_clean_summaries(self, tmp_store):
        from gradglass.analysis.builtins import test_grad_nan_inf
        _write_grad_summaries(tmp_store, "gni-pass", [_healthy_mnist_layers()])
        ctx = _ctx(tmp_store, "gni-pass")
        assert test_grad_nan_inf(ctx).status == TestStatus.PASS

    def test_fail_nan_in_mean(self, tmp_store):
        """NaN in mean → FAIL."""
        from gradglass.analysis.builtins import test_grad_nan_inf
        layers = {"fc1.weight": {"mean": float("nan"), "var": 0.01, "max": 0.1, "norm": 0.5}}
        _write_grad_summaries(tmp_store, "gni-nan", [layers])
        ctx = _ctx(tmp_store, "gni-nan")
        assert test_grad_nan_inf(ctx).status == TestStatus.FAIL

    def test_fail_inf_in_norm(self, tmp_store):
        """Inf in norm → FAIL."""
        from gradglass.analysis.builtins import test_grad_nan_inf
        layers = {"fc1.weight": {"mean": 0.01, "var": 0.01, "max": 0.1, "norm": float("inf")}}
        _write_grad_summaries(tmp_store, "gni-inf", [layers])
        ctx = _ctx(tmp_store, "gni-inf")
        assert test_grad_nan_inf(ctx).status == TestStatus.FAIL


class TestGradLayerImbalanceBuiltin:
    """Tests for the GRAD_LAYER_IMBALANCE builtin (max/min norm ratio > 100)."""

    def test_pass_balanced(self, tmp_store):
        from gradglass.analysis.builtins import test_grad_layer_imbalance
        layers = {
            "fc1.weight": {"mean": 0.01, "var": 1e-4, "max": 0.1, "norm": 1.0},
            "fc2.weight": {"mean": 0.005, "var": 5e-5, "max": 0.05, "norm": 0.5},
        }
        _write_grad_summaries(tmp_store, "gli-pass", [layers])
        ctx = _ctx(tmp_store, "gli-pass")
        assert test_grad_layer_imbalance(ctx).status == TestStatus.PASS

    def test_warn_extreme_imbalance(self, tmp_store):
        """max/min norm ratio > 100 → WARN."""
        from gradglass.analysis.builtins import test_grad_layer_imbalance
        layers = {
            "fc1.weight": {"mean": 0.01, "var": 1e-4, "max": 0.1, "norm": 200.0},
            "fc2.weight": {"mean": 1e-6, "var": 1e-12, "max": 1e-5, "norm": 1.0},  # ratio 200x
        }
        _write_grad_summaries(tmp_store, "gli-warn", [layers])
        ctx = _ctx(tmp_store, "gli-warn")
        assert test_grad_layer_imbalance(ctx).status == TestStatus.WARN


# ── Integration: mnist should be Healthy ──────────────────────────────────────

class TestMnistHealthIntegration:
    """
    Simulate what the StoryMode health computation sees for a healthy,
    converged MNIST run.  After the diff.py fix, gradient_flow_analysis()
    must not return any VANISHING/EXPLODING flagged layers for this data,
    and the builtin GRAD_VANISHING / GRAD_EXPLODING tests must both PASS.
    """

    def test_gradient_instability_returns_empty_for_mnist(self):
        """
        gradient_flow_analysis() on a healthy converged MNIST run must
        return zero VANISHING or EXPLODING flagged layers.
        """
        # Simulate multiple steps of a converged run
        summaries = _make_summaries([_healthy_mnist_layers()] * 10)
        results = gradient_flow_analysis(summaries)

        vanishing = [r for r in results if "VANISHING" in r["flags"]]
        exploding = [r for r in results if "EXPLODING" in r["flags"]]

        assert vanishing == [], (
            f"MNIST should have no VANISHING flags, got: {[l['layer'] for l in vanishing]}"
        )
        assert exploding == [], (
            f"MNIST should have no EXPLODING flags, got: {[l['layer'] for l in exploding]}"
        )

    def test_builtin_grad_vanishing_passes_for_mnist(self, tmp_store):
        """GRAD_VANISHING builtin must PASS for a converged MNIST-like run."""
        from gradglass.analysis.builtins import test_grad_vanishing
        # 10 steps of healthy data (including the problematic fc2.* with near-zero mean)
        _write_grad_summaries(tmp_store, "mnist-healthy", [_healthy_mnist_layers()] * 10)
        ctx = _ctx(tmp_store, "mnist-healthy")
        result = test_grad_vanishing(ctx)
        assert result.status == TestStatus.PASS, (
            f"GRAD_VANISHING must PASS for a healthy MNIST run, got {result.status}: "
            f"{result.details}"
        )

    def test_builtin_grad_exploding_passes_for_mnist(self, tmp_store):
        """GRAD_EXPLODING builtin must PASS for a converged MNIST-like run."""
        from gradglass.analysis.builtins import test_grad_exploding
        _write_grad_summaries(tmp_store, "mnist-healthy-exp", [_healthy_mnist_layers()] * 10)
        ctx = _ctx(tmp_store, "mnist-healthy-exp")
        result = test_grad_exploding(ctx)
        assert result.status == TestStatus.PASS, (
            f"GRAD_EXPLODING must PASS for a healthy MNIST run, got {result.status}"
        )
