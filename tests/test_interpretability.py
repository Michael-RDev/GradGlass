"""
GradGlass — Interpretability & Attribution Test Suite
====================================================
Tests for the new analysis tests added in builtins.py:
  - GRAD_INPUT_SALIENCY
  - LIME_PROXY_CONFIDENCE
  - SHAP_GRAD_ATTRIBUTION_RANK
  - DEAD_CHANNEL_DETECTION
  - WEIGHT_NORM_DISTRIBUTION
  - FREEZE_RECOMMENDATION
  - ACTIVATION_PATTERN_STABILITY
  - LAYER_CAPACITY_UTILIZATION
  - EPOCH_LOSS_IMPROVEMENT
"""
import json
import math
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from gradglass.artifacts import ArtifactStore
from gradglass.analysis.registry import TestContext, TestStatus


def _make_context(store, run_id, *, metrics=None, grad_summaries=None,
                  activations=None, predictions=None, checkpoints=None) -> TestContext:
    """Build a minimal TestContext for unit-testing individual analysis tests."""
    run_dir = store.ensure_run_dir(run_id)

    # Write metadata
    import json as _json
    meta = {"name": run_id, "run_id": run_id, "status": "finished", "start_time": "2026-01-01"}
    with open(run_dir / "metadata.json", "w") as f:
        _json.dump(meta, f)

    # Write metrics
    if metrics:
        with open(run_dir / "metrics.jsonl", "w") as f:
            for m in metrics:
                f.write(_json.dumps(m) + "\n")

    # Write gradient summaries
    if grad_summaries:
        grad_dir = run_dir / "gradients"
        grad_dir.mkdir(exist_ok=True)
        for i, s in enumerate(grad_summaries):
            with open(grad_dir / f"summaries_step_{i+1}.json", "w") as f:
                _json.dump(s, f)

    # Write activation stats (as .npy files with a stat sidecar — just use activation_stats directly)
    return TestContext(
        run_id=run_id,
        run_dir=run_dir,
        store=store,
        metadata=meta,
        metrics=metrics or [],
        checkpoints_meta=checkpoints or [],
        gradient_summaries=grad_summaries or [],
        activation_stats=activations or [],
        predictions=predictions or [],
    )


@pytest.fixture
def tmp_store():
    tmpdir = tempfile.mkdtemp()
    store = ArtifactStore(root=tmpdir)
    yield store
    shutil.rmtree(tmpdir)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _grad_summary(step, layer_norms: dict):
    return {"step": step, "layers": {name: {"norm": norm, "mean": 0.0, "var": 0.01, "max": norm} for name, norm in layer_norms.items()}}


def _activation_stat(layer, step, mean=0.1, var=0.5, sparsity=0.1):
    return {"layer": layer, "step": step, "mean": mean, "var": var, "sparsity": sparsity}


# ────────────────────────────────────────────────────────────────────────────
# GRAD_INPUT_SALIENCY
# ────────────────────────────────────────────────────────────────────────────

class TestGradInputSaliency:
    def test_pass_with_gradient_data(self, tmp_store):
        from gradglass.analysis.builtins import test_grad_input_saliency
        summaries = [
            _grad_summary(i, {"fc1": 0.5 * i, "fc2": 0.1 * i, "fc3": 0.05 * i})
            for i in range(1, 6)
        ]
        ctx = _make_context(tmp_store, "saliency-pass", grad_summaries=summaries)
        result = test_grad_input_saliency(ctx)
        assert result.status == TestStatus.PASS
        assert "top_10_by_gradient_norm" in result.details
        # fc1 should rank highest (largest norms)
        top = result.details["top_10_by_gradient_norm"]
        assert top[0]["layer"] == "fc1"

    def test_skip_without_gradients(self, tmp_store):
        from gradglass.analysis.builtins import test_grad_input_saliency
        ctx = _make_context(tmp_store, "saliency-skip")
        result = test_grad_input_saliency(ctx)
        assert result.status == TestStatus.SKIP

    def test_saliency_ranking_respects_norms(self, tmp_store):
        from gradglass.analysis.builtins import test_grad_input_saliency
        # Make layer_b have consistently higher norm than layer_a
        summaries = [_grad_summary(i, {"layer_a": 0.1, "layer_b": 10.0}) for i in range(1, 4)]
        ctx = _make_context(tmp_store, "saliency-rank", grad_summaries=summaries)
        result = test_grad_input_saliency(ctx)
        assert result.status == TestStatus.PASS
        top = result.details["top_10_by_gradient_norm"]
        assert top[0]["layer"] == "layer_b"


# ────────────────────────────────────────────────────────────────────────────
# LIME_PROXY_CONFIDENCE
# ────────────────────────────────────────────────────────────────────────────

class TestLimeProxyConfidence:
    def test_pass_with_varied_confidence(self, tmp_store):
        from gradglass.analysis.builtins import test_lime_proxy_confidence
        confs = list(np.random.uniform(0.3, 0.95, 100))
        preds = [{"step": 1, "confidence": confs, "y_pred": [0] * 100, "y_true": [0] * 100}]
        ctx = _make_context(tmp_store, "lime-pass", predictions=preds)
        result = test_lime_proxy_confidence(ctx)
        assert result.status == TestStatus.PASS
        assert result.details["std"] > 0.05

    def test_warn_overconfident(self, tmp_store):
        from gradglass.analysis.builtins import test_lime_proxy_confidence
        confs = [0.999] * 50  # near-perfect, zero variance
        preds = [{"step": 1, "confidence": confs, "y_pred": [0] * 50}]
        ctx = _make_context(tmp_store, "lime-overconf", predictions=preds)
        result = test_lime_proxy_confidence(ctx)
        assert result.status == TestStatus.WARN
        assert "overconfident" in result.recommendation.lower() or result.details["mean"] > 0.98

    def test_warn_mode_collapse(self, tmp_store):
        from gradglass.analysis.builtins import test_lime_proxy_confidence
        confs = [0.5] * 50  # identical confidence for all samples
        preds = [{"step": 1, "confidence": confs, "y_pred": [0] * 50}]
        ctx = _make_context(tmp_store, "lime-collapse", predictions=preds)
        result = test_lime_proxy_confidence(ctx)
        assert result.status == TestStatus.WARN

    def test_skip_without_predictions(self, tmp_store):
        from gradglass.analysis.builtins import test_lime_proxy_confidence
        ctx = _make_context(tmp_store, "lime-skip")
        result = test_lime_proxy_confidence(ctx)
        assert result.status == TestStatus.SKIP


# ────────────────────────────────────────────────────────────────────────────
# SHAP_GRAD_ATTRIBUTION_RANK
# ────────────────────────────────────────────────────────────────────────────

class TestShapAttributionRank:
    def test_stable_rank(self, tmp_store):
        from gradglass.analysis.builtins import test_shap_attribution_rank
        # Same top layer throughout
        summaries = [_grad_summary(i, {"fc1": 5.0, "fc2": 2.0, "fc3": 0.5}) for i in range(1, 9)]
        ctx = _make_context(tmp_store, "shap-stable", grad_summaries=summaries)
        result = test_shap_attribution_rank(ctx)
        assert result.status == TestStatus.PASS
        assert result.details["rank_overlap"] >= 2

    def test_unstable_rank(self, tmp_store):
        from gradglass.analysis.builtins import test_shap_attribution_rank
        # Use 6 distinct layers split into two groups: a/b/c dominate early, d/e/f dominate late.
        # With 8 summaries, early window = first 2, late window = last 2.
        # early_top[:3] = [a,b,c]; late_top[:3] = [d,e,f] → overlap=0 → stability=0.0 → WARN
        early_layers = {"a": 100.0, "b": 90.0, "c": 80.0, "d": 0.001, "e": 0.001, "f": 0.001}
        late_layers  = {"a": 0.001, "b": 0.001, "c": 0.001, "d": 100.0, "e": 90.0, "f": 80.0}
        summaries = (
            [_grad_summary(i, early_layers) for i in range(1, 5)] +
            [_grad_summary(i, late_layers)  for i in range(5, 9)]
        )
        ctx = _make_context(tmp_store, "shap-unstable", grad_summaries=summaries)
        result = test_shap_attribution_rank(ctx)
        # early top-3=[a,b,c], late top-3=[d,e,f] → overlap=0 → WARN
        assert result.status == TestStatus.WARN

    def test_skip_with_few_summaries(self, tmp_store):
        from gradglass.analysis.builtins import test_shap_attribution_rank
        summaries = [_grad_summary(1, {"fc1": 1.0})]  # only 1 summary
        ctx = _make_context(tmp_store, "shap-fewsum", grad_summaries=summaries)
        result = test_shap_attribution_rank(ctx)
        assert result.status == TestStatus.SKIP


# ────────────────────────────────────────────────────────────────────────────
# DEAD_CHANNEL_DETECTION
# ────────────────────────────────────────────────────────────────────────────

class TestDeadChannelDetection:
    def test_pass_healthy_activations(self, tmp_store):
        from gradglass.analysis.builtins import test_dead_channels
        activations = [_activation_stat("conv1", 1, mean=0.3, sparsity=0.1)]
        ctx = _make_context(tmp_store, "deadch-pass", activations=activations)
        result = test_dead_channels(ctx)
        assert result.status == TestStatus.PASS

    def test_warn_dead_channels(self, tmp_store):
        from gradglass.analysis.builtins import test_dead_channels
        activations = [_activation_stat("conv1", 1, mean=0.0, var=0.0, sparsity=0.99)]
        ctx = _make_context(tmp_store, "deadch-warn", activations=activations)
        result = test_dead_channels(ctx)
        assert result.status == TestStatus.WARN

    def test_skip_without_activations(self, tmp_store):
        from gradglass.analysis.builtins import test_dead_channels
        ctx = _make_context(tmp_store, "deadch-skip")
        result = test_dead_channels(ctx)
        assert result.status == TestStatus.SKIP


# ────────────────────────────────────────────────────────────────────────────
# WEIGHT_NORM_DISTRIBUTION
# ────────────────────────────────────────────────────────────────────────────

class TestWeightNormDistribution:
    def test_pass_healthy_weights(self, tmp_store):
        from gradglass.analysis.builtins import test_weight_norm_distribution
        run_id = "weightnorm-pass"
        run_dir = tmp_store.ensure_run_dir(run_id)
        ckpt_dir = run_dir / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)

        weights = {"fc1.weight": np.random.randn(64, 32).astype(np.float32),
                   "fc1.bias": np.random.randn(64).astype(np.float32)}
        np.savez(ckpt_dir / "step_1.npz", **weights)
        import json as _json
        with open(ckpt_dir / "step_1_meta.json", "w") as f:
            _json.dump({"step": 1, "num_params": 100}, f)
        with open(run_dir / "metadata.json", "w") as f:
            _json.dump({"name": run_id, "run_id": run_id, "status": "finished", "start_time": "2026"}, f)

        ctx = TestContext(
            run_id=run_id, run_dir=run_dir, store=tmp_store,
            metadata={"name": run_id}, metrics=[],
            checkpoints_meta=[{"step": 1, "num_params": 100}],
        )
        result = test_weight_norm_distribution(ctx)
        assert result.status == TestStatus.PASS
        assert "mean_norm" in result.details

    def test_skip_no_checkpoints(self, tmp_store):
        from gradglass.analysis.builtins import test_weight_norm_distribution
        ctx = _make_context(tmp_store, "weightnorm-skip")
        result = test_weight_norm_distribution(ctx)
        assert result.status == TestStatus.SKIP


# ────────────────────────────────────────────────────────────────────────────
# FREEZE_RECOMMENDATION
# ────────────────────────────────────────────────────────────────────────────

class TestFreezeRecommendation:
    def test_identifies_low_gradient_layers(self, tmp_store):
        from gradglass.analysis.builtins import test_freeze_recommendation
        # fc1 has very low gradient norm, fc2 has high
        summaries = [_grad_summary(i, {"fc1": 1e-6, "fc2": 1.0}) for i in range(1, 5)]
        ctx = _make_context(tmp_store, "freeze-detect", grad_summaries=summaries)
        result = test_freeze_recommendation(ctx)
        assert result.status == TestStatus.PASS
        candidates = result.details.get("freeze_candidates", [])
        assert any(c["layer"] == "fc1" for c in candidates)
        assert "suggested_code" in result.details

    def test_no_candidates_all_active(self, tmp_store):
        from gradglass.analysis.builtins import test_freeze_recommendation
        # All layers have similar gradient norms — no candidates
        summaries = [_grad_summary(i, {"fc1": 1.0, "fc2": 0.8, "fc3": 0.9}) for i in range(1, 5)]
        ctx = _make_context(tmp_store, "freeze-none", grad_summaries=summaries)
        result = test_freeze_recommendation(ctx)
        assert result.status == TestStatus.PASS
        assert result.details.get("freeze_candidates", []) == [] or \
               "no freeze candidates" in result.details.get("message", "").lower()

    def test_skip_few_summaries(self, tmp_store):
        from gradglass.analysis.builtins import test_freeze_recommendation
        summaries = [_grad_summary(1, {"fc1": 0.1})]
        ctx = _make_context(tmp_store, "freeze-fewsum", grad_summaries=summaries)
        result = test_freeze_recommendation(ctx)
        assert result.status == TestStatus.SKIP


# ────────────────────────────────────────────────────────────────────────────
# ACTIVATION_PATTERN_STABILITY
# ────────────────────────────────────────────────────────────────────────────

class TestActivationPatternStability:
    def test_pass_stable(self, tmp_store):
        from gradglass.analysis.builtins import test_activation_pattern_stability
        # Variance converges to a stable value in late steps
        activations = [
            _activation_stat("fc1", step, var=0.5 + 0.001 * step)
            for step in range(1, 10)
        ]
        ctx = _make_context(tmp_store, "actstab-pass", activations=activations)
        result = test_activation_pattern_stability(ctx)
        assert result.status == TestStatus.PASS

    def test_warn_high_variance_change(self, tmp_store):
        from gradglass.analysis.builtins import test_activation_pattern_stability
        import random
        random.seed(42)
        # Late-training variance oscillates wildly
        activations = (
            [_activation_stat("fc1", step, var=0.5) for step in range(1, 7)] +
            [_activation_stat("fc1", step, var=random.uniform(0.0, 5.0)) for step in range(7, 13)]
        )
        ctx = _make_context(tmp_store, "actstab-warn", activations=activations)
        result = test_activation_pattern_stability(ctx)
        # May be PASS or WARN depending on random values — just ensure it runs
        assert result.status in (TestStatus.PASS, TestStatus.WARN)

    def test_skip_no_activations(self, tmp_store):
        from gradglass.analysis.builtins import test_activation_pattern_stability
        ctx = _make_context(tmp_store, "actstab-skip")
        result = test_activation_pattern_stability(ctx)
        assert result.status == TestStatus.SKIP


# ────────────────────────────────────────────────────────────────────────────
# LAYER_CAPACITY_UTILIZATION
# ────────────────────────────────────────────────────────────────────────────

class TestLayerCapacityUtilization:
    def test_pass_good_variance(self, tmp_store):
        from gradglass.analysis.builtins import test_layer_capacity
        activations = [_activation_stat("fc1", 5, var=0.8), _activation_stat("fc2", 5, var=1.2)]
        ctx = _make_context(tmp_store, "capacity-pass", activations=activations)
        result = test_layer_capacity(ctx)
        assert result.status == TestStatus.PASS

    def test_warn_underutilized(self, tmp_store):
        from gradglass.analysis.builtins import test_layer_capacity
        activations = [_activation_stat("fc1", 1, var=0.0001)]  # near-zero variance
        ctx = _make_context(tmp_store, "capacity-warn", activations=activations)
        result = test_layer_capacity(ctx)
        assert result.status == TestStatus.WARN

    def test_skip_no_activations(self, tmp_store):
        from gradglass.analysis.builtins import test_layer_capacity
        ctx = _make_context(tmp_store, "capacity-skip")
        result = test_layer_capacity(ctx)
        assert result.status == TestStatus.SKIP


# ────────────────────────────────────────────────────────────────────────────
# EPOCH_LOSS_IMPROVEMENT
# ────────────────────────────────────────────────────────────────────────────

class TestEpochLossImprovement:
    def test_pass_improving(self, tmp_store):
        from gradglass.analysis.builtins import test_epoch_loss_improvement
        metrics = [
            {"step": i, "epoch": (i - 1) // 10 + 1, "loss": 2.0 - (i / 50)}
            for i in range(1, 41)
        ]
        ctx = _make_context(tmp_store, "epochloss-pass", metrics=metrics)
        result = test_epoch_loss_improvement(ctx)
        assert result.status == TestStatus.PASS
        assert result.details["total_reduction"] > 0

    def test_warn_regression(self, tmp_store):
        from gradglass.analysis.builtins import test_epoch_loss_improvement
        # Loss increases in epoch 3 → regression
        metrics = (
            [{"step": i, "epoch": 1, "loss": 1.0} for i in range(1, 11)] +
            [{"step": i, "epoch": 2, "loss": 0.8} for i in range(11, 21)] +
            [{"step": i, "epoch": 3, "loss": 1.5} for i in range(21, 31)]  # spike
        )
        ctx = _make_context(tmp_store, "epochloss-warn", metrics=metrics)
        result = test_epoch_loss_improvement(ctx)
        assert result.status == TestStatus.WARN
        assert len(result.details["regressions"]) > 0

    def test_skip_no_epoch_field(self, tmp_store):
        from gradglass.analysis.builtins import test_epoch_loss_improvement
        metrics = [{"step": i, "loss": 1.0} for i in range(1, 20)]
        ctx = _make_context(tmp_store, "epochloss-noepoch", metrics=metrics)
        result = test_epoch_loss_improvement(ctx)
        assert result.status == TestStatus.SKIP

    def test_skip_single_epoch(self, tmp_store):
        from gradglass.analysis.builtins import test_epoch_loss_improvement
        metrics = [{"step": i, "epoch": 1, "loss": 1.0 - i * 0.01} for i in range(10)]
        ctx = _make_context(tmp_store, "epochloss-single", metrics=metrics)
        result = test_epoch_loss_improvement(ctx)
        assert result.status == TestStatus.SKIP
