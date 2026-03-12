import json
import shutil
import tempfile
from pathlib import Path
import numpy as np
import pytest
from gradglass.artifacts import ArtifactStore
from gradglass.diff import weight_diff, full_diff, classify_severity, Severity, gradient_flow_analysis, prediction_diff, architecture_diff

@pytest.fixture
def tmp_store():
    tmpdir = tempfile.mkdtemp()
    store = ArtifactStore(root=tmpdir)
    yield store
    shutil.rmtree(tmpdir)

class TestWeightDiff:

    def test_identical_weights(self):
        w = np.random.randn(64, 64).astype(np.float32)
        result = weight_diff(w, w, layer_name='test')
        assert result.frob_norm == 0.0
        assert result.cos_sim == pytest.approx(1.0)
        assert result.percent_changed == 0.0
        assert result.severity == Severity.LOW

    def test_small_change(self):
        w_a = np.random.randn(64, 64).astype(np.float32)
        w_b = w_a + np.random.randn(64, 64).astype(np.float32) * 1e-05
        result = weight_diff(w_a, w_b, layer_name='test')
        assert result.frob_norm > 0
        assert result.cos_sim > 0.99
        assert result.severity in (Severity.LOW, Severity.MEDIUM)

    def test_large_change(self):
        w_a = np.random.randn(64, 64).astype(np.float32)
        w_b = np.random.randn(64, 64).astype(np.float32)
        result = weight_diff(w_a, w_b, layer_name='test')
        assert result.frob_norm > 0.1
        assert result.severity in (Severity.HIGH, Severity.CRITICAL)

    def test_delta_shape(self):
        w_a = np.random.randn(32, 16).astype(np.float32)
        w_b = np.random.randn(32, 16).astype(np.float32)
        result = weight_diff(w_a, w_b, layer_name='test')
        assert result.delta.shape == (32, 16)
        assert result.shape == (32, 16)

class TestSeverity:

    def test_low(self):
        assert classify_severity(0.01, 0.999, 0.02) == Severity.LOW

    def test_medium(self):
        assert classify_severity(0.1, 0.98, 0.1) == Severity.MEDIUM

    def test_high(self):
        assert classify_severity(0.2, 0.95, 0.3) == Severity.HIGH

    def test_critical(self):
        assert classify_severity(0.5, 0.8, 0.6) == Severity.CRITICAL

    def test_worst_case_wins(self):
        result = classify_severity(0.01, 0.999, 0.25)
        assert result == Severity.HIGH

class TestFullDiff:

    def test_full_diff(self):
        weights_a = {'layer1.weight': np.random.randn(64, 64).astype(np.float32), 'layer2.weight': np.random.randn(128, 64).astype(np.float32)}
        weights_b = {'layer1.weight': weights_a['layer1.weight'] + np.random.randn(64, 64).astype(np.float32) * 0.001, 'layer2.weight': np.random.randn(128, 64).astype(np.float32)}
        result = full_diff(weights_a, weights_b, run_id='test', step_a=100, step_b=200)
        assert result.run_id == 'test'
        assert result.step_a == 100
        assert result.step_b == 200
        assert len(result.layers) == 2
        assert result.summary['total_layers'] == 2

    def test_to_dict(self):
        weights_a = {'w': np.ones((4, 4), dtype=np.float32)}
        weights_b = {'w': np.ones((4, 4), dtype=np.float32) * 1.1}
        result = full_diff(weights_a, weights_b)
        d = result.to_dict(include_deltas=True)
        assert 'layers' in d
        assert 'summary' in d
        assert 'delta_histogram' in d['layers'][0]

class TestGradientFlowAnalysis:

    def test_empty(self):
        assert gradient_flow_analysis([]) == []

    def test_vanishing_detection(self):
        summaries = [{'step': 100, 'layers': {'fc.weight': {'mean': 1e-08, 'var': 1e-12, 'max': 1e-09, 'norm': 1e-08}}}]
        analysis = gradient_flow_analysis(summaries)
        assert len(analysis) == 1
        assert 'VANISHING' in analysis[0]['flags'] or 'DEAD' in analysis[0]['flags']

    def test_exploding_detection(self):
        summaries = [{'step': 100, 'layers': {'fc.weight': {'mean': 500, 'var': 100, 'max': 1000, 'norm': 800}}}]
        analysis = gradient_flow_analysis(summaries)
        assert 'EXPLODING' in analysis[0]['flags']

class TestPredictionDiff:

    def test_label_flips(self):
        pred_a = {'step': 100, 'y_pred': [0, 1, 2, 3, 4]}
        pred_b = {'step': 200, 'y_pred': [0, 1, 3, 3, 0]}
        result = prediction_diff(pred_a, pred_b)
        assert result['label_flips'] == 2

    def test_confidence_change(self):
        pred_a = {'step': 100, 'confidence': [0.9, 0.8, 0.7]}
        pred_b = {'step': 200, 'confidence': [0.95, 0.85, 0.75]}
        result = prediction_diff(pred_a, pred_b)
        assert result['confidence_delta'] > 0

class TestArchitectureDiff:

    def test_identical(self):
        arch = {'layers': [{'id': 'conv1', 'type': 'Conv2D'}], 'edges': []}
        result = architecture_diff(arch, arch)
        assert result['is_identical'] is True

    def test_added_layer(self):
        arch_a = {'layers': [{'id': 'conv1', 'type': 'Conv2D'}], 'edges': []}
        arch_b = {'layers': [{'id': 'conv1', 'type': 'Conv2D'}, {'id': 'conv2', 'type': 'Conv2D'}], 'edges': []}
        result = architecture_diff(arch_a, arch_b)
        assert len(result['added_layers']) == 1
        assert result['is_identical'] is False

class TestArtifactStore:

    def test_create_run_dir(self, tmp_store):
        run_dir = tmp_store.ensure_run_dir('test-run')
        assert run_dir.exists()
        assert (run_dir / 'checkpoints').exists()
        assert (run_dir / 'gradients').exists()

    def test_save_and_load_checkpoint(self, tmp_store):
        run_dir = tmp_store.ensure_run_dir('test-run')
        weights = {'layer1': np.random.randn(10, 10).astype(np.float32)}
        ckpt_path = run_dir / 'checkpoints' / 'step_100.npz'
        np.savez_compressed(str(ckpt_path), **weights)
        loaded = tmp_store.load_checkpoint('test-run', 100)
        np.testing.assert_array_almost_equal(loaded['layer1'], weights['layer1'])

    def test_metrics(self, tmp_store):
        run_dir = tmp_store.ensure_run_dir('test-run')
        metrics_path = run_dir / 'metrics.jsonl'
        with open(metrics_path, 'w') as f:
            f.write('{"step":1,"loss":0.5}\n')
            f.write('{"step":2,"loss":0.3}\n')
        metrics = tmp_store.get_metrics('test-run')
        assert len(metrics) == 2
        assert metrics[1]['loss'] == 0.3

    def test_list_runs(self, tmp_store):
        run_dir = tmp_store.ensure_run_dir('test-run')
        meta = {'name': 'test', 'status': 'complete'}
        with open(run_dir / 'metadata.json', 'w') as f:
            json.dump(meta, f)
        runs = tmp_store.list_runs()
        assert len(runs) == 1
        assert runs[0]['name'] == 'test'
