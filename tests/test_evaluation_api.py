from __future__ import annotations

import json
import shutil
import tempfile

import pytest
from fastapi.testclient import TestClient

from gradglass.artifacts import ArtifactStore
from gradglass.server import create_app


@pytest.fixture
def tmp_store():
    tmpdir = tempfile.mkdtemp()
    store = ArtifactStore(root=tmpdir)
    yield store
    shutil.rmtree(tmpdir)


def _seed_metadata(store: ArtifactStore, run_id: str, *, task: str, enable_benchmarks: bool | None = None) -> None:
    run_dir = store.ensure_run_dir(run_id)
    config = {'task': task}
    if enable_benchmarks is not None:
        config['enable_benchmarks'] = enable_benchmarks
    meta = {
        'name': run_id,
        'run_id': run_id,
        'framework': 'pytorch',
        'status': 'complete',
        'start_time_epoch': 100.0,
        'config': config,
    }
    (run_dir / 'metadata.json').write_text(json.dumps(meta))


def test_eval_endpoint_returns_structured_classification_report(tmp_store):
    run_id = 'classification-eval-run'
    _seed_metadata(tmp_store, run_id, task='multiclass_classification')
    run_dir = tmp_store.get_run_dir(run_id)

    with open(run_dir / 'metrics.jsonl', 'w') as f:
        f.write(json.dumps({'step': 1, 'loss': 1.2, 'val_loss': 1.3}) + '\n')
        f.write(json.dumps({'step': 2, 'loss': 0.9, 'val_loss': 1.0}) + '\n')

    pred_dir = run_dir / 'predictions'
    (pred_dir / 'probe_step_1.json').write_text(
        json.dumps(
            {
                'step': 1,
                'y_true': [0, 1, 1, 0],
                'y_pred': [0, 0, 1, 0],
                'confidence': [0.92, 0.85, 0.94, 0.81],
                'prediction_type': 'class_scores',
            }
        )
    )
    (pred_dir / 'probe_step_2.json').write_text(
        json.dumps(
            {
                'step': 2,
                'y_true': [0, 1, 1, 0],
                'y_pred': [0, 1, 1, 0],
                'confidence': [0.96, 0.91, 0.93, 0.88],
                'prediction_type': 'class_scores',
            }
        )
    )

    client = TestClient(create_app(tmp_store))
    response = client.get(f'/api/runs/{run_id}/eval')

    assert response.status_code == 200
    payload = response.json()
    report = payload['report']

    assert payload['run_id'] == run_id
    assert report['inferred_task_type'] == 'classification'
    assert report['confidence_in_task_inference'] > 0.5
    assert report['benchmark_state']['enabled'] is False
    assert report['benchmark_state']['eligible_families'] == []
    assert report['benchmark_state']['message'] == 'Benchmarks disabled for this run.'
    assert 'vision' in report['task_type_distribution']
    assert report['trend_analysis']['status'] == 'improving'
    assert report['performance_summary']['headline_metrics'][0]['name'] == 'accuracy'
    assert report['error_analysis']['generalization']['status'] == 'insufficient_data'
    assert report['evaluations'][-1]['confusion_matrix']['matrix'] == [[2, 0], [0, 2]]
    assert any(metric['name'] == 'macro_f1' for metric in report['selected_metrics'])


def test_eval_endpoint_falls_back_to_metrics_only_forecasting_report(tmp_store):
    run_id = 'forecast-eval-run'
    _seed_metadata(tmp_store, run_id, task='time-series/forecasting')
    run_dir = tmp_store.get_run_dir(run_id)

    with open(run_dir / 'metrics.jsonl', 'w') as f:
        f.write(json.dumps({'step': 1, 'val_rmse': 0.83, 'val_mae': 0.61, 'val_mape': 0.19}) + '\n')
        f.write(json.dumps({'step': 2, 'val_rmse': 0.74, 'val_mae': 0.56, 'val_mape': 0.17}) + '\n')
        f.write(json.dumps({'step': 3, 'val_rmse': 0.68, 'val_mae': 0.51, 'val_mape': 0.14}) + '\n')

    client = TestClient(create_app(tmp_store))
    response = client.get(f'/api/runs/{run_id}/eval')

    assert response.status_code == 200
    payload = response.json()
    report = payload['report']

    assert report['inferred_task_type'] == 'time_series_forecasting'
    assert report['performance_summary']['headline_metrics'][0]['name'] == 'rmse'
    assert report['evaluations'][-1]['rmse'] == pytest.approx(0.68)
    assert report['trend_analysis']['status'] == 'improving'
    assert report['missing_artifacts']
    assert 'metrics only' in report['missing_artifacts'][0].lower()


def test_eval_endpoint_enabling_benchmarks_does_not_activate_regression_families(tmp_store):
    run_id = 'regression-bench-run'
    _seed_metadata(tmp_store, run_id, task='regression', enable_benchmarks=True)
    run_dir = tmp_store.get_run_dir(run_id)

    with open(run_dir / 'metrics.jsonl', 'w') as f:
        f.write(json.dumps({'step': 1, 'val_rmse': 0.83, 'val_mae': 0.61, 'val_mse': 0.69}) + '\n')
        f.write(json.dumps({'step': 2, 'val_rmse': 0.74, 'val_mae': 0.56, 'val_mse': 0.55}) + '\n')

    client = TestClient(create_app(tmp_store))
    response = client.get(f'/api/runs/{run_id}/eval')

    assert response.status_code == 200
    report = response.json()['report']

    assert report['inferred_task_type'] == 'regression'
    assert report['benchmark_state']['enabled'] is True
    assert report['benchmark_state']['eligible_families'] == []
    assert 'do not have a compatible benchmark family' in report['benchmark_state']['message']


def test_eval_endpoint_returns_llm_benchmark_family_when_explicitly_enabled(tmp_store):
    run_id = 'seqgen-bench-run'
    _seed_metadata(tmp_store, run_id, task='sequence_generation', enable_benchmarks=True)
    run_dir = tmp_store.get_run_dir(run_id)

    with open(run_dir / 'metrics.jsonl', 'w') as f:
        f.write(json.dumps({'step': 1, 'bleu': 0.31, 'rouge_l': 0.42, 'semantic_similarity': 0.56}) + '\n')
        f.write(json.dumps({'step': 2, 'bleu': 0.38, 'rouge_l': 0.49, 'semantic_similarity': 0.61}) + '\n')

    client = TestClient(create_app(tmp_store))
    response = client.get(f'/api/runs/{run_id}/eval')

    assert response.status_code == 200
    report = response.json()['report']

    assert report['inferred_task_type'] == 'sequence_generation'
    assert report['benchmark_state']['enabled'] is True
    assert report['benchmark_state']['eligible_families'] == ['llm']
    assert report['benchmark_state']['message'] == 'Compatible benchmark suites: LLM Benchmarks.'


def test_eval_endpoint_returns_vision_benchmark_family_when_explicitly_enabled(tmp_store):
    run_id = 'vision-bench-run'
    _seed_metadata(tmp_store, run_id, task='vision', enable_benchmarks=True)
    run_dir = tmp_store.get_run_dir(run_id)

    with open(run_dir / 'metrics.jsonl', 'w') as f:
        f.write(json.dumps({'step': 1, 'top_1_accuracy': 0.72, 'top_5_accuracy': 0.94}) + '\n')
        f.write(json.dumps({'step': 2, 'top_1_accuracy': 0.78, 'top_5_accuracy': 0.97}) + '\n')

    client = TestClient(create_app(tmp_store))
    response = client.get(f'/api/runs/{run_id}/eval')

    assert response.status_code == 200
    report = response.json()['report']

    assert report['inferred_task_type'] == 'vision'
    assert report['benchmark_state']['enabled'] is True
    assert report['benchmark_state']['eligible_families'] == ['vision']
    assert report['benchmark_state']['message'] == 'Compatible benchmark suites: Vision Benchmarks.'
