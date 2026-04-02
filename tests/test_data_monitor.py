import asyncio

import numpy as np

from gradglass.analysis.data_monitor import DatasetMonitorBuilder, PipelineStage
from gradglass.analysis.data_monitor.inspectors import detect_modality
from gradglass.artifacts import ArtifactStore
from gradglass.run import Run
from gradglass.server import create_app


def test_dataset_monitor_builder_persists_report_and_computes_stage_deltas(tmp_path):
    run_dir = tmp_path / "demo-run"
    builder = DatasetMonitorBuilder(
        "classification",
        dataset_name="demo-dataset",
        task_hint="tabular",
        run_dir=run_dir,
        run_id="demo-run",
    )
    raw_rows = [
        {"feature_a": 1.0, "feature_b": 2.0, "optional": "x"},
        {"feature_a": 3.0, "feature_b": 4.0, "optional": None},
        {"feature_a": 5.0, "feature_b": 6.0, "optional": "z"},
    ]
    cleaned_rows = [
        {"feature_a": 1.0, "feature_b": 2.0},
        {"feature_a": 5.0, "feature_b": 6.0},
    ]

    builder.record_stage(PipelineStage.RAW_DATA, split="train", data=raw_rows, labels=np.array([0, 1, 0]))
    builder.record_stage(PipelineStage.CLEANING, split="train", data=cleaned_rows, labels=np.array([0, 0]))
    report = builder.finalize(save=True)

    snapshots = report.pipeline["snapshots"]
    assert len(snapshots) == 2
    assert snapshots[1]["dropped_samples"] == 1
    assert any(change["type"] == "removed_fields" for change in snapshots[1]["schema_changes"])
    assert report.metadata.recorded_stage_count == 2
    assert (run_dir / "analysis" / "dataset_monitor.json").exists()
    assert (run_dir / "analysis" / "dataset_monitor_summary.txt").exists()


def test_dataset_monitor_supports_multiple_modalities_and_unknown_safe_results(tmp_path):
    builder = DatasetMonitorBuilder("multimodal", dataset_name="mixed")
    builder.record_stage(PipelineStage.RAW_DATA, split="train", data=["hello world", "a much longer example"], labels=np.array([0, 1]))
    builder.record_stage(
        PipelineStage.RAW_DATA,
        split="validation",
        data=np.random.randn(3, 32, 32).astype(np.float32),
        labels=np.array([0, 1, 1]),
    )
    builder.record_stage(
        PipelineStage.RAW_DATA,
        split="test",
        data=np.random.randn(2, 16000).astype(np.float32),
        labels=np.array([1, 0]),
    )
    report = builder.finalize(save=False)

    assert set(report.metadata.available_splits) == {"train", "validation", "test"}
    assert report.composition.latest_by_split["train"].sequence_length_distribution["available"] is True
    assert report.composition.latest_by_split["validation"].image_size_distribution["available"] is True
    assert report.composition.latest_by_split["test"].audio_duration_distribution["available"] is True


def test_run_monitor_dataset_and_check_leakage_emit_artifacts(tmp_path):
    store = ArtifactStore(root=tmp_path)
    run = Run(name="monitor-demo", store=store, auto_open=False)

    builder = run.monitor_dataset("classification", dataset_name="run-data", task_hint="tabular")
    builder.record_stage(
        PipelineStage.SPLITTING,
        split="train",
        data=np.array([[1.0, 2.0], [2.0, 3.0], [1.0, 2.0]], dtype=np.float32),
        labels=np.array([0, 1, 0], dtype=np.int64),
    )
    builder.record_stage(
        PipelineStage.SPLITTING,
        split="test",
        data=np.array([[1.0, 2.0], [9.0, 9.0]], dtype=np.float32),
        labels=np.array([0, 1], dtype=np.int64),
    )
    report = builder.finalize(save=True)

    leakage = run.check_leakage(
        np.array([[1.0, 2.0], [2.0, 3.0], [1.0, 2.0]], dtype=np.float32),
        np.array([0, 1, 0], dtype=np.int64),
        np.array([[1.0, 2.0], [9.0, 9.0]], dtype=np.float32),
        np.array([0, 1], dtype=np.int64),
        max_samples=10,
        print_summary=False,
    )

    assert report.metadata.run_id == run.run_id
    assert leakage.num_failed >= 1
    assert (run.run_dir / "analysis" / "dataset_monitor.json").exists()
    assert (run.run_dir / "analysis" / "leakage_report.json").exists()


def test_data_monitor_and_leakage_endpoints(tmp_path):
    store = ArtifactStore(root=tmp_path)
    run_dir = store.ensure_run_dir("api-run")
    builder = DatasetMonitorBuilder(
        "classification",
        dataset_name="api-dataset",
        run_dir=run_dir,
        run_id="api-run",
    )
    builder.record_stage(
        PipelineStage.SPLITTING,
        split="train",
        data=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        labels=np.array([0, 1], dtype=np.int64),
    )
    builder.record_stage(
        PipelineStage.SPLITTING,
        split="test",
        data=np.array([[1.0, 2.0], [5.0, 6.0]], dtype=np.float32),
        labels=np.array([0, 1], dtype=np.int64),
    )
    builder.finalize(save=True)

    app = create_app(store)
    route_map = {
        route.path: route.endpoint
        for route in app.router.routes
        if hasattr(route, "path") and hasattr(route, "endpoint")
    }
    payload = asyncio.run(route_map["/api/runs/{run_id}/data-monitor"]("api-run"))
    assert payload["metadata"]["dataset_name"] == "api-dataset"
    assert payload["dashboard"]["summary"]["total_checks"] >= 1

    leakage_payload = asyncio.run(route_map["/api/runs/{run_id}/leakage"]("api-run"))
    assert "results" in leakage_payload
    assert any(result["check_id"] == "EXACT_OVERLAP" for result in leakage_payload["results"])


def test_detect_modality_covers_text_image_and_audio_like_inputs():
    assert detect_modality("hello world").value == "text"
    assert detect_modality(np.random.randn(32, 32).astype(np.float32)).value == "image"
    assert detect_modality(np.random.randn(16000, 2).astype(np.float32), task_hint="audio waveform").value == "audio"
