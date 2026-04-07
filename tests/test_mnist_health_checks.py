from __future__ import annotations

import numpy as np

from gradglass.analysis.data_monitor import DatasetMonitorBuilder, PipelineStage
from gradglass.analysis.leakage import build_monitor_report_for_arrays, project_monitor_report_to_legacy
from gradglass.artifacts import ArtifactStore
from gradglass.run import Run

from tests._mnist import mnist_subset


TRAIN_INDICES = list(range(256))
TEST_INDICES = list(range(128))
CLEAN_POOL = 400


def _variance_filtered(
    train_x: np.ndarray, test_x: np.ndarray, *, min_std: float = 1e-3
) -> tuple[np.ndarray, np.ndarray]:
    train_std = np.std(train_x, axis=0)
    test_std = np.std(test_x, axis=0)
    keep = (train_std > min_std) | (test_std > min_std)
    return train_x[:, keep], test_x[:, keep]


def _standardize(train_x: np.ndarray, test_x: np.ndarray, *, use_combined_stats: bool) -> tuple[np.ndarray, np.ndarray]:
    reference = np.vstack([train_x, test_x]) if use_combined_stats else train_x
    mean = reference.mean(axis=0, dtype=np.float64)
    std = reference.std(axis=0, dtype=np.float64)
    std = np.where(std < 1e-6, 1.0, std)
    train_scaled = ((train_x - mean) / std).astype(np.float32)
    test_scaled = ((test_x - mean) / std).astype(np.float32)
    return train_scaled, test_scaled


def _result_by_id(report, check_id: str):
    return next(result for result in report.results if result.check_id == check_id)


def _clean_projected_mnist_pair() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_images, train_labels = mnist_subset("train", list(range(CLEAN_POOL)), flatten=True)
    test_images, test_labels = mnist_subset("test", list(range(CLEAN_POOL)), flatten=True)

    train_indices = [int(np.where(train_labels == label)[0][0]) for label in range(10)]
    test_indices = [int(np.where(test_labels == label)[0][0]) for label in range(10)]

    train_x = train_images[train_indices]
    test_x = test_images[test_indices]
    train_y = train_labels[train_indices]
    test_y = test_labels[test_indices]

    # Use a fixed projection so MNIST stays real-data-derived while the leakage
    # monitor sees compact tabular features rather than 784-long "audio-like" rows.
    basis = np.random.RandomState(7).normal(size=(train_x.shape[1], 16)).astype(np.float32)
    train_projected = train_x @ basis
    test_projected = test_x @ basis

    train_scaled, _ = _standardize(train_projected, test_projected, use_combined_stats=False)
    _, test_scaled = _standardize(test_projected, test_projected, use_combined_stats=False)
    return train_scaled, train_y.astype(np.int64), test_scaled, test_y.astype(np.int64)


def _scaler_probe_pair() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_x, train_y = mnist_subset("train", TRAIN_INDICES, flatten=True)
    test_x, test_y = mnist_subset("test", TEST_INDICES, flatten=True)
    train_x, test_x = _variance_filtered(train_x, test_x)
    return train_x[:, :256], train_y, test_x[:, :256], test_y


def test_mnist_clean_split_passes_leakage_checks_and_persists_artifacts(tmp_path):
    train_x, train_y, test_x, test_y = _clean_projected_mnist_pair()

    store = ArtifactStore(root=tmp_path)
    run = Run(name="mnist-clean", store=store, auto_open=False)

    monitor_report = build_monitor_report_for_arrays(
        train_x,
        train_y,
        test_x,
        test_y,
        dataset_name="MNIST Clean Split",
        run_dir=run.run_dir,
        run_id=run.run_id,
        save=True,
    )
    leakage = run.check_leakage(train_x, train_y, test_x, test_y, max_samples=512, print_summary=False)

    assert monitor_report.metadata.dataset_name == "MNIST Clean Split"
    assert leakage.passed is True
    assert leakage.num_failed == 0
    assert _result_by_id(leakage, "EXACT_OVERLAP").passed is True
    assert _result_by_id(leakage, "PREPROCESSING_LEAKAGE").passed is True
    assert (run.run_dir / "analysis" / "dataset_monitor.json").exists()
    assert (run.run_dir / "analysis" / "dataset_monitor_summary.txt").exists()
    assert (run.run_dir / "analysis" / "leakage_report.json").exists()


def test_mnist_exact_overlap_is_reported_as_unhealthy(tmp_path):
    train_x, train_y, test_x, test_y = _clean_projected_mnist_pair()

    test_x[:3] = train_x[:3]
    test_y[:3] = train_y[:3]

    store = ArtifactStore(root=tmp_path)
    run = Run(name="mnist-overlap", store=store, auto_open=False)
    leakage = run.check_leakage(train_x, train_y, test_x, test_y, max_samples=64, print_summary=False)

    exact_overlap = _result_by_id(leakage, "EXACT_OVERLAP")
    assert leakage.passed is False
    assert leakage.num_failed >= 1
    assert exact_overlap.passed is False
    assert exact_overlap.details["num_overlapping"] >= 3


def test_mnist_combined_standardization_triggers_preprocessing_leakage():
    train_x, train_y, test_x, test_y = _scaler_probe_pair()
    train_scaled, test_scaled = _standardize(train_x, test_x, use_combined_stats=True)

    report = project_monitor_report_to_legacy(
        build_monitor_report_for_arrays(
            train_scaled, train_y, test_scaled, test_y, dataset_name="MNIST Combined Stats", save=False
        )
    )

    scaler = _result_by_id(report, "PREPROCESSING_LEAKAGE")
    assert report.passed is False
    assert scaler.passed is False


def test_mnist_train_only_standardization_does_not_trigger_preprocessing_leakage():
    train_x, train_y, test_x, test_y = _scaler_probe_pair()
    train_scaled, test_scaled = _standardize(train_x, test_x, use_combined_stats=False)

    report = project_monitor_report_to_legacy(
        build_monitor_report_for_arrays(
            train_scaled, train_y, test_scaled, test_y, dataset_name="MNIST Train Stats", save=False
        )
    )

    scaler = _result_by_id(report, "PREPROCESSING_LEAKAGE")
    assert scaler.passed is True


def test_mnist_label_as_feature_triggers_target_correlation_failure():
    train_x, train_y, test_x, test_y = _clean_projected_mnist_pair()
    train_augmented = np.column_stack([train_x, train_y.astype(np.float32)])
    test_augmented = np.column_stack([test_x, test_y.astype(np.float32)])

    report = project_monitor_report_to_legacy(
        build_monitor_report_for_arrays(
            train_augmented.astype(np.float32),
            train_y,
            test_augmented.astype(np.float32),
            test_y,
            dataset_name="MNIST Target Correlation",
            save=False,
        )
    )

    target_correlation = _result_by_id(report, "TARGET_CORRELATION")
    assert report.passed is False
    assert target_correlation.passed is False
    assert target_correlation.details["num_suspicious"] >= 1


def test_mnist_vision_monitor_prefers_image_summaries():
    train_x, train_y = mnist_subset("train", list(range(24)), flatten=False)
    test_x, test_y = mnist_subset("test", list(range(16)), flatten=False)

    builder = DatasetMonitorBuilder("vision", dataset_name="MNIST Vision")
    builder.record_stage(PipelineStage.SPLITTING, split="train", data=train_x, labels=train_y)
    builder.record_stage(PipelineStage.SPLITTING, split="test", data=test_x, labels=test_y)
    report = builder.finalize(save=False)

    train_slice = report.composition.latest_by_split["train"]
    test_slice = report.composition.latest_by_split["test"]
    panel_titles = {panel.title for panel in report.dashboard.composition_panels}

    assert train_slice.image_size_distribution["available"] is True
    assert test_slice.image_size_distribution["available"] is True
    assert train_slice.outlier_features == []
    assert test_slice.outlier_features == []
    assert "Train Image Heights" in panel_titles
    assert "Train Image Widths" in panel_titles
    assert "Train Image Aspect Ratios" in panel_titles
    assert "Train Top Outlier Features" not in panel_titles
