from __future__ import annotations

from gradglass.experiment_tracking import (
    build_overview_snapshot,
    infer_total_epochs_from_config,
    infer_total_steps_from_config,
)


def test_infer_total_steps_from_config():
    assert infer_total_steps_from_config({"total_steps": 400}) == 400
    assert infer_total_steps_from_config({"epochs": 3, "steps_per_epoch": 100}) == 300
    assert infer_total_steps_from_config({"n_estimators": 250}) == 250
    assert infer_total_steps_from_config({"max_iter": 75}) == 75
    assert infer_total_steps_from_config({"num_iterations": 80}) == 80
    assert infer_total_steps_from_config({}) is None


def test_infer_total_epochs_from_config():
    assert infer_total_epochs_from_config({"epochs": 5}) == 5
    assert infer_total_epochs_from_config({"num_epochs": 6}) == 6
    assert infer_total_epochs_from_config({"phase1_epochs": 2, "phase2_epochs": 3}) == 5
    assert infer_total_epochs_from_config({}) is None


def test_pytorch_overview_normalization_and_eta():
    metadata = {
        "framework": "pytorch",
        "status": "running",
        "start_time_epoch": 100.0,
        "config": {"epochs": 2, "steps_per_epoch": 5, "monitor": True},
    }
    metrics = [
        {"step": 1, "timestamp": 101.0, "loss": 1.0, "val_loss": 1.2, "lr": 0.01},
        {"step": 2, "timestamp": 102.0, "loss": 0.8, "val_loss": 1.0, "lr": 0.01},
        {"step": 3, "timestamp": 103.0, "loss": 0.7, "val_loss": 0.9, "lr": 0.005},
    ]
    runtime = {
        "status": "running",
        "heartbeat_ts": 103.2,
        "current_step": 3,
        "monitor_enabled": True,
        "resource_tracking_available": True,
    }

    snapshot = build_overview_snapshot(
        run_id="demo-run",
        metadata=metadata,
        metrics=metrics,
        runtime_state=runtime,
        now_ts=103.3,
    )

    assert snapshot["framework"] == "pytorch"
    assert snapshot["health_state"] == "HEALTHY"
    assert snapshot["current_step"] == 3
    assert snapshot["total_steps"] == 10
    assert snapshot["total_steps_source"] == "config"
    assert snapshot["eta_s"] is not None
    assert snapshot["eta_is_live"] is True
    assert snapshot["loss_history"][-1] == [3.0, 0.7]
    assert snapshot["val_loss_history"][-1] == [3.0, 0.9]
    assert snapshot["lr_history"][-1] == [3.0, 0.005]


def test_eta_unavailable_when_total_steps_unknown():
    metadata = {"framework": "pytorch", "status": "running", "start_time_epoch": 10.0, "config": {}}
    metrics = [
        {"step": 1, "timestamp": 11.0, "loss": 1.0},
        {"step": 2, "timestamp": 12.0, "loss": 0.9},
    ]

    snapshot = build_overview_snapshot(
        run_id="unknown-total",
        metadata=metadata,
        metrics=metrics,
        runtime_state={"status": "running", "heartbeat_ts": 12.0, "current_step": 2},
        now_ts=12.5,
    )

    assert snapshot["eta_s"] is None
    assert snapshot["total_steps_source"] == "unknown"
    assert "unknown" in snapshot["eta_reason"].lower()


def test_eta_infers_total_steps_from_epoch_progress():
    metadata = {"framework": "pytorch", "status": "running", "start_time_epoch": 10.0, "config": {"epochs": 4}}
    metrics = [
        {"step": 1, "timestamp": 11.0, "loss": 1.0, "epoch": 1},
        {"step": 5, "timestamp": 15.0, "loss": 0.8, "epoch": 1},
        {"step": 6, "timestamp": 16.0, "loss": 0.79, "epoch": 2},
        {"step": 10, "timestamp": 20.0, "loss": 0.7, "epoch": 2},
    ]

    snapshot = build_overview_snapshot(
        run_id="epoch-total",
        metadata=metadata,
        metrics=metrics,
        runtime_state={"status": "running", "heartbeat_ts": 20.0, "current_step": 10},
        now_ts=20.5,
    )

    assert snapshot["total_steps"] == 20
    assert snapshot["total_steps_source"] == "epoch_inference"
    assert snapshot["eta_s"] is not None


def test_epoch_inference_requires_multiple_epochs():
    metadata = {"framework": "pytorch", "status": "running", "start_time_epoch": 10.0, "config": {"epochs": 4}}
    metrics = [
        {"step": 1, "timestamp": 11.0, "loss": 1.0, "epoch": 1},
        {"step": 5, "timestamp": 15.0, "loss": 0.8, "epoch": 1},
    ]

    snapshot = build_overview_snapshot(
        run_id="epoch-single",
        metadata=metadata,
        metrics=metrics,
        runtime_state={"status": "running", "heartbeat_ts": 15.0, "current_step": 5},
        now_ts=15.2,
    )

    assert snapshot["total_steps"] is None
    assert snapshot["total_steps_source"] == "unknown"


def test_eta_recalibrates_when_active_progress_reaches_estimated_total():
    metadata = {"framework": "pytorch", "status": "running", "start_time_epoch": 10.0, "config": {"total_steps": 4}}
    metrics = [
        {"step": 1, "timestamp": 11.0, "loss": 1.0},
        {"step": 2, "timestamp": 12.0, "loss": 0.9},
        {"step": 3, "timestamp": 13.0, "loss": 0.85},
        {"step": 4, "timestamp": 14.0, "loss": 0.82},
        {"step": 5, "timestamp": 15.0, "loss": 0.8},
    ]

    snapshot = build_overview_snapshot(
        run_id="eta-recalibrate",
        metadata=metadata,
        metrics=metrics,
        runtime_state={"status": "running", "heartbeat_ts": 15.0, "current_step": 5},
        now_ts=15.2,
    )

    assert snapshot["eta_s"] is None
    assert "recalibrating" in snapshot["eta_reason"].lower()


def test_terminal_run_eta_is_zero():
    metadata = {"framework": "pytorch", "status": "complete", "start_time_epoch": 10.0, "config": {"total_steps": 4}}
    metrics = [
        {"step": 1, "timestamp": 11.0, "loss": 1.0},
        {"step": 2, "timestamp": 12.0, "loss": 0.9},
        {"step": 3, "timestamp": 13.0, "loss": 0.85},
        {"step": 4, "timestamp": 14.0, "loss": 0.82},
    ]

    snapshot = build_overview_snapshot(
        run_id="eta-complete",
        metadata=metadata,
        metrics=metrics,
        runtime_state={"status": "complete", "heartbeat_ts": 14.0, "current_step": 4},
        now_ts=15.0,
    )

    assert snapshot["eta_s"] == 0.0
    assert snapshot["eta_reason"] is None


def test_eta_smoothing_ignores_large_timing_spikes():
    metadata = {"framework": "pytorch", "status": "running", "start_time_epoch": 10.0, "config": {"total_steps": 20}}
    metrics = [
        {"step": 1, "timestamp": 11.0, "loss": 1.0},
        {"step": 2, "timestamp": 12.0, "loss": 0.9},
        {"step": 3, "timestamp": 13.0, "loss": 0.8},
        {"step": 4, "timestamp": 113.0, "loss": 0.7},
        {"step": 5, "timestamp": 114.0, "loss": 0.65},
        {"step": 6, "timestamp": 115.0, "loss": 0.6},
        {"step": 7, "timestamp": 116.0, "loss": 0.58},
        {"step": 8, "timestamp": 117.0, "loss": 0.56},
        {"step": 9, "timestamp": 118.0, "loss": 0.54},
        {"step": 10, "timestamp": 119.0, "loss": 0.52},
    ]

    snapshot = build_overview_snapshot(
        run_id="eta-stability",
        metadata=metadata,
        metrics=metrics,
        runtime_state={"status": "running", "heartbeat_ts": 119.0, "current_step": 10},
        now_ts=119.2,
    )

    assert snapshot["eta_s"] is not None
    assert snapshot["eta_s"] < 30.0


def test_health_state_transitions():
    metadata = {"framework": "pytorch", "status": "running", "start_time_epoch": 100.0, "config": {}}
    metrics = [
        {"step": 1, "timestamp": 101.0, "loss": 1.0},
        {"step": 2, "timestamp": 102.0, "loss": 0.9},
        {"step": 3, "timestamp": 103.0, "loss": 0.8},
    ]

    stalled = build_overview_snapshot(
        run_id="stalled",
        metadata=metadata,
        metrics=metrics,
        runtime_state={"status": "running", "heartbeat_ts": 103.0, "current_step": 3},
        now_ts=150.0,
    )
    assert stalled["health_state"] == "STALLED"

    failed = build_overview_snapshot(
        run_id="failed",
        metadata={**metadata, "status": "failed"},
        metrics=metrics,
        runtime_state={"status": "failed", "heartbeat_ts": 103.0, "current_step": 3, "fatal_exception": "boom"},
        now_ts=104.0,
    )
    assert failed["health_state"] == "FAILED"


def test_sklearn_loss_curve_fallback():
    metadata = {
        "framework": "sklearn",
        "status": "running",
        "start_time_epoch": 100.0,
        "config": {"monitor": False},
    }

    snapshot = build_overview_snapshot(
        run_id="sklearn-loss-curve",
        metadata=metadata,
        metrics=[],
        runtime_state={"status": "running", "heartbeat_ts": 104.0, "current_step": 0},
        sklearn_diagnostics=[{"step": 1, "n_iter": 3, "loss_curve": [0.8, 0.7, 0.5]}],
        now_ts=104.0,
    )

    assert snapshot["framework"] == "sklearn"
    assert snapshot["current_step"] == 3
    assert snapshot["total_steps"] == 3
    assert snapshot["total_steps_source"] == "diagnostics"
    assert snapshot["loss_history"] == [[1.0, 0.8], [2.0, 0.7], [3.0, 0.5]]


def test_xgboost_unknown_framework_inference_and_loss_mapping():
    metadata = {
        "framework": "unknown",
        "status": "running",
        "start_time_epoch": 10.0,
        "config": {"objective": "binary:logistic", "num_boost_round": 4, "eta": 0.1},
    }
    metrics = [
        {"step": 1, "timestamp": 11.0, "train_logloss": 0.7, "test_logloss": 0.75},
        {"step": 2, "timestamp": 12.0, "train_logloss": 0.6, "test_logloss": 0.7},
    ]

    snapshot = build_overview_snapshot(
        run_id="xgb-functional",
        metadata=metadata,
        metrics=metrics,
        runtime_state={"status": "running", "heartbeat_ts": 12.0, "current_step": 2},
        now_ts=12.2,
    )

    assert snapshot["framework"] == "xgboost"
    assert snapshot["total_steps"] == 4
    assert snapshot["total_steps_source"] == "config"
    assert snapshot["loss_history"][-1] == [2.0, 0.6]
    assert snapshot["val_loss_history"][-1] == [2.0, 0.7]
