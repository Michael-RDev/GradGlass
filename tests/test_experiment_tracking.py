from gradglass.experiment_tracking import build_overview_snapshot


def test_terminal_runs_use_final_event_time_for_elapsed_duration():
    metadata = {
        "framework": "pytorch",
        "status": "complete",
        "start_time_epoch": 100.0,
        "config": {},
    }
    runtime_state = {
        "status": "complete",
        "start_time_epoch": 100.0,
        "last_event_ts": 405.0,
        "heartbeat_ts": 405.0,
        "current_step": 12,
        "last_event": "finish",
    }
    metrics = [
        {"step": 1, "timestamp": 120.0, "loss": 1.2},
        {"step": 12, "timestamp": 404.0, "loss": 0.7},
    ]

    snapshot = build_overview_snapshot(
        run_id="demo-run",
        metadata=metadata,
        metrics=metrics,
        runtime_state=runtime_state,
        now_ts=1000.0,
    )

    assert snapshot["status"] == "completed"
    assert snapshot["elapsed_time_s"] == 305.0
    assert snapshot["eta_s"] == 0.0
