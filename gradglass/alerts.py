from __future__ import annotations

import json
from typing import Any, Optional

from gradglass.analysis.builtins import (
    test_grad_exploding,
    test_grad_nan_inf,
    test_grad_vanishing,
    test_loss_finite,
    test_loss_spikes,
    test_overfitting,
    test_train_val_gap,
    test_val_loss_divergence,
)
from gradglass.analysis.registry import TestContext
from gradglass.analysis.report import PostRunReport
from gradglass.diff import gradient_flow_analysis
from gradglass.experiment_tracking import build_overview_snapshot

SEVERITY_ORDER = {"CRITICAL": 3, "HIGH": 2, "MEDIUM": 1, "LOW": 0}
STATUS_ORDER = {"fail": 0, "warn": 1, "info": 2}

CTA_PATHS = {
    "Training Metrics": "/training",
    "Predictions": "/evaluation",
    "Checkpoint Diff": "/overview",
    "Gradient Flow": "/overview",
    "Model Structure": "/overview",
    "Activations": "/overview",
    "Data": "/data",
    "Distributed Training": "/infrastructure",
    "Runtime Health": "/infrastructure",
}

TEST_TITLES = {
    "LEARNING_RATE_LOGGED": "Learning rate is not being tracked",
    "WEIGHT_DIFF_SEVERITY_COUNTS": "Large checkpoint changes detected",
    "LABEL_FLIP_RATE": "Prediction labels changed significantly",
    "SEED_LOGGED": "Random seed was not recorded",
    "LOSS_FINITE": "Loss became NaN or Inf",
    "LOSS_SPIKE_DETECTION": "Sudden loss spikes detected",
    "TRAIN_VAL_GAP": "Large train/validation gap",
    "OVERFITTING_HEURISTIC": "Overfitting detected",
    "VAL_LOSS_DIVERGENCE": "Validation loss is diverging",
    "GRAD_NAN_INF": "NaN/Inf gradients detected",
    "GRAD_VANISHING": "Vanishing gradients detected",
    "GRAD_EXPLODING": "Exploding gradients detected",
}

TEST_RECOMMENDATION_FALLBACKS = {
    "LEARNING_RATE_LOGGED": "Pass optimizer to run.watch() so GradGlass can track LR changes.",
    "WEIGHT_DIFF_SEVERITY_COUNTS": "Review recent checkpoint changes and reduce learning-rate pressure if weights are shifting too aggressively.",
    "LABEL_FLIP_RATE": "Review evaluation drift and reduce training instability before promoting this checkpoint.",
    "SEED_LOGGED": "Pass seed= in run config so the run can be reproduced later.",
}

GROUPED_FLAG_META = {
    "NOISY": {
        "severity": "MEDIUM",
        "title": "Noisy gradients detected",
        "recommendation": "Gradients are noisy across multiple layers. Lower the learning rate, increase batch size, or add LR warmup.",
    },
    "DISTRIBUTION_SHIFT": {
        "severity": "MEDIUM",
        "title": "Gradient distribution shifted",
        "recommendation": "Gradient distributions changed sharply. Lower LR during fine-tuning or unfreeze layers more gradually.",
    },
    "DEAD": {
        "severity": "HIGH",
        "title": "Inactive layers detected",
        "recommendation": "Some layers appear inactive. Check initialization, activations, or whether layers were frozen too aggressively.",
    },
}


def build_alert_snapshot(
    store,
    run_id: str,
    *,
    metadata: Optional[dict[str, Any]] = None,
    metrics: Optional[list[dict[str, Any]]] = None,
    runtime_state: Optional[dict[str, Any]] = None,
    overview: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    metadata = metadata or store.get_run_metadata(run_id) or {}
    metrics = metrics if metrics is not None else store.get_metrics(run_id)
    runtime_state = runtime_state if runtime_state is not None else store.get_runtime_state(run_id)
    run_dir = store.get_run_dir(run_id)

    if overview is None:
        overview = build_overview_snapshot(
            run_id=run_id,
            metadata=metadata,
            metrics=metrics,
            runtime_state=runtime_state,
        )

    ctx = _build_live_context(store, run_id, run_dir, metadata, metrics)
    report = _load_analysis_report(store, run_id, run_dir, overview)

    dedupe_ids: set[str] = set()
    alerts: list[dict[str, Any]] = []

    for alert in _runtime_alerts(overview, runtime_state):
        _push_alert(alerts, dedupe_ids, alert)

    for alert in _analysis_report_alerts(report):
        _push_alert(alerts, dedupe_ids, alert)

    for alert in _live_builtin_alerts(ctx):
        _push_alert(alerts, dedupe_ids, alert)

    for alert in _grouped_gradient_flag_alerts(ctx):
        _push_alert(alerts, dedupe_ids, alert)

    alerts.sort(key=_alert_sort_key)
    summary = _build_summary(alerts, overview)

    return {
        "run_id": run_id,
        "status": overview.get("status"),
        "health_state": overview.get("health_state", "WARNING"),
        "summary": summary,
        "alerts": alerts,
    }


def _build_live_context(store, run_id, run_dir, metadata, metrics) -> TestContext:
    return TestContext(
        run_id=run_id,
        run_dir=run_dir,
        store=store,
        metadata=metadata,
        metrics=metrics,
        gradient_summaries=store.get_gradient_summaries(run_id),
    )


def _load_analysis_report(store, run_id, run_dir, overview: dict[str, Any]) -> Optional[dict[str, Any]]:
    report_path = run_dir / "analysis" / "report.json"
    if report_path.exists():
        try:
            return json.loads(report_path.read_text())
        except (OSError, json.JSONDecodeError):
            return None

    if overview.get("status") != "completed":
        return None

    try:
        report = PostRunReport.generate(
            run_id=run_id,
            store=store,
            run_dir=run_dir,
            save=True,
            print_summary=False,
        )
        return report.to_dict()
    except Exception:
        return None


def _runtime_alerts(overview: dict[str, Any], runtime_state: Optional[dict[str, Any]]) -> list[dict[str, Any]]:
    runtime_state = runtime_state or {}
    alerts: list[dict[str, Any]] = []
    status = (overview.get("status") or "").lower()
    health_state = (overview.get("health_state") or "WARNING").upper()
    status_reason = overview.get("status_reason") or ""
    heartbeat_age = _heartbeat_age_seconds(overview)

    if status == "failed":
        alerts.append(
            _alert(
                alert_id="RUNTIME_FAILED",
                source="runtime",
                status="fail",
                severity="CRITICAL",
                category="Runtime Health",
                title="Training run failed",
                message=status_reason or "The training process reported a fatal failure.",
                recommendation="Inspect the training logs, fix the fatal error, and restart from the last known good checkpoint.",
                details={"status_reason": status_reason} if status_reason else {},
                cta_path="/infrastructure",
            )
        )

    if status == "interrupted" and health_state != "STALLED":
        alerts.append(
            _alert(
                alert_id="RUNTIME_INTERRUPTED",
                source="runtime",
                status="fail",
                severity="HIGH",
                category="Runtime Health",
                title="Training process was interrupted",
                message=status_reason or "Heartbeat stopped and the training process no longer looks active.",
                recommendation="Confirm the worker process is still running, then resume or restart the run if it exited unexpectedly.",
                details={"status_reason": status_reason} if status_reason else {},
                cta_path="/infrastructure",
            )
        )

    if health_state == "STALLED":
        details = {}
        if heartbeat_age is not None:
            details["heartbeat_age_s"] = heartbeat_age
        alerts.append(
            _alert(
                alert_id="RUNTIME_STALLED",
                source="runtime",
                status="fail",
                severity="HIGH",
                category="Runtime Health",
                title="Training heartbeat stalled",
                message=status_reason or "GradGlass has not seen a fresh heartbeat recently, so live training may be stuck.",
                recommendation="Check the trainer process, data loader, and resource usage. Restart the run if the process is no longer making progress.",
                details=details,
                cta_path="/infrastructure",
            )
        )

    if status == "cancelled":
        alerts.append(
            _alert(
                alert_id="RUNTIME_CANCELLED",
                source="runtime",
                status="warn",
                severity="LOW",
                category="Runtime Health",
                title="Training was cancelled",
                message=status_reason or "This run stopped due to an explicit cancel or interrupt signal.",
                recommendation="No automatic fix is required unless the run stopped unexpectedly. Re-run if you still need fresh training artifacts.",
                details={"status_reason": status_reason} if status_reason else {},
                cta_path="/infrastructure",
            )
        )

    if runtime_state.get("resource_tracking_required") and not overview.get("resource_tracking_available"):
        alerts.append(
            _alert(
                alert_id="RESOURCE_TRACKING_MISSING",
                source="runtime",
                status="warn",
                severity="LOW",
                category="Runtime Health",
                title="Required resource tracking is unavailable",
                message="The run requested infrastructure telemetry, but resource tracking data is missing.",
                recommendation="Re-run with working telemetry access so CPU, memory, and accelerator pressure can be monitored.",
                details={"resource_tracking_available": False},
                cta_path="/infrastructure",
            )
        )

    return alerts


def _analysis_report_alerts(report: Optional[dict[str, Any]]) -> list[dict[str, Any]]:
    if not report:
        return []

    results = report.get("tests", {}).get("results", [])
    alerts = []
    for result in results:
        status = result.get("status")
        if status in {"pass", "skip"}:
            continue
        alerts.append(_alert_from_test_result(result, source="analysis"))
    return alerts


def _live_builtin_alerts(ctx: TestContext) -> list[dict[str, Any]]:
    tests = [
        test_loss_finite,
        test_loss_spikes,
        test_train_val_gap,
        test_overfitting,
        test_val_loss_divergence,
        test_grad_nan_inf,
        test_grad_vanishing,
        test_grad_exploding,
    ]
    alerts = []
    for fn in tests:
        result = fn(ctx).to_dict()
        if result.get("status") in {"pass", "skip"}:
            continue
        alerts.append(_alert_from_test_result(result, source="metrics" if result.get("category") == "Training Metrics" else "gradient"))
    return alerts


def _grouped_gradient_flag_alerts(ctx: TestContext) -> list[dict[str, Any]]:
    if not ctx.gradient_summaries:
        return []

    grouped: dict[str, list[dict[str, Any]]] = {}
    for entry in gradient_flow_analysis(ctx.gradient_summaries):
        for flag in entry.get("flags", []):
            if flag not in GROUPED_FLAG_META:
                continue
            grouped.setdefault(flag, []).append(entry)

    latest_step = ctx.gradient_summaries[-1].get("step")
    alerts = []
    for flag, layers in grouped.items():
        meta = GROUPED_FLAG_META[flag]
        layer_names = [layer["layer"] for layer in layers]
        message = f"{len(layer_names)} layer{'s' if len(layer_names) != 1 else ''} flagged with {flag.lower().replace('_', ' ')}."
        if layer_names:
            message = f"{message} Latest affected layers: {', '.join(layer_names[:3])}."
        evidence = []
        if latest_step is not None:
            evidence.append(f"Latest gradient summary step: {latest_step}")
        evidence.append(f"Affected layers: {len(layer_names)}")
        alerts.append(
            _alert(
                alert_id=f"GRAD_FLAG_{flag}",
                source="gradient",
                status="warn",
                severity=meta["severity"],
                category="Gradient Flow",
                title=meta["title"],
                message=message,
                recommendation=meta["recommendation"],
                details={"flag": flag, "layers": layer_names, "step": latest_step},
                evidence=evidence,
                step=latest_step,
                cta_path="/overview",
            )
        )
    return alerts


def _alert_from_test_result(result: dict[str, Any], *, source: str) -> dict[str, Any]:
    test_id = result.get("id", "UNKNOWN_ALERT")
    details = result.get("details") or {}
    title = TEST_TITLES.get(test_id) or test_id.replace("_", " ").title()
    message = _message_for_test_result(test_id, details, result.get("recommendation") or "")
    recommendation = result.get("recommendation") or TEST_RECOMMENDATION_FALLBACKS.get(test_id, "")
    category = result.get("category", "Training Metrics")
    step = _extract_step(details)
    evidence = _evidence_for_test_result(test_id, details)

    return _alert(
        alert_id=test_id,
        source=source,
        status=result.get("status", "warn"),
        severity=result.get("severity", "MEDIUM"),
        category=category,
        title=title,
        message=message,
        recommendation=recommendation,
        details=details,
        evidence=evidence,
        step=step,
        cta_path=CTA_PATHS.get(category),
    )


def _message_for_test_result(test_id: str, details: dict[str, Any], recommendation: str) -> str:
    if test_id == "LEARNING_RATE_LOGGED":
        return "Optimizer learning-rate history was not logged for this run."
    if test_id == "WEIGHT_DIFF_SEVERITY_COUNTS":
        counts = details.get("severity_counts") or {}
        critical = counts.get("critical", 0)
        total_layers = details.get("total_layers", 0)
        return f"{critical} of {total_layers} checkpointed layers show CRITICAL-sized changes between the first and last checkpoints."
    if test_id == "LABEL_FLIP_RATE":
        flips = details.get("flips", 0)
        total = details.get("total", 0)
        flip_rate = details.get("flip_rate")
        if flip_rate is not None:
            return f"{flips} of {total} probed predictions changed label ({flip_rate:.0%} flip rate) across training."
        return f"{flips} of {total} probed predictions changed label across training."
    if test_id == "SEED_LOGGED":
        return "Run metadata does not include a random seed, so this training run is not fully reproducible."
    if test_id == "LOSS_FINITE":
        total_bad = details.get("total_bad")
        bad_steps = details.get("nan_inf_steps") or []
        if bad_steps:
            first = bad_steps[0]
            return f"Loss became non-finite at step {first.get('step')}."
        return f"Found {total_bad} non-finite loss value(s) in the metric history."
    if test_id == "LOSS_SPIKE_DETECTION":
        spikes = details.get("spikes") or []
        if spikes:
            first = spikes[0]
            return f"Loss spiked to {first.get('value')} around step {first.get('step')}."
        return "Recent metrics show sudden loss spikes that may indicate instability."
    if test_id == "TRAIN_VAL_GAP":
        ratio = details.get("ratio")
        train_loss = details.get("train_loss")
        val_loss = details.get("val_loss")
        if ratio is not None and train_loss is not None and val_loss is not None:
            return f"Validation loss ({val_loss}) is {ratio}x training loss ({train_loss})."
        return "Validation performance is trailing training performance by a large margin."
    if test_id == "OVERFITTING_HEURISTIC":
        inc = details.get("val_loss_increase_ratio")
        if inc is not None:
            return f"Validation loss rose by {inc:.0%} during the final training segment while training loss kept falling."
        return "Validation loss is rising while training loss continues to improve."
    if test_id == "VAL_LOSS_DIVERGENCE":
        rise_rate = details.get("rise_rate")
        if rise_rate is not None:
            return f"Validation loss rose on {rise_rate:.0%} of the checked final-half steps."
        return "Validation loss is rising consistently late in training."
    if test_id == "GRAD_NAN_INF":
        total = details.get("total")
        issues = details.get("issues") or []
        if issues:
            first = issues[0]
            return f"Detected non-finite gradient values in {first.get('layer')} at step {first.get('step')}."
        return f"Detected {total} non-finite gradient issue(s)."
    if test_id == "GRAD_VANISHING":
        total = details.get("total", 0)
        return f"{total} gradient entry{'ies' if total != 1 else ''} fell below the vanishing threshold."
    if test_id == "GRAD_EXPLODING":
        total = details.get("total", 0)
        return f"{total} gradient entry{'ies' if total != 1 else ''} exceeded the exploding threshold."
    return recommendation or "GradGlass detected an issue that may need attention."


def _evidence_for_test_result(test_id: str, details: dict[str, Any]) -> list[str]:
    if test_id == "WEIGHT_DIFF_SEVERITY_COUNTS":
        counts = details.get("severity_counts") or {}
        return [
            f"Critical layers: {counts.get('critical', 0)}",
            f"High layers: {counts.get('high', 0)}",
            f"Compared layers: {details.get('total_layers', 0)}",
        ]
    if test_id == "LABEL_FLIP_RATE":
        return [
            f"Flips: {details.get('flips', 0)} / {details.get('total', 0)}",
            f"First probe step: {details.get('step_a')}",
            f"Last probe step: {details.get('step_b')}",
        ]
    if test_id == "TRAIN_VAL_GAP":
        return [
            f"Train loss: {details.get('train_loss')}",
            f"Val loss: {details.get('val_loss')}",
            f"Gap ratio: {details.get('ratio')}",
        ]
    if test_id == "OVERFITTING_HEURISTIC":
        return [
            f"Val increase ratio: {details.get('val_loss_increase_ratio')}",
            "Train loss kept falling late in the run",
        ]
    if test_id == "VAL_LOSS_DIVERGENCE":
        return [
            f"Val loss start: {details.get('val_loss_start')}",
            f"Val loss end: {details.get('val_loss_end')}",
            f"Rise rate: {details.get('rise_rate')}",
        ]
    if test_id == "LOSS_SPIKE_DETECTION":
        spikes = details.get("spikes") or []
        if spikes:
            first = spikes[0]
            return [
                f"Spike step: {first.get('step')}",
                f"Spike value: {first.get('value')}",
                f"Local mean: {first.get('local_mean')}",
            ]
    if test_id == "GRAD_NAN_INF":
        issues = details.get("issues") or []
        if issues:
            first = issues[0]
            return [
                f"Layer: {first.get('layer')}",
                f"Metric: {first.get('metric')}",
                f"Step: {first.get('step')}",
            ]
    if test_id == "GRAD_VANISHING":
        entries = details.get("vanishing_entries") or []
        if entries:
            first = entries[0]
            return [
                f"Layer: {first.get('layer')}",
                f"Norm: {first.get('norm')}",
                f"Step: {first.get('step')}",
            ]
    if test_id == "GRAD_EXPLODING":
        entries = details.get("exploding_entries") or []
        if entries:
            first = entries[0]
            return [
                f"Layer: {first.get('layer')}",
                f"Norm: {first.get('norm')}",
                f"Step: {first.get('step')}",
            ]
    return _generic_evidence(details)


def _generic_evidence(details: dict[str, Any]) -> list[str]:
    evidence = []
    for key, value in details.items():
        if value in (None, ""):
            continue
        if isinstance(value, (int, float, str, bool)):
            evidence.append(f"{key.replace('_', ' ').title()}: {value}")
        if len(evidence) == 3:
            break
    return evidence


def _extract_step(details: dict[str, Any]) -> Optional[int]:
    for key in ("step", "step_a", "step_b"):
        value = details.get(key)
        if isinstance(value, int):
            return value
    for list_key in ("nan_inf_steps", "spikes", "vanishing_entries", "exploding_entries", "issues"):
        entries = details.get(list_key)
        if isinstance(entries, list) and entries:
            step = entries[0].get("step")
            if isinstance(step, int):
                return step
    return None


def _heartbeat_age_seconds(overview: dict[str, Any]) -> Optional[float]:
    heartbeat_ts = overview.get("heartbeat_ts")
    updated_at = overview.get("updated_at")
    if heartbeat_ts is None or updated_at is None:
        return None
    try:
        return round(max(float(updated_at) - float(heartbeat_ts), 0.0), 2)
    except (TypeError, ValueError):
        return None


def _build_summary(alerts: list[dict[str, Any]], overview: dict[str, Any]) -> dict[str, Any]:
    counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
    fail_count = 0
    warn_count = 0
    for alert in alerts:
        severity = alert["severity"]
        counts[severity] = counts.get(severity, 0) + 1
        if alert["status"] == "fail":
            fail_count += 1
        elif alert["status"] == "warn":
            warn_count += 1

    return {
        "total": len(alerts),
        "critical": counts["CRITICAL"],
        "high": counts["HIGH"],
        "medium": counts["MEDIUM"],
        "low": counts["LOW"],
        "high_severity": counts["CRITICAL"] + counts["HIGH"],
        "warnings": counts["MEDIUM"] + counts["LOW"],
        "fail_count": fail_count,
        "warn_count": warn_count,
        "health_state": overview.get("health_state", "WARNING"),
        "health_reason": overview.get("status_reason") or overview.get("eta_reason"),
        "updated_at": overview.get("updated_at"),
        "top_alert_id": alerts[0]["id"] if alerts else None,
    }


def _alert(
    *,
    alert_id: str,
    source: str,
    status: str,
    severity: str,
    category: str,
    title: str,
    message: str,
    recommendation: str,
    details: Optional[dict[str, Any]] = None,
    evidence: Optional[list[str]] = None,
    step: Optional[int] = None,
    cta_path: Optional[str] = None,
) -> dict[str, Any]:
    return {
        "id": alert_id,
        "source": source,
        "status": status,
        "severity": severity,
        "category": category,
        "title": title,
        "message": message,
        "recommendation": recommendation,
        "details": details or {},
        "evidence": evidence or [],
        "step": step,
        "cta_path": cta_path,
    }


def _push_alert(alerts: list[dict[str, Any]], seen_ids: set[str], alert: dict[str, Any]) -> None:
    alert_id = alert.get("id")
    if not alert_id or alert_id in seen_ids:
        return
    seen_ids.add(alert_id)
    alerts.append(alert)


def _alert_sort_key(alert: dict[str, Any]) -> tuple[int, int, str]:
    severity_score = -SEVERITY_ORDER.get(alert.get("severity", "LOW"), 0)
    status_score = STATUS_ORDER.get(alert.get("status", "warn"), 9)
    return (severity_score, status_score, alert.get("title", ""))
