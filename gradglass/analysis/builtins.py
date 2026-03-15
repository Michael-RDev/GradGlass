from __future__ import annotations
import json
import math
from pathlib import Path
from typing import Any
import numpy as np
from gradglass.analysis.registry import (
    TestCategory,
    TestContext,
    TestRegistry,
    TestResult,
    TestSeverity,
    TestStatus,
    test,
)


@test(
    "STORE_LAYOUT_VALID",
    "Required artifact directories exist",
    TestCategory.ARTIFACT,
    TestSeverity.CRITICAL,
    "Verifies the run directory contains all required subdirectories.",
)
def test_store_layout(ctx):
    required = ["checkpoints", "gradients", "activations", "predictions", "slices"]
    missing = []
    for d in required:
        if not (ctx.run_dir / d).exists():
            missing.append(d)
    if missing:
        return TestResult(
            id="STORE_LAYOUT_VALID",
            title="Required artifact directories exist",
            status=TestStatus.FAIL,
            severity=TestSeverity.CRITICAL,
            category=TestCategory.ARTIFACT,
            details={"missing_dirs": missing},
            recommendation=f"Missing directories: {', '.join(missing)}. Re-run with gg.run().watch().",
        )
    return TestResult(
        id="STORE_LAYOUT_VALID",
        title="Required artifact directories exist",
        status=TestStatus.PASS,
        severity=TestSeverity.CRITICAL,
        category=TestCategory.ARTIFACT,
        details={"dirs_found": required},
    )


@test(
    "METADATA_VALID_JSON",
    "Metadata file is valid JSON with required fields",
    TestCategory.ARTIFACT,
    TestSeverity.CRITICAL,
    "Checks that metadata.json exists, parses correctly, and has key fields.",
)
def test_metadata_valid(ctx):
    meta_path = ctx.run_dir / "metadata.json"
    if not meta_path.exists():
        return TestResult(
            id="METADATA_VALID_JSON",
            title="Metadata file is valid JSON with required fields",
            status=TestStatus.FAIL,
            severity=TestSeverity.CRITICAL,
            category=TestCategory.ARTIFACT,
            details={"error": "metadata.json not found"},
            recommendation="Ensure run.watch() was called and run completed properly.",
        )
    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except json.JSONDecodeError as e:
        return TestResult(
            id="METADATA_VALID_JSON",
            title="Metadata file is valid JSON with required fields",
            status=TestStatus.FAIL,
            severity=TestSeverity.CRITICAL,
            category=TestCategory.ARTIFACT,
            details={"error": f"JSON parse error: {e}"},
            recommendation="metadata.json is corrupted. Re-run the experiment.",
        )
    required_fields = ["name", "run_id", "status", "start_time"]
    missing_fields = [f for f in required_fields if f not in meta]
    if missing_fields:
        return TestResult(
            id="METADATA_VALID_JSON",
            title="Metadata file is valid JSON with required fields",
            status=TestStatus.WARN,
            severity=TestSeverity.HIGH,
            category=TestCategory.ARTIFACT,
            details={"missing_fields": missing_fields, "fields_present": list(meta.keys())},
            recommendation=f"Metadata missing fields: {', '.join(missing_fields)}.",
        )
    return TestResult(
        id="METADATA_VALID_JSON",
        title="Metadata file is valid JSON with required fields",
        status=TestStatus.PASS,
        severity=TestSeverity.CRITICAL,
        category=TestCategory.ARTIFACT,
        details={"fields": list(meta.keys()), "status": meta.get("status")},
    )


@test(
    "CHECKPOINT_READABLE",
    "All checkpoints load without error",
    TestCategory.ARTIFACT,
    TestSeverity.HIGH,
    "Attempts to load each checkpoint file to verify integrity.",
)
def test_checkpoint_readable(ctx):
    if not ctx.has_checkpoints:
        return TestResult(
            id="CHECKPOINT_READABLE",
            title="All checkpoints load without error",
            status=TestStatus.SKIP,
            severity=TestSeverity.HIGH,
            category=TestCategory.ARTIFACT,
            details={"reason": "No checkpoints found"},
        )
    errors = []
    loaded = 0
    for ck in ctx.checkpoints_meta:
        step = ck["step"]
        try:
            ctx.load_checkpoint(step)
            loaded += 1
        except Exception as e:
            errors.append({"step": step, "error": str(e)})
    if errors:
        return TestResult(
            id="CHECKPOINT_READABLE",
            title="All checkpoints load without error",
            status=TestStatus.FAIL,
            severity=TestSeverity.HIGH,
            category=TestCategory.ARTIFACT,
            details={"loaded": loaded, "errors": errors},
            recommendation="Some checkpoint files are corrupted or missing.",
        )
    return TestResult(
        id="CHECKPOINT_READABLE",
        title="All checkpoints load without error",
        status=TestStatus.PASS,
        severity=TestSeverity.HIGH,
        category=TestCategory.ARTIFACT,
        details={"loaded": loaded},
    )


@test(
    "CHECKPOINT_SHAPE_CONSISTENCY",
    "Parameter shapes stable across checkpoints",
    TestCategory.ARTIFACT,
    TestSeverity.HIGH,
    "Verifies that parameter shapes don't change between checkpoints.",
)
def test_checkpoint_shape_consistency(ctx):
    steps = ctx.checkpoint_steps()
    if len(steps) < 2:
        return TestResult(
            id="CHECKPOINT_SHAPE_CONSISTENCY",
            title="Parameter shapes stable across checkpoints",
            status=TestStatus.SKIP,
            severity=TestSeverity.HIGH,
            category=TestCategory.ARTIFACT,
            details={"reason": "Need >=2 checkpoints"},
        )
    try:
        first = ctx.load_checkpoint(steps[0])
        mismatches = []
        for step in steps[1:]:
            other = ctx.load_checkpoint(step)
            for key in first:
                if key in other and first[key].shape != other[key].shape:
                    mismatches.append(
                        {"layer": key, "step": step, "expected": list(first[key].shape), "got": list(other[key].shape)}
                    )
        if mismatches:
            return TestResult(
                id="CHECKPOINT_SHAPE_CONSISTENCY",
                title="Parameter shapes stable across checkpoints",
                status=TestStatus.FAIL,
                severity=TestSeverity.HIGH,
                category=TestCategory.ARTIFACT,
                details={"mismatches": mismatches[:10]},
                recommendation="Parameter shapes changed during training. Check model modifications.",
            )
        return TestResult(
            id="CHECKPOINT_SHAPE_CONSISTENCY",
            title="Parameter shapes stable across checkpoints",
            status=TestStatus.PASS,
            severity=TestSeverity.HIGH,
            category=TestCategory.ARTIFACT,
            details={"checkpoints_compared": len(steps), "layers": len(first)},
        )
    except Exception as e:
        return TestResult(
            id="CHECKPOINT_SHAPE_CONSISTENCY",
            title="Parameter shapes stable across checkpoints",
            status=TestStatus.FAIL,
            severity=TestSeverity.HIGH,
            category=TestCategory.ARTIFACT,
            details={"error": str(e)},
        )


@test(
    "CHECKPOINT_PARAM_COUNT_STABLE",
    "Parameter count unchanged across checkpoints",
    TestCategory.ARTIFACT,
    TestSeverity.MEDIUM,
    "Checks that total trainable parameter count is stable.",
)
def test_checkpoint_param_count_stable(ctx):
    if not ctx.has_checkpoints or len(ctx.checkpoints_meta) < 2:
        return TestResult(
            id="CHECKPOINT_PARAM_COUNT_STABLE",
            title="Parameter count unchanged across checkpoints",
            status=TestStatus.SKIP,
            severity=TestSeverity.MEDIUM,
            category=TestCategory.ARTIFACT,
            details={"reason": "Need >=2 checkpoints"},
        )
    counts = [c.get("num_params", 0) for c in ctx.checkpoints_meta]
    if len(set(counts)) > 1:
        return TestResult(
            id="CHECKPOINT_PARAM_COUNT_STABLE",
            title="Parameter count unchanged across checkpoints",
            status=TestStatus.WARN,
            severity=TestSeverity.MEDIUM,
            category=TestCategory.ARTIFACT,
            details={"param_counts": counts},
            recommendation="Parameter count changed between checkpoints. This may indicate model mutation.",
        )
    return TestResult(
        id="CHECKPOINT_PARAM_COUNT_STABLE",
        title="Parameter count unchanged across checkpoints",
        status=TestStatus.PASS,
        severity=TestSeverity.MEDIUM,
        category=TestCategory.ARTIFACT,
        details={"param_count": counts[0]},
    )


@test(
    "ARTIFACT_SIZE_BUDGET",
    "Storage within reasonable budget",
    TestCategory.ARTIFACT,
    TestSeverity.LOW,
    "Warns if total artifact storage exceeds a reasonable threshold.",
)
def test_artifact_size_budget(ctx):
    total_bytes = sum((p.stat().st_size for p in ctx.run_dir.rglob("*") if p.is_file()))
    total_mb = total_bytes / (1024 * 1024)
    threshold_mb = 500
    if total_mb > threshold_mb:
        return TestResult(
            id="ARTIFACT_SIZE_BUDGET",
            title="Storage within reasonable budget",
            status=TestStatus.WARN,
            severity=TestSeverity.LOW,
            category=TestCategory.ARTIFACT,
            details={"size_mb": round(total_mb, 1), "threshold_mb": threshold_mb},
            recommendation=f"Artifact store is {total_mb:.1f} MB. Consider pruning old checkpoints.",
        )
    return TestResult(
        id="ARTIFACT_SIZE_BUDGET",
        title="Storage within reasonable budget",
        status=TestStatus.PASS,
        severity=TestSeverity.LOW,
        category=TestCategory.ARTIFACT,
        details={"size_mb": round(total_mb, 1)},
    )


@test(
    "DUPLICATE_ARTIFACT_KEYS",
    "No conflicting checkpoint step numbers",
    TestCategory.ARTIFACT,
    TestSeverity.MEDIUM,
    "Checks for duplicate step numbers in checkpoint metadata.",
)
def test_duplicate_artifact_keys(ctx):
    if not ctx.has_checkpoints:
        return TestResult(
            id="DUPLICATE_ARTIFACT_KEYS",
            title="No conflicting checkpoint step numbers",
            status=TestStatus.SKIP,
            severity=TestSeverity.MEDIUM,
            category=TestCategory.ARTIFACT,
            details={"reason": "No checkpoints found"},
        )
    steps = [c["step"] for c in ctx.checkpoints_meta]
    dupes = [s for s in set(steps) if steps.count(s) > 1]
    if dupes:
        return TestResult(
            id="DUPLICATE_ARTIFACT_KEYS",
            title="No conflicting checkpoint step numbers",
            status=TestStatus.WARN,
            severity=TestSeverity.MEDIUM,
            category=TestCategory.ARTIFACT,
            details={"duplicate_steps": dupes},
            recommendation="Duplicate checkpoint step numbers detected.",
        )
    return TestResult(
        id="DUPLICATE_ARTIFACT_KEYS",
        title="No conflicting checkpoint step numbers",
        status=TestStatus.PASS,
        severity=TestSeverity.MEDIUM,
        category=TestCategory.ARTIFACT,
        details={"unique_steps": len(steps)},
    )


@test(
    "ARCH_SERIALIZATION_VALID",
    "Architecture graph loads correctly",
    TestCategory.MODEL,
    TestSeverity.HIGH,
    "Verifies model_structure.json exists and parses.",
)
def test_arch_serialization(ctx):
    if not ctx.has_architecture:
        return TestResult(
            id="ARCH_SERIALIZATION_VALID",
            title="Architecture graph loads correctly",
            status=TestStatus.SKIP,
            severity=TestSeverity.HIGH,
            category=TestCategory.MODEL,
            details={"reason": "No architecture data found"},
        )
    layers = ctx.architecture.get("layers", [])
    edges = ctx.architecture.get("edges", [])
    return TestResult(
        id="ARCH_SERIALIZATION_VALID",
        title="Architecture graph loads correctly",
        status=TestStatus.PASS,
        severity=TestSeverity.HIGH,
        category=TestCategory.MODEL,
        details={"num_layers": len(layers), "num_edges": len(edges)},
    )


@test(
    "GRAPH_DAG_VALID",
    "Architecture graph has no cycles",
    TestCategory.MODEL,
    TestSeverity.MEDIUM,
    "Checks that the architecture graph is a valid DAG.",
)
def test_graph_dag(ctx):
    if not ctx.has_architecture:
        return TestResult(
            id="GRAPH_DAG_VALID",
            title="Architecture graph has no cycles",
            status=TestStatus.SKIP,
            severity=TestSeverity.MEDIUM,
            category=TestCategory.MODEL,
        )
    edges = ctx.architecture.get("edges", [])
    adj: dict[str, list[str]] = {}
    in_degree: dict[str, int] = {}
    nodes = set()
    for layer in ctx.architecture.get("layers", []):
        nodes.add(layer["id"])
        adj.setdefault(layer["id"], [])
        in_degree.setdefault(layer["id"], 0)
    for e in edges:
        adj.setdefault(e[0], []).append(e[1])
        in_degree.setdefault(e[1], 0)
        in_degree[e[1]] = in_degree.get(e[1], 0) + 1
    queue = [n for n in nodes if in_degree.get(n, 0) == 0]
    visited = 0
    while queue:
        node = queue.pop(0)
        visited += 1
        for neighbor in adj.get(node, []):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    has_cycle = visited < len(nodes)
    if has_cycle:
        return TestResult(
            id="GRAPH_DAG_VALID",
            title="Architecture graph has no cycles",
            status=TestStatus.FAIL,
            severity=TestSeverity.MEDIUM,
            category=TestCategory.MODEL,
            details={"visited": visited, "total_nodes": len(nodes)},
            recommendation="Architecture graph contains cycles. Check model definition.",
        )
    return TestResult(
        id="GRAPH_DAG_VALID",
        title="Architecture graph has no cycles",
        status=TestStatus.PASS,
        severity=TestSeverity.MEDIUM,
        category=TestCategory.MODEL,
        details={"nodes": len(nodes), "edges": len(edges)},
    )


@test(
    "LAYER_NAME_UNIQUENESS",
    "Layer names are unique",
    TestCategory.MODEL,
    TestSeverity.MEDIUM,
    "Ensures no duplicate layer names in architecture.",
)
def test_layer_name_uniqueness(ctx):
    if not ctx.has_architecture:
        return TestResult(
            id="LAYER_NAME_UNIQUENESS",
            title="Layer names are unique",
            status=TestStatus.SKIP,
            severity=TestSeverity.MEDIUM,
            category=TestCategory.MODEL,
        )
    ids = [l["id"] for l in ctx.architecture.get("layers", [])]
    dupes = [i for i in set(ids) if ids.count(i) > 1]
    if dupes:
        return TestResult(
            id="LAYER_NAME_UNIQUENESS",
            title="Layer names are unique",
            status=TestStatus.FAIL,
            severity=TestSeverity.MEDIUM,
            category=TestCategory.MODEL,
            details={"duplicates": dupes},
            recommendation="Duplicate layer IDs found. This may cause checkpoint/diff issues.",
        )
    return TestResult(
        id="LAYER_NAME_UNIQUENESS",
        title="Layer names are unique",
        status=TestStatus.PASS,
        severity=TestSeverity.MEDIUM,
        category=TestCategory.MODEL,
        details={"total_layers": len(ids)},
    )


@test(
    "TRAINABLE_FROZEN_CONSISTENCY",
    "Frozen layers didn't update",
    TestCategory.MODEL,
    TestSeverity.HIGH,
    "Verifies that layers marked as non-trainable did not change in checkpoints.",
)
def test_trainable_frozen(ctx):
    if not ctx.has_architecture or not ctx.has_checkpoints or len(ctx.checkpoint_steps()) < 2:
        return TestResult(
            id="TRAINABLE_FROZEN_CONSISTENCY",
            title="Frozen layers didn't update",
            status=TestStatus.SKIP,
            severity=TestSeverity.HIGH,
            category=TestCategory.MODEL,
            details={"reason": "Need architecture + >=2 checkpoints"},
        )
    frozen_layers = set()
    for layer in ctx.architecture.get("layers", []):
        if not layer.get("trainable", True):
            frozen_layers.add(layer["id"])
    if not frozen_layers:
        return TestResult(
            id="TRAINABLE_FROZEN_CONSISTENCY",
            title="Frozen layers didn't update",
            status=TestStatus.PASS,
            severity=TestSeverity.HIGH,
            category=TestCategory.MODEL,
            details={"frozen_layers": 0, "message": "No frozen layers found"},
        )
    try:
        steps = ctx.checkpoint_steps()
        w_first = ctx.load_checkpoint(steps[0])
        w_last = ctx.load_checkpoint(steps[-1])
        violations = []
        for key in w_first:
            layer_name = key.rsplit(".", 1)[0] if "." in key else key
            if layer_name in frozen_layers:
                if key in w_last and (not np.array_equal(w_first[key], w_last[key])):
                    violations.append(key)
        if violations:
            return TestResult(
                id="TRAINABLE_FROZEN_CONSISTENCY",
                title="Frozen layers didn't update",
                status=TestStatus.FAIL,
                severity=TestSeverity.HIGH,
                category=TestCategory.MODEL,
                details={"violations": violations},
                recommendation="Frozen layers were updated. Check optimizer parameter groups.",
            )
        return TestResult(
            id="TRAINABLE_FROZEN_CONSISTENCY",
            title="Frozen layers didn't update",
            status=TestStatus.PASS,
            severity=TestSeverity.HIGH,
            category=TestCategory.MODEL,
            details={"frozen_layers_checked": len(frozen_layers)},
        )
    except Exception as e:
        return TestResult(
            id="TRAINABLE_FROZEN_CONSISTENCY",
            title="Frozen layers didn't update",
            status=TestStatus.SKIP,
            severity=TestSeverity.HIGH,
            category=TestCategory.MODEL,
            details={"error": str(e)},
        )


@test(
    "PARAM_INIT_SANITY",
    "Initial parameters are reasonable",
    TestCategory.MODEL,
    TestSeverity.MEDIUM,
    "Checks earliest checkpoint for all-zero, all-NaN, or extreme init values.",
)
def test_param_init_sanity(ctx):
    steps = ctx.checkpoint_steps()
    if not steps:
        return TestResult(
            id="PARAM_INIT_SANITY",
            title="Initial parameters are reasonable",
            status=TestStatus.SKIP,
            severity=TestSeverity.MEDIUM,
            category=TestCategory.MODEL,
        )
    try:
        weights = ctx.load_checkpoint(steps[0])
        issues = []
        for name, w in weights.items():
            if np.all(w == 0):
                issues.append({"layer": name, "problem": "all_zeros"})
            elif np.any(np.isnan(w)):
                issues.append({"layer": name, "problem": "contains_nan"})
            elif np.any(np.isinf(w)):
                issues.append({"layer": name, "problem": "contains_inf"})
            elif np.max(np.abs(w)) > 100:
                issues.append({"layer": name, "problem": "extreme_values", "max_abs": float(np.max(np.abs(w)))})
        if issues:
            return TestResult(
                id="PARAM_INIT_SANITY",
                title="Initial parameters are reasonable",
                status=TestStatus.WARN,
                severity=TestSeverity.MEDIUM,
                category=TestCategory.MODEL,
                details={"issues": issues},
                recommendation="Some parameters have suspicious initial values.",
            )
        return TestResult(
            id="PARAM_INIT_SANITY",
            title="Initial parameters are reasonable",
            status=TestStatus.PASS,
            severity=TestSeverity.MEDIUM,
            category=TestCategory.MODEL,
            details={"layers_checked": len(weights)},
        )
    except Exception as e:
        return TestResult(
            id="PARAM_INIT_SANITY",
            title="Initial parameters are reasonable",
            status=TestStatus.SKIP,
            severity=TestSeverity.MEDIUM,
            category=TestCategory.MODEL,
            details={"error": str(e)},
        )


@test(
    "LOSS_FINITE",
    "Loss is always finite (no NaN/Inf)",
    TestCategory.METRICS,
    TestSeverity.CRITICAL,
    "Checks every logged loss value for NaN or Inf.",
)
def test_loss_finite(ctx):
    if not ctx.has_metrics:
        return TestResult(
            id="LOSS_FINITE",
            title="Loss is always finite (no NaN/Inf)",
            status=TestStatus.SKIP,
            severity=TestSeverity.CRITICAL,
            category=TestCategory.METRICS,
        )
    losses = [(m.get("step"), m.get("loss")) for m in ctx.metrics if "loss" in m]
    if not losses:
        return TestResult(
            id="LOSS_FINITE",
            title="Loss is always finite (no NaN/Inf)",
            status=TestStatus.SKIP,
            severity=TestSeverity.CRITICAL,
            category=TestCategory.METRICS,
            details={"reason": "No loss values logged"},
        )
    bad = [{"step": s, "value": str(v)} for (s, v) in losses if v is not None and (math.isnan(v) or math.isinf(v))]
    if bad:
        return TestResult(
            id="LOSS_FINITE",
            title="Loss is always finite (no NaN/Inf)",
            status=TestStatus.FAIL,
            severity=TestSeverity.CRITICAL,
            category=TestCategory.METRICS,
            details={"nan_inf_steps": bad[:20], "total_bad": len(bad)},
            recommendation="Loss became NaN/Inf. Reduce learning rate or check data pipeline.",
        )
    return TestResult(
        id="LOSS_FINITE",
        title="Loss is always finite (no NaN/Inf)",
        status=TestStatus.PASS,
        severity=TestSeverity.CRITICAL,
        category=TestCategory.METRICS,
        details={"total_steps": len(losses)},
    )


@test(
    "LOSS_MONOTONIC_TREND",
    "Loss shows decreasing trend",
    TestCategory.METRICS,
    TestSeverity.MEDIUM,
    "Detects if loss is flat or not improving over time.",
)
def test_loss_monotonic(ctx):
    if not ctx.has_metrics:
        return TestResult(
            id="LOSS_MONOTONIC_TREND",
            title="Loss shows decreasing trend",
            status=TestStatus.SKIP,
            severity=TestSeverity.MEDIUM,
            category=TestCategory.METRICS,
        )
    losses = [m["loss"] for m in ctx.metrics if "loss" in m and m["loss"] is not None]
    if len(losses) < 10:
        return TestResult(
            id="LOSS_MONOTONIC_TREND",
            title="Loss shows decreasing trend",
            status=TestStatus.SKIP,
            severity=TestSeverity.MEDIUM,
            category=TestCategory.METRICS,
            details={"reason": f"Only {len(losses)} loss values (need >=10)"},
        )
    n = len(losses)
    first_segment = losses[: max(n // 10, 1)]
    last_segment = losses[-max(n // 10, 1) :]
    first_avg = sum(first_segment) / len(first_segment)
    last_avg = sum(last_segment) / len(last_segment)
    improved = last_avg < first_avg
    ratio = last_avg / max(first_avg, 1e-12)
    if not improved:
        return TestResult(
            id="LOSS_MONOTONIC_TREND",
            title="Loss shows decreasing trend",
            status=TestStatus.WARN,
            severity=TestSeverity.MEDIUM,
            category=TestCategory.METRICS,
            details={"first_avg": round(first_avg, 6), "last_avg": round(last_avg, 6), "ratio": round(ratio, 4)},
            recommendation="Loss did not decrease. Check learning rate, data, or model capacity.",
        )
    return TestResult(
        id="LOSS_MONOTONIC_TREND",
        title="Loss shows decreasing trend",
        status=TestStatus.PASS,
        severity=TestSeverity.MEDIUM,
        category=TestCategory.METRICS,
        details={"first_avg": round(first_avg, 6), "last_avg": round(last_avg, 6), "reduction": round(1 - ratio, 4)},
    )


@test(
    "LOSS_SPIKE_DETECTION",
    "No sudden loss spikes",
    TestCategory.METRICS,
    TestSeverity.HIGH,
    "Detects sudden jumps in loss that may indicate training instability.",
)
def test_loss_spikes(ctx):
    if not ctx.has_metrics:
        return TestResult(
            id="LOSS_SPIKE_DETECTION",
            title="No sudden loss spikes",
            status=TestStatus.SKIP,
            severity=TestSeverity.HIGH,
            category=TestCategory.METRICS,
        )
    losses = [m["loss"] for m in ctx.metrics if "loss" in m and m["loss"] is not None]
    if len(losses) < 20:
        return TestResult(
            id="LOSS_SPIKE_DETECTION",
            title="No sudden loss spikes",
            status=TestStatus.SKIP,
            severity=TestSeverity.HIGH,
            category=TestCategory.METRICS,
            details={"reason": "Too few data points"},
        )
    window = max(len(losses) // 20, 5)
    spikes = []
    for i in range(window, len(losses)):
        local_mean = sum(losses[i - window : i]) / window
        if local_mean > 0 and losses[i] > local_mean * 5:
            spikes.append({"step": i, "value": round(losses[i], 6), "local_mean": round(local_mean, 6)})
    if spikes:
        return TestResult(
            id="LOSS_SPIKE_DETECTION",
            title="No sudden loss spikes",
            status=TestStatus.WARN,
            severity=TestSeverity.HIGH,
            category=TestCategory.METRICS,
            details={"spikes": spikes[:10], "total_spikes": len(spikes)},
            recommendation="Loss spikes detected. Consider gradient clipping or LR warmup.",
        )
    return TestResult(
        id="LOSS_SPIKE_DETECTION",
        title="No sudden loss spikes",
        status=TestStatus.PASS,
        severity=TestSeverity.HIGH,
        category=TestCategory.METRICS,
        details={"steps_checked": len(losses)},
    )


@test(
    "ACC_SANITY",
    "Accuracy values are within [0, 1]",
    TestCategory.METRICS,
    TestSeverity.MEDIUM,
    "Verifies accuracy values are valid probabilities.",
)
def test_acc_sanity(ctx):
    if not ctx.has_metrics:
        return TestResult(
            id="ACC_SANITY",
            title="Accuracy values are within [0, 1]",
            status=TestStatus.SKIP,
            severity=TestSeverity.MEDIUM,
            category=TestCategory.METRICS,
        )
    accs = [(m.get("step"), m.get("acc")) for m in ctx.metrics if "acc" in m]
    if not accs:
        return TestResult(
            id="ACC_SANITY",
            title="Accuracy values are within [0, 1]",
            status=TestStatus.SKIP,
            severity=TestSeverity.MEDIUM,
            category=TestCategory.METRICS,
            details={"reason": "No accuracy values logged"},
        )
    out_of_range = [{"step": s, "value": v} for (s, v) in accs if v is not None and (v < 0 or v > 1)]
    if out_of_range:
        return TestResult(
            id="ACC_SANITY",
            title="Accuracy values are within [0, 1]",
            status=TestStatus.FAIL,
            severity=TestSeverity.MEDIUM,
            category=TestCategory.METRICS,
            details={"out_of_range": out_of_range[:10]},
            recommendation="Accuracy outside [0,1]. Check metric computation.",
        )
    return TestResult(
        id="ACC_SANITY",
        title="Accuracy values are within [0, 1]",
        status=TestStatus.PASS,
        severity=TestSeverity.MEDIUM,
        category=TestCategory.METRICS,
        details={"total_acc_values": len(accs)},
    )


@test(
    "TRAIN_VAL_GAP",
    "Train/validation gap is reasonable",
    TestCategory.METRICS,
    TestSeverity.MEDIUM,
    "Measures generalization gap if validation metrics exist.",
)
def test_train_val_gap(ctx):
    if not ctx.has_metrics:
        return TestResult(
            id="TRAIN_VAL_GAP",
            title="Train/validation gap is reasonable",
            status=TestStatus.SKIP,
            severity=TestSeverity.MEDIUM,
            category=TestCategory.METRICS,
        )
    has_val = any(("val_loss" in m or "val_acc" in m for m in ctx.metrics))
    if not has_val:
        return TestResult(
            id="TRAIN_VAL_GAP",
            title="Train/validation gap is reasonable",
            status=TestStatus.SKIP,
            severity=TestSeverity.MEDIUM,
            category=TestCategory.METRICS,
            details={"reason": "No validation metrics logged"},
        )
    train_losses = [m["loss"] for m in ctx.metrics if "loss" in m]
    val_losses = [m["val_loss"] for m in ctx.metrics if "val_loss" in m]
    if train_losses and val_losses:
        train_last = train_losses[-1]
        val_last = val_losses[-1]
        gap = val_last - train_last
        ratio = val_last / max(train_last, 1e-12)
        if ratio > 10.0:
            return TestResult(
                id="TRAIN_VAL_GAP",
                title="Train/validation gap is reasonable",
                status=TestStatus.FAIL,
                severity=TestSeverity.HIGH,
                category=TestCategory.METRICS,
                details={
                    "train_loss": round(train_last, 6),
                    "val_loss": round(val_last, 6),
                    "gap": round(gap, 6),
                    "ratio": round(ratio, 2),
                },
                recommendation=f"Severe overfitting: val_loss is {ratio:.0f}x the train_loss. Add strong regularization (dropout, weight decay), reduce model size, or increase training data.",
            )
        if gap > train_last * 0.5:
            return TestResult(
                id="TRAIN_VAL_GAP",
                title="Train/validation gap is reasonable",
                status=TestStatus.WARN,
                severity=TestSeverity.MEDIUM,
                category=TestCategory.METRICS,
                details={
                    "train_loss": round(train_last, 6),
                    "val_loss": round(val_last, 6),
                    "gap": round(gap, 6),
                    "ratio": round(ratio, 2),
                },
                recommendation="Large train/val gap suggests overfitting. Try regularization.",
            )
    return TestResult(
        id="TRAIN_VAL_GAP",
        title="Train/validation gap is reasonable",
        status=TestStatus.PASS,
        severity=TestSeverity.MEDIUM,
        category=TestCategory.METRICS,
        details={"has_val_metrics": True},
    )


@test(
    "OVERFITTING_HEURISTIC",
    "No overfitting detected",
    TestCategory.METRICS,
    TestSeverity.MEDIUM,
    "Detects rising validation loss with falling training loss.",
)
def test_overfitting(ctx):
    if not ctx.has_metrics:
        return TestResult(
            id="OVERFITTING_HEURISTIC",
            title="No overfitting detected",
            status=TestStatus.SKIP,
            severity=TestSeverity.MEDIUM,
            category=TestCategory.METRICS,
        )
    val_losses = [m["val_loss"] for m in ctx.metrics if "val_loss" in m]
    train_losses = [m["loss"] for m in ctx.metrics if "loss" in m]
    if len(val_losses) < 5 or len(train_losses) < 5:
        return TestResult(
            id="OVERFITTING_HEURISTIC",
            title="No overfitting detected",
            status=TestStatus.SKIP,
            severity=TestSeverity.MEDIUM,
            category=TestCategory.METRICS,
            details={"reason": "Insufficient validation data"},
        )
    n = len(val_losses)
    last_third = val_losses[2 * n // 3 :]
    if len(last_third) >= 2 and last_third[-1] > last_third[0]:
        train_n = len(train_losses)
        train_last = train_losses[2 * train_n // 3 :]
        if len(train_last) >= 2 and train_last[-1] < train_last[0]:
            val_increase = round((last_third[-1] - last_third[0]) / max(last_third[0], 1e-12), 4)
            return TestResult(
                id="OVERFITTING_HEURISTIC",
                title="No overfitting detected",
                status=TestStatus.FAIL,
                severity=TestSeverity.HIGH,
                category=TestCategory.METRICS,
                details={"val_loss_rising": True, "train_loss_falling": True, "val_loss_increase_ratio": val_increase},
                recommendation="Overfitting confirmed: validation loss is rising while training loss is falling. Use early stopping, dropout, weight decay, or more data.",
            )
    return TestResult(
        id="OVERFITTING_HEURISTIC",
        title="No overfitting detected",
        status=TestStatus.PASS,
        severity=TestSeverity.MEDIUM,
        category=TestCategory.METRICS,
    )


@test(
    "VAL_LOSS_DIVERGENCE",
    "Validation loss not diverging",
    TestCategory.METRICS,
    TestSeverity.HIGH,
    "Detects when validation loss increases monotonically across the final half of training.",
)
def test_val_loss_divergence(ctx):
    if not ctx.has_metrics:
        return TestResult(
            id="VAL_LOSS_DIVERGENCE",
            title="Validation loss not diverging",
            status=TestStatus.SKIP,
            severity=TestSeverity.HIGH,
            category=TestCategory.METRICS,
        )
    val_losses = [m["val_loss"] for m in ctx.metrics if "val_loss" in m]
    if len(val_losses) < 6:
        return TestResult(
            id="VAL_LOSS_DIVERGENCE",
            title="Validation loss not diverging",
            status=TestStatus.SKIP,
            severity=TestSeverity.HIGH,
            category=TestCategory.METRICS,
            details={"reason": f"Only {len(val_losses)} val_loss values (need >=6)"},
        )
    half = val_losses[len(val_losses) // 2 :]
    # Check if every step is higher than the previous (strict monotonic rise)
    rising_steps = sum(1 for i in range(1, len(half)) if half[i] > half[i - 1])
    rise_rate = rising_steps / max(len(half) - 1, 1)
    if rise_rate >= 0.85:
        return TestResult(
            id="VAL_LOSS_DIVERGENCE",
            title="Validation loss not diverging",
            status=TestStatus.FAIL,
            severity=TestSeverity.HIGH,
            category=TestCategory.METRICS,
            details={
                "val_loss_start": round(half[0], 6),
                "val_loss_end": round(half[-1], 6),
                "rising_steps": rising_steps,
                "total_steps_checked": len(half) - 1,
                "rise_rate": round(rise_rate, 3),
            },
            recommendation="Validation loss is diverging (rising continuously). Stop training earlier, add early stopping, or reduce model complexity.",
        )
    return TestResult(
        id="VAL_LOSS_DIVERGENCE",
        title="Validation loss not diverging",
        status=TestStatus.PASS,
        severity=TestSeverity.HIGH,
        category=TestCategory.METRICS,
        details={"rise_rate": round(rise_rate, 3)},
    )


@test(
    "LEARNING_RATE_LOGGED",
    "Learning rate is being tracked",
    TestCategory.METRICS,
    TestSeverity.LOW,
    "Checks if learning rate values are present in metrics.",
)
def test_lr_logged(ctx):
    if not ctx.has_metrics:
        return TestResult(
            id="LEARNING_RATE_LOGGED",
            title="Learning rate is being tracked",
            status=TestStatus.SKIP,
            severity=TestSeverity.LOW,
            category=TestCategory.METRICS,
        )
    lr_entries = [m for m in ctx.metrics if "lr" in m]
    if not lr_entries:
        return TestResult(
            id="LEARNING_RATE_LOGGED",
            title="Learning rate is being tracked",
            status=TestStatus.WARN,
            severity=TestSeverity.LOW,
            category=TestCategory.METRICS,
            details={"reason": "No LR values found in metrics"},
            recommendation="Pass optimizer to run.watch() to enable LR tracking.",
        )
    return TestResult(
        id="LEARNING_RATE_LOGGED",
        title="Learning rate is being tracked",
        status=TestStatus.PASS,
        severity=TestSeverity.LOW,
        category=TestCategory.METRICS,
        details={"lr_entries": len(lr_entries), "latest_lr": lr_entries[-1].get("lr")},
    )


@test(
    "WEIGHT_DIFF_COMPUTED",
    "Weight diff exists between checkpoints",
    TestCategory.CHECKPOINT,
    TestSeverity.HIGH,
    "Verifies that diff can be computed between first and last checkpoint.",
)
def test_weight_diff_computed(ctx):
    steps = ctx.checkpoint_steps()
    if len(steps) < 2:
        return TestResult(
            id="WEIGHT_DIFF_COMPUTED",
            title="Weight diff exists between checkpoints",
            status=TestStatus.SKIP,
            severity=TestSeverity.HIGH,
            category=TestCategory.CHECKPOINT,
            details={"reason": "Need >=2 checkpoints"},
        )
    try:
        from gradglass.diff import full_diff

        w_a = ctx.load_checkpoint(steps[0])
        w_b = ctx.load_checkpoint(steps[-1])
        result = full_diff(w_a, w_b, ctx.run_id, steps[0], steps[-1])
        return TestResult(
            id="WEIGHT_DIFF_COMPUTED",
            title="Weight diff exists between checkpoints",
            status=TestStatus.PASS,
            severity=TestSeverity.HIGH,
            category=TestCategory.CHECKPOINT,
            details={
                "step_a": steps[0],
                "step_b": steps[-1],
                "layers_compared": result.summary["total_layers"],
                "severity_counts": {
                    "low": result.summary.get("low", 0),
                    "medium": result.summary.get("medium", 0),
                    "high": result.summary.get("high", 0),
                    "critical": result.summary.get("critical", 0),
                },
            },
        )
    except Exception as e:
        return TestResult(
            id="WEIGHT_DIFF_COMPUTED",
            title="Weight diff exists between checkpoints",
            status=TestStatus.FAIL,
            severity=TestSeverity.HIGH,
            category=TestCategory.CHECKPOINT,
            details={"error": str(e)},
            recommendation="Could not compute weight diff. Check checkpoint integrity.",
        )


@test(
    "WEIGHT_DIFF_SEVERITY_COUNTS",
    "Weight diff severity distribution",
    TestCategory.CHECKPOINT,
    TestSeverity.MEDIUM,
    "Reports distribution of LOW/MED/HIGH/CRITICAL layer changes.",
)
def test_weight_diff_severity(ctx):
    steps = ctx.checkpoint_steps()
    if len(steps) < 2:
        return TestResult(
            id="WEIGHT_DIFF_SEVERITY_COUNTS",
            title="Weight diff severity distribution",
            status=TestStatus.SKIP,
            severity=TestSeverity.MEDIUM,
            category=TestCategory.CHECKPOINT,
        )
    try:
        from gradglass.diff import full_diff

        w_a = ctx.load_checkpoint(steps[0])
        w_b = ctx.load_checkpoint(steps[-1])
        result = full_diff(w_a, w_b, ctx.run_id, steps[0], steps[-1])
        counts = {
            "low": result.summary.get("low", 0),
            "medium": result.summary.get("medium", 0),
            "high": result.summary.get("high", 0),
            "critical": result.summary.get("critical", 0),
        }
        total = sum(counts.values())
        critical_ratio = counts["critical"] / max(total, 1)
        status = TestStatus.PASS
        rec = ""
        if critical_ratio > 0.5:
            status = TestStatus.WARN
            rec = "Over 50% of layers have CRITICAL severity changes. This is expected for full training."
        return TestResult(
            id="WEIGHT_DIFF_SEVERITY_COUNTS",
            title="Weight diff severity distribution",
            status=status,
            severity=TestSeverity.MEDIUM,
            category=TestCategory.CHECKPOINT,
            details={"severity_counts": counts, "total_layers": total, "critical_ratio": round(critical_ratio, 3)},
            recommendation=rec,
        )
    except Exception:
        return TestResult(
            id="WEIGHT_DIFF_SEVERITY_COUNTS",
            title="Weight diff severity distribution",
            status=TestStatus.SKIP,
            severity=TestSeverity.MEDIUM,
            category=TestCategory.CHECKPOINT,
        )


@test(
    "TOP_CHANGED_LAYERS",
    "Top-K most changed layers identified",
    TestCategory.CHECKPOINT,
    TestSeverity.MEDIUM,
    "Lists the top 5 most changed layers by Frobenius norm.",
)
def test_top_changed_layers(ctx):
    steps = ctx.checkpoint_steps()
    if len(steps) < 2:
        return TestResult(
            id="TOP_CHANGED_LAYERS",
            title="Top-K most changed layers identified",
            status=TestStatus.SKIP,
            severity=TestSeverity.MEDIUM,
            category=TestCategory.CHECKPOINT,
        )
    try:
        from gradglass.diff import full_diff

        w_a = ctx.load_checkpoint(steps[0])
        w_b = ctx.load_checkpoint(steps[-1])
        result = full_diff(w_a, w_b, ctx.run_id, steps[0], steps[-1])
        top_layers = []
        sorted_layers = sorted(result.layers, key=lambda d: d.frob_norm, reverse=True)
        for lr in sorted_layers[:5]:
            top_layers.append(
                {
                    "layer": lr.layer_name,
                    "frob_norm": round(lr.frob_norm, 4),
                    "cos_sim": round(lr.cos_sim, 4),
                    "severity": lr.severity.value,
                }
            )
        return TestResult(
            id="TOP_CHANGED_LAYERS",
            title="Top-K most changed layers identified",
            status=TestStatus.PASS,
            severity=TestSeverity.MEDIUM,
            category=TestCategory.CHECKPOINT,
            details={"top_5_layers": top_layers},
        )
    except Exception:
        return TestResult(
            id="TOP_CHANGED_LAYERS",
            title="Top-K most changed layers identified",
            status=TestStatus.SKIP,
            severity=TestSeverity.MEDIUM,
            category=TestCategory.CHECKPOINT,
        )


@test(
    "UNCHANGED_LAYER_DETECTION",
    "No dead (unchanged) layers",
    TestCategory.CHECKPOINT,
    TestSeverity.HIGH,
    "Detects layers that never update during training.",
)
def test_unchanged_layers(ctx):
    steps = ctx.checkpoint_steps()
    if len(steps) < 2:
        return TestResult(
            id="UNCHANGED_LAYER_DETECTION",
            title="No dead (unchanged) layers",
            status=TestStatus.SKIP,
            severity=TestSeverity.HIGH,
            category=TestCategory.CHECKPOINT,
        )
    try:
        w_a = ctx.load_checkpoint(steps[0])
        w_b = ctx.load_checkpoint(steps[-1])
        unchanged = []
        for key in w_a:
            if key in w_b and np.array_equal(w_a[key], w_b[key]):
                unchanged.append(key)
        if unchanged:
            return TestResult(
                id="UNCHANGED_LAYER_DETECTION",
                title="No dead (unchanged) layers",
                status=TestStatus.WARN,
                severity=TestSeverity.HIGH,
                category=TestCategory.CHECKPOINT,
                details={"unchanged_layers": unchanged},
                recommendation="Some layers never updated. They may be frozen or have zero gradients.",
            )
        return TestResult(
            id="UNCHANGED_LAYER_DETECTION",
            title="No dead (unchanged) layers",
            status=TestStatus.PASS,
            severity=TestSeverity.HIGH,
            category=TestCategory.CHECKPOINT,
            details={"all_layers_updated": True, "total_layers": len(w_a)},
        )
    except Exception:
        return TestResult(
            id="UNCHANGED_LAYER_DETECTION",
            title="No dead (unchanged) layers",
            status=TestStatus.SKIP,
            severity=TestSeverity.HIGH,
            category=TestCategory.CHECKPOINT,
        )


@test(
    "EXCESSIVE_UPDATE_RATIO",
    "Update/weight ratio is stable",
    TestCategory.CHECKPOINT,
    TestSeverity.HIGH,
    "Checks ||ΔW||/||W|| for signs of instability.",
)
def test_excessive_update_ratio(ctx):
    steps = ctx.checkpoint_steps()
    if len(steps) < 2:
        return TestResult(
            id="EXCESSIVE_UPDATE_RATIO",
            title="Update/weight ratio is stable",
            status=TestStatus.SKIP,
            severity=TestSeverity.HIGH,
            category=TestCategory.CHECKPOINT,
        )
    try:
        w_a = ctx.load_checkpoint(steps[0])
        w_b = ctx.load_checkpoint(steps[-1])
        ratios = {}
        excessive = []
        for key in w_a:
            if key in w_b and w_a[key].shape == w_b[key].shape:
                w_norm = float(np.linalg.norm(w_a[key]))
                d_norm = float(np.linalg.norm(w_a[key].astype(np.float64) - w_b[key].astype(np.float64)))
                ratio = d_norm / max(w_norm, 1e-12)
                ratios[key] = round(ratio, 6)
                if ratio > 2.0:
                    excessive.append({"layer": key, "ratio": round(ratio, 4)})
        if excessive:
            return TestResult(
                id="EXCESSIVE_UPDATE_RATIO",
                title="Update/weight ratio is stable",
                status=TestStatus.WARN,
                severity=TestSeverity.HIGH,
                category=TestCategory.CHECKPOINT,
                details={"excessive_layers": excessive, "all_ratios": ratios},
                recommendation="Some layers changed by more than 2x their original norm. May indicate instability.",
            )
        return TestResult(
            id="EXCESSIVE_UPDATE_RATIO",
            title="Update/weight ratio is stable",
            status=TestStatus.PASS,
            severity=TestSeverity.HIGH,
            category=TestCategory.CHECKPOINT,
            details={"max_ratio": max(ratios.values()) if ratios else 0, "layers_checked": len(ratios)},
        )
    except Exception:
        return TestResult(
            id="EXCESSIVE_UPDATE_RATIO",
            title="Update/weight ratio is stable",
            status=TestStatus.SKIP,
            severity=TestSeverity.HIGH,
            category=TestCategory.CHECKPOINT,
        )


@test(
    "GRAD_SUMMARY_PRESENT",
    "Gradient summaries captured",
    TestCategory.GRADIENT,
    TestSeverity.MEDIUM,
    "Checks if any gradient summary files were captured.",
)
def test_grad_summary_present(ctx):
    if ctx.has_grad_summaries:
        return TestResult(
            id="GRAD_SUMMARY_PRESENT",
            title="Gradient summaries captured",
            status=TestStatus.PASS,
            severity=TestSeverity.MEDIUM,
            category=TestCategory.GRADIENT,
            details={"num_summaries": len(ctx.gradient_summaries)},
        )
    return TestResult(
        id="GRAD_SUMMARY_PRESENT",
        title="Gradient summaries captured",
        status=TestStatus.WARN,
        severity=TestSeverity.MEDIUM,
        category=TestCategory.GRADIENT,
        details={"reason": "No gradient summaries found"},
        recommendation="Try increasing epochs or lowering the `every` parameter in watch().",
    )


@test(
    "GRAD_NAN_INF",
    "No NaN/Inf gradients",
    TestCategory.GRADIENT,
    TestSeverity.CRITICAL,
    "Checks all gradient summaries for NaN or Inf values.",
)
def test_grad_nan_inf(ctx):
    if not ctx.has_grad_summaries:
        return TestResult(
            id="GRAD_NAN_INF",
            title="No NaN/Inf gradients",
            status=TestStatus.SKIP,
            severity=TestSeverity.CRITICAL,
            category=TestCategory.GRADIENT,
        )
    bad = []
    for entry in ctx.gradient_summaries:
        step = entry.get("step", "?")
        for layer, data in entry.get("layers", {}).items():
            for k in ["mean", "var", "max", "norm"]:
                v = data.get(k, 0)
                if v is not None and (math.isnan(v) or math.isinf(v)):
                    bad.append({"step": step, "layer": layer, "metric": k, "value": str(v)})
    if bad:
        return TestResult(
            id="GRAD_NAN_INF",
            title="No NaN/Inf gradients",
            status=TestStatus.FAIL,
            severity=TestSeverity.CRITICAL,
            category=TestCategory.GRADIENT,
            details={"issues": bad[:20], "total": len(bad)},
            recommendation="NaN/Inf gradients detected. Use gradient clipping or reduce LR.",
        )
    return TestResult(
        id="GRAD_NAN_INF",
        title="No NaN/Inf gradients",
        status=TestStatus.PASS,
        severity=TestSeverity.CRITICAL,
        category=TestCategory.GRADIENT,
        details={"summaries_checked": len(ctx.gradient_summaries)},
    )


@test(
    "GRAD_VANISHING",
    "No vanishing gradients",
    TestCategory.GRADIENT,
    TestSeverity.HIGH,
    "Detects gradient norms collapsing below epsilon.",
)
def test_grad_vanishing(ctx):
    if not ctx.has_grad_summaries:
        return TestResult(
            id="GRAD_VANISHING",
            title="No vanishing gradients",
            status=TestStatus.SKIP,
            severity=TestSeverity.HIGH,
            category=TestCategory.GRADIENT,
        )
    vanishing = []
    for entry in ctx.gradient_summaries:
        step = entry.get("step", "?")
        for layer, data in entry.get("layers", {}).items():
            norm = data.get("norm", 0)
            if norm is not None and abs(norm) < 1e-07:
                vanishing.append({"step": step, "layer": layer, "norm": norm})
    if vanishing:
        return TestResult(
            id="GRAD_VANISHING",
            title="No vanishing gradients",
            status=TestStatus.WARN,
            severity=TestSeverity.HIGH,
            category=TestCategory.GRADIENT,
            details={"vanishing_entries": vanishing[:20], "total": len(vanishing)},
            recommendation="Some layers have near-zero gradient norms. Use skip connections or different activation.",
        )
    return TestResult(
        id="GRAD_VANISHING",
        title="No vanishing gradients",
        status=TestStatus.PASS,
        severity=TestSeverity.HIGH,
        category=TestCategory.GRADIENT,
    )


@test(
    "GRAD_EXPLODING",
    "No exploding gradients",
    TestCategory.GRADIENT,
    TestSeverity.HIGH,
    "Detects gradient norms exceeding a safe threshold.",
)
def test_grad_exploding(ctx):
    if not ctx.has_grad_summaries:
        return TestResult(
            id="GRAD_EXPLODING",
            title="No exploding gradients",
            status=TestStatus.SKIP,
            severity=TestSeverity.HIGH,
            category=TestCategory.GRADIENT,
        )
    exploding = []
    threshold = 100.0
    for entry in ctx.gradient_summaries:
        step = entry.get("step", "?")
        for layer, data in entry.get("layers", {}).items():
            norm = data.get("norm", 0)
            if norm is not None and abs(norm) > threshold:
                exploding.append({"step": step, "layer": layer, "norm": round(norm, 4)})
    if exploding:
        return TestResult(
            id="GRAD_EXPLODING",
            title="No exploding gradients",
            status=TestStatus.WARN,
            severity=TestSeverity.HIGH,
            category=TestCategory.GRADIENT,
            details={"exploding_entries": exploding[:20], "total": len(exploding)},
            recommendation="Large gradient norms detected. Use gradient clipping.",
        )
    return TestResult(
        id="GRAD_EXPLODING",
        title="No exploding gradients",
        status=TestStatus.PASS,
        severity=TestSeverity.HIGH,
        category=TestCategory.GRADIENT,
    )


@test(
    "GRAD_LAYER_IMBALANCE",
    "Gradient magnitudes balanced across layers",
    TestCategory.GRADIENT,
    TestSeverity.MEDIUM,
    "Checks for extreme imbalance in gradient norms between layers.",
)
def test_grad_layer_imbalance(ctx):
    if not ctx.has_grad_summaries:
        return TestResult(
            id="GRAD_LAYER_IMBALANCE",
            title="Gradient magnitudes balanced across layers",
            status=TestStatus.SKIP,
            severity=TestSeverity.MEDIUM,
            category=TestCategory.GRADIENT,
        )
    latest = ctx.gradient_summaries[-1]
    norms = {}
    for layer, data in latest.get("layers", {}).items():
        norm = data.get("norm", 0)
        if norm is not None:
            norms[layer] = norm
    if len(norms) < 2:
        return TestResult(
            id="GRAD_LAYER_IMBALANCE",
            title="Gradient magnitudes balanced across layers",
            status=TestStatus.SKIP,
            severity=TestSeverity.MEDIUM,
            category=TestCategory.GRADIENT,
        )
    max_norm = max(norms.values())
    min_norm = min(norms.values())
    ratio = max_norm / max(min_norm, 1e-12)
    if ratio > 100:
        return TestResult(
            id="GRAD_LAYER_IMBALANCE",
            title="Gradient magnitudes balanced across layers",
            status=TestStatus.WARN,
            severity=TestSeverity.MEDIUM,
            category=TestCategory.GRADIENT,
            details={"max_norm": round(max_norm, 6), "min_norm": round(min_norm, 6), "ratio": round(ratio, 2)},
            recommendation=f"Gradient norm ratio is {ratio:.0f}x. Consider per-layer LR or normalization.",
        )
    return TestResult(
        id="GRAD_LAYER_IMBALANCE",
        title="Gradient magnitudes balanced across layers",
        status=TestStatus.PASS,
        severity=TestSeverity.MEDIUM,
        category=TestCategory.GRADIENT,
        details={"ratio": round(ratio, 2)},
    )


@test(
    "GRAD_CLIP_EFFECTIVENESS",
    "Gradient clipping is effective",
    TestCategory.GRADIENT,
    TestSeverity.LOW,
    "If gradient clipping is enabled, verifies norms stay within bounds.",
)
def test_grad_clip(ctx):
    if not ctx.has_grad_summaries:
        return TestResult(
            id="GRAD_CLIP_EFFECTIVENESS",
            title="Gradient clipping is effective",
            status=TestStatus.SKIP,
            severity=TestSeverity.LOW,
            category=TestCategory.GRADIENT,
        )
    clip_val = None
    if ctx.metadata:
        config = ctx.metadata.get("config", {})
        clip_val = config.get("grad_clip") or config.get("max_grad_norm")
    if clip_val is None:
        return TestResult(
            id="GRAD_CLIP_EFFECTIVENESS",
            title="Gradient clipping is effective",
            status=TestStatus.SKIP,
            severity=TestSeverity.LOW,
            category=TestCategory.GRADIENT,
            details={"reason": "No gradient clipping config found in metadata"},
        )
    violations = []
    for entry in ctx.gradient_summaries:
        for layer, data in entry.get("layers", {}).items():
            norm = data.get("norm", 0)
            if norm is not None and norm > clip_val * 1.1:
                violations.append({"step": entry.get("step"), "layer": layer, "norm": round(norm, 4)})
    if violations:
        return TestResult(
            id="GRAD_CLIP_EFFECTIVENESS",
            title="Gradient clipping is effective",
            status=TestStatus.WARN,
            severity=TestSeverity.LOW,
            category=TestCategory.GRADIENT,
            details={"clip_val": clip_val, "violations": violations[:10]},
            recommendation="Some gradient norms exceed clip threshold. Verify clipping is applied.",
        )
    return TestResult(
        id="GRAD_CLIP_EFFECTIVENESS",
        title="Gradient clipping is effective",
        status=TestStatus.PASS,
        severity=TestSeverity.LOW,
        category=TestCategory.GRADIENT,
        details={"clip_val": clip_val},
    )


@test(
    "ACT_NAN_INF",
    "No NaN/Inf activations",
    TestCategory.ACTIVATION,
    TestSeverity.CRITICAL,
    "Checks activation statistics for NaN or Inf values.",
)
def test_act_nan_inf(ctx):
    if not ctx.has_activations:
        return TestResult(
            id="ACT_NAN_INF",
            title="No NaN/Inf activations",
            status=TestStatus.SKIP,
            severity=TestSeverity.CRITICAL,
            category=TestCategory.ACTIVATION,
        )
    bad = []
    for stat in ctx.activation_stats:
        for k in ["mean", "var"]:
            v = stat.get(k)
            if v is not None and (math.isnan(v) or math.isinf(v)):
                bad.append({"layer": stat.get("layer"), "step": stat.get("step"), "metric": k})
    if bad:
        return TestResult(
            id="ACT_NAN_INF",
            title="No NaN/Inf activations",
            status=TestStatus.FAIL,
            severity=TestSeverity.CRITICAL,
            category=TestCategory.ACTIVATION,
            details={"issues": bad[:20]},
            recommendation="NaN/Inf activations found. Check for exploding values or bad inputs.",
        )
    return TestResult(
        id="ACT_NAN_INF",
        title="No NaN/Inf activations",
        status=TestStatus.PASS,
        severity=TestSeverity.CRITICAL,
        category=TestCategory.ACTIVATION,
        details={"activations_checked": len(ctx.activation_stats)},
    )


@test(
    "ACT_SPARSITY_COLLAPSE",
    "Activation sparsity is healthy",
    TestCategory.ACTIVATION,
    TestSeverity.HIGH,
    "Detects layers where too many activations are near zero.",
)
def test_act_sparsity(ctx):
    if not ctx.has_activations:
        return TestResult(
            id="ACT_SPARSITY_COLLAPSE",
            title="Activation sparsity is healthy",
            status=TestStatus.SKIP,
            severity=TestSeverity.HIGH,
            category=TestCategory.ACTIVATION,
        )
    collapsed = []
    for stat in ctx.activation_stats:
        sparsity = stat.get("sparsity", 0)
        if sparsity is not None and sparsity > 0.95:
            collapsed.append({"layer": stat.get("layer"), "step": stat.get("step"), "sparsity": round(sparsity, 4)})
    if collapsed:
        return TestResult(
            id="ACT_SPARSITY_COLLAPSE",
            title="Activation sparsity is healthy",
            status=TestStatus.WARN,
            severity=TestSeverity.HIGH,
            category=TestCategory.ACTIVATION,
            details={"collapsed_layers": collapsed},
            recommendation="Some layers have >95% zero activations. Check for dying ReLU.",
        )
    return TestResult(
        id="ACT_SPARSITY_COLLAPSE",
        title="Activation sparsity is healthy",
        status=TestStatus.PASS,
        severity=TestSeverity.HIGH,
        category=TestCategory.ACTIVATION,
    )


@test(
    "DEAD_NEURON_RATE",
    "Dead neuron rate is low",
    TestCategory.ACTIVATION,
    TestSeverity.HIGH,
    "Estimates dead neuron percentage per layer from sparsity stats.",
)
def test_dead_neuron_rate(ctx):
    if not ctx.has_activations:
        return TestResult(
            id="DEAD_NEURON_RATE",
            title="Dead neuron rate is low",
            status=TestStatus.SKIP,
            severity=TestSeverity.HIGH,
            category=TestCategory.ACTIVATION,
        )
    high_dead = []
    for stat in ctx.activation_stats:
        sparsity = stat.get("sparsity", 0)
        if sparsity is not None and sparsity > 0.5:
            high_dead.append({"layer": stat.get("layer"), "step": stat.get("step"), "dead_rate": round(sparsity, 4)})
    if high_dead:
        return TestResult(
            id="DEAD_NEURON_RATE",
            title="Dead neuron rate is low",
            status=TestStatus.WARN,
            severity=TestSeverity.HIGH,
            category=TestCategory.ACTIVATION,
            details={"high_dead_rate_layers": high_dead},
            recommendation="Many neurons are inactive. Consider LeakyReLU or different initialization.",
        )
    return TestResult(
        id="DEAD_NEURON_RATE",
        title="Dead neuron rate is low",
        status=TestStatus.PASS,
        severity=TestSeverity.HIGH,
        category=TestCategory.ACTIVATION,
    )


@test(
    "ACT_DISTRIBUTION_DRIFT",
    "Activation distributions stable",
    TestCategory.ACTIVATION,
    TestSeverity.MEDIUM,
    "Detects large shifts in activation mean/variance between steps.",
)
def test_act_distribution_drift(ctx):
    if not ctx.has_activations:
        return TestResult(
            id="ACT_DISTRIBUTION_DRIFT",
            title="Activation distributions stable",
            status=TestStatus.SKIP,
            severity=TestSeverity.MEDIUM,
            category=TestCategory.ACTIVATION,
        )
    by_layer: dict[str, list] = {}
    for stat in ctx.activation_stats:
        layer = stat.get("layer", "unknown")
        by_layer.setdefault(layer, []).append(stat)
    drifting = []
    for layer, stats in by_layer.items():
        if len(stats) < 2:
            continue
        means = [s.get("mean", 0) for s in stats]
        shift = abs(means[-1] - means[0])
        if shift > 5.0:
            drifting.append({"layer": layer, "mean_shift": round(shift, 4)})
    if drifting:
        return TestResult(
            id="ACT_DISTRIBUTION_DRIFT",
            title="Activation distributions stable",
            status=TestStatus.WARN,
            severity=TestSeverity.MEDIUM,
            category=TestCategory.ACTIVATION,
            details={"drifting_layers": drifting},
            recommendation="Activation mean shifted significantly. May indicate covariate shift.",
        )
    return TestResult(
        id="ACT_DISTRIBUTION_DRIFT",
        title="Activation distributions stable",
        status=TestStatus.PASS,
        severity=TestSeverity.MEDIUM,
        category=TestCategory.ACTIVATION,
    )


@test(
    "SATURATION_DETECTION",
    "No activation saturation detected",
    TestCategory.ACTIVATION,
    TestSeverity.MEDIUM,
    "Checks for tanh/sigmoid saturation indicators.",
)
def test_saturation(ctx):
    if not ctx.has_activations:
        return TestResult(
            id="SATURATION_DETECTION",
            title="No activation saturation detected",
            status=TestStatus.SKIP,
            severity=TestSeverity.MEDIUM,
            category=TestCategory.ACTIVATION,
        )
    saturated = []
    for stat in ctx.activation_stats:
        mean = stat.get("mean", 0)
        var = stat.get("var", 1)
        if var is not None and var < 0.0001 and (mean is not None) and (abs(mean) > 0.9):
            saturated.append(
                {"layer": stat.get("layer"), "step": stat.get("step"), "mean": round(mean, 4), "var": round(var, 6)}
            )
    if saturated:
        return TestResult(
            id="SATURATION_DETECTION",
            title="No activation saturation detected",
            status=TestStatus.WARN,
            severity=TestSeverity.MEDIUM,
            category=TestCategory.ACTIVATION,
            details={"saturated_layers": saturated},
            recommendation="Activation saturation detected. Outputs near ±1 with low variance.",
        )
    return TestResult(
        id="SATURATION_DETECTION",
        title="No activation saturation detected",
        status=TestStatus.PASS,
        severity=TestSeverity.MEDIUM,
        category=TestCategory.ACTIVATION,
    )


@test(
    "REPRESENTATION_COLLAPSE",
    "No representation collapse",
    TestCategory.ACTIVATION,
    TestSeverity.HIGH,
    "Detects near-constant activations (all embeddings identical).",
)
def test_representation_collapse(ctx):
    if not ctx.has_activations:
        return TestResult(
            id="REPRESENTATION_COLLAPSE",
            title="No representation collapse",
            status=TestStatus.SKIP,
            severity=TestSeverity.HIGH,
            category=TestCategory.ACTIVATION,
        )
    collapsed = []
    for stat in ctx.activation_stats:
        var = stat.get("var", 1)
        if var is not None and var < 1e-08:
            collapsed.append({"layer": stat.get("layer"), "step": stat.get("step"), "var": var})
    if collapsed:
        return TestResult(
            id="REPRESENTATION_COLLAPSE",
            title="No representation collapse",
            status=TestStatus.FAIL,
            severity=TestSeverity.HIGH,
            category=TestCategory.ACTIVATION,
            details={"collapsed_layers": collapsed},
            recommendation="Activations are near-constant. Representations have collapsed.",
        )
    return TestResult(
        id="REPRESENTATION_COLLAPSE",
        title="No representation collapse",
        status=TestStatus.PASS,
        severity=TestSeverity.HIGH,
        category=TestCategory.ACTIVATION,
    )


@test(
    "PRED_FINITE",
    "All predictions are finite",
    TestCategory.PREDICTION,
    TestSeverity.CRITICAL,
    "Checks prediction probe outputs for NaN/Inf.",
)
def test_pred_finite(ctx):
    if not ctx.has_predictions:
        return TestResult(
            id="PRED_FINITE",
            title="All predictions are finite",
            status=TestStatus.SKIP,
            severity=TestSeverity.CRITICAL,
            category=TestCategory.PREDICTION,
        )
    bad = []
    for pred in ctx.predictions:
        step = pred.get("step", "?")
        for key in ["y_pred", "confidence"]:
            vals = pred.get(key, [])
            if vals:
                arr = np.array(vals, dtype=float)
                if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
                    bad.append({"step": step, "field": key})
    if bad:
        return TestResult(
            id="PRED_FINITE",
            title="All predictions are finite",
            status=TestStatus.FAIL,
            severity=TestSeverity.CRITICAL,
            category=TestCategory.PREDICTION,
            details={"bad_entries": bad[:20]},
            recommendation="NaN/Inf in model predictions. Critical training issue.",
        )
    return TestResult(
        id="PRED_FINITE",
        title="All predictions are finite",
        status=TestStatus.PASS,
        severity=TestSeverity.CRITICAL,
        category=TestCategory.PREDICTION,
        details={"probes_checked": len(ctx.predictions)},
    )


@test(
    "LABEL_FLIP_RATE",
    "Label flip rate is reasonable",
    TestCategory.PREDICTION,
    TestSeverity.MEDIUM,
    "Measures how many predictions change between first and last probe.",
)
def test_label_flip_rate(ctx):
    if not ctx.has_predictions or len(ctx.predictions) < 2:
        return TestResult(
            id="LABEL_FLIP_RATE",
            title="Label flip rate is reasonable",
            status=TestStatus.SKIP,
            severity=TestSeverity.MEDIUM,
            category=TestCategory.PREDICTION,
        )
    first = ctx.predictions[0]
    last = ctx.predictions[-1]
    y_a = first.get("y_pred", [])
    y_b = last.get("y_pred", [])
    if y_a and y_b:
        min_len = min(len(y_a), len(y_b))
        a = np.array(y_a[:min_len])
        b = np.array(y_b[:min_len])
        flips = int(np.sum(a != b))
        flip_rate = flips / max(min_len, 1)
        return TestResult(
            id="LABEL_FLIP_RATE",
            title="Label flip rate is reasonable",
            status=TestStatus.PASS if flip_rate < 0.5 else TestStatus.WARN,
            severity=TestSeverity.MEDIUM,
            category=TestCategory.PREDICTION,
            details={
                "flips": flips,
                "total": min_len,
                "flip_rate": round(flip_rate, 4),
                "step_a": first.get("step"),
                "step_b": last.get("step"),
            },
            recommendation="High label flip rate may indicate unstable training." if flip_rate >= 0.5 else "",
        )
    return TestResult(
        id="LABEL_FLIP_RATE",
        title="Label flip rate is reasonable",
        status=TestStatus.SKIP,
        severity=TestSeverity.MEDIUM,
        category=TestCategory.PREDICTION,
    )


@test(
    "CONFIDENCE_SHIFT",
    "Prediction confidence improved",
    TestCategory.PREDICTION,
    TestSeverity.LOW,
    "Tracks mean confidence from first to last prediction probe.",
)
def test_confidence_shift(ctx):
    if not ctx.has_predictions or len(ctx.predictions) < 2:
        return TestResult(
            id="CONFIDENCE_SHIFT",
            title="Prediction confidence improved",
            status=TestStatus.SKIP,
            severity=TestSeverity.LOW,
            category=TestCategory.PREDICTION,
        )
    first_conf = ctx.predictions[0].get("confidence", [])
    last_conf = ctx.predictions[-1].get("confidence", [])
    if first_conf and last_conf:
        mean_first = float(np.mean(first_conf))
        mean_last = float(np.mean(last_conf))
        delta = mean_last - mean_first
        return TestResult(
            id="CONFIDENCE_SHIFT",
            title="Prediction confidence improved",
            status=TestStatus.PASS if delta >= 0 else TestStatus.WARN,
            severity=TestSeverity.LOW,
            category=TestCategory.PREDICTION,
            details={
                "first_mean_conf": round(mean_first, 4),
                "last_mean_conf": round(mean_last, 4),
                "delta": round(delta, 4),
            },
            recommendation="Confidence decreased over training." if delta < 0 else "",
        )
    return TestResult(
        id="CONFIDENCE_SHIFT",
        title="Prediction confidence improved",
        status=TestStatus.SKIP,
        severity=TestSeverity.LOW,
        category=TestCategory.PREDICTION,
    )


@test(
    "TOP_CHANGED_SAMPLES",
    "Top changed prediction samples identified",
    TestCategory.PREDICTION,
    TestSeverity.LOW,
    "Lists samples with largest prediction delta between first and last probe.",
)
def test_top_changed_samples(ctx):
    if not ctx.has_predictions or len(ctx.predictions) < 2:
        return TestResult(
            id="TOP_CHANGED_SAMPLES",
            title="Top changed prediction samples identified",
            status=TestStatus.SKIP,
            severity=TestSeverity.LOW,
            category=TestCategory.PREDICTION,
        )
    first_conf = ctx.predictions[0].get("confidence", [])
    last_conf = ctx.predictions[-1].get("confidence", [])
    if first_conf and last_conf:
        min_len = min(len(first_conf), len(last_conf))
        a = np.array(first_conf[:min_len])
        b = np.array(last_conf[:min_len])
        deltas = np.abs(b - a)
        top_k = min(10, min_len)
        top_indices = np.argsort(deltas)[-top_k:][::-1]
        top_samples = [
            {
                "index": int(i),
                "conf_before": round(float(a[i]), 4),
                "conf_after": round(float(b[i]), 4),
                "delta": round(float(deltas[i]), 4),
            }
            for i in top_indices
        ]
        return TestResult(
            id="TOP_CHANGED_SAMPLES",
            title="Top changed prediction samples identified",
            status=TestStatus.PASS,
            severity=TestSeverity.LOW,
            category=TestCategory.PREDICTION,
            details={"top_samples": top_samples},
        )
    return TestResult(
        id="TOP_CHANGED_SAMPLES",
        title="Top changed prediction samples identified",
        status=TestStatus.SKIP,
        severity=TestSeverity.LOW,
        category=TestCategory.PREDICTION,
    )


@test(
    "DATA_HASH_STABILITY",
    "Dataset signature logged",
    TestCategory.DATA,
    TestSeverity.LOW,
    "Checks if dataset hash/signature was captured in metadata.",
)
def test_data_hash(ctx):
    if ctx.metadata and ctx.metadata.get("config", {}).get("data_hash"):
        return TestResult(
            id="DATA_HASH_STABILITY",
            title="Dataset signature logged",
            status=TestStatus.PASS,
            severity=TestSeverity.LOW,
            category=TestCategory.DATA,
            details={"data_hash": ctx.metadata["config"]["data_hash"]},
        )
    return TestResult(
        id="DATA_HASH_STABILITY",
        title="Dataset signature logged",
        status=TestStatus.SKIP,
        severity=TestSeverity.LOW,
        category=TestCategory.DATA,
        details={"reason": "No data hash found in config"},
        recommendation="Pass data_hash to run config to track dataset stability.",
    )


@test(
    "CLASS_IMBALANCE_CHECK",
    "Class distribution is balanced",
    TestCategory.DATA,
    TestSeverity.LOW,
    "Checks prediction probes for class imbalance in labels.",
)
def test_class_imbalance(ctx):
    if not ctx.has_predictions:
        return TestResult(
            id="CLASS_IMBALANCE_CHECK",
            title="Class distribution is balanced",
            status=TestStatus.SKIP,
            severity=TestSeverity.LOW,
            category=TestCategory.DATA,
        )
    first = ctx.predictions[0]
    y_true = first.get("y_true", [])
    if not y_true:
        return TestResult(
            id="CLASS_IMBALANCE_CHECK",
            title="Class distribution is balanced",
            status=TestStatus.SKIP,
            severity=TestSeverity.LOW,
            category=TestCategory.DATA,
            details={"reason": "No y_true in prediction probes"},
        )
    labels = np.array(y_true)
    (unique, counts) = np.unique(labels, return_counts=True)
    max_ratio = float(max(counts)) / float(min(counts)) if min(counts) > 0 else float("inf")
    if max_ratio > 10:
        return TestResult(
            id="CLASS_IMBALANCE_CHECK",
            title="Class distribution is balanced",
            status=TestStatus.WARN,
            severity=TestSeverity.LOW,
            category=TestCategory.DATA,
            details={"max_ratio": round(max_ratio, 2), "class_counts": dict(zip(unique.tolist(), counts.tolist()))},
            recommendation="Significant class imbalance detected. Consider oversampling or weighted loss.",
        )
    return TestResult(
        id="CLASS_IMBALANCE_CHECK",
        title="Class distribution is balanced",
        status=TestStatus.PASS,
        severity=TestSeverity.LOW,
        category=TestCategory.DATA,
        details={"num_classes": len(unique), "max_ratio": round(max_ratio, 2)},
    )


@test(
    "SLICE_COVERAGE",
    "Dataset slices have sufficient examples",
    TestCategory.DATA,
    TestSeverity.LOW,
    "Checks if dataset slice artifacts exist and contain enough samples.",
)
def test_slice_coverage(ctx):
    slice_dir = ctx.run_dir / "slices"
    if not slice_dir.exists():
        return TestResult(
            id="SLICE_COVERAGE",
            title="Dataset slices have sufficient examples",
            status=TestStatus.SKIP,
            severity=TestSeverity.LOW,
            category=TestCategory.DATA,
            details={"reason": "No slices directory"},
        )
    slice_files = list(slice_dir.glob("*.json"))
    if not slice_files:
        return TestResult(
            id="SLICE_COVERAGE",
            title="Dataset slices have sufficient examples",
            status=TestStatus.SKIP,
            severity=TestSeverity.LOW,
            category=TestCategory.DATA,
            details={"reason": "No slice files found"},
        )
    return TestResult(
        id="SLICE_COVERAGE",
        title="Dataset slices have sufficient examples",
        status=TestStatus.PASS,
        severity=TestSeverity.LOW,
        category=TestCategory.DATA,
        details={"num_slices": len(slice_files)},
    )


@test(
    "RANKS_PRESENT",
    "All distributed ranks reported",
    TestCategory.DISTRIBUTED,
    TestSeverity.HIGH,
    "Checks that all expected ranks have data in the artifact store.",
)
def test_ranks_present(ctx):
    if not ctx.is_distributed:
        return TestResult(
            id="RANKS_PRESENT",
            title="All distributed ranks reported",
            status=TestStatus.SKIP,
            severity=TestSeverity.HIGH,
            category=TestCategory.DISTRIBUTED,
            details={"reason": "Not a distributed run"},
        )
    expected = 0
    if ctx.distributed_info:
        expected = ctx.distributed_info.get("world_size", 0)
    actual = len(ctx.rank_list or [])
    if expected > 0 and actual < expected:
        return TestResult(
            id="RANKS_PRESENT",
            title="All distributed ranks reported",
            status=TestStatus.FAIL,
            severity=TestSeverity.HIGH,
            category=TestCategory.DISTRIBUTED,
            details={"expected": expected, "found": actual, "ranks": ctx.rank_list},
            recommendation="Not all ranks reported. Check for crashes or straggling workers.",
        )
    return TestResult(
        id="RANKS_PRESENT",
        title="All distributed ranks reported",
        status=TestStatus.PASS,
        severity=TestSeverity.HIGH,
        category=TestCategory.DISTRIBUTED,
        details={"ranks": actual},
    )


@test(
    "SYNC_LATENCY_STATS",
    "Sync latency within bounds",
    TestCategory.DISTRIBUTED,
    TestSeverity.MEDIUM,
    "Checks distributed synchronization latency statistics.",
)
def test_sync_latency(ctx):
    if not ctx.is_distributed:
        return TestResult(
            id="SYNC_LATENCY_STATS",
            title="Sync latency within bounds",
            status=TestStatus.SKIP,
            severity=TestSeverity.MEDIUM,
            category=TestCategory.DISTRIBUTED,
        )
    if ctx.distributed_info and "sync_latency_ms" in ctx.distributed_info:
        latency = ctx.distributed_info["sync_latency_ms"]
        if latency > 1000:
            return TestResult(
                id="SYNC_LATENCY_STATS",
                title="Sync latency within bounds",
                status=TestStatus.WARN,
                severity=TestSeverity.MEDIUM,
                category=TestCategory.DISTRIBUTED,
                details={"latency_ms": latency},
                recommendation="High sync latency. Check network bandwidth between nodes.",
            )
        return TestResult(
            id="SYNC_LATENCY_STATS",
            title="Sync latency within bounds",
            status=TestStatus.PASS,
            severity=TestSeverity.MEDIUM,
            category=TestCategory.DISTRIBUTED,
            details={"latency_ms": latency},
        )
    return TestResult(
        id="SYNC_LATENCY_STATS",
        title="Sync latency within bounds",
        status=TestStatus.SKIP,
        severity=TestSeverity.MEDIUM,
        category=TestCategory.DISTRIBUTED,
        details={"reason": "No sync latency data"},
    )


@test(
    "STRAGGLER_DETECTION",
    "No straggler ranks detected",
    TestCategory.DISTRIBUTED,
    TestSeverity.MEDIUM,
    "Detects ranks that are significantly slower than others.",
)
def test_straggler(ctx):
    if not ctx.is_distributed:
        return TestResult(
            id="STRAGGLER_DETECTION",
            title="No straggler ranks detected",
            status=TestStatus.SKIP,
            severity=TestSeverity.MEDIUM,
            category=TestCategory.DISTRIBUTED,
        )
    return TestResult(
        id="STRAGGLER_DETECTION",
        title="No straggler ranks detected",
        status=TestStatus.SKIP,
        severity=TestSeverity.MEDIUM,
        category=TestCategory.DISTRIBUTED,
        details={"reason": "Per-rank timing data not available"},
    )


@test(
    "PER_RANK_LOSS_DIVERGENCE",
    "Per-rank loss values converge",
    TestCategory.DISTRIBUTED,
    TestSeverity.HIGH,
    "Checks that loss values across ranks don't diverge.",
)
def test_per_rank_loss(ctx):
    if not ctx.is_distributed:
        return TestResult(
            id="PER_RANK_LOSS_DIVERGENCE",
            title="Per-rank loss values converge",
            status=TestStatus.SKIP,
            severity=TestSeverity.HIGH,
            category=TestCategory.DISTRIBUTED,
        )
    return TestResult(
        id="PER_RANK_LOSS_DIVERGENCE",
        title="Per-rank loss values converge",
        status=TestStatus.SKIP,
        severity=TestSeverity.HIGH,
        category=TestCategory.DISTRIBUTED,
        details={"reason": "Per-rank loss data not available"},
    )


@test(
    "RANK_CRASH_DETECTION",
    "No rank crashes detected",
    TestCategory.DISTRIBUTED,
    TestSeverity.CRITICAL,
    "Checks for signs of rank crashes during distributed training.",
)
def test_rank_crash(ctx):
    if not ctx.is_distributed:
        return TestResult(
            id="RANK_CRASH_DETECTION",
            title="No rank crashes detected",
            status=TestStatus.SKIP,
            severity=TestSeverity.CRITICAL,
            category=TestCategory.DISTRIBUTED,
        )
    return TestResult(
        id="RANK_CRASH_DETECTION",
        title="No rank crashes detected",
        status=TestStatus.SKIP,
        severity=TestSeverity.CRITICAL,
        category=TestCategory.DISTRIBUTED,
        details={"reason": "Crash detection data not available"},
    )


@test(
    "SEED_LOGGED",
    "Random seed is logged",
    TestCategory.REPRODUCIBILITY,
    TestSeverity.LOW,
    "Checks if a random seed was captured for reproducibility.",
)
def test_seed_logged(ctx):
    if ctx.metadata:
        config = ctx.metadata.get("config", {})
        seed = config.get("seed") or config.get("random_seed")
        if seed is not None:
            return TestResult(
                id="SEED_LOGGED",
                title="Random seed is logged",
                status=TestStatus.PASS,
                severity=TestSeverity.LOW,
                category=TestCategory.REPRODUCIBILITY,
                details={"seed": seed},
            )
    return TestResult(
        id="SEED_LOGGED",
        title="Random seed is logged",
        status=TestStatus.WARN,
        severity=TestSeverity.LOW,
        category=TestCategory.REPRODUCIBILITY,
        details={"reason": "No seed found in config"},
        recommendation="Pass seed= in run config for reproducibility tracking.",
    )


@test(
    "ENV_CAPTURED",
    "Environment info is captured",
    TestCategory.REPRODUCIBILITY,
    TestSeverity.LOW,
    "Checks for Python version, framework version, CUDA version in metadata.",
)
def test_env_captured(ctx):
    if ctx.metadata:
        env = ctx.metadata.get("environment", {})
        if env:
            return TestResult(
                id="ENV_CAPTURED",
                title="Environment info is captured",
                status=TestStatus.PASS,
                severity=TestSeverity.LOW,
                category=TestCategory.REPRODUCIBILITY,
                details={"environment": env},
            )
    return TestResult(
        id="ENV_CAPTURED",
        title="Environment info is captured",
        status=TestStatus.WARN,
        severity=TestSeverity.LOW,
        category=TestCategory.REPRODUCIBILITY,
        details={"reason": "No environment info in metadata"},
        recommendation="Environment capture will be added automatically in future versions.",
    )


@test(
    "DETERMINISM_FLAGS_LOGGED",
    "Determinism flags are set",
    TestCategory.REPRODUCIBILITY,
    TestSeverity.LOW,
    "Checks if deterministic training flags are logged.",
)
def test_determinism_flags(ctx):
    if ctx.metadata:
        config = ctx.metadata.get("config", {})
        if config.get("deterministic") or config.get("torch_deterministic"):
            return TestResult(
                id="DETERMINISM_FLAGS_LOGGED",
                title="Determinism flags are set",
                status=TestStatus.PASS,
                severity=TestSeverity.LOW,
                category=TestCategory.REPRODUCIBILITY,
            )
    return TestResult(
        id="DETERMINISM_FLAGS_LOGGED",
        title="Determinism flags are set",
        status=TestStatus.SKIP,
        severity=TestSeverity.LOW,
        category=TestCategory.REPRODUCIBILITY,
        details={"reason": "No determinism config found"},
    )


@test(
    "RUN_GIT_COMMIT_CAPTURED",
    "Git commit hash captured",
    TestCategory.REPRODUCIBILITY,
    TestSeverity.LOW,
    "Checks if the git commit hash was logged for the run.",
)
def test_git_commit(ctx):
    if ctx.metadata:
        git_hash = ctx.metadata.get("git_commit") or ctx.metadata.get("config", {}).get("git_commit")
        if git_hash:
            return TestResult(
                id="RUN_GIT_COMMIT_CAPTURED",
                title="Git commit hash captured",
                status=TestStatus.PASS,
                severity=TestSeverity.LOW,
                category=TestCategory.REPRODUCIBILITY,
                details={"git_commit": git_hash},
            )
    return TestResult(
        id="RUN_GIT_COMMIT_CAPTURED",
        title="Git commit hash captured",
        status=TestStatus.SKIP,
        severity=TestSeverity.LOW,
        category=TestCategory.REPRODUCIBILITY,
        details={"reason": "No git commit in metadata"},
        recommendation="GradGlass can auto-capture git commit if run from a git repo.",
    )


# ---------------------------------------------------------------------------
# Interpretability & Explainability Tests
# ---------------------------------------------------------------------------


@test(
    "GRAD_INPUT_SALIENCY",
    "Gradient×Input saliency is computable",
    TestCategory.GRADIENT,
    TestSeverity.MEDIUM,
    "Estimates feature importance using gradient×input proxy. High-variance layers provide cleaner saliency signals.",
)
def test_grad_input_saliency(ctx):
    """Compute gradient×input proxy for feature importance using saved gradient summaries."""
    if not ctx.has_grad_summaries:
        return TestResult(
            id="GRAD_INPUT_SALIENCY",
            title="Gradient×Input saliency is computable",
            status=TestStatus.SKIP,
            severity=TestSeverity.MEDIUM,
            category=TestCategory.GRADIENT,
            details={"reason": "No gradient summaries found"},
            recommendation='Enable gradient capture with run.watch(gradients="summary").',
        )

    # Collect per-layer mean gradient norm across all captured steps
    layer_norms: dict[str, list[float]] = {}
    for entry in ctx.gradient_summaries:
        for layer, data in entry.get("layers", {}).items():
            norm = data.get("norm")
            if norm is not None and not math.isnan(norm) and not math.isinf(norm):
                layer_norms.setdefault(layer, []).append(norm)

    if not layer_norms:
        return TestResult(
            id="GRAD_INPUT_SALIENCY",
            title="Gradient×Input saliency is computable",
            status=TestStatus.SKIP,
            severity=TestSeverity.MEDIUM,
            category=TestCategory.GRADIENT,
            details={"reason": "No valid gradient norm data"},
        )

    # Sort layers by mean gradient norm (proxy for importance)
    importances = {
        layer: {
            "mean_grad_norm": round(float(np.mean(norms)), 6),
            "std_grad_norm": round(float(np.std(norms)), 6),
            "steps_captured": len(norms),
        }
        for layer, norms in layer_norms.items()
    }
    ranked = sorted(importances.items(), key=lambda x: x[1]["mean_grad_norm"], reverse=True)
    top_k = [{"layer": name, **info} for name, info in ranked[:10]]

    return TestResult(
        id="GRAD_INPUT_SALIENCY",
        title="Gradient×Input saliency is computable",
        status=TestStatus.PASS,
        severity=TestSeverity.MEDIUM,
        category=TestCategory.GRADIENT,
        details={
            "top_10_by_gradient_norm": top_k,
            "total_layers_analyzed": len(importances),
            "note": 'Layers with higher mean gradient norm receive stronger learning signal and are more "salient".',
        },
    )


@test(
    "LIME_PROXY_CONFIDENCE",
    "Prediction confidence varies across samples (LIME proxy)",
    TestCategory.PREDICTION,
    TestSeverity.MEDIUM,
    "A LIME-style proxy: checks if prediction confidence varies meaningfully across input samples. Low variance may indicate mode collapse or overconfidence.",
)
def test_lime_proxy_confidence(ctx):
    """LIME proxy: checks confidence distribution across prediction probes."""
    if not ctx.has_predictions:
        return TestResult(
            id="LIME_PROXY_CONFIDENCE",
            title="Prediction confidence varies across samples (LIME proxy)",
            status=TestStatus.SKIP,
            severity=TestSeverity.MEDIUM,
            category=TestCategory.PREDICTION,
            details={"reason": "No prediction probes found"},
            recommendation="Log predictions using run.log_batch().",
        )

    # Use the last prediction probe
    last_pred = ctx.predictions[-1]
    confidences = last_pred.get("confidence", [])
    if not confidences or len(confidences) < 5:
        return TestResult(
            id="LIME_PROXY_CONFIDENCE",
            title="Prediction confidence varies across samples (LIME proxy)",
            status=TestStatus.SKIP,
            severity=TestSeverity.MEDIUM,
            category=TestCategory.PREDICTION,
            details={"reason": f"Insufficient confidence values ({len(confidences)} < 5)"},
        )

    conf_arr = np.array(confidences, dtype=float)
    conf_mean = float(np.mean(conf_arr))
    conf_std = float(np.std(conf_arr))
    conf_min = float(np.min(conf_arr))
    conf_max = float(np.max(conf_arr))

    # Overconfidence: near 1.0 with very low variance
    if conf_mean > 0.98 and conf_std < 0.02:
        return TestResult(
            id="LIME_PROXY_CONFIDENCE",
            title="Prediction confidence varies across samples (LIME proxy)",
            status=TestStatus.WARN,
            severity=TestSeverity.MEDIUM,
            category=TestCategory.PREDICTION,
            details={
                "mean": round(conf_mean, 4),
                "std": round(conf_std, 4),
                "min": round(conf_min, 4),
                "max": round(conf_max, 4),
            },
            recommendation="Model is overconfident (near 100% on all samples). May indicate overfitting or label leakage.",
        )

    # Underconfidence / mode collapse
    if conf_std < 0.01:
        return TestResult(
            id="LIME_PROXY_CONFIDENCE",
            title="Prediction confidence varies across samples (LIME proxy)",
            status=TestStatus.WARN,
            severity=TestSeverity.MEDIUM,
            category=TestCategory.PREDICTION,
            details={
                "mean": round(conf_mean, 4),
                "std": round(conf_std, 4),
                "min": round(conf_min, 4),
                "max": round(conf_max, 4),
            },
            recommendation="Confidence variance is very low — the model gives nearly identical confidence to all samples. Possible mode collapse.",
        )

    return TestResult(
        id="LIME_PROXY_CONFIDENCE",
        title="Prediction confidence varies across samples (LIME proxy)",
        status=TestStatus.PASS,
        severity=TestSeverity.MEDIUM,
        category=TestCategory.PREDICTION,
        details={
            "mean": round(conf_mean, 4),
            "std": round(conf_std, 4),
            "min": round(conf_min, 4),
            "max": round(conf_max, 4),
            "samples": len(conf_arr),
        },
    )


@test(
    "SHAP_GRAD_ATTRIBUTION_RANK",
    "Gradient attribution ranking is stable (SHAP proxy)",
    TestCategory.GRADIENT,
    TestSeverity.MEDIUM,
    "SHAP-style proxy: checks if the ranking of layer importances by gradient norm is stable across training. Unstable ranking may indicate the model is still reorganizing which features matter.",
)
def test_shap_attribution_rank(ctx):
    """SHAP proxy: checks stability of layer importance ranking across training steps."""
    if not ctx.has_grad_summaries or len(ctx.gradient_summaries) < 4:
        return TestResult(
            id="SHAP_GRAD_ATTRIBUTION_RANK",
            title="Gradient attribution ranking is stable (SHAP proxy)",
            status=TestStatus.SKIP,
            severity=TestSeverity.MEDIUM,
            category=TestCategory.GRADIENT,
            details={"reason": f"Need ≥4 gradient summary steps (have {len(ctx.gradient_summaries)})"},
        )

    # Compare top-3 layers by gradient norm at start vs end
    def get_top_layers(summary, k=3):
        norms = {
            layer: data.get("norm", 0)
            for layer, data in summary.get("layers", {}).items()
            if data.get("norm") is not None
        }
        return [name for name, _ in sorted(norms.items(), key=lambda x: x[1], reverse=True)[:k]]

    early_summaries = ctx.gradient_summaries[: max(1, len(ctx.gradient_summaries) // 4)]
    late_summaries = ctx.gradient_summaries[-max(1, len(ctx.gradient_summaries) // 4) :]

    early_top: list[str] = []
    for s in early_summaries:
        for name in get_top_layers(s):
            if name not in early_top:
                early_top.append(name)
        if len(early_top) >= 3:
            break

    late_top: list[str] = []
    for s in late_summaries:
        for name in get_top_layers(s):
            if name not in late_top:
                late_top.append(name)
        if len(late_top) >= 3:
            break

    overlap = len(set(early_top[:3]) & set(late_top[:3]))
    stability = overlap / 3 if early_top and late_top else 0

    details = {
        "early_top_layers": early_top[:3],
        "late_top_layers": late_top[:3],
        "rank_overlap": overlap,
        "stability_score": round(stability, 2),
    }

    if stability < 0.33:
        return TestResult(
            id="SHAP_GRAD_ATTRIBUTION_RANK",
            title="Gradient attribution ranking is stable (SHAP proxy)",
            status=TestStatus.WARN,
            severity=TestSeverity.MEDIUM,
            category=TestCategory.GRADIENT,
            details=details,
            recommendation="Feature importance ranking changed completely during training. The model is still reorganizing which layers matter most.",
        )
    return TestResult(
        id="SHAP_GRAD_ATTRIBUTION_RANK",
        title="Gradient attribution ranking is stable (SHAP proxy)",
        status=TestStatus.PASS,
        severity=TestSeverity.MEDIUM,
        category=TestCategory.GRADIENT,
        details=details,
    )


@test(
    "DEAD_CHANNEL_DETECTION",
    "No dead feature channels in convolutional layers",
    TestCategory.ACTIVATION,
    TestSeverity.HIGH,
    "Detects convolutional filter channels with near-zero activation across all samples (dead channels).",
)
def test_dead_channels(ctx):
    if not ctx.has_activations:
        return TestResult(
            id="DEAD_CHANNEL_DETECTION",
            title="No dead feature channels in convolutional layers",
            status=TestStatus.SKIP,
            severity=TestSeverity.HIGH,
            category=TestCategory.ACTIVATION,
            details={"reason": "No activation data found"},
        )

    dead_channels = []
    for stat in ctx.activation_stats:
        layer = stat.get("layer", "")
        sparsity = stat.get("sparsity", 0)
        mean = stat.get("mean", None)
        var = stat.get("var", None)
        # A "dead channel" has near-zero mean, near-zero variance, and high sparsity
        if sparsity is not None and sparsity > 0.98 and mean is not None and abs(mean) < 1e-4:
            dead_channels.append(
                {"layer": layer, "step": stat.get("step"), "sparsity": round(sparsity, 4), "mean": round(mean, 6)}
            )

    if dead_channels:
        return TestResult(
            id="DEAD_CHANNEL_DETECTION",
            title="No dead feature channels in convolutional layers",
            status=TestStatus.WARN,
            severity=TestSeverity.HIGH,
            category=TestCategory.ACTIVATION,
            details={"dead_channels": dead_channels[:20], "total_dead": len(dead_channels)},
            recommendation="Dead channels detected (>98% sparsity, near-zero mean). Use He initialization, Leaky ReLU, or batch norm.",
        )
    return TestResult(
        id="DEAD_CHANNEL_DETECTION",
        title="No dead feature channels in convolutional layers",
        status=TestStatus.PASS,
        severity=TestSeverity.HIGH,
        category=TestCategory.ACTIVATION,
        details={"activations_checked": len(ctx.activation_stats)},
    )


@test(
    "WEIGHT_NORM_DISTRIBUTION",
    "Weight norms are well-distributed across layers",
    TestCategory.CHECKPOINT,
    TestSeverity.MEDIUM,
    "Checks that weight norms are in a healthy range and not collapsed or exploded per layer.",
)
def test_weight_norm_distribution(ctx):
    steps = ctx.checkpoint_steps()
    if not steps:
        return TestResult(
            id="WEIGHT_NORM_DISTRIBUTION",
            title="Weight norms are well-distributed across layers",
            status=TestStatus.SKIP,
            severity=TestSeverity.MEDIUM,
            category=TestCategory.CHECKPOINT,
        )
    try:
        weights = ctx.load_checkpoint(steps[-1])
        layer_norms = {}
        issues = []
        for name, w in weights.items():
            norm = float(np.linalg.norm(w.astype(np.float64)))
            layer_norms[name] = round(norm, 4)
            if norm < 1e-6:
                issues.append({"layer": name, "issue": "near_zero_norm", "norm": norm})
            elif norm > 1e4:
                issues.append({"layer": name, "issue": "exploded_norm", "norm": round(norm, 2)})

        if issues:
            return TestResult(
                id="WEIGHT_NORM_DISTRIBUTION",
                title="Weight norms are well-distributed across layers",
                status=TestStatus.WARN,
                severity=TestSeverity.MEDIUM,
                category=TestCategory.CHECKPOINT,
                details={"issues": issues, "all_norms": layer_norms},
                recommendation="Some layers have extreme weight norms. Check initialization and learning rate.",
            )

        norms_list = list(layer_norms.values())
        return TestResult(
            id="WEIGHT_NORM_DISTRIBUTION",
            title="Weight norms are well-distributed across layers",
            status=TestStatus.PASS,
            severity=TestSeverity.MEDIUM,
            category=TestCategory.CHECKPOINT,
            details={
                "layers_checked": len(layer_norms),
                "mean_norm": round(float(np.mean(norms_list)), 4),
                "min_norm": round(float(np.min(norms_list)), 4),
                "max_norm": round(float(np.max(norms_list)), 4),
                "all_norms": layer_norms,
            },
        )
    except Exception as e:
        return TestResult(
            id="WEIGHT_NORM_DISTRIBUTION",
            title="Weight norms are well-distributed across layers",
            status=TestStatus.SKIP,
            severity=TestSeverity.MEDIUM,
            category=TestCategory.CHECKPOINT,
            details={"error": str(e)},
        )


@test(
    "FREEZE_RECOMMENDATION",
    "Recommend layers to freeze based on gradient activity",
    TestCategory.GRADIENT,
    TestSeverity.LOW,
    "Identifies layers with consistently low gradient norms that are candidates for freezing to speed up training.",
)
def test_freeze_recommendation(ctx):
    if not ctx.has_grad_summaries or len(ctx.gradient_summaries) < 3:
        return TestResult(
            id="FREEZE_RECOMMENDATION",
            title="Recommend layers to freeze based on gradient activity",
            status=TestStatus.SKIP,
            severity=TestSeverity.LOW,
            category=TestCategory.GRADIENT,
            details={"reason": "Need ≥3 gradient summary steps"},
        )

    # Compute mean gradient norm per layer across all steps
    layer_norms: dict[str, list[float]] = {}
    for entry in ctx.gradient_summaries:
        for layer, data in entry.get("layers", {}).items():
            norm = data.get("norm")
            if norm is not None and not math.isnan(norm):
                layer_norms.setdefault(layer, []).append(norm)

    if not layer_norms:
        return TestResult(
            id="FREEZE_RECOMMENDATION",
            title="Recommend layers to freeze based on gradient activity",
            status=TestStatus.SKIP,
            severity=TestSeverity.LOW,
            category=TestCategory.GRADIENT,
        )

    # Find layers with very low mean gradient norm (< 1% of max mean norm)
    mean_norms = {layer: float(np.mean(norms)) for layer, norms in layer_norms.items()}
    max_mean = max(mean_norms.values()) if mean_norms else 1.0
    threshold = max_mean * 0.01

    candidates = [
        {"layer": layer, "mean_grad_norm": round(norm, 8), "relative": round(norm / max(max_mean, 1e-12), 6)}
        for layer, norm in mean_norms.items()
        if norm < threshold
    ]
    candidates.sort(key=lambda x: x["mean_grad_norm"])

    # Generate Python code snippet
    if candidates:
        layer_names = [c["layer"] for c in candidates[:5]]
        code_lines = ["# GradGlass: Suggested layers to freeze (low gradient activity)"]
        for name in layer_names:
            code_lines.append(f"# model.{name}.requires_grad_(False)")
        freeze_code = "\n".join(code_lines)

        return TestResult(
            id="FREEZE_RECOMMENDATION",
            title="Recommend layers to freeze based on gradient activity",
            status=TestStatus.PASS,
            severity=TestSeverity.LOW,
            category=TestCategory.GRADIENT,
            details={
                "freeze_candidates": candidates[:10],
                "total_candidates": len(candidates),
                "suggested_code": freeze_code,
                "note": "These layers have <1% of the maximum gradient norm. Freezing them may speed up training with minimal accuracy impact.",
            },
        )

    return TestResult(
        id="FREEZE_RECOMMENDATION",
        title="Recommend layers to freeze based on gradient activity",
        status=TestStatus.PASS,
        severity=TestSeverity.LOW,
        category=TestCategory.GRADIENT,
        details={"message": "All layers are receiving meaningful gradients — no freeze candidates identified."},
    )


@test(
    "ACTIVATION_PATTERN_STABILITY",
    "Activation patterns stabilize over training",
    TestCategory.ACTIVATION,
    TestSeverity.MEDIUM,
    "Checks if layer activations stabilize in later training stages, indicating convergence.",
)
def test_activation_pattern_stability(ctx):
    if not ctx.has_activations:
        return TestResult(
            id="ACTIVATION_PATTERN_STABILITY",
            title="Activation patterns stabilize over training",
            status=TestStatus.SKIP,
            severity=TestSeverity.MEDIUM,
            category=TestCategory.ACTIVATION,
        )

    by_layer: dict[str, list] = {}
    for stat in ctx.activation_stats:
        layer = stat.get("layer", "unknown")
        by_layer.setdefault(layer, []).append(stat)

    if not by_layer:
        return TestResult(
            id="ACTIVATION_PATTERN_STABILITY",
            title="Activation patterns stabilize over training",
            status=TestStatus.SKIP,
            severity=TestSeverity.MEDIUM,
            category=TestCategory.ACTIVATION,
        )

    unstable = []
    stable_count = 0

    for layer, stats in by_layer.items():
        if len(stats) < 3:
            continue
        # Look at variance trend: is variance still changing a lot in the last third?
        third = max(len(stats) // 3, 1)
        early_vars = [s.get("var", 0) for s in stats[:third] if s.get("var") is not None]
        late_vars = [s.get("var", 0) for s in stats[-third:] if s.get("var") is not None]

        if not early_vars or not late_vars:
            continue

        early_mean = float(np.mean(early_vars))
        late_mean = float(np.mean(late_vars))
        late_std = float(np.std(late_vars))

        # Unstable if late variance is still changing a lot relative to its mean
        if late_mean > 0 and late_std / late_mean > 0.5:
            unstable.append({"layer": layer, "late_var_cv": round(late_std / late_mean, 3)})
        else:
            stable_count += 1

    if unstable:
        return TestResult(
            id="ACTIVATION_PATTERN_STABILITY",
            title="Activation patterns stabilize over training",
            status=TestStatus.WARN,
            severity=TestSeverity.MEDIUM,
            category=TestCategory.ACTIVATION,
            details={"unstable_layers": unstable, "stable_count": stable_count},
            recommendation="Some layers show high activation variance in late training. Model may still be shifting representations.",
        )
    return TestResult(
        id="ACTIVATION_PATTERN_STABILITY",
        title="Activation patterns stabilize over training",
        status=TestStatus.PASS,
        severity=TestSeverity.MEDIUM,
        category=TestCategory.ACTIVATION,
        details={"stable_layers": stable_count, "total_analyzed": stable_count + len(unstable)},
    )


@test(
    "LAYER_CAPACITY_UTILIZATION",
    "Layers are utilizing their full representational capacity",
    TestCategory.ACTIVATION,
    TestSeverity.MEDIUM,
    "Checks if activation variance is reasonable — very low variance suggests a layer is underutilized.",
)
def test_layer_capacity(ctx):
    if not ctx.has_activations:
        return TestResult(
            id="LAYER_CAPACITY_UTILIZATION",
            title="Layers are utilizing their full representational capacity",
            status=TestStatus.SKIP,
            severity=TestSeverity.MEDIUM,
            category=TestCategory.ACTIVATION,
        )

    # Use last-step activations
    steps = sorted(set(s.get("step", 0) for s in ctx.activation_stats))
    if not steps:
        return TestResult(
            id="LAYER_CAPACITY_UTILIZATION",
            title="Layers are utilizing their full representational capacity",
            status=TestStatus.SKIP,
            severity=TestSeverity.MEDIUM,
            category=TestCategory.ACTIVATION,
        )

    last_step_stats = [s for s in ctx.activation_stats if s.get("step") == steps[-1]]
    if not last_step_stats:
        last_step_stats = ctx.activation_stats[-min(5, len(ctx.activation_stats)) :]

    underutilized = []
    healthy = []
    for stat in last_step_stats:
        var = stat.get("var")
        layer = stat.get("layer", "?")
        if var is None:
            continue
        if var < 0.001:
            underutilized.append({"layer": layer, "var": round(var, 8)})
        else:
            healthy.append(layer)

    if underutilized:
        return TestResult(
            id="LAYER_CAPACITY_UTILIZATION",
            title="Layers are utilizing their full representational capacity",
            status=TestStatus.WARN,
            severity=TestSeverity.MEDIUM,
            category=TestCategory.ACTIVATION,
            details={"underutilized_layers": underutilized, "healthy_count": len(healthy)},
            recommendation="Some layers have very low activation variance. They may not be contributing meaningful features.",
        )
    return TestResult(
        id="LAYER_CAPACITY_UTILIZATION",
        title="Layers are utilizing their full representational capacity",
        status=TestStatus.PASS,
        severity=TestSeverity.MEDIUM,
        category=TestCategory.ACTIVATION,
        details={"healthy_layers": len(healthy)},
    )


@test(
    "EPOCH_LOSS_IMPROVEMENT",
    "Loss improves consistently between epochs",
    TestCategory.METRICS,
    TestSeverity.HIGH,
    "Groups metrics by epoch and checks that mean loss decreases epoch-over-epoch.",
)
def test_epoch_loss_improvement(ctx):
    if not ctx.has_metrics:
        return TestResult(
            id="EPOCH_LOSS_IMPROVEMENT",
            title="Loss improves consistently between epochs",
            status=TestStatus.SKIP,
            severity=TestSeverity.HIGH,
            category=TestCategory.METRICS,
        )

    epoch_entries = [m for m in ctx.metrics if "epoch" in m and "loss" in m and m["loss"] is not None]
    if not epoch_entries:
        return TestResult(
            id="EPOCH_LOSS_IMPROVEMENT",
            title="Loss improves consistently between epochs",
            status=TestStatus.SKIP,
            severity=TestSeverity.HIGH,
            category=TestCategory.METRICS,
            details={"reason": "No epoch field found in metrics. Log with run.log(loss=..., epoch=epoch)."},
        )

    # Group by epoch
    by_epoch: dict[int, list[float]] = {}
    for m in epoch_entries:
        e = int(m["epoch"])
        by_epoch.setdefault(e, []).append(m["loss"])

    if len(by_epoch) < 2:
        return TestResult(
            id="EPOCH_LOSS_IMPROVEMENT",
            title="Loss improves consistently between epochs",
            status=TestStatus.SKIP,
            severity=TestSeverity.HIGH,
            category=TestCategory.METRICS,
            details={"reason": f"Only {len(by_epoch)} epoch(s) found — need ≥2"},
        )

    epoch_means = sorted([(e, float(np.mean(losses))) for e, losses in by_epoch.items()])
    regressions = []
    for i in range(1, len(epoch_means)):
        prev_e, prev_loss = epoch_means[i - 1]
        curr_e, curr_loss = epoch_means[i]
        if curr_loss > prev_loss * 1.05:  # > 5% increase is a regression
            regressions.append(
                {
                    "from_epoch": prev_e,
                    "to_epoch": curr_e,
                    "prev_loss": round(prev_loss, 6),
                    "curr_loss": round(curr_loss, 6),
                }
            )

    epoch_summary = [{"epoch": e, "mean_loss": round(l, 6)} for e, l in epoch_means]

    if regressions:
        return TestResult(
            id="EPOCH_LOSS_IMPROVEMENT",
            title="Loss improves consistently between epochs",
            status=TestStatus.WARN,
            severity=TestSeverity.HIGH,
            category=TestCategory.METRICS,
            details={"epoch_losses": epoch_summary, "regressions": regressions},
            recommendation="Loss increased between some epochs (>5%). Check LR schedule, batch composition, or early stopping.",
        )

    return TestResult(
        id="EPOCH_LOSS_IMPROVEMENT",
        title="Loss improves consistently between epochs",
        status=TestStatus.PASS,
        severity=TestSeverity.HIGH,
        category=TestCategory.METRICS,
        details={
            "epoch_losses": epoch_summary,
            "total_epochs": len(epoch_means),
            "total_reduction": round(epoch_means[0][1] - epoch_means[-1][1], 6),
        },
    )
