from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np

TASK_TYPES = (
    "classification",
    "regression",
    "sequence_generation",
    "vision",
    "reinforcement_learning",
    "retrieval_embedding",
    "time_series_forecasting",
)

TASK_DISPLAY_NAMES = {
    "classification": "Classification",
    "regression": "Regression",
    "sequence_generation": "Sequence Generation",
    "vision": "Vision",
    "reinforcement_learning": "Reinforcement Learning",
    "retrieval_embedding": "Retrieval / Embeddings",
    "time_series_forecasting": "Time Series Forecasting",
}

HIGHER_IS_BETTER = {
    "accuracy",
    "precision",
    "recall",
    "macro_f1",
    "micro_f1",
    "r2",
    "mape_inverse",
    "bleu",
    "rouge_l",
    "semantic_similarity",
    "cosine_similarity",
    "recall_at_1",
    "recall_at_5",
    "top_1_accuracy",
    "top_5_accuracy",
    "mean_iou",
    "mAP_50",
    "mean_return",
    "success_rate",
}

METRIC_DISPLAY_NAMES = {
    "accuracy": "Latest Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "macro_f1": "Macro F1",
    "micro_f1": "Micro F1",
    "mse": "MSE",
    "rmse": "RMSE",
    "mae": "MAE",
    "r2": "R²",
    "mape": "MAPE",
    "bleu": "BLEU",
    "rouge_l": "ROUGE-L",
    "semantic_similarity": "Semantic Similarity",
    "cosine_similarity": "Cosine Similarity",
    "recall_at_1": "Recall@1",
    "recall_at_5": "Recall@5",
    "top_1_accuracy": "Top-1 Accuracy",
    "top_5_accuracy": "Top-5 Accuracy",
    "mean_iou": "Mean IoU",
    "mAP_50": "mAP@0.50",
    "mean_return": "Mean Return",
    "success_rate": "Success Rate",
}

METRIC_JUSTIFICATIONS = {
    "accuracy": "Discrete ground-truth labels and predictions are available.",
    "precision": "Class predictions can be compared against true labels to measure false positives.",
    "recall": "Class predictions can be compared against true labels to measure false negatives.",
    "macro_f1": "Per-class precision and recall are available, so balanced class performance can be summarized.",
    "micro_f1": "Single-label classification predictions support an aggregate F1 summary.",
    "mse": "Numeric targets and predictions are available, so squared error can be measured.",
    "rmse": "Numeric targets and predictions are available, and RMSE keeps error in target units.",
    "mae": "Numeric targets and predictions are available, so absolute error can be measured.",
    "r2": "Numeric targets and predictions are available, so explained variance can be estimated.",
    "mape": "Targets are numeric and mostly non-zero, so relative percentage error is measurable.",
    "bleu": "Generated text and reference text are both available.",
    "rouge_l": "Generated text and reference text are both available for overlap-based comparison.",
    "semantic_similarity": "Embeddings are available for output/reference similarity.",
    "cosine_similarity": "Embedding vectors are available for similarity measurement.",
    "recall_at_1": "Retrieved rankings and relevance labels are available.",
    "recall_at_5": "Retrieved rankings and relevance labels are available.",
    "top_1_accuracy": "Vision-classification labels and predictions are available.",
    "top_5_accuracy": "Ranked class predictions are available.",
    "mean_iou": "Predicted and reference masks are available.",
    "mAP_50": "Predicted and reference boxes with labels are available.",
    "mean_return": "Episode returns or rewards are available.",
    "success_rate": "Outcome signals are available for success/failure aggregation.",
}

PERCENT_METRICS = {
    "accuracy",
    "precision",
    "recall",
    "macro_f1",
    "micro_f1",
    "mape",
    "bleu",
    "rouge_l",
    "semantic_similarity",
    "recall_at_1",
    "recall_at_5",
    "top_1_accuracy",
    "top_5_accuracy",
    "mean_iou",
    "mAP_50",
    "success_rate",
}

METRIC_PREFERENCES = {
    "classification": ["accuracy", "macro_f1", "precision", "recall", "micro_f1"],
    "regression": ["rmse", "mae", "mse", "r2", "mape"],
    "time_series_forecasting": ["rmse", "mae", "mape", "mse", "r2"],
    "sequence_generation": ["rouge_l", "bleu", "semantic_similarity"],
    "retrieval_embedding": ["recall_at_5", "recall_at_1", "cosine_similarity"],
    "vision": ["mAP_50", "mean_iou", "top_1_accuracy", "top_5_accuracy", "accuracy"],
    "reinforcement_learning": ["mean_return", "success_rate"],
}

METRIC_KEY_ALIASES = {
    "classification": {
        "accuracy": ["val_accuracy", "validation_accuracy", "test_accuracy", "accuracy", "val_acc", "acc"],
        "macro_f1": ["val_macro_f1", "macro_f1", "f1", "val_f1"],
        "precision": ["val_precision", "precision", "macro_precision"],
        "recall": ["val_recall", "recall", "macro_recall"],
    },
    "regression": {
        "rmse": ["val_rmse", "test_rmse", "rmse"],
        "mae": ["val_mae", "test_mae", "mae"],
        "mse": ["val_mse", "test_mse", "mse", "val_loss", "loss"],
        "r2": ["val_r2", "test_r2", "r2"],
        "mape": ["val_mape", "test_mape", "mape"],
    },
    "time_series_forecasting": {
        "rmse": ["val_rmse", "forecast_rmse", "rmse"],
        "mae": ["val_mae", "forecast_mae", "mae"],
        "mse": ["val_mse", "forecast_mse", "mse", "val_loss", "loss"],
        "mape": ["val_mape", "forecast_mape", "mape"],
        "r2": ["val_r2", "r2"],
    },
    "sequence_generation": {
        "bleu": ["bleu", "val_bleu", "test_bleu"],
        "rouge_l": ["rouge_l", "rouge", "val_rouge_l", "test_rouge_l"],
        "semantic_similarity": ["semantic_similarity", "embedding_similarity"],
    },
    "retrieval_embedding": {
        "recall_at_1": ["recall_at_1", "r_at_1"],
        "recall_at_5": ["recall_at_5", "r_at_5"],
        "cosine_similarity": ["cosine_similarity", "embedding_cosine_similarity"],
    },
    "vision": {
        "mAP_50": ["map_50", "mAP_50", "map", "mAP"],
        "mean_iou": ["mean_iou", "iou", "val_iou"],
        "top_1_accuracy": ["top_1_accuracy", "top1_accuracy", "top1", "accuracy", "acc"],
        "top_5_accuracy": ["top_5_accuracy", "top5_accuracy", "top5"],
    },
    "reinforcement_learning": {
        "mean_return": ["mean_return", "avg_return", "return", "episode_return", "reward"],
        "success_rate": ["success_rate", "win_rate"],
    },
}

BENCHMARK_FAMILY_ELIGIBILITY = {
    "classification": [],
    "regression": [],
    "sequence_generation": ["llm"],
    "vision": ["vision"],
    "reinforcement_learning": [],
    "retrieval_embedding": [],
    "time_series_forecasting": [],
}

BENCHMARK_FAMILY_DISPLAY_NAMES = {
    "llm": "LLM Benchmarks",
    "vision": "Vision Benchmarks",
}


def build_evaluation_payload(
    run_id: str,
    *,
    metadata: Optional[dict[str, Any]] = None,
    metrics: Optional[list[dict[str, Any]]] = None,
    predictions: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    metadata = metadata or {}
    metrics = metrics or []
    predictions = predictions or []

    task_distribution, evidence = infer_task_distribution(predictions=predictions, metrics=metrics, metadata=metadata)
    inferred_task = max(task_distribution, key=task_distribution.get)
    modalities = detect_modalities(predictions=predictions, metadata=metadata, inferred_task=inferred_task)
    benchmark_state = build_benchmark_state(metadata=metadata, inferred_task=inferred_task)

    evaluations = build_evaluations(
        predictions=predictions,
        metrics=metrics,
        metadata=metadata,
        inferred_task=inferred_task,
        modalities=modalities,
    )
    latest_evaluation = evaluations[-1] if evaluations else None

    missing_artifacts = collect_missing_artifacts(
        inferred_task=inferred_task,
        predictions=predictions,
        latest_evaluation=latest_evaluation,
    )
    selected_metrics = build_selected_metrics(inferred_task=inferred_task, latest_evaluation=latest_evaluation)
    performance_summary = build_performance_summary(
        inferred_task=inferred_task,
        latest_evaluation=latest_evaluation,
        selected_metrics=selected_metrics,
    )
    trend_analysis = build_trend_analysis(inferred_task=inferred_task, evaluations=evaluations)
    error_analysis = build_error_analysis(
        inferred_task=inferred_task,
        latest_evaluation=latest_evaluation,
        metrics=metrics,
        missing_artifacts=missing_artifacts,
    )
    modality_analysis = build_modality_analysis(modalities=modalities, predictions=predictions)
    recommendations = build_recommendations(
        inferred_task=inferred_task,
        latest_evaluation=latest_evaluation,
        missing_artifacts=missing_artifacts,
        error_analysis=error_analysis,
        ambiguity=(task_distribution[inferred_task] < 0.6),
    )

    return {
        "run_id": run_id,
        "output_format_version": 2,
        "inferred_task_type": inferred_task,
        "inferred_task_type_display": TASK_DISPLAY_NAMES[inferred_task],
        "confidence_in_task_inference": float(task_distribution[inferred_task]),
        "task_type_distribution": task_distribution,
        "task_inference_evidence": evidence,
        "selected_metrics": selected_metrics,
        "performance_summary": performance_summary,
        "trend_analysis": trend_analysis,
        "error_analysis": error_analysis,
        "recommendations": recommendations,
        "modality_analysis": modality_analysis,
        "missing_artifacts": missing_artifacts,
        "evaluations": evaluations,
        "benchmark_state": benchmark_state,
        "benchmark_alignment": {
            "available": False,
            "summary": None,
            "message": "No benchmark alignment claim is shown without compatible benchmark artifacts.",
        },
    }


def _coerce_bool_flag(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off", ""}:
            return False
    return bool(value)


def benchmarks_enabled(metadata: dict[str, Any]) -> bool:
    config = metadata.get("config") if isinstance(metadata, dict) else {}
    if isinstance(config, dict) and "enable_benchmarks" in config:
        return _coerce_bool_flag(config.get("enable_benchmarks"), default=False)
    if isinstance(metadata, dict) and "enable_benchmarks" in metadata:
        return _coerce_bool_flag(metadata.get("enable_benchmarks"), default=False)
    return False


def build_benchmark_state(*, metadata: dict[str, Any], inferred_task: str) -> dict[str, Any]:
    enabled = benchmarks_enabled(metadata)
    if not enabled:
        return {
            "enabled": False,
            "eligible_families": [],
            "message": "Benchmarks disabled for this run.",
        }

    eligible_families = list(BENCHMARK_FAMILY_ELIGIBILITY.get(inferred_task, []))
    if eligible_families:
        eligible_names = [BENCHMARK_FAMILY_DISPLAY_NAMES.get(name, name) for name in eligible_families]
        message = f"Compatible benchmark suites: {', '.join(eligible_names)}."
    else:
        task_label = TASK_DISPLAY_NAMES.get(inferred_task, inferred_task.replace("_", " "))
        message = f"Benchmarks are enabled, but {task_label} runs do not have a compatible benchmark family."

    return {
        "enabled": True,
        "eligible_families": eligible_families,
        "message": message,
    }


def infer_task_distribution(
    *, predictions: list[dict[str, Any]], metrics: list[dict[str, Any]], metadata: dict[str, Any]
) -> tuple[dict[str, float], list[str]]:
    scores = {task: 0.25 for task in TASK_TYPES}
    evidence: list[str] = []

    task_hint = str(((metadata.get("config") or {}).get("task") or metadata.get("task") or "")).strip().lower()
    if task_hint:
        if any(token in task_hint for token in ("class", "label")):
            scores["classification"] += 4.0
            evidence.append(f"Metadata task hint suggests classification ({task_hint}).")
        if any(token in task_hint for token in ("regress",)):
            scores["regression"] += 4.0
            evidence.append(f"Metadata task hint suggests regression ({task_hint}).")
        if any(token in task_hint for token in ("time", "forecast", "timeseries", "time-series", "sequence forecast")):
            scores["time_series_forecasting"] += 5.0
            evidence.append(f"Metadata task hint suggests time-series forecasting ({task_hint}).")
        if any(token in task_hint for token in ("retriev", "embed", "search")):
            scores["retrieval_embedding"] += 5.0
            evidence.append(f"Metadata task hint suggests retrieval/embeddings ({task_hint}).")
        if any(token in task_hint for token in ("rl", "reinforcement", "policy", "control")):
            scores["reinforcement_learning"] += 5.0
            evidence.append(f"Metadata task hint suggests reinforcement learning ({task_hint}).")
        if any(token in task_hint for token in ("generation", "translate", "summar", "llm", "text")):
            scores["sequence_generation"] += 5.0
            evidence.append(f"Metadata task hint suggests sequence generation ({task_hint}).")
        if any(token in task_hint for token in ("vision", "image", "detect", "segment")):
            scores["vision"] += 5.0
            evidence.append(f"Metadata task hint suggests a vision task ({task_hint}).")

    metric_keys = {str(key).lower() for row in metrics for key, value in row.items() if isinstance(value, (int, float))}
    if metric_keys:
        if {"accuracy", "acc", "macro_f1", "f1", "precision", "recall"} & metric_keys:
            scores["classification"] += 2.5
            evidence.append("Logged metrics contain classification-style keys (accuracy/F1/precision/recall).")
        if {"rmse", "mse", "mae", "r2"} & metric_keys:
            scores["regression"] += 2.0
            evidence.append("Logged metrics contain regression-style keys (RMSE/MSE/MAE/R²).")
        if {"val_mape", "mape", "forecast_rmse", "forecast_mae"} & metric_keys:
            scores["time_series_forecasting"] += 3.0
            evidence.append("Logged metrics contain forecasting-style keys (MAPE/forecast errors).")
        if {"bleu", "rouge", "rouge_l", "perplexity"} & metric_keys:
            scores["sequence_generation"] += 3.0
            evidence.append("Logged metrics contain text-generation-style keys (BLEU/ROUGE/perplexity).")
        if any(key.startswith("recall_at_") for key in metric_keys) or "cosine_similarity" in metric_keys:
            scores["retrieval_embedding"] += 3.0
            evidence.append("Logged metrics contain retrieval or embedding similarity keys.")
        if {"map", "map_50", "iou", "mean_iou", "top1", "top_1_accuracy"} & metric_keys:
            scores["vision"] += 3.0
            evidence.append("Logged metrics contain vision-style keys (mAP/IoU/Top-k accuracy).")
        if {"reward", "return", "episode_return", "success_rate"} & metric_keys:
            scores["reinforcement_learning"] += 3.0
            evidence.append("Logged metrics contain reinforcement-learning-style keys (reward/return).")

    for record in predictions:
        record_scores, record_evidence = _score_prediction_record(record)
        for task, value in record_scores.items():
            scores[task] += value
        evidence.extend(record_evidence)

    total = sum(max(value, 0.0) for value in scores.values()) or 1.0
    distribution = {task: float(max(value, 0.0) / total) for task, value in scores.items()}
    distribution = dict(sorted(distribution.items(), key=lambda item: item[1], reverse=True))
    return distribution, evidence[:8]


def build_evaluations(
    *,
    predictions: list[dict[str, Any]],
    metrics: list[dict[str, Any]],
    metadata: dict[str, Any],
    inferred_task: str,
    modalities: list[str],
) -> list[dict[str, Any]]:
    evaluations: list[dict[str, Any]] = []

    for record in predictions:
        snapshot = evaluate_prediction_record(record=record, metadata=metadata, modalities=modalities)
        if snapshot is not None:
            evaluations.append(snapshot)

    evaluations.sort(key=lambda item: (_safe_int(item.get("step")) is None, _safe_int(item.get("step")) or 0))
    if evaluations:
        return evaluations

    metric_snapshots = build_metric_snapshots(metrics=metrics, inferred_task=inferred_task)
    if metric_snapshots:
        return metric_snapshots

    return []


def evaluate_prediction_record(
    *, record: dict[str, Any], metadata: dict[str, Any], modalities: list[str]
) -> Optional[dict[str, Any]]:
    distribution, evidence = infer_task_distribution(predictions=[record], metrics=[], metadata=metadata)
    task_type = max(distribution, key=distribution.get)

    step = _safe_int(record.get("step"))
    snapshot: dict[str, Any] = {
        "step": step,
        "task_type": task_type,
        "task_type_display": TASK_DISPLAY_NAMES[task_type],
        "task_distribution": distribution,
        "task_confidence": float(distribution[task_type]),
        "modalities": modalities,
        "task_inference_evidence": evidence[:4],
        "metric_values": {},
        "selected_metrics": [],
        "missing_artifacts": [],
        "is_classification": task_type == "classification",
    }

    y_true = _to_array(record.get("y_true"))
    y_pred = _to_array(record.get("y_pred"))
    confidence = _to_float_array(record.get("confidence"))

    if task_type == "classification":
        if y_true is None or y_pred is None:
            return None
        if _is_multilabel_pair(y_true, y_pred):
            payload = _evaluate_multilabel_classification(y_true, y_pred)
        else:
            payload = _evaluate_classification(y_true, y_pred, confidence=confidence)
        snapshot.update(payload)
    elif task_type == "vision":
        payload = _evaluate_vision_record(record, y_true=y_true, y_pred=y_pred)
        if payload is None:
            return None
        snapshot.update(payload)
    elif task_type in {"regression", "time_series_forecasting"}:
        if y_true is None or y_pred is None:
            return None
        payload = _evaluate_regression_like(y_true, y_pred, time_series=(task_type == "time_series_forecasting"))
        snapshot.update(payload)
    elif task_type == "sequence_generation":
        payload = _evaluate_generation(record)
        if payload is None:
            return None
        snapshot.update(payload)
    elif task_type == "retrieval_embedding":
        payload = _evaluate_retrieval(record)
        if payload is None:
            return None
        snapshot.update(payload)
    elif task_type == "reinforcement_learning":
        payload = _evaluate_reinforcement_learning(record)
        if payload is None:
            return None
        snapshot.update(payload)
    else:
        return None

    snapshot["selected_metrics"] = build_selected_metrics(inferred_task=task_type, latest_evaluation=snapshot)
    snapshot["missing_artifacts"] = collect_missing_artifacts(
        inferred_task=task_type,
        predictions=[record],
        latest_evaluation=snapshot,
    )
    snapshot["metric_values"] = extract_metric_values(snapshot)
    return snapshot


def build_metric_snapshots(metrics: list[dict[str, Any]], inferred_task: str) -> list[dict[str, Any]]:
    alias_map = METRIC_KEY_ALIASES.get(inferred_task, {})
    if not alias_map:
        return []

    preferred_rows = [row for row in metrics if any(str(key).lower().startswith(("val_", "test_")) for key in row.keys())]
    rows = preferred_rows or metrics
    snapshots: list[dict[str, Any]] = []

    for row in rows:
        metric_values: dict[str, float] = {}
        for canonical_name, aliases in alias_map.items():
            value = _extract_first_numeric(row, aliases)
            if value is not None:
                metric_values[canonical_name] = value

        if not metric_values:
            continue

        step = _safe_int(row.get("step"))
        snapshot = {
            "step": step,
            "task_type": inferred_task,
            "task_type_display": TASK_DISPLAY_NAMES[inferred_task],
            "metric_values": metric_values,
            "is_classification": inferred_task == "classification",
            "selected_metrics": [
                {"name": name, "justification": METRIC_JUSTIFICATIONS.get(name, "Logged metric available.")}
                for name in metric_values.keys()
            ],
            "missing_artifacts": [
                "Per-example predictions and targets were not logged for this step.",
            ],
        }
        snapshot.update(metric_values)
        if inferred_task == "classification":
            snapshot.setdefault("accuracy", metric_values.get("accuracy"))
            snapshot.setdefault("macro_f1", metric_values.get("macro_f1"))
            snapshot.setdefault("precision", metric_values.get("precision"))
            snapshot.setdefault("recall", metric_values.get("recall"))
        snapshots.append(snapshot)

    return snapshots


def build_selected_metrics(inferred_task: str, latest_evaluation: Optional[dict[str, Any]]) -> list[dict[str, str]]:
    if not latest_evaluation:
        return []

    metric_values = extract_metric_values(latest_evaluation)
    selected = []
    for metric_name in METRIC_PREFERENCES.get(inferred_task, []):
        if metric_name in metric_values:
            selected.append(
                {
                    "name": metric_name,
                    "display_name": METRIC_DISPLAY_NAMES.get(metric_name, metric_name),
                    "justification": METRIC_JUSTIFICATIONS.get(metric_name, "Metric available from logged artifacts."),
                }
            )
    return selected


def build_performance_summary(
    *, inferred_task: str, latest_evaluation: Optional[dict[str, Any]], selected_metrics: list[dict[str, str]]
) -> dict[str, Any]:
    if latest_evaluation is None:
        return {
            "latest_step": None,
            "latest_metrics": {},
            "headline_metrics": [],
            "summary": "Insufficient data to build a performance summary.",
        }

    metric_values = extract_metric_values(latest_evaluation)
    headline_metrics = []
    for metric in selected_metrics[:4]:
        name = metric["name"]
        if name not in metric_values:
            continue
        value = metric_values[name]
        headline_metrics.append(
            {
                "name": name,
                "label": METRIC_DISPLAY_NAMES.get(name, name),
                "value": value,
                "display": format_metric(name, value),
            }
        )

    summary = "; ".join(f"{item['label']}: {item['display']}" for item in headline_metrics)
    if not summary:
        summary = "Performance metrics were logged, but none match the preferred summary metrics for this task."

    return {
        "latest_step": latest_evaluation.get("step"),
        "latest_metrics": metric_values,
        "headline_metrics": headline_metrics,
        "summary": summary,
    }


def build_trend_analysis(inferred_task: str, evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    if len(evaluations) < 2:
        return {
            "status": "insufficient_history",
            "summary": "Trend analysis needs at least two evaluation snapshots.",
            "primary_metric": None,
            "series": [],
            "series_keys": [],
            "significance": {
                "status": "insufficient_history",
                "message": "Formal change significance is not assessed without repeated snapshots.",
            },
        }

    available_keys = {key for item in evaluations for key in extract_metric_values(item).keys()}
    preferred_keys = [key for key in METRIC_PREFERENCES.get(inferred_task, []) if key in available_keys]
    series_keys = preferred_keys[:3]
    if not series_keys:
        series_keys = sorted(available_keys)[:3]

    primary_metric = series_keys[0] if series_keys else None
    series = []
    primary_values = []
    for item in evaluations:
        row = {"step": item.get("step")}
        metric_values = extract_metric_values(item)
        for key in series_keys:
            if key in metric_values:
                row[key] = metric_values[key]
        if primary_metric and primary_metric in metric_values:
            primary_values.append(metric_values[primary_metric])
        series.append(row)

    if primary_metric and len(primary_values) >= 2:
        higher_is_better = primary_metric in HIGHER_IS_BETTER
        start_value = primary_values[0]
        latest_value = primary_values[-1]
        delta = latest_value - start_value
        effective_delta = delta if higher_is_better else -delta
        tolerance = max(abs(start_value) * 0.02, np.std(primary_values) * 0.15, 1e-6)
        if effective_delta > tolerance:
            status = "improving"
        elif effective_delta < -tolerance:
            status = "degrading"
        else:
            status = "stable"
        significance = assess_change_significance(values=primary_values)
        summary = (
            f"{METRIC_DISPLAY_NAMES.get(primary_metric, primary_metric)} moved from "
            f"{format_metric(primary_metric, start_value)} to {format_metric(primary_metric, latest_value)}."
        )
    else:
        status = "insufficient_history"
        significance = {
            "status": "insufficient_history",
            "message": "Formal change significance is not assessed without repeated snapshots.",
        }
        summary = "Trend analysis could not identify a primary metric series."

    return {
        "status": status,
        "summary": summary,
        "primary_metric": primary_metric,
        "series": series,
        "series_keys": series_keys,
        "significance": significance,
    }


def build_error_analysis(
    *,
    inferred_task: str,
    latest_evaluation: Optional[dict[str, Any]],
    metrics: list[dict[str, Any]],
    missing_artifacts: list[str],
) -> dict[str, Any]:
    summary: list[str] = []
    weak_classes = []
    dominant_misclassifications = []
    overconfident_errors: dict[str, Any] = {
        "status": "insufficient_data",
        "count": 0,
        "rate": None,
        "message": "Confidence-calibrated error analysis requires labels, predictions, and confidences.",
    }
    data_imbalance: dict[str, Any] = {
        "status": "insufficient_data",
        "message": "Per-class support data is unavailable.",
    }

    if latest_evaluation:
        weak_classes = latest_evaluation.get("weak_classes", [])
        dominant_misclassifications = latest_evaluation.get("dominant_misclassifications", [])
        if weak_classes:
            summary.append(f"{len(weak_classes)} weak class/label segments need attention.")
        if dominant_misclassifications:
            top = dominant_misclassifications[0]
            summary.append(
                f"Most common error pattern: {top['actual']} predicted as {top['predicted']} ({top['count']} samples)."
            )
        if latest_evaluation.get("overconfident_errors"):
            overconfident_errors = latest_evaluation["overconfident_errors"]
            if overconfident_errors.get("count", 0) > 0:
                summary.append(
                    f"{overconfident_errors['count']} high-confidence mistakes were detected in the latest snapshot."
                )
        if latest_evaluation.get("data_imbalance"):
            data_imbalance = latest_evaluation["data_imbalance"]
            if data_imbalance.get("status") in {"moderate", "severe"}:
                summary.append(data_imbalance.get("message", "Class imbalance detected."))
        if inferred_task in {"regression", "time_series_forecasting"} and latest_evaluation.get("residual_summary"):
            residual = latest_evaluation["residual_summary"]
            summary.append(
                f"Residual bias is {residual['bias']:+.4f} with a 95th percentile absolute error of "
                f"{residual['p95_abs_error']:.4f}."
            )

    generalization = assess_generalization(metrics=metrics, latest_evaluation=latest_evaluation, inferred_task=inferred_task)
    if generalization.get("status") not in {"stable", "insufficient_data"}:
        summary.append(generalization.get("message", "Generalization warning detected."))
    elif not summary and missing_artifacts:
        summary.append("Evaluation is partially available, but richer diagnostics need more logged artifacts.")

    return {
        "summary": summary,
        "weak_classes": weak_classes,
        "dominant_misclassifications": dominant_misclassifications,
        "overconfident_errors": overconfident_errors,
        "data_imbalance": data_imbalance,
        "generalization": generalization,
    }


def build_modality_analysis(modalities: list[str], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    per_modality = []
    for modality in modalities:
        label = modality.replace("_", " ")
        per_modality.append(
            {
                "modality": modality,
                "status": "evaluated",
                "message": f"Observable artifacts suggest a {label} component.",
            }
        )

    if len(modalities) <= 1:
        cross_modal_alignment = {
            "status": "not_applicable",
            "message": "Cross-modal alignment requires at least two paired modalities.",
        }
    else:
        paired_alignment = any("alignment" in str(key).lower() for record in predictions for key in record.keys())
        cross_modal_alignment = {
            "status": "available" if paired_alignment else "not_available",
            "message": (
                "Paired alignment signals were logged."
                if paired_alignment
                else "Multiple modalities were detected, but no paired alignment artifacts were logged."
            ),
        }

    return {
        "modalities": modalities,
        "per_modality": per_modality,
        "cross_modal_alignment": cross_modal_alignment,
    }


def collect_missing_artifacts(
    *, inferred_task: str, predictions: list[dict[str, Any]], latest_evaluation: Optional[dict[str, Any]]
) -> list[str]:
    missing: list[str] = []
    if not predictions:
        missing.append("Per-example predictions and targets were not logged; deeper evaluation is based on run metrics only.")

    if inferred_task == "classification":
        if latest_evaluation and not latest_evaluation.get("confusion_matrix"):
            missing.append("Confusion-matrix detail requires class labels for each prediction.")
        if not any("confidence" in record for record in predictions):
            missing.append("Prediction confidences are missing, so overconfident-error analysis is limited.")
    elif inferred_task in {"regression", "time_series_forecasting"}:
        if latest_evaluation and not latest_evaluation.get("residual_summary"):
            missing.append("Aligned numeric targets and predictions are needed for residual diagnostics.")
        if inferred_task == "time_series_forecasting" and latest_evaluation and not latest_evaluation.get("per_horizon"):
            missing.append("Multi-horizon targets/predictions are needed for horizon-wise forecast diagnostics.")
    elif inferred_task == "sequence_generation":
        if not any(any(key in record for key in ("reference", "references", "y_true")) for record in predictions):
            missing.append("Reference text is missing, so overlap metrics like BLEU and ROUGE cannot be computed.")
        if not any(any(key in record for key in ("output_embeddings", "reference_embeddings")) for record in predictions):
            missing.append("Embeddings are missing, so semantic-similarity checks are limited.")
    elif inferred_task == "retrieval_embedding":
        if not any(any(key in record for key in ("relevant_ids", "relevance", "labels")) for record in predictions):
            missing.append("Relevance labels are missing, so Recall@K is limited.")
        if not any(any(key in record for key in ("query_embeddings", "reference_embeddings", "embeddings")) for record in predictions):
            missing.append("Embedding vectors are missing, so cosine-similarity analysis is limited.")
    elif inferred_task == "vision":
        if latest_evaluation and not any(key in latest_evaluation for key in ("mean_iou", "mAP_50", "top_1_accuracy")):
            missing.append("Vision-specific labels, masks, or boxes are needed for vision metrics.")
    elif inferred_task == "reinforcement_learning":
        if not any(any(key in record for key in ("returns", "rewards", "success")) for record in predictions):
            missing.append("Episode returns or reward traces are needed for reinforcement-learning evaluation.")

    deduped: list[str] = []
    seen = set()
    for item in missing:
        if item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped


def build_recommendations(
    *,
    inferred_task: str,
    latest_evaluation: Optional[dict[str, Any]],
    missing_artifacts: list[str],
    error_analysis: dict[str, Any],
    ambiguity: bool,
) -> list[str]:
    recommendations: list[str] = []

    if ambiguity:
        recommendations.append("Set an explicit `task` in run metadata so task inference does not need to rely on heuristics.")
    if missing_artifacts:
        recommendations.append("Log richer per-example artifacts with `run.log_batch()` to unlock deeper diagnostics and confusion/error views.")

    weak_classes = error_analysis.get("weak_classes") or []
    if weak_classes:
        recommendations.append("Prioritize the weakest classes/labels with targeted sampling, augmentation, or threshold tuning.")

    overconfident = error_analysis.get("overconfident_errors") or {}
    if overconfident.get("count", 0) > 0:
        recommendations.append("Calibrate prediction confidence or add hard-negative examples to reduce overconfident mistakes.")

    generalization = error_analysis.get("generalization") or {}
    if generalization.get("status") == "possible_overfitting":
        recommendations.append("Add regularization, early stopping, or more validation coverage to reduce overfitting pressure.")
    elif generalization.get("status") == "possible_underfitting":
        recommendations.append("Increase model capacity, training time, or feature richness to address likely underfitting.")

    if inferred_task == "time_series_forecasting":
        if latest_evaluation and not latest_evaluation.get("per_horizon"):
            recommendations.append("Log multi-horizon predictions so forecast quality can be inspected horizon by horizon.")
    elif inferred_task == "sequence_generation":
        recommendations.append("Attach reference outputs or embeddings when possible so text quality can be scored beyond qualitative review.")
    elif inferred_task == "retrieval_embedding":
        recommendations.append("Log ranked candidates plus relevance labels so Recall@K can be tracked over time.")

    trimmed: list[str] = []
    seen = set()
    for item in recommendations:
        if item not in seen:
            trimmed.append(item)
            seen.add(item)
        if len(trimmed) == 5:
            break
    return trimmed


def detect_modalities(
    *, predictions: list[dict[str, Any]], metadata: dict[str, Any], inferred_task: str
) -> list[str]:
    detected: list[str] = []
    task_hint = str(((metadata.get("config") or {}).get("task") or metadata.get("task") or "")).lower()

    for record in predictions:
        hint = str(record.get("input_modality_hint") or "").strip().lower()
        if hint:
            detected.append(hint)
        input_shape = _shape_from_record(record.get("input_shape"))
        if input_shape:
            if len(input_shape) >= 4:
                detected.append("vision")
            elif len(input_shape) == 3 and inferred_task == "time_series_forecasting":
                detected.append("time_series")
            elif len(input_shape) == 2:
                detected.append("structured_data")
        if any(key in record for key in ("prompt", "prompts", "generated_text", "reference", "references")):
            detected.append("text")
        if any(key in record for key in ("pred_masks", "masks", "images", "pred_boxes", "boxes")):
            detected.append("vision")
        if any(key in record for key in ("audio", "waveform", "spectrogram")):
            detected.append("audio")

    if not detected:
        if inferred_task == "vision" or any(token in task_hint for token in ("vision", "image", "detect", "segment")):
            detected.append("vision")
        elif inferred_task == "sequence_generation":
            detected.append("text")
        elif inferred_task == "time_series_forecasting":
            detected.append("time_series")
        else:
            detected.append("structured_data")

    unique: list[str] = []
    for item in detected:
        if item not in unique:
            unique.append(item)
    return unique


def assess_change_significance(values: list[float]) -> dict[str, str]:
    if len(values) < 5:
        return {
            "status": "insufficient_history",
            "message": "Formal change significance is not assessed without at least five snapshots.",
        }

    values_arr = np.asarray(values, dtype=float)
    window = max(2, len(values_arr) // 3)
    early = values_arr[:window]
    late = values_arr[-window:]
    var_early = float(np.var(early, ddof=1)) if len(early) > 1 else 0.0
    var_late = float(np.var(late, ddof=1)) if len(late) > 1 else 0.0
    denom = math.sqrt(var_early / max(len(early), 1) + var_late / max(len(late), 1) + 1e-12)
    t_like = abs(float(np.mean(late) - np.mean(early))) / max(denom, 1e-12)

    if t_like >= 2.0:
        return {
            "status": "likely_significant",
            "message": "Early and late windows are separated enough to suggest a statistically meaningful change.",
        }
    return {
        "status": "not_significant",
        "message": "Observed changes are directional, but they do not clearly exceed run-to-run variability.",
    }


def assess_generalization(
    *, metrics: list[dict[str, Any]], latest_evaluation: Optional[dict[str, Any]], inferred_task: str
) -> dict[str, Any]:
    train_candidates = ["loss", "train_loss", "training_loss"]
    val_candidates = ["val_loss", "validation_loss", "test_loss"]
    train_series = _extract_metric_series(metrics, train_candidates)
    val_series = _extract_metric_series(metrics, val_candidates)

    if len(train_series) < 3 or len(val_series) < 3:
        return {
            "status": "insufficient_data",
            "message": "Train/validation loss history is too sparse for overfitting or underfitting checks.",
        }

    train_values = np.asarray([value for _, value in train_series], dtype=float)
    val_values = np.asarray([value for _, value in val_series], dtype=float)
    train_improved = train_values[0] - train_values[-1]
    best_val = float(np.min(val_values))
    val_rebound = val_values[-1] - best_val
    gap = val_values[-1] - train_values[-1]

    if train_improved > max(train_values[0] * 0.05, 1e-6) and val_rebound > max(best_val * 0.05, 1e-6) and gap > 0:
        return {
            "status": "possible_overfitting",
            "message": "Training loss keeps improving while validation loss has rebounded from its best point.",
            "train_loss_delta": float(train_improved),
            "validation_rebound": float(val_rebound),
        }

    primary_metrics = extract_metric_values(latest_evaluation or {})
    weak_primary = False
    if inferred_task in {"classification", "vision"}:
        value = primary_metrics.get("macro_f1", primary_metrics.get("accuracy", primary_metrics.get("top_1_accuracy")))
        weak_primary = value is not None and value < 0.6
    elif inferred_task in {"regression", "time_series_forecasting"}:
        value = primary_metrics.get("r2")
        weak_primary = value is not None and value < 0.2

    train_plateau = abs(train_values[-1] - np.mean(train_values[-3:])) <= max(np.std(train_values[-3:]), 1e-6)
    val_plateau = abs(val_values[-1] - np.mean(val_values[-3:])) <= max(np.std(val_values[-3:]), 1e-6)
    if weak_primary and abs(gap) <= max(np.std(val_values[-3:]) * 0.5, 0.02) and train_plateau and val_plateau:
        return {
            "status": "possible_underfitting",
            "message": "Training and validation losses are closely matched and plateaued while task performance remains weak.",
        }

    return {
        "status": "stable",
        "message": "No strong overfitting or underfitting signal was detected from the available loss history.",
    }


def extract_metric_values(payload: dict[str, Any]) -> dict[str, float]:
    metric_values = payload.get("metric_values")
    if isinstance(metric_values, dict) and metric_values:
        return {key: float(value) for key, value in metric_values.items() if _is_number(value)}

    extracted = {}
    for key in (
        "accuracy",
        "precision",
        "recall",
        "macro_f1",
        "micro_f1",
        "mse",
        "rmse",
        "mae",
        "r2",
        "mape",
        "bleu",
        "rouge_l",
        "semantic_similarity",
        "cosine_similarity",
        "recall_at_1",
        "recall_at_5",
        "top_1_accuracy",
        "top_5_accuracy",
        "mean_iou",
        "mAP_50",
        "mean_return",
        "success_rate",
    ):
        if _is_number(payload.get(key)):
            extracted[key] = float(payload[key])
    return extracted


def format_metric(name: str, value: float) -> str:
    if name in PERCENT_METRICS:
        return f"{value * 100:.2f}%" if value <= 1.0 else f"{value:.2f}%"
    if abs(value) >= 1000:
        return f"{value:,.2f}"
    return f"{value:.4f}"


def _score_prediction_record(record: dict[str, Any]) -> tuple[dict[str, float], list[str]]:
    scores = {task: 0.0 for task in TASK_TYPES}
    evidence: list[str] = []

    y_true = _to_array(record.get("y_true"))
    y_pred = _to_array(record.get("y_pred"))
    prediction_type = str(record.get("prediction_type") or "").lower()
    input_shape = _shape_from_record(record.get("input_shape"))

    if any(key in record for key in ("prompt", "prompts", "generated_text", "reference", "references")):
        scores["sequence_generation"] += 5.0
        evidence.append("Text prompts or generation/reference fields were logged.")

    if any(key in record for key in ("query_embeddings", "reference_embeddings", "ranked_ids", "relevant_ids")):
        scores["retrieval_embedding"] += 5.0
        evidence.append("Retrieval or embedding fields were logged.")

    if any(key in record for key in ("pred_masks", "masks", "pred_boxes", "boxes", "images")):
        scores["vision"] += 5.0
        evidence.append("Vision-specific artifacts such as masks or boxes were logged.")

    if any(key in record for key in ("rewards", "returns", "episode_returns", "success")):
        scores["reinforcement_learning"] += 5.0
        evidence.append("Reward or return traces were logged.")

    if prediction_type == "class_scores":
        scores["classification"] += 5.0
        evidence.append("Predictions were logged as class scores.")
    elif prediction_type == "numeric_array":
        scores["regression"] += 1.5

    if input_shape:
        if len(input_shape) >= 4:
            scores["vision"] += 2.0
            evidence.append(f"Input shape {input_shape} suggests image-like data.")
        elif len(input_shape) == 3:
            scores["time_series_forecasting"] += 1.5
            evidence.append(f"Input shape {input_shape} suggests sequential or temporal data.")
        elif len(input_shape) == 2:
            scores["classification"] += 0.5
            scores["regression"] += 0.5

    if y_true is not None and y_pred is not None:
        if _is_multilabel_pair(y_true, y_pred):
            scores["classification"] += 4.0
            evidence.append("Targets and predictions look like multilabel binary matrices.")
        elif _looks_like_classification_labels(y_true, y_pred):
            scores["classification"] += 4.0
            evidence.append("Targets and predictions look like discrete labels.")
        elif _looks_like_time_series(y_true, y_pred, input_shape):
            scores["time_series_forecasting"] += 4.0
            evidence.append("Numeric prediction shapes and temporal cues suggest forecasting.")
        elif _is_numeric_pair(y_true, y_pred):
            scores["regression"] += 4.0
            evidence.append("Aligned numeric targets and predictions suggest regression-style evaluation.")

    return scores, evidence


def _evaluate_classification(y_true: np.ndarray, y_pred: np.ndarray, confidence: Optional[np.ndarray]) -> dict[str, Any]:
    true_labels = _flatten_labels(y_true)
    pred_labels = _flatten_labels(y_pred)
    n = min(len(true_labels), len(pred_labels))
    if n == 0:
        raise ValueError("Classification payload is empty")
    true_labels = true_labels[:n]
    pred_labels = pred_labels[:n]

    classes = _ordered_unique(true_labels + pred_labels)
    index = {cls: idx for idx, cls in enumerate(classes)}
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    for true_label, pred_label in zip(true_labels, pred_labels):
        cm[index[true_label], index[pred_label]] += 1

    support = int(cm.sum())
    accuracy = float(np.trace(cm) / support) if support else 0.0
    per_class = []
    weak_classes = []
    dominant_misclassifications = []
    macro_precision = 0.0
    macro_recall = 0.0
    macro_f1 = 0.0

    for i, cls in enumerate(classes):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        class_support = int(cm[i, :].sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        macro_precision += precision
        macro_recall += recall
        macro_f1 += f1
        payload = {
            "class": _to_jsonable_scalar(cls),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": class_support,
        }
        per_class.append(payload)
        if recall < 0.6 or precision < 0.6:
            weak_classes.append(payload)

    for i, actual in enumerate(classes):
        for j, predicted in enumerate(classes):
            if i == j or cm[i, j] == 0:
                continue
            dominant_misclassifications.append(
                {
                    "actual": _to_jsonable_scalar(actual),
                    "predicted": _to_jsonable_scalar(predicted),
                    "count": int(cm[i, j]),
                }
            )
    dominant_misclassifications.sort(key=lambda item: item["count"], reverse=True)

    num_classes = max(len(classes), 1)
    supports = [item["support"] for item in per_class if item["support"] > 0]
    imbalance_status = "none"
    imbalance_message = "No strong class imbalance signal was detected."
    if supports:
        max_support = max(supports)
        min_support = min(supports)
        ratio = max_support / max(min_support, 1)
        if ratio >= 5:
            imbalance_status = "severe"
            imbalance_message = f"Class support ratio is {ratio:.1f}x between the largest and smallest classes."
        elif ratio >= 2.5:
            imbalance_status = "moderate"
            imbalance_message = f"Class support ratio is {ratio:.1f}x between the largest and smallest classes."
    else:
        ratio = None

    overconfident_errors = {
        "status": "insufficient_data",
        "count": 0,
        "rate": None,
        "mean_confidence": None,
        "message": "Confidence values were not logged for this snapshot.",
    }
    if confidence is not None:
        conf = confidence.flatten()[:n]
        mistakes = [(float(c), t, p) for c, t, p in zip(conf, true_labels, pred_labels) if t != p and c >= 0.8]
        if mistakes:
            mean_conf = float(np.mean([item[0] for item in mistakes]))
            overconfident_errors = {
                "status": "detected",
                "count": len(mistakes),
                "rate": float(len(mistakes) / n),
                "mean_confidence": mean_conf,
                "message": f"{len(mistakes)} mistakes were made with confidence >= 0.80.",
            }
        else:
            overconfident_errors = {
                "status": "clear",
                "count": 0,
                "rate": 0.0,
                "mean_confidence": None,
                "message": "No high-confidence mistakes were detected in this snapshot.",
            }

    return {
        "is_classification": True,
        "classification_type": "single_label",
        "accuracy": accuracy,
        "precision": float(macro_precision / num_classes),
        "recall": float(macro_recall / num_classes),
        "macro_precision": float(macro_precision / num_classes),
        "macro_recall": float(macro_recall / num_classes),
        "macro_f1": float(macro_f1 / num_classes),
        "micro_f1": accuracy,
        "support": support,
        "confusion_matrix": {
            "classes": [_to_jsonable_scalar(cls) for cls in classes],
            "matrix": cm.tolist(),
        },
        "per_class": per_class,
        "weak_classes": weak_classes,
        "dominant_misclassifications": dominant_misclassifications[:5],
        "overconfident_errors": overconfident_errors,
        "data_imbalance": {
            "status": imbalance_status,
            "support_ratio": None if ratio is None else float(ratio),
            "message": imbalance_message,
        },
    }


def _evaluate_multilabel_classification(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    y_true_bin = np.asarray(y_true, dtype=int)
    y_pred_bin = np.asarray(y_pred, dtype=int)
    y_true_bin = (y_true_bin > 0).astype(int)
    y_pred_bin = (y_pred_bin > 0).astype(int)

    tp = (y_true_bin * y_pred_bin).sum(axis=0)
    fp = ((1 - y_true_bin) * y_pred_bin).sum(axis=0)
    fn = (y_true_bin * (1 - y_pred_bin)).sum(axis=0)
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp, dtype=float), where=(precision + recall) > 0)

    support = y_true_bin.sum(axis=0)
    per_class = []
    weak_classes = []
    for idx in range(y_true_bin.shape[1]):
        payload = {
            "class": idx,
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
            "support": int(support[idx]),
        }
        per_class.append(payload)
        if payload["precision"] < 0.6 or payload["recall"] < 0.6:
            weak_classes.append(payload)

    subset_accuracy = float(np.mean(np.all(y_true_bin == y_pred_bin, axis=1)))
    micro_tp = float(tp.sum())
    micro_fp = float(fp.sum())
    micro_fn = float(fn.sum())
    micro_precision = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) else 0.0
    micro_recall = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall)
        else 0.0
    )

    return {
        "is_classification": True,
        "classification_type": "multilabel",
        "accuracy": subset_accuracy,
        "precision": float(np.mean(precision)),
        "recall": float(np.mean(recall)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "micro_f1": float(micro_f1),
        "support": int(y_true_bin.shape[0]),
        "per_class": per_class,
        "weak_classes": weak_classes,
        "dominant_misclassifications": [],
        "overconfident_errors": {
            "status": "insufficient_data",
            "count": 0,
            "rate": None,
            "message": "Confidence-calibrated multilabel error analysis was not logged.",
        },
        "data_imbalance": {
            "status": "moderate" if (support.max() / max(support[support > 0].min(), 1)) >= 2.5 else "none",
            "support_ratio": float(support.max() / max(support[support > 0].min(), 1)) if np.any(support > 0) else None,
            "message": "Multilabel support distribution computed from positive labels.",
        },
    }


def _evaluate_regression_like(y_true: np.ndarray, y_pred: np.ndarray, *, time_series: bool) -> dict[str, Any]:
    true_arr = np.asarray(y_true, dtype=float)
    pred_arr = np.asarray(y_pred, dtype=float)
    if true_arr.size == 0 or pred_arr.size == 0:
        raise ValueError("Regression payload is empty")

    if true_arr.shape != pred_arr.shape:
        flat_n = min(true_arr.size, pred_arr.size)
        true_flat = true_arr.reshape(-1)[:flat_n]
        pred_flat = pred_arr.reshape(-1)[:flat_n]
    else:
        true_flat = true_arr.reshape(-1)
        pred_flat = pred_arr.reshape(-1)

    error = pred_flat - true_flat
    mse = float(np.mean(error ** 2))
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(mse))
    variance = float(np.var(true_flat))
    r2 = float(1.0 - np.sum(error ** 2) / max(np.sum((true_flat - np.mean(true_flat)) ** 2), 1e-12)) if variance > 0 else 0.0

    metric_values = {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }
    non_zero_mask = np.abs(true_flat) > 1e-8
    if np.any(non_zero_mask):
        mape = float(np.mean(np.abs(error[non_zero_mask] / true_flat[non_zero_mask])))
        metric_values["mape"] = mape

    payload = {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "metric_values": metric_values,
        "residual_summary": {
            "bias": float(np.mean(error)),
            "std": float(np.std(error)),
            "p95_abs_error": float(np.percentile(np.abs(error), 95)),
        },
    }
    if "mape" in metric_values:
        payload["mape"] = metric_values["mape"]

    if time_series and true_arr.ndim >= 2 and pred_arr.ndim >= 2 and true_arr.shape == pred_arr.shape:
        per_horizon = []
        horizon = true_arr.shape[-1]
        for idx in range(horizon):
            horizon_true = true_arr[..., idx].reshape(-1)
            horizon_pred = pred_arr[..., idx].reshape(-1)
            horizon_err = horizon_pred - horizon_true
            horizon_mse = float(np.mean(horizon_err ** 2))
            per_horizon.append(
                {
                    "horizon": idx,
                    "rmse": float(np.sqrt(horizon_mse)),
                    "mae": float(np.mean(np.abs(horizon_err))),
                }
            )
        payload["per_horizon"] = per_horizon

    return payload


def _evaluate_generation(record: dict[str, Any]) -> Optional[dict[str, Any]]:
    outputs = _extract_text_list(record, ["generated_text", "generated", "outputs", "predictions", "y_pred"])
    references = _extract_text_list(record, ["references", "reference", "targets", "y_true"])
    if not outputs or not references:
        return None

    n = min(len(outputs), len(references))
    outputs = outputs[:n]
    references = references[:n]

    bleu_scores = [_sentence_bleu(output, reference) for output, reference in zip(outputs, references)]
    rouge_scores = [_rouge_l_f1(output, reference) for output, reference in zip(outputs, references)]
    metric_values = {
        "bleu": float(np.mean(bleu_scores)),
        "rouge_l": float(np.mean(rouge_scores)),
    }

    semantic_similarity = _paired_cosine_similarity(record.get("output_embeddings"), record.get("reference_embeddings"))
    if semantic_similarity is not None:
        metric_values["semantic_similarity"] = semantic_similarity

    return {
        "metric_values": metric_values,
        **metric_values,
        "quality_notes": {
            "avg_output_length": float(np.mean([len(item.split()) for item in outputs])),
            "avg_reference_length": float(np.mean([len(item.split()) for item in references])),
        },
    }


def _evaluate_retrieval(record: dict[str, Any]) -> Optional[dict[str, Any]]:
    metric_values: dict[str, float] = {}
    cosine = _paired_cosine_similarity(record.get("query_embeddings"), record.get("reference_embeddings"))
    if cosine is not None:
        metric_values["cosine_similarity"] = cosine

    ranked_ids = record.get("ranked_ids") or record.get("retrieved_ids")
    relevant_ids = record.get("relevant_ids") or record.get("labels")
    if ranked_ids is not None and relevant_ids is not None:
        recall_1, recall_5 = _compute_recall_at_k(ranked_ids, relevant_ids, ks=(1, 5))
        metric_values["recall_at_1"] = recall_1
        metric_values["recall_at_5"] = recall_5

    if not metric_values:
        return None
    return {"metric_values": metric_values, **metric_values}


def _evaluate_vision_record(
    record: dict[str, Any], *, y_true: Optional[np.ndarray], y_pred: Optional[np.ndarray]
) -> Optional[dict[str, Any]]:
    pred_masks = _to_array(record.get("pred_masks") or record.get("pred_mask"))
    true_masks = _to_array(record.get("masks") or record.get("true_masks") or record.get("y_true_masks"))
    if pred_masks is not None and true_masks is not None and pred_masks.shape == true_masks.shape:
        iou = _mean_iou(true_masks, pred_masks)
        return {"metric_values": {"mean_iou": iou}, "mean_iou": iou}

    pred_boxes = record.get("pred_boxes")
    true_boxes = record.get("boxes") or record.get("true_boxes")
    if pred_boxes is not None and true_boxes is not None:
        map_50 = _map_at_50(pred_boxes=pred_boxes, true_boxes=true_boxes)
        return {"metric_values": {"mAP_50": map_50}, "mAP_50": map_50}

    if y_true is not None and y_pred is not None:
        if _looks_like_classification_labels(y_true, y_pred):
            payload = _evaluate_classification(y_true, y_pred, confidence=_to_float_array(record.get("confidence")))
            payload["top_1_accuracy"] = payload["accuracy"]
            payload.setdefault("metric_values", {})["top_1_accuracy"] = payload["accuracy"]
            return payload
        if _is_numeric_pair(y_true, y_pred):
            return _evaluate_regression_like(y_true, y_pred, time_series=False)

    return None


def _evaluate_reinforcement_learning(record: dict[str, Any]) -> Optional[dict[str, Any]]:
    returns = _to_float_array(record.get("returns") or record.get("episode_returns") or record.get("rewards"))
    if returns is None or returns.size == 0:
        return None
    metric_values = {"mean_return": float(np.mean(returns))}
    success = _to_float_array(record.get("success") or record.get("success_rate"))
    if success is not None and success.size > 0:
        metric_values["success_rate"] = float(np.mean(success))
    return {
        "metric_values": metric_values,
        **metric_values,
        "return_summary": {
            "best_return": float(np.max(returns)),
            "std_return": float(np.std(returns)),
        },
    }


def _compute_recall_at_k(ranked_ids: Any, relevant_ids: Any, ks: tuple[int, int]) -> tuple[float, float]:
    ranked = ranked_ids if isinstance(ranked_ids, list) else []
    relevant = relevant_ids if isinstance(relevant_ids, list) else []
    if ranked and not isinstance(ranked[0], list):
        ranked = [ranked]
    if relevant and not isinstance(relevant[0], list):
        relevant = [relevant]
    n = min(len(ranked), len(relevant))
    if n == 0:
        return 0.0, 0.0

    values = []
    for k in ks:
        hits = 0
        for rank_row, rel_row in zip(ranked[:n], relevant[:n]):
            rel_set = set(rel_row if isinstance(rel_row, list) else [rel_row])
            top_k = rank_row[:k] if isinstance(rank_row, list) else [rank_row]
            if any(item in rel_set for item in top_k):
                hits += 1
        values.append(float(hits / n))
    return values[0], values[1]


def _paired_cosine_similarity(left: Any, right: Any) -> Optional[float]:
    left_arr = _to_float_array(left)
    right_arr = _to_float_array(right)
    if left_arr is None or right_arr is None:
        return None
    if left_arr.shape != right_arr.shape:
        return None
    if left_arr.ndim == 1:
        left_arr = left_arr[None, :]
        right_arr = right_arr[None, :]
    numerators = np.sum(left_arr * right_arr, axis=1)
    denominators = np.linalg.norm(left_arr, axis=1) * np.linalg.norm(right_arr, axis=1)
    similarities = np.divide(numerators, denominators, out=np.zeros_like(numerators, dtype=float), where=denominators > 0)
    return float(np.mean(similarities))


def _sentence_bleu(output: str, reference: str) -> float:
    output_tokens = output.split()
    reference_tokens = reference.split()
    if not output_tokens or not reference_tokens:
        return 0.0

    unigram_matches = sum(1 for token in output_tokens if token in reference_tokens)
    unigram_precision = unigram_matches / len(output_tokens)

    output_bigrams = list(zip(output_tokens, output_tokens[1:]))
    reference_bigrams = set(zip(reference_tokens, reference_tokens[1:]))
    if output_bigrams:
        bigram_matches = sum(1 for token in output_bigrams if token in reference_bigrams)
        bigram_precision = bigram_matches / len(output_bigrams)
    else:
        bigram_precision = unigram_precision

    brevity_penalty = min(1.0, math.exp(1 - len(reference_tokens) / max(len(output_tokens), 1)))
    return float(brevity_penalty * math.sqrt(max(unigram_precision, 0.0) * max(bigram_precision, 0.0)))


def _rouge_l_f1(output: str, reference: str) -> float:
    output_tokens = output.split()
    reference_tokens = reference.split()
    if not output_tokens or not reference_tokens:
        return 0.0

    dp = [[0] * (len(reference_tokens) + 1) for _ in range(len(output_tokens) + 1)]
    for i, output_token in enumerate(output_tokens, start=1):
        for j, reference_token in enumerate(reference_tokens, start=1):
            if output_token == reference_token:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[-1][-1]
    precision = lcs / len(output_tokens)
    recall = lcs / len(reference_tokens)
    return float((2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0)


def _mean_iou(true_masks: np.ndarray, pred_masks: np.ndarray) -> float:
    true_bin = np.asarray(true_masks) > 0
    pred_bin = np.asarray(pred_masks) > 0
    intersection = np.logical_and(true_bin, pred_bin).sum(axis=tuple(range(1, true_bin.ndim)))
    union = np.logical_or(true_bin, pred_bin).sum(axis=tuple(range(1, true_bin.ndim)))
    scores = np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float), where=union > 0)
    return float(np.mean(scores))


def _map_at_50(*, pred_boxes: Any, true_boxes: Any) -> float:
    predictions = pred_boxes if isinstance(pred_boxes, list) else []
    targets = true_boxes if isinstance(true_boxes, list) else []
    if predictions and isinstance(predictions[0], dict):
        predictions = [predictions]
    if targets and isinstance(targets[0], dict):
        targets = [targets]
    n = min(len(predictions), len(targets))
    if n == 0:
        return 0.0

    hits = 0
    total = 0
    for pred_row, true_row in zip(predictions[:n], targets[:n]):
        remaining = list(true_row)
        for prediction in sorted(pred_row, key=lambda item: float(item.get("score", 1.0)), reverse=True):
            label = prediction.get("label")
            best_idx = None
            best_iou = 0.0
            for idx, target in enumerate(remaining):
                if target.get("label") != label:
                    continue
                iou = _box_iou(prediction.get("box") or prediction.get("bbox"), target.get("box") or target.get("bbox"))
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            total += 1
            if best_idx is not None and best_iou >= 0.5:
                hits += 1
                remaining.pop(best_idx)
    return float(hits / max(total, 1))


def _box_iou(left: Any, right: Any) -> float:
    if not isinstance(left, (list, tuple)) or not isinstance(right, (list, tuple)) or len(left) != 4 or len(right) != 4:
        return 0.0
    x1 = max(float(left[0]), float(right[0]))
    y1 = max(float(left[1]), float(right[1]))
    x2 = min(float(left[2]), float(right[2]))
    y2 = min(float(left[3]), float(right[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_left = max(0.0, float(left[2]) - float(left[0])) * max(0.0, float(left[3]) - float(left[1]))
    area_right = max(0.0, float(right[2]) - float(right[0])) * max(0.0, float(right[3]) - float(right[1]))
    union = area_left + area_right - inter
    return inter / union if union > 0 else 0.0


def _extract_metric_series(metrics: list[dict[str, Any]], keys: list[str]) -> list[tuple[int, float]]:
    series = []
    for row in metrics:
        value = _extract_first_numeric(row, keys)
        step = _safe_int(row.get("step"))
        if value is not None and step is not None:
            series.append((step, value))
    return series


def _extract_first_numeric(row: dict[str, Any], keys: list[str]) -> Optional[float]:
    lowered = {str(key).lower(): value for key, value in row.items()}
    for key in keys:
        value = lowered.get(key.lower())
        if _is_number(value):
            return float(value)
    return None


def _extract_text_list(record: dict[str, Any], keys: list[str]) -> list[str]:
    for key in keys:
        value = record.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            out = []
            for item in value:
                if isinstance(item, str):
                    out.append(item)
            if out:
                return out
        arr = _to_array(value)
        if arr is not None and arr.dtype.kind in {"U", "S", "O"}:
            out = [str(item) for item in arr.reshape(-1).tolist() if item is not None]
            if out:
                return out
    return []


def _to_array(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    try:
        arr = np.asarray(value)
    except Exception:
        return None
    return arr if arr.size > 0 else None


def _to_float_array(value: Any) -> Optional[np.ndarray]:
    arr = _to_array(value)
    if arr is None:
        return None
    try:
        return arr.astype(float)
    except (TypeError, ValueError):
        return None


def _flatten_labels(arr: np.ndarray) -> list[Any]:
    flat = arr.reshape(-1).tolist()
    return [_to_jsonable_scalar(item) for item in flat]


def _shape_from_record(value: Any) -> Optional[tuple[int, ...]]:
    if isinstance(value, (list, tuple)) and all(isinstance(item, (int, float)) for item in value):
        return tuple(int(item) for item in value)
    return None


def _is_numeric_pair(left: np.ndarray, right: np.ndarray) -> bool:
    left_numeric = _to_float_array(left)
    right_numeric = _to_float_array(right)
    return left_numeric is not None and right_numeric is not None


def _is_integer_like(arr: np.ndarray) -> bool:
    float_arr = _to_float_array(arr)
    if float_arr is None:
        return False
    if np.any(~np.isfinite(float_arr)):
        return False
    return bool(np.all(np.isclose(float_arr, np.round(float_arr))))


def _is_binary_matrix(arr: np.ndarray) -> bool:
    float_arr = _to_float_array(arr)
    if float_arr is None or float_arr.ndim < 2:
        return False
    unique = np.unique(float_arr)
    return bool(set(unique.tolist()).issubset({0.0, 1.0}))


def _is_multilabel_pair(y_true: np.ndarray, y_pred: np.ndarray) -> bool:
    return y_true.shape == y_pred.shape and y_true.ndim >= 2 and _is_binary_matrix(y_true) and _is_binary_matrix(y_pred)


def _looks_like_classification_labels(y_true: np.ndarray, y_pred: np.ndarray) -> bool:
    if y_true.size == 0 or y_pred.size == 0:
        return False
    if y_true.shape == y_pred.shape and _is_binary_matrix(y_true) and _is_binary_matrix(y_pred):
        return False
    if _is_integer_like(y_true) and _is_integer_like(y_pred):
        flat_true = _to_float_array(y_true).reshape(-1)
        flat_pred = _to_float_array(y_pred).reshape(-1)
        unique_count = len(np.unique(np.concatenate([flat_true, flat_pred])))
        return unique_count <= max(20, int(math.sqrt(max(len(flat_true), 1)) * 2))
    if y_true.dtype.kind in {"U", "S", "O"} and y_pred.dtype.kind in {"U", "S", "O"}:
        tokens = [str(item) for item in y_true.reshape(-1).tolist()[:32]]
        avg_words = np.mean([len(item.split()) for item in tokens]) if tokens else 0
        unique_count = len(set(tokens + [str(item) for item in y_pred.reshape(-1).tolist()[:32]]))
        return avg_words <= 3 and unique_count <= 50
    return False


def _looks_like_time_series(y_true: np.ndarray, y_pred: np.ndarray, input_shape: Optional[tuple[int, ...]]) -> bool:
    if not _is_numeric_pair(y_true, y_pred):
        return False
    if input_shape and len(input_shape) == 3 and y_true.ndim >= 2 and y_pred.ndim >= 2:
        return True
    return y_true.ndim >= 2 and y_pred.ndim >= 2 and y_true.shape == y_pred.shape and y_true.shape[-1] > 1


def _ordered_unique(items: list[Any]) -> list[Any]:
    unique = []
    seen = set()
    for item in items:
        key = repr(item)
        if key in seen:
            continue
        unique.append(item)
        seen.add(key)
    return unique


def _safe_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _to_jsonable_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value
