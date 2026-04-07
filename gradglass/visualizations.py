from __future__ import annotations

import re
from typing import Any, Optional

import numpy as np


_LAYER_SUFFIX_RE = re.compile(
    r"\.(weight|bias|running_mean|running_var|num_batches_tracked|gamma|beta)$", re.IGNORECASE
)


def normalize_layer_id(layer_id: str) -> str:
    normalized = _LAYER_SUFFIX_RE.sub("", str(layer_id or ""))
    return normalized or str(layer_id or "")


def _array_stats(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    flat = arr.reshape(-1) if arr.size else arr
    if flat.size == 0:
        return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "sparsity": 0.0}
    return {
        "count": int(flat.size),
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "sparsity": float(np.mean(np.abs(flat) < 1e-6)),
    }


def _histogram(values: np.ndarray, bins: int = 40) -> dict[str, Any]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    if flat.size == 0:
        return {"counts": [], "bin_edges": []}
    counts, bin_edges = np.histogram(flat, bins=bins)
    return {"counts": counts.astype(int).tolist(), "bin_edges": bin_edges.astype(float).tolist()}


def _warning_labels(stats: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    scale = max(abs(stats["max"]), abs(stats["min"]), abs(stats["mean"]), 1e-12)
    if stats["sparsity"] > 0.95:
        warnings.append("dead")
    if stats["std"] < max(scale * 0.1, 1e-6) and scale > 1e-3:
        warnings.append("saturated")
    if abs(stats["mean"]) > max(stats["std"], 1e-6) * 0.75:
        warnings.append("skewed")
    return warnings


def _summarize_layer_array(
    layer: str, values: np.ndarray, *, shape: Optional[list[int]] = None, kind: str
) -> dict[str, Any]:
    arr = np.asarray(values)
    stats = _array_stats(arr)
    return {
        "layer": layer,
        "kind": kind,
        "shape": shape or list(arr.shape),
        "stats": stats,
        "histogram": _histogram(arr),
        "warnings": _warning_labels(stats),
    }


def build_distributions_payload(store, run_id: str, step: Optional[int] = None) -> dict[str, Any]:
    checkpoints = store.list_checkpoints(run_id)
    checkpoint_step = checkpoints[-1]["step"] if checkpoints else None
    weight_layers = []

    if checkpoint_step is not None:
        weights = store.load_checkpoint(run_id, checkpoint_step)
        grouped: dict[str, list[np.ndarray]] = {}
        for param_name, values in weights.items():
            layer = normalize_layer_id(param_name)
            grouped.setdefault(layer, []).append(np.asarray(values).reshape(-1))
        for layer, chunks in sorted(grouped.items()):
            combined = np.concatenate(chunks, axis=0) if chunks else np.asarray([])
            entry = _summarize_layer_array(layer, combined, shape=[int(combined.size)], kind="weights")
            entry["checkpoint_step"] = checkpoint_step
            entry["parameter_count"] = int(combined.size)
            weight_layers.append(entry)

    activation_layers = []
    probe_step = None
    probe_reason = None
    try:
        probe_bundle = store.load_probe_bundle(run_id, step=step)
        probe_step = probe_bundle["meta"].get("step")
        arrays = probe_bundle["arrays"]
        for layer_meta in probe_bundle["meta"].get("activation_layers", []):
            key = layer_meta.get("key")
            if not key or key not in arrays:
                continue
            arr = np.asarray(arrays[key])
            entry = _summarize_layer_array(
                layer_meta.get("layer", key), arr, shape=layer_meta.get("shape") or list(arr.shape), kind="activations"
            )
            entry["probe_step"] = probe_step
            entry["sample_count"] = int(arr.shape[0]) if arr.ndim > 0 else 0
            activation_layers.append(entry)
    except FileNotFoundError:
        probe_reason = "No probe activations were captured for this run."

    return {
        "run_id": run_id,
        "default_mode": "activations" if activation_layers else "weights",
        "weights": {
            "available": bool(weight_layers),
            "step": checkpoint_step,
            "reason": None if weight_layers else "No checkpoints were captured for this run.",
            "layers": weight_layers,
        },
        "activations": {
            "available": bool(activation_layers),
            "step": probe_step,
            "reason": None if activation_layers else probe_reason,
            "layers": activation_layers,
        },
    }


def _normalize_vision_tensor(values: np.ndarray) -> list[Any]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim == 3:
        arr = np.mean(arr, axis=-1)
    min_v = float(np.min(arr)) if arr.size else 0.0
    max_v = float(np.max(arr)) if arr.size else 0.0
    if max_v > min_v:
        arr = (arr - min_v) / (max_v - min_v)
    return arr.astype(float).tolist()


def _normalize_vision_saliency(values: np.ndarray) -> list[Any]:
    arr = np.abs(np.asarray(values, dtype=np.float64))
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.mean(arr, axis=0)
    elif arr.ndim == 3 and arr.shape[-1] in (1, 3):
        arr = np.mean(arr, axis=-1)
    max_v = float(np.max(arr)) if arr.size else 0.0
    if max_v > 0:
        arr = arr / max_v
    return arr.astype(float).tolist()


def _to_python_list(values: Optional[np.ndarray]) -> Optional[list[Any]]:
    if values is None:
        return None
    arr = np.asarray(values)
    return arr.tolist()


def build_saliency_payload(store, run_id: str, step: Optional[int] = None) -> dict[str, Any]:
    try:
        probe_bundle = store.load_probe_bundle(run_id, step=step)
    except FileNotFoundError:
        return {
            "run_id": run_id,
            "available": False,
            "reason": "No probe bundle was captured for this run. Re-run with `saliency='auto'`.",
        }

    meta = probe_bundle["meta"]
    saliency_meta = meta.get("saliency") or {}
    arrays = probe_bundle["arrays"]
    saliency = arrays.get("saliency")
    if saliency is None or not saliency_meta.get("available"):
        return {
            "run_id": run_id,
            "step": meta.get("step"),
            "available": False,
            "reason": saliency_meta.get("reason") or "Saliency was not captured for this run.",
        }

    input_arr = np.asarray(arrays.get("input"))
    target_arr = arrays.get("targets")
    prediction_arr = arrays.get("predictions")
    confidence_arr = arrays.get("confidence")
    modality = saliency_meta.get("kind") or meta.get("input_modality")

    if modality == "vision":
        sample_count = min(int(input_arr.shape[0]), int(np.asarray(saliency).shape[0]), 6)
        samples = []
        for index in range(sample_count):
            samples.append(
                {
                    "index": index,
                    "target": None if target_arr is None else np.asarray(target_arr)[index].tolist(),
                    "prediction": None if prediction_arr is None else np.asarray(prediction_arr)[index].tolist(),
                    "confidence": None if confidence_arr is None else float(np.asarray(confidence_arr)[index]),
                    "input": _normalize_vision_tensor(input_arr[index]),
                    "saliency": _normalize_vision_saliency(np.asarray(saliency)[index]),
                }
            )
        return {"run_id": run_id, "step": meta.get("step"), "available": True, "modality": "vision", "samples": samples}

    if modality == "structured":
        saliency_arr = np.asarray(saliency, dtype=np.float64)
        feature_scores = np.mean(np.abs(saliency_arr), axis=0)
        top_indices = np.argsort(feature_scores)[::-1][: min(16, feature_scores.shape[0])]
        feature_importance = [
            {"index": int(idx), "feature": f"Feature {int(idx) + 1}", "score": float(feature_scores[idx])}
            for idx in top_indices
        ]
        sample_count = min(int(input_arr.shape[0]), int(saliency_arr.shape[0]), 6)
        samples = []
        for index in range(sample_count):
            samples.append(
                {
                    "index": index,
                    "target": None if target_arr is None else np.asarray(target_arr)[index].tolist(),
                    "prediction": None if prediction_arr is None else np.asarray(prediction_arr)[index].tolist(),
                    "confidence": None if confidence_arr is None else float(np.asarray(confidence_arr)[index]),
                    "input": np.asarray(input_arr[index], dtype=float).tolist(),
                    "saliency": np.asarray(saliency_arr[index], dtype=float).tolist(),
                }
            )
        return {
            "run_id": run_id,
            "step": meta.get("step"),
            "available": True,
            "modality": "structured",
            "samples": samples,
            "feature_importance": feature_importance,
        }

    return {
        "run_id": run_id,
        "step": meta.get("step"),
        "available": False,
        "reason": saliency_meta.get("reason") or "The captured saliency format is not supported by this view yet.",
    }


def _prepare_embedding_matrix(values: np.ndarray) -> tuple[Optional[np.ndarray], Optional[str]]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim < 2 or arr.shape[0] < 2:
        return None, None
    if arr.ndim == 2:
        matrix = arr
        pooling = "none"
    elif arr.ndim == 3:
        matrix = arr.mean(axis=1)
        pooling = "sequence_mean"
    else:
        matrix = arr.reshape(arr.shape[0], arr.shape[1], -1).mean(axis=-1)
        pooling = "spatial_mean"
    if matrix.ndim != 2 or matrix.shape[1] < 2:
        return None, None
    return matrix, pooling


def _pca_2d(matrix: np.ndarray) -> tuple[np.ndarray, list[float]]:
    centered = matrix - np.mean(matrix, axis=0, keepdims=True)
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    components = min(2, vt.shape[0], centered.shape[1])
    projection = centered @ vt[:components].T
    if components == 1:
        projection = np.column_stack([projection[:, 0], np.zeros(centered.shape[0])])
    explained = (singular_values**2) / max(centered.shape[0] - 1, 1)
    total = float(np.sum(explained))
    if total <= 0:
        ratios = [0.0, 0.0]
    else:
        ratios = explained[:components] / total
        ratios = ratios.astype(float).tolist()
        if len(ratios) == 1:
            ratios.append(0.0)
    return projection.astype(float), ratios[:2]


def build_embeddings_payload(store, run_id: str, step: Optional[int] = None) -> dict[str, Any]:
    try:
        probe_bundle = store.load_probe_bundle(run_id, step=step)
    except FileNotFoundError:
        return {"run_id": run_id, "available": False, "reason": "No probe activations were captured for this run."}

    meta = probe_bundle["meta"]
    arrays = probe_bundle["arrays"]
    layers = []

    for layer_meta in meta.get("activation_layers", []):
        key = layer_meta.get("key")
        if not key or key not in arrays:
            continue
        matrix, pooling = _prepare_embedding_matrix(arrays[key])
        if matrix is None:
            continue
        projection, explained_variance_ratio = _pca_2d(matrix)
        layers.append(
            {
                "layer": layer_meta.get("layer", key),
                "original_shape": layer_meta.get("shape") or list(np.asarray(arrays[key]).shape),
                "matrix_shape": list(matrix.shape),
                "pooling": pooling,
                "explained_variance_ratio": explained_variance_ratio,
                "projection": projection.tolist(),
            }
        )

    if not layers:
        return {
            "run_id": run_id,
            "step": meta.get("step"),
            "available": False,
            "reason": "No activation layers in the latest probe bundle could be projected into 2D.",
        }

    return {
        "run_id": run_id,
        "step": meta.get("step"),
        "available": True,
        "default_layer": layers[0]["layer"],
        "targets": _to_python_list(arrays.get("targets")),
        "predictions": _to_python_list(arrays.get("predictions")),
        "confidence": _to_python_list(arrays.get("confidence")),
        "layers": layers,
    }
