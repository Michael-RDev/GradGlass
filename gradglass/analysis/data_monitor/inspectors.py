from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from gradglass.analysis.data_monitor.fingerprinting import (
    build_audio_signature,
    build_image_signature,
    build_numeric_signature,
    build_text_signature,
    exact_fingerprint,
    normalized_fingerprint,
    normalize_text,
    tokenize_text,
)
from gradglass.analysis.data_monitor.models import ModalityType, TaskType


@dataclass
class SampleObservation:
    index: int
    split: str
    modality: ModalityType
    sample: Any
    label: Any = None
    exact_fingerprint: Optional[str] = None
    normalized_fingerprint: Optional[str] = None
    approximate_signature: np.ndarray = field(default_factory=lambda: np.zeros(16, dtype=np.float32))
    schema_fields: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    missing_rate: float = 0.0
    feature_names: list[str] = field(default_factory=list)
    feature_vector: Optional[np.ndarray] = None
    normalized_label: Any = None


def _normalize_label(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _normalize_label(value.item())
        return [_normalize_label(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _normalize_label(value.item())
    if isinstance(value, str):
        return normalize_text(value)
    if isinstance(value, (list, tuple, set)):
        return [_normalize_label(item) for item in value]
    return value


def _missing_rate_for_dict(sample: dict[str, Any]) -> float:
    if not sample:
        return 0.0
    missing = 0
    total = 0
    for value in sample.values():
        total += 1
        if value is None:
            missing += 1
        elif isinstance(value, float) and np.isnan(value):
            missing += 1
        elif isinstance(value, np.ndarray) and value.size > 0:
            missing += int(np.isnan(value).sum())
            total += int(value.size) - 1
    return float(missing / max(total, 1))


def _task_implied_modality(task: TaskType) -> Optional[ModalityType]:
    if task == TaskType.LANGUAGE_MODELING:
        return ModalityType.TEXT
    if task == TaskType.VISION:
        return ModalityType.IMAGE
    if task == TaskType.AUDIO:
        return ModalityType.AUDIO
    if task == TaskType.TABULAR:
        return ModalityType.TABULAR
    if task == TaskType.SEQUENCE_TIME_SERIES:
        return ModalityType.TOKEN_SEQUENCE
    return None


def detect_modality(sample: Any, task: TaskType = TaskType.UNKNOWN, task_hint: str | None = None) -> ModalityType:
    task_modality = _task_implied_modality(task)
    if task_modality is not None:
        return task_modality
    hint = (task_hint or "").lower()
    if any(token in hint for token in ("text", "token", "language", "llm")):
        return ModalityType.TEXT
    if any(token in hint for token in ("image", "vision")):
        return ModalityType.IMAGE
    if any(token in hint for token in ("audio", "speech", "wave")):
        return ModalityType.AUDIO
    if any(token in hint for token in ("tabular", "table", "csv")):
        return ModalityType.TABULAR
    if isinstance(sample, str):
        return ModalityType.TEXT
    if isinstance(sample, dict):
        field_modalities = {detect_modality(value, TaskType.UNKNOWN, None) for value in sample.values() if value is not None}
        field_modalities.discard(ModalityType.UNKNOWN)
        if len(field_modalities) > 1:
            return ModalityType.MULTIMODAL
        if len(field_modalities) == 1:
            return list(field_modalities)[0]
        return ModalityType.TABULAR
    if isinstance(sample, (list, tuple)) and sample:
        if all(isinstance(item, str) for item in sample):
            return ModalityType.TEXT
        if all(isinstance(item, (int, np.integer)) for item in sample):
            return ModalityType.TOKEN_SEQUENCE
        if any(isinstance(item, dict) for item in sample):
            return ModalityType.MULTIMODAL
    if isinstance(sample, np.ndarray):
        arr = np.asarray(sample)
        if arr.ndim == 0:
            return ModalityType.TABULAR
        if arr.ndim == 1:
            if np.issubdtype(arr.dtype, np.integer) and arr.size >= 4:
                return ModalityType.TOKEN_SEQUENCE
            if np.issubdtype(arr.dtype, np.floating) and arr.size >= 512:
                return ModalityType.AUDIO
            return ModalityType.TABULAR
        if arr.ndim == 2:
            shape = arr.shape
            if min(shape) >= 8 and max(shape) >= 8 and abs(shape[0] - shape[1]) <= max(shape) * 0.75:
                return ModalityType.IMAGE
            if shape[1] <= 16:
                return ModalityType.TENSOR
            return ModalityType.AUDIO
        if arr.ndim == 3:
            channels = arr.shape[0] if arr.shape[0] in (1, 3, 4) else arr.shape[-1]
            if channels in (1, 3, 4):
                return ModalityType.IMAGE
            return ModalityType.TENSOR
        return ModalityType.TENSOR
    return ModalityType.UNKNOWN


def _flatten_numeric_features(sample: Any) -> tuple[list[str], np.ndarray]:
    if isinstance(sample, np.ndarray):
        arr = np.asarray(sample, dtype=np.float32).reshape(-1)
        return [f"feature_{index}" for index in range(arr.size)], arr
    if isinstance(sample, dict):
        names = []
        values = []
        for key in sorted(sample):
            value = sample[key]
            if isinstance(value, np.ndarray):
                arr = np.asarray(value, dtype=np.float32).reshape(-1)
                for idx, item in enumerate(arr.tolist()):
                    names.append(f"{key}[{idx}]")
                    values.append(float(item))
            elif isinstance(value, (int, float, np.integer, np.floating)):
                names.append(str(key))
                values.append(float(value))
        return names, np.asarray(values, dtype=np.float32)
    return [], np.zeros(0, dtype=np.float32)


def inspect_sample(
    sample: Any,
    *,
    index: int,
    split: str,
    label: Any = None,
    task: TaskType = TaskType.UNKNOWN,
    task_hint: str | None = None,
    signature_dims: int = 16,
) -> SampleObservation:
    modality = detect_modality(sample, task=task, task_hint=task_hint)
    exact_fp = exact_fingerprint(sample)
    normalized_fp = normalized_fingerprint(sample)
    schema_fields: list[str] = []
    metrics: dict[str, Any] = {}
    feature_names: list[str] = []
    feature_vector: Optional[np.ndarray] = None
    missing_rate = 0.0

    if modality == ModalityType.TEXT:
        text = sample if isinstance(sample, str) else " ".join(str(item) for item in sample)
        tokens = tokenize_text(text)
        metrics["sequence_length"] = len(tokens)
        metrics["text_preview"] = normalize_text(text)[:120]
        approx = build_text_signature(text, dims=signature_dims)
    elif modality == ModalityType.TOKEN_SEQUENCE:
        arr = np.asarray(sample)
        metrics["sequence_length"] = int(arr.reshape(-1).size)
        feature_names, feature_vector = _flatten_numeric_features(arr)
        approx = build_numeric_signature(arr, dims=signature_dims)
    elif modality == ModalityType.IMAGE:
        arr = np.asarray(sample)
        height = int(arr.shape[-2]) if arr.ndim >= 2 else 0
        width = int(arr.shape[-1]) if arr.ndim >= 2 else 0
        metrics["image_height"] = height
        metrics["image_width"] = width
        metrics["aspect_ratio"] = round(float(width / max(height, 1)), 4) if width and height else None
        feature_names = ["image_height", "image_width"]
        feature_vector = np.asarray([height, width], dtype=np.float32)
        approx = build_image_signature(arr, dims=signature_dims)
    elif modality == ModalityType.AUDIO:
        arr = np.asarray(sample)
        waveform = arr.reshape(-1)
        metrics["audio_length"] = int(waveform.size)
        metrics["duration_s"] = round(float(waveform.size / 16000.0), 4)
        metrics["sample_rate"] = 16000
        approx = build_audio_signature(waveform, dims=signature_dims)
    elif modality == ModalityType.MULTIMODAL:
        parts = {}
        part_vectors = []
        if isinstance(sample, dict):
            schema_fields = sorted(sample.keys())
            missing_rate = _missing_rate_for_dict(sample)
            for key in schema_fields:
                field_modality = detect_modality(sample[key], TaskType.UNKNOWN, None)
                parts[key] = field_modality.value
                _, part_feature_vector = _flatten_numeric_features(sample[key])
                if part_feature_vector.size:
                    part_vectors.append(build_numeric_signature(part_feature_vector, dims=signature_dims))
            metrics["field_modalities"] = parts
        approx = np.mean(part_vectors, axis=0) if part_vectors else np.zeros(signature_dims, dtype=np.float32)
    else:
        if isinstance(sample, dict):
            schema_fields = sorted(sample.keys())
            missing_rate = _missing_rate_for_dict(sample)
        feature_names, feature_vector = _flatten_numeric_features(sample)
        if feature_vector is None or feature_vector.size == 0:
            approx = np.zeros(signature_dims, dtype=np.float32)
        else:
            approx = build_numeric_signature(feature_vector, dims=signature_dims)
        if modality == ModalityType.TABULAR:
            metrics["feature_count"] = len(feature_names)
        elif modality == ModalityType.TENSOR:
            arr = np.asarray(sample)
            metrics["tensor_shape"] = list(arr.shape)

    return SampleObservation(
        index=index,
        split=split,
        modality=modality,
        sample=sample,
        label=label,
        exact_fingerprint=exact_fp,
        normalized_fingerprint=normalized_fp,
        approximate_signature=np.asarray(approx, dtype=np.float32),
        schema_fields=schema_fields,
        metrics=metrics,
        missing_rate=missing_rate,
        feature_names=feature_names,
        feature_vector=feature_vector,
        normalized_label=_normalize_label(label),
    )
