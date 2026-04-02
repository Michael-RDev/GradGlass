from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class AdaptedSample:
    index: int
    sample: Any
    label: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AdaptedDataset:
    records: list[AdaptedSample]
    total_count: Optional[int]
    source_type: str
    errors: list[str] = field(default_factory=list)


def _to_numpy(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu().numpy()
    if hasattr(value, "numpy") and callable(value.numpy):
        try:
            return value.numpy()
        except Exception:
            return value
    return value


def _safe_len(value: Any) -> Optional[int]:
    try:
        return len(value)
    except Exception:
        return None


def _labels_from_data(data: Any, labels: Any):
    if labels is not None:
        return data, labels
    if isinstance(data, tuple) and len(data) == 2:
        left_len = _safe_len(data[0])
        right_len = _safe_len(data[1])
        if left_len is not None and right_len is not None and left_len == right_len:
            return data[0], data[1]
    return data, labels


def _build_from_columnar_dict(data: dict[str, Any], limit: int) -> AdaptedDataset:
    columns = {key: _to_numpy(value) for key, value in data.items()}
    lengths = [_safe_len(value) for value in columns.values()]
    valid_lengths = [length for length in lengths if length is not None]
    total_count = min(valid_lengths) if valid_lengths else None
    if total_count is None:
        return AdaptedDataset(records=[], total_count=None, source_type="dict", errors=["Could not infer dict length"])
    records = []
    for index in range(min(total_count, limit)):
        row = {}
        for key, column in columns.items():
            try:
                row[key] = _to_numpy(column[index])
            except Exception:
                row[key] = None
        records.append(AdaptedSample(index=index, sample=row))
    return AdaptedDataset(records=records, total_count=total_count, source_type="dict")


def _build_from_array_like(data: Any, labels: Any, limit: int) -> AdaptedDataset:
    array = _to_numpy(data)
    label_array = _to_numpy(labels) if labels is not None else None
    total_count = _safe_len(array)
    if total_count is None:
        return AdaptedDataset(records=[], total_count=None, source_type=type(data).__name__, errors=["Unsupported array-like input"])
    records = []
    for index in range(min(total_count, limit)):
        label = None
        if label_array is not None:
            try:
                label = _to_numpy(label_array[index])
            except Exception:
                label = None
        records.append(AdaptedSample(index=index, sample=_to_numpy(array[index]), label=label))
    return AdaptedDataset(records=records, total_count=total_count, source_type=type(data).__name__)


def _iter_batches(loader: Any, limit: int) -> AdaptedDataset:
    records: list[AdaptedSample] = []
    errors: list[str] = []
    total_count = None
    try:
        total_count = len(loader.dataset)  # type: ignore[attr-defined]
    except Exception:
        total_count = None
    seen = 0
    for batch in loader:
        if seen >= limit:
            break
        try:
            if isinstance(batch, dict):
                x_batch = batch.get("x", batch.get("inputs", batch.get("input", batch)))
                y_batch = batch.get("y", batch.get("labels", batch.get("targets")))
            elif isinstance(batch, (tuple, list)) and len(batch) >= 2:
                x_batch, y_batch = batch[0], batch[1]
            else:
                x_batch, y_batch = batch, None
            x_np = _to_numpy(x_batch)
            y_np = _to_numpy(y_batch) if y_batch is not None else None
            batch_len = _safe_len(x_np) or 1
            for local_idx in range(batch_len):
                if seen >= limit:
                    break
                sample = _to_numpy(x_np[local_idx]) if batch_len > 1 else _to_numpy(x_np)
                label = None
                if y_np is not None:
                    try:
                        label = _to_numpy(y_np[local_idx]) if batch_len > 1 else _to_numpy(y_np)
                    except Exception:
                        label = None
                records.append(AdaptedSample(index=seen, sample=sample, label=label))
                seen += 1
        except Exception as exc:
            errors.append(str(exc))
    return AdaptedDataset(records=records, total_count=total_count, source_type=type(loader).__name__, errors=errors)


def _iter_dataset(dataset: Any, limit: int) -> AdaptedDataset:
    total_count = _safe_len(dataset)
    records = []
    errors = []
    if total_count is None:
        return AdaptedDataset(records=[], total_count=None, source_type=type(dataset).__name__, errors=["Dataset length unavailable"])
    for index in range(min(total_count, limit)):
        try:
            item = dataset[index]
            if isinstance(item, (tuple, list)) and len(item) >= 2:
                sample, label = item[0], item[1]
            else:
                sample, label = item, None
            records.append(AdaptedSample(index=index, sample=_to_numpy(sample), label=_to_numpy(label)))
        except Exception as exc:
            errors.append(str(exc))
    return AdaptedDataset(records=records, total_count=total_count, source_type=type(dataset).__name__, errors=errors)


def _iter_tf_dataset(dataset: Any, limit: int) -> AdaptedDataset:
    records = []
    errors = []
    total_count = None
    iterator = None
    try:
        iterator = dataset.as_numpy_iterator()
    except Exception:
        iterator = iter(dataset)
    for index, item in enumerate(iterator):
        if index >= limit:
            break
        try:
            if isinstance(item, (tuple, list)) and len(item) >= 2:
                sample, label = item[0], item[1]
            else:
                sample, label = item, None
            records.append(AdaptedSample(index=index, sample=_to_numpy(sample), label=_to_numpy(label)))
        except Exception as exc:
            errors.append(str(exc))
    return AdaptedDataset(records=records, total_count=total_count, source_type=type(dataset).__name__, errors=errors)


def adapt_input(data: Any, labels: Any = None, limit: int = 5000) -> AdaptedDataset:
    data, labels = _labels_from_data(data, labels)

    if data is None:
        return AdaptedDataset(records=[], total_count=0, source_type="none", errors=["No data provided"])

    if isinstance(data, dict):
        return _build_from_columnar_dict(data, limit)

    module_name = type(data).__module__
    qualname = type(data).__qualname__

    if "torch.utils.data" in module_name and "DataLoader" in qualname:
        return _iter_batches(data, limit)
    if "torch.utils.data" in module_name and "Dataset" in qualname:
        return _iter_dataset(data, limit)
    if module_name.startswith("tensorflow") and "Dataset" in qualname:
        return _iter_tf_dataset(data, limit)

    if isinstance(data, np.ndarray):
        return _build_from_array_like(data, labels, limit)

    if isinstance(data, (list, tuple)):
        if data and isinstance(data[0], (dict, str, bytes)):
            total_count = len(data)
            records = []
            label_values = _to_numpy(labels) if labels is not None else None
            for index in range(min(total_count, limit)):
                label = None
                if label_values is not None:
                    try:
                        label = _to_numpy(label_values[index])
                    except Exception:
                        label = None
                records.append(AdaptedSample(index=index, sample=_to_numpy(data[index]), label=label))
            return AdaptedDataset(records=records, total_count=total_count, source_type=type(data).__name__)
        return _build_from_array_like(np.asarray(data, dtype=object), labels, limit)

    if hasattr(data, "__getitem__") and hasattr(data, "__len__"):
        return _iter_dataset(data, limit)

    return AdaptedDataset(
        records=[AdaptedSample(index=0, sample=_to_numpy(data), label=_to_numpy(labels))],
        total_count=1,
        source_type=type(data).__name__,
        errors=["Fell back to single-record adaptation"],
    )
