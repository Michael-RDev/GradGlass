from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any

import numpy as np


def _json_default(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def _hash_payload(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=_json_default, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def normalize_text(text: str) -> str:
    lowered = text.strip().lower()
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered


def tokenize_text(text: str) -> list[str]:
    return [token for token in re.findall(r"[a-z0-9_]+", normalize_text(text)) if token]


def build_text_signature(text: str, dims: int = 16) -> np.ndarray:
    tokens = tokenize_text(text)
    if not tokens:
        return np.zeros(dims, dtype=np.float32)
    shingles = []
    for token in tokens:
        shingles.append(token)
    for idx in range(max(0, len(tokens) - 1)):
        shingles.append(f"{tokens[idx]}::{tokens[idx + 1]}")
    vec = np.zeros(dims, dtype=np.float32)
    for item in shingles:
        digest = hashlib.md5(item.encode("utf-8")).hexdigest()
        bucket = int(digest[:8], 16) % dims
        vec[bucket] += 1.0
    norm = np.linalg.norm(vec)
    return vec if norm <= 1e-12 else vec / norm


def _normalize_float(value: float) -> float:
    if math.isnan(value):
        return 0.0
    if math.isinf(value):
        return 0.0
    return round(float(value), 6)


def normalize_numeric_array(array: np.ndarray) -> list[float]:
    flat = np.asarray(array, dtype=np.float32).reshape(-1)
    return [_normalize_float(v) for v in flat.tolist()]


def build_numeric_signature(array: np.ndarray, dims: int = 16) -> np.ndarray:
    flat = np.asarray(array, dtype=np.float32).reshape(-1)
    if flat.size == 0:
        return np.zeros(dims, dtype=np.float32)
    if flat.size >= dims:
        positions = np.linspace(0, flat.size - 1, num=dims).astype(int)
        sampled = flat[positions]
    else:
        sampled = np.pad(flat, (0, dims - flat.size), mode="constant")
    stats = np.array(
        [
            float(np.mean(flat)),
            float(np.std(flat)),
            float(np.min(flat)),
            float(np.max(flat)),
            float(np.median(flat)),
            float(np.percentile(flat, 10)),
            float(np.percentile(flat, 90)),
        ],
        dtype=np.float32,
    )
    vec = np.concatenate([stats, sampled.astype(np.float32)])[:dims]
    if vec.size < dims:
        vec = np.pad(vec, (0, dims - vec.size), mode="constant")
    norm = np.linalg.norm(vec)
    return vec if norm <= 1e-12 else vec / norm


def build_image_signature(array: np.ndarray, dims: int = 16) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim == 3:
        arr = np.mean(arr, axis=-1)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        return np.zeros(dims, dtype=np.float32)
    side = max(2, int(round(np.sqrt(dims))))
    y_idx = np.linspace(0, max(arr.shape[0] - 1, 0), num=side).astype(int)
    x_idx = np.linspace(0, max(arr.shape[1] - 1, 0), num=side).astype(int)
    down = arr[np.ix_(y_idx, x_idx)].reshape(-1)
    if down.size < dims:
        down = np.pad(down, (0, dims - down.size), mode="constant")
    down = down[:dims]
    mean = float(np.mean(down)) if down.size else 0.0
    phash_bits = (down > mean).astype(np.float32)
    norm = np.linalg.norm(phash_bits)
    return phash_bits if norm <= 1e-12 else phash_bits / norm


def build_audio_signature(array: np.ndarray, dims: int = 16) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return np.zeros(dims, dtype=np.float32)
    positions = np.linspace(0, arr.size - 1, num=max(8, dims // 2)).astype(int)
    sampled = arr[positions]
    spectrum = np.abs(np.fft.rfft(arr[: min(arr.size, 2048)]))
    if spectrum.size == 0:
        spectrum = np.zeros(max(1, dims // 2), dtype=np.float32)
    freq_positions = np.linspace(0, spectrum.size - 1, num=max(8, dims // 2)).astype(int)
    freq = spectrum[freq_positions].astype(np.float32)
    vec = np.concatenate([sampled.astype(np.float32), freq])[:dims]
    if vec.size < dims:
        vec = np.pad(vec, (0, dims - vec.size), mode="constant")
    norm = np.linalg.norm(vec)
    return vec if norm <= 1e-12 else vec / norm


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    a = np.asarray(vec_a, dtype=np.float32)
    b = np.asarray(vec_b, dtype=np.float32)
    if a.size == 0 or b.size == 0:
        return 0.0
    if a.size != b.size:
        size = min(a.size, b.size)
        a = a[:size]
        b = b[:size]
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def canonicalize_value(value: Any) -> Any:
    if isinstance(value, str):
        return normalize_text(value)
    if isinstance(value, np.ndarray):
        return normalize_numeric_array(value)
    if isinstance(value, (list, tuple)):
        return [canonicalize_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): canonicalize_value(value[key]) for key in sorted(value)}
    if isinstance(value, np.generic):
        return canonicalize_value(value.item())
    if isinstance(value, float):
        return _normalize_float(value)
    return value


def exact_fingerprint(value: Any) -> str:
    if isinstance(value, np.ndarray):
        return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()
    return _hash_payload(value)


def normalized_fingerprint(value: Any) -> str:
    return _hash_payload(canonicalize_value(value))
