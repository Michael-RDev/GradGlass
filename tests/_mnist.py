from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np


_MNIST_FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "mnist_fixture.npz"


@lru_cache(maxsize=2)
def _mnist_split(split: str) -> tuple[np.ndarray, np.ndarray]:
    with np.load(_MNIST_FIXTURE_PATH) as fixture:
        dataset = {
            "train": (fixture["train_images"], fixture["train_labels"]),
            "test": (fixture["test_images"], fixture["test_labels"]),
        }

    normalized = str(split).strip().lower()
    if normalized not in dataset:
        raise ValueError(f"Unsupported MNIST split: {split}")
    images, labels = dataset[normalized]
    return images.copy(), labels.copy()


def mnist_subset(
    split: str, indices: list[int] | np.ndarray, *, flatten: bool = False, normalize: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    images, labels = _mnist_split(split)
    selected = np.asarray(indices, dtype=np.int64)
    subset_images = images[selected].astype(np.float32)
    if normalize:
        subset_images /= 255.0
    if flatten:
        subset_images = subset_images.reshape(len(selected), -1)
    subset_labels = labels[selected].astype(np.int64)
    return subset_images.copy(), subset_labels.copy()
