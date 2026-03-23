from __future__ import annotations

import numpy as np

import gradglass as gg

RND = 42


def make_dataset(n_samples: int = 2000, n_features: int = 20, seed: int = RND):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_samples, n_features)).astype(np.float32)
    weights = rng.normal(0.0, 1.0, size=(n_features,)).astype(np.float32)
    logits = x @ weights + 0.25 * rng.standard_normal(n_samples).astype(np.float32)
    y = (logits > 0.0).astype(np.int64)
    return x, y


def train_test_split(x, y, test_size: float = 0.3, seed: int = RND):
    rng = np.random.default_rng(seed)
    idx = np.arange(x.shape[0])
    rng.shuffle(idx)
    split = int(round(x.shape[0] * (1.0 - test_size)))
    train_idx = idx[:split]
    test_idx = idx[split:]
    return x[train_idx], x[test_idx], y[train_idx], y[test_idx]


def fit_standardizer(x):
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return mean, std


def transform_standardizer(x, mean, std):
    return (x - mean) / std


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def train_logistic_regression(x_train, y_train, lr: float = 0.05, steps: int = 250):
    n_features = x_train.shape[1]
    w = np.zeros(n_features, dtype=np.float32)
    b = 0.0
    y = y_train.astype(np.float32)

    for _ in range(steps):
        p = sigmoid(x_train @ w + b)
        err = p - y
        grad_w = (x_train.T @ err) / x_train.shape[0]
        grad_b = float(err.mean())
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


def predict_logistic_regression(x, w, b):
    return (sigmoid(x @ w + b) >= 0.5).astype(np.int64)


def accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


def leakage_scaling_before_split(x, y):
    mean, std = fit_standardizer(x)
    x_scaled = transform_standardizer(x, mean, std)  # uses test distribution information
    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.3, seed=RND)
    gg.run_leakage_detection(x_train, y_train, x_test, y_test)

    w, b = train_logistic_regression(x_train, y_train)
    preds = predict_logistic_regression(x_test, w, b)
    return accuracy(y_test, preds)


def correct_scaling_after_split(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, seed=RND)
    gg.run_leakage_detection(x_train, y_train, x_test, y_test)

    mean, std = fit_standardizer(x_train)
    x_train_scaled = transform_standardizer(x_train, mean, std)
    x_test_scaled = transform_standardizer(x_test, mean, std)

    w, b = train_logistic_regression(x_train_scaled, y_train)
    preds = predict_logistic_regression(x_test_scaled, w, b)
    return accuracy(y_test, preds)


def main():
    x, y = make_dataset()

    acc_leak = leakage_scaling_before_split(x, y)
    acc_correct = correct_scaling_after_split(x, y)

    print(f"Accuracy (scaling BEFORE split): {acc_leak:.4f}  <-- potential data leakage")
    print(f"Accuracy (scaling AFTER split) : {acc_correct:.4f}  <-- correct procedure")

    mean_full, _ = fit_standardizer(x)
    x_train, _, _, _ = train_test_split(x, y, test_size=0.3, seed=RND)
    mean_train, _ = fit_standardizer(x_train)
    print("First 5 feature means (full-data fit) :", np.round(mean_full[:5], 3).tolist())
    print("First 5 feature means (train-only fit):", np.round(mean_train[:5], 3).tolist())


if __name__ == "__main__":
    main()
