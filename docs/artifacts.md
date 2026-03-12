# Artifact Storage

GradGlass writes all run data to a local directory tree. Nothing is sent to the network. This document covers the directory layout, every file that gets written, and the `ArtifactStore` API for reading them programmatically.

---

## Table of Contents

- [Default Location](#default-location)
- [Directory Layout](#directory-layout)
- [File Reference](#file-reference)
  - [metadata.json](#metadatajson)
  - [metrics.jsonl](#metricsjsonl)
  - [model_structure.json](#model_structurejson)
  - [checkpoints/](#checkpoints)
  - [gradients/](#gradients)
  - [activations/](#activations)
  - [predictions/](#predictions)
  - [analysis/](#analysis)
- [`ArtifactStore` API](#artifactstore-api)

---

## Default Location

Artifacts are written to `gg_artifacts/` relative to your current working directory:

```
<cwd>/
  gg_artifacts/
    runs/
      <run-id>/
        ...
```

Override the root in Python:

```python
gg.configure(root='/data/ml-artifacts')
# or per-run:
run = gg.run('my-exp', root='/data/ml-artifacts')
```

Or via environment variable (before starting Python):

```bash
export GRADGLASS_ROOT=/data/ml-artifacts
```

See [Configuration](configuration.md) for the full list of options.

---

## Directory Layout

```
gg_artifacts/
└── runs/
    └── {name}-{YYYYMMDDHHMMSS}-{6hex}/     ← run directory
        │
        ├── metadata.json                   ← run config, environment, status
        ├── metrics.jsonl                   ← one JSON line per run.log() call
        ├── model_structure.json            ← architecture DAG
        │
        ├── checkpoints/
        │   ├── step_0.npz                  ← deep-learning checkpoint (NumPy)
        │   ├── step_0_meta.json            ← checkpoint metadata
        │   ├── step_500.npz
        │   ├── step_500_meta.json
        │   ├── step_0.pkl                  ← sklearn/XGBoost/LightGBM checkpoint
        │   └── step_0_meta.json
        │
        ├── gradients/
        │   ├── summaries_step_0.json       ← per-parameter gradient stats (deep learning)
        │   ├── summaries_step_50.json
        │   └── sklearn_diagnostics_step_0.json  ← sklearn post-fit diagnostics
        │
        ├── activations/
        │   ├── {layer}_step_0.npy          ← raw activation arrays
        │   └── {layer}_step_0_stats.json   ← fallback stats if .npy write fails
        │
        ├── predictions/
        │   ├── probe_step_0.json           ← prediction probe
        │   └── probe_step_100.json
        │
        ├── slices/                         ← reserved for future slice analysis
        │
        └── analysis/
            ├── report.json                 ← full PostRunReport
            ├── summary.txt                 ← human-readable text report
            ├── tests.jsonl                 ← one TestResult per line with timestamp
            └── leakage_report.json         ← optional, from run.check_leakage()
```

---

## File Reference

### `metadata.json`

Written at run creation and updated on `run.finish()`. Contains all static run information.

```json
{
  "run_id": "mnist-cnn-20260309194202-474f8f",
  "name": "mnist-cnn",
  "framework": "pytorch",
  "status": "complete",
  "start_time": "2026-03-09T19:42:02.000000",
  "end_time": "2026-03-09T19:43:15.000000",
  "config": {
    "lr": 0.001,
    "epochs": 10,
    "batch_size": 256
  },
  "capture_config": {
    "activations": "auto",
    "gradients": "summary",
    "activation_batches": 2,
    "grad_every": 50
  },
  "environment": {
    "python_version": "3.11.4",
    "torch_version": "2.1.0",
    "tf_version": null,
    "cuda_available": true,
    "cuda_version": "12.1",
    "gpu_name": "NVIDIA A100"
  },
  "git_commit": "a3f9c21"
}
```

**`status`** is one of: `"running"` · `"complete"` · `"failed"`

---

### `metrics.jsonl`

Appended to on every `run.log()` call. Each line is a self-contained JSON object.

```jsonl
{"step": 0, "timestamp": "2026-03-09T19:42:05.123", "loss": 2.312, "acc": 0.11, "lr": 0.001, "epoch": 0}
{"step": 1, "timestamp": "2026-03-09T19:42:05.456", "loss": 2.198, "acc": 0.14, "lr": 0.001, "epoch": 0}
{"step": 100, "timestamp": "2026-03-09T19:42:35.789", "loss": 0.412, "acc": 0.87, "lr": 0.0005, "epoch": 2}
```

Fields:
- `step` — auto-incremented integer, starts at 0
- `timestamp` — ISO timestamp
- `lr` — included automatically if an optimizer is being watched
- All other keys — passed by the user to `run.log()`

---

### `model_structure.json`

Written once when `run.watch()` is called. Describes the model architecture.

```json
{
  "framework": "pytorch",
  "is_boosting": false,
  "layers": [
    {
      "id": "conv1",
      "type": "Conv2d",
      "input_shape": [1, 28, 28],
      "output_shape": [32, 26, 26],
      "num_params": 320,
      "trainable": true
    }
  ],
  "edges": [
    {"from": "conv1", "to": "relu1"}
  ],
  "pipeline_steps": [],
  "feature_importances": null,
  "trees_info": null
}
```

For sklearn `Pipeline` objects, `pipeline_steps` contains per-step hyperparameters. For XGBoost/LightGBM, `trees_info` contains booster-level statistics and `feature_importances` contains per-feature scores.

---

### `checkpoints/`

Each checkpoint produces two files: the weights file and a metadata sidecar.

**`step_N.npz`** — deep learning (PyTorch / Keras):
A compressed NumPy archive where each key is a parameter name and each value is a numpy array of the parameter's weights.

```python
import numpy as np
ckpt = np.load('checkpoints/step_100.npz')
weights = dict(ckpt)  # {'layer.weight': array(...), 'layer.bias': array(...)}
```

**`step_N.pkl`** — sklearn / XGBoost / LightGBM:
The fitted estimator serialised via joblib (falls back to pickle).

```python
import joblib
model = joblib.load('checkpoints/step_0.pkl')
```

**`step_N_meta.json`** — sidecar for every checkpoint:

```json
{
  "step": 100,
  "tag": "epoch_2",
  "timestamp": "2026-03-09T19:42:40.000000",
  "num_params": 431080,
  "format": "npz",
  "size_mb": 1.64
}
```

---

### `gradients/`

**`summaries_step_N.json`** — one file per gradient capture (deep learning):

```json
{
  "step": 50,
  "timestamp": "2026-03-09T19:42:20.000000",
  "summaries": {
    "conv1.weight": {
      "mean": 0.0012,
      "var": 0.000004,
      "max": 0.021,
      "norm": 0.034,
      "kl_div": 0.003
    }
  }
}
```

`kl_div` is the KL divergence between this step's gradient distribution and the previous step's distribution. Used by `GRADIENT_DISTRIBUTION_SHIFT`.

**`sklearn_diagnostics_step_N.json`** — post-fit sklearn diagnostics:

```json
{
  "step": 0,
  "feature_importances": [0.12, 0.08, 0.31, ...],
  "coef": [[0.5, -0.3, ...]],
  "intercept": [0.01],
  "loss_curve": [2.3, 1.8, 1.2, ...],
  "oob_score": 0.934,
  "inertia": 1452.3,
  "explained_variance_ratio": [0.81, 0.12],
  "n_components": 2
}
```

Not all fields are present for every estimator — only those that the fitted model exposes.

---

### `activations/`

Written by the forward hooks attached in `run.watch()`. Only present for deep-learning runs.

**`{layer}_step_N.npy`** — raw activation array (numpy `.npy` format).

The array shape is `(N_samples, ...)` where the trailing dimensions match the layer's output shape.

**`{layer}_step_N_stats.json`** — fallback if the `.npy` write fails (e.g. activation is too large):

```json
{
  "layer": "relu1",
  "step": 0,
  "mean": 0.412,
  "var": 0.089,
  "sparsity": 0.23,
  "shape": [256, 32, 26, 26]
}
```

---

### `predictions/`

Written by `run.log_batch()`.

**`probe_step_N.json`**:

```json
{
  "step": 100,
  "timestamp": "2026-03-09T19:42:38.000000",
  "y_true": [3, 7, 1, 0, 5],
  "y_pred": [3, 7, 1, 0, 4],
  "confidence": [0.97, 0.91, 0.88, 0.99, 0.61],
  "logits_sample": [[0.1, -0.2, ...], ...],
  "loss": 0.142
}
```

`logits_sample` contains raw logits for the first 16 samples in the batch. `y_true`, `y_pred`, and `confidence` cover the entire batch.

---

### `analysis/`

Written by `run.analyze()` or `gg.analyze()`.

| File | Description |
|------|-------------|
| `report.json` | Full `PostRunReport` — all `TestResult` objects, summary counts |
| `summary.txt` | Human-readable terminal-style report with pass/warn/fail breakdown |
| `tests.jsonl` | One `TestResult` per line with an added `timestamp` field |
| `leakage_report.json` | Written by `run.check_leakage()` — see [Data Leakage Detection](leakage.md) |

---

## `ArtifactStore` API

`ArtifactStore` is the low-level I/O layer. You rarely need to call it directly — use the `Run` methods instead. It is useful for tooling, scripting, and writing custom analysis tests.

```python
from gradglass.artifacts import ArtifactStore

store = ArtifactStore(root='gg_artifacts')
```

### Reading runs

| Method | Returns | Description |
|--------|---------|-------------|
| `store.list_runs()` | `list[dict]` | All runs with enriched metadata (storage_mb, num_checkpoints, latest_loss, etc.) |
| `store.get_run_dir(run_id)` | `Path` | Absolute path to the run directory (no creation) |
| `store.get_metadata(run_id)` | `dict` | Parsed `metadata.json` |
| `store.get_metrics(run_id)` | `list[dict]` | All rows from `metrics.jsonl` |
| `store.get_latest_metric(run_id)` | `dict \| None` | Last row of `metrics.jsonl` |
| `store.get_checkpoints(run_id)` | `list[dict]` | Checkpoint metadata dicts, sorted by step, with `has_weights` and `size_mb` |
| `store.load_checkpoint(run_id, step)` | `dict \| object` | numpy dict (NPZ) or unpickled object (PKL) |
| `store.get_gradients(run_id)` | `list[dict]` | All gradient summary dicts |
| `store.get_activations(run_id)` | `list[dict]` | Activation stats from `*_stats.json` files and `.npy` metadata |
| `store.get_predictions(run_id)` | `list[dict]` | All prediction probe dicts |
| `store.get_model_structure(run_id)` | `dict \| None` | Parsed `model_structure.json` |
| `store.get_analysis_report(run_id)` | `dict \| None` | Parsed `analysis/report.json` |
| `store.get_rank_dirs(run_id)` | `list[Path]` | Subdirectories matching `rank_*/` for distributed runs |

### Writing runs

| Method | Description |
|--------|-------------|
| `store.create_run_dir(run_id)` | Create the run directory and all required subdirectories |

### Example: reading all metrics for a run

```python
from gradglass.artifacts import ArtifactStore

store = ArtifactStore()
metrics = store.get_metrics('mnist-cnn-20260309194202-474f8f')
losses = [m['loss'] for m in metrics if 'loss' in m]
```
