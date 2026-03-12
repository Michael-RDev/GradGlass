# Python API Reference

---

## Table of Contents

- [The `gg` Singleton — `GradGlass`](#the-gg-singleton--gradglass)
- [The `Run` Class](#the-run-class)
  - [Watching a Model](#watching-a-model)
  - [Logging](#logging)
  - [Checkpointing](#checkpointing)
  - [Analysis & Leakage](#analysis--leakage)
  - [Dashboard](#dashboard)
  - [Finishing a Run](#finishing-a-run)
  - [Framework Callbacks](#framework-callbacks)
  - [Class Methods](#class-methods)
- [Custom Analysis Tests — `@test`](#custom-analysis-tests--test)
  - [`TestContext`](#testcontext)
  - [`TestResult` & `TestStatus`](#testresult--teststatus)
  - [`TestCategory`](#testcategory)
  - [`TestSeverity`](#testseverity)
- [Data Leakage — `LeakageDetector`](#data-leakage--leakagedetector)

---

## The `gg` Singleton — `GradGlass`

```python
from gradglass import gg
```

`gg` is a module-level singleton of `GradGlass`. It is created at import time; you never need to instantiate it yourself.

---

### `gg.configure(**kwargs)`

Configure the singleton before creating runs.

```python
gg.configure(root='./my_artifacts', auto_open=True)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `root` | `str \| Path` | `'gg_artifacts'` | Root directory for all artifact storage |
| `auto_open` | `bool` | `False` | Auto-launch the browser after every `run.finish()` |

Also respects the `GRADGLASS_OPEN=1` environment variable (sets `auto_open=True`).

---

### `gg.run(name, **kwargs) -> Run`

Create a new training run.

```python
run = gg.run('my-experiment', lr=0.001, epochs=50, batch_size=256)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | — | Human-readable run name. Used as the directory prefix. |
| `activations` | `str` | `'auto'` | `'auto'` to capture activations, `'off'` to disable |
| `gradients` | `str` | `'summary'` | `'summary'` to capture per-parameter gradient stats, `'off'` to disable |
| `checkpoint_layers` | `str \| list[str] \| None` | `None` (all layers) | Layer name(s) to include in checkpoints. `None` saves all. |
| `activation_batches` | `int` | `2` | Number of forward-pass batches to accumulate activations from |
| `grad_every` | `int` | `50` | Steps between gradient summary captures |
| `monitor` | `bool` | `False` | Start the live dashboard server immediately on run creation |
| `port` | `int` | `0` | Server port. `0` means pick a free port automatically. |
| `task` | `str \| None` | `None` | Optional label for the task type, e.g. `'nlp/text-classification'` |
| `**kwargs` | any | — | Arbitrary key-value pairs stored as run config (e.g. `lr=0.001`) |

The run ID is auto-generated as `{name}-{YYYYMMDDHHMMSS}-{6hex}`.

---

### `gg.list_runs() -> list[dict]`

Return metadata for every run in the artifact store, enriched with storage stats.

```python
runs = gg.list_runs()
for r in runs:
    print(r['run_id'], r['latest_loss'], r['storage_mb'])
```

Each dict contains: `run_id`, `name`, `framework`, `status`, `start_time`, `total_steps`, `latest_loss`, `latest_acc`, `num_checkpoints`, `storage_mb`.

---

### `gg.open_last()`

Open the most recently created run in the browser dashboard (starts the server if not running).

---

### `gg.load_run(run_id) -> Run`

Load an existing run from disk by its ID. Does not re-attach hooks.

```python
run = gg.load_run('my-experiment-20260309194202-474f8f')
```

---

### `gg.analyze(run_id, tests=None) -> PostRunReport`

Trigger the analysis suite on an existing run.

```python
report = gg.analyze('my-experiment-20260309194202-474f8f')
report.print_summary()
```

Pass a list of test IDs to `tests` to run only a subset.

---

### `gg.serve(port=0) -> int`

Start the dashboard server in a background thread. Returns the port it is listening on.

---

### `gg.test()`

Returns the `@test` decorator for registering custom analysis tests. See [Custom Analysis Tests](#custom-analysis-tests--test).

---

## The `Run` Class

A `Run` is created by `gg.run(...)`. It is the central object you interact with during training.

---

### Watching a Model

#### `run.watch(model, optimizer=None, **kwargs) -> Run`

Attach GradGlass to a model (and optionally an optimizer). Returns `self` for chaining.

```python
run = gg.run('my-exp').watch(model, optimizer)
```

- **PyTorch / Keras:** Registers forward and gradient hooks, saves `model_structure.json`.
- **sklearn / XGBoost / LightGBM:** Creates a capture adapter (no hooks). Also saves `model_structure.json`.

The `**kwargs` override any capture config set in `gg.run()` (e.g. `activations`, `gradients`, `monitor`).

---

### Logging

#### `run.log(**metrics)`

Log a dict of scalar metrics for the current step. Call once per training step or once per epoch.

```python
run.log(loss=0.312, acc=0.87, val_loss=0.45, epoch=3)
```

- Increments the internal step counter.
- Appends a JSON line to `metrics.jsonl` (includes `step`, `timestamp`, and all provided keys).
- Automatically includes the current learning rate if an optimizer is being watched.
- Triggers a gradient summary capture every `grad_every` steps.
- Triggers an auto-checkpoint if `set_auto_checkpoint()` was called.

---

#### `run.log_batch(x, y=None, y_pred=None, loss=None)`

Save a prediction probe for the current batch. Useful for tracking per-sample predictions over time.

```python
run.log_batch(x=x, y=y, y_pred=logits, loss=loss)
```

Saves a `predictions/probe_step_N.json` file containing:
- `y_true` — ground truth class indices
- `y_pred` — predicted class indices
- `confidence` — softmax confidence scores
- `logits_sample` — raw logits for the first 16 samples
- `loss` — scalar loss value

---

### Checkpointing

#### `run.checkpoint(step=None, tag='')`

Save a model checkpoint at the current (or specified) step.

```python
run.checkpoint(tag='best')
run.checkpoint(tag=f'epoch_{epoch}')
```

- **Deep learning (PyTorch / Keras):** Saves a compressed `.npz` file containing all named parameters as numpy arrays.
- **sklearn / XGBoost / LightGBM:** Saves a `.pkl` file via joblib (falls back to pickle).
- Always writes a `_meta.json` file alongside the checkpoint with: `step`, `tag`, `timestamp`, `num_params`, `format`, `size_mb`.

---

#### `run.set_auto_checkpoint(every)`

Automatically save a checkpoint every `every` steps (triggered inside `run.log()`).

```python
run.set_auto_checkpoint(every=500)
```

---

### Analysis & Leakage

#### `run.analyze(tests=None) -> PostRunReport`

Run the full analysis suite against this run's artifacts.

```python
report = run.analyze()
report.print_summary()
```

Pass a list of test IDs to `tests` to run only a subset.

---

#### `run.check_leakage(X_train, y_train, X_test, y_test, max_samples=2000)`

Run all 7 data leakage checks on numpy arrays (or anything array-like).

```python
run.check_leakage(X_train, y_train, X_test, y_test)
```

Results are saved to `analysis/leakage_report.json` and shown in the **Leakage Report** dashboard page. See [Data Leakage Detection](leakage.md) for a full description of all checks.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `X_train` | array-like | — | Training features |
| `y_train` | array-like | — | Training labels |
| `X_test` | array-like | — | Test features |
| `y_test` | array-like | — | Test labels |
| `max_samples` | `int` | `2000` | Subsample size to keep checks fast on large datasets |

---

#### `run.check_leakage_from_loaders(train_loader, test_loader, max_samples=2000)`

Same as `check_leakage()` but accepts PyTorch `DataLoader` objects. Gathers up to `max_samples` from each loader automatically.

---

### Dashboard

#### `run.open()`

Start the dashboard server (if not already running) and open `http://localhost:{port}/?run={run_id}` in the default browser. **Blocks until Ctrl+C.**

---

#### `run.start_server(port=0) -> int`

Start the dashboard server in a background thread. Returns the port. Non-blocking — use for embedding in a larger script.

---

#### `run.monitor(port=0, open_browser=True)`

Start the live monitoring dashboard before or during training. Idempotent — does nothing if the server is already running. The dashboard will update in real time as `run.log()` is called.

```python
run = gg.run('my-exp', monitor=True).watch(model, optimizer)
# or:
run.monitor(port=8432, open_browser=True)
```

---

### Finishing a Run

#### `run.finish(analyze=False, open=False)`

Flush all pending writes, mark the run as `"complete"`, and clean up hooks.

```python
run.finish()
run.finish(analyze=True)           # run analysis before finishing
run.finish(analyze=True, open=True) # run analysis then open the dashboard
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `analyze` | `bool` | `False` | Run the full analysis suite before finishing |
| `open` | `bool` | `False` | Open the dashboard in the browser after finishing |

`run.finish()` should always be called at the end of training. If `gg.configure(auto_open=True)` or `GRADGLASS_OPEN=1` is set, the browser opens regardless of the `open` parameter.

---

### Framework Callbacks

#### `run.keras_callback()`

Returns a Keras `Callback` instance. Pass it to `model.fit(callbacks=[...])`.

```python
model.fit(X, y, epochs=10, callbacks=[run.keras_callback()])
```

The callback calls `run.log()` after each epoch and saves a checkpoint at the end of training.

---

#### `run.xgboost_callback()`

Returns an `xgb.callback.TrainingCallback` instance. Pass it to `xgb.train(callbacks=[...])` or to `XGBClassifier.fit(callbacks=[...])`.

```python
booster = xgb.train(params, dtrain, callbacks=[run.xgboost_callback()])
```

Streams per-round eval metrics (from the `evals` argument) into `metrics.jsonl` after each boosting round.

---

#### `run.lightgbm_callback()`

Returns a LightGBM callback. Pass it to `lgb.train(callbacks=[...])` or `LGBMClassifier.fit(callbacks=[...])`.

```python
gbm = lgb.train(params, train_data, callbacks=[run.lightgbm_callback()])
```

Streams per-round eval metrics into `metrics.jsonl` after each boosting round.

---

### `run.fit(X, y=None, X_val=None, y_val=None, **kwargs) -> estimator`

**sklearn / XGBoost / LightGBM only.** Wraps the model's native `fit()` call and automatically captures post-fit diagnostics:

- Train and validation accuracy or R² (`acc`, `val_acc`)
- Per-round metrics from XGBoost/LightGBM eval results
- A checkpoint of the fitted model
- A prediction probe on the validation set (if provided)

```python
run = gg.run('random-forest').watch(clf)
run.fit(X_train, y_train, X_val=X_test, y_val=y_test)
```

Returns the fitted estimator.

---

### Class Methods

#### `Run.from_existing(run_id, store) -> Run`

Load an existing run from disk. Reads `metadata.json`; does not re-attach hooks.

```python
from gradglass.run import Run
from gradglass.artifacts import ArtifactStore

store = ArtifactStore()
run = Run.from_existing('my-experiment-20260309194202-474f8f', store)
```

---

## Custom Analysis Tests — `@test`

Register a custom analysis function that runs automatically with `run.analyze()`.

```python
from gradglass import gg, test, TestContext, TestResult, TestCategory, TestSeverity, TestStatus

@test(
    id='HIGH_FINAL_LOSS',
    title='Final loss below threshold',
    category=TestCategory.METRICS,
    severity=TestSeverity.HIGH,
    description='Fails if the final logged loss exceeds 1.0.',
)
def check_final_loss(ctx: TestContext) -> TestResult:
    losses = [m['loss'] for m in ctx.metrics if 'loss' in m]

    if not losses:
        return TestResult(
            id='HIGH_FINAL_LOSS', title='Final loss below threshold',
            status=TestStatus.SKIP, severity=TestSeverity.HIGH,
            category=TestCategory.METRICS,
            details={'reason': 'No loss values found'},
        )

    final = losses[-1]
    if final > 1.0:
        return TestResult(
            id='HIGH_FINAL_LOSS', title='Final loss below threshold',
            status=TestStatus.FAIL, severity=TestSeverity.HIGH,
            category=TestCategory.METRICS,
            details={'final_loss': final},
            recommendation='Lower the learning rate or train for more epochs.',
        )

    return TestResult(
        id='HIGH_FINAL_LOSS', title='Final loss below threshold',
        status=TestStatus.PASS, severity=TestSeverity.HIGH,
        category=TestCategory.METRICS,
        details={'final_loss': final},
    )
```

The decorated function is registered in the global `TestRegistry` and will be included in every subsequent call to `run.analyze()`.

### `@test` Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `id` | `str` | ✅ | Unique test ID in `SCREAMING_SNAKE_CASE` |
| `title` | `str` | ✅ | Short human-readable title |
| `category` | `TestCategory` | ✅ | Which category this test belongs to |
| `severity` | `TestSeverity` | ✅ | Default severity level |
| `description` | `str` | — | Longer explanation (shown in the Analysis dashboard page) |

Alternatively, use `gg.test()` to access the same decorator via the singleton:

```python
@gg.test()
def my_check(ctx):
    ...
```

---

### `TestContext`

All artifacts for the run are pre-loaded and available as properties.

| Property | Type | Description |
|----------|------|-------------|
| `run_id` | `str` | The run identifier |
| `metadata` | `dict` | Parsed `metadata.json` |
| `metrics` | `list[dict]` | All rows from `metrics.jsonl` (each row is one `run.log()` call) |
| `checkpoints` | `list[dict]` | Checkpoint metadata dicts (from `*_meta.json` files) |
| `model_structure` | `dict` | Parsed `model_structure.json` |
| `gradient_summaries` | `list[dict]` | Per-step gradient stat dicts (from `gradients/summaries_step_*.json`) |
| `activation_stats` | `list[dict]` | Per-layer activation stat dicts |
| `predictions` | `list[dict]` | Prediction probe dicts (from `predictions/probe_step_*.json`) |
| `framework` | `str` | e.g. `"pytorch"`, `"keras"`, `"sklearn"`, `"xgboost"`, `"lightgbm"` |
| `is_deep_learning` | `bool` | `True` for PyTorch / Keras runs |
| `is_sklearn` | `bool` | `True` for sklearn runs |
| `is_boosting` | `bool` | `True` for XGBoost / LightGBM runs |
| `has_checkpoints` | `bool` | At least one checkpoint exists |
| `has_gradients` | `bool` | At least one gradient summary exists |
| `has_activations` | `bool` | At least one activation stat exists |
| `has_metrics` | `bool` | At least one metric row exists |
| `has_predictions` | `bool` | At least one prediction probe exists |
| `has_model_structure` | `bool` | `model_structure.json` is present |
| `store` | `ArtifactStore` | Direct access to the artifact store (for advanced use) |
| `run_dir` | `Path` | Absolute path to the run's artifact directory |

---

### `TestResult` & `TestStatus`

```python
from gradglass import TestResult
from gradglass.analysis.registry import TestStatus
```

`TestResult` is a dataclass:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | `str` | ✅ | Must match the test's registered ID |
| `title` | `str` | ✅ | Human-readable title |
| `status` | `TestStatus` | ✅ | `PASS`, `WARN`, `FAIL`, or `SKIP` |
| `severity` | `TestSeverity` | ✅ | `LOW`, `MEDIUM`, `HIGH`, or `CRITICAL` |
| `category` | `TestCategory` | ✅ | Which category |
| `details` | `dict` | — | Arbitrary key-value pairs shown in the dashboard |
| `recommendation` | `str` | — | Actionable suggestion shown on WARN/FAIL |

`TestStatus` values:

| Value | Meaning |
|-------|---------|
| `TestStatus.PASS` | Check passed |
| `TestStatus.WARN` | Check passed but with a concern worth noting |
| `TestStatus.FAIL` | Check failed — action recommended |
| `TestStatus.SKIP` | Check could not run (missing data) |

---

### `TestCategory`

| Value | Description |
|-------|-------------|
| `TestCategory.STORE` | Artifact & store integrity |
| `TestCategory.ARCHITECTURE` | Model structure |
| `TestCategory.METRICS` | Training metrics |
| `TestCategory.CHECKPOINT` | Checkpoint diffs |
| `TestCategory.GRADIENTS` | Gradient flow |
| `TestCategory.ACTIVATIONS` | Activation distributions |
| `TestCategory.PREDICTIONS` | Prediction probes |
| `TestCategory.DATA` | Dataset integrity |
| `TestCategory.DISTRIBUTED` | Distributed training |
| `TestCategory.REPRODUCIBILITY` | Reproducibility |

---

### `TestSeverity`

| Value | Meaning |
|-------|---------|
| `TestSeverity.LOW` | Informational — worth noting |
| `TestSeverity.MEDIUM` | Should be investigated |
| `TestSeverity.HIGH` | Likely indicates a real problem |
| `TestSeverity.CRITICAL` | Indicates a serious or blocking issue |

---

## Data Leakage — `LeakageDetector`

For a full reference see [Data Leakage Detection](leakage.md). The class is exported for advanced use:

```python
from gradglass import LeakageDetector

detector = LeakageDetector(max_samples=2000)
report = detector.run_all_checks(X_train, y_train, X_test, y_test)
```
