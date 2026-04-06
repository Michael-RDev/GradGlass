# Python API

## Top-level imports

The package exports a compact public surface:

```python
from gradglass import (
    gg,
    test,
    TestContext,
    TestResult,
    TestCategory,
    TestSeverity,
    DatasetMonitorBuilder,
    DatasetMonitorConfig,
    DatasetMonitorReport,
    LeakageDetector,
    DataLeakageReport,
    run_leakage_detection,
)
```

## `gg`: shared controller

`gg` is an instance of `GradGlass` defined in `gradglass/core.py`.

Common methods:

- `gg.configure(auto_open=False, root=None)` updates browser behavior and workspace root
- `gg.run(name, **options)` creates a new `Run`
- `gg.list_runs()` returns metadata summaries for existing runs
- `gg.get_run(run_id)` rehydrates a `Run` from disk
- `gg.open_last()` opens the most recent run in the dashboard
- `gg.analyze_run(run_id, **kwargs)` runs analysis on an existing run
- `gg.monitor_dataset(...)` creates a dataset monitor builder
- `gg.test()` returns the custom test decorator
- `gg.monitor(port=8432, open_browser=True)` starts the dashboard server for the current store

## `Run`

`Run` is the core unit of tracked work.

### Construction

Use `gg.run(name=..., ...)` rather than instantiating `Run` directly.

Useful options passed at creation time include:

- `task`
- `monitor`
- `port`
- `monitor_open_browser`
- configuration values that help infer `total_steps`

### Core methods

- `watch(model, optimizer=None, activations="auto", gradients="summary", saliency="auto", layers="trainable", sample_batches=2, probe_examples=16, every=50, monitor=None, monitor_port=None, monitor_open_browser=None)`
- `log(**metrics)`
- `log_batch(x, y=None, y_pred=None, loss=None)`
- `checkpoint(step=None, tag=None)`
- `checkpoint_every(interval)`
- `flush()`
- `analyze(tests="all", print_summary=True)`
- `finish(open=False, analyze=True, print_summary=True)`
- `fail(error, open=False, analyze=False, print_summary=True)`
- `cancel(reason=None, open=False, analyze=False, print_summary=True)`
- `interrupt(reason=None, open=False, analyze=False, print_summary=True)`
- `open()`
- `serve(port=0, open_browser=True)`
- `monitor(port=0, open_browser=True)`

### Explainability methods

- `log_shap(feature_names, shap_values, message=None, top_k=20)`
- `log_lime(samples, message=None)`

These write summary artifacts into the run directory so the API and dashboard can load them later.

### Data quality methods

- `monitor_dataset(task, dataset_name=None, task_hint=None, config=None)`
- `check_leakage(train_x, train_y, test_x, test_y, max_samples=2000, print_summary=True)`
- `check_leakage_from_loaders(train_loader, test_loader, max_samples=2000, print_summary=True)`

### Framework helpers

- `keras_callback()` returns a Keras callback for TensorFlow training
- `get_lr()` resolves a learning rate from the attached optimizer when possible

## Custom tests

Register your own analysis checks with the `test` decorator:

```python
from gradglass.analysis.registry import test, TestCategory, TestSeverity, TestResult, TestStatus


@test(
    id="MY_CUSTOM_CHECK",
    title="Flag suspicious confidence drift",
    category=TestCategory.PREDICTION,
    severity=TestSeverity.MEDIUM,
)
def my_check(ctx):
    suspicious = False
    return TestResult(
        id="MY_CUSTOM_CHECK",
        title="Flag suspicious confidence drift",
        status=TestStatus.WARN if suspicious else TestStatus.PASS,
        severity=TestSeverity.MEDIUM,
        category=TestCategory.PREDICTION,
        details={"suspicious": suspicious},
        recommendation="Inspect recent prediction probes." if suspicious else "",
    )
```

Registered tests become part of `run.analyze()`.

## Dataset monitor API

`DatasetMonitorBuilder` is a staged recorder for pipeline-level data checks.

Typical usage:

```python
monitor = gg.monitor_dataset(task="classification", dataset_name="customer_churn")
monitor.record_stage("raw", split="train", data=train_x, labels=train_y)
monitor.record_stage("raw", split="test", data=test_x, labels=test_y)
report = monitor.finalize(save=True)
```

It writes structured summaries and recommendations that also power `/api/runs/{run_id}/data-monitor`.

## Leakage API

There are two ways to use leakage detection:

- standalone through `LeakageDetector` or `run_leakage_detection(...)`
- run-scoped through `Run.check_leakage(...)`

The modern implementation is built on the dataset monitor system, then projected into the legacy leakage-report format.
