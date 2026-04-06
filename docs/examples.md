# Examples

The repository includes nine examples that double as documentation and feature coverage.

## Recommended reading order

1. `01_pytorch_core_tracking.py`
2. `02_keras_integration.py`
3. `03_data_leakage.py`
4. `04_dataset_monitoring.py`
5. `05_custom_tests_and_shap.py`
6. `06_workspace_and_server.py`
7. `07_pytorch_full_observability.py`
8. `08_api_feature_coverage.py`
9. `09_dashboard_showcase.py`

## What each example teaches

### 01: minimal PyTorch tracking

Shows the basic `gg.run(...)` and `run.watch(...)` pattern, plus metric logging, probe logging, checkpoint cadence, and analysis.

### 02: Keras integration

Shows how GradGlass fits a callback-based TensorFlow workflow without taking over training.

### 03: data leakage

Shows both standalone leakage detection and run-scoped leakage analysis.

### 04: dataset monitoring

Shows how to record dataset state across pipeline stages and emit a monitor report.

### 05: custom tests and SHAP

Shows how to register custom analysis checks and log SHAP summary artifacts for interpretability views.

### 06: workspace and server

Shows listing runs, reopening them, and manually starting a dashboard server.

### 07: full observability

Shows a richer PyTorch path with activations, gradients, saliency, probes, checkpoints, and dashboard-driven artifacts.

### 08: API coverage

Creates synthetic runs, starts the server, and programmatically exercises the entire API surface.

### 09: dashboard showcase

Creates a polished multi-run workspace intended to populate every dashboard section with realistic data.

## Example workspace behavior

The examples intentionally write into `examples/gg_workspace/` by default so you can run them from the repo root and
still get a coherent demo workspace.

That workspace is generated on demand and is not part of the launch branch or release artifacts.

For more detail, see [examples/README.md](../examples/README.md).
