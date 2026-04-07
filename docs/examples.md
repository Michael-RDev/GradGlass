# Examples

The repository includes nine examples that double as documentation and feature coverage.

If you want a real training script first, start with `01_pytorch_core_tracking.py`, `02_keras_integration.py`,
`05_custom_tests_and_shap.py`, or `07_pytorch_full_observability.py`. Those train actual models and then print the
exact `gradglass serve` command for the repo-root workspace they created.

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

Shows listing runs, completing a small run, and then serving the resulting workspace afterward.

### 07: full observability

Shows a richer PyTorch path with activations, gradients, saliency, probes, checkpoints, and dashboard-driven artifacts.

### 08: API coverage

Creates synthetic runs for API coverage. This is a tooling example, not the primary “real training script” entrypoint.
Use `--serve` if you want it to launch the dashboard and exercise the API automatically.

### 09: dashboard showcase

Creates a synthetic multi-run workspace intended to populate every dashboard section with realistic-looking data. By
default it prepares the workspace and prints the `gradglass serve` command to run next.

## Example workspace behavior

The examples intentionally write into the repo-root `gg_workspace/` by default so you can run them from the repo root
and then immediately use plain `gradglass serve`.

That workspace is generated on demand and is not part of the launch branch or release artifacts.

Common conventions across the examples:

- real training examples print the exact next `gradglass serve` command for the workspace they just created
- synthetic tooling examples prefer generating artifacts first and only start the dashboard when you opt into `--serve`
- when you override `--root`, the printed command switches to `GRADGLASS_ROOT='...' gradglass serve --port ...`

If you are running from a source checkout, make sure the dashboard bundle exists before serving:

```bash
npm --prefix gradglass/dashboard install
npm --prefix gradglass/dashboard run build
```
