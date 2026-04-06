# GradGlass Example Pack

Install the example dependencies with:

```bash
pip install -e .[torch,explainability]
```

## Example Guide

- `01_pytorch_core_tracking.py`
  Smallest useful PyTorch example: create a model, watch it with GradGlass, log metrics, capture logits-based prediction probes, checkpoint, and analyze the run.
- `02_keras_integration.py`
  TensorFlow / Keras callback integration.
- `03_data_leakage.py`
  Standalone and run-scoped leakage checks.
- `04_dataset_monitoring.py`
  Dataset monitoring across pipeline stages and transformations.
- `05_custom_tests_and_shap.py`
  Real tabular PyTorch + SHAP workflow with a custom attribution-dominance test layered onto the built-in GradGlass analysis suite.
- `06_workspace_and_server.py`
  Workspace browsing, retrieving prior runs, and serving the dashboard manually.
- `07_pytorch_full_observability.py`
  A fuller PyTorch vision example that lights up metrics, checkpoints, evaluation, prediction probes, saliency, gradients, activations, and the live dashboard flow.
- `08_api_feature_coverage.py`
  Full API coverage harness: creates multiple synthetic runs, starts the server on the same workspace, and programmatically exercises every current `/api` REST feature plus the websocket stream.
- `09_dashboard_showcase.py`
  Flagship workspace demo: generates multiple coordinated runs, launches the dashboard, and prints a guided tour that walks through every current dashboard section.

## Notes

- Examples now write to a single `gg_workspace/` folder beside the example script by default, so `python examples/01_pytorch_core_tracking.py` will create `examples/gg_workspace/` even if you launch it from the repository root.
- The generated `examples/gg_workspace/` tree is a runtime artifact for local exploration, not checked-in launch content.
- The dashboard gets richer prediction diagnostics when `run.log_batch(..., y_pred=logits)` receives raw class scores rather than argmax labels.
- The data-monitoring and leakage workflows already live in `03` and `04`, so the new examples focus on PyTorch training and explainability.
- `08` is intentionally a coverage harness rather than a beginner tutorial. Use it when you want to validate the full API surface end to end.
