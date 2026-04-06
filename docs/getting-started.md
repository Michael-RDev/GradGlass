# Getting Started

## Install

GradGlass targets Python 3.9+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Optional extras:

```bash
pip install -e .[torch]
pip install -e .[tensorflow]
pip install -e .[explainability]
pip install -e .[all]
```

## Create your first run

The main entry point is the shared `gg` object:

```python
from gradglass import gg
```

Create a run, attach it to a model, log metrics, and finish it:

```python
run = gg.run(name="experiment_name", task="classification")
run.watch(model, optimizer=optimizer, gradients="summary", activations="auto")

for batch in loader:
    ...
    run.log(loss=loss.item(), acc=acc)

run.finish()
```

## Understand the lifecycle

1. `gg.run(...)` allocates a run ID and creates its artifact directory.
2. `run.watch(...)` detects the framework, attaches capture hooks, writes metadata, and optionally starts a live monitor.
3. `run.log(...)` appends a metric row and may trigger gradient capture or auto-checkpointing.
4. `run.log_batch(...)` saves prediction probes and the payloads needed by evaluation, saliency, and embeddings views.
5. `run.analyze()` runs the post-training test suite and writes analysis files.
6. `run.finish()`, `run.fail()`, `run.cancel()`, or `run.interrupt()` finalize the run state.

## Configure the workspace

GradGlass writes to a `gg_workspace/` directory by default. The location is derived from the launching script when
possible, otherwise from the current working directory.

Ways to override it:

```python
gg.configure(root="/path/to/workspace")
```

```bash
export GRADGLASS_ROOT=/path/to/workspace
```

## Open the dashboard

From Python:

```python
run.serve(port=8432, open_browser=True)
```

or

```python
gg.monitor(port=8432)
```

From the CLI:

```bash
gradglass serve --port 8432
gradglass open
gradglass list
```

## PyTorch pattern

PyTorch integration is explicit and training-loop friendly:

- pass the model and optimizer to `run.watch(...)`
- call `run.log(...)` inside your loop
- call `run.log_batch(...)` when you want richer probe artifacts
- use `run.checkpoint_every(n)` or manual `run.checkpoint()`

The minimal working example lives at [examples/01_pytorch_core_tracking.py](../examples/01_pytorch_core_tracking.py).

## TensorFlow / Keras pattern

TensorFlow integration is callback-based:

```python
run = gg.run(name="keras_run")
run.watch(model)
model.fit(..., callbacks=[run.keras_callback()])
run.finish()
```

See [examples/02_keras_integration.py](../examples/02_keras_integration.py).

## Important behavioral notes

- `run.log()` and `run.log_batch()` both increment the internal step counter.
- `run.fit()` intentionally raises an error in this release.
- `run.log_batch()` is most useful when you pass raw logits instead of argmax labels.
- `saliency='auto'` and `activations='auto'` only produce useful dashboard panels if compatible probe data is captured.
- `run.finish(analyze=True)` runs post-run analysis by default.
