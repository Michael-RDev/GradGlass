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

If you are running GradGlass from this repo checkout instead of an installed release, build the dashboard once before
using `gradglass serve`:

```bash
npm --prefix gradglass/dashboard install
npm --prefix gradglass/dashboard run build
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

The repository examples intentionally override that to use the repo-root `gg_workspace/` so `gradglass serve` works
immediately after you run an example from the repo root.

Ways to override it:

```python
gg.configure(root="/path/to/workspace")
```

```bash
export GRADGLASS_ROOT=/path/to/workspace
```

## Open the dashboard

After your training script finishes:

```bash
gradglass serve --port 8432
gradglass open
gradglass list
```

`gradglass serve` reads the current workspace and requires a built dashboard bundle. In this repo, examples are wired
to the repo-root `gg_workspace/`, so a real training example followed by `gradglass serve --port 8432` is the normal
post-run flow.

If you need to inspect a different workspace from the shell, prefix the command:

```bash
GRADGLASS_ROOT=/path/to/workspace gradglass serve --port 8432
```

For live viewing during training, enable monitoring when you create the run:

```python
run = gg.run(name="experiment_name", task="classification", monitor=True)
```

Advanced Python-side server helpers still exist when you explicitly want them:

```python
run.serve(port=8432, open_browser=True)
gg.monitor(port=8432, open_browser=True)
```

Both the CLI and Python helpers fail fast with a helpful runtime error when the dashboard bundle is missing or the
requested port is already occupied. Detached monitor servers can be stopped later with `gradglass stop <run_id>`,
`gradglass stop --port 8432`, or `gradglass stop --all`.

## PyTorch pattern

PyTorch integration is explicit and training-loop friendly:

- pass the model and optimizer to `run.watch(...)`
- call `run.log(...)` inside your loop
- call `run.log_batch(...)` when you want richer probe artifacts
- use `run.checkpoint_every(n)` or manual `run.checkpoint()`
- use `monitor=True` on `gg.run(...)` if you want the dashboard live during training

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
