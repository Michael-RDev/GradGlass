<p align="center">
  <img alt="GradGlass badge" src="./gradglass/dashboard/public/gradglass-badge.svg" width="96" height="96">
  <br/>
  <strong style="font-size: 2rem; letter-spacing: -0.03em;">GradGlass</strong>
  <br/>
  <br/>
</p>

<p align="center">
  <a href="./pyproject.toml"><img alt="Version" src="https://img.shields.io/badge/version-1.0.0-1f7a8c"></a>
  <a href="./LICENSE"><img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-0f4c5c"></a>
  <a href="./docs/index.md"><img alt="Documentation" src="https://img.shields.io/badge/docs-ready-2a9d8f"></a>
  <a href="./examples"><img alt="Examples" src="https://img.shields.io/badge/examples-9%20workflows-e9c46a"></a>
  <a href="./tests"><img alt="Tests" src="https://img.shields.io/badge/test%20suite-pytest-264653"></a>
</p>

<h3 align="center">Inspect, compare, debug, and monitor model training with local-first artifacts and a live dashboard.</h3>

GradGlass is a Python library for neural network transparency. It wraps your training loop with run tracking, checkpoints,
gradient and activation capture, prediction probes, data quality checks, leakage detection, post-run analysis, and a
browser dashboard backed by a local FastAPI server.

The library is designed around one idea: training runs should leave behind enough structured evidence to explain what
happened, not just whether the loss went down.

## Why GradGlass

- Local-first workflow. Artifacts are written into a workspace directory you can inspect, diff, archive, and serve later.
- Framework-aware capture. PyTorch and TensorFlow/Keras runs can emit checkpoints, architecture graphs, probes, and summaries.
- Built-in diagnosis. Analysis, alerts, data monitoring, and leakage checks are part of the package instead of external glue.
- Dashboard included. The bundled React app reads the same artifacts and exposes overview, training, evaluation, data, and interpretability views.
- Extensible testing model. You can register your own post-run tests and keep them alongside the built-in suite.

## Installation

Install gradglass from PyPI:

```bash
pip install gradglass
```

## Contributing

GradGlass requires Python 3.9+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Install optional extras as needed:

```bash
pip install -e .[torch]
pip install -e .[tensorflow]
pip install -e .[explainability]
pip install -e .[all]
```

If you are working from this source checkout, build the dashboard bundle once before using `gradglass serve`:

```bash
npm --prefix gradglass/dashboard install
npm --prefix gradglass/dashboard run build
```

When the dashboard bundle is missing, GradGlass now tries to build it while creating wheels and sdists. If the
frontend toolchain is unavailable, packaging fails with a clear message instead of silently shipping an incomplete UI.

## Quickstart

```python
import torch
import torch.nn as nn
import torch.optim as optim

from gradglass import gg


model = nn.Sequential(
    nn.Linear(4, 16),
    nn.ReLU(),
    nn.Linear(16, 2),
)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

run = gg.run(name="demo_run", task="classification")
run.checkpoint_every(10)
run.watch(
    model,
    optimizer=optimizer,
    activations="auto",
    gradients="summary",
    saliency="auto",
    every=10,
    probe_examples=16,
)

for step in range(100):
    x = torch.randn(32, 4)
    y = (x[:, 0] > 0).long()

    optimizer.zero_grad()
    logits = model(x)
    loss = criterion(logits, y)
    loss.backward()
    optimizer.step()

    acc = (logits.argmax(dim=1) == y).float().mean().item()
    run.log(loss=loss.item(), acc=acc)

    if (step + 1) % 10 == 0:
        run.log_batch(x=x, y=y, y_pred=logits.detach(), loss=loss.item())

run.analyze()
run.finish(open=False, analyze=False)
```

This creates a workspace with a structured run directory, including metadata, metrics, checkpoints, probes, analysis
outputs, and any explainability artifacts you recorded.

To inspect those results after training:

```bash
gradglass serve --port 8432
```

If the workspace lives somewhere other than the default `gg_workspace/` near your entrypoint, point the CLI at it with
`GRADGLASS_ROOT=/path/to/workspace gradglass serve --port 8432`.

For live viewing during training, create the run with `monitor=True`.

## CLI

Use the packaged CLI to inspect completed runs in the current workspace:

```bash
gradglass list
gradglass serve --port 8432
gradglass monitor --port 8432
gradglass open
gradglass analyze <run_id>
gradglass stop --port 8432
gradglass stop --all
```

In a dev checkout, `gradglass serve` requires the built frontend bundle under `gradglass/dashboard/dist`.
When startup succeeds it prints the workspace path; when it fails, the CLI now exits with a clear error if the bundle
is missing or the requested port is already in use.

## Release Checks

Use these commands as the baseline `1.0.0` release gate:

```bash
pytest
npm --prefix gradglass/dashboard test
python -m build
```

PyPI publishing is wired for GitHub Releases via Trusted Publisher. The launch repository is
[`Michael-RDev/GradGlass`](https://github.com/Michael-RDev/GradGlass), and the release workflow is designed to publish
only from a published GitHub Release.

Before tagging a release, make sure the final docs/examples/test set is committed and generated workspaces or local
build outputs are not present in the launch branch.

## Documentation

- [Docs index](./docs/index.md)
- [Getting started](./docs/getting-started.md)
- [Python API](./docs/python-api.md)
- [Architecture](./docs/architecture.md)
- [Analysis and data quality](./docs/analysis-and-data.md)
- [Dashboard and API](./docs/dashboard-and-api.md)
- [Examples](./docs/examples.md)
- [Artifact layout](./docs/artifact-layout.md)
- [Releasing](./docs/releasing.md)

## Example Workflows

The repository includes focused examples for:

- Minimal PyTorch tracking
- TensorFlow / Keras callback integration
- Data leakage detection
- Dataset monitoring
- Custom tests with SHAP
- Workspace browsing and server usage
- Full PyTorch observability
- API coverage generation
- Dashboard showcase generation

See [docs/examples.md](./docs/examples.md).

When you run examples from the repo root, they now write into the repo-root `gg_workspace/` by default so the next
step is simply:

```bash
gradglass serve --port 8432
```

The synthetic tooling demos also separate workspace generation from dashboard launch by default:

- `08_api_feature_coverage.py` prints the serve command unless you pass `--serve`
- `09_dashboard_showcase.py` prepares the showcase workspace unless you pass `--serve`

## Workspace Philosophy

By default GradGlass resolves a workspace named `gg_workspace/` near the entrypoint script you launched. Runs are then
stored under `gg_workspace/runs/<run_id>/...`. You can override the root explicitly with `gg.configure(root=...)` or
the `GRADGLASS_ROOT` environment variable.

## Current Scope

GradGlass currently supports:

- PyTorch model watching and manual training-loop logging
- TensorFlow / Keras callback-based integration
- Local dashboard serving with FastAPI + Vite-built frontend assets
- Post-run analysis and custom registered tests
- Dataset monitoring and legacy-style leakage reporting
- SHAP and LIME summary artifact logging

`run.fit()` is intentionally not supported in this release. PyTorch users should keep their own training loop and call
`run.log()`. TensorFlow users should use `run.keras_callback()`.

## License

GradGlass is licensed under the Apache License 2.0. See [LICENSE](./LICENSE).
