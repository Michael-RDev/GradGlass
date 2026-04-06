# GradGlass Docs

GradGlass is a local-first observability library for model training runs. These docs are organized so you can start
quickly, then drill into the parts of the system that matter most for your workflow.

## Start here

- [Getting started](./getting-started.md) for installation, workspace setup, and a first tracked run
- [Python API](./python-api.md) for the main `gg` and `Run` interfaces
- [Examples](./examples.md) for guided scripts included in this repository

## Reference guides

- [Architecture](./architecture.md) for how the package is structured internally
- [Analysis and data quality](./analysis-and-data.md) for built-in tests, custom tests, leakage detection, and dataset monitoring
- [Dashboard and API](./dashboard-and-api.md) for the FastAPI endpoints and frontend views
- [Artifact layout](./artifact-layout.md) for the on-disk workspace schema
- [Releasing](./releasing.md) for launch, GitHub Actions, and PyPI publishing steps

## What GradGlass covers

- Training metrics and learning-rate tracking
- Model structure extraction
- Checkpoints and parameter diffs
- Gradient summaries and activation statistics
- Prediction probes, saliency payloads, and embeddings views
- Evaluation summaries and alert generation
- Dataset monitoring and data leakage detection
- Local run browsing through a bundled dashboard

## Public entry points

- `from gradglass import gg`
- `gradglass` CLI
- `gradglass.server:create_app`
- `gradglass.analysis.registry:test`
- `gradglass.analysis.data_monitor.DatasetMonitorBuilder`
- `gradglass.analysis.leakage.LeakageDetector`
