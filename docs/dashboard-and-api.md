# Dashboard and API

## Serving model

The dashboard is served by FastAPI from `gradglass/server.py`. If a built frontend bundle exists, the app serves it as
static files.

In a source checkout, build that bundle once with:

```bash
npm --prefix gradglass/dashboard install
npm --prefix gradglass/dashboard run build
```

Recommended entry points:

- `gradglass serve`
- `monitor=True` when creating a run

Advanced entry points:

- `gg.monitor(...)`
- `run.serve(...)`
- `run.open()`

If the dashboard bundle is missing, `gradglass serve` fails clearly instead of serving a partial API-only placeholder.

Startup guardrails:

- the server verifies that `gradglass/dashboard/dist/index.html` exists before it starts
- successful startup prints the workspace root along with the dashboard URL
- explicit ports are checked before bind, and the error includes the conflicting pid/command when available
- `gradglass stop --port <port>` and `gradglass stop --all` only stop verified GradGlass-owned detached servers

## REST endpoints

Current run endpoints include:

- `GET /api/runs`
- `GET /api/compare`
- `GET /api/runs/{run_id}`
- `GET /api/runs/{run_id}/metrics`
- `GET /api/runs/{run_id}/overview`
- `GET /api/runs/{run_id}/alerts`
- `GET /api/runs/{run_id}/checkpoints`
- `GET /api/runs/{run_id}/diff`
- `GET /api/runs/{run_id}/gradients`
- `GET /api/runs/{run_id}/activations`
- `GET /api/runs/{run_id}/distributions`
- `GET /api/runs/{run_id}/saliency`
- `GET /api/runs/{run_id}/embeddings`
- `GET /api/runs/{run_id}/shap`
- `GET /api/runs/{run_id}/lime`
- `GET /api/runs/{run_id}/predictions`
- `GET /api/runs/{run_id}/architecture`
- `GET /api/runs/{run_id}/analysis`
- `GET /api/runs/{run_id}/freeze_code`
- `GET /api/runs/{run_id}/architecture/diff`
- `GET /api/runs/{run_id}/leakage`
- `GET /api/runs/{run_id}/data-monitor`
- `POST /api/runs/{run_id}/architecture/mutate`
- `GET /api/runs/{run_id}/distributed`
- `GET /api/runs/{run_id}/infrastructure`
- `GET /api/runs/{run_id}/eval`

Live streaming:

- `WS /api/runs/{run_id}/stream`

## What powers each view

- Overview: metadata, status, overview snapshot, alert summary
- Training: metric history and checkpoint cadence
- Evaluation: predictions, inferred task type, selected metrics, recommendations
- Data: data-monitor report and leakage output
- Interpretability: saliency payloads, SHAP summaries, LIME samples, embeddings
- Model internals: architecture graph, gradient summaries, weight/activation distributions
- Infrastructure: runtime heartbeat, process health, distributed rank artifacts, telemetry
- Compare: side-by-side run metrics and diffs

## Frontend structure

The dashboard source lives in `gradglass/dashboard/src/`.

High-level areas:

- layout and theme providers
- API client and store
- route controller
- page modules for overview, training, evaluation, analysis, data, interpretability, infrastructure, and comparison
- the shared GradGlass badge asset used by the README, dashboard header, and browser tab icon

## Operational notes

- Most API endpoints degrade gracefully when an optional artifact is missing.
- Some views become meaningfully richer only if you capture probe batches and checkpoints.
- The websocket stream is useful for live dashboards while training is still active.
- The intended post-run workflow is: run training, then launch `gradglass serve` for that workspace.
- Monitor servers can be stopped with `gradglass stop`, which updates run runtime state as well as terminating the process when possible.
- If you are serving a non-default workspace from the shell, use `GRADGLASS_ROOT=/path/to/workspace gradglass serve`.
- The top-left dashboard logo and browser tab favicon both use the same GradGlass badge shown in the README.
