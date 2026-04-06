# Artifact Layout

GradGlass stores data under a workspace root, usually `gg_workspace/`.

## Directory structure

```text
gg_workspace/
  runs/
    <run_id>/
      metadata.json
      runtime_state.json
      metrics.jsonl
      model_structure.json
      distributed_index.json
      analysis/
      checkpoints/
      gradients/
      activations/
      predictions/
      probes/
      shap/
      lime/
      rank_*/
```

## Important files

- `metadata.json`
  Static-ish run metadata, captured config, environment details, and run status.
- `runtime_state.json`
  Live state such as heartbeat timestamps, current step, monitor PID/port, and fatal exception markers.
- `metrics.jsonl`
  Append-only metric history used by training, overview, and evaluation views.
- `model_structure.json`
  Extracted architecture graph.
- `analysis/report.json`
  Structured post-run summary.
- `analysis/summary.txt`
  Human-readable analysis summary.
- `analysis/tests.jsonl`
  Flat stream of test results.

## Checkpoints

Saved under `checkpoints/` as:

- `step_<n>.npz`
- `step_<n>_meta.json`

Older pickle-style checkpoints are still readable as a fallback.

## Probe bundles

Probe captures are split into:

- `predictions/probe_step_<n>.json`
- `probes/probe_step_<n>.json`
- `probes/probe_step_<n>.npz`

These power:

- prediction timelines
- evaluation payloads
- saliency views
- activation distributions
- embeddings projections

## Explainability artifacts

- `shap/summary.json`
- `lime/samples.json`

## Distributed and infrastructure artifacts

Optional files such as `distributed_index.json` and `rank_*/heartbeat.json` are used by infrastructure and distributed-training views.

## Why the file layout matters

This layout is the stable handoff point between:

- your training code
- the analysis engine
- the REST API
- the dashboard

Because the system is artifact-first, you can re-open runs, compare them later, or serve them from another process
without rerunning training.
