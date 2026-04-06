# Analysis and Data Quality

## Post-run analysis

`run.analyze()` calls `PostRunReport.generate(...)`, which:

1. builds a `TestContext`
2. runs the registered test suite
3. computes summary sections for checkpoints, gradients, metrics, and storage
4. writes:
   - `analysis/report.json`
   - `analysis/summary.txt`
   - `analysis/tests.jsonl`

## Built-in analysis categories

The registry organizes checks into categories such as:

- Artifact and store integrity
- Model structure
- Training metrics
- Checkpoint diffs
- Gradient flow
- Activations
- Predictions
- Data
- Distributed training
- Reproducibility

The exact enum values live in `gradglass/analysis/registry.py`.

## Custom tests

Custom tests use the same registry as built-ins. Each test receives a `TestContext` with access to:

- metadata
- metrics
- checkpoint metadata
- architecture
- gradient summaries
- activation stats
- predictions
- distributed metadata

That means your checks can stay file-format aware without manually reloading artifacts.

## Alerts

The alerts system combines several signal sources:

- runtime status and heartbeat state
- persisted analysis report failures and warnings
- live built-in heuristics from metrics and gradients
- grouped gradient anomaly flags

The result is served through `/api/runs/{run_id}/alerts` and summarized into a dashboard-friendly payload.

## Leakage detection

Leakage detection now sits on top of the dataset monitor engine.

Paths:

- standalone: `LeakageDetector(...).run_all()`
- convenience function: `run_leakage_detection(...)`
- run scoped: `run.check_leakage(...)`

Typical checks include:

- exact overlap between train and test
- duplicates within a split
- near-duplicate sample detection
- label-distribution consistency
- feature-statistics consistency
- preprocessing leakage heuristics

## Dataset monitoring

`DatasetMonitorBuilder` records snapshots across pipeline stages such as ingest, splitting, preprocessing, and
training-ready datasets.

It uses:

- adapters to normalize inputs from arrays, loaders, datasets, and columnar dictionaries
- inspectors to infer modality and sample-level characteristics
- fingerprinting to compare exact and normalized representations
- analyzers to build checks, compositions, comparisons, and recommendations

Outputs include:

- structured report JSON
- human-readable summary text
- dashboard-oriented view models

## When to use which system

- Use `run.analyze()` for model-run diagnostics after or during training.
- Use `DatasetMonitorBuilder` when you want to inspect data evolution through a pipeline.
- Use `check_leakage()` when your immediate question is train/test contamination risk.
- Use custom tests when your team has domain-specific definitions of “healthy training.”
