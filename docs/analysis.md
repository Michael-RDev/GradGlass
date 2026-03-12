# Analysis System

GradGlass runs a suite of automated checks against your run's artifacts after training. These checks cover everything from artifact integrity and gradient health to overfitting heuristics and model architecture consistency.

---

## Table of Contents

- [Running Analysis](#running-analysis)
- [Built-in Test Catalog](#built-in-test-catalog)
  - [Artifact & Store Integrity](#artifact--store-integrity)
  - [Model Structure](#model-structure)
  - [Training Metrics](#training-metrics)
  - [Checkpoint Diff](#checkpoint-diff)
  - [Gradient Flow](#gradient-flow)
  - [Activations](#activations)
  - [Predictions](#predictions)
  - [Data](#data)
  - [Reproducibility](#reproducibility)
- [Writing Custom Tests](#writing-custom-tests)
- [`AnalysisRunner` Internals](#analysisrunner-internals)
- [`PostRunReport`](#postrunreport)

---

## Running Analysis

### After training (recommended)

```python
run.finish(analyze=True)
```

### On demand during a run

```python
report = run.analyze()
report.print_summary()
```

### On an existing run from Python

```python
report = gg.analyze('my-experiment-20260309194202-474f8f')
```

### From the terminal

```bash
gradglass analyze my-experiment-20260309194202-474f8f
gradglass analyze           # analyze the most recent run
gradglass analyze --open    # analyze and open the dashboard
```

### Running a subset of tests

Pass a list of test IDs to skip the rest:

```python
report = run.analyze(tests=['LOSS_FINITE', 'OVERFITTING_HEURISTIC', 'GRAD_VANISHING'])
```

---

## Built-in Test Catalog

### Artifact & Store Integrity

These tests verify that the run's saved files are readable and structurally valid.

| ID | Severity | Description | FAIL / WARN Condition |
|----|----------|-------------|----------------------|
| `STORE_LAYOUT_VALID` | CRITICAL | All required subdirectories exist inside the run directory | Any required subdir is missing |
| `METADATA_VALID_JSON` | CRITICAL | `metadata.json` parses without error and contains all required fields | Parse failure or missing required field |
| `CHECKPOINT_READABLE` | HIGH | Every saved checkpoint file can be loaded without error | Any checkpoint fails to load |
| `CHECKPOINT_SHAPE_CONSISTENCY` | HIGH | Parameter tensor shapes are stable across all checkpoints | Any parameter changes shape between checkpoints |
| `CHECKPOINT_PARAM_COUNT_STABLE` | MEDIUM | Total parameter count is the same across all checkpoints | Param count changes between any two checkpoints |
| `ARTIFACT_SIZE_BUDGET` | LOW | Total run storage is within a reasonable limit | Total storage exceeds 500 MB (WARN) |
| `DUPLICATE_ARTIFACT_KEYS` | MEDIUM | No two checkpoints share the same step number | Duplicate step number found in checkpoint metadata |

---

### Model Structure

These tests check the saved architecture graph for validity and consistency.

| ID | Severity | Description | FAIL / WARN Condition |
|----|----------|-------------|----------------------|
| `ARCH_SERIALIZATION_VALID` | HIGH | `model_structure.json` parses correctly and has a valid `layers` list | Parse failure or missing `layers` key |
| `GRAPH_DAG_VALID` | MEDIUM | The layer graph has no cycles (verified via Kahn's algorithm) | A cycle is detected in the layer graph |
| `LAYER_NAME_UNIQUENESS` | MEDIUM | No two layers share the same ID | Duplicate layer ID found |
| `TRAINABLE_FROZEN_CONSISTENCY` | HIGH | Frozen layers did not change weight values between the first and last checkpoint | A frozen layer's weights changed |
| `PARAM_INIT_SANITY` | MEDIUM | Initial weights are not all-zero, NaN, Inf, or suspiciously large (> 100) | Any of the above conditions found at step 0 |

---

### Training Metrics

These tests analyse the values logged via `run.log()`.

| ID | Severity | Description | FAIL / WARN Condition |
|----|----------|-------------|----------------------|
| `LOSS_FINITE` | CRITICAL | Every logged `loss` value is finite | Any `loss` value is NaN or Inf |
| `LOSS_MONOTONIC_TREND` | MEDIUM | Loss has a downward trend overall | Final 25% of steps has a higher mean loss than the first 25% |
| `LOSS_SPIKE_DETECTION` | HIGH | No single step has an anomalously high loss | Any step's loss exceeds 5× the local window mean |
| `ACC_SANITY` | MEDIUM | All logged accuracy values are in [0, 1] | Any `acc` or `val_acc` value is outside [0, 1] |
| `TRAIN_VAL_GAP` | MEDIUM | Validation loss is not dramatically worse than training loss | `val_loss / train_loss > 10` (FAIL); gap > 50% (WARN) |
| `OVERFITTING_HEURISTIC` | MEDIUM | Validation loss is not rising while training loss is falling | In the last third of training, `val_loss` increases while `train_loss` decreases for a majority of steps |
| `VAL_LOSS_DIVERGENCE` | HIGH | Validation loss is not consistently rising in the second half of training | `val_loss` is rising in ≥ 85% of steps during the second half of training |
| `LEARNING_RATE_LOGGED` | LOW | Learning rate is present in the logged metrics | No `lr` key found in any metric row |

---

### Checkpoint Diff

These tests compare the first and last saved checkpoints to assess how much the model changed.

| ID | Severity | Description | FAIL / WARN Condition |
|----|----------|-------------|----------------------|
| `WEIGHT_DIFF_COMPUTED` | HIGH | The weight diff between the first and last checkpoint succeeds without error | Diff computation fails (e.g. incompatible shapes) |
| `WEIGHT_DIFF_SEVERITY_COUNTS` | MEDIUM | Severity distribution of layer-level diffs is reasonable | More than 50% of layers have CRITICAL diff severity |
| `TOP_CHANGED_LAYERS` | MEDIUM | Informational — reports the top 5 most changed layers by Frobenius norm | N/A (always PASS; details shown in dashboard) |
| `UNCHANGED_LAYER_DETECTION` | HIGH | At least some parameters changed between first and last checkpoint | Any parameters are identical between the first and last checkpoint (possible frozen/dead layer) |
| `EXCESSIVE_UPDATE_RATIO` | HIGH | Weight update ratio is within a healthy range | `‖ΔW‖ / ‖W‖` ratio is excessively high for any layer |

**Diff severity thresholds** (per layer, worst of three metrics wins):

| Metric | LOW | MEDIUM | HIGH | CRITICAL |
|--------|-----|--------|------|----------|
| Frobenius norm of delta | ≤ 0.05 | ≤ 0.15 | ≤ 0.30 | > 0.30 |
| Cosine similarity | ≥ 0.995 | ≥ 0.97 | ≥ 0.90 | < 0.90 |
| % weights changed | ≤ 5% | ≤ 20% | ≤ 50% | > 50% |

---

### Gradient Flow

These tests use the gradient summaries captured every `grad_every` steps.

| ID | Severity | Description | FAIL / WARN Condition |
|----|----------|-------------|----------------------|
| `GRAD_SUMMARY_PRESENT` | MEDIUM | Gradient summary files were captured | No gradient summaries found |
| `GRAD_NAN_INF` | CRITICAL | No NaN or Inf values in any gradient summary | Any layer has NaN or Inf gradient values |
| `GRAD_VANISHING` | HIGH | No layer has consistently near-zero gradient norms | Any layer’s mean gradient norm is < 1e-7 across summaries |
| `GRAD_EXPLODING` | HIGH | No layer has catastrophically large gradient norms | Any layer’s gradient norm exceeds the explosion threshold |
| `GRAD_LAYER_IMBALANCE` | MEDIUM | Gradient magnitudes are balanced across layers | Extreme imbalance between the max and min per-layer gradient norm |
| `GRAD_CLIP_EFFECTIVENESS` | LOW | Gradient clipping keeps norms within bounds (if enabled) | Norms exceed the clip threshold after clipping is applied |
| `GRAD_INPUT_SALIENCY` | MEDIUM | Gradient×Input saliency is computable from captured data | Saliency computation fails or returns degenerate values |
| `SHAP_GRAD_ATTRIBUTION_RANK` | MEDIUM | Gradient attribution ranking is stable (gradient-based proxy) | Attribution ranks are unstable across probes |
| `FREEZE_RECOMMENDATION` | LOW | Informational — layers with low gradient activity are identified as freeze candidates | N/A (always PASS; recommendations shown in dashboard) |

---

### Activations

These tests analyse the activation distributions captured via forward hooks.

| ID | Severity | Description | FAIL / WARN Condition |
|----|----------|-------------|----------------------|
| `ACT_NAN_INF` | CRITICAL | No NaN or Inf values in any activation statistic | Any layer has NaN or Inf activation mean or variance |
| `ACT_SPARSITY_COLLAPSE` | HIGH | Activation sparsity is healthy | Any layer has sparsity > 90% (potential dead ReLUs) |
| `DEAD_NEURON_RATE` | HIGH | Dead neuron rate is low | Estimated dead neuron percentage is too high for any layer |
| `ACT_DISTRIBUTION_DRIFT` | MEDIUM | Activation distributions are stable over training | Large shifts in activation mean or variance between observations |
| `SATURATION_DETECTION` | MEDIUM | No activation saturation detected | Signs of tanh/sigmoid saturation in any layer |
| `REPRESENTATION_COLLAPSE` | HIGH | No representation collapse | Activations are near-constant across samples (all embeddings identical) |
| `DEAD_CHANNEL_DETECTION` | HIGH | No dead feature channels in convolutional layers | Any convolutional channel has near-zero activations throughout training |
| `ACTIVATION_PATTERN_STABILITY` | MEDIUM | Activation patterns stabilise over training | Activation patterns are still highly variable late in training |
| `LAYER_CAPACITY_UTILIZATION` | MEDIUM | Layers are utilising their representational capacity | Any layer’s effective rank is too low relative to its output dimensionality |

---

### Predictions

These tests check the prediction probes saved via `run.log_batch()`.

| ID | Severity | Description | FAIL / WARN Condition |
|----|----------|-------------|----------------------|
| `PRED_FINITE` | CRITICAL | All prediction probe outputs are finite | Any prediction value is NaN or Inf |
| `LABEL_FLIP_RATE` | MEDIUM | Low rate of prediction label flips between consecutive probes | Label flip rate between consecutive steps is high |
| `CONFIDENCE_SHIFT` | LOW | Informational — tracks whether mean confidence improved over training | N/A (always PASS; trend shown in dashboard) |
| `TOP_CHANGED_SAMPLES` | LOW | Informational — identifies samples with the largest prediction delta | N/A (always PASS; details shown in dashboard) |
| `LIME_PROXY_CONFIDENCE` | MEDIUM | Prediction confidence varies meaningfully across samples (gradient proxy) | Confidence is near-identical for all samples (model is not discriminating) |

---

### Data

These tests check for data integrity issues separate from the full leakage suite.

| ID | Severity | Description | FAIL / WARN Condition |
|----|----------|-------------|----------------------|
| `DATA_HASH_STABILITY` | LOW | A dataset hash/signature was logged for this run | No `dataset_hash` field found in `metadata.json` |
| `CLASS_IMBALANCE_CHECK` | LOW | Class distribution in prediction probes is balanced | Severe class imbalance detected in logged prediction labels |
| `SLICE_COVERAGE` | LOW | Dataset slice artifacts exist and contain enough samples | Slice directory is empty or individual slices have too few samples |

---

### Reproducibility

| ID | Severity | Description | FAIL / WARN Condition |
|----|----------|-------------|----------------------|
| `SEED_LOGGED` | LOW | A random seed was captured for this run | No `seed` field found in `metadata.json` |
| `ENV_CAPTURED` | LOW | Python and framework version info was captured | No `environment` field found in `metadata.json` |
| `DETERMINISM_FLAGS_LOGGED` | LOW | Deterministic-training flags are set | No `deterministic` flag found in `metadata.json` |
| `RUN_GIT_COMMIT_CAPTURED` | LOW | A git commit hash was captured at run creation | No `git_commit` field in `metadata.json` |

---

## Writing Custom Tests

See the [Python API — Custom Analysis Tests](python-api.md#custom-analysis-tests--test) section for the full guide and `TestContext` reference.

Quick template:

```python
from gradglass import gg, test, TestContext, TestResult, TestCategory, TestSeverity, TestStatus

@test(
    id='MY_UNIQUE_ID',
    title='Short human-readable title',
    category=TestCategory.METRICS,   # choose the most relevant category
    severity=TestSeverity.HIGH,
    description='One sentence explaining what this check does.',
)
def my_check(ctx: TestContext) -> TestResult:
    # 1. Guard: skip if required data is absent
    if not ctx.has_metrics:
        return TestResult(id='MY_UNIQUE_ID', title='...', status=TestStatus.SKIP,
                          severity=TestSeverity.HIGH, category=TestCategory.METRICS,
                          details={'reason': 'No metrics logged'})

    # 2. Evaluate
    ...

    # 3. Return PASS, WARN, or FAIL
    return TestResult(id='MY_UNIQUE_ID', title='...', status=TestStatus.PASS,
                      severity=TestSeverity.HIGH, category=TestCategory.METRICS,
                      details={'value': ...})
```

Custom tests are registered globally and run on every subsequent `run.analyze()` call in the same Python process.

---

## `AnalysisRunner` Internals

```python
from gradglass.analysis.runner import AnalysisRunner
from gradglass.artifacts import ArtifactStore

store = ArtifactStore()
runner = AnalysisRunner(run_id='my-run-id', store=store)
```

| Method | Description |
|--------|-------------|
| `runner.build_context()` | Load all artifacts from the store into a `TestContext` |
| `runner.run(tests=None)` | Execute all registered tests (or a subset by ID). Catches exceptions and converts them to FAIL results. Returns `list[TestResult]`. |
| `runner.summarize(results)` | Produce a dict with four sections: `passed`, `warned`, `failed`, `skipped` |
| `runner.print_report(results)` | Print a human-readable summary to stdout |

---

## `PostRunReport`

`PostRunReport` is the serialisable container for a completed analysis.

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `run_id` | `str` | Run identifier |
| `timestamp` | `str` | ISO timestamp of when analysis ran |
| `results` | `list[TestResult]` | All test results |
| `passed` | `int` | Number of PASS results |
| `warned` | `int` | Number of WARN results |
| `failed` | `int` | Number of FAIL results |
| `skipped` | `int` | Number of SKIP results |

### Methods

| Method | Description |
|--------|-------------|
| `report.print_summary()` | Print a human-readable terminal report grouped by category |
| `report.to_dict()` | Serialise to a JSON-compatible dict |
| `PostRunReport.from_dict(d)` | Deserialise from a dict |
| `PostRunReport.generate(run_id, store)` | Full pipeline: build context → run tests → save `analysis/report.json`, `analysis/summary.txt`, and `analysis/tests.jsonl` → return the report |

### Saved files

After analysis runs, three files are written inside `analysis/`:

| File | Format | Description |
|------|--------|-------------|
| `report.json` | JSON | Full `PostRunReport` serialised |
| `summary.txt` | Plain text | Human-readable terminal-style report |
| `tests.jsonl` | JSON Lines | One `TestResult` per line with timestamp |
