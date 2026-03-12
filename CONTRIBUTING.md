# Contributing to GradGlass

Thank you for your interest in contributing! This document covers everything you need to get from zero to a merged pull request.

---

## Table of Contents

- [Project Layout](#project-layout)
- [Dev Setup](#dev-setup)
- [Running Tests](#running-tests)
- [Building the Dashboard](#building-the-dashboard)
- [Writing a Built-in Analysis Test](#writing-a-built-in-analysis-test)
- [Code Style](#code-style)
- [PR Conventions](#pr-conventions)

---

## Project Layout

```
gradglass/
  __init__.py        # Public exports: gg, test, TestContext/Result/Category/Severity, LeakageDetector
  core.py            # GradGlass singleton + configure/run/list helpers
  run.py             # Run class — the main user-facing training object
  capture.py         # CaptureEngine (hooks), SklearnCaptureAdapter, XGBoost/LightGBM/Keras callbacks
  artifacts.py       # ArtifactStore — all filesystem I/O
  diff.py            # weight_diff, full_diff, gradient_flow_analysis, activation_diff, arch_diff
  cli.py             # argparse CLI (entry point: `gg`)
  server.py          # FastAPI app — all REST endpoints and the WebSocket stream
  analysis/
    registry.py      # TestRegistry, @test decorator, TestCategory/Severity/Status enums
    builtins.py      # 40+ built-in analysis tests
    leakage.py       # LeakageDetector, LeakageReport, run_leakage_detection
    runner.py        # AnalysisRunner — orchestrates tests against a run's artifacts
    report.py        # PostRunReport — serialisation and terminal output
dashboard/
  src/               # React 18 + Vite + Tailwind — the browser UI
  dist/              # Pre-built production bundle (committed to the repo)
examples/            # Runnable demo scripts for every supported framework
tests/               # pytest test suite
```

---

## Dev Setup

You need **Python ≥ 3.9** and **Node.js ≥ 18** (only if you plan to modify the dashboard).

```bash
# Clone and enter
git clone https://github.com/your-org/gradglass.git
cd gradglass

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install the package in editable mode with all dev dependencies
pip install -e ".[dev]"

# Optional framework extras
pip install -e ".[torch]"
pip install -e ".[tensorflow]"
```

The `[dev]` extra installs: `pytest`, `pytest-asyncio`, `httpx`, and `ruff`.

---

## Running Tests

```bash
# Run the full test suite
pytest

# Run a specific test file
pytest tests/test_core.py

# Run with verbose output
pytest -v

# Run only tests matching a keyword
pytest -k "leakage"
```

Tests live in `tests/`. Each file maps to a subsystem:

| File | Covers |
|------|--------|
| `tests/test_core.py` | `GradGlass` singleton, `Run` creation, `watch`, `log`, `checkpoint`, `finish` |
| `tests/test_capture_and_runner.py` | `CaptureEngine`, `SklearnCaptureAdapter`, `AnalysisRunner` |
| `tests/test_diff_analysis.py` | `weight_diff`, `full_diff`, gradient/activation/arch diff functions |
| `tests/test_interpretability.py` | Feature importance capture, coefficient extraction |
| `tests/data_leakage.py` | All 7 `LeakageDetector` checks |

---

## Building the Dashboard

The React dashboard source lives in `gradglass/dashboard/`. A pre-built `dist/` bundle is committed to the repository so users don't need Node.js to run GradGlass. Rebuild it only if you change the frontend code.

```bash
cd gradglass/dashboard

# Install Node dependencies
npm install

# Development server (proxies API to a local GradGlass server)
npm run dev

# Production build — outputs to gradglass/dashboard/dist/
npm run build
```

After running `npm run build`, commit the updated `dist/` directory along with your frontend changes.

**Dashboard tech stack:** React 18 · React Router v6 · Vite · Tailwind CSS · Recharts · Lucide React

---

## Writing a Built-in Analysis Test

Built-in tests live in `gradglass/analysis/builtins.py`. Every test is a plain Python function decorated with `@test(...)` from `gradglass.analysis.registry`.

### Step-by-step

1. **Choose a category and severity.** See `TestCategory` and `TestSeverity` in `gradglass/analysis/registry.py`.

2. **Pick a unique ID.** Use `SCREAMING_SNAKE_CASE`. Check `builtins.py` to make sure the ID isn't already taken.

3. **Write the function.** It receives a `TestContext` (all artifacts pre-loaded) and returns a `TestResult`.

```python
# gradglass/analysis/builtins.py

from .registry import test, TestContext, TestResult, TestStatus, TestCategory, TestSeverity

@test(
    id='MY_NEW_CHECK',
    title='My new check',
    category=TestCategory.METRICS,
    severity=TestSeverity.HIGH,
    description='One-sentence explanation of what this checks.',
)
def my_new_check(ctx: TestContext) -> TestResult:
    # Pull what you need from ctx
    losses = [m['loss'] for m in ctx.metrics if 'loss' in m]

    if not losses:
        return TestResult(
            id='MY_NEW_CHECK', title='My new check',
            status=TestStatus.SKIP, severity=TestSeverity.HIGH,
            category=TestCategory.METRICS,
            details={'reason': 'No loss values logged'},
        )

    if losses[-1] > 10.0:
        return TestResult(
            id='MY_NEW_CHECK', title='My new check',
            status=TestStatus.FAIL, severity=TestSeverity.HIGH,
            category=TestCategory.METRICS,
            details={'final_loss': losses[-1]},
            recommendation='Reduce the learning rate or check for data issues.',
        )

    return TestResult(
        id='MY_NEW_CHECK', title='My new check',
        status=TestStatus.PASS, severity=TestSeverity.HIGH,
        category=TestCategory.METRICS,
        details={'final_loss': losses[-1]},
    )
```

4. **Write a test.** Add a corresponding case in `tests/test_capture_and_runner.py` (or a new file) to verify your check passes, fails, and skips as expected.

5. **Add it to the catalog in the docs.** Update [docs/analysis.md](docs/analysis.md) with a row for your new test.

### `TestContext` quick reference

| Property | Type | Description |
|----------|------|-------------|
| `run_id` | `str` | The run identifier |
| `metadata` | `dict` | Parsed `metadata.json` |
| `metrics` | `list[dict]` | All rows from `metrics.jsonl` |
| `checkpoints` | `list[dict]` | Checkpoint metadata dicts |
| `model_structure` | `dict` | Parsed `model_structure.json` |
| `gradient_summaries` | `list[dict]` | Per-step gradient stats |
| `activation_stats` | `list[dict]` | Per-layer activation stats |
| `predictions` | `list[dict]` | Prediction probe dicts |
| `has_checkpoints` | `bool` | At least one checkpoint exists |
| `has_gradients` | `bool` | At least one gradient summary exists |
| `has_activations` | `bool` | At least one activation stat exists |
| `has_metrics` | `bool` | At least one metric row exists |
| `framework` | `str` | e.g. `"pytorch"`, `"sklearn"` |
| `is_deep_learning` | `bool` | `True` for PyTorch / Keras runs |
| `is_sklearn` | `bool` | `True` for sklearn runs |

---

## Code Style

GradGlass uses **[ruff](https://docs.astral.sh/ruff/)** for linting and formatting.

```bash
# Check for issues
ruff check .

# Auto-fix safe issues
ruff check --fix .

# Format
ruff format .
```

Configuration is in `pyproject.toml` under `[tool.ruff]`. Please run `ruff check .` and `ruff format .` before opening a PR. The CI pipeline will reject PRs that fail these checks.

Key conventions:
- Type-annotate all public function signatures.
- Keep functions focused — if a function is growing beyond ~60 lines, consider splitting it.
- Prefer explicit over implicit (no `*` imports in library code).
- All new public symbols need a docstring.

---

## PR Conventions

1. **Branch naming:** `feat/<short-description>`, `fix/<short-description>`, `docs/<short-description>`, `test/<short-description>`.

2. **Commit messages:** Use the [Conventional Commits](https://www.conventionalcommits.org/) format:
   ```
   feat: add MY_NEW_CHECK to builtins
   fix: handle missing loss key in LOSS_SPIKE_DETECTION
   docs: add leakage detection thresholds table
   ```

3. **Scope of a PR:** Keep PRs focused on one thing. A PR that adds a new analysis test should not also refactor the dashboard.

4. **Tests are required** for all new features and bug fixes. PRs without tests will not be merged.

5. **Update the docs** if your change affects the public API, CLI, configuration options, or built-in test catalog.

6. **Changelog:** Add a line to `CHANGELOG.md` (if it exists) under the appropriate section (`Added`, `Fixed`, `Changed`).

7. Open a draft PR early if you want feedback before your work is complete.

---

## Questions?

Open a [GitHub Discussion](https://github.com/your-org/gradglass/discussions) for general questions or ideas. Use [Issues](https://github.com/your-org/gradglass/issues) for confirmed bugs and specific feature requests.
