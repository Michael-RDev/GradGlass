# CLI Reference

GradGlass ships with a `gradglass` command-line tool. After `pip install gradglass` it is available on your `PATH`.

```bash
gradglass --help
```

---

## Commands

- [`gradglass serve`](#gradglass-serve) — Start the dashboard server
- [`gradglass ls`](#gradglass-ls) — List all runs
- [`gradglass open`](#gradglass-open) — Open a run in the browser
- [`gradglass monitor`](#gradglass-monitor) — Live monitoring during training
- [`gradglass analyze`](#gradglass-analyze) — Run post-training analysis

---

## `gradglass serve`

Start the GradGlass dashboard server and open it in your default browser.

```bash
gradglass serve
gradglass serve --port 9000
gradglass serve --no-browser
```

| Flag | Default | Description |
|------|---------|-------------|
| `--port PORT` | `8432` | Port to listen on |
| `--no-browser` | off | Start the server without opening the browser |

The server runs at `http://localhost:{port}/` and serves all runs found in the local `gg_artifacts/` directory. The command blocks until you press Ctrl+C.

**Examples:**

```bash
# Start on the default port, open browser automatically
gradglass serve

# Start on a custom port without opening the browser
gradglass serve --port 9000 --no-browser
```

---

## `gradglass ls`

Print a table of all captured runs.

```bash
gradglass ls
```

**Output columns:** Name · Framework · Steps · Latest Loss · Status · Storage

**Example output:**

```
NAME                            FRAMEWORK   STEPS   LOSS     STATUS     STORAGE
mnist-cnn-20260309194202        pytorch     1200    0.041    complete   14.2 MB
random-forest-20260309201500    sklearn     1       0.121    complete    2.1 MB
xgb-experiment-20260309203000   xgboost     300     0.083    complete    1.4 MB
```

No flags. Reads runs from the `gg_artifacts/` directory in the current working directory.

---

## `gradglass open`

Open a specific run (or the most recent run) in the browser dashboard.

```bash
gradglass open
gradglass open my-experiment-20260309194202-474f8f
```

| Argument | Description |
|----------|-------------|
| `[run_id]` | Optional. Run ID to open. Defaults to the most recently created run (sorted by `start_time`). |

Starts the server on port `8432` if it is not already running, then opens `http://localhost:8432/?run={run_id}` in the default browser.

**Examples:**

```bash
# Open the most recent run
gradglass open

# Open a specific run by ID
gradglass open mnist-cnn-20260309194202-474f8f
```

---

## `gradglass monitor`

Start the live monitoring dashboard. Identical to `gradglass serve` but intended for use *during* training — the dashboard will update in real time as your training script calls `run.log()`.

```bash
gradglass monitor
gradglass monitor --port 9000
gradglass monitor --no-browser
```

| Flag | Default | Description |
|------|---------|-------------|
| `--port PORT` | `8432` | Port to listen on |
| `--no-browser` | off | Start the server without opening the browser |

New metric values are pushed to connected browser clients every second over a WebSocket connection. You do not need to refresh the page.

**Typical workflow:**

```bash
# Terminal 1 — start the monitor before training begins
gradglass monitor

# Terminal 2 — run your training script
python train.py
```

Or start it from Python inside your training script:

```python
run.monitor(open_browser=True)
```

---

## `gradglass analyze`

Run the built-in analysis suite against a run and print a summary.

```bash
gradglass analyze
gradglass analyze my-experiment-20260309194202-474f8f
gradglass analyze --tests LOSS_FINITE,OVERFITTING_HEURISTIC
gradglass analyze --open
```

| Argument / Flag | Default | Description |
|-----------------|---------|-------------|
| `[run_id]` | most recent run | Run ID to analyse |
| `--tests ID1,ID2,...` | all tests | Comma-separated list of test IDs to run (skips all others) |
| `--open` | off | Open the dashboard after analysis completes |

Results are written to `analysis/report.json`, `analysis/summary.txt`, and `analysis/tests.jsonl` inside the run directory. If those files already exist they will be overwritten.

**Examples:**

```bash
# Analyze the most recent run
gradglass analyze

# Analyze a specific run and open the dashboard when done
gradglass analyze mnist-cnn-20260309194202-474f8f --open

# Run only gradient-related tests
gradglass analyze --tests GRAD_VANISHING,GRAD_EXPLODING,DEAD_NEURON_RATE
```
