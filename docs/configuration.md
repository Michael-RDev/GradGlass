# Configuration Reference

GradGlass has a small set of configuration options. They can be set at three levels (in increasing priority):

1. **Environment variable** — set once before the Python process starts
2. **`gg.configure()`** — set globally for all runs in the current process
3. **`gg.run()`** argument — set per-run, overrides all of the above

---

## All Options

### `root`

**Type:** `str | Path`
**Default:** `'gg_artifacts'`

Root directory for all artifact storage. GradGlass creates `runs/` inside this directory.

```bash
# Environment variable
export GRADGLASS_ROOT=/data/ml-artifacts
```

```python
# Global
gg.configure(root='/data/ml-artifacts')

# Per-run
run = gg.run('my-exp', root='/data/ml-artifacts')
```

---

### `auto_open`

**Type:** `bool`
**Default:** `False`

Automatically open the browser after every `run.finish()` call. Equivalent to always passing `open=True` to `run.finish()`.

```bash
# Environment variable (any of: 1, true, yes)
export GRADGLASS_OPEN=1
```

```python
# Global
gg.configure(auto_open=True)
```

There is no per-run argument for `auto_open` — use `run.finish(open=True)` instead.

---

### `activations`

**Type:** `str`
**Default:** `'auto'`
**Valid values:** `'auto'` | `'off'`

Controls activation capture for deep-learning runs.

- `'auto'` — forward hooks are registered on every module; activations are buffered and written to `activations/`.
- `'off'` — no activation hooks are attached. Use this to reduce disk usage and I/O overhead for large models.

```python
# Global
gg.configure(activations='off')

# Per-run
run = gg.run('my-exp', activations='off')

# Via watch() (overrides gg.run() value)
run.watch(model, activations='off')
```

Has no effect on sklearn / XGBoost / LightGBM runs (they have no forward hooks).

---

### `gradients`

**Type:** `str`
**Default:** `'summary'`
**Valid values:** `'summary'` | `'off'`

Controls gradient capture for deep-learning runs.

- `'summary'` — per-parameter gradient stats (mean, var, max, norm, KL divergence) are captured every `grad_every` steps and written to `gradients/summaries_step_N.json`.
- `'off'` — no gradient hooks are attached.

```python
# Per-run
run = gg.run('my-exp', gradients='off')

# Via watch()
run.watch(model, optimizer, gradients='off')
```

Has no effect on sklearn / XGBoost / LightGBM runs.

---

### `checkpoint_layers`

**Type:** `str | list[str] | None`
**Default:** `None` (save all layers)

Restrict which layers are included in checkpoints.

- `None` — save all named parameters.
- A list of layer name strings — save only those layers.

```python
# Save only the final classification head
run = gg.run('my-exp', checkpoint_layers=['classifier.weight', 'classifier.bias'])
```

Useful for reducing checkpoint file size in large models when you only care about specific components (e.g. a fine-tuned head).

---

### `activation_batches`

**Type:** `int`
**Default:** `2`

Number of forward-pass batches to accumulate before writing activations to disk. Higher values give more representative activation statistics but use more memory during accumulation.

```python
run = gg.run('my-exp', activation_batches=4)
```

---

### `grad_every`

**Type:** `int`
**Default:** `50`

Number of `run.log()` steps between gradient summary captures. A value of `1` captures gradients on every step; higher values reduce disk writes for long runs.

```python
run = gg.run('my-exp', grad_every=100)
```

---

### `monitor`

**Type:** `bool`
**Default:** `False`

Start the live dashboard server immediately when the run is created (before training begins). The dashboard will update in real time as `run.log()` is called.

```python
run = gg.run('my-exp', monitor=True)
# equivalent to:
run = gg.run('my-exp').watch(model, monitor=True)
# or:
run.monitor(open_browser=True)
```

---

### `port`

**Type:** `int`
**Default:** `0` (pick a free port automatically)

Port for the monitoring or dashboard server started by `monitor=True` or `run.monitor()`.

```python
run = gg.run('my-exp', monitor=True, port=8432)
```

A value of `0` lets the OS pick a free port. The chosen port is printed to stdout and returned by `run.start_server()`.

---

### `task`

**Type:** `str | None`
**Default:** `None`

Optional metadata label describing the ML task. Stored in `metadata.json` and displayed in the dashboard. Has no effect on analysis logic — it is purely informational.

```python
run = gg.run('my-exp', task='nlp/text-classification')
run = gg.run('my-exp', task='time-series/forecasting')
run = gg.run('my-exp', task='anomaly-detection')
```

---

## Quick Reference

| Option | Env Variable | `gg.configure()` | `gg.run()` | Default |
|--------|-------------|------------------|------------|---------|
| `root` | `GRADGLASS_ROOT` | ✅ | ✅ | `'gg_artifacts'` |
| `auto_open` | `GRADGLASS_OPEN` | ✅ | ❌ | `False` |
| `activations` | — | ✅ | ✅ | `'auto'` |
| `gradients` | — | ✅ | ✅ | `'summary'` |
| `checkpoint_layers` | — | — | ✅ | `None` |
| `activation_batches` | — | — | ✅ | `2` |
| `grad_every` | — | — | ✅ | `50` |
| `monitor` | — | — | ✅ | `False` |
| `port` | — | — | ✅ | `0` |
| `task` | — | — | ✅ | `None` |
