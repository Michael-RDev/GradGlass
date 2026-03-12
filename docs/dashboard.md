# Dashboard

The GradGlass dashboard is a local React web app that lets you explore, compare, and debug your runs. It runs entirely on `localhost` — no data ever leaves your machine.

---

## Launching the Dashboard

```python
# From Python — open a specific run
run.open()

# From Python — open the most recent run
gg.open_last()
```

```bash
# From the terminal — open the most recent run
gradglass open

# From the terminal — start the server (browse all runs)
gradglass serve
```

The server starts on port `8432` by default and serves `http://localhost:8432/`.

---

## Navigation

The top navigation bar has two levels:

**Root level** (when no run is selected):
- **Runs** — the home page listing all captured runs

**Run level** (when viewing a specific run):
- Breadcrumb showing the current run name
- Tab bar: **Story Mode · Eval Lab · Behavior Explorer · Root Cause Map · Diff Explorer · Architecture**

Additional pages are accessible via direct URL but not surfaced in the top nav: Gradient Flow, Checkpoint Browser, Analysis, and Leakage Report.

---

## Pages

### Home (`/`)

A card list of all runs in your artifact store.

**What it shows:**
- Run name and framework badge (PyTorch, Keras, sklearn, XGBoost, LightGBM)
- Run status (Running / Complete / Failed)
- Total steps logged
- Latest loss and accuracy
- Storage size on disk

**How to use it:**
- Click any card to jump to that run's **Story Mode** page.
- Runs can be sorted by date, loss, or storage size.

---

### Story Mode (`/runs/{runId}/story`)

The top-level health summary for a run. **This is the best page to start with.**

**What it shows:**

**Health banner** — one of five states based on analysis results:
| Banner | Meaning |
|--------|---------|
| ✅ Healthy | All checks passed |
| ⚠️ Overfitting | Overfitting heuristics triggered |
| 🔥 Unstable Gradients | Vanishing, exploding, or dead gradient flags raised |
| ⚠️ Issues Detected | Some tests failed or warned |
| ❌ Failed | Critical failures found |

**Loss / metric chart** — automatically plots up to 4 relevant metric keys from `metrics.jsonl` (e.g. `loss`, `val_loss`, `acc`, `val_acc`) over training steps.

**Run metadata panel** — framework, task label, model parameter count, training config (from `metadata.json`), and environment info (Python/framework versions, GPU name).

**Epoch-level stats** — if `epoch` is logged, shows per-epoch aggregates.

---

### Eval Lab (`/runs/{runId}/eval`)

Evaluation metrics computed from the saved prediction probes.

**Classification runs show:**
- **Confusion matrix** — colour-coded cells (green diagonal = correct, red off-diagonal = errors). Click a cell to filter the Behavior Explorer to that true/predicted class pair.
- **Per-class table** — Precision, Recall, F1 for each class.
- **Macro F1 / Precision / Recall** — overall aggregate metrics.

**Regression runs show:**
- MSE, RMSE, and MAE computed from the most recent prediction probe.

**Step selector** — browse eval snapshots across training to see how predictions evolved.

**How to interpret it:** A confusion matrix with a strong diagonal and low off-diagonal values indicates the model is discriminating well. A row with many errors concentrated in one column suggests the model is confusing that true class with one particular predicted class.

---

### Behavior Explorer (`/runs/{runId}/behavior`)

A per-sample table of predictions from the saved probes.

**What it shows:**
- True label · Predicted label · Confidence score · Correct / Error status
- Confidence shift between consecutive steps (e.g. "confidence dropped 0.23")

**Filters:**
- **All** — show every sample
- **Errors** — show only misclassified samples
- **Label Flips** — show samples where the predicted class changed between two consecutive steps

**How to use it:** Filter to **Errors** to find systematically misclassified samples. Filter to **Label Flips** to find unstable predictions that flip classes during training — a sign of training instability or an overly aggressive learning rate.

---

### Root Cause Map (`/runs/{runId}/rootcause`)

A three-column diagnostic layout linking symptoms to their likely causes.

**Columns:**

1. **Symptoms** — gradient health flags surfaced from the analysis (VANISHING, EXPLODING, NOISY, DEAD, DISTRIBUTION_SHIFT). Each flag shows which layers are affected.

2. **Affected Slices** — reserved for future slice-based analysis.

3. **Layer Suspects** — reserved for future layer-level attribution.

**How to use it:** Start here when your model is not converging. The Symptoms column maps directly to the affected layers so you know exactly where to look.

---

### Diff Explorer (`/runs/{runId}/diff`)

Weight-level comparison between two checkpoints.

**What it shows:**
- A dropdown to select **Checkpoint A** and **Checkpoint B**
- Per-layer diff table: Frobenius norm of delta, cosine similarity, % weights changed, severity badge
- Summary counts: how many layers are LOW / MEDIUM / HIGH / CRITICAL

**Severity thresholds:**

| Metric | LOW | MEDIUM | HIGH | CRITICAL |
|--------|-----|--------|------|----------|
| Frobenius norm | ≤ 0.05 | ≤ 0.15 | ≤ 0.30 | > 0.30 |
| Cosine similarity | ≥ 0.995 | ≥ 0.97 | ≥ 0.90 | < 0.90 |
| % weights changed | ≤ 5% | ≤ 20% | ≤ 50% | > 50% |

The worst of the three metrics determines a layer's severity.

**How to use it:** Layers with CRITICAL severity changed dramatically — if those are layers you expected to be stable (e.g. frozen layers in a transfer learning setup), that is a bug. Layers with LOW severity barely moved — if those are layers you expected to train, the learning rate for that layer may be too low.

---

### Architecture (`/runs/{runId}/architecture`)

An interactive DAG of the model's layer graph.

**What it shows:**
- Each node is a layer, labelled with its name, type, output shape, and parameter count
- Directed edges show data flow between layers
- Frozen layers are visually distinguished
- Clicking a node shows full layer details

**How to use it:** Verify that the model's actual architecture matches your design intention. Useful for catching subtle bugs like unintentionally disconnected layers or wrong input shapes.

---

### Gradient Flow (`/runs/{runId}/gradients`)

*(Accessible at `/runs/{runId}/gradients` — not in the main tab bar)*

Per-layer gradient history over training steps.

**What it shows:**
- A chart of gradient norm over time for each layer
- Health flags detected per layer: VANISHING · EXPLODING · NOISY · DISTRIBUTION_SHIFT · DEAD

**Flag meanings:**

| Flag | Meaning |
|------|---------|
| VANISHING | Gradient norm < 1e-7 — the layer is not learning |
| EXPLODING | Gradient norm > 1000 — numerically unstable |
| NOISY | Signal-to-noise ratio < 0.1 — gradients are dominated by noise |
| DISTRIBUTION_SHIFT | KL divergence between consecutive summaries > 0.5 |
| DEAD | Gradient is exactly 0 for ≥ 5 consecutive summaries — dead neurons |

**How to use it:** Layers flagged as VANISHING in the early layers of a deep network are a classic symptom of the vanishing gradient problem — consider gradient clipping, residual connections, or a different activation function. DEAD flags in layers after ReLU suggest ReLU saturation — consider switching to Leaky ReLU or ELU.

---

### Checkpoint Browser (`/runs/{runId}/checkpoints`)

*(Accessible at `/runs/{runId}/checkpoints` — not in the main tab bar)*

A list of all saved checkpoints.

**What it shows:**
- Step number, tag (e.g. `epoch_2`), timestamp, total parameter count, file size

**How to use it:** Navigate here to pick checkpoint steps to compare in the **Diff Explorer**. Useful for verifying that auto-checkpointing fired at the expected intervals.

---

### Analysis (`/runs/{runId}/analysis`)

*(Accessible at `/runs/{runId}/analysis` — not in the main tab bar)*

The full results of the built-in analysis suite, grouped by category.

**What it shows:**
- Each test shows: status icon (✅ PASS / ⚠️ WARN / ❌ FAIL / — SKIP), severity badge, title, details dict, and recommendation text
- Tests grouped by `TestCategory`
- Pass / Warn / Fail / Skip counts in the header

**Freeze Layer Weights panel:**
- Queries the server for low-gradient layers
- Lists layers that contributed little gradient signal during training
- Shows copy-pasteable code for freezing those layers in PyTorch or Keras

**How to use it:** Read FAIL and WARN results with their recommendations to triage issues. Use the freeze panel to build leaner fine-tuned models by freezing layers that are already well-trained.

---

### Leakage Report (`/runs/{runId}/leakage`)

*(Accessible at `/runs/{runId}/leakage` — not in the main tab bar)*

Results of the 7 data leakage checks.

**What it shows:**
- One card per check: status badge, severity badge, check title, details dict, and recommendation
- A summary header showing whether any CRITICAL or FAIL-status checks were found

**How to use it:** A FAIL on `EXACT_OVERLAP` means train samples appear verbatim in your test set — your evaluation metrics are meaningless and must be recomputed on a clean split. A FAIL on `TARGET_CORRELATION` means one or more features directly encode the label — the model is learning from future information it would not have in production.

See [Data Leakage Detection](leakage.md) for a full description of every check, its thresholds, and remediation advice.
