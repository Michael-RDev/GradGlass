# Dashboard Features Summary

> **Scope:** This document reflects the **current React UI** as of **March 12, 2026**.  
> It is based on the routed pages defined in `gradglass/dashboard/src/App.jsx`
> and their corresponding component implementations.  
> See [Current UI vs legacy docs](#current-ui-vs-legacy-docs) for context.

---

## Table of Contents

1. [Cross-Cutting Architecture](#cross-cutting-architecture)
2. [Route: `/` — Home](#route----home)
3. [Route: `/run/:runId/overview` — Overview](#route-runrunidoverview--overview)
4. [Route: `/run/:runId/training` — Training Metrics](#route-runrunidtraining--training-metrics)
5. [Route: `/run/:runId/infrastructure` — Infrastructure Telemetry](#route-runrunidinfrastructure--infrastructure-telemetry)
6. [Route: `/run/:runId/evaluation` — Evaluation & Benchmarks](#route-runrunidevaluation--evaluation--benchmarks)
7. [Route: `/run/:runId/compare` — Compare Experiments](#route-runrunidcompare--compare-experiments)
8. [Route: `/run/:runId/alerts` — System Alerts](#route-runrunidalerts--system-alerts)
9. [Route: `/run/:runId/internals` — Model Architecture Explorer](#route-runrunidinternals--model-architecture-explorer)
10. [Route: `/run/:runId/data` — Dataset & Pipeline Monitor](#route-runruniddata--dataset--pipeline-monitor)
11. [Route: `/run/:runId/interpretability` — Interpretability & Debugging](#route-runrunidinterpretability--interpretability--debugging)
12. [Current UI vs Legacy Docs](#current-ui-vs-legacy-docs)

---

## Cross-Cutting Architecture

### Shared State — `useRunStore`

All run-scoped pages use a single [Zustand](https://github.com/pmndrs/zustand) store defined in
`gradglass/dashboard/src/store/useRunStore.js`. The store holds:

| Field | Type | Description |
|---|---|---|
| `activeRunId` | `string \| null` | The currently active run identifier |
| `metadata` | `object \| null` | Full run metadata from `GET /api/runs/:runId` |
| `metrics` | `array` | All logged scalar metrics from `GET /api/runs/:runId/metrics`, extended by the live WebSocket stream |
| `alerts` | `array` | Alert objects from `GET /api/runs/:runId/alerts` |
| `ws` | `WebSocket \| null` | Live WebSocket connection for streaming metrics |

**Activation:** every run-scoped page calls `setActiveRun(runId)` in a `useEffect`. The store
tears down any existing WebSocket, fetches `metadata`, `metrics`, and `alerts` in parallel, and then
opens a new WebSocket.

**Live metric stream:** `ws://localhost:8432/api/runs/:runId/stream`.  
Messages must be JSON with shape `{ type: "metrics_update", data: [...] }`. Each incoming
`metrics_update` is appended to `metrics`, so charts update live without polling.

### Sidebar & Navigation — `Layout.jsx`

`gradglass/dashboard/src/components/Layout.jsx` renders a persistent 260 px left sidebar and a
64 px top header around every page.

**Top header:**
- GradGlass logo/wordmark (links to `/`)
- Placeholder navigation buttons (Dashboard, Experiments, Models, Datasets) — these are currently
  `<button>` elements with no routing target; they do not navigate anywhere.
- Theme toggle (sun/moon icon)
- Notification bell (no action wired)
- User avatar placeholder (no action wired)

**Sidebar sections (only visible on run pages):**

| Section | Label in sidebar | Route suffix |
|---|---|---|
| Monitor | Dashboard | `/overview` |
| Monitor | Metrics | `/training` |
| Monitor | Infrastructure | `/infrastructure` |
| Analyze | Evaluation | `/evaluation` |
| Analyze | Compare | `/compare` |
| Analyze | Alerts | `/alerts` |
| Deep Dive | Visualizations | `/internals` |
| Deep Dive | Data | `/data` |
| Deep Dive | Interpretability | `/interpretability` |

When not on a run page the sidebar shows "Global Overview" text with no nav items. When on a run
page it displays an "Active Run" chip with the decoded `runId`.

Active state is determined by `location.pathname.includes(item.path)`.

### Theme — `ThemeProvider`

`gradglass/dashboard/src/components/ThemeProvider.jsx` wraps the entire app in a React context.

- **Default:** `dark` (system preference is checked, but dark is the explicit fallback).
- **Persistence:** stored in `localStorage` under the key `"theme"`.
- **Toggle:** the button in the top header calls `toggleTheme()`, which flips between `"dark"` and
  `"light"` and adds/removes the `dark` class on `<html>`.
- **Consumer:** every chart-rendering page reads `const { theme } = useTheme()` to select
  ECharts-specific colors (`textColor`, `gridColor`, `visualMap` palettes).

---

## Route: `/` — Home

**Component:** `gradglass/dashboard/src/pages/Home.jsx`

### Purpose

Entry point. Shows every captured experiment as a clickable card so users can answer:
*"Which runs do I have? Which performed best? What should I open first?"*

### Data Sources

| Source | Details |
|---|---|
| `GET /api/runs` | Returns `{ runs: [...] }`. All data on this page is backend-driven. |

No `useRunStore` is used; state is entirely local (`useState`).

### Key Interactions

- **Sort selector** — three buttons: `Date` (default, newest first by `start_time_epoch`), `Loss`
  (ascending by `latest_loss`), `Storage` (descending by `storage_bytes`).
- **Run cards** — each card is a `<Link to="/run/:runId">` and shows: run name, framework badge,
  status badge (`StatusBadge`), start time, run ID (monospace), step count, latest loss, latest
  accuracy, storage usage.
- Clicking a card navigates to `/run/:runId` which immediately redirects to `/run/:runId/overview`
  via `<Navigate to="overview" replace />`.

### Empty / Error / Loading Behavior

| State | Display |
|---|---|
| Loading | `<LoadingSpinner />` (full page) |
| API error | Centered text: "Could not connect to GradGlass server" with the error message and a hint to check `localhost:8432` |
| No runs | `<EmptyState>` with a `Database` icon and a code hint to start capturing |

### Data Origin

✅ **Fully backend-driven** — all run data is served from the GradGlass artifact store via the local Python API.

---

## Route: `/run/:runId/overview` — Overview

**Component:** `gradglass/dashboard/src/pages/Overview.jsx`

### Purpose

At-a-glance health dashboard for a single run. Answers:
*"Is training healthy right now? What is the current loss / LR / throughput? Are there any active alerts?"*

### Data Sources

| Source | Details |
|---|---|
| `useRunStore` → `metadata` | `GET /api/runs/:runId` — run name, status, start time |
| `useRunStore` → `metrics` | `GET /api/runs/:runId/metrics` + live WebSocket stream |
| `useRunStore` → `alerts` | `GET /api/runs/:runId/alerts` |

### Key Interactions

- **Health banner** — top-right pill switches between "SYSTEM: HEALTHY" (emerald) and
  "SYSTEM: WARNINGS DETECTED" (amber) based on whether `alerts` is non-empty.
- **Alert cards** — if alerts exist, each alert renders with red (high severity) or amber
  (medium) background, title, and message.
- **KPI strip** — shows up to five tiles: Current Step, Time Elapsed (seconds since
  `start_time_epoch`), Throughput (`tokens_per_sec` / `throughput`), Perplexity, Reward.
  Tiles are only rendered if the corresponding metric series is non-empty.
- **Charts (2 × 2 grid):**
  - *Training Performance (Loss)* — `loss` (train) and `val_loss` as smoothed line series.
  - *Learning Rate Schedule* — `lr` on a log-scale y-axis, step-mode line.
  - *Advanced Metrics (LLM/RL)* — overlaid lines for `perplexity`, `reward`/`mean_reward`,
    `kl_divergence`/`kl`. Only rendered when at least one of these metrics is present.
  - *System Throughput* — `tokens_per_sec` or `throughput` as an area chart. Only rendered
    when the metric is present.

### Empty / Error / Loading Behavior

| State | Display |
|---|---|
| `metadata === null` | "Loading run data…" paragraph |
| No `val_loss` | Val Loss series silently omitted from the Loss chart |
| No advanced metrics | Advanced Metrics and Throughput chart cards not rendered |
| `alerts.length === 0` | No alert cards shown; health banner is green |

### Data Origin

✅ **Fully backend-driven.** All KPI values, chart data, and alerts are served from the GradGlass artifact store and (for live runs) extended via the WebSocket stream.

---

## Route: `/run/:runId/training` — Training Metrics

**Component:** `gradglass/dashboard/src/pages/Training.jsx`

### Purpose

Full-resolution interactive scalar explorer. Answers:
*"How did metric X evolve across every step? What does it look like with smoothing? Can I zoom in on a spike?"*

### Data Sources

| Source | Details |
|---|---|
| `useRunStore` → `metrics` | `GET /api/runs/:runId/metrics` + live WebSocket stream |
| `useRunStore` → `discoverMetricKeys()` | Derives the set of all non-`step`/non-`timestamp` keys logged across all metric records |

### Key Interactions

- **Metric selector sidebar (left pane)** — scrollable list of every discovered metric key.
  Each is a toggle button (eye-open / eye-off icon). The first three metrics are pre-selected
  on mount.
- **Smoothing slider (top toolbar)** — exponential moving average weight, range `0.00–0.99`,
  default `0.20`. At weight `> 0` each selected metric renders as two series: a raw faded line
  and a smoothed foreground line.
- **ECharts toolbox** — built-in DataZoom (box-select) and Restore buttons, plus a slider
  scrubber below the chart.
- **DataZoom** — `inside` mode (scroll wheel / pinch) and `slider` mode (drag handles) both
  active simultaneously, scoped to the x-axis.

### Empty / Error / Loading Behavior

| State | Display |
|---|---|
| `availableMetrics.length === 0` | "No scalar metrics found matching this run." paragraph |
| No metrics selected | Centered placeholder text in the chart area |

### Data Origin

✅ **Fully backend-driven** (same `metrics` array as Overview, extended live via WebSocket).

---

## Route: `/run/:runId/infrastructure` — Infrastructure Telemetry

**Component:** `gradglass/dashboard/src/pages/Infrastructure.jsx`

### Purpose

Hardware utilization overview for distributed training. Answers:
*"Are my GPUs loaded? Is memory healthy? Is inter-GPU communication a bottleneck?"*

### Data Sources

| Source | Details |
|---|---|
| `GET /api/runs/:runId/infrastructure` | Returns `{ run_id, mode, metrics, gpu_devices, collected_at }`, where `metrics` includes `cluster_nodes`, `system_cpu`, `system_ram`, `power_draw`, `multi_gpu_compute_utilization`, and `gpu_memory_fragmentation`. |

### Key Interactions

- **KPI tiles** — live Cluster Nodes, System CPU, System RAM, and Power Draw metrics from backend probes.
- **GPU Compute Utilization chart** — rolling live history built from 2s polling; supports no-GPU, single-GPU, and multi-GPU states.
- **GPU Memory Fragmentation chart** — rolling live history when allocator counters are available; otherwise explicit unavailable state.
- **Transparency-first diagnostics** — each metric/section includes a debug disclosure with source, probe function, command/API call, error reason, and last-updated time.

### Empty / Error / Loading Behavior

| State | Display |
|---|---|
| Initial load | "Loading infrastructure telemetry..." |
| Poll failure | Inline error banner plus last successful telemetry snapshot (if present) |
| Metric unavailable/error | Explicit "Unavailable" UI with function and probe command |

### Data Origin

✅ **Backend-driven live telemetry.** Infrastructure values come from runtime probes via
`/api/runs/:runId/infrastructure`; unsupported probes are surfaced as `unavailable`/`error`
with retrieval metadata instead of placeholder numbers.

---

## Route: `/run/:runId/evaluation` — Evaluation & Benchmarks

**Component:** `gradglass/dashboard/src/pages/Evaluation.jsx`

### Purpose

Structured quality metrics and benchmark comparisons. Answers:
*"What is the model's accuracy/F1/MSE over time? Does it pass the confusion matrix? How does it score on LLM or vision benchmarks?"*

### Data Sources

| Source | Details |
|---|---|
| `GET /api/runs/:runId/eval` | Returns `{ evaluations: [...] }` — an array of evaluation snapshots, each containing step, `is_classification` flag, and depending on task type: `accuracy`, `macro_f1`, `macro_precision`, `macro_recall`, or `mse`, `rmse`, `mae`, plus optional `confusion_matrix: { classes, matrix }`. |

LLM and Vision benchmark tabs do **not** call any endpoint.

### Key Interactions

**Three tabs:**

1. **Standard Metrics** (backend-driven)
   - KPI tiles: latest Accuracy, Macro F1, Precision, Recall (classification) **or** MSE, RMSE,
     MAE, Evaluated Steps (regression).
   - Trend chart: Accuracy + Macro F1 over steps (classification) or MSE + MAE over steps
     (regression).
   - Confusion matrix heatmap (classification only, if `confusion_matrix` data is present in
     the latest evaluation snapshot).

2. **LLM Benchmarks** (mock)
   - Grouped bar chart comparing 0-shot vs 5-shot performance across MMLU, GSM8K, HellaSwag,
     TruthfulQA, HumanEval. All values are hardcoded.

3. **Vision Benchmarks** (mock)
   - Grouped bar chart comparing a Baseline vs Current Run across mAP, IoU, Top-1 Acc, Top-5
     Acc. All values are hardcoded.

### Empty / Error / Loading Behavior

| State | Display |
|---|---|
| Loading | "Loading evaluation data…" paragraph |
| `evalData.length === 0` on Standard tab | Inline message with `run.log_batch()` hint |
| No `confusion_matrix` on latest eval | Confusion matrix section not rendered |

### Data Origin

✅ **Standard Metrics tab** — backend-driven via `GET /api/runs/:runId/eval`.  
⚠️ **LLM Benchmarks tab** and **Vision Benchmarks tab** — entirely mock / demo data.

---

## Route: `/run/:runId/compare` — Compare Experiments

**Component:** `gradglass/dashboard/src/pages/Compare.jsx`

### Purpose

Side-by-side multi-run metric overlay. Answers:
*"How does experiment A's loss curve compare to B and C? Which run found the best accuracy?"*

### Data Sources

| Source | Details |
|---|---|
| `GET /api/runs` | Populates the run selector sidebar with all available runs. |
| `GET /api/compare?run_ids=id1,id2,...` | Returns a map of `{ [runId]: { metrics: [...] } }` for the selected runs. Called every time the selection changes. |

No `useRunStore` is used; state is entirely local.

### Key Interactions

- **Run selector sidebar** — text filter input that searches run name and ID. Each run row has a
  checkbox-style toggle; selected rows are highlighted in indigo.
- **Metric dropdown (top toolbar)** — populated from the union of all keys found in
  `compareData`. Defaults to `loss`; falls back to the first available metric if `loss` is not
  present.
- **Overlay chart** — one line per selected run, each assigned a distinct color from a fixed
  palette. ECharts DataZoom (inside + slider) active. Legend labels use the run's human-readable
  name.

### Empty / Error / Loading Behavior

| State | Display |
|---|---|
| No runs selected | Full-height empty state with `GitCompare` icon and instruction text |
| Loading after selection | "Loading comparison data…" text in the chart area |
| No common numeric metrics | "No common numerical metrics found" message |

### Data Origin

✅ **Fully backend-driven** via `GET /api/runs` and `GET /api/compare`.

---

## Route: `/run/:runId/alerts` — System Alerts

**Component:** `gradglass/dashboard/src/pages/Alerts.jsx`

### Purpose

Dedicated alert feed for the active run. Answers:
*"What anomalies has GradGlass detected? How many are critical vs warnings?"*

### Data Sources

| Source | Details |
|---|---|
| `useRunStore` → `alerts` | `GET /api/runs/:runId/alerts` |
| `useRunStore` → `metadata` | `GET /api/runs/:runId` (used for `metadata.status` in the health tile) |

### Key Interactions

- **Summary tiles** — High Severity count, Warnings count, System Health (from `metadata.status`).
- **Alert list** — one card per alert, color-coded by severity:
  - `high` — red flame icon, red background
  - `medium` — amber triangle icon, amber background
  - `low` / default — blue info icon, blue background
- **High severity** alerts include an inline "Recommendation" block with text advising to stop the
  run or revert to a checkpoint.

### Empty / Error / Loading Behavior

| State | Display |
|---|---|
| `metadata === null` | "Loading alerts data…" paragraph |
| `alerts.length === 0` | Full-height green "All Clear" state with `ShieldCheck` icon |

### Data Origin

✅ **Fully backend-driven** via `useRunStore` (which fetches from `GET /api/runs/:runId/alerts`).

---

## Route: `/run/:runId/internals` — Model Architecture Explorer

**Component:** `gradglass/dashboard/src/pages/ModelInternals.jsx`

### Purpose

Layer-level gradient and architecture diagnostics. Answers:
*"Which layers have vanishing or exploding gradients? Are any layers dormant enough to freeze?"*

### Data Sources

| Source | Details |
|---|---|
| `GET /api/runs/:runId/gradients` | Returns `{ summaries: [{ step, layers: { [name]: { norm } } }] }` — one summary per captured step, each with per-layer L2 gradient norms. |
| `GET /api/runs/:runId/freeze_code` | Returns `{ message, candidates: [{ layer, relative_norm }], pytorch_code }` — layers with near-zero gradient activity and auto-generated PyTorch freeze code. |

Both requests are made in `Promise.all`; a failure in either is silently swallowed (`catch(() => null)`).

### Key Interactions

- **KPI tiles** — Tracked Modules (layer count), Latest Step, Est. Parameters
  (`numLayers × 1.5M`, approximate), Ablation Candidates.
- **Architecture Flow pane (left)** — scrollable alphabetical list of all tracked layer names
  from the most recent gradient summary. Hovering a row reveals an eye icon (no action wired
  beyond hover).
- **Global Gradient Stability chart** — horizontal log-scale bar chart of per-layer L2 norms at
  the latest step. Bar color encodes health: orange (`>1e-5` and `≤10` — healthy), slate (`<1e-5`
  — vanishing), red (`>10` — exploding). Chart height scales with layer count (`numLayers × 25px`,
  minimum 500 px).
- **Layer Ablation / Freeze Tools section** — lists each freeze candidate with a relative activity
  bar and percentage. If `pytorch_code` is present, renders a read-only `<pre>` code block.
- **Export Graph button** — renders but has no action wired.

### Empty / Error / Loading Behavior

| State | Display |
|---|---|
| Loading | "Loading architecture data…" paragraph |
| No gradient summaries | "No gradient data found. Run with `gradients='summary'`" paragraph |

### Data Origin

✅ **Fully backend-driven** via `GET /api/runs/:runId/gradients` and `GET /api/runs/:runId/freeze_code`.

---

## Route: `/run/:runId/data` — Dataset & Pipeline Monitor

**Component:** `gradglass/dashboard/src/pages/Data.jsx`

### Purpose

Data health and leakage visibility. Answers:
*"Did any data leak between train and test? What does my dataset look like? What stage is the pipeline at?"*

### Data Sources

| Source | Details |
|---|---|
| `GET /api/runs/:runId/leakage` | Returns the leakage report: `{ passed, num_failed, results: [{ title, description, severity, passed, recommendation, details }] }`. Used for the Leakage Detection Results section only. |

The pipeline stages widget and both charts (modality breakdown, token distribution) do **not**
call any endpoint.

### Key Interactions

- **Leakage status badge** — top-right pill shows "ALL CHECKS PASSED" (emerald) or
  "`n` CHECKS FAILED" (red) derived from the leakage report. Only shown if a report exists.
- **Data Pipeline Stages widget** — horizontal stepper with five nodes: Raw Data → Cleaning →
  Augmentation → Tokenization → Loader. Stage state (complete / active / pending) and sample
  counts are hardcoded.
- **Dataset Composition (Modality) chart** — donut chart with hardcoded values: Text 1048,
  Images 735, Code 580, Audio 300.
- **Token Length Distribution chart** — bar chart with hardcoded bucket counts across six
  sequence-length ranges.
- **Leakage Detection Results** — one card per result from the backend report, showing title,
  description, severity badge, optional recommendation block, and a raw JSON `details` pre-block.
  When no report is found, a dashed empty state is shown with a `run.check_leakage(...)` code hint.

### Empty / Error / Loading Behavior

| State | Display |
|---|---|
| Loading | "Loading data diagnostics…" paragraph |
| No leakage report (`report === null`) | Pipeline and charts still render; Leakage section shows the empty/hint state |

### Data Origin

✅ **Leakage Detection Results section** — backend-driven via `GET /api/runs/:runId/leakage`.  
⚠️ **Data Pipeline Stages widget, Dataset Composition chart, and Token Distribution chart** — entirely mock / demo data.

---

## Route: `/run/:runId/interpretability` — Interpretability & Debugging

**Component:** `gradglass/dashboard/src/pages/Interpretability.jsx`

### Purpose

Black-box inspection tools for attention patterns, feature attribution, and failure analysis.
Answers:
*"Which tokens or pixels drive the model's predictions? Which examples does the model fail on most badly?"*

### Data Sources

| Source | Details |
|---|---|
| *(none)* | This page makes **no API calls** and does not use `useRunStore`. |

### Key Interactions

**Three tabs:**

1. **Attention Maps**
   - 9 × 9 self-attention heatmap using tokens `["The", "quick", "brown", "fox", "jumps",
     "over", "the", "lazy", "dog"]`. Weights are generated with a diagonal-biased formula
     plus `Math.random()`.
   - "Head Controls" side panel with Layer and Attention Head dropdowns (styled but non-functional
     — selecting a different head/layer does not update the chart).
   - Insight callout card describing Head 4's attention pattern (static text).

2. **Feature Attribution**
   - Horizontal bar chart of hardcoded "Global SHAP" values for five named features.
     Positive values are orange, negative values are red.

3. **Worst Predictions**
   - Table of three hardcoded rows (sample ID, input snippet, predicted label, ground truth,
     loss score).

### Empty / Error / Loading Behavior

No loading/error/empty states — the page renders immediately on mount in all scenarios.

### Data Origin

⚠️ **Entirely mock / demo data.** All charts, tables, and controls are hardcoded or use
`Math.random()`. There is no backend integration for interpretability features in the current
build.

---

## Current UI vs Legacy Docs

[docs/dashboard.md](dashboard.md) documents an **older page model** for the GradGlass dashboard.
It describes a different navigation structure (tab-bar with **Story Mode · Eval Lab · Behavior
Explorer · Root Cause Map · Diff Explorer · Architecture**) and mentions pages (Checkpoint Browser,
Gradient Flow, Analysis, Leakage Report, Diff Viewer) that are **not present as routed pages** in
the current `App.jsx`.

The following component files exist in `gradglass/dashboard/src/pages/` but are **not registered
in the router** as of this writing and therefore not reachable via normal navigation:

- `Analysis.jsx`
- `ArchitectureGraph.jsx`
- `BehaviorExplorer/`
- `CheckpointBrowser.jsx`
- `DiffViewer.jsx`
- `EvalLab/`
- `GradientFlow.jsx`
- `LeakageReport.jsx`
- `RootCauseMap/`
- `RunOverview.jsx`
- `StoryMode/`

Use **this document** for the current UI feature set. Refer to [docs/dashboard.md](dashboard.md)
only for historical context about the older navigation model.
