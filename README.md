<p align="center">
  <br/>
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/gradglass-dark.svg" width="480" style="max-width: 100%;">
    <source media="(prefers-color-scheme: light)" srcset="docs/gradglass-light.svg" width="480" style="max-width: 100%;">
    <img alt="GradGlass — ML Transparency Engine" src="docs/gradglass-light.svg" width="480" style="max-width: 100%;">
  </picture>
  <br/>
</p>

<p align="center">
  <a href="https://pypi.org/project/gradglass"><img alt="PyPI" src="https://img.shields.io/pypi/v/gradglass?color=6366f1&logo=python&logoColor=white"></a>
  <a href="https://pypi.org/project/gradglass"><img alt="Python" src="https://img.shields.io/pypi/pyversions/gradglass?color=6366f1&logo=python&logoColor=white"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-6366f1"></a>
  <img alt="Telemetry: None" src="https://img.shields.io/badge/telemetry-none-22c55e">
  <img alt="100% Local" src="https://img.shields.io/badge/100%25-local-22c55e">
</p>

<h3 align="center">
  <p>ML Explaniability — Understand Training. Diagnose Models. Ship Faster</p>
</h3>

<p align="center">
  Inspect, compare, debug, and analyse your models without leaving your local machine.<br/>
  Works with <strong>PyTorch</strong>, <strong>Keras&nbsp;/&nbsp;TensorFlow</strong>, <strong>scikit-learn</strong>, <strong>XGBoost</strong>, <strong>LightGBM</strong>, and anything else you can train.
</p>

---

## ✨ Features

| | |
|---|---|
| 🔍 **Checkpoint Diffing** | Git-style diffs for neural network weights at the tensor level |
| 📈 **Gradient Flow Analysis** | Detect vanishing gradients, neuron death, and NaN propagation |
| 🏗️ **Architecture Visualization** | Interactive DAG rendering with layer icons and shape annotations |
| 🌲 **Feature Importance Capture** | Auto-extracts importances, coefficients, and diagnostics for tree models |
| 📡 **Per-Round Metric Streaming** | XGBoost / LightGBM callbacks stream eval metrics in real time |
| 🕵️ **Data Leakage Detection** | Statistical checks across train / test splits |
| 🖥️ **Distributed Training Monitor** | Per-rank diagnostics across GPUs and nodes |
| 🔒 **100% Local** | No cloud, no telemetry , all data stays on your machine |

---

## 🚀 Quick Start

```bash
pip install gradglass
```

---

## 🔬 Framework Examples

<details open>
<summary><b>PyTorch</b></summary>

```python
from gradglass import gg

run = gg.run('resnet-exp1').watch(model, optimizer)

for epoch in range(num_epochs):
    for batch in train_loader:
        loss, acc = train_step(batch)
        run.log(loss=float(loss), acc=float(acc))
    run.checkpoint(tag=f'epoch_{epoch}')

run.open()  # launches dashboard in browser
```
</details>

<details>
<summary><b>Keras / TensorFlow</b></summary>

```python
from gradglass import gg

run = gg.run('lstm-experiment').watch(model)
model.fit(dataset, epochs=10, callbacks=[run.keras_callback()])
run.open()
```
</details>

<details>
<summary><b>scikit-learn</b></summary>

GradGlass auto-detects any `sklearn` estimator or `Pipeline`:

```python
from gradglass import gg
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Single estimator
clf = RandomForestClassifier(n_estimators=200, oob_score=True)
run = gg.run('random-forest').watch(clf)
run.fit(X_train, y_train, X_val=X_test, y_val=y_test)
run.log(test_f1=f1_score(y_test, clf.predict(X_test), average='macro'))
run.finish()

# Pipeline , GradGlass records every step's hyperparameters
pipe = Pipeline([('scaler', StandardScaler()),
                 ('clf',    LogisticRegression(max_iter=500))])
run2 = gg.run('lr-pipeline').watch(pipe)
run2.fit(X_train, y_train, X_val=X_test, y_val=y_test)
run2.finish()
```

What GradGlass auto-captures for sklearn:

| Attribute | Auto-captured |
|---|---|
| All hyperparameters | ✅ via `get_params()` |
| Feature importances | ✅ `feature_importances_` |
| Coefficients | ✅ `coef_` / `intercept_` |
| Training loss curve | ✅ `loss_curve_` (MLP etc.) |
| OOB score | ✅ `oob_score_` |
| Inertia | ✅ KMeans / clustering |
| Explained variance | ✅ PCA / decomposition |
| Serialised checkpoint | ✅ via joblib |
</details>

<details>
<summary><b>XGBoost</b></summary>

```python
from gradglass import gg
import xgboost as xgb

# Functional API
run = gg.run('xgb-experiment')
cb  = run.xgboost_callback()
booster = xgb.train(params, dtrain, num_boost_round=300,
                    evals=[(dtrain, 'train'), (dval, 'val')],
                    callbacks=[cb])
run.finish()

# Sklearn API
clf = xgb.XGBClassifier(n_estimators=200, max_depth=6)
run2 = gg.run('xgb-sklearn').watch(clf)
run2.fit(X_train, y_train, X_val=X_test, y_val=y_test,
         callbacks=[run2.xgboost_callback()],
         eval_set=[(X_test, y_test)], verbose=False)
run2.finish()
```
</details>

<details>
<summary><b>LightGBM</b></summary>

```python
from gradglass import gg
import lightgbm as lgb

# Functional API
run = gg.run('lgbm-experiment')
gbm = lgb.train(params, train_data, valid_sets=[val_data],
                callbacks=[run.lightgbm_callback()])
run.finish()

# Sklearn API
clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05)
run2 = gg.run('lgbm-sklearn').watch(clf)
run2.fit(X_train, y_train, X_val=X_test, y_val=y_test,
         callbacks=[run2.lightgbm_callback()],
         eval_set=[(X_test, y_test)])
run2.finish()
```
</details>

---

## 🗺️ Use Cases

GradGlass is **not** limited to classification or regression , it is a framework-agnostic transparency layer for any ML / DL task:

| Task | Framework | Example |
|------|-----------|---------|
| Image classification | PyTorch / Keras | [`examples/mnist_demo.py`](examples/mnist_demo.py) |
| Tabular regression | PyTorch | [`examples/regression_demo.py`](examples/regression_demo.py) |
| Gradient boosting | XGBoost / LightGBM | [`examples/xgboost_demo.py`](examples/xgboost_demo.py) |
| Tree ensembles, Pipelines | scikit-learn | [`examples/sklearn_demo.py`](examples/sklearn_demo.py) |
| Clustering (KMeans, DBSCAN) | scikit-learn | [`examples/sklearn_demo.py`](examples/sklearn_demo.py) |
| NLP / text classification | PyTorch LSTM | [`examples/time_series_demo.py`](examples/time_series_demo.py) |
| Time-series forecasting | PyTorch LSTM | [`examples/time_series_demo.py`](examples/time_series_demo.py) |
| Anomaly detection | PyTorch Autoencoder | see [Python API docs](docs/python-api.md) |
| Generative (VAE, GAN) | PyTorch | see [Python API docs](docs/python-api.md) |
| Reinforcement learning | PyTorch policy net | bring your own model |
| Graph neural networks | PyTorch Geometric | bring your own model |
| Self-supervised / contrastive | PyTorch | bring your own model |
| Object detection / segmentation | PyTorch / YOLO | bring your own model |
| Recommendation systems | any | bring your own model |

---

## 📊 Dashboard

Launch the local React dashboard to explore your runs:

```python
run.open()       # open current run in browser
gg.open_last()   # open the most recent run
gg.list_runs()   # list all captured runs
```

Or from the terminal:

```bash
gg serve   # start the dashboard server (opens browser)
gg open    # open the most recent run
gg ls      # list all runs
```

---

## 🔎 Data Leakage Detection

```python
# NumPy / sklearn arrays
run.check_leakage(X_train, y_train, X_test, y_test)

# PyTorch DataLoaders
run.check_leakage_from_loaders(train_loader, test_loader)
```

---

## 🧪 Custom Analysis Tests

```python
@gg.test()
def my_custom_check(ctx):
    # ctx provides checkpoints, gradients, metrics, and run metadata
    ...
```

---

## 📖 Documentation

| Doc | Description |
|-----|-------------|
| [Getting Started](docs/getting-started.md) | Installation, first run, per-framework walkthroughs |
| [Python API](docs/python-api.md) | Full reference for `gg`, `Run`, `@test`, `TestContext`, and all callbacks |
| [CLI](docs/cli.md) | All `gg` terminal commands with flags and examples |
| [Data Leakage Detection](docs/leakage.md) | The 7 leakage checks, thresholds, and the `LeakageDetector` class |
| [Analysis System](docs/analysis.md) | Built-in test catalog, custom test authoring, `AnalysisRunner` |
| [Dashboard](docs/dashboard.md) | Every dashboard page , what it shows and how to read it |
| [Artifact Storage](docs/artifacts.md) | Directory layout, every file explained, `ArtifactStore` API |
| [Configuration](docs/configuration.md) | All config options with types, defaults, and where to set them |
| [Contributing](CONTRIBUTING.md) | Dev setup, project layout, tests, PR conventions |
