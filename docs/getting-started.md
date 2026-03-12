# Getting Started

This guide walks you through installing GradGlass and instrumenting your first training run.

---

## Requirements

- **Python ≥ 3.9**
- No cloud account, no API keys.

Optional (only needed if you modify the dashboard source):
- Node.js ≥ 18

---

## Installation

```bash
pip install gradglass
```

GradGlass's core dependencies are intentionally minimal (`numpy`, `fastapi`, `uvicorn`, `websockets`, `pydantic`). Install extras for your framework:

```bash
pip install "gradglass[torch]"        # adds torch ≥ 1.12
pip install "gradglass[tensorflow]"   # adds tensorflow ≥ 2.10
pip install "gradglass[all]"          # torch + tensorflow
```

sklearn, XGBoost, and LightGBM are supported out of the box, just have them installed in your environment.

---

## PyTorch — First Run

```python
from gradglass import gg

# 1. Create a run and attach it to your model and optimizer
run = gg.run('resnet-exp1', lr=0.001, epochs=10).watch(model, optimizer)

# 2. Log metrics inside your training loop
for epoch in range(num_epochs):
    for x, y in train_loader:
        logits = model(x)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = (logits.argmax(1) == y).float().mean().item()
        run.log(loss=loss.item(), acc=acc, epoch=epoch)

    # Save a checkpoint at the end of each epoch
    run.checkpoint(tag=f'epoch_{epoch}')

# 3. Finish the run — writes metadata, runs analysis, opens the dashboard
run.finish(analyze=True, open=True)
```

`run.finish()` marks the run as complete. Pass `analyze=True` to immediately run the built-in analysis suite (results appear in the dashboard). Pass `open=True` (or `open_browser=True`) to launch the dashboard automatically.

---

## Keras / TensorFlow

```python
from gradglass import gg

run = gg.run('lstm-experiment').watch(model)

model.fit(
    dataset,
    epochs=10,
    callbacks=[run.keras_callback()],
)

run.finish()
run.open()
```

The Keras callback automatically calls `run.log()` after each epoch and saves a checkpoint at the end of training.

---

## scikit-learn

GradGlass auto-detects any sklearn estimator or `Pipeline`:

```python
from gradglass import gg
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=200, oob_score=True)

run = gg.run('random-forest').watch(clf)
run.fit(X_train, y_train, X_val=X_test, y_val=y_test)

run.finish()
```

Use `run.fit()` instead of `clf.fit()` — it wraps the fit call and automatically captures hyperparameters, feature importances, coefficients, OOB score, and more. See [Python API — Run.fit()](python-api.md#runfit) for the full list of auto-captured attributes.

---

## XGBoost

```python
from gradglass import gg
import xgboost as xgb

run = gg.run('xgb-experiment')

booster = xgb.train(
    params,
    dtrain,
    num_boost_round=300,
    evals=[(dtrain, 'train'), (dval, 'val')],
    callbacks=[run.xgboost_callback()],
)

run.finish()
```

The XGBoost callback streams per-round eval metrics directly into the dashboard in real time.

---

## LightGBM

```python
from gradglass import gg
import lightgbm as lgb

run = gg.run('lgbm-experiment')

gbm = lgb.train(
    params,
    train_data,
    valid_sets=[val_data],
    callbacks=[run.lightgbm_callback()],
)

run.finish()
```

---

## Opening the Dashboard

From Python:

```python
run.open()        # open this run in the browser
gg.open_last()    # open the most recently created run
```

From the terminal (after `pip install gradglass`):

```bash
gradglass serve   # start the dashboard server and open the browser
gradglass open    # open the most recent run directly
gradglass ls      # print a table of all captured runs
```

The dashboard runs on `localhost` and never makes outbound network connections. See [Dashboard](dashboard.md) for a full description of every page.

---

## Auto-Open on Finish

Set the `GRADGLASS_OPEN` environment variable to skip the `open=True` argument:

```bash
export GRADGLASS_OPEN=1
```

Or configure it globally in Python:

```python
gg.configure(auto_open=True)
```

From that point on, every `run.finish()` call will automatically open the dashboard.

---

## Where Artifacts Are Stored

GradGlass writes all artifacts to `gg_artifacts/runs/<run-id>/` relative to your working directory. See [Artifact Storage](artifacts.md) for the full directory layout and a description of every file written.
