# GradGlass

GradGlass is a lightweight training observability toolkit for deep learning runs.
It captures metrics, checkpoints, gradients, activations, predictions, and architecture snapshots, then serves them in a local dashboard.

## Framework Support

GradGlass currently supports:

- PyTorch
- TensorFlow / Keras

## Install

```bash
pip install gradglass
```

## Quick Start (PyTorch)

```python
import torch
import torch.nn.functional as F
from gradglass import gg

model = MyModel()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

run = gg.run("my-pytorch-run", epochs=10, lr=1e-3)
run.watch(model, optimizer, every=10, monitor=True)

for epoch in range(10):
    for x, y in train_loader:
        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()

        run.log(loss=loss.item(), epoch=epoch)
        run.log_batch(x=x, y=y, y_pred=logits, loss=loss)

run.finish(open=True)
```

## Quick Start (TensorFlow / Keras)

```python
import tensorflow as tf
from gradglass import gg

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(10),
])
model.compile(optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

run = gg.run("my-keras-run", epochs=5)
cb = run.keras_callback()

model.fit(train_x, train_y, validation_data=(val_x, val_y), epochs=5, callbacks=[cb])
run.finish(open=True)
```

## Notes

- Use `run.watch(model, optimizer)` for PyTorch and `run.keras_callback()` for Keras flows.
- `run.fit(...)` is intentionally disabled in this release.
- Historical runs with other framework labels remain readable through the API/dashboard, but new execution paths are PT/TF only.

## Included Examples

- `examples/mnist_demo.py`
- `examples/regression_demo.py`
- `examples/tensorflow_demo.py`
- `examples/overfitting_demo.py`
- `examples/gradient_instability_demo.py`
- `examples/interpretability_demo.py`
- `examples/leakage_demo.py`
- `examples/time_series_demo.py`
- `examples/transfer_learning_demo.py`

## Dashboard

The dashboard is served from the local API and includes:

- Overview and health state
- Training metrics
- Model architecture visualization
- Gradients, activations, predictions
- Infrastructure telemetry

Start a dashboard while training with `monitor=True` in `run.watch(...)`, or open after training with `run.open()` / `run.finish(open=True)`.
