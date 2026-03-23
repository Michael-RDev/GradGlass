import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).parent.parent))
from gradglass import gg

EPOCHS = 8
BATCH_SIZE = 64
LR = 1e-3


def make_dataset(seed=42, n_samples=4000, n_features=20):
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, size=(n_samples, n_features)).astype(np.float32)
    w = rng.normal(0.0, 1.0, size=(n_features,)).astype(np.float32)
    logits = x @ w + 0.25 * rng.normal(0.0, 1.0, size=(n_samples,)).astype(np.float32)
    y = (logits > 0.0).astype(np.int32)

    split = int(0.8 * n_samples)
    return (x[:split], y[:split]), (x[split:], y[split:])


def build_model(n_features):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(n_features,)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy")],
    )
    return model


def main():
    print("GradGlass TensorFlow Demo")
    (x_train, y_train), (x_val, y_val) = make_dataset()
    model = build_model(n_features=x_train.shape[1])

    run = gg.run(
        "tensorflow-demo",
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LR,
        task="binary_classification",
    )
    callback = run.keras_callback()

    model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[callback],
        verbose=1,
    )

    run.finish(open=True, analyze=True, print_summary=True)
    print("Done")


if __name__ == "__main__":
    main()
