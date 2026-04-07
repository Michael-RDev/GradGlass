import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from gradglass import gg

from _example_output import print_dashboard_next_steps, repo_workspace_root


def main():
    gg.configure(root=str(repo_workspace_root()), auto_open=False)

    # Prepare dummy dataset
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0

    # Just grab a small slice for demonstration
    x_train = x_train[:2000]
    y_train = y_train[:2000]

    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Conv2D(16, 3, activation="relu", input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    run = gg.run(name="mnist_keras_integration")

    # Watch model to hook layers
    run.watch(model)

    print(f"Started Keras run: {run.run_id}")

    # Train model using standard .fit(), passing in out-of-the-box callback
    model.fit(x_train, y_train, batch_size=64, epochs=2, callbacks=[run.keras_callback()])

    # Flush artifacts and close
    run.finish(open=False, analyze=True)
    print("Done!")
    print_dashboard_next_steps(gg.store.root)


if __name__ == "__main__":
    main()
