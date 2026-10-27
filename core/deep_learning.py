"""ANN and CNN on MNIST. TensorFlow is imported lazily so the rest of the app works without it."""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

MODEL_KINDS = ["ANN (dense)", "CNN"]


def tf_available() -> bool:
    return importlib.util.find_spec("tensorflow") is not None


@lru_cache(maxsize=1)
def load_mnist():
    """Normalised MNIST with a channel axis: (x_train, y_train, x_test, y_test)."""
    import tensorflow as tf

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = (x_train / 255.0).astype("float32")[..., None]
    x_test = (x_test / 255.0).astype("float32")[..., None]
    return x_train, y_train, x_test, y_test


def build_model(kind: str):
    from tensorflow.keras import layers, models

    if kind == "ANN (dense)":
        model = models.Sequential([
            layers.Input(shape=(28, 28, 1)),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(10, activation="softmax"),
        ])
    elif kind == "CNN":
        model = models.Sequential([
            layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dense(10, activation="softmax"),
        ])
    else:
        raise ValueError(f"Unknown model kind: {kind}")
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


@dataclass
class TrainResult:
    kind: str
    model: object
    history: dict
    test_accuracy: float
    test_loss: float
    n_samples: int
    epochs: int


def train_model(kind: str, epochs: int = 3, n_samples: int = 10000, batch_size: int = 128, on_epoch=None) -> TrainResult:
    """Train on the first ``n_samples`` images. ``on_epoch(epoch, total, logs)`` reports progress."""
    import tensorflow as tf

    x_train, y_train, x_test, y_test = load_mnist()
    tf.keras.utils.set_random_seed(42)
    model = build_model(kind)
    callbacks = []
    if on_epoch is not None:
        class _Progress(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                on_epoch(epoch + 1, epochs, logs or {})

        callbacks.append(_Progress())
    hist = model.fit(x_train[:n_samples], y_train[:n_samples], epochs=epochs, batch_size=batch_size,
                     validation_split=0.1, verbose=0, callbacks=callbacks)
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    return TrainResult(kind, model, hist.history, float(acc), float(loss), n_samples, epochs)


def preprocess_image(img) -> np.ndarray:
    """Convert a photo/drawing of a digit to a (1, 28, 28, 1) MNIST-style array.

    Dark ink on a light background is inverted, the digit is cropped, scaled to fit
    a 20x20 box and centred on a 28x28 canvas, like the original MNIST images.
    """
    from PIL import Image

    arr = np.asarray(img.convert("L"), dtype="float32") / 255.0
    if arr.mean() > 0.5:
        arr = 1.0 - arr
    ys, xs = np.where(arr > 0.2)
    if len(ys):
        arr = arr[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]
    h, w = arr.shape
    scale = 20.0 / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    small = Image.fromarray((arr * 255).astype("uint8")).resize((nw, nh), Image.LANCZOS)
    canvas = np.zeros((28, 28), dtype="float32")
    top, left = (28 - nh) // 2, (28 - nw) // 2
    canvas[top: top + nh, left: left + nw] = np.asarray(small, dtype="float32") / 255.0
    return canvas[None, ..., None]


def predict_digit(model, image_batch: np.ndarray) -> np.ndarray:
    """Class probabilities (length 10) for one (1, 28, 28, 1) image."""
    return model.predict(image_batch, verbose=0)[0]
