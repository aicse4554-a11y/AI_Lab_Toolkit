import random

import numpy as np
import pandas as pd
import streamlit as st

import ui
from core import deep_learning as dl

ui.setup("Deep Learning", "🧬", "Train an ANN and a CNN on MNIST handwritten digits, then test them.")

if not dl.tf_available():
    st.error("TensorFlow is not installed, so this page is disabled. The other pages work without it.")
    st.code("pip install -r requirements-dl.txt", language="bash")
    st.caption("TensorFlow needs Python 3.9 to 3.12. The first run also downloads the MNIST dataset (about 11 MB).")
    ui.show_originals("08_ANN_Keras.py", "09_CNN.py")
    st.stop()

if "dl_results" not in st.session_state:
    st.session_state["dl_results"] = {}

# --------------------------------------------------------------------- training
st.subheader("Train a model")
c1, c2, c3, c4 = st.columns(4)
kind = c1.selectbox("Model", dl.MODEL_KINDS)
epochs = c2.slider("Epochs", 1, 10, 3)
n_samples = c3.select_slider("Training images", [5000, 10000, 30000, 60000], value=10000)
batch = c4.select_slider("Batch size", [32, 64, 128, 256], value=128)

if st.button(f"Train {kind}", type="primary"):
    bar = st.progress(0.0, text="Loading MNIST and building the model...")

    def on_epoch(done, total, logs):
        bar.progress(done / total, text=f"Epoch {done}/{total}: accuracy {logs.get('accuracy', 0):.3f}, "
                                        f"validation accuracy {logs.get('val_accuracy', 0):.3f}")

    with st.spinner("Training... the CNN can take a minute on a CPU."):
        st.session_state["dl_results"][kind] = dl.train_model(kind, epochs, n_samples, batch, on_epoch)
    bar.empty()
    st.session_state["dl_model_choice"] = kind

trained = st.session_state["dl_results"]
if not trained:
    st.info("Train a model to see results and try predictions.")
    ui.show_originals("08_ANN_Keras.py", "09_CNN.py")
    st.stop()

# ---------------------------------------------------------------------- results
st.subheader("Results")
st.dataframe(
    pd.DataFrame([
        {"Model": r.kind, "Test accuracy": r.test_accuracy, "Test loss": r.test_loss,
         "Training images": r.n_samples, "Epochs": r.epochs, "Parameters": r.model.count_params()}
        for r in trained.values()
    ]).style.format({"Test accuracy": "{:.2%}", "Test loss": "{:.3f}"}),
    hide_index=True,
)

names = list(trained)
active_name = st.selectbox("Model to inspect", names, index=names.index(st.session_state.get("dl_model_choice", names[0])))
active = trained[active_name]
history = pd.DataFrame({"train accuracy": active.history["accuracy"], "validation accuracy": active.history["val_accuracy"]})
history.index = range(1, len(history) + 1)
st.line_chart(history)

# ------------------------------------------------------------------ predictions
st.subheader("Try it")
tab_test, tab_upload = st.tabs(["A test-set digit", "Your own digit image"])

with tab_test:
    _, _, x_test, y_test = dl.load_mnist()
    st.session_state.setdefault("dl_idx", 0)
    pick, rand = st.columns([3, 1])
    pick.number_input("Test image index", 0, len(x_test) - 1, key="dl_idx")
    rand.button("🎲 Random", on_click=lambda: st.session_state.update(dl_idx=random.randrange(len(x_test))))
    idx = int(st.session_state["dl_idx"])
    probs = dl.predict_digit(active.model, x_test[idx: idx + 1])
    a, b = st.columns(2)
    a.image(x_test[idx, :, :, 0], width=200, clamp=True)
    guess = int(np.argmax(probs))
    (b.success if guess == int(y_test[idx]) else b.error)(f"Predicted **{guess}**, actual **{int(y_test[idx])}**")
    b.bar_chart(pd.Series(probs, index=[str(i) for i in range(10)]))

with tab_upload:
    st.caption("Upload a photo or drawing of a single digit (dark ink on a plain background works best).")
    file = st.file_uploader("Image", type=["png", "jpg", "jpeg"])
    if file is not None:
        from PIL import Image

        image = Image.open(file)
        batch_img = dl.preprocess_image(image)
        probs = dl.predict_digit(active.model, batch_img)
        a, b, c = st.columns(3)
        a.image(image, caption="Your image", width=200)
        b.image((batch_img[0, :, :, 0] * 255).astype("uint8"), caption="What the model sees (28×28)", width=200)
        c.success(f"Predicted **{int(np.argmax(probs))}** ({probs.max():.1%} confident)")
        c.bar_chart(pd.Series(probs, index=[str(i) for i in range(10)]))

ui.show_originals("08_ANN_Keras.py", "09_CNN.py")
