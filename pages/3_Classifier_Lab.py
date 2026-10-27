import time

import pandas as pd
import streamlit as st

import ui
from core import ml_models as ml

ui.setup("Classifier Lab", "📊", "Compare classic classifiers on built-in data or your own CSV.")

# ------------------------------------------------------------------ data source
source = st.radio("Dataset source", ["Built-in dataset", "Upload CSV"], horizontal=True)
if source == "Built-in dataset":
    label = st.selectbox("Dataset", list(ml.DATASETS))
    X, y = ml.load_dataset(label)
else:
    upload = st.file_uploader("CSV file (one row per example, one column holds the class)", type="csv")
    if upload is None:
        st.info("Upload a CSV to continue, or switch to a built-in dataset.")
        st.stop()
    raw = pd.read_csv(upload)
    target = st.selectbox("Target column (the class to predict)", raw.columns, index=len(raw.columns) - 1)
    try:
        X, y = ml.prepare_uploaded(raw, target)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    label = upload.name

c1, c2, c3 = st.columns(3)
c1.metric("Rows", len(X))
c2.metric("Features", X.shape[1])
c3.metric("Classes", y.nunique())
with st.expander("Preview data"):
    st.dataframe(X.assign(target=y).head(50))

# --------------------------------------------------------------------- settings
st.subheader("Settings")
s1, s2, s3 = st.columns(3)
chosen = s1.multiselect("Models", list(ml.make_models()), default=list(ml.make_models()))
test_size = s2.slider("Test size", 0.1, 0.5, 0.2, 0.05)
seed = int(s3.number_input("Random seed", 0, 9999, 42))
use_cv = st.checkbox("Also run cross-validation (more reliable on small datasets)")
folds = st.slider("Folds", 2, 10, 5) if use_cv else 5

if st.button("Train and compare", type="primary", disabled=not chosen):
    try:
        X_tr, X_te, y_tr, y_te = ml.split_data(X, y, test_size, seed)
        results, fitted, preds = ml.train_and_evaluate(chosen, X_tr, X_te, y_tr, y_te, seed)
        if use_cv:
            cv = {name: ml.cross_validate(name, X, y, folds, seed) for name in chosen}
            results["CV accuracy"] = [cv[n][0] if cv[n] else None for n in results.index]
            results["CV std"] = [cv[n][1] if cv[n] else None for n in results.index]
    except ValueError as exc:
        st.error(f"Could not train: {exc}")
        st.stop()
    st.session_state["clf_run"] = {
        "id": time.time(), "label": label, "results": results, "fitted": fitted, "preds": preds,
        "y_test": y_te, "columns": list(X.columns), "labels": sorted(y.unique()),
        "ref": pd.DataFrame({"mean": X.mean(), "min": X.min(), "max": X.max()}),
    }

run = st.session_state.get("clf_run")
if run is None:
    st.stop()

# ---------------------------------------------------------------------- results
st.divider()
st.subheader(f"Results on {run['label']}")
results = run["results"]
best = results["F1"].idxmax()
st.success(f"Best model by F1 score: **{best}** (F1 = {results.loc[best, 'F1']:.3f})")
st.dataframe(results.style.format("{:.3f}", na_rep="-").highlight_max(axis=0, subset=["Accuracy", "Precision", "Recall", "F1"], color="#cfe8ff"))
st.bar_chart(results[["Accuracy", "Precision", "Recall", "F1"]])
st.download_button("Download results (CSV)", results.to_csv(), "classifier_results.csv", "text/csv")

st.subheader("Inspect a model")
names = list(results.index)
inspect = st.selectbox("Model", names, index=names.index(best))
model, pred = run["fitted"][inspect], run["preds"][inspect]

col_a, col_b = st.columns(2, gap="large")
with col_a:
    st.markdown("**Confusion matrix** (rows = true class, columns = predicted)")
    from sklearn.metrics import confusion_matrix

    labels = run["labels"]
    st.dataframe(pd.DataFrame(confusion_matrix(run["y_test"], pred, labels=labels),
                              index=[f"true {l}" for l in labels], columns=[f"pred {l}" for l in labels]))
with col_b:
    importance = ml.feature_importance(model, run["columns"])
    if importance is not None:
        st.markdown("**Feature importance**")
        st.bar_chart(importance.head(15))
    else:
        st.caption("This model (SVM with an RBF kernel) has no simple feature-importance measure.")

with st.expander("🔮 Try a prediction"):
    ref, columns = run["ref"], run["columns"]
    editable = columns[:12]
    with st.form("predict_form"):
        grid = st.columns(3)
        values = {
            col: grid[i % 3].number_input(col, value=float(ref.loc[col, "mean"]), key=f"in_{run['id']}_{col}")
            for i, col in enumerate(editable)
        }
        go = st.form_submit_button("Predict")
    if len(columns) > 12:
        st.caption("Only the first 12 features are editable here; the rest use their average value.")
    if go:
        row = ref["mean"].to_dict()
        row.update(values)
        frame = pd.DataFrame([row])[columns]
        st.success(f"{inspect} predicts: **{model.predict(frame)[0]}**")
        try:
            proba = pd.Series(model.predict_proba(frame)[0], index=model.classes_)
            st.bar_chart(proba)
        except (AttributeError, NotImplementedError):
            pass

ui.show_originals("04_Logistic_Regression.py", "05_Decision_Tree_and_Random_Forest.py", "07_SVM_Classification.py",
                  "10_ML_Evaluation_Metrics.py")
