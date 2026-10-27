"""Classic supervised-learning helpers: datasets, models, evaluation and feature importance."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

TOY_NAME = "Toy 1D (from the lab scripts)"


def toy_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """The tiny dataset used in the Logistic Regression / Tree / SVM lab programs."""
    X = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    y = pd.Series(["0", "0", "0", "1", "1", "1"], name="target")
    return X, y


def _sklearn_dataset(loader):
    def load() -> tuple[pd.DataFrame, pd.Series]:
        d = loader()
        X = pd.DataFrame(d.data, columns=list(d.feature_names))
        y = pd.Series(d.target).map(dict(enumerate(d.target_names))).astype(str).rename("target")
        return X, y

    return load


DATASETS = {
    "Iris": _sklearn_dataset(load_iris),
    "Wine": _sklearn_dataset(load_wine),
    "Breast Cancer": _sklearn_dataset(load_breast_cancer),
    TOY_NAME: toy_dataset,
}


def load_dataset(name: str) -> tuple[pd.DataFrame, pd.Series]:
    return DATASETS[name]()


def prepare_uploaded(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    """Turn an uploaded table into numeric features X and string labels y (with checks)."""
    df = df.dropna()
    if target not in df.columns:
        raise ValueError(f"Column '{target}' not found.")
    y = df[target].astype(str)
    if y.nunique() < 2:
        raise ValueError("The target column needs at least two different classes.")
    if y.nunique() > 20:
        raise ValueError("The target has more than 20 distinct values, so it looks like a number to predict "
                         "(regression), not classes. Pick a different target column.")
    X = pd.get_dummies(df.drop(columns=[target])).astype(float)
    if X.shape[1] == 0:
        raise ValueError("No feature columns left after removing the target.")
    return X, y.rename("target")


def make_models(seed: int = 42) -> dict:
    return {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "Decision Tree": DecisionTreeClassifier(random_state=seed),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=seed),
        "SVM": make_pipeline(StandardScaler(), SVC()),
    }


def split_data(X, y, test_size: float = 0.2, seed: int = 42):
    """Train/test split, stratified when the class counts allow it."""
    try:
        return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    except ValueError:
        return train_test_split(X, y, test_size=test_size, random_state=seed)


def metrics_row(y_true, y_pred) -> dict[str, float]:
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def train_and_evaluate(names, X_train, X_test, y_train, y_test, seed: int = 42):
    """Fit each chosen model. Returns (results table, fitted models, predictions)."""
    available = make_models(seed)
    rows, fitted, preds = [], {}, {}
    for name in names:
        t0 = time.perf_counter()
        model = available[name].fit(X_train, y_train)
        elapsed = time.perf_counter() - t0
        pred = model.predict(X_test)
        fitted[name], preds[name] = model, pred
        rows.append({"Model": name, **metrics_row(y_test, pred), "Train time (s)": elapsed})
    return pd.DataFrame(rows).set_index("Model"), fitted, preds


def cross_validate(name: str, X, y, folds: int = 5, seed: int = 42) -> tuple[float, float] | None:
    """Mean and std accuracy over stratified folds, or None when the data is too small."""
    smallest = int(pd.Series(y).value_counts().min())
    if smallest < 2:
        return None
    folds = max(2, min(folds, smallest))
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = cross_val_score(make_models(seed)[name], X, y, cv=cv, scoring="accuracy")
    return float(scores.mean()), float(scores.std())


def feature_importance(model, columns) -> pd.Series | None:
    """Importance for trees/forests, mean |coefficient| for logistic regression, else None."""
    est = model[-1] if hasattr(model, "steps") else model
    if hasattr(est, "feature_importances_"):
        values = est.feature_importances_
    elif hasattr(est, "coef_"):
        values = np.abs(est.coef_).mean(axis=0)
    else:
        return None
    return pd.Series(values, index=list(columns)).sort_values(ascending=False)
