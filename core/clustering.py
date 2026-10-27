"""K-Means clustering helpers: fit, project to 2D, elbow/silhouette analysis, predict."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

TOY_NAME = "Toy 1D (from the lab script)"


def toy_points() -> pd.DataFrame:
    return pd.DataFrame({"x": [1.0, 2.0, 3.0, 10.0, 11.0, 12.0]})


def _silhouette(Z, labels) -> float | None:
    n_labels = len(set(labels))
    if 2 <= n_labels < len(Z):
        return float(silhouette_score(Z, labels))
    return None


@dataclass
class ClusterResult:
    labels: np.ndarray
    centers: pd.DataFrame
    inertia: float
    silhouette: float | None
    projection: pd.DataFrame
    model: KMeans
    scaler: StandardScaler | None
    columns: list[str]

    def predict(self, values: dict[str, float]) -> int:
        row = np.array([[float(values[c]) for c in self.columns]])
        if self.scaler is not None:
            row = self.scaler.transform(row)
        return int(self.model.predict(row)[0])


def run_kmeans(df: pd.DataFrame, k: int, scale: bool = True, seed: int = 0) -> ClusterResult:
    X = df.to_numpy(dtype=float)
    scaler = StandardScaler().fit(X) if scale else None
    Z = scaler.transform(X) if scaler is not None else X
    model = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(Z)
    centers = scaler.inverse_transform(model.cluster_centers_) if scaler is not None else model.cluster_centers_
    if Z.shape[1] >= 2:
        pts = PCA(n_components=2, random_state=seed).fit_transform(Z)
    else:
        pts = np.column_stack([Z[:, 0], np.zeros(len(Z))])
    projection = pd.DataFrame({"PC1": pts[:, 0], "PC2": pts[:, 1], "cluster": model.labels_.astype(str)})
    return ClusterResult(
        labels=model.labels_,
        centers=pd.DataFrame(centers, columns=list(df.columns)).rename_axis("cluster"),
        inertia=float(model.inertia_),
        silhouette=_silhouette(Z, model.labels_),
        projection=projection,
        model=model,
        scaler=scaler,
        columns=list(df.columns),
    )


def elbow(df: pd.DataFrame, max_k: int = 10, scale: bool = True, seed: int = 0) -> pd.DataFrame:
    """Inertia (and silhouette for k >= 2) for k = 1..max_k."""
    X = df.to_numpy(dtype=float)
    Z = StandardScaler().fit_transform(X) if scale else X
    rows = []
    for k in range(1, min(max_k, len(df) - 1) + 1):
        km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(Z)
        rows.append({"k": k, "Inertia": float(km.inertia_), "Silhouette": _silhouette(Z, km.labels_)})
    return pd.DataFrame(rows).set_index("k")
