import pandas as pd
import streamlit as st

import ui
from core import clustering as cl
from core import ml_models as ml

ui.setup("Clustering", "🎯", "K-Means finds groups in unlabeled data. Use sample data or upload your own CSV.")


@st.cache_data(show_spinner=False)
def cached_elbow(df: pd.DataFrame, max_k: int, scale: bool) -> pd.DataFrame:
    return cl.elbow(df, max_k, scale)


source = st.radio("Data", [cl.TOY_NAME, "Iris (labels ignored)", "Upload CSV"], horizontal=True)
if source == cl.TOY_NAME:
    data = cl.toy_points()
elif source.startswith("Iris"):
    data, _ = ml.load_dataset("Iris")
else:
    upload = st.file_uploader("CSV with numeric columns", type="csv")
    if upload is None:
        st.info("Upload a CSV to continue, or pick a sample dataset.")
        st.stop()
    table = pd.read_csv(upload).select_dtypes("number").dropna()
    if table.shape[1] == 0:
        st.error("No numeric columns found in that file.")
        st.stop()
    features = st.multiselect("Feature columns", list(table.columns), default=list(table.columns))
    if not features:
        st.warning("Pick at least one feature column.")
        st.stop()
    data = table[features]

if len(data) < 4:
    st.error("K-Means needs at least 4 rows.")
    st.stop()

c1, c2, c3 = st.columns(3)
max_k = min(10, len(data) - 1)
k = c1.slider("Number of clusters (k)", 2, max_k, min(3, max_k)) if max_k > 2 else 2
scale = c2.checkbox("Standardise features", value=True, help="Recommended when columns use different units.")
c3.metric("Rows × features", f"{len(data)} × {data.shape[1]}")

result = cl.run_kmeans(data, k, scale)

m1, m2 = st.columns(2)
m1.metric("Inertia (lower = tighter clusters)", f"{result.inertia:.2f}")
m2.metric("Silhouette (closer to 1 = better separated)", f"{result.silhouette:.3f}" if result.silhouette is not None else "n/a")

left, right = st.columns(2, gap="large")
with left:
    st.markdown("**Clusters** (2D view; PCA when there are 2+ features)")
    st.scatter_chart(result.projection, x="PC1", y="PC2", color="cluster")
with right:
    st.markdown("**Cluster centres and sizes**")
    summary = result.centers.round(3).copy()
    summary["size"] = pd.Series(result.labels).value_counts().sort_index().values
    st.dataframe(summary)

st.subheader("Choosing k")
elbow = cached_elbow(data, 10, scale)
e1, e2 = st.columns(2)
e1.markdown("**Elbow curve** (look for the bend)")
e1.line_chart(elbow["Inertia"])
e2.markdown("**Silhouette by k** (higher is better)")
e2.line_chart(elbow["Silhouette"].dropna())

with st.expander("🔮 Which cluster does a new point belong to?"):
    with st.form("new_point"):
        grid = st.columns(min(4, len(data.columns)))
        point = {
            col: grid[i % len(grid)].number_input(col, value=float(data[col].mean()), key=f"pt_{col}")
            for i, col in enumerate(data.columns[:8])
        }
        go = st.form_submit_button("Predict cluster")
    if len(data.columns) > 8:
        st.caption("Only the first 8 features are editable; others use their average.")
    if go:
        full = {c: float(data[c].mean()) for c in data.columns}
        full.update(point)
        st.success(f"The point belongs to **cluster {result.predict(full)}**.")

out = data.assign(cluster=result.labels)
st.download_button("Download data with cluster labels (CSV)", out.to_csv(index=False), "clustered_data.csv", "text/csv")

ui.show_originals("06_KMeans_Clustering.py")
