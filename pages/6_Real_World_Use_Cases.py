import time

import numpy as np
import pandas as pd
import streamlit as st

import ui
from core import clustering as cl
from core import graph_search as gs
from core import usecases as uc

ui.setup("Real-World Use Cases", "🌍",
         "The lab algorithms applied to practical problems. All data is simulated, so replace it with your own for real use.")

tab_metro, tab_seg, tab_fraud = st.tabs(["🚇 Metro route planner", "🛍️ Customer segments", "🚨 Live fraud monitor"])

# ============================================================== 1. metro (BFS / DFS)
with tab_metro:
    st.markdown("**Algorithm:** BFS / DFS. Plan a trip and re-route around closed stations, like a transit app does.")
    stations = sorted({s for line in uc.METRO_LINES.values() for s in line})
    c1, c2, c3 = st.columns(3)
    start = c1.selectbox("From", stations, index=stations.index("Airport"))
    goal = c2.selectbox("To", stations, index=stations.index("University"))
    algo = c3.radio("Algorithm", ["BFS", "DFS"], horizontal=True, help="BFS finds the fewest stops. DFS finds *a* route, often a longer one.")
    closed = st.multiselect("Closed stations (maintenance, incident...)", [s for s in stations if s not in (start, goal)])

    trip = uc.plan_trip(start, goal, closed, algo)
    if trip is None:
        st.error("No route available with these closures.")
    elif trip.stops == 0:
        st.info("You are already there.")
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric("Stops", trip.stops)
        m2.metric("Transfers", trip.transfers)
        m3.metric("Estimated time", f"{trip.minutes:.0f} min")
        for line, a, b, n in trip.legs:
            st.write(f"{uc.LINE_EMOJI[line]} **{line} line**: {a} → {b} ({n} stop{'s' if n != 1 else ''})")
        graph = uc.metro_graph(set(closed) - {start, goal})
        st.graphviz_chart(gs.graph_to_dot(graph, trip.path, trip.path, directed=False, start=start, goal=goal))
        if closed:
            baseline = uc.plan_trip(start, goal, (), algo)
            if baseline and baseline.minutes != trip.minutes:
                st.caption(f"The closures add {trip.minutes - baseline.minutes:.0f} min compared with the normal route.")

# ================================================================ 2. segments (K-Means)
with tab_seg:
    st.markdown("**Algorithm:** K-Means. Group customers by behaviour and get a suggested marketing action for each group.")
    src = st.radio("Customer data", ["Sample shop (400 customers)", "Upload CSV"], horizontal=True)
    if src.startswith("Sample"):
        data = uc.make_customers()
    else:
        up = st.file_uploader("CSV of customers (numeric columns)", type="csv", key="seg_csv")
        if up is None:
            st.info("Upload a CSV, or switch to the sample data. Columns named like the sample get business names.")
            data = None
        else:
            table = pd.read_csv(up).select_dtypes("number").dropna()
            cols = st.multiselect("Columns to cluster on", list(table.columns), default=list(table.columns))
            data = table[cols] if cols else None
    if data is not None and len(data) >= 4:
        k = st.slider("Number of segments", 2, min(8, len(data) - 1), 4)
        res = cl.run_kmeans(data, k)
        names = uc.name_segments(res.centers)
        sizes = pd.Series(res.labels).value_counts()
        cards = st.columns(min(k, 4))
        for i in range(k):
            title, action = names[i]
            with cards[i % len(cards)], st.container(border=True):
                st.markdown(f"**{title}**")
                st.caption(f"{sizes.get(i, 0)} customers ({sizes.get(i, 0) / len(data):.0%})")
                st.write(action)
        left, right = st.columns(2, gap="large")
        left.scatter_chart(res.projection, x="PC1", y="PC2", color="cluster")
        right.dataframe(res.centers.round(1).assign(customers=sizes.sort_index().values))
        out = data.assign(segment=[names[int(l)][0] for l in res.labels])
        st.download_button("Download customers with segment (CSV)", out.to_csv(index=False), "customer_segments.csv", "text/csv")

# ============================================================ 3. fraud (Random Forest)
with tab_fraud:
    st.markdown("**Algorithm:** Random Forest. A simulated stream of card payments is scored as it arrives, "
                "and risky ones are flagged for review.")

    @st.cache_resource(show_spinner="Training the fraud model on simulated history...")
    def get_model():
        return uc.train_fraud_model()

    fm = get_model()
    with st.expander("Model quality on held-out history"):
        cols = st.columns(4)
        for col, (name, value) in zip(cols, fm.metrics.items()):
            col.metric(name, f"{value:.1%}")

    s1, s2, s3 = st.columns(3)
    threshold = s1.slider("Flag when fraud probability is above", 0.1, 0.9, 0.5, 0.05)
    fraud_rate = s2.slider("Share of fraud in the stream", 0.0, 0.3, 0.08, 0.01)
    per_tick = s3.slider("Transactions per second", 1, 10, 3)

    st.session_state.setdefault("feed", pd.DataFrame())
    st.session_state.setdefault("live", False)
    st.session_state.setdefault("rng", np.random.default_rng(1))
    b1, b2, _ = st.columns([1, 1, 4])
    if b1.button("▶ Start" if not st.session_state["live"] else "⏸ Pause", type="primary"):
        st.session_state["live"] = not st.session_state["live"]
        st.rerun()
    if b2.button("🗑 Clear"):
        st.session_state["feed"] = pd.DataFrame()
        st.rerun()

    @st.fragment(run_every=1.0 if st.session_state["live"] else None)
    def live_panel():
        if st.session_state["live"]:
            batch = uc.sample_transactions(per_tick, fraud_rate, st.session_state["rng"])
            batch["fraud_probability"] = uc.score(fm.model, batch)
            batch.insert(0, "time", time.strftime("%H:%M:%S"))
            st.session_state["feed"] = pd.concat([st.session_state["feed"], batch], ignore_index=True).tail(300)
        feed = st.session_state["feed"]
        if feed.empty:
            st.info("Press **Start** to begin the live feed.")
            return
        flagged = feed["fraud_probability"] > threshold
        caught = int((flagged & (feed["is_fraud"] == 1)).sum())
        real = int(feed["is_fraud"].sum())
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Transactions scored", len(feed))
        k2.metric("Flagged for review", int(flagged.sum()))
        k3.metric("Fraud caught", f"{caught}/{real}")
        k4.metric("False alarms", int((flagged & (feed["is_fraud"] == 0)).sum()))
        st.line_chart(feed["fraud_probability"].reset_index(drop=True).rolling(5, min_periods=1).mean(), height=150)
        st.markdown("**Latest flagged transactions**")
        shown = feed[flagged].tail(8).iloc[::-1].rename(columns={"is_fraud": "truly_fraud"})
        st.dataframe(shown.style.format({"fraud_probability": "{:.0%}"}), hide_index=True)

    live_panel()

    with st.expander("🔍 Score a single transaction yourself"):
        with st.form("one_txn"):
            g = st.columns(5)
            row = pd.DataFrame([{
                "amount": g[0].number_input("Amount", 0.0, 10000.0, 900.0),
                "hour": g[1].slider("Hour", 0, 23, 3),
                "distance_km": g[2].number_input("Km from home", 0.0, 20000.0, 400.0),
                "merchant_risk": g[3].slider("Merchant risk", 0.0, 1.0, 0.8),
                "txns_last_hour": g[4].number_input("Txns in last hour", 0, 50, 5),
            }])
            go = st.form_submit_button("Score")
        if go:
            p = float(uc.score(fm.model, row)[0])
            (st.error if p > threshold else st.success)(f"Fraud probability **{p:.0%}**: " + ("flag for review." if p > threshold else "looks normal."))

ui.show_originals("01_BFS.py", "02_DFS.py", "06_KMeans_Clustering.py", "05_Decision_Tree_and_Random_Forest.py")
