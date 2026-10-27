import pandas as pd
import streamlit as st

import ui
from core import graph_search as gs

ui.setup("Graph Search", "🕸️", "Watch how Breadth-First and Depth-First Search explore a graph.")

left, right = st.columns([1, 2], gap="large")

with left:
    preset = st.selectbox("Example graph", list(gs.PRESETS))
    if st.session_state.get("_graph_preset") != preset:
        st.session_state["_graph_preset"] = preset
        st.session_state["graph_text"] = gs.PRESETS[preset]
    st.text_area(
        "Graph definition",
        key="graph_text",
        height=220,
        help="One node per line: `A: B, C` means A connects to B and C. Use # for comments.",
    )
    directed = st.toggle("Directed edges", value=True)
    graph = gs.parse_graph(st.session_state["graph_text"], directed=directed)
    if not graph:
        st.warning("Enter at least one line such as `A: B, C`.")
        st.stop()
    nodes = list(graph)
    start = st.selectbox("Start node", nodes)
    goal_choice = st.selectbox("Goal node (optional)", ["(none)"] + nodes)
    goal = None if goal_choice == "(none)" else goal_choice
    algo = st.radio("Algorithm", ["BFS", "DFS"], horizontal=True)

result = (gs.bfs if algo == "BFS" else gs.dfs)(graph, start, goal)
path = result.path_to(goal)

with right:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Nodes visited", len(result.order))
    m2.metric("Nodes in graph", len(graph))
    m3.metric("Deepest level", max(result.depth[n] for n in result.order))
    m4.metric("Path length", len(path) - 1 if path else "-")
    st.graphviz_chart(gs.graph_to_dot(graph, result.order, path, directed, start, goal))
    st.caption("Green = start, orange = goal, blue = visited (number = visit order), red = path found.")

st.success(f"**{algo} visit order:** " + " → ".join(result.order))
if goal:
    if result.goal_found:
        st.info(f"Goal **{goal}** reached after visiting {len(result.order)} node(s). Path: " + " → ".join(path))
    else:
        st.error(f"Goal **{goal}** is not reachable from **{start}**.")

tab_steps, tab_compare = st.tabs(["Step log", "Compare BFS vs DFS"])

with tab_steps:
    st.dataframe(pd.DataFrame(result.steps), hide_index=True)

with tab_compare:
    rows = []
    for runner, structure in ((gs.bfs, "Queue (FIFO)"), (gs.dfs, "Stack (LIFO)")):
        r = runner(graph, start, goal)
        p = r.path_to(goal)
        rows.append({
            "Algorithm": r.algorithm,
            "Data structure": structure,
            "Visit order": " → ".join(r.order),
            "Nodes visited": len(r.order),
            "Path to goal": " → ".join(p) if p else ("-" if goal is None else "not found"),
            "Path length": len(p) - 1 if p else None,
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.markdown(
        "- **BFS** explores level by level, so it finds the *shortest* path in an unweighted graph.\n"
        "- **DFS** dives deep before backtracking. It may find a *longer* path but needs less memory on wide graphs."
    )

ui.show_originals("01_BFS.py", "02_DFS.py")
