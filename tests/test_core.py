"""Tests for the pure logic in core/. Run with:  python -m pytest   (or)   python tests/test_core.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from core import clustering as cl
from core import expert_system as es
from core import graph_search as gs
from core import ml_models as ml
from core import theme

LAB_GRAPH = gs.parse_graph(gs.PRESETS["Tree from the lab (A to F)"])


# ----------------------------------------------------------------- graph search
def test_parse_graph_adds_neighbour_only_nodes():
    g = gs.parse_graph("A: B, C\n# comment\nB: D")
    assert g == {"A": ["B", "C"], "B": ["D"], "C": [], "D": []}


def test_parse_graph_undirected_mirrors_edges():
    g = gs.parse_graph("A: B", directed=False)
    assert g == {"A": ["B"], "B": ["A"]}


def test_bfs_matches_lab_output():
    assert gs.bfs(LAB_GRAPH, "A").order == ["A", "B", "C", "D", "E", "F"]


def test_dfs_matches_lab_output():
    assert gs.dfs(LAB_GRAPH, "A").order == ["A", "B", "D", "E", "C", "F"]


def test_bfs_finds_shortest_path_and_dfs_finds_some_path():
    g = gs.parse_graph(gs.PRESETS["Two routes to a goal"])
    bfs_path = gs.bfs(g, "S", "G").path_to("G")
    assert bfs_path is not None and len(bfs_path) == 4
    dfs_res = gs.dfs(g, "S", "G")
    assert dfs_res.goal_found and dfs_res.path_to("G")[0] == "S"


def test_search_stops_at_goal_and_handles_unreachable():
    res = gs.bfs(LAB_GRAPH, "A", "C")
    assert res.order == ["A", "B", "C"]
    assert gs.bfs(LAB_GRAPH, "B", "F").goal_found is False


def test_cycles_do_not_loop_forever():
    g = gs.parse_graph(gs.PRESETS["Graph with a cycle"])
    assert sorted(gs.bfs(g, "A").order) == sorted(g)
    assert sorted(gs.dfs(g, "A").order) == sorted(g)


def test_dot_output_contains_nodes_and_highlighted_path():
    res = gs.bfs(LAB_GRAPH, "A", "F")
    dot = gs.graph_to_dot(LAB_GRAPH, res.order, res.path_to("F"), start="A", goal="F")
    assert dot.startswith("digraph") and '"A" -> "C" [color="#e63946"' in dot


# ---------------------------------------------------------------- expert system
def test_expert_system_matches_original_lab_rules():
    assert es.diagnose({"fever", "cough"}).primary.conclusion == "Flu"
    assert es.diagnose({"cold", "cough"}).primary.conclusion == "Common Cold"
    assert es.diagnose({"fever"}).primary.conclusion == "Infection"
    assert es.diagnose({"cold"}).primary is None
    assert es.diagnose(set()).primary is None


def test_expert_system_priority_and_urgent_rule():
    d = es.diagnose({"fever", "cough", "breathless"})
    assert d.primary.urgent and len(d.fired) >= 2


def test_custom_rule_takes_priority_when_first():
    custom = es.Rule("C1", "Custom", frozenset({"headache"}))
    d = es.diagnose({"headache", "fever"}, [custom] + es.DEFAULT_RULES)
    assert d.primary.conclusion == "Custom"


# ------------------------------------------------------------------- ML models
def test_toy_dataset_predictions_match_lab_scripts():
    X, y = ml.toy_dataset()
    tree = ml.make_models()["Decision Tree"].fit(X, y)
    logreg = ml.make_models()["Logistic Regression"].fit(X, y)
    assert tree.predict(pd.DataFrame({"x": [5.0]}))[0] == "1"
    assert logreg.predict(pd.DataFrame({"x": [5.0]}))[0] == "1"


def test_train_and_evaluate_on_iris():
    X, y = ml.load_dataset("Iris")
    X_tr, X_te, y_tr, y_te = ml.split_data(X, y, 0.2, 42)
    res, fitted, preds = ml.train_and_evaluate(list(ml.make_models()), X_tr, X_te, y_tr, y_te)
    assert set(res.index) == set(ml.make_models())
    assert ((res[["Accuracy", "Precision", "Recall", "F1"]] >= 0) & (res[["Accuracy", "Precision", "Recall", "F1"]] <= 1)).all().all()
    assert res["Accuracy"].max() > 0.9
    assert ml.feature_importance(fitted["Random Forest"], X.columns).shape[0] == 4
    assert ml.feature_importance(fitted["Logistic Regression"], X.columns) is not None
    assert ml.feature_importance(fitted["SVM"], X.columns) is None


def test_cross_validation_and_small_data_guard():
    X, y = ml.load_dataset("Iris")
    mean, std = ml.cross_validate("Decision Tree", X, y, folds=5)
    assert 0.8 < mean <= 1.0 and std >= 0
    assert ml.cross_validate("Decision Tree", pd.DataFrame({"x": [1, 2]}), pd.Series(["a", "b"])) is None


def test_prepare_uploaded_validations():
    df = pd.DataFrame({"a": [1, 2, 3, 4], "color": list("xyxy"), "label": list("pqpq")})
    X, y = ml.prepare_uploaded(df, "label")
    assert "color_x" in X.columns and list(y) == list("pqpq")
    try:
        ml.prepare_uploaded(df.assign(label="same"), "label")
    except ValueError:
        pass
    else:
        raise AssertionError("single class should be rejected")
    try:
        ml.prepare_uploaded(pd.DataFrame({"a": range(50), "t": range(50)}), "t")
    except ValueError:
        pass
    else:
        raise AssertionError("continuous target should be rejected")


# ------------------------------------------------------------------- clustering
def test_kmeans_toy_matches_lab_script():
    res = cl.run_kmeans(cl.toy_points(), k=2, scale=False)
    assert list(res.labels[:3]) == [res.labels[0]] * 3 and list(res.labels[3:]) == [res.labels[3]] * 3
    assert res.labels[0] != res.labels[3]
    assert res.predict({"x": 4.0}) == res.labels[0]  # 4 is closer to the low group
    assert res.silhouette is not None and res.silhouette > 0.8


def test_kmeans_on_iris_and_elbow():
    X, _ = ml.load_dataset("Iris")
    res = cl.run_kmeans(X, k=3)
    assert res.projection.shape == (150, 3) and res.centers.shape == (3, 4)
    table = cl.elbow(X, max_k=6)
    assert table["Inertia"].is_monotonic_decreasing and len(table) == 6


# ------------------------------------------------------------------------ theme
def test_theme_css_for_each_mode():
    assert theme.CHOICES == ["Auto", "Light", "Dark"]
    assert theme.theme_css("Auto") == "" and theme.theme_css("anything else") == ""
    dark, light = theme.theme_css("Dark"), theme.theme_css("Light")
    assert "prefers-color-scheme: light" in dark and "invert(1)" in dark   # dark only flips a light device
    assert "prefers-color-scheme: dark" in light and "invert(1)" in light  # light only flips a dark device
    assert "img, video" in dark  # images are flipped back so photos look normal


# ----------------------------------------------------------------- deep learning
def test_preprocess_image_shape_and_range():
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return
    from core import deep_learning as dl

    img = Image.new("L", (120, 90), 255)
    ImageDraw.Draw(img).line((30, 20, 90, 70), fill=0, width=8)  # dark stroke on white
    out = dl.preprocess_image(img)
    assert out.shape == (1, 28, 28, 1) and 0.0 <= out.min() and out.max() <= 1.0 and out.max() > 0.5


if __name__ == "__main__":
    failures = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print("FAIL", name, "->", repr(exc))
    sys.exit(1 if failures else 0)
