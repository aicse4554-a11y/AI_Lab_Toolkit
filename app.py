"""AI Lab Toolkit - home page.  Run with:  streamlit run app.py"""
import sys

import streamlit as st

import ui
from core import deep_learning as dl

st.set_page_config(page_title="AI Lab Toolkit", page_icon="🧠", layout="wide")
ui.theme_switcher()

st.title("🧠 AI Lab Toolkit")
st.markdown(
    "An interactive workbench that turns ten classic AI/ML lab programs into one application. "
    "Explore search algorithms, a rule-based expert system, machine-learning classifiers, "
    "clustering and neural networks, using the built-in data or **your own CSV files**."
)

cards = [
    ("pages/1_Graph_Search.py", "🕸️", "Graph Search", "BFS and DFS on any graph you type, with a step log, goal search and a drawn graph."),
    ("pages/2_Expert_System.py", "🩺", "Expert System", "Tick symptoms, see which rules fire, and add your own rules."),
    ("pages/3_Classifier_Lab.py", "📊", "Classifier Lab", "Logistic Regression, Decision Tree, Random Forest and SVM compared on accuracy, precision, recall and F1."),
    ("pages/4_Clustering.py", "🎯", "Clustering", "K-Means with a 2D plot, elbow and silhouette analysis, and new-point prediction."),
    ("pages/5_Deep_Learning.py", "🧬", "Deep Learning", "Train an ANN and a CNN on MNIST and test them on your own handwritten digits."),
]
cols = st.columns(len(cards))
for col, (path, icon, title, text) in zip(cols, cards):
    with col, st.container(border=True):
        st.markdown(f"### {icon} {title}")
        st.write(text)
        st.page_link(path, label=f"Open {title}")

st.subheader("How the lab programs map to this app")
st.table(
    {
        "Lab program": ["BFS, DFS", "Rule-Based Expert System", "Logistic Regression, Decision Tree / Random Forest, SVM, Evaluation Metrics",
                        "K-Means Clustering", "ANN (Keras), CNN"],
        "App page": ["Graph Search", "Expert System", "Classifier Lab", "Clustering", "Deep Learning"],
    }
)
st.caption("Every page has a panel at the bottom with the original program it is based on (see the `original_scripts/` folder).")

with st.sidebar:
    st.header("Environment")
    st.write(f"Python {sys.version.split()[0]}")
    if dl.tf_available():
        st.success("TensorFlow installed: Deep Learning page is ready.")
    else:
        st.warning("TensorFlow not installed. Everything except the Deep Learning page still works.")
        st.code("pip install -r requirements-dl.txt")
