# 🧠 AI Lab Toolkit

One Streamlit application that combines ten classic AI/ML lab programs: search algorithms, a rule-based expert
system, classifiers, clustering and neural networks. Every page works with built-in data or your own CSV.

## Features

| Page | Built from | What you can do |
|---|---|---|
| **Graph Search** | BFS, DFS | Type any graph, run BFS/DFS, see a drawn graph, step log, goal search and a BFS-vs-DFS comparison |
| **Expert System** | Rule-Based Expert System | Tick symptoms, see which rules fire and why, add your own rules |
| **Classifier Lab** | Logistic Regression, Decision Tree, Random Forest, SVM, Evaluation Metrics | Compare 4 models on accuracy / precision / recall / F1, optional cross-validation, confusion matrix, feature importance, live prediction |
| **Clustering** | K-Means | 2D cluster plot, elbow + silhouette analysis, predict a new point, download labelled data |
| **Deep Learning** | ANN, CNN | Train an ANN or CNN on MNIST, compare them, predict test digits or your own digit image |

## Project structure

```
AI_Lab_Toolkit/
├── app.py                      # Home page (run this)
├── ui.py                       # Shared page setup, theme switch, "show original code" panel
├── pages/
│   ├── 1_Graph_Search.py
│   ├── 2_Expert_System.py
│   ├── 3_Classifier_Lab.py
│   ├── 4_Clustering.py
│   └── 5_Deep_Learning.py
├── core/                       # Pure logic, no Streamlit (easy to test and reuse)
│   ├── graph_search.py
│   ├── expert_system.py
│   ├── ml_models.py
│   ├── clustering.py
│   ├── deep_learning.py
│   └── theme.py                # CSS behind the Light / Dark / Auto switch
├── tests/test_core.py          # Tests for the core logic
├── original_scripts/           # Your ten original lab programs, untouched
├── .streamlit/config.toml
├── requirements.txt            # Everything except TensorFlow
├── requirements-dl.txt         # + TensorFlow for the Deep Learning page
├── Dockerfile
├── .gitignore
└── README.md
```

## Quick start

```bash
cd AI_Lab_Toolkit
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt    # everything except the Deep Learning page
# or, to include the Deep Learning page too:
pip install -r requirements-dl.txt

streamlit run app.py
```

The app opens at http://localhost:8501. Without TensorFlow the Deep Learning page shows install instructions and the
rest of the app works normally. TensorFlow needs **Python 3.9 - 3.12**, and the first training run downloads MNIST
(about 11 MB).

## Light / Dark mode

Every page has a **🌗 Theme** switch in the sidebar with three options:

- **Auto** (default): follows your device's light/dark setting.
- **Light** / **Dark**: force that look. The choice is kept as you move between pages.

Streamlit also has its own theme menu (⋮ top-right → Settings → Theme). If you pick a theme there, leave the
sidebar switch on **Auto** so the two do not fight. To change the default colours permanently, edit
`.streamlit/config.toml`.

## Using your own data

- **Classifier Lab**: upload a CSV, pick the column holding the class (for example `species`). Text columns are
  one-hot encoded automatically; rows with missing values are dropped.
- **Clustering**: upload a CSV and pick the numeric columns to cluster on.
- **Deep Learning**: upload a photo or drawing of one digit; it is cropped, centred and resized to 28×28 like MNIST.

## Tests

```bash
python -m pytest            # or:  python tests/test_core.py
```

## Docker

```bash
docker build -t ai-lab-toolkit .
docker run -p 8501:8501 ai-lab-toolkit
```

## Troubleshooting

- **`ModuleNotFoundError: core`**: run `streamlit run app.py` from inside the `AI_Lab_Toolkit` folder.
- **Graph does not draw**: check each line looks like `A: B, C` (a colon after the node name).
- **TensorFlow install fails**: check `python --version`; use 3.9 - 3.12 or skip TensorFlow.
- **Slow CNN training**: lower the number of training images or epochs on the Deep Learning page.

*The expert system is an educational demo and not medical advice.*
