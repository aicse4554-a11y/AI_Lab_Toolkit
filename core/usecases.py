"""Real-world scenarios built on the lab algorithms.

1. Metro routing with station closures      (BFS / DFS)
2. Customer segmentation for marketing      (K-Means)
3. Live card-transaction fraud scoring      (Random Forest classifier)

All data here is synthetic so the app works out of the box; swap in real data for real use.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from core import graph_search as gs

# ======================================================================== 1. metro
METRO_LINES: dict[str, list[str]] = {
    "Red": ["Airport", "Tech Park", "Central", "Old Town", "Harbor"],
    "Blue": ["North Gate", "Central", "Stadium", "Mall", "South Bay"],
    "Green": ["Hospital", "Old Town", "Stadium", "Library", "University"],
    "Yellow": ["North Gate", "Tech Park", "Hospital"],
}
LINE_EMOJI = {"Red": "🟥", "Blue": "🟦", "Green": "🟩", "Yellow": "🟨"}

EDGE_LINE: dict[frozenset, str] = {
    frozenset((a, b)): line for line, stations in METRO_LINES.items() for a, b in zip(stations, stations[1:])
}


def metro_graph(closed=()) -> gs.Graph:
    """Undirected station graph. Closed stations (and their connections) are removed."""
    closed = set(closed)
    graph: gs.Graph = {}
    for stations in METRO_LINES.values():
        for s in stations:
            if s not in closed:
                graph.setdefault(s, [])
        for a, b in zip(stations, stations[1:]):
            if a in closed or b in closed:
                continue
            if b not in graph[a]:
                graph[a].append(b)
            if a not in graph[b]:
                graph[b].append(a)
    return graph


@dataclass
class Trip:
    path: list[str]
    legs: list[tuple[str, str, str, int]]  # (line, from, to, number of stops)
    stops: int
    transfers: int
    minutes: float


def plan_trip(start: str, goal: str, closed=(), algorithm: str = "BFS",
              minutes_per_stop: float = 2.5, transfer_minutes: float = 4.0) -> Trip | None:
    """Route between two stations, avoiding closed ones. BFS minimises the number of stops."""
    closed = set(closed) - {start, goal}
    graph = metro_graph(closed)
    if start not in graph or goal not in graph:
        return None
    result = (gs.bfs if algorithm == "BFS" else gs.dfs)(graph, start, goal)
    path = result.path_to(goal)
    if not path:
        return None
    legs: list[tuple[str, str, str, int]] = []
    for a, b in zip(path, path[1:]):
        line = EDGE_LINE[frozenset((a, b))]
        if legs and legs[-1][0] == line:
            legs[-1] = (line, legs[-1][1], b, legs[-1][3] + 1)
        else:
            legs.append((line, a, b, 1))
    stops, transfers = len(path) - 1, max(0, len(legs) - 1)
    return Trip(path, legs, stops, transfers, stops * minutes_per_stop + transfers * transfer_minutes)


# ================================================================ 2. customer segments
CUSTOMER_COLUMNS = ["Annual Spend", "Visits per Month", "Avg Basket", "Days Since Last Purchase"]


def make_customers(n: int = 400, seed: int = 7) -> pd.DataFrame:
    """Synthetic retail customers drawn from four hidden behaviour groups."""
    rng = np.random.default_rng(seed)
    #        share  visits/month  basket (mean, sd)  days since last purchase
    groups = [(0.30, (6.0, 1.5), (18, 5), (12, 6)),    # frequent, small baskets
              (0.20, (4.0, 1.2), (85, 15), (20, 8)),   # big baskets
              (0.30, (1.2, 0.5), (45, 12), (70, 20)),  # rare visits, long gap
              (0.20, (9.0, 1.8), (30, 8), (6, 3))]     # very frequent
    sizes = rng.multinomial(n, [g[0] for g in groups])
    frames = []
    for size, (_, visits, basket, recency) in zip(sizes, groups):
        v = np.clip(rng.normal(*visits, size), 0.3, None)
        b = np.clip(rng.normal(*basket, size), 5, None)
        frames.append(pd.DataFrame({
            "Annual Spend": (v * 12 * b * rng.uniform(0.85, 1.1, size)).round(0),
            "Visits per Month": v.round(1),
            "Avg Basket": b.round(0),
            "Days Since Last Purchase": np.clip(rng.normal(*recency, size), 1, None).round(0),
        }))
    return pd.concat(frames, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)


def name_segments(centers: pd.DataFrame) -> dict[int, tuple[str, str]]:
    """Give each cluster a business name and a suggested action (only when the sample columns are present)."""
    names = {int(i): (f"Segment {int(i)}", "Review this group's typical behaviour and design a targeted offer.")
             for i in centers.index}
    if not set(CUSTOMER_COLUMNS) <= set(centers.columns) or len(centers) < 2:
        return names
    spread = centers.std(ddof=0).replace(0, 1)
    z = (centers - centers.mean()) / spread
    rules = [
        ("💎 Big spenders", "Annual Spend", 1, "VIP perks, early access and premium bundles."),
        ("😴 Lapsing customers", "Days Since Last Purchase", 1, "Win-back email with a time-limited discount."),
        ("🔁 Loyal regulars", "Visits per Month", 1, "Loyalty points and referral rewards."),
        ("🏷️ Bargain hunters", "Avg Basket", -1, "Coupons, multi-buy deals and clearance alerts."),
    ]
    free = set(names)
    for title, column, sign, action in rules:
        if not free:
            break
        best = max(free, key=lambda i: sign * z.loc[i, column])
        names[best] = (title, action)
        free.remove(best)
    return names


# ================================================================ 3. fraud monitoring
FRAUD_FEATURES = ["amount", "hour", "distance_km", "merchant_risk", "txns_last_hour"]


def sample_transactions(n: int, fraud_rate: float, rng: np.random.Generator) -> pd.DataFrame:
    """Simulate card transactions. Fraud tends to be larger, at night, far from home, at risky merchants, in bursts."""
    is_fraud = rng.random(n) < fraud_rate
    amount = np.where(is_fraud, rng.lognormal(np.log(250), 0.9, n), rng.lognormal(np.log(40), 0.8, n))
    night = rng.choice([0, 1, 2, 3, 4, 5, 23], n)
    day = np.clip(rng.normal(14, 4, n).round(), 6, 22)
    hour = np.where(is_fraud & (rng.random(n) < 0.7), night, day).astype(int)
    distance = np.where(is_fraud, rng.exponential(60, n), rng.exponential(5, n))
    risk = np.where(is_fraud, rng.beta(6, 3, n), rng.beta(2, 8, n))
    burst = np.where(is_fraud, rng.poisson(4, n), rng.poisson(1, n))
    return pd.DataFrame({
        "amount": amount.round(2), "hour": hour, "distance_km": distance.round(1),
        "merchant_risk": risk.round(2), "txns_last_hour": burst, "is_fraud": is_fraud.astype(int),
    })


@dataclass
class FraudModel:
    model: RandomForestClassifier
    metrics: dict[str, float]


def train_fraud_model(seed: int = 0, n: int = 6000, fraud_rate: float = 0.06) -> FraudModel:
    """Train on simulated history and report hold-out precision / recall."""
    data = sample_transactions(n, fraud_rate, np.random.default_rng(seed))
    X_tr, X_te, y_tr, y_te = train_test_split(
        data[FRAUD_FEATURES], data["is_fraud"], test_size=0.25, random_state=seed, stratify=data["is_fraud"])
    model = RandomForestClassifier(n_estimators=150, class_weight="balanced", random_state=seed).fit(X_tr, y_tr)
    pred = model.predict(X_te)
    return FraudModel(model, {
        "Precision": precision_score(y_te, pred, zero_division=0),
        "Recall": recall_score(y_te, pred, zero_division=0),
        "F1": f1_score(y_te, pred, zero_division=0),
        "Accuracy": accuracy_score(y_te, pred),
    })


def score(model: RandomForestClassifier, frame: pd.DataFrame) -> np.ndarray:
    """Probability that each transaction is fraudulent."""
    return model.predict_proba(frame[FRAUD_FEATURES])[:, 1]
