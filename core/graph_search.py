"""Breadth-First and Depth-First Search with step logs, goal search and Graphviz output."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

Graph = dict[str, list[str]]

PRESETS: dict[str, str] = {
    "Tree from the lab (A to F)": "A: B, C\nB: D, E\nC: F\nD:\nE:\nF:",
    "Graph with a cycle": "A: B, C\nB: D\nC: D, E\nD: F\nE: F\nF: A",
    "Two routes to a goal": "S: A, B\nA: C\nB: C, D\nC: G\nD: G\nG:",
}


def parse_graph(text: str, directed: bool = True) -> Graph:
    """Parse lines like ``A: B, C`` into an adjacency dict.

    Lines starting with ``#`` (or text after ``#``) are comments. Nodes that only appear
    as neighbours are added automatically. For undirected graphs every edge is mirrored.
    """
    graph: Graph = {}
    for raw in text.strip().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        node, rest = line.split(":", 1)
        node = node.strip().replace('"', "")
        if not node:
            continue
        graph.setdefault(node, [])
        for nb in (x.strip().replace('"', "") for x in rest.split(",")):
            if not nb:
                continue
            if nb not in graph[node]:
                graph[node].append(nb)
            graph.setdefault(nb, [])
            if not directed and node not in graph[nb]:
                graph[nb].append(node)
    return graph


@dataclass
class SearchResult:
    algorithm: str
    start: str
    goal: str | None
    order: list[str] = field(default_factory=list)
    steps: list[dict] = field(default_factory=list)
    parent: dict[str, str | None] = field(default_factory=dict)
    depth: dict[str, int] = field(default_factory=dict)

    @property
    def goal_found(self) -> bool:
        return self.goal is not None and self.goal in self.order

    def path_to(self, target: str | None) -> list[str] | None:
        """Path from start to ``target`` along the search tree (shortest for BFS only)."""
        if target is None or target not in self.order:
            return None
        path = [target]
        while self.parent.get(path[-1]) is not None:
            path.append(self.parent[path[-1]])  # type: ignore[arg-type]
        return path[::-1]


def bfs(graph: Graph, start: str, goal: str | None = None) -> SearchResult:
    """Breadth-First Search (FIFO queue). Finds shortest paths in unweighted graphs."""
    res = SearchResult("BFS", start, goal, parent={start: None}, depth={start: 0})
    queue: deque[str] = deque([start])
    visited: set[str] = set()
    while queue:
        node = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        res.order.append(node)
        for nb in graph.get(node, []):
            if nb not in res.parent:  # first time discovered
                res.parent[nb] = node
                res.depth[nb] = res.depth[node] + 1
                queue.append(nb)
        res.steps.append({"Step": len(res.order), "Visit": node, "Queue after": " ".join(queue) or "(empty)"})
        if node == goal:
            break
    return res


def dfs(graph: Graph, start: str, goal: str | None = None) -> SearchResult:
    """Depth-First Search (LIFO stack). Visits neighbours left to right like recursion."""
    res = SearchResult("DFS", start, goal)
    stack: list[tuple[str, str | None, int]] = [(start, None, 0)]
    visited: set[str] = set()
    while stack:
        node, par, depth = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        res.order.append(node)
        res.parent[node] = par
        res.depth[node] = depth
        for nb in reversed(graph.get(node, [])):
            if nb not in visited:
                stack.append((nb, node, depth + 1))
        top_first = " ".join(n for n, _, _ in reversed(stack)) or "(empty)"
        res.steps.append({"Step": len(res.order), "Visit": node, "Stack after (top first)": top_first})
        if node == goal:
            break
    return res


def graph_to_dot(
    graph: Graph,
    order: list[str] | tuple = (),
    path: list[str] | None = None,
    directed: bool = True,
    start: str | None = None,
    goal: str | None = None,
) -> str:
    """Graphviz DOT source. Visited nodes show their visit number, the path is highlighted."""
    rank = {n: i + 1 for i, n in enumerate(order)}
    path_edges: set = set()
    for a, b in zip(path or [], (path or [])[1:]):
        path_edges.add((a, b) if directed else frozenset((a, b)))
    kind, op = ("digraph", "->") if directed else ("graph", "--")
    lines = [
        f"{kind} G {{",
        '  bgcolor="transparent";',
        '  node [shape=circle, style=filled, fillcolor="#ffffff", fontname="Helvetica", color="#555555"];',
        '  edge [color="#888888"];',
    ]
    for n in graph:
        label = f"{n}\\n#{rank[n]}" if n in rank else n
        attrs = [f'label="{label}"']
        if n in rank:
            attrs.append('fillcolor="#cfe8ff"')
        if n == start:
            attrs.append('fillcolor="#b7e4c7"')
        if n == goal:
            attrs.append('fillcolor="#ffd6a5"')
        lines.append(f'  "{n}" [{", ".join(attrs)}];')
    seen: set = set()
    for n, nbrs in graph.items():
        for m in nbrs:
            key = (n, m) if directed else frozenset((n, m))
            if key in seen:
                continue
            seen.add(key)
            style = ' [color="#e63946", penwidth=2.5]' if key in path_edges else ""
            lines.append(f'  "{n}" {op} "{m}"{style};')
    lines.append("}")
    return "\n".join(lines)
