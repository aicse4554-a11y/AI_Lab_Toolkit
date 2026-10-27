def bfs(graph, start):
    visited = []
    queue = [start]

    while queue:
        node = queue.pop(0)

        if node not in visited:
            print(node, end=" ")
            visited.append(node)
            queue.extend(graph[node])


graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': [],
    'F': []
}

print("BFS:")
bfs(graph, 'A')
