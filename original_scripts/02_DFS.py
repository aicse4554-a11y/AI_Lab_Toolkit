def dfs(graph, node, visited=[]):
    if node not in visited:
        print(node, end=" ")
        visited.append(node)

        for next_node in graph[node]:
            dfs(graph, next_node, visited)


graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': [],
    'F': []
}

print("DFS:")
dfs(graph, 'A')
