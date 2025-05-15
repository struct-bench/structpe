"""
graph_handling.py

Utilities for attribute-dependency graph logic:
 - invert_adjacency for reversed adjacency
 - topological_sort for cycle detection & ordering
"""

def invert_adjacency(rev_adj):
    """
    Invert reversed adjacency: node -> list_of_predecessors
    into normal adjacency: predecessor -> [successors].
    """
    normal_adj = {}
    for node in rev_adj:
        normal_adj[node] = []
    for node, preds in rev_adj.items():
        for p in preds:
            normal_adj[p].append(node)
    return normal_adj


def topological_sort(rev_adj):
    """
    Kahn's algorithm to detect cycles & produce ordering.
    'rev_adj' is reversed adjacency, e.g.:
      emotion -> [sentiment]
      rating -> [sentiment]
      text -> [sentiment, emotion]
      sentiment -> []
    meaning sentiment precedes emotion/rating, emotion+sentiment precede text.
    """
    normal = invert_adjacency(rev_adj)

    # compute in-degrees
    in_degree = {n: 0 for n in normal}
    for n in normal:
        for suc in normal[n]:
            in_degree[suc] += 1

    queue = [n for n in in_degree if in_degree[n] == 0]
    order = []
    while queue:
        cur = queue.pop(0)
        order.append(cur)
        for suc in normal[cur]:
            in_degree[suc] -= 1
            if in_degree[suc] == 0:
                queue.append(suc)

    if len(order) != len(normal):
        raise ValueError("Cycle detected in attribute graph.")
    return order
