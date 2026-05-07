from __future__ import annotations


def path_cost(path: list[int], edge_costs: dict[tuple[int, int], float]) -> float:
    return sum(edge_costs[(a, b)] for a, b in zip(path, path[1:], strict=False))
