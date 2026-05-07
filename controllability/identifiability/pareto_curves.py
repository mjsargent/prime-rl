from __future__ import annotations


def pareto_points(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    frontier: list[tuple[float, float]] = []
    for point in sorted(points):
        if not frontier or point[1] > frontier[-1][1]:
            frontier.append(point)
    return frontier
