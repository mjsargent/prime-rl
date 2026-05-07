from __future__ import annotations

import numpy as np


def graph_laplacian(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    degree = np.diag(weights.sum(axis=1))
    return degree - weights, degree
