from __future__ import annotations

import numpy as np


def nearest_neighbor_extend(train_z: np.ndarray, train_values: np.ndarray, query_z: np.ndarray) -> np.ndarray:
    distances = ((query_z[:, None, :] - train_z[None, :, :]) ** 2).sum(axis=-1)
    return train_values[distances.argmin(axis=1)]
