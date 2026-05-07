from __future__ import annotations

import numpy as np


def pairwise_l2(features: np.ndarray) -> np.ndarray:
    diff = features[:, None, :] - features[None, :, :]
    return np.sqrt((diff * diff).sum(axis=-1))
