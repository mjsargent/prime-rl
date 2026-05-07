from __future__ import annotations

import numpy as np


def choose_landmarks(x: np.ndarray, n_landmarks: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.choice(len(x), size=min(n_landmarks, len(x)), replace=False)
