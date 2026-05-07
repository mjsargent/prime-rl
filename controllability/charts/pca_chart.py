from __future__ import annotations

import numpy as np


class PCAChart:
    def __init__(self, n_components: int):
        self.n_components = n_components
        self.mean_: np.ndarray | None = None
        self.components_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "PCAChart":
        self.mean_ = x.mean(axis=0)
        centered = x - self.mean_
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        self.components_ = vh[: self.n_components]
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("PCAChart must be fit before transform")
        return (x - self.mean_) @ self.components_.T
