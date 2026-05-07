from __future__ import annotations

from pathlib import Path

import numpy as np
from torch import Tensor


class ActivationCache:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, env_id: str, model_id: str, layer: int) -> Path:
        safe_env = env_id.replace("/", "_")
        safe_model = model_id.replace("/", "_")
        return self.root / f"{safe_env}_{safe_model}_residuals_l{layer}.npy"

    def save(self, env_id: str, model_id: str, layer: int, residuals: Tensor | np.ndarray) -> Path:
        path = self.path_for(env_id, model_id, layer)
        array = residuals.detach().cpu().float().numpy() if isinstance(residuals, Tensor) else residuals
        np.save(path, array)
        return path

    def load(self, env_id: str, model_id: str, layer: int) -> np.ndarray:
        return np.load(self.path_for(env_id, model_id, layer))
