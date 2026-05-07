from __future__ import annotations

from torch import Tensor


def clip_to_norm(delta: Tensor, max_norm: float) -> Tensor:
    norm = delta.norm()
    if norm <= max_norm:
        return delta
    return delta * (max_norm / norm.clamp_min(1e-12))
