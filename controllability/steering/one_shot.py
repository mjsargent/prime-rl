from __future__ import annotations

from torch import Tensor

from controllability.steering.trust_region import clip_to_norm


def one_shot_delta(direction: Tensor, edit_norm: float) -> Tensor:
    return clip_to_norm(direction, edit_norm)
