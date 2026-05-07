from __future__ import annotations

import torch
from torch import Tensor


def random_unit_direction_like(reference: Tensor, seed: int) -> Tensor:
    generator = torch.Generator(device=reference.device).manual_seed(seed)
    direction = torch.randn(reference.shape, generator=generator, device=reference.device, dtype=reference.dtype)
    return direction / direction.norm().clamp_min(1e-12)
