from __future__ import annotations

from torch import Tensor

from controllability.models.frozen_model import FrozenModel


def suffix_vjp(
    model: FrozenModel,
    prefix_tokens,
    patch_layer: int,
    patch_position: int,
    readout_layer: int,
    readout_position: int,
    chart_grad: Tensor,
) -> Tensor:
    return model.vjp_to_residual(prefix_tokens, patch_layer, patch_position, readout_layer, readout_position, chart_grad)
