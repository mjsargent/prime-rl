from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LayerPair:
    patch_layer: int
    readout_layer: int

    def __post_init__(self) -> None:
        if self.readout_layer <= self.patch_layer:
            raise ValueError("readout_layer must be greater than patch_layer")


def default_layer_pairs(num_layers: int) -> list[LayerPair]:
    if num_layers < 2:
        raise ValueError("Need at least two layers to form a patch/readout pair")
    mid = num_layers // 2
    candidates = {(0, mid), (mid, num_layers - 1)}
    if num_layers > 4:
        candidates.add((num_layers // 4, 3 * num_layers // 4))
    return [LayerPair(patch, readout) for patch, readout in sorted(candidates) if readout > patch]
