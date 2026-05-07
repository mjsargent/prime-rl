from __future__ import annotations

import pytest

from controllability.models.layer_pair_registry import LayerPair, default_layer_pairs


def test_layer_pair_requires_readout_after_patch():
    with pytest.raises(ValueError):
        LayerPair(patch_layer=4, readout_layer=4)


def test_default_layer_pairs_are_ordered():
    pairs = default_layer_pairs(12)
    assert pairs
    assert all(pair.patch_layer < pair.readout_layer for pair in pairs)
