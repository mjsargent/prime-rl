from __future__ import annotations

import os

import pytest
import torch

from controllability.models.frozen_model import FrozenModel


def test_frozen_model_zero_patch_identity_vjp_and_decode():
    if os.environ.get("CONTROLLABILITY_RUN_HF_TESTS") != "1":
        pytest.skip("Set CONTROLLABILITY_RUN_HF_TESTS=1 to run HF model tests.")

    model = FrozenModel("sshleifer/tiny-gpt2", dtype="float32", device_map="cpu")
    if len(model.layers) < 2:
        pytest.skip("Tiny model does not expose at least two transformer layers.")

    tokens = model.tokenizer.encode("Hello world", return_tensors="pt")
    patch_layer = 0
    readout_layer = 1
    hidden_dim = model.get_residual(tokens, layer=patch_layer).shape[-1]
    zero = torch.zeros(hidden_dim)

    baseline = model.get_residual(tokens, layer=readout_layer)
    patched = model.patched_forward(tokens, patch_layer, -1, zero, readout_layer)
    assert torch.allclose(patched.cpu(), baseline.cpu(), atol=1e-5, rtol=1e-5)

    chart_grad = torch.randn_like(baseline)
    direction = torch.randn(hidden_dim)
    eps = 1e-3
    plus = model.patched_forward(tokens, patch_layer, -1, eps * direction, readout_layer)
    minus = model.patched_forward(tokens, patch_layer, -1, -eps * direction, readout_layer)
    finite_difference = ((plus - minus).cpu() * chart_grad.cpu()).sum() / (2 * eps)
    vjp = model.vjp_to_residual(tokens, patch_layer, -1, readout_layer, -1, chart_grad)
    vjp_direction = (vjp.cpu() * direction).sum()
    relative_error = (finite_difference - vjp_direction).abs() / finite_difference.abs().clamp_min(1e-6)
    assert relative_error < 1e-2

    plain = model.model.generate(
        input_ids=tokens.to(model.device),
        max_new_tokens=2,
        do_sample=False,
        pad_token_id=model.tokenizer.eos_token_id,
    )[0].cpu().tolist()
    patched_tokens = model.patched_decode(tokens, patch_layer, -1, zero, max_new_tokens=2, temperature=0.0)
    assert patched_tokens == plain
